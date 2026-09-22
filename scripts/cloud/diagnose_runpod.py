#!/usr/bin/env python3
"""Launch bounded B3 sampling diagnostics or checkpoint comparisons on an existing RunPod.

With --ssh, transfer the current source and follow the detached job's logs.
Use download --ssh to save verified results and logs locally before deleting the Pod.
On the Pod, fetch the selected HF transformer, reuse frozen model components,
and supervise the diagnostic without creating a training or W&B run. A paired
config evaluates both exports on identical tasks, noise seeds and Euler budgets.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.cloud import train_runpod as pod


DEFAULT_CONFIG = ROOT / "configs/diagnostics/partial_denoising_b3.json"


def load_config(path: Path) -> dict:
    """Validate the experiment and download bounds before opening a connection."""
    config = json.loads(path.read_text())
    return validate_config(config, path)


def validate_config(config: dict, path: Path) -> dict:
    """A comparison shares every setting except the two checkpoint identities."""
    if isinstance(config, dict) and "checkpoints" in config:
        checkpoints = config["checkpoints"]
        if ("checkpoint" in config or not isinstance(checkpoints, dict) or len(checkpoints) != 2
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name)
                       for name in checkpoints)):
            raise ValueError("checkpoints must name exactly two exports, without a singular checkpoint")
        if "generation_steps" not in config:
            raise ValueError("checkpoint comparisons require pure-noise generation_steps")
        common = {key: value for key, value in config.items() if key != "checkpoints"}
        for checkpoint in checkpoints.values():
            validate_config({**common, "checkpoint": checkpoint}, path)
        if len({tuple(sorted(spec.items())) for spec in checkpoints.values()}) != 2:
            raise ValueError("select two different checkpoints")
        return config
    fields = {"schema_version", "training_config", "checkpoint", "task_count", "split_seed",
              "sampling_seeds", "max_runtime_seconds",
              "max_setup_seconds", "max_checkpoint_gib"}
    trajectory_fields = {"start_times", "steps_per_unit", "generation_schedules"}
    optional = {"task_pool", "generation_steps", *trajectory_fields}
    if (not isinstance(config, dict) or not fields <= set(config) or set(config) - fields - optional
            or type(config["schema_version"]) is not int or config["schema_version"] != 1):
        raise ValueError("unsupported partial-denoising config")
    pool = config.get("task_pool", "sampling")
    if pool not in ("sampling", "validation"):
        raise ValueError("task_pool must select sampling or validation; final test tasks are excluded")
    if config.get("generation_schedules", ["linear"]) not in (["linear"], ["linear", "cosine_high_noise"]):
        raise ValueError("generation_schedules must be linear, optionally followed by cosine_high_noise")
    for name, maximum in (("task_count", 100 if pool == "validation" else 32),
                          ("max_runtime_seconds", 1800), ("max_setup_seconds", 1800),
                          ("max_checkpoint_gib", 24)):
        if type(config[name]) is not int or not 1 <= config[name] <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if type(config["split_seed"]) is not int or config["split_seed"] != 167:
        raise ValueError("use the frozen validation split with seed 167")
    seeds = config["sampling_seeds"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 2
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("sampling_seeds must contain one or two distinct integer seeds")
    if "generation_steps" in config:
        counts = config["generation_steps"]
        if (not isinstance(counts, list) or not 1 <= len(counts) <= 8
                or any(type(n) is not int or not 1 <= n <= 1000 for n in counts)
                or len(set(counts)) != len(counts)):
            raise ValueError("generation_steps must contain one to eight distinct step counts in [1, 1000]")
        if trajectory_fields & set(config):
            raise ValueError("generation_steps selects linear t=1 runs; omit start_times, steps_per_unit and generation_schedules")
    else:
        steps = config.get("steps_per_unit")
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValueError("steps_per_unit must be an integer in [1, 1000]")
        times = config.get("start_times")
        if (not isinstance(times, list) or not times
                or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t <= 1
                       or not math.isclose(t * steps, round(t * steps), abs_tol=1e-7) for t in times)
                or len(set(times)) != len(times) or 1.0 not in times):
            raise ValueError("start_times must be unique Euler-grid points in (0, 1], including 1")
    checkpoint = config["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"repo_id", "revision", "subfolder"}:
        raise ValueError("checkpoint requires repo_id, revision and subfolder")
    if not isinstance(checkpoint["repo_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", checkpoint["repo_id"]):
        raise ValueError("checkpoint repo_id must be a Hugging Face owner/repository")
    for name in ("revision", "subfolder"):
        value = checkpoint[name]
        if (not isinstance(value, str) or not value or value.startswith("/")
                or any(part in ("", ".", "..") for part in value.split("/"))
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)):
            raise ValueError(f"invalid checkpoint {name}")
    if not isinstance(config["training_config"], str):
        raise ValueError("training_config must be a repository config path")
    training = (path.resolve().parent / config["training_config"]).resolve()
    if not training.is_relative_to(ROOT / "configs") or not training.is_file() or training.suffix != ".json":
        raise ValueError("training_config must reference an existing repository config")
    return config


def transformer_metadata(spec: dict, *, max_gib: int):
    """Read Hub metadata only; fail missing exports before downloading any weights."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hub did not return an immutable commit SHA")
    prefix = spec["subfolder"] + "/"
    files = [entry for entry in info.siblings
             if entry.rfilename.startswith(prefix)
             and "/" not in entry.rfilename[len(prefix):]
             and entry.rfilename.endswith((".json", ".safetensors"))]
    names = {entry.rfilename for entry in files}
    if prefix + "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError(f"Missing transformer export: {spec['repo_id']}/{spec['subfolder']} at {info.sha}")
    if any(type(entry.size) is not int or entry.size <= 0 for entry in files):
        raise ValueError("Hub file sizes are missing or invalid")
    total = sum(entry.size for entry in files)
    if total > max_gib * 1024**3:
        raise ValueError(f"Selected checkpoint exceeds the {max_gib} GiB download limit")
    return info.sha, files, total


def resolve_checkpoints(config: dict) -> dict:
    """Resolve every checkpoint before starting the first expensive evaluation."""
    specs = config.get("checkpoints", {"checkpoint": config.get("checkpoint")})
    resolved = {}
    for name, spec in specs.items():
        revision, files, total = transformer_metadata(spec, max_gib=config["max_checkpoint_gib"])
        resolved[name] = {**spec, "revision": revision}
        print(f"HF source ready: {name}, {len(files)} files, {total / 1024**3:.2f} GiB, {revision}", flush=True)
    return {**config, **({"checkpoints": resolved} if "checkpoints" in config else resolved)}


def download_transformer(spec: dict, *, workspace: Path, max_gib: int,
                         elastic_volume: bool = False) -> tuple[Path, dict]:
    """Fetch a pinned export; inspect sizes and the index without rehashing weights."""
    from huggingface_hub import snapshot_download

    revision, files, total = transformer_metadata(spec, max_gib=max_gib)
    prefix = spec["subfolder"] + "/"
    names = {entry.rfilename for entry in files}
    # Modal Volumes have elastic capacity and statfs only returns placeholder values.
    if not elastic_volume and shutil.disk_usage(workspace).free < total + 1024**3:
        raise ValueError("Insufficient workspace space for the selected transformer and results")
    print(f"HF download: {spec['repo_id']}/{spec['subfolder']} at {revision}; "
          f"{len(files)} files, {total / 1024**3:.2f} GiB before cache reuse", flush=True)
    snapshot = Path(snapshot_download(spec["repo_id"], revision=revision,
                    allow_patterns=sorted(names), token=os.environ["HF_TOKEN"]))
    identity = {**spec, "resolved_revision": revision, "bytes": total, "files": {},
                "verification": "file_sizes_and_shard_index; hashes_from_hub_metadata"}
    for entry in files:
        path = snapshot / entry.rfilename
        if path.stat().st_size != entry.size:
            raise ValueError(f"Incomplete downloaded file: {entry.rfilename}")
        identity["files"][entry.rfilename[len(prefix):]] = {
            "bytes": entry.size, "hub_sha256": entry.lfs.sha256 if entry.lfs is not None else None,
        }
    folder = snapshot / spec["subfolder"]
    for index in folder.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map")
        if (not isinstance(mapping, dict) or not mapping
                or any(not isinstance(shard, str) or prefix + shard not in names
                       or not shard.endswith(".safetensors") for shard in mapping.values())):
            raise ValueError("Transformer index references missing or invalid weight shards")
    return folder, identity


def prepare_bundle(workspace: Path) -> Path:
    """Prepare only the existing validation split; never mount final-test payloads."""
    from scripts.data.prepare_arc_validation_benchmark import ARCHIVE_URL, ARCHIVE_SHA256, prepare
    from diffusion_arc.training.run.validation import validate_validation_bundle

    bundle = workspace / "data/benchmarks/arc_agi_1_validation"
    if not bundle.exists():
        with TemporaryDirectory(dir=workspace) as temporary:
            archive = Path(temporary) / "arc-agi-1.tar.gz"
            with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            pod.verify_file(archive, ARCHIVE_SHA256)
            staged = Path(temporary) / "validation"
            prepare(archive, staged, seed=167)
            bundle.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(bundle)
    validate_validation_bundle(bundle, split_seed=167, verify_payloads=False)
    return bundle


def load_runtime(config_path: Path, config: dict):
    """Require the B3 model contract before downloads or GPU model allocation."""
    from diffusion_arc.training.config import load_training_config

    runtime = load_training_config((config_path.parent / config["training_config"]).resolve(), expected_type="baseline")
    if (runtime.flux_packing is None or runtime.flux_packing.max_tokens != 12288
            or runtime.nvarc_max_tokens != 134400
            or (runtime.models.nvarc_dtype, runtime.models.vae_dtype,
                runtime.models.flux_parameter_dtype, runtime.models.flux_compute_dtype)
            != ("bfloat16", "float32", "float32", "bfloat16")):
        raise ValueError("diagnostic requires the production B3 packing and precision configuration")
    return runtime


def prepare_job(config_path: Path, workspace: Path, output: Path) -> None:
    """Prepare pinned dependencies, frozen components and diagnostic inputs on the Pod."""
    from diffusion_arc.training.setup import prepare_device

    config = load_config(config_path)
    pod.require_secrets("HF_TOKEN")
    pod.prepare_runtime()
    runtime = load_runtime(config_path, config)
    prepare_device(runtime)
    embedded = pod.embedded_model_paths(verify_payloads=False)
    if embedded is None:
        paths = {"nvarc_checkpoint": workspace / "models/nvarc-qwen3-4b-grids15-sft139",
                 "flux_checkpoint": workspace / "models/flux2-klein-base-4b"}
        if shutil.disk_usage(workspace).free < (config["max_checkpoint_gib"] + 10) * 1024**3:
            raise ValueError("Need checkpoint headroom plus 10 GiB for frozen NVARC/VAE downloads")
        pod.require_secrets("KAGGLE_API_TOKEN")
        runtime = replace(runtime, models=replace(runtime.models, **paths))
        pod.download_models(runtime, workspace=workspace, include_transformer=False)
    else:
        paths = embedded
    transformer, identity = download_transformer(config["checkpoint"], workspace=workspace,
                                                max_gib=config["max_checkpoint_gib"])
    bundle = prepare_bundle(workspace)
    pod.write_json(output / "prepared.json", {
        "models": {name: str(path) for name, path in paths.items()},
        "transformer": str(transformer), "checkpoint": identity, "bundle": str(bundle),
        "frozen_sources": {"nvarc": pod.NVARC_HANDLE, "flux_repo": pod.FLUX_REPO,
                           "flux_revision": pod.FLUX_REVISION},
    })


def stop_child(process) -> None:
    """Release the diagnostic session, including setup subprocesses such as pip."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # A descendant can outlive its direct parent or ignore SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(config_path: Path, workspace: Path, output: Path, config: dict) -> int:
    """Bound setup and GPU runtime separately, releasing the child on timeout."""
    command = [sys.executable, str(ROOT / "scripts/diagnose_partial_denoising.py"),
               "--config", str(config_path), "--workspace", str(workspace), "--output", str(output)]
    started = time.monotonic()
    ready_at = None

    def terminate(_signum, _frame):
        raise SystemExit(143)

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        return _supervise_child(command, config, output, started, ready_at)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _supervise_child(command, config, output, started, ready_at) -> int:
    with subprocess.Popen(command, cwd=ROOT, start_new_session=True) as process:
        try:
            while process.poll() is None:
                now = time.monotonic()
                if ready_at is None and (output / "models-ready.json").is_file():
                    ready_at = json.loads((output / "models-ready.json").read_text())["monotonic_seconds"]
                elapsed = now - (ready_at if ready_at is not None else started)
                limit = config["max_runtime_seconds"] if ready_at is not None else config["max_setup_seconds"]
                if elapsed > limit:
                    stop_child(process)
                    pod.write_json(output / "timeout.json", {
                        "phase": "diagnostic" if ready_at is not None else "setup",
                        "limit_seconds": limit, "elapsed_seconds": elapsed,
                    })
                    report_path = output / "report.json"
                    if report_path.is_file():
                        report = json.loads(report_path.read_text())
                        report.update(status="timed_out", timeout_phase="diagnostic" if ready_at is not None else "setup")
                        pod.write_json(report_path, report)
                    print(f"Diagnostic stopped after {limit}s in {'runtime' if ready_at is not None else 'setup'}; "
                          f"partial artifacts: {output}", flush=True)
                    return 124
                time.sleep(0.5)
        except BaseException:
            stop_child(process)
            raise
        return process.returncode if process.returncode >= 0 else 128 - process.returncode


def summarize_comparison(output: Path, config: dict) -> dict:
    """Compare task-level outcomes at each budget, keeping all queries under one seed."""
    from scripts.diagnose_partial_denoising import summarize

    summaries, scores, reference = {}, {}, None
    for name in config["checkpoints"]:
        folder = output / name
        report = json.loads((folder / "report.json").read_text())
        if report["status"] != "succeeded":
            raise ValueError(f"Cannot compare incomplete checkpoint: {name}")
        contract = {key: report[key] for key in
                    ("queries", "selected_task_ids", "dataset_manifest", "trajectories")}
        contract["frozen_sources"] = report["preparation"]["frozen_sources"]
        if reference is not None and contract != reference:
            raise ValueError("Checkpoint evaluations used different queries, sources or trajectories")
        reference = contract
        records = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        summaries[name] = [row for row in summarize(records, report["queries"], config)
                           if row["kind"] == "denoised"]
        scores[name] = {(r["sampling_steps"], r["task_id"], r["query_index"], r["seed"]): r["exact_grid"]
                        for r in records if r["kind"] == "denoised"}
    first, second = config["checkpoints"]
    task_queries = {}
    for query in reference["queries"]:
        task_queries.setdefault(query["task_id"], []).append(query["query_index"])
    paired = []
    for steps in config["generation_steps"]:
        for attempts in range(1, len(config["sampling_seeds"]) + 1):
            solved = {name: {task for task, queries in task_queries.items()
                            if any(all(scores[name][steps, task, query, seed] for query in queries)
                                   for seed in config["sampling_seeds"][:attempts])}
                      for name in config["checkpoints"]}
            paired.append({"sampling_steps": steps, "attempts": attempts,
                           "task_count": len(task_queries), "first": first, "second": second,
                           "first_only": sorted(solved[first] - solved[second]),
                           "second_only": sorted(solved[second] - solved[first]),
                           "both": sorted(solved[first] & solved[second]),
                           "neither": sorted(task_queries.keys() - (solved[first] | solved[second]))})
    return {"summaries": summaries, "paired_tasks": paired}


def run_comparison(config_path: Path, workspace: Path, output: Path, config: dict) -> int:
    """Run the existing diagnostic twice, reusing downloads and preserving partial results."""
    started = time.monotonic()
    report = {"schema_version": 1, "status": "setting_up", "config": config,
              "completed_checkpoints": [], "checkpoint_reports": {}}
    pod.write_json(output / "report.json", report)
    try:
        config = resolve_checkpoints(config)
        report.update(config=config, status="running")
        common = {key: value for key, value in config.items() if key != "checkpoints"}
        common["training_config"] = str((config_path.parent / config["training_config"]).resolve())
        for name, checkpoint in config["checkpoints"].items():
            folder = output / name
            folder.mkdir()
            child_config = {**common, "checkpoint": checkpoint}
            child_path = folder / "diagnostic.json"
            pod.write_json(child_path, child_config)
            report["active_checkpoint"] = name
            report["checkpoint_reports"][name] = f"{name}/report.json"
            pod.write_json(output / "report.json", report)
            print(f"\nEvaluating {name}: t=1, Euler steps {config['generation_steps']}", flush=True)
            code = supervise(child_path, workspace, folder, child_config)
            if code:
                report.update(status="timed_out" if code == 124 else "failed", exit_code=code)
                return code
            report["completed_checkpoints"].append(name)
        report.update(summarize_comparison(output, config), status="succeeded", active_checkpoint=None)
        print(json.dumps(report["summaries"], indent=2), flush=True)
        print(f"Comparison complete: {output / 'report.json'}", flush=True)
        return 0
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["total_seconds"] = time.monotonic() - started
        pod.write_json(output / "report.json", report)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["download"]:
        from scripts.cloud.download_diagnostic import main as download
        return download(arguments[1:])
    if any(arg == "--ssh" or arg.startswith("--ssh=") for arg in arguments):
        from scripts.cloud.launch_runpod import main as launch
        return launch(["diagnose", *arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-sources", action="store_true",
                        help="Check HF export metadata only; no SSH, downloads or GPU")
    args = parser.parse_args(arguments)
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.check_sources:
        resolve_checkpoints(config)
        return 0
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name or ""):
        parser.error("requires a valid --run-name")
    workspace = args.workspace.resolve()
    os.environ.setdefault("HF_HOME", str(workspace / ".cache/huggingface"))
    os.environ.setdefault("KAGGLEHUB_CACHE", str(workspace / ".cache/kagglehub"))
    with pod.workspace_lock(workspace):
        output = workspace / "diagnostics" / args.run_name
        output.mkdir(parents=True, exist_ok=False)
        pod.write_json(output / "diagnostic.json", config)
        print(f"Diagnostic artifacts: {output}\nLimits per checkpoint: setup {config['max_setup_seconds']}s; "
              f"runtime after model loading {config['max_runtime_seconds']}s", flush=True)
        if "checkpoints" in config:
            return run_comparison(config_path, workspace, output, config)
        return supervise(config_path, workspace, output, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"RunPod diagnostic: {error}", file=sys.stderr)
        raise SystemExit(1)
