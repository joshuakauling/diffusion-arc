#!/usr/bin/env python3
"""Start the third epoch from pinned HF weights with fresh AdamW and a high-noise recipe.

With --ssh, upload local source to an existing Pod and follow the detached job.
Full checkpoints created by this stage can later be resumed with --resume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.cloud import train_runpod as pod
from scripts.cloud.diagnose_runpod import download_transformer

DEFAULT_CONFIG = ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json"


def check_parent_run(api, config) -> dict:
    """Read parent identity/counters and reject stale history before changing W&B."""
    options = config.recipe["logging"]
    entity = options["entity"] or api.default_entity
    run = api.run(f"{entity}/{options['project']}/{options['parent_run_id']}")
    expected_step = config.recipe["models"]["completed_epochs"] * config.loop.max_optimizer_steps
    if run.state == "running":
        raise ValueError("The parent W&B run is still active")
    if run.lastHistoryStep != expected_step:
        raise ValueError(f"Parent W&B history must end at step {expected_step}; found {run.lastHistoryStep}")
    # The source experiment's validation split and scientific settings must be retained.
    recipe = config.recipe_dict()
    for key in ("loop", "loss", "optimizer", "scheduler", "execution", "canvas", "validation"):
        previous, current = dict(run.config[key]), dict(recipe[key])
        if key == "validation":
            previous.pop("bundle")
            current.pop("bundle")
        if previous != current:
            raise ValueError(f"Parent W&B {key} differs from the agreed recipe")
    counters = {}
    if options["resume_parent_run"]:
        workload = options["workload_counters"]
        for phase in workload["phases"]:
            for name in workload["names"]:
                key = f"{phase}/{name}"
                value = run.summary[key + workload["cumulative_suffix"]]
                if type(value) not in (int, float) or value < 0 or int(value) != value:
                    raise ValueError(f"Invalid parent workload counter: {key}")
                counters[key] = int(value)
    return {"wandb_run": {"id": run.id, "entity": run.entity, "project": run.project},
            "optimizer_step": expected_step, "workload_totals": counters}


def train(config_path: str, startup_path: str) -> None:
    """Keep GPU allocations in a child process and restore only new-stage resume state."""
    from diffusion_arc.training.config import load_training_config
    from diffusion_arc.training.runner import run_training

    config = load_training_config(Path(config_path), expected_type="training_run")
    startup = json.loads(Path(startup_path).read_text())
    run_training(config, weights_parent=startup["parent_run"])


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--ssh" or arg.startswith("--ssh=") for arg in arguments):
        from scripts.cloud.launch_runpod import main as launch
        return launch(["finetune", *arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-preparation", action="store_true",
                        help="With --prepare-only, download/validate on CPU; GPU checks run before training")
    parser.add_argument("--use-prepared", action="store_true",
                        help="Require completed preparation; never download or rebuild data in this process")
    parser.add_argument("--output-repo", default=os.environ.get("HF_OUTPUT_REPO"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "diffusion-arc"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args(arguments)
    if args.cpu_preparation and not args.prepare_only:
        parser.error("--cpu-preparation requires --prepare-only")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("invalid run name")
    from diffusion_arc.training.config import load_training_config
    template = load_training_config(args.config, expected_type="training_run")
    if template.recipe["models"]["initialization"] != "transformer_weights":
        raise ValueError("finetune requires a transformer_weights initialization recipe")
    pod.require_secrets("HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN")
    args.workspace = args.workspace.resolve()
    os.environ["HF_HOME"] = str(args.workspace / ".cache/huggingface")
    os.environ["KAGGLEHUB_CACHE"] = str(args.workspace / ".cache/kagglehub")
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    from huggingface_hub import HfApi

    with pod.workspace_lock(args.workspace, local_only=template.recipe["deployment"]["provider"] == "modal"):
        directory = args.workspace / "runs" / args.run_name
        startup_path = directory / "startup.json"
        previous = json.loads(startup_path.read_text()) if startup_path.exists() else None
        if previous is None and (args.use_prepared or args.resume):
            raise ValueError("Completed preparation requires this stage's saved startup receipt")
        if previous:
            saved = load_training_config(directory / "training.json", expected_type="training_run")
            model_paths = {name: saved.recipe["models"][name] for name in (
                "nvarc_checkpoint", "flux_checkpoint", "transformer_checkpoint")}
        else:
            model_paths = pod.embedded_model_paths()
        config = pod.resolve_config(args, directory, model_paths=model_paths)
        if previous:
            expected, actual = config.recipe_dict(), saved.recipe_dict()
            for payload in (expected, actual):
                payload["models"]["resume_from"] = None
            if expected != actual or previous["frozen_sources"] != {
                    "nvarc": pod.NVARC_HANDLE, "flux_repo": pod.FLUX_REPO, "flux_revision": pod.FLUX_REVISION}:
                raise ValueError("Prepared run differs from this recipe; use a new run name")
        start_step = pod.validate_start(config, directory, args.resume)
        pod.prepare_runtime()
        api = HfApi(token=os.environ["HF_TOKEN"])
        output_repo = args.output_repo or (previous["output_repo"] if previous else
                                          template.recipe["models"]["transformer_source"]["repo_id"])
        pod.prepare_output_repository(api, output_repo, "model")
        import wandb
        wandb_api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=30)
        if not wandb_api.viewer:
            raise ValueError("W&B credentials could not be verified")
        if args.resume:
            from scripts.cloud.continue_runpod import check_wandb_run
            latest = pod.run_checkpoints(config)[-1]
            check_wandb_run(wandb_api, latest, last_allowed_step=latest.metadata["progress"]["epoch"]
                            * config.loop.max_optimizer_steps)
            parent = previous["parent_run"]
        else:
            parent = check_parent_run(wandb_api, config)
        if args.cpu_preparation:
            pod.preflight_disk(config, args.workspace, start_step)
        else:
            pod.preflight_gpu_and_disk(config, args.workspace, start_step)
        if previous:
            print("Reusing prepared models and training data; no repeated file scan or download", flush=True)
            transformer, source = Path(previous["transformer"]), previous["transformer_source"]
        else:
            pod.preflight_data_access()
            if model_paths is None:
                api.auth_check(pod.FLUX_REPO)
            print("Preparing frozen models, pinned transformer and a fresh augmented training view", flush=True)
            if model_paths is None:
                pod.download_models(config, workspace=args.workspace, include_transformer=False)
            storage = {"elastic_volume": True} if config.recipe["deployment"]["provider"] == "modal" else {}
            transformer, source = download_transformer(
                dict(template.recipe["models"]["transformer_source"]), workspace=args.workspace,
                max_gib=24, **storage)
        config = pod.resolve_config(args, directory, model_paths={
            **(model_paths or {}), "transformer_checkpoint": transformer,
        })
        # Large downloads must not consume the space reserved for all full checkpoints.
        pod.write_json(config.path, config.recipe_dict())
        if previous is None:
            pod.prepare_data(config, workspace=args.workspace)
        pod.preflight_disk(config, args.workspace, start_step)
        pod.write_json(startup_path, {
            "transformer": str(transformer), "transformer_source": source,
            "parent_run": parent, "optimizer_reset": True, "output_repo": output_repo,
            "frozen_sources": {"nvarc": pod.NVARC_HANDLE, "flux_repo": pod.FLUX_REPO,
                               "flux_revision": pod.FLUX_REVISION},
            "source_snapshot": json.loads((ROOT / "_source.json").read_text())
                               if (ROOT / "_source.json").exists() else None,
        })
        flow = config.recipe["flow"]
        endpoint, high = flow["endpoint_targets_per_full_batch"], flow["high_noise_targets_per_full_batch"]
        print(f"Ready: {config.loop.max_optimizer_steps} updates, seed {config.seed}, "
              f"{config.loop.optimizer_batch_size}-target batches with {endpoint} endpoint / {high} high-noise / "
              f"{config.loop.optimizer_batch_size - endpoint - high} uniform draws; "
              f"{config.scheduler.warmup_steps}-step warmup. Config: {config.path}", flush=True)
        if args.prepare_only:
            return 0
        pod.write_json(directory / "training-started.json", {"resume_from": config.recipe_dict()["models"]["resume_from"]})
        subprocess.run([sys.executable, "-c", "from scripts.cloud.finetune_runpod import train; "
                        "import sys; train(*sys.argv[1:])", str(config.path), str(startup_path)],
                       cwd=ROOT, check=True)
        pod.upload_weights(config, api=api, output_repo=output_repo)
        suffix = " The Pod remains running." if config.recipe["deployment"]["provider"] == "runpod" else ""
        print("Training and weight uploads completed." + suffix, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"RunPod finetuning: {error}", file=sys.stderr)
        raise SystemExit(1)
