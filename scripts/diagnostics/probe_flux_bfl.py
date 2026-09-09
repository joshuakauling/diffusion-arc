#!/usr/bin/env python3
"""Validate the BFL reference, Diffusers singletons and our masked batch path.

Uses existing local checkpoints and the recorded short A/B exposures. This
forward-only diagnostic never modifies production modules or model weights.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import torch

from scripts.diagnostics import probe_flux_packing as inputs_probe
from scripts.diagnostics import _flux_bfl_forward as forward
from scripts.diagnostics._flux_packing_forward import forward_policy, slice_inputs
from scripts.diagnostics.probe_nvarc_packing import dataset_hashes

PROGRESS_MARKER = "FLUX_BFL_PROGRESS"
VARIANTS = ("bfl_a", "bfl_b", "diffusers_a", "diffusers_b", "diffusers_ab_masked")


@dataclass(frozen=True)
class ProbeConfig:
    input_config: Path
    training_config: Path
    input_policy: inputs_probe.ProbeConfig
    max_duration_seconds: int
    sdpa_backend: str
    precisions: tuple[str, ...]
    numerical_tolerances: dict


def load_probe_config(path, *, require_checkpoints=True):
    path = Path(path)
    raw = json.loads(path.read_text())
    fields = {"schema_version", "input_config", "max_duration_seconds", "sdpa_backend",
              "precisions", "numerical_tolerances"}
    if (not isinstance(raw, dict) or set(raw) != fields
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("unexpected BFL probe configuration fields or schema")
    if type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 480:
        raise ValueError("BFL worker budget must be an integer in [1, 480] seconds")
    if raw["sdpa_backend"] != "math" or raw["precisions"] != ["fp32", "bf16"]:
        raise ValueError("BFL probe requires Math SDPA with FP32 followed by BF16")
    if not isinstance(raw["input_config"], str) or not raw["input_config"]:
        raise ValueError("input_config must name the recorded packing input configuration")
    tolerances = raw["numerical_tolerances"]
    if not isinstance(tolerances, dict) or set(tolerances) != {"fp32", "bf16"}:
        raise ValueError("numerical_tolerances must cover both precisions")
    for precision, limits in tolerances.items():
        if (not isinstance(limits, dict) or set(limits) != {"output_relative_l2", "loss_relative"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1
                       for v in limits.values())):
            raise ValueError(f"invalid {precision} output/loss tolerances")
    input_path = (path.resolve().parent / raw["input_config"]).resolve()
    input_policy, training = inputs_probe.load_probe_config(input_path, require_checkpoints=require_checkpoints)
    policy = ProbeConfig(input_config=input_path, training_config=input_policy.training_config,
                         input_policy=forward_policy(input_policy),
                         precisions=tuple(raw["precisions"]), numerical_tolerances=tolerances,
                         max_duration_seconds=raw["max_duration_seconds"], sdpa_backend="math")
    build_plan(policy, training)
    return policy, training


def build_plan(policy, training):
    parent = {key: value for key, value in inputs_probe.build_plan(policy.input_policy, training)[0].items()
              if key != "round_orders"}
    return dict(source_group=parent, sample_labels=["A", "B"],
                target_indices=parent["target_indices"][:2], text_lengths=parent["text_lengths"][:2],
                image_tokens_per_sample=900, batch_padded_text_length=parent["padded_length"],
                cases=[f"{precision}/{variant}" for precision in policy.precisions for variant in VARIANTS],
                stages=forward.stage_order(5, 20), sdpa_backend=policy.sdpa_backend,
                full_forward_calls=2 * len(VARIANTS), backward_calls=0, optimizer_updates=0,
                source_input_precision="existing NVARC/VAE preparation; promoted identically to FP32",
                parameter_dtype="torch.float32", tf32=False,
                numerical_tolerances=policy.numerical_tolerances,
                scope="implementation and mask parity; not cuDNN/FA3 or training equivalence")


def source_hashes():
    paths = {*sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
             *sorted((REPOSITORY_ROOT / "scripts/diagnostics").glob("*.py")),
             *sorted((REPOSITORY_ROOT / "scripts/diagnostics/_third_party/bfl_flux2").glob("*")),
             REPOSITORY_ROOT / "scripts/cloud/probe_flux_bfl_modal.py",
             REPOSITORY_ROOT / "scripts/cloud/probe_flux_packing_modal.py", REPOSITORY_ROOT / "pyproject.toml"}
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def source_metadata():
    metadata = inputs_probe.source_metadata()
    metadata["source_sha256"] = source_hashes()
    return metadata


def config_hashes(path, policy):
    return {name: hashlib.sha256(Path(file).read_bytes()).hexdigest() for name, file in (
        ("probe", path), ("inputs", policy.input_config), ("training", policy.training_config))}


def run_comparisons(bfl, diffusers, data, policy, *, profile=False, on_case=lambda row: None):
    prepared_identity = forward.input_identity(data)
    calls = {
        "a": forward.call_inputs(data, (0,)), "b": forward.call_inputs(data, (1,)),
        "ab": forward.call_inputs(data, (0, 1), policy.input_policy.groups[0]["padded_length"]),
    }
    call_identities = {name: forward.call_identity(inputs) for name, inputs in calls.items()}
    rows = []
    # Retain only the current precision's references. Comparison observers use
    # CPU snapshots; no 4B model activation graph is kept.
    for precision in policy.precisions:
        snapshots, losses = {"bfl_singletons": {}, "diffusers_singletons": {}}, {
            "bfl_singletons": {}, "diffusers_singletons": {}}
        for variant in VARIANTS:
            implementation = "bfl" if variant.startswith("bfl_") else "diffusers"
            key = "ab" if variant.endswith("masked") else variant[-1]
            model = bfl if implementation == "bfl" else diffusers
            reference_names = (() if implementation == "bfl" else
                               ("bfl_singletons", "diffusers_singletons") if key == "ab" else ("bfl_singletons",))
            on_case(dict(variant=f"{precision}/{variant}", status="running"))
            if forward.call_identity(calls[key]) != call_identities[key]:
                raise ValueError("forward inputs changed before a comparison call")
            row, captured = forward.run_case(model, implementation, calls[key], precision,
                policy.numerical_tolerances[precision],
                references={name: snapshots[name] for name in reference_names},
                reference_losses={name: losses[name] for name in reference_names}, profile=profile)
            row["variant"] = f"{precision}/{variant}"
            row["input_identity"] = call_identities[key]
            if forward.call_identity(calls[key]) != call_identities[key]:
                raise ValueError("forward mutated a comparison call's input tensors")
            if key != "ab":
                name = f"{implementation}_singletons"
                snapshots[name].update(captured)
                losses[name].update(row["losses"])
            rows.append(row)
            on_case(row)
        if forward.input_identity(data) != prepared_identity:
            raise ValueError("comparison mutated the shared prepared input tensors")
    if any(parameter.grad is not None for model in (bfl, diffusers) for parameter in model.parameters()):
        raise ValueError("forward-only comparison populated parameter gradients")
    return rows


def validate_complete(result):
    plan, cases = result["plan"], result.get("cases", [])
    if [row["variant"] for row in cases] != plan["cases"]:
        raise ValueError("missing or reordered BFL comparison cases")
    reference_losses = {}
    for row in cases:
        precision, variant = row["variant"].split("/")
        implementation = "bfl" if variant.startswith("bfl_") else "diffusers"
        labels = ["A", "B"] if variant.endswith("masked") else [variant[-1].upper()]
        if (row["status"] != "complete" or list(row["outputs"]) != plan["stages"]
                or row["sdpa_backend"] != "math" or row["grad_enabled"]
                or row["implementation"] != implementation or row["precision"] != precision
                or row["labels"] != labels or set(row["losses"]) != set(labels)
                or any(not math.isfinite(value) for value in row["losses"].values())):
            raise ValueError("incomplete forward or precision evidence")
        expected = ([] if row["implementation"] == "bfl" else
                    ["bfl_singletons", "diffusers_singletons"] if len(row["labels"]) == 2 else ["bfl_singletons"])
        if list(row["comparisons"]) != expected or list(row["comparison_summary"]) != expected:
            raise ValueError("missing independent or mask comparisons")
        for reference in expected:
            if (set(row["comparisons"][reference]) != set(labels)
                    or set(row["comparison_summary"][reference]) != set(labels)):
                raise ValueError("comparison omitted an example")
            for stages in row["comparisons"][reference].values():
                if list(stages) != plan["stages"]:
                    raise ValueError("comparison omitted a boundary")
                for stage, streams in stages.items():
                    required = ({"text", "image"} if stage.startswith(("double.", "single.")) else
                                {"conditioning"} if stage == "time_embedding" else
                                {"text"} if stage == "text_projection" else {"image"})
                    if set(streams) != required:
                        raise ValueError("comparison omitted a text/image stream")
                    for metrics in streams.values():
                        relative = metrics["relative_l2"]
                        if relative is not None and (not math.isfinite(relative) or relative < 0):
                            raise ValueError("invalid relative output error")
        expected_summary = forward.comparison_summary(row["comparisons"], row["losses"],
            {name: reference_losses[(precision, name)] for name in expected},
            plan["numerical_tolerances"][precision])
        if row["comparison_summary"] != expected_summary:
            raise ValueError("reported numerical summary disagrees with boundary metrics")
        if len(labels) == 1:
            reference_losses.setdefault((precision, f"{implementation}_singletons"), {}).update(row["losses"])
    return all(sample["passed"] for row in cases for comparison in row["comparison_summary"].values()
               for sample in comparison.values())


def run_worker(args):
    from diffusion_arc.training.single_gpu_setup import load_single_gpu_models, seed_single_gpu_process
    from scripts.diagnostics._flux_bfl_weights import build_bfl_model, validate_source

    result = dict(status="running", phase="configuration", mode="bfl_forward", experiment_version=1, cases=[])
    def persist(phase=None):
        if phase:
            result["phase"] = phase
        inputs_probe.shapes._write_json(args.output_dir / "result.json", result)
        print(PROGRESS_MARKER, result["phase"], flush=True)
    def on_case(row):
        if row["status"] == "complete":
            result["cases"].append(row)
        persist(f"{row['variant']}/{row['status']}")
    try:
        policy, training = load_probe_config(args.config)
        result.update(plan=build_plan(policy, training), bfl_source=validate_source(),
                      optimizer_initialized=False, optimizer_updates=0, backward_calls=0)
        device = inputs_probe.shapes.prepare_single_gpu_device(training)
        persist("load_existing_models")
        seed_single_gpu_process(training.seed)
        models = load_single_gpu_models(training, device=device)
        model = models.transformer
        if (model.config.attention_head_dim != 128 or len(model.transformer_blocks) != 5
                or len(model.single_transformer_blocks) != 20 or model.gradient_checkpointing
                or any(p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError("BFL probe requires the original 5+20 FLUX blocks with FP32 weights and no checkpointing")
        result.update(environment=inputs_probe.shapes._environment_metadata(), flow_seed=training.seed,
                      checkpoint_config=dict(model.config), checkpoint_weights_hashed=False)
        persist("prepare_real_inputs_once")
        stored = inputs_probe.prepare_inputs(policy.input_policy, training, models)
        parent = result["plan"]["source_group"]
        full = inputs_probe.group_inputs(stored[parent["name"]], parent, training.seed, device)
        pair = slice_inputs(full, 0, 2)
        data = forward.prepare_comparison_inputs(pair)
        if [p.shape[0] for p in data["prefixes"]] != result["plan"]["text_lengths"]:
            raise ValueError("prepared prefix lengths differ from the declared plan")
        result.update(input_identity=forward.input_identity(data), input_sha256=full["input_sha256"])
        persist("map_existing_weights_to_bfl")
        bfl, mapping = build_bfl_model(model)
        result["weight_mapping"] = mapping
        model.eval()
        persist("weight_mapping_verified")
        result["cases"] = run_comparisons(bfl, model, data, policy, profile=True, on_case=on_case)
        result["numerical_passed"] = validate_complete(result)
        result.update(status="complete", phase="complete")
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        persist()
    return result


def write_report(path, result):
    lines = ["# BFL / Diffusers forward diagnostic", "", f"Execution: **{result['status']}**.",
             f"Numerical screening passed: **{result.get('numerical_passed', 'not established')}**.", "",
             "Both precisions use Math SDPA, FP32 persistent parameters and disabled TF32. "
             "BF16 enables autocast while retaining each implementation's native dtype transitions.", "",
             "| Case | Reference | Example | Velocity rel. L2 | Maximum boundary rel. L2 | First boundary above tolerance | Passed |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    def percent(value):
        return "undefined" if value is None else f"{100 * value:.6f}%"
    for row in result.get("cases", []):
        for reference, samples in row["comparison_summary"].items():
            for label, sample in samples.items():
                lines.append(f"| {row['variant']} | {reference} | {label} | "
                    f"{percent(sample['velocity_relative_l2'])} | {percent(sample['maximum_relative_l2'])} | "
                    f"{sample['first_outside_tolerance'] or 'none'} | {sample['passed']} |")
    lines += ["", "All captured valid text/image streams and per-example flow losses are screened against "
              "the tolerances declared in the configuration. First nonidentical boundaries and full metrics "
              "are retained in summary.json. Source identity, weight mapping and input fingerprints are recorded.",
              "", "This establishes only the tested implementation and mask comparisons. It does not establish "
              "cuDNN/FA3 agreement, gradient/update agreement, checkpointing correctness, throughput, or learning quality. "
              "Shared errors in upstream checkpoint conversion remain outside this comparison."]
    if result.get("error"):
        lines += ["", f"Failure during `{result.get('phase')}`: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    from scripts.diagnostics._flux_bfl_weights import validate_source

    policy, training = load_probe_config(args.config)
    bfl_source = validate_source()
    metadata = source_metadata() if metadata is None else metadata
    if metadata.get("source_sha256") != source_hashes():
        raise ValueError("uploaded source hashes differ from launch metadata")
    hashes, data_hashes = config_hashes(args.config, policy), dataset_hashes(training)
    for key, actual in (("config_sha256", hashes), ("dataset_sha256", data_hashes)):
        if key in metadata and metadata[key] != actual:
            raise ValueError(f"uploaded {key} differs from launch metadata")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(metadata, config_sha256=hashes, dataset_sha256=data_hashes,
                    config=json.loads(args.config.read_text()), input_config=json.loads(policy.input_config.read_text()),
                    training_config=json.loads(policy.training_config.read_text()), bfl_source=bfl_source,
                    plan=build_plan(policy, training), effective_worker_seconds=policy.max_duration_seconds,
                    status="running", checkpoint_weights_hashed=False)
    inputs_probe.shapes._write_json(args.output_dir / "manifest.json", manifest)
    result, started = dict(status="error", phase="worker_start", cases=[]), time.monotonic()
    try:
        returncode, timed_out = inputs_probe.shapes.launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config),
             "--output-dir", str(args.output_dir)], log_path=args.output_dir / "worker.log",
            timeout_seconds=policy.max_duration_seconds, on_progress=on_progress, progress_marker=PROGRESS_MARKER)
        if (args.output_dir / "result.json").is_file():
            result = json.loads((args.output_dir / "result.json").read_text())
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error="worker exceeded its configured time budget")
        elif returncode != 0:
            if result.get("status") != "oom":
                result["status"] = "error"
            result.setdefault("error", "worker failed; inspect worker.log")
        elif result.get("status") == "complete":
            result["numerical_passed"] = validate_complete(result)
        else:
            result.update(status="error", error=result.get("error", "worker exited with incomplete results"))
    except Exception as error:
        partial = args.output_dir / "result.json"
        if partial.is_file():
            try:
                result = json.loads(partial.read_text())
            except (OSError, ValueError):
                pass
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        if result["status"] != "complete":
            result.pop("numerical_passed", None)
        manifest["status"] = result["status"]
        inputs_probe.shapes._write_json(args.output_dir / "summary.json", result)
        inputs_probe.shapes._write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", result)
        on_progress()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/diagnostics/flux_bfl.json")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan:
        from scripts.diagnostics._flux_bfl_weights import validate_source
        policy, training = load_probe_config(args.config, require_checkpoints=False)
        print(json.dumps(dict(plan=build_plan(policy, training), bfl_source=validate_source()), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for execution")
    result = run_worker(args) if args.worker else run_probe(args)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
