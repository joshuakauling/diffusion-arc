#!/usr/bin/env python3
"""Bounded complete-gradient comparison of FLUX singleton, padding and packing."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import json
import math
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_arc.training.single_gpu_setup import load_single_gpu_models, prepare_single_gpu_device, seed_single_gpu_process
from scripts.diagnostics import probe_attention_execution as first_wave
from scripts.diagnostics import probe_flux_packing as packing
from scripts.diagnostics import probe_flux_backend_conditions as evidence
from scripts.diagnostics import probe_flux_batch_shapes as worker_tools
from scripts.diagnostics import _flux_backward_layouts as execution
from scripts.diagnostics._flux_backend_replay import fingerprint

PROGRESS_MARKER = "FLUX_BACKWARD_LAYOUTS_PROGRESS"


@dataclass(frozen=True)
class ProbeConfig:
    raw: dict
    fixture: Path
    packing_policy: object
    training: object
    config_paths: tuple
    checkpoint_paths: tuple
    max_duration_seconds: int


def load_probe_config(path, *, require_checkpoints=True):
    raw = json.loads(Path(path).read_text())
    expected = dict(schema_version=1, source_group="short_b4", selected_labels=["A", "B"],
                    text_lengths=[468, 406], padded_length=487, global_batch_size=48,
                    repetitions=2, edge_width=32, capture_blocks=list(execution.BLOCKS))
    if (not isinstance(raw, dict) or set(raw) != {*expected, "packing_fixture", "max_duration_seconds"}
            or any(raw[key] != value or type(raw[key]) is not type(value) for key, value in expected.items())
            or any(any(type(item) is not type(template) for item, template in zip(raw[key], value))
                   for key, value in expected.items() if isinstance(value, list))
            or not isinstance(raw["packing_fixture"], str) or not raw["packing_fixture"]
            or type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 840):
        raise ValueError("configuration differs from approved backward layout matrix")
    fixture = (Path(path).resolve().parent / raw["packing_fixture"]).resolve()
    parent, training = packing.load_probe_config(fixture, require_checkpoints=require_checkpoints)
    groups = tuple(group for group in parent.groups if group["name"] == raw["source_group"])
    if len(groups) != 1 or training.loop.optimizer_batch_size != raw["global_batch_size"]:
        raise ValueError("historical source group or production weighting changed")
    selected = replace(parent, groups=groups, windows=tuple(w for w in parent.windows if w["name"] == groups[0]["window"]))
    planned = packing.build_plan(selected, training)[0]
    if planned["text_lengths"][:2] != raw["text_lengths"] or planned["padded_length"] != raw["padded_length"]:
        raise ValueError("historical inputs changed")
    return ProbeConfig(raw, fixture, selected, training, (Path(path).resolve(), fixture, parent.training_config),
                       (training.models.nvarc_checkpoint, training.models.flux_checkpoint), raw["max_duration_seconds"])


def build_plan(policy):
    return dict(cases=execution.case_plan(), full_model_backward_calls=22, logical_batches=16,
                source_group=policy.raw["source_group"], text_lengths=policy.raw["text_lengths"], image_tokens=900,
                padded_length=policy.raw["padded_length"], sample_weights={"A": 1 / 48, "B": 1 / 48},
                capture_blocks=policy.raw["capture_blocks"], operation_boundary="text.mlp.linear_out", edge_width=32,
                kernel_replay_evaluations=72, kernel_replay_upstream="real weighted singleton flow loss",
                max_worker_seconds=policy.max_duration_seconds, max_function_seconds=960, max_lifecycle_seconds=1200,
                model_parameters="float32", parameter_gradients="float32", compute="bfloat16",
                optimizer_updates=0, gradient_clipping=False, checkpointing=False, retries=0, model_downloads=False,
                estimated_compute_ceiling_usd=1.809456, planned_budget_usd=2.0, user_balance_usd=2.75,
                gradient_references_resident_max=2, training_qualified=False,
                sdpa_packing="per-example cuDNN calls inside packed model; not native ragged",
                comparison_policy="report magnitudes, directions, repeatability and boundary enrichment; no new acceptance threshold")


def source_metadata():
    result = evidence.source_metadata()
    for name in ("probe_flux_backward_layouts_modal.py", "probe_attention_execution_modal.py"):
        path = ROOT / "scripts/cloud" / name
        result["source_sha256"][str(path.relative_to(ROOT))] = evidence.file_sha256(path)
    return result


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): evidence.file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return packing.dataset_hashes(policy.training)


def validate_complete(result, policy):
    plan = build_plan(policy)
    if result.get("status") != "complete" or result.get("plan") != plan:
        raise ValueError("backward diagnostic incomplete or plan changed")
    if set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"}:
        raise ValueError("missing checkpoint identities")
    if result.get("weights_unchanged") is not True or result.get("inputs_unchanged") is not True:
        raise ValueError("input/weight invariance not established")
    if result.get("max_gradient_references_resident", 999) > 2:
        raise ValueError("gradient reference memory contract exceeded")
    cases = result.get("cases", [])
    if [case.get("name") for case in cases] != [case["name"] for case in plan["cases"]]:
        raise ValueError("missing or reordered gradient cases")
    for row, spec in zip(cases, plan["cases"]):
        if (row.get("variant") != spec["variant"] or row.get("backward_calls") != spec["backward_calls"]
                or row.get("sample_weights") != plan["sample_weights"] or row.get("dispatch_verified") is not True
                or row.get("parameter_count") != 3875544576 or row.get("gradient_tensors") != 169
                or set(row.get("comparisons", {})) != set(spec["references"])
                or not math.isfinite(row.get("gradient_norm", float("nan")))):
            raise ValueError("gradient case coverage, weighting or dispatch failed")
        execution.validate_dispatch(row["operators"], spec["variant"], spec["backward_calls"], 25)
        if any(check["nonzero_gradient_elements"] != 0 for check in row["padding_checks"]):
            raise ValueError("nonzero padding input gradients")
        for comparison in row["comparisons"].values():
            grads = comparison["gradients"]
            if grads["total"]["count"] != row["parameter_count"] or len(grads["parameters"]) != row["gradient_tensors"]:
                raise ValueError("incomplete full gradient comparison")
            reconstructed = execution.metrics.aggregate(grads["parameters"].values())
            for key in ("error_squared", "reference_squared", "actual_squared", "dot", "count", "max_abs", "rms", "relative_l2", "cosine"):
                if reconstructed[key] != grads["total"][key]:
                    raise ValueError("gradient summary does not match parameter metrics")
            if set(comparison["boundaries"]) != {*execution.BLOCKS, "text.mlp.linear_out"}:
                raise ValueError("missing backward boundary observations")
            for fields in comparison["boundaries"].values():
                if set(fields) != {"output", "gradient"} or any(set(samples) != {"A", "B"} for samples in fields.values()):
                    raise ValueError("missing sample or backward boundary")
            for stage, fields in comparison["boundaries"].items():
                for samples in fields.values():
                    for label, metric in samples.items():
                        length = policy.raw["text_lengths"]["AB".index(label)]
                        count = length if stage == "text.mlp.linear_out" else length + 900
                        if metric["sequence_length"] != count or metric["text_length"] != length or metric["edge_width"] != 32:
                            raise ValueError("token boundary identity mismatch")
    if sum(row["backward_calls"] for row in cases) != 22:
        raise ValueError("unexpected full backward count")
    replay = result.get("kernel_boundaries", {})
    if (replay.get("status") != "complete" or replay.get("evaluations_completed") != 72
            or replay.get("evaluations_attempted") != 72 or set(replay.get("stages", {})) != set(execution.BLOCKS)):
        raise ValueError("kernel boundary replay incomplete")
    profiles = 0
    for stage in execution.BLOCKS:
        backends = replay["stages"][stage]["backends"]
        if set(backends) != {"cudnn", "fa3"}:
            raise ValueError("kernel replay omitted a backend")
        for name, row in backends.items():
            layouts = {"singletons", "ab", "ba", *(["padded"] if name == "cudnn" else [])}
            expected_comparisons = {"ab_vs_singletons", "ba_vs_singletons", "ba_vs_ab", *(["padded_vs_singletons"] if name == "cudnn" else [])}
            if [value["repetition"] for value in row["executions"]] != [0, 1] or set(row["repeat_comparisons"]) != layouts:
                raise ValueError("kernel repeat coverage failed")
            for value in row["executions"]:
                if set(value["comparisons"]) != expected_comparisons:
                    raise ValueError("kernel layout comparison missing")
                for dispatch in value["dispatch"].values():
                    if dispatch["profiled"]:
                        profiles += 1
                        if dispatch.get("backend_verified") is not True or not dispatch.get("cuda_kernel_names"):
                            raise ValueError("missing profiled CUDA kernel evidence")
    if profiles != 3:
        raise ValueError("kernel replay profiler coverage failed")
    return True


def worker(args):
    from scripts.diagnostics._flux_boundary_replay import replay_boundaries

    policy = load_probe_config(args.config)
    result = dict(status="running", phase="preflight", plan=build_plan(policy), cases=[], checkpoints={},
                  max_gradient_references_resident=0)
    started = time.monotonic()

    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(f"{PROGRESS_MARKER} {phase} elapsed={result['worker_elapsed_seconds']:.1f}s", flush=True)

    try:
        device = prepare_single_gpu_device(policy.training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("diagnostic requires the pinned Hopper execution path")
        seed_single_gpu_process(policy.training.seed)
        torch.set_num_threads(8)
        result["environment"] = worker_tools._environment_metadata()
        for name, folder in (("flux", policy.training.models.flux_checkpoint / "transformer"),
                             ("nvarc", policy.training.models.nvarc_checkpoint),
                             ("vae", policy.training.models.flux_checkpoint / "vae")):
            persist(f"checkpoint/{name}")
            result["checkpoints"][name] = evidence.checkpoint_evidence(folder)
        persist("load_cached_models")
        models = load_single_gpu_models(policy.training, device=device)
        model = models.transformer
        if model.gradient_checkpointing or any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
            raise ValueError("checkpointing and nonzero dropout are outside this diagnostic")
        persist("prepare_frozen_inputs")
        stored = first_wave.prepare_frozen_inputs(policy.packing_policy, policy.training, models)
        selected = stored[policy.raw["source_group"]]
        singletons = first_wave.singleton_inputs(selected, seed=policy.training.seed, device=device)
        if not torch.equal(singletons[0]["image_positions"], singletons[1]["image_positions"]):
            raise ValueError("native batch requires the same image position system for both samples")
        result["source_sha256"] = selected[2]
        del selected, stored, models
        gc.collect()
        torch.cuda.empty_cache()
        model.train().requires_grad_(True)
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        identity = [{key: fingerprint(value) for key, value in item.items() if isinstance(value, torch.Tensor)} for item in singletons]
        result["input_identity"] = identity
        lengths = dict(zip("AB", policy.raw["text_lengths"]))
        references, kernel_records = {}, {}
        specs = execution.case_plan()
        for index, spec in enumerate(specs):
            persist(f"backward/{spec['name']}")
            row, states, captured = execution.run_case(model, singletons, spec["variant"],
                padded_length=policy.raw["padded_length"], global_batch_size=policy.raw["global_batch_size"],
                retain_attention=spec["name"] == "cudnn_single/0")
            row.update(name=spec["name"], repeat=spec["repeat"])
            if row.pop("parameter_versions") != versions:
                raise ValueError("model weights changed between comparisons")
            if captured:
                kernel_records = captured
            for reference_name in spec["references"]:
                persist(f"compare/{spec['name']}/{reference_name}")
                reference = references[reference_name]
                row["comparisons"][reference_name] = {
                    "gradients": execution.compare_gradients(model, reference["gradients"]),
                    **execution.compare_states(states, reference["states"], lengths, edge_width=policy.raw["edge_width"]),
                    "loss_relative": abs(row["logical_mean_loss"] - reference["loss"]) / max(abs(reference["loss"]), 1e-30),
                }
            future = execution.retained_after(specs, index)
            for key in list(references):
                if key not in future:
                    del references[key]
            # Loop locals must not keep expired 16-GB snapshots alive.
            if spec["references"]:
                del reference
            if spec["name"] in future:
                references[spec["name"]] = dict(gradients=execution.gradient_snapshot(model), states=states,
                                                 loss=row["logical_mean_loss"])
            result["max_gradient_references_resident"] = max(result["max_gradient_references_resident"], len(references))
            if len(references) > 2:
                raise ValueError("too many resident parameter gradient references")
            result["cases"].append(row)
            del states
            persist(f"completed/{spec['name']}")
        references.clear()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        gc.collect()
        torch.cuda.empty_cache()
        persist("kernel_boundary_replays")
        def replay_progress(partial):
            result["kernel_boundaries"] = partial
            persist("kernel_boundary_progress")
        result["kernel_boundaries"] = replay_boundaries(kernel_records, text_lengths=lengths, device=device,
            padded_length=policy.raw["padded_length"], edge_width=policy.raw["edge_width"], on_result=replay_progress)
        result["weights_unchanged"] = versions == {name: parameter._version for name, parameter in model.named_parameters()}
        result["inputs_unchanged"] = identity == [
            {key: fingerprint(value) for key, value in item.items() if isinstance(value, torch.Tensor)} for item in singletons]
        result["status"] = "complete"
        validate_complete(result, policy)
        persist("complete")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def write_report(path, result):
    lines = ["# FLUX backward layout diagnostic", "", f"Status: **{result['status']}**.", "",
             "All populated parameter gradients of the same two-example weighted flow loss.",
             "No optimizer updates, checkpointing or training-throughput qualification.", "",
             "| Case / reference | Gradient relative L2 | Cosine | Norm ratio |", "| --- | ---: | ---: | ---: |"]
    for row in result.get("cases", []):
        for name, comparison in row.get("comparisons", {}).items():
            metric = comparison["gradients"]["total"]
            lines.append(f"| {row['name']} / {name} | {100 * metric['relative_l2']:.6f}% | {metric['cosine']:.9f} | {metric['norm_ratio']:.6f} |")
    if result.get("error"):
        lines += ["", "Error: " + result["error"]]
    lines += ["", "Per-token forward/backward boundary profiles, parameter/component metrics,",
              "padding-gradient checks and dispatch/source identities are in `summary.json`.",
              f"Worker/supervisor seconds: {result.get('worker_elapsed_seconds')} / {result.get('elapsed_seconds')}."]
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    checks = {**source_metadata(), "config_sha256": config_hashes(args.config, policy), "dataset_sha256": dataset_hashes(policy)}
    if metadata is not None and any(metadata.get(key) != value for key, value in checks.items()):
        raise ValueError("uploaded source/config/data differs from launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**(metadata or {}), **checks, "config": policy.raw, "plan": build_plan(policy), "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start", cases=[])
    try:
        returncode, timed_out = worker_tools.launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config), "--output-dir", str(args.output_dir)],
            log_path=args.output_dir / "worker.log", timeout_seconds=policy.max_duration_seconds,
            on_progress=on_progress, progress_marker=PROGRESS_MARKER)
        partial = args.output_dir / "result.json"
        if partial.is_file():
            result = json.loads(partial.read_text())
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error=f"worker exceeded {policy.max_duration_seconds} seconds")
        elif returncode != 0 or result.get("status") != "complete":
            result.update(status="error", error=result.get("error", "worker exited without complete evidence"))
        else:
            validate_complete(result, policy)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = result["status"]
        worker_tools._write_json(args.output_dir / "summary.json", result)
        worker_tools._write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_backward_layouts.json")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(load_probe_config(args.config, require_checkpoints=False)), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    result = worker(args) if args.worker else run_probe(args)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
