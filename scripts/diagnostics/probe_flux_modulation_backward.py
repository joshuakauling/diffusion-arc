#!/usr/bin/env python3
"""Isolate FP32 modulation-gradient reduction in the full BF16 packed FLUX model."""

from dataclasses import replace
from pathlib import Path
import argparse
import gc
import json
import math
import re
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnostics import probe_flux_backward_layouts as baseline
from scripts.diagnostics import _flux_backward_layouts as execution
from scripts.diagnostics._flux_backend_replay import fingerprint

PROGRESS_MARKER = "FLUX_MODULATION_BACKWARD_PROGRESS"
worker_tools = baseline.worker_tools
evidence = baseline.evidence
MODULATIONS = ("double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation")


def load_probe_config(path, *, require_checkpoints=True):
    raw = json.loads(Path(path).read_text())
    expected = dict(schema_version=1, backend="fa3", modulation_reduction="fp32_segment_sum", repetitions=2)
    if (not isinstance(raw, dict) or set(raw) != {*expected, "backward_fixture", "max_duration_seconds"}
            or any(raw[key] != value or type(raw[key]) is not type(value) for key, value in expected.items())
            or not isinstance(raw["backward_fixture"], str) or not raw["backward_fixture"]
            or type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 420):
        raise ValueError("configuration differs from the approved modulation intervention")
    parent_path = (Path(path).resolve().parent / raw["backward_fixture"]).resolve()
    parent = baseline.load_probe_config(parent_path, require_checkpoints=require_checkpoints)
    return replace(parent, raw={**parent.raw, **raw}, config_paths=(Path(path).resolve(), *parent.config_paths),
                   max_duration_seconds=raw["max_duration_seconds"])


def case_plan():
    definitions = (
        ("single", "fa3_single", "native", ()),
        ("packed_native_ab", "fa3_packed_ab", "native", ("single/0",)),
        ("packed_fp32_ab", "fa3_packed_ab", "fp32", ("single/0", "packed_native_ab/0")),
        ("packed_fp32_ba", "fa3_packed_ba", "fp32", ("single/0", "packed_fp32_ab/0")),
    )
    result = []
    for name, variant, reduction, references in definitions:
        for repeat in range(2):
            refs = list(references) if repeat == 0 else list(dict.fromkeys(("single/0", f"{name}/0")))
            result.append(dict(name=f"{name}/{repeat}", variant=variant, repeat=repeat,
                modulation_reduction=reduction, references=refs,
                backward_calls=2 if variant == "fa3_single" else 1,
                labels=["B", "A"] if variant.endswith("ba") else ["A", "B"]))
    return result


def build_plan(policy):
    return dict(cases=case_plan(), logical_batches=8, full_model_backward_calls=10,
        source_group=policy.raw["source_group"], text_lengths=policy.raw["text_lengths"], image_tokens=900,
        sample_weights={"A": 1 / 48, "B": 1 / 48}, backend="fa3", compute="bfloat16",
        model_parameters="float32", parameter_gradients="float32", forward_intervention="none",
        backward_intervention="three modulation ownership reductions: FP32 segment sums, one cast back",
        capture_blocks=policy.raw["capture_blocks"], edge_width=policy.raw["edge_width"],
        forward_invariance="exact output and captured activation fingerprints against packed native AB, by sample identity",
        profiler_evidence="CPU forward/backward backend operators; CUDA kernel-event capture not requested",
        optimizer_updates=0, gradient_clipping=False, checkpointing=False, kernel_replay_evaluations=0,
        gradient_references_resident_max=2, retries=0, model_downloads=False,
        max_worker_seconds=policy.max_duration_seconds, max_function_seconds=480, max_lifecycle_seconds=600,
        planned_budget_usd=1.0, estimated_compute_ceiling_usd=0.904728,
        comparison_policy="report effect on full gradients and modulation separately against unchanged-repeat controls",
        training_qualified=False)


def source_metadata():
    result = baseline.source_metadata()
    path = ROOT / "scripts/cloud/probe_flux_modulation_backward_modal.py"
    result["source_sha256"][str(path.relative_to(ROOT))] = evidence.file_sha256(path)
    return result


config_hashes = baseline.config_hashes
dataset_hashes = baseline.dataset_hashes


def forward_identity(states):
    return dict(velocity={label: fingerprint(value) for label, value in states["outputs"].items()},
        activations={stage: {label: fingerprint(value) for label, value in fields["output"].items()}
                     for stage, fields in states["boundaries"].items()})


def modulation_metrics(gradients):
    groups = {name: execution.metrics.aggregate(value for key, value in gradients["parameters"].items()
              if key.split(".")[0] == name) for name in MODULATIONS}
    groups["all_modulation"] = execution.metrics.aggregate(groups.values())
    groups["other_parameters"] = execution.metrics.aggregate(value for key, value in gradients["parameters"].items()
                                                            if key.split(".")[0] not in MODULATIONS)
    for metric in groups.values():
        metric["norm_ratio"] = math.sqrt(metric["actual_squared"] / metric["reference_squared"]) if metric["reference_squared"] else None
        metric["share_of_squared_error"] = metric["error_squared"] / gradients["total"]["error_squared"] if gradients["total"]["error_squared"] else 0.
    return groups


def validate_forward_identity(identity, policy):
    if (set(identity) != {"velocity", "activations"} or set(identity["velocity"]) != {"A", "B"}
            or set(identity["activations"]) != {*execution.BLOCKS, "text.mlp.linear_out"}):
        raise ValueError("incomplete forward identity evidence")
    for stage, samples in {"velocity": identity["velocity"], **identity["activations"]}.items():
        if set(samples) != {"A", "B"}:
            raise ValueError("forward identity lost sample ownership")
        for label, value in samples.items():
            text = policy.raw["text_lengths"]["AB".index(label)]
            shape = [900, 128] if stage == "velocity" else [text if stage == "text.mlp.linear_out" else text + 900, 3072]
            if (value.get("shape") != shape or value.get("dtype") not in ("torch.bfloat16", "torch.float32")
                    or re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")) is None):
                raise ValueError("forward fingerprint has invalid shape, dtype or hash")


def validate_complete(result, policy):
    plan = build_plan(policy)
    if result.get("status") != "complete" or result.get("plan") != plan:
        raise ValueError("modulation diagnostic incomplete or plan changed")
    if set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"}:
        raise ValueError("missing checkpoint identities")
    if result.get("weights_unchanged") is not True or result.get("inputs_unchanged") is not True:
        raise ValueError("input/weight invariance not established")
    if result.get("max_gradient_references_resident", 999) > 2:
        raise ValueError("gradient reference memory contract exceeded")
    cases = result.get("cases", [])
    if [row.get("name") for row in cases] != [spec["name"] for spec in plan["cases"]]:
        raise ValueError("missing or reordered gradient cases")
    native_forward = next(row["forward_identity"] for row in cases if row["name"] == "packed_native_ab/0")
    for row, spec in zip(cases, plan["cases"]):
        if (any(row.get(key) != spec[key] for key in ("variant", "repeat", "modulation_reduction", "backward_calls"))
                or row.get("sample_weights") != plan["sample_weights"] or row.get("dispatch_verified") is not True
                or row.get("parameter_count") != 3875544576 or row.get("gradient_tensors") != 169
                or set(row.get("comparisons", {})) != set(spec["references"])
                or not math.isfinite(row.get("gradient_norm", float("nan")))):
            raise ValueError("gradient case coverage, weighting or dispatch failed")
        execution.validate_dispatch(row["operators"], spec["variant"], spec["backward_calls"], 25)
        validate_forward_identity(row["forward_identity"], policy)
        reduction = row.get("modulation_reduction_evidence", {})
        expected_calls = 3 if spec["modulation_reduction"] == "fp32" else 0
        if reduction.get("forward_calls") != expected_calls or reduction.get("backward_calls") != expected_calls:
            raise ValueError("modulation intervention coverage mismatch")
        if expected_calls and (reduction.get("source_dtypes") != ["torch.bfloat16"] * 3
                               or reduction.get("reduction_dtype") != "torch.float32"):
            raise ValueError("modulation reduction precision mismatch")
        if row.get("modulation_reduction") == "fp32":
            if row.get("forward_identity") != native_forward or row.get("forward_unchanged") is not True:
                raise ValueError("modulation intervention changed forward values")
            if row["losses"] != next(value["losses"] for value in cases if value["name"] == "packed_native_ab/0"):
                raise ValueError("modulation intervention changed per-example losses")
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
            if comparison.get("modulation") != modulation_metrics(grads):
                raise ValueError("modulation summary does not match parameter metrics")
    if sum(row["backward_calls"] for row in cases) != 10:
        raise ValueError("unexpected full backward count")
    return True


def worker(args):
    policy = load_probe_config(args.config)
    result = dict(status="running", phase="preflight", plan=build_plan(policy), cases=[], checkpoints={},
                  max_gradient_references_resident=0)
    started = time.monotonic()

    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(f"{PROGRESS_MARKER} {phase} elapsed={result['worker_elapsed_seconds']:.1f}s", flush=True)

    try:
        device = baseline.prepare_single_gpu_device(policy.training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("diagnostic requires the pinned Hopper execution path")
        baseline.seed_single_gpu_process(policy.training.seed)
        torch.set_num_threads(8)
        result["environment"] = worker_tools._environment_metadata()
        for name, folder in (("flux", policy.training.models.flux_checkpoint / "transformer"),
                             ("nvarc", policy.training.models.nvarc_checkpoint),
                             ("vae", policy.training.models.flux_checkpoint / "vae")):
            persist(f"checkpoint/{name}")
            result["checkpoints"][name] = evidence.checkpoint_evidence(folder)
        persist("load_cached_models")
        models = baseline.load_single_gpu_models(policy.training, device=device)
        model = models.transformer
        if model.gradient_checkpointing or any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
            raise ValueError("checkpointing and nonzero dropout are outside this diagnostic")
        persist("prepare_frozen_inputs")
        stored = baseline.first_wave.prepare_frozen_inputs(policy.packing_policy, policy.training, models)
        selected = stored[policy.raw["source_group"]]
        singletons = baseline.first_wave.singleton_inputs(selected, seed=policy.training.seed, device=device)
        result["source_sha256"] = selected[2]
        del selected, stored, models
        gc.collect()
        torch.cuda.empty_cache()
        model.train().requires_grad_(True)
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        identity = [{key: fingerprint(value) for key, value in item.items() if isinstance(value, torch.Tensor)} for item in singletons]
        result["input_identity"] = identity
        lengths = dict(zip("AB", policy.raw["text_lengths"]))
        references, native_forward = {}, None
        specs = case_plan()
        for index, spec in enumerate(specs):
            persist(f"backward/{spec['name']}")
            row, states, _ = execution.run_case(model, singletons, spec["variant"],
                padded_length=policy.raw["padded_length"], global_batch_size=policy.raw["global_batch_size"],
                modulation_reduction=spec["modulation_reduction"])
            row.update(name=spec["name"], repeat=spec["repeat"], forward_identity=forward_identity(states))
            if row.pop("parameter_versions") != versions:
                raise ValueError("model weights changed between comparisons")
            if spec["name"] == "packed_native_ab/0":
                native_forward = row["forward_identity"]
            if spec["modulation_reduction"] == "fp32":
                row["forward_unchanged"] = row["forward_identity"] == native_forward
            for reference_name in spec["references"]:
                persist(f"compare/{spec['name']}/{reference_name}")
                reference = references[reference_name]
                gradients = execution.compare_gradients(model, reference["gradients"])
                row["comparisons"][reference_name] = dict(gradients=gradients, modulation=modulation_metrics(gradients),
                    **execution.compare_states(states, reference["states"], lengths, edge_width=policy.raw["edge_width"]),
                    loss_relative=abs(row["logical_mean_loss"] - reference["loss"]) / max(abs(reference["loss"]), 1e-30))
            future = execution.retained_after(specs, index)
            for key in list(references):
                if key not in future:
                    del references[key]
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
    lines = ["# FLUX modulation backward intervention", "", f"Status: **{result['status']}**.", "",
             "Only the three modulation ownership sums change; BF16 forward values must remain exact.",
             "All gradients use the same two-example weighted flow loss. No optimizer or throughput qualification.", "",
             "| Case / reference | Gradient relative L2 | Cosine | Modulation relative L2 | Other parameters |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in result.get("cases", []):
        for name, comparison in row.get("comparisons", {}).items():
            metric, groups = comparison["gradients"]["total"], comparison["modulation"]
            lines.append(f"| {row['name']} / {name} | {100 * metric['relative_l2']:.6f}% | {metric['cosine']:.9f} | "
                         f"{100 * groups['all_modulation']['relative_l2']:.6f}% | {100 * groups['other_parameters']['relative_l2']:.6f}% |")
    if result.get("error"):
        lines += ["", "Error: " + result["error"]]
    lines += ["", "Backend evidence is CPU forward/backward operator dispatch; CUDA kernel-event capture was not requested.",
              "Exact forward fingerprints, full parameter/component comparisons and passive boundary observations are in `summary.json`.",
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_modulation_backward.json")
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
