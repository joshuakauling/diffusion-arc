#!/usr/bin/env python3
"""Compare FA3 reduction cost, then bound packed capacity using whole real examples."""

from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import argparse
import gc
import json
import math
import statistics
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnostics import probe_flux_packing_performance as performance

packing = performance.packing
worker_tools = performance.worker_tools
evidence = performance.evidence
PROGRESS_MARKER = "FLUX_PACKED_CAPACITY_PROGRESS"


@dataclass(frozen=True)
class ProbeConfig:
    raw: dict
    packing_policy: object
    training: object
    config_paths: tuple
    checkpoint_paths: tuple
    max_duration_seconds: int


def load_probe_config(path, *, require_checkpoints=True):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    fixed = dict(schema_version=1, backend="fa3", comparison_group="medium_b4",
                 warmup_calls=2, rounds=4, measured_calls=5, confirmation_calls=3)
    if (not isinstance(raw, dict) or set(raw) != {*fixed, "packing_fixture", "families", "max_duration_seconds"}
            or any(type(raw[k]) is not type(v) or raw[k] != v for k, v in fixed.items())
            or not isinstance(raw["packing_fixture"], str) or not raw["packing_fixture"]
            or type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 480):
        raise ValueError("configuration differs from the approved capacity and reduction comparison")
    fixture = (path.parent / raw["packing_fixture"]).resolve()
    parent, training = packing.load_probe_config(fixture, require_checkpoints=require_checkpoints)
    windows = {w["name"]: w for w in parent.windows}
    families = raw["families"]
    if (not isinstance(families, list) or len(families) != 2
            or any(not isinstance(f, dict) for f in families)
            or [f.get("name") for f in families] != ["typical", "long_source"]):
        raise ValueError("exactly typical and long_source families are required")
    for family in families:
        if (set(family) != {"name", "window", "target_indices", "start_count", "flow_seed"}
                or family["window"] not in windows or type(family["flow_seed"]) is not int
                or family["flow_seed"] < 0 or type(family["start_count"]) is not int):
            raise ValueError("invalid capacity family")
        indices = family["target_indices"]
        if (not isinstance(indices, list) or not indices or len(indices) > 12
                or any(type(n) is not int or not 0 <= n < 48 for n in indices)
                or len(indices) != len(set(indices)) or not 1 <= family["start_count"] < len(indices)):
            raise ValueError("capacity families require distinct whole targets and a bounded growth sequence")
    source_group = next(g for g in parent.groups if g["name"] == raw["comparison_group"])
    first = families[0]
    if (first["window"] != source_group["window"] or first["start_count"] != 4
            or first["target_indices"][:4] != source_group["target_indices"]):
        raise ValueError("typical family must begin with the unchanged B2 comparison group")
    long = families[1]
    if windows[long["window"]]["target_prefix_lengths"][long["target_indices"][0]] != 8400 or long["start_count"] != 1:
        raise ValueError("long_source family must start with the 8400-token source")
    return ProbeConfig(raw, parent, training, (path, fixture, parent.training_config),
        (training.models.nvarc_checkpoint, training.models.flux_checkpoint), raw["max_duration_seconds"])


def build_plan(policy):
    windows = {w["name"]: w for w in policy.packing_policy.windows}
    families = []
    for family in policy.raw["families"]:
        lengths = windows[family["window"]]["target_prefix_lengths"]
        candidates = []
        for count in range(family["start_count"], len(family["target_indices"]) + 1):
            indices = family["target_indices"][:count]
            selected = [lengths[i] for i in indices]
            candidates.append(dict(name=f"{family['name']}_{count}", target_indices=indices,
                text_lengths=selected, batch_size=count, real_joint_tokens=sum(selected) + count * 900,
                padded_length=max(selected)))
        families.append({**family, "candidates": candidates})
    return dict(families=families, comparison=families[0]["candidates"][0], backend="fa3",
        comparison_reductions=["native", "fp32"], timed_comparison_calls=40, warmup_comparison_calls=4,
        round_orders=[["native", "fp32"] if r % 2 == 0 else ["fp32", "native"] for r in range(4)],
        capacity_reduction="fp32", confirmation_calls=3, max_capacity_candidates=sum(len(f["candidates"]) for f in families),
        max_worker_seconds=policy.max_duration_seconds, memory="resident frozen models, FP32 parameters/gradients/AdamW and preparation reserve",
        model_parameters="float32", compute="bfloat16", tf32=False, checkpointing=False,
        optimizer_updates_during_comparison_and_search=0, loss_weight="target count / 48",
        oom_policy="catch allocation OOM only; release failed graph, clear cache, then confirm last success",
        capacity_scope="tested whole-example growth sequences, not a universal token-exact limit",
        timing_scope="assembly + forward/loss + backward; excludes gradient reset, preparation and optimizer step")


def source_metadata():
    result = evidence.source_metadata()
    path = ROOT / "scripts/cloud/probe_flux_packed_capacity_modal.py"
    result["source_sha256"][str(path.relative_to(ROOT))] = evidence.file_sha256(path)
    return result


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): evidence.file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return packing.dataset_hashes(policy.training)


def prepare_pools(policy, models):
    # Reuse audited source replay; these groups select CPU pools, not a new B2 schedule.
    groups = tuple(dict(name=f["name"], window=f["window"], target_indices=f["target_indices"])
                   for f in policy.raw["families"])
    return packing.prepare_inputs(replace(policy.packing_policy, groups=groups), policy.training, models)


def candidate_data(pool, spec, seed, device):
    count = spec["batch_size"]
    selected = (pool[0][:count], pool[1][:count], pool[2])
    data = packing.group_inputs(selected, spec, seed, device)
    performance.validate_layout(data["layout"])
    return data


def call(model, optimizer, backend, spec, data, reduction):
    return performance.measure_call(model, optimizer, "fa3", spec, data, backend,
                                    compute_dtype=torch.bfloat16, modulation_reduction=reduction)


def check_reduction(observation, reduction):
    performance.validate_observation(observation, "fa3" if reduction == "fp32" else "sdpa_b2")


def run_comparison(policy, models, optimizer, backend, pool, result, persist):
    plan = result["plan"]
    spec = plan["comparison"]
    family = plan["families"][0]
    model = models.transformer
    data = candidate_data(pool, spec, family["flow_seed"], next(model.parameters()).device)
    identity = performance.input_identity(data)
    comparison = dict(spec=spec, input_identity=identity, cases=[], status="running")
    result["comparison"] = comparison
    reference = None
    for reduction in plan["comparison_reductions"]:
        case = dict(reduction=reduction, warmup=[], calls=[])
        comparison["cases"].append(case)
        persist(f"comparison/{reduction}/warmup")
        for _ in range(policy.raw["warmup_calls"]):
            with performance.observe_modulation_reduction() as observation:
                row, output = call(model, optimizer, backend, spec, data, reduction)
            check_reduction(observation, reduction)
            case["warmup"].append(row)
            case["modulation_observation"] = observation
        output_identity = performance.fingerprint(output)
        case["output_identity"] = output_identity
        if reference is None:
            reference = (output_identity, row["loss"])
        case["forward_identical"] = output_identity == reference[0] and row["loss"] == reference[1]
        del output
        if not case["forward_identical"]:
            raise ValueError("reduction comparison changed forward output or loss")
        case["gradient_health"] = performance.gradient_health(model)
    persist("comparison/fa3_backend_profile")
    comparison["backend_profile"] = performance.profile_backend(
        lambda: call(model, optimizer, backend, spec, data, "fp32"), model, "fa3", backend)
    cases = {case["reduction"]: case for case in comparison["cases"]}
    for round_index, order in enumerate(plan["round_orders"]):
        for reduction in order:
            persist(f"comparison/{reduction}/round_{round_index}")
            for call_index in range(policy.raw["measured_calls"]):
                row, output = call(model, optimizer, backend, spec, data, reduction)
                del output
                cases[reduction]["calls"].append(dict(round=round_index, call=call_index, **row))
    comparison.update(status="complete", inputs_unchanged=identity == performance.input_identity(data))
    persist("comparison/complete")


def attempt_candidate(models, optimizer, backend, pool, spec, seed, repeats):
    """Return no GPU tensors or exception objects, so a failed graph can be collected."""
    phase = "input_preparation"
    try:
        model = models.transformer
        data = candidate_data(pool, spec, seed, next(model.parameters()).device)
        identity = performance.input_identity(data)
        rows = []
        for _ in range(repeats):
            phase = "forward_backward"
            with performance.observe_modulation_reduction() as observation:
                row, output = call(model, optimizer, backend, spec, data, "fp32")
            del output
            check_reduction(observation, "fp32")
            rows.append(row)
        phase = "gradient_health"
        health = performance.gradient_health(model)
        if identity != performance.input_identity(data):
            raise ValueError("capacity input mutation")
        return dict(status="passed", spec=spec, calls=rows, gradient_health=health,
                    input_identity=identity, inputs_unchanged=True, modulation_observation=observation)
    except torch.OutOfMemoryError as error:
        if phase != "forward_backward":
            raise RuntimeError(f"OOM outside measured model execution ({phase}): {error}") from error
        # Do not retain the traceback or failed graph in the result.
        return dict(status="oom", spec=spec, phase=phase, error=str(error))


def recover_memory(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def search_family(family, attempt, recover, *, confirmation_calls, publish):
    """Grow by whole examples, stop at first OOM, and independently confirm the last fit."""
    result = dict(name=family["name"], status="running", attempts=[])
    last = None
    for spec in family["candidates"]:
        row = attempt(spec, 1)
        result["attempts"].append(row)
        publish(result)
        if row["status"] == "passed":
            last = spec
            continue
        if row["status"] != "oom":
            raise ValueError("invalid capacity attempt status")
        result["first_oom"] = row
        recover()
        if last is None:
            raise RuntimeError("initial capacity control did not fit; no lower bound established")
        break
    # Release successful-shape caches too, so confirmation includes a fresh allocation.
    recover()
    confirmation = attempt(last, confirmation_calls)
    result["confirmation"] = confirmation
    if confirmation["status"] != "passed":
        raise RuntimeError("last fitting pack failed confirmation after cleanup")
    result.update(status="bounded" if "first_oom" in result else "lower_bound_only", largest_passed=last)
    publish(result)
    return result


def worker(args):
    policy = load_probe_config(args.config)
    torch.set_num_threads(8)
    started = time.monotonic()
    result = dict(status="running", phase="configuration", plan=build_plan(policy), families=[])
    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(PROGRESS_MARKER, phase, flush=True)
    try:
        persist("fa3/preflight")
        device = worker_tools.prepare_single_gpu_device(policy.training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("cached image requires Hopper/SM90")
        backend = packing.FlashBackend("fa3")
        result["backend_preflight"] = packing.backend_preflight(backend, device, policy.packing_policy.numerical_tolerances)
        persist("load_models_and_training_memory")
        device, models, optimizer, reserve, setup = worker_tools.prepare_runtime(policy.training)
        model = models.transformer
        if model.config.attention_head_dim != 128 or models.flux_compute_dtype != torch.bfloat16:
            raise ValueError("capacity experiment requires BF16 compute and head dimension 128")
        with performance.training_precision(device):
            environment = worker_tools._environment_metadata()
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                           torch_cpu_threads=torch.get_num_threads())
        environment["packages"]["flash-attn-3"] = version("flash-attn-3")
        result.update(setup=setup, environment=environment, backend_binding=backend.binding,
            checkpoints={name: evidence.checkpoint_evidence(folder) for name, folder in (
                ("flux", policy.training.models.flux_checkpoint / "transformer"),
                ("vae", policy.training.models.flux_checkpoint / "vae"),
                ("nvarc", policy.training.models.nvarc_checkpoint))})
        versions = {n: p._version for n, p in model.named_parameters()}
        pointers = {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None}
        persist("prepare_real_inputs")
        pools = prepare_pools(policy, models)
        run_comparison(policy, models, optimizer, backend, pools["typical"], result, persist)
        recover_memory(device)
        for family in result["plan"]["families"]:
            position = len(result["families"])
            result["families"].append(dict(name=family["name"], status="running", attempts=[]))
            def publish(record):
                result["families"][position] = record
                persist(f"capacity/{family['name']}/{record['status']}/{len(record['attempts'])}")
            def attempt(spec, repeats):
                persist(f"capacity/{spec['name']}/{repeats}_calls")
                return attempt_candidate(models, optimizer, backend, pools[family["name"]],
                                         spec, family["flow_seed"], repeats)
            search_family(family, attempt, lambda: recover_memory(device),
                          confirmation_calls=policy.raw["confirmation_calls"], publish=publish)
        result.update(weights_unchanged=versions == {n: p._version for n, p in model.named_parameters()},
            gradient_storage_unchanged=pointers == {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None},
            status="complete")
        validate_complete(result, policy)
        persist("complete")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def validate_calls(calls, count):
    if len(calls) != count or any(not math.isfinite(c[k]) or c[k] <= 0 for c in calls for k in packing.TIMINGS):
        raise ValueError("incomplete or invalid timing calls")
    if any(not math.isfinite(c["loss"]) or c["peak_allocated_bytes"] <= 0 for c in calls):
        raise ValueError("invalid loss or memory evidence")


def validate_passed(row, spec, count):
    if (row.get("status") != "passed" or row.get("spec") != spec or row.get("inputs_unchanged") is not True
            or row.get("gradient_health", {}).get("finite") is not True):
        raise ValueError("invalid successful capacity evidence")
    check_reduction(row["modulation_observation"], "fp32")
    validate_calls(row["calls"], count)


def validate_complete(result, policy):
    plan = build_plan(policy)
    if result.get("status") != "complete" or result.get("plan") != plan:
        raise ValueError("incomplete capacity experiment or changed plan")
    if result.get("weights_unchanged") is not True or result.get("gradient_storage_unchanged") is not True:
        raise ValueError("weight or gradient storage changed")
    if set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"} or not result.get("backend_preflight", {}).get("passed"):
        raise ValueError("missing checkpoint or backend evidence")
    comparison = result.get("comparison", {})
    if (comparison.get("status") != "complete" or comparison.get("spec") != plan["comparison"]
            or comparison.get("inputs_unchanged") is not True
            or comparison.get("backend_profile", {}).get("verified") is not True
            or [c.get("reduction") for c in comparison.get("cases", [])] != ["native", "fp32"]):
        raise ValueError("incomplete reduction comparison")
    expected = {(r, c) for r in range(4) for c in range(5)}
    for case in comparison["cases"]:
        if case.get("forward_identical") is not True or case.get("gradient_health", {}).get("finite") is not True:
            raise ValueError("comparison forward/gradient health failed")
        check_reduction(case["modulation_observation"], case["reduction"])
        validate_calls(case["warmup"], 2)
        validate_calls(case["calls"], 20)
        if {(c["round"], c["call"]) for c in case["calls"]} != expected:
            raise ValueError("reduction timing round coverage changed")
    if len(result.get("families", [])) != len(plan["families"]):
        raise ValueError("capacity family coverage changed")
    for actual, family in zip(result["families"], plan["families"]):
        attempts = actual.get("attempts", [])
        if actual.get("name") != family["name"] or not attempts or len(attempts) > len(family["candidates"]):
            raise ValueError("invalid capacity family")
        for index, row in enumerate(attempts):
            if row.get("spec") != family["candidates"][index]:
                raise ValueError("capacity search changed whole-example growth order")
            if row["status"] == "oom":
                if (index != len(attempts) - 1 or row.get("phase") != "forward_backward"
                        or actual.get("first_oom") != row or actual.get("status") != "bounded"):
                    raise ValueError("invalid OOM boundary")
            else:
                validate_passed(row, family["candidates"][index], 1)
        bounded = attempts[-1]["status"] == "oom"
        if not bounded and (len(attempts) != len(family["candidates"]) or actual.get("status") != "lower_bound_only"):
            raise ValueError("unbounded result claimed as a maximum")
        successes = [r["spec"] for r in attempts if r["status"] == "passed"]
        if not successes or actual.get("largest_passed") != successes[-1]:
            raise ValueError("missing successful capacity control")
        validate_passed(actual["confirmation"], successes[-1], 3)


def summarize(result):
    rows = []
    for case in result.get("comparison", {}).get("cases", []):
        calls = case["calls"]
        if calls:
            rows.append(dict(reduction=case["reduction"], measured_calls=len(calls),
                medians={k: statistics.median(c[k] for c in calls) for k in packing.TIMINGS},
                round_medians={str(r): statistics.median(c["total_wall_ms"] for c in calls if c["round"] == r)
                               for r in sorted({c["round"] for c in calls})},
                peak_allocated_bytes=max(c["peak_allocated_bytes"] for c in calls),
                peak_reserved_bytes=max(c["peak_reserved_bytes"] for c in calls)))
    return rows


def write_report(path, result):
    rows = summarize(result)
    lines = ["# FA3 reduction performance and packed capacity", "", f"Status: **{result['status']}**.", "",
        "Timing includes assembly, forward/loss and weighted backward; excludes optimizer steps and input preparation.",
        "The comparison uses 11,659 joint tokens. Medians in milliseconds; allocated peaks in GiB.", "",
        "| Reduction | Samples | Forward | Backward | Total | Peak GiB |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        m = row["medians"]
        lines.append(f"| {row['reduction']} | {row['measured_calls']} | {m['forward_ms']:.3f} | {m['backward_ms']:.3f} | "
                     f"{m['total_wall_ms']:.3f} | {row['peak_allocated_bytes'] / 2**30:.3f} |")
    lines += ["", "| Family | Status | Largest confirmed tokens | First OOM tokens | Confirmed peak GiB |",
              "| --- | --- | ---: | ---: | ---: |"]
    for family in result.get("families", []):
        best = family.get("largest_passed", {}).get("real_joint_tokens", "—")
        bad = family.get("first_oom", {}).get("spec", {}).get("real_joint_tokens", "—")
        calls = family.get("confirmation", {}).get("calls", [])
        peak = f"{max(c['peak_allocated_bytes'] for c in calls) / 2**30:.3f}" if calls else "—"
        lines.append(f"| {family['name']} | {family['status']} | {best} | {bad} | {peak} |")
    lines += ["", "These are bounds for the recorded whole-example growth sequences, not universal token limits.",
        "FP32 weights, gradients, AdamW moments, frozen models and preparation reserve remain resident; checkpointing is off.",
        "Capacity uses FP32 modulation sums. CPU operator events verify FA3; CUDA events time GPU work."]
    if result.get("error"):
        lines += ["", f"Error during {result.get('phase')}: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")
    return rows


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    checks = {**source_metadata(), "config_sha256": config_hashes(args.config, policy), "dataset_sha256": dataset_hashes(policy)}
    if metadata is not None and any(metadata.get(k) != v for k, v in checks.items()):
        raise ValueError("uploaded source/config/data differs from launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**(metadata or {}), **checks, "config": policy.raw, "plan": build_plan(policy), "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start", families=[])
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
        rows = write_report(args.output_dir / "report.md", result)
        worker_tools._write_json(args.output_dir / "summary.json", {**result, "rows": rows})
        worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_packed_capacity.json")
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
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
