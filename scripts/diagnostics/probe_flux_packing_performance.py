#!/usr/bin/env python3
"""Measure unchanged B2 microbatches with padded cuDNN and fully packed FlashAttention."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
import argparse
import gc
import json
import math
import sys
import time
import traceback

import torch
from torch.nn.attention import SDPBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_arc.model.flow_matching import flow_matching_loss
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from scripts.diagnostics import probe_flux_packing as packing
from scripts.diagnostics import probe_flux_backend_conditions as evidence
from scripts.diagnostics._flux_backward_layouts import training_precision
from scripts.diagnostics._flux_backend_replay import fingerprint
from scripts.diagnostics._flux_modulation_backward import observe_modulation_reduction
from scripts.diagnostics._flux_packing_correctness import observed_sdpa_backend
from scripts.diagnostics._flux_packing_preflight import validate_layout

PROGRESS_MARKER = "FLUX_PACKING_PERFORMANCE_PROGRESS"
worker_tools = packing.shapes


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
    fixed = dict(schema_version=1, backends=["fa2", "fa3", "fa4"], modulation_reduction="fp32")
    if (not isinstance(raw, dict) or set(raw) != {*fixed, "packing_fixture", "groups", "warmup_calls",
                                                "measured_calls", "rounds", "max_duration_seconds"}
            or any(type(raw[k]) is not type(v) or raw[k] != v for k, v in fixed.items())
            or not isinstance(raw["packing_fixture"], str) or not raw["packing_fixture"]):
        raise ValueError("configuration differs from the approved performance comparison")
    for key, low, high in (("warmup_calls", 1, 2), ("measured_calls", 3, 5), ("rounds", 2, 3),
                           ("max_duration_seconds", 1, 420)):
        if type(raw[key]) is not int or not low <= raw[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    fixture = (path.parent / raw["packing_fixture"]).resolve()
    parent, training = packing.load_probe_config(fixture, require_checkpoints=require_checkpoints)
    names = raw["groups"]
    catalog = {group["name"]: group for group in parent.groups}
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or not set(names) <= catalog.keys()):
        raise ValueError("groups must select distinct existing B2 fixture groups")
    selected = replace(parent, groups=tuple(catalog[n] for n in names), backends=tuple(raw["backends"]),
        **{k: raw[k] for k in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds")})
    packing.build_plan(selected, training)
    return ProbeConfig(raw, selected, training, (path, fixture, parent.training_config),
        (training.models.nvarc_checkpoint, training.models.flux_checkpoint), raw["max_duration_seconds"])


def build_plan(policy):
    groups = packing.build_plan(policy.packing_policy, policy.training)
    return dict(groups=groups, variants=["sdpa_b2", *policy.raw["backends"]],
        timed_calls=len(groups) * 4 * policy.raw["rounds"] * policy.raw["measured_calls"],
        warmup_calls=len(groups) * 4 * policy.raw["warmup_calls"], backend_profiles=4,
        timing="assembly (including layout and processor setup), forward/loss, weighted backward",
        excluded="input preparation, gradient reset, optimizer step, gradient clipping, diagnostics",
        optimizer_batch_size=48, loss_weight="physical batch size / 48", optimizer_learning_rate=0.0,
        model_parameters="float32", parameter_gradients="float32", compute="bfloat16", tf32=False,
        modulation_reduction="fp32", checkpointing=False, optimizer_updates_during_comparisons=0,
        memory="frozen models, FP32 parameters/gradients/AdamW moments, preparation reserve",
        cuda_kernel_event_capture=False, backend_evidence="CPU forward/backward operator dispatch",
        max_worker_seconds=policy.max_duration_seconds, training_qualified=False, capacity_search=False)


def source_metadata():
    result = evidence.source_metadata()
    path = ROOT / "scripts/cloud/probe_flux_packing_performance_modal.py"
    result["source_sha256"][str(path.relative_to(ROOT))] = evidence.file_sha256(path)
    return result


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): evidence.file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return packing.dataset_hashes(policy.training)


def measure_call(model, optimizer, variant, group, data, backend, *, compute_dtype,
                 modulation_reduction="fp32"):
    """Time useful work, rebuilding layout metadata and processors for every packed call."""
    device = data["clean"].device
    optimizer.zero_grad(set_to_none=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    with training_precision(device):
        started = time.perf_counter()
        if backend is None:
            assembled = pad_flux_text_prefixes(data["prefixes"], padded_length=group["padded_length"])
            current = data
            route = nullcontext()
        else:
            assembled = torch.cat(data["prefixes"])
            layout = packing.packing_layout(tuple(group["text_lengths"]), 900, device)
            current = {**data, "layout": layout}
            route = packing.packed_processors(model, layout, backend)
        with route:
            torch.cuda.synchronize(device)
            assembled_at = time.perf_counter()
            events[0].record()
            output = packing.forward_velocity(model, variant, assembled, current, compute_dtype,
                                               modulation_reduction=modulation_reduction)
            loss = flow_matching_loss(output, data["flow"].target_velocity)
            events[1].record()
            (loss * (group["batch_size"] / 48)).backward()
            events[2].record()
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
    loss_value = loss.detach().item()
    if not math.isfinite(loss_value):
        raise FloatingPointError("non-finite FLUX loss")
    return dict(assembly_ms=(assembled_at - started) * 1000,
        forward_ms=events[0].elapsed_time(events[1]), backward_ms=events[1].elapsed_time(events[2]),
        forward_backward_ms=events[0].elapsed_time(events[2]), total_wall_ms=(finished - started) * 1000,
        loss=loss_value, **worker_tools._memory(device)), output.detach()


def gradient_health(model):
    """Check complete FP32 trainable-gradient coverage outside timed calls."""
    count = elements = 0
    finite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dtype != torch.float32 or parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise ValueError(f"missing FP32 parameter/gradient: {name}")
        finite.append(torch.isfinite(parameter.grad).all())
        count += 1
        elements += parameter.numel()
    if not count or not torch.stack(finite).all().item():
        raise FloatingPointError("non-finite or absent trainable gradients")
    return dict(tensors=count, elements=elements, dtype="torch.float32", finite=True)


def validate_observation(observation, variant):
    expected = 0 if variant == "sdpa_b2" else 3
    if (observation.get("forward_calls") != expected or observation.get("backward_calls") != expected
            or (expected and (observation.get("reduction_dtype") != "torch.float32"
                             or observation.get("source_dtypes") != ["torch.bfloat16"] * 3))):
        raise ValueError("FP32 modulation reduction coverage mismatch")


def validate_backend_evidence(counts, variant, layers):
    """Count dispatched kernels; autocast can nest the generic SDPA entry twice."""
    if variant == "sdpa_b2":
        if (observed_sdpa_backend(counts, layers) != SDPBackend.CUDNN_ATTENTION
                or counts.get("aten::_scaled_dot_product_cudnn_attention_backward") != layers):
            raise ValueError(f"cuDNN forward/backward dispatch not established: {counts}")
    else:
        packing.validate_backend_events(counts, variant, layers)


def profile_backend(call, model, variant, backend):
    if backend is not None:
        backend.observe = True
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            _, output = call()
        del output
    finally:
        if backend is not None:
            backend.observe = False
    counts = {e.key: e.count for e in profiler.key_averages()
              if any(s in e.key.lower() for s in ("scaled_dot_product", "attention", "flash", "flux_packing::"))}
    layers = len(model.transformer_blocks) + len(model.single_transformer_blocks)
    validate_backend_evidence(counts, variant, layers)
    return dict(verified=True, operators=counts, cuda_kernel_events_requested=False)


def input_identity(data):
    return dict(prefixes=[fingerprint(t) for t in data["prefixes"]],
        flow={k: fingerprint(getattr(data["flow"], k)) for k in ("noisy_latents", "timesteps", "target_velocity")})


def worker(args):
    policy = load_probe_config(args.config)
    torch.set_num_threads(8)
    parent, training = policy.packing_policy, policy.training
    result = dict(status="running", phase="configuration", plan=build_plan(policy), groups=[])
    started = time.monotonic()
    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(PROGRESS_MARKER, phase, flush=True)
    try:
        persist("backend_preflight")
        device = worker_tools.prepare_single_gpu_device(training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("cached performance image requires Hopper/SM90")
        backends = {}
        result["backend_preflights"] = {}
        for name in parent.backends:
            persist(f"{name}/small_forward_backward_preflight")
            backends[name] = packing.FlashBackend(name)
            result["backend_preflights"][name] = packing.backend_preflight(backends[name], device, parent.numerical_tolerances)
        persist("load_models_and_training_memory")
        device, models, optimizer, reserve, setup = worker_tools.prepare_runtime(training)
        model = models.transformer
        if model.config.attention_head_dim != 128 or models.flux_compute_dtype != torch.bfloat16:
            raise ValueError("performance comparison requires BF16 compute and head dimension 128")
        with training_precision(device):
            environment = worker_tools._environment_metadata()
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory,
                           torch_cpu_threads=torch.get_num_threads())
        environment["packages"].update({n: version(n) for n in ("flash-attn", "flash-attn-3", "flash-attn-4")})
        result.update(setup=setup, backend_bindings={n: b.binding for n, b in backends.items()},
            environment=environment,
            checkpoints={name: evidence.checkpoint_evidence(folder) for name, folder in (
                ("flux", training.models.flux_checkpoint / "transformer"),
                ("vae", training.models.flux_checkpoint / "vae"), ("nvarc", training.models.nvarc_checkpoint))})
        versions = {n: p._version for n, p in model.named_parameters()}
        gradient_pointers = {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None}
        persist("prepare_real_inputs")
        stored = packing.prepare_inputs(parent, training, models)
        result["backend_profiles"] = {}
        for index, spec in enumerate(result["plan"]["groups"]):
            group = {**spec, "status": "running", "cases": []}
            result["groups"].append(group)
            data = packing.group_inputs(stored[group["name"]], group, training.seed + index, device)
            validate_layout(data["layout"])
            group.update(flow_seed=training.seed + index, input_identity=input_identity(data),
                         input_sha256=data["input_sha256"])
            cases = {v: dict(variant=v, status="running", calls=[], warmup=[]) for v in result["plan"]["variants"]}
            group["cases"] = list(cases.values())
            reference = None
            for variant, case in cases.items():
                persist(f"{group['name']}/{variant}/warmup")
                backend = backends.get(variant)
                def call():
                    return measure_call(model, optimizer, variant, group, data, backend,
                                        compute_dtype=models.flux_compute_dtype)
                for _ in range(parent.warmup_calls):
                    with observe_modulation_reduction() as observation:
                        row, output = call()
                    validate_observation(observation, variant)
                    case["warmup"].append(row)
                    case["modulation_observation"] = observation
                case["gradient_health"] = gradient_health(model)
                if variant == "sdpa_b2":
                    reference = (output.cpu().clone(), row["loss"])
                velocity = packing.tensor_difference([(output, reference[0])])
                loss_relative = abs(row["loss"] - reference[1]) / max(abs(reference[1]), 1e-12)
                case["comparison"] = dict(velocity=velocity, loss_relative=loss_relative,
                    passed=velocity["relative_l2"] <= parent.numerical_tolerances["velocity_relative_l2"]
                           and loss_relative <= parent.numerical_tolerances["loss_relative"],
                    full_gradient_comparison=False, training_equivalence_established=False)
                del output
                if not case["comparison"]["passed"]:
                    raise ValueError(f"{variant} output/loss screening failed")
                if index == 0:
                    persist(f"{group['name']}/{variant}/backend_profile")
                    result["backend_profiles"][variant] = profile_backend(call, model, variant, backend)
            del reference
            for round_index, order in enumerate(group["round_orders"]):
                for variant in order:
                    persist(f"{group['name']}/{variant}/timed_round_{round_index}")
                    for call_index in range(parent.measured_calls):
                        row, output = measure_call(model, optimizer, variant, group, data, backends.get(variant),
                                                   compute_dtype=models.flux_compute_dtype)
                        del output
                        cases[variant]["calls"].append(dict(round=round_index, call=call_index, phase="measured", **row))
            for case in cases.values():
                case["status"] = "complete"
            group.update(status="complete", inputs_unchanged=group["input_identity"] == input_identity(data))
            del data
            gc.collect()
            persist(f"completed/{group['name']}")
        result["weights_unchanged"] = versions == {n: p._version for n, p in model.named_parameters()}
        result["gradient_storage_unchanged"] = gradient_pointers == {
            n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None}
        result["status"] = "complete"
        validate_complete(result, policy)
        persist("complete")
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def validate_complete(result, policy):
    plan = build_plan(policy)
    if result.get("status") != "complete" or result.get("plan") != plan:
        raise ValueError("performance evidence incomplete or plan changed")
    if result.get("weights_unchanged") is not True or result.get("gradient_storage_unchanged") is not True:
        raise ValueError("weight/gradient storage invariance not established")
    if set(result.get("backend_profiles", {})) != set(plan["variants"]) or any(
            p.get("verified") is not True for p in result["backend_profiles"].values()):
        raise ValueError("missing backend dispatch evidence")
    if set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"}:
        raise ValueError("missing checkpoint identities")
    if [g.get("name") for g in result.get("groups", [])] != [g["name"] for g in plan["groups"]]:
        raise ValueError("microbatch coverage differs from plan")
    expected = {(r, c) for r in range(policy.raw["rounds"]) for c in range(policy.raw["measured_calls"])}
    for group, spec in zip(result["groups"], plan["groups"]):
        if (any(group.get(k) != v for k, v in spec.items()) or group.get("status") != "complete"
                or group.get("inputs_unchanged") is not True):
            raise ValueError("microbatch identity or completeness changed")
        if [c.get("variant") for c in group.get("cases", [])] != plan["variants"]:
            raise ValueError("backend coverage changed")
        for case in group["cases"]:
            calls = case["calls"]
            if (case.get("status") != "complete" or len(calls) != len(expected)
                    or {(c["round"], c["call"]) for c in calls} != expected
                    or len(case["warmup"]) != policy.raw["warmup_calls"]
                    or case.get("comparison", {}).get("passed") is not True
                    or case.get("gradient_health", {}).get("finite") is not True):
                raise ValueError("missing timing, screening or gradient evidence")
            validate_observation(case["modulation_observation"], case["variant"])
            if any(not math.isfinite(row[k]) or row[k] <= 0 for row in calls for k in packing.TIMINGS):
                raise ValueError("invalid timing sample")
            if any(not math.isfinite(row["loss"]) or row["peak_allocated_bytes"] <= 0 for row in calls):
                raise ValueError("invalid loss/memory sample")


def write_report(path, result, policy):
    rows = packing.summarize(result, policy.packing_policy)
    packing.write_report(path, result, rows)
    with path.open("a") as stream:
        stream.write("\nThis performance run forces cuDNN in the padded baseline and FP32 modulation sums in all packed variants.\n"
            "Layout creation and processor installation/restoration are included in wall time. Gradient reset, optimizer work "
            "and input preparation are excluded. Optimizer storage and the preparation reserve remain resident.\n"
            "Output/loss screening and finite full-gradient coverage run outside timing; full gradient differences are not recomputed.\n"
            "Backend dispatch is verified with CPU operator events. CUDA events time GPU work; CUDA kernel tracing is not requested.\n")
    return rows


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    checks = {**source_metadata(), "config_sha256": config_hashes(args.config, policy), "dataset_sha256": dataset_hashes(policy)}
    if metadata is not None and any(metadata.get(key) != value for key, value in checks.items()):
        raise ValueError("uploaded source/config/data differs from launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**(metadata or {}), **checks, "config": policy.raw, "plan": build_plan(policy), "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start", groups=[])
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
        rows = write_report(args.output_dir / "report.md", result, policy)
        worker_tools._write_json(args.output_dir / "summary.json", {**result, "rows": rows})
        worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_packing_performance.json")
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
