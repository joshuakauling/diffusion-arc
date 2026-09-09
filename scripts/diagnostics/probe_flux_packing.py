#!/usr/bin/env python3
"""Compare production FLUX execution with standalone packed-attention diagnostics.

Standalone diagnostic: no source edits, scheduler changes, weight updates,
alignment sweep, homogeneity sweep, or model/dataset downloads.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import argparse
import gc
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.model.contracts import FlowMatchingBatch
from diffusion_arc.model.flow_matching import flow_matching_loss, sample_flow_matching_batch
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids, predict_flux_velocity
from diffusion_arc.training.dataset import ArcTrainingDataset
from diffusion_arc.training.flux_input import _select_eligible_sequence, prepare_flux_inputs_batch
from diffusion_arc.training.flux_schedule import schedule_flux_targets
from scripts.diagnostics import probe_flux_batch_shapes as shapes
from scripts.diagnostics._flux_alignment import statistics_for
from scripts.diagnostics._flux_packing import FlashBackend, packed_processors, packed_velocity, packing_layout
from scripts.diagnostics.probe_nvarc_packing import dataset_hashes

PROGRESS_MARKER = "FLUX_PACKING_PROGRESS"
TIMINGS = ("assembly_ms", "forward_ms", "backward_ms", "forward_backward_ms", "total_wall_ms")


@dataclass(frozen=True)
class ProbeConfig:
    training_config: Path
    warmup_calls: int
    measured_calls: int
    rounds: int
    max_duration_seconds: int
    backends: tuple[str, ...]
    numerical_tolerances: dict
    length_source: dict
    windows: tuple[dict, ...]
    groups: tuple[dict, ...]


def load_probe_config(path, *, require_checkpoints=True):
    raw = json.loads(path.read_text())
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", *ProbeConfig.__dataclass_fields__}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("unexpected FLUX packing configuration fields or schema")
    for key in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    backends = raw["backends"]
    if (not isinstance(backends, list) or not backends or len(set(backends)) != len(backends)
            or any(b not in FlashBackend.MODULES for b in backends)):
        raise ValueError("backends must be distinct entries from fa2, fa3, fa4")
    tolerances = raw["numerical_tolerances"]
    if (not isinstance(tolerances, dict)
            or set(tolerances) != {"velocity_relative_l2", "gradient_relative_l2", "loss_relative"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1
                   for v in tolerances.values())):
        raise ValueError("numerical_tolerances must define three finite relative limits in (0, 1)")
    if not isinstance(raw["training_config"], str) or not raw["training_config"]:
        raise ValueError("training_config must name a file")
    if not isinstance(raw["length_source"], dict) or not raw["length_source"]:
        raise ValueError("length_source must record provenance")
    windows = raw["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be nonempty")
    for window in windows:
        if (not isinstance(window, dict) or set(window) != {
                "name", "optimizer_step", "source_keys", "source_lengths", "target_prefix_lengths"}
                or not isinstance(window["name"], str) or not window["name"].isidentifier()
                or type(window["optimizer_step"]) is not int or window["optimizer_step"] < 1):
            raise ValueError("invalid input window")
        for key, size in (("source_lengths", 16), ("target_prefix_lengths", 48)):
            if (not isinstance(window[key], list) or len(window[key]) != size
                    or any(type(n) is not int or not 1 <= n <= 8400 for n in window[key])):
                raise ValueError(f"{key} must contain {size} lengths in [1, 8400]")
        if (not isinstance(window["source_keys"], list) or len(window["source_keys"]) != 16
                or any(not isinstance(k, str) or not k for k in window["source_keys"])
                or window["source_lengths"] != window["target_prefix_lengths"][2::3]):
            raise ValueError("source keys or consumed prefix lengths are inconsistent")
    if len({w["name"] for w in windows}) != len(windows):
        raise ValueError("window names must be unique")
    if not isinstance(raw["groups"], list) or not raw["groups"]:
        raise ValueError("groups must be nonempty")
    policy = ProbeConfig(
        training_config=(path.resolve().parent / raw["training_config"]).resolve(),
        backends=tuple(backends), windows=tuple(windows), groups=tuple(raw["groups"]),
        **{k: raw[k] for k in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds",
                              "numerical_tolerances", "length_source")},
    )
    training = shapes.load_single_gpu_training_config(policy.training_config)
    shapes.validate_training_layout(training, require_checkpoints=require_checkpoints)
    if (training.flux_scheduling is None or training.nvarc_scheduling is None
            or training.loop.optimizer_batch_size != 48 or training.loop.nvarc_source_batch_size != 16
            or training.data.shuffle or training.loop.min_context_pairs_before_target != 2):
        raise ValueError("diagnostic requires the fixed B2 16-source/48-target input policy")
    build_plan(policy, training)
    return policy, training


def build_plan(policy, training):
    windows = {w["name"]: w for w in policy.windows}
    plan = []
    for index, group in enumerate(policy.groups):
        if (not isinstance(group, dict) or set(group) != {
                "name", "window", "batch_index", "target_indices", "padded_length"}
                or not isinstance(group["name"], str) or not group["name"].isidentifier()
                or group["window"] not in windows or type(group["batch_index"]) is not int
                or group["batch_index"] < 0 or type(group["padded_length"]) is not int
                or not isinstance(group["target_indices"], list)
                or any(type(i) is not int for i in group["target_indices"])):
            raise ValueError("invalid FLUX group specification")
        window = windows[group["window"]]
        lengths = window["target_prefix_lengths"]
        schedule = schedule_flux_targets(lengths, training.flux_scheduling)
        if group["batch_index"] >= len(schedule):
            raise ValueError("batch_index is outside the recorded B2 schedule")
        batch = schedule[group["batch_index"]]
        if list(batch.target_indices) != group["target_indices"] or batch.padded_length != group["padded_length"]:
            raise ValueError("configured group differs from the current B2 schedule")
        selected = [lengths[i] for i in batch.target_indices]
        variants = ["sdpa_b2", *policy.backends]
        random.Random(training.seed + index).shuffle(variants)
        orders = [variants[r % len(variants):] + variants[:r % len(variants)] for r in range(policy.rounds)]
        plan.append({**group, "text_lengths": selected, "batch_size": len(selected),
                     "real_joint_tokens": sum(selected) + len(selected) * 900,
                     "baseline_joint_tokens": len(selected) * (batch.padded_length + 900),
                     "round_orders": orders})
    if len({g["name"] for g in plan}) != len(plan):
        raise ValueError("group names must be unique")
    return plan


def prepare_inputs(policy, training, models):
    """Replay real ARC contents with production preparation; retain CPU copies."""
    dataset = ArcTrainingDataset(training.data.split_directory, training.data.view_name, models.tokenizer)
    selected = {}
    for window in policy.windows:
        groups = [g for g in policy.groups if g["window"] == window["name"]]
        if not groups:
            continue
        start = (window["optimizer_step"] - 1) * 16
        if [str(k) for k in dataset.keys[start:start + 16]] != window["source_keys"]:
            raise ValueError("original benchmark augmentation keys differ")
        samples = tuple(dataset[i] for i in range(start, start + 16))
        eligible = [_select_eligible_sequence(s, min_context_pairs_before_target=2) for s in samples]
        if (any(s is None for s in eligible)
                or [len(s.input_ids) for s in eligible] != window["source_lengths"]
                or [n for s in eligible for n in s.prefix_ends] != window["target_prefix_lengths"]):
            raise ValueError("original benchmark token/prefix lengths differ")
        prepared = prepare_flux_inputs_batch(
            samples, nvarc_runner=models.nvarc_runner, vae_encoder=models.vae_encoder,
            min_context_pairs_before_target=2, vae_microbatch_size=3,
            nvarc_scheduling=training.nvarc_scheduling,
        )
        for group in groups:
            targets = [prepared[i] for i in group["target_indices"]]
            selected[group["name"]] = (
                tuple(t.nvarc_sequence_state._prefix_for(t.prefix_end).detach().cpu().clone() for t in targets),
                torch.stack([t.clean_latent for t in targets]).detach().cpu(),
                hashlib.sha256(json.dumps([list(s.input_ids) for s in eligible]).encode()).hexdigest(),
            )
        # Avoid retaining the last target/source owners across GPU measurements.
        del targets, prepared
    return selected


def group_inputs(stored, group, seed, device):
    prefixes, clean, digest = stored
    prefixes = tuple(p.to(device) for p in prefixes)
    clean = clean.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    draws = [sample_flow_matching_batch(latent.unsqueeze(0), generator=generator) for latent in clean]
    flow = FlowMatchingBatch(**{
        k: torch.cat([getattr(f, k) for f in draws])
        for k in ("noisy_latents", "timesteps", "target_velocity")
    })
    return dict(prefixes=prefixes, clean=clean, flow=flow, input_sha256=digest,
                image_positions=build_flux_image_position_ids(len(prefixes), device),
                layout=packing_layout(tuple(group["text_lengths"]), 900, device))


def forward_velocity(model, variant, assembled, data, compute_dtype, *, modulation_reduction="native"):
    flow = data["flow"]
    if variant == "sdpa_b2":
        return predict_flux_velocity(model, flow.noisy_latents, assembled, data["image_positions"],
                                     flow.timesteps, compute_dtype=compute_dtype)
    with torch.autocast(device_type=flow.noisy_latents.device.type, dtype=compute_dtype):
        return packed_velocity(model, assembled, flow.noisy_latents, data["image_positions"],
                               flow.timesteps, data["layout"], modulation_reduction=modulation_reduction)


def measure_call(model, optimizer, variant, group, data, *, compute_dtype, weight, profile=False):
    """Input assembly + production flow loss + weighted backward; no updates."""
    device = data["clean"].device
    optimizer.zero_grad(set_to_none=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    scope = torch.profiler.record_function if profile else lambda _: nullcontext()
    started = time.perf_counter()
    with scope("flux_packing_assembly"):
        assembled = (pad_flux_text_prefixes(data["prefixes"], padded_length=group["padded_length"])
                     if variant == "sdpa_b2" else torch.cat(data["prefixes"]))
    torch.cuda.synchronize(device)
    assembled_at = time.perf_counter()
    events[0].record()
    with scope("flux_packing_forward"):
        output = forward_velocity(model, variant, assembled, data, compute_dtype)
        loss = flow_matching_loss(output, data["flow"].target_velocity)
    events[1].record()
    with scope("flux_packing_backward"):
        (loss * weight).backward()
    events[2].record()
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    loss_value = loss.detach().item()
    if not math.isfinite(loss_value):
        raise FloatingPointError("non-finite FLUX loss")
    return {
        "assembly_ms": (assembled_at - started) * 1000,
        "forward_ms": events[0].elapsed_time(events[1]),
        "backward_ms": events[1].elapsed_time(events[2]),
        "forward_backward_ms": events[0].elapsed_time(events[2]),
        "total_wall_ms": (finished - started) * 1000,
        "loss": loss_value, **shapes._memory(device),
    }, output.detach()


def tensor_difference(pairs):
    """Stream full FP32 outputs/gradients through bounded GPU scratch space."""
    stats, maximum, count = None, None, 0
    for actual, reference in pairs:
        if actual.shape != reference.shape:
            raise ValueError("numerical comparison shape mismatch")
        if stats is None:
            stats = torch.zeros(2, dtype=torch.float64, device=actual.device)
            maximum = torch.zeros((), device=actual.device)
        for left, right in zip(actual.detach().reshape(-1).split(1 << 20), reference.reshape(-1).split(1 << 20)):
            a, b = left.float(), right.to(device=actual.device, dtype=torch.float32)
            delta = a - b
            maximum = torch.maximum(maximum, delta.abs().max())
            stats += torch.stack((delta.double().square().sum(), b.double().square().sum()))
            count += a.numel()
    if not count:
        raise ValueError("numerical comparison contains no values")
    error, scale = stats.cpu().tolist()
    maximum = maximum.item()
    if not all(math.isfinite(v) for v in (error, scale, maximum)):
        raise FloatingPointError("non-finite output or gradient")
    return dict(max_abs=maximum, rms=(error / count) ** .5,
                relative_l2=(error / scale) ** .5 if scale else (0.0 if error == 0 else math.inf), count=count)


def compare(model, output, row, reference, tolerances):
    ref_output, ref_gradients, ref_loss = reference
    actual = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    if actual.keys() != ref_gradients.keys():
        raise ValueError("trainable gradient coverage differs from reference")
    velocity = tensor_difference([(output, ref_output)])
    gradient = tensor_difference((actual[n], ref_gradients[n]) for n in actual)
    loss_relative = abs(row["loss"] - ref_loss) / max(abs(ref_loss), 1e-12)
    passed = (velocity["relative_l2"] <= tolerances["velocity_relative_l2"]
              and gradient["relative_l2"] <= tolerances["gradient_relative_l2"]
              and loss_relative <= tolerances["loss_relative"])
    return dict(velocity=velocity, gradient=gradient, loss_relative=loss_relative,
                passed=passed, thresholds=tolerances, training_equivalence_established=False)


def validate_backend_events(counts, variant, layers):
    if variant == "sdpa_b2":
        if counts.get("aten::scaled_dot_product_attention", 0) != layers:
            raise RuntimeError("baseline did not execute the expected SDPA calls")
        return
    if counts.get(f"flux_packing::{variant}", 0) != layers:
        raise RuntimeError(f"{variant}: incorrect explicit backend call count")
    extension_ops = {
        "fa2": ("flash_attn::_flash_attn_varlen_forward", "flash_attn::_flash_attn_varlen_backward"),
        "fa3": ("flash_attn_3::_flash_attn_forward", "flash_attn_3::_flash_attn_backward"),
    }
    for other, operators in extension_ops.items():
        for operator in operators:
            expected = layers if variant == other else 0
            if counts.get(operator, 0) != expected:
                raise RuntimeError(f"{variant}: unexpected {operator} count")
    if any("scaled_dot_product" in n for n in counts):
        raise RuntimeError(f"{variant}: unintended SDPA fallback")
    if not any("flash" in n.lower() and ("backward" in n.lower() or "bwd" in n.lower()) for n in counts):
        raise RuntimeError(f"{variant}: missing FlashAttention backward evidence")


def backend_preflight(backend, device, tolerances):
    """Small noncausal Q/K/V forward/backward and segment-isolation preflight."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    layout = packing_layout((3, 5), 2, device)
    generator = torch.Generator(device=device).manual_seed(181)
    inputs = [torch.randn(12, 2, 128, device=device, dtype=torch.bfloat16,
                          generator=generator, requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH):
        reference = torch.cat([
            F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            .transpose(0, 1) for q, k, v in zip(*(t.split((5, 7)) for t in inputs))
        ])
    gradients = torch.autograd.grad(reference.float().square().mean(), inputs)
    started = time.perf_counter()
    actual = backend(*inputs, layout)
    actual_gradients = torch.autograd.grad(actual.float().square().mean(), inputs)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output_error = tensor_difference([(actual, reference.detach())])
    gradient_error = tensor_difference(zip(actual_gradients, gradients))
    with torch.no_grad():
        changed = [t.detach().clone() for t in inputs]
        for t in changed:
            t[:5].add_(2)
        isolated = backend(*changed, layout)
    # Different grad modes may select different execution details; compare both
    # isolation forwards under the same mode and physical shape.
    with torch.no_grad():
        unchanged = backend(*(t.detach() for t in inputs), layout)
    if not torch.equal(unchanged[5:], isolated[5:]):
        raise RuntimeError(f"{backend.name}: cross-example attention leakage")
    passed = (output_error["relative_l2"] <= tolerances["velocity_relative_l2"]
              and gradient_error["relative_l2"] <= tolerances["gradient_relative_l2"])
    if not passed:
        raise RuntimeError(f"{backend.name}: Q/K/V preflight tolerance exceeded")
    return dict(passed=True, output=output_error, gradients=gradient_error,
                source_isolation="bitwise_equal_unmodified_segment", first_forward_backward_seconds=elapsed)


def profile_case(call, model, variant, backend, path):
    if backend is not None:
        backend.observe = True
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) as profiler:
            row, output = call(profile=True)
        del output
    finally:
        if backend is not None:
            backend.observe = False
    counts = {e.key: e.count for e in profiler.key_averages()
              if any(s in e.key.lower() for s in ("attention", "flash", "flux_packing::"))}
    kernels = sorted({e.name for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    if not kernels:
        raise RuntimeError("profile contains no CUDA kernels")
    validate_backend_events(counts, variant, len(model.transformer_blocks) + len(model.single_transformer_blocks))
    temporary = path.with_suffix("")
    profiler.export_chrome_trace(str(temporary))
    with temporary.open("rb") as source, gzip.open(path, "wb") as destination:
        shutil.copyfileobj(source, destination)
    temporary.unlink()
    return dict(trace=str(path.name), operators=counts, cuda_kernel_names=kernels)


def run_worker(args):
    if getattr(args, "correctness", False) and getattr(args, "forward_only", False):
        raise ValueError("--correctness and --forward-only are mutually exclusive")
    if getattr(args, "forward_only", False):
        from scripts.diagnostics._flux_packing_forward import run_worker as forward_worker
        return forward_worker(args)
    if getattr(args, "correctness", False):
        from scripts.diagnostics._flux_packing_correctness import run_worker as correctness_worker
        return correctness_worker(args)
    result = dict(status="running", phase="configuration", groups=[])
    def persist(*, notify=True):
        shapes._write_json(args.output_dir / "result.json", result)
        if notify:
            print(PROGRESS_MARKER, result["phase"], flush=True)
    try:
        policy, training = load_probe_config(args.config)
        plan = build_plan(policy, training)
        result["phase"] = "load_backends_and_models"
        persist()
        backends = {name: FlashBackend(name) for name in policy.backends}
        device = shapes.prepare_single_gpu_device(training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("diagnostic image requires Hopper/SM90")
        result["backend_preflights"] = {}
        for name, backend in backends.items():
            result["phase"] = f"{name}/small_forward_backward_preflight"
            persist()
            result["backend_preflights"][name] = backend_preflight(backend, device, policy.numerical_tolerances)
            persist()
        result["phase"] = "load_models_and_training_memory"
        persist()
        device, models, optimizer, reserve, setup = shapes.prepare_runtime(training)
        model = models.transformer
        if torch.cuda.get_device_capability(device) != (9, 0) or model.config.attention_head_dim != 128:
            raise ValueError("diagnostic image requires Hopper and FLUX head dimension 128")
        result.update(setup=setup, environment={**shapes._environment_metadata(),
            "packages": {n: version(n) for n in ("flash-attn", "flash-attn-3", "flash-attn-4")}},
            backend_bindings={n: b.binding for n, b in backends.items()}, phase="prepare_real_inputs")
        persist()
        stored = prepare_inputs(policy, training, models)
        (args.output_dir / "traces").mkdir()
        for index, specification in enumerate(plan):
            group = {**specification, "status": "running", "cases": []}
            result["groups"].append(group)
            data = group_inputs(stored[group["name"]], group, training.seed + index, device)
            group.update(flow_seed=training.seed + index, input_sha256=data["input_sha256"])
            def call(variant, **kwargs):
                return measure_call(model, optimizer, variant, group, data,
                    compute_dtype=models.flux_compute_dtype,
                    weight=group["batch_size"] / training.loop.optimizer_batch_size, **kwargs)
            result["phase"] = f"{group['name']}/reference_forward_backward"
            persist()
            row, output = call("sdpa_b2")
            reference = (output.cpu(), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                                       if p.grad is not None}, row["loss"])
            group["reference_call"] = row
            del output
            cases = {variant: dict(variant=variant, status="running", calls=[]) for variant in ("sdpa_b2", *policy.backends)}
            group["cases"] = list(cases.values())
            for round_index, order in enumerate(group["round_orders"]):
                for variant in order:
                    backend = backends.get(variant)
                    context = packed_processors(model, data["layout"], backend) if backend else nullcontext()
                    with context:
                        case = cases[variant]
                        if round_index == 0:
                            result["phase"] = f"{group['name']}/{variant}/first_forward_backward"
                            persist()
                            first, output = call(variant)
                            case["first_call"] = first
                            result["phase"] = f"{group['name']}/{variant}/numerical_comparison"
                            persist()
                            case["comparison"] = compare(model, output, first, reference, policy.numerical_tolerances)
                            del output
                            if not case["comparison"]["passed"]:
                                persist()
                                raise RuntimeError("FLUX numerical diagnostic tolerance exceeded")
                        for i in range(policy.warmup_calls + policy.measured_calls):
                            phase = "warmup" if i < policy.warmup_calls else "measured"
                            result["phase"] = f"{group['name']}/{variant}/round_{round_index}/{phase}_{i}"
                            persist(notify=False)
                            row, output = call(variant)
                            del output
                            case["calls"].append(dict(round=round_index, call=i, phase=phase, **row))
                            persist(notify=False)
                        persist()
            del reference
            gc.collect()
            for variant, case in cases.items():
                result["phase"] = f"{group['name']}/{variant}/profile"
                persist()
                backend = backends.get(variant)
                context = packed_processors(model, data["layout"], backend) if backend else nullcontext()
                with context:
                    case["profile"] = profile_case(lambda **kw: call(variant, **kw), model, variant, backend,
                        args.output_dir / "traces" / f"{group['name']}_{variant}.json.gz")
                case["status"] = "complete"
                persist()
            group["status"] = "complete"
            del data
            gc.collect()
            persist()
        result.update(status="complete", phase="complete")
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        persist()
    return result


def summarize(result, policy):
    rows = []
    for group in result.get("groups", []):
        for case in group["cases"]:
            calls = [c for c in case["calls"] if c["phase"] == "measured"]
            complete = case["status"] == "complete" and len(calls) == policy.rounds * policy.measured_calls
            rows.append(dict(group=group["name"], variant=case["variant"], complete=complete,
                text_lengths=group["text_lengths"], padded_length=group["padded_length"],
                measured_calls=len(calls), timings={k: statistics_for([c[k] for c in calls]) for k in TIMINGS} if calls else {},
                round_medians={str(r): statistics.median(c["total_wall_ms"] for c in calls if c["round"] == r)
                               for r in sorted({c["round"] for c in calls})},
                peak_allocated_bytes=max((c["peak_allocated_bytes"] for c in calls), default=None),
                peak_reserved_bytes=max((c["peak_reserved_bytes"] for c in calls), default=None),
                comparison=case.get("comparison")))
    for row in rows:
        baseline = next((r for r in rows if r["group"] == row["group"] and r["variant"] == "sdpa_b2"), None)
        if row["complete"] and baseline and baseline["complete"]:
            row["speedup_vs_baseline"] = baseline["timings"]["total_wall_ms"]["median"] / row["timings"]["total_wall_ms"]["median"]
    return rows


def validate_complete(result, plan, policy, output_dir):
    groups = result.get("groups", [])
    if result.get("status") != "complete" or [g["name"] for g in groups] != [g["name"] for g in plan]:
        raise ValueError("worker did not complete the requested groups")
    for group, expected in zip(groups, plan):
        if any(group.get(k) != v for k, v in expected.items()):
            raise ValueError("worker group differs from the planned inputs")
        if [c["variant"] for c in group["cases"]] != ["sdpa_b2", *policy.backends]:
            raise ValueError("missing or duplicate backend cases")
        for case in group["cases"]:
            identities = [(c["round"], c["call"], c["phase"]) for c in case["calls"]]
            expected_calls = [(r, i, "warmup" if i < policy.warmup_calls else "measured")
                              for r in range(policy.rounds) for i in range(policy.warmup_calls + policy.measured_calls)]
            if (case["status"] != "complete" or identities != expected_calls
                    or not case.get("comparison", {}).get("passed")
                    or not (output_dir / "traces" / case.get("profile", {}).get("trace", "MISSING")).is_file()):
                raise ValueError("incomplete samples, comparison or backend profile")
            if any(not math.isfinite(c[k]) or c[k] < 0 for c in case["calls"] for k in TIMINGS):
                raise ValueError("invalid timing observation")


def source_hashes():
    files = [*sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
             *sorted((REPOSITORY_ROOT / "scripts/diagnostics").glob("*.py")),
             REPOSITORY_ROOT / "scripts/cloud/probe_flux_packing_modal.py", REPOSITORY_ROOT / "pyproject.toml"]
    return {str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def config_hashes(path, policy):
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in
            (("probe", path), ("training", policy.training_config))}


def source_metadata():
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True).strip(),
                source_sha256=source_hashes())


def write_report(path, result, rows):
    lines = ["# FLUX attention packing diagnostic", "", f"Status: **{result['status']}**", "",
             "Times are medians in ms; memory is peak allocated GiB. First calls, numerical checks and profiles are excluded.", "",
             "| Group | Text lengths | Backend | Forward | Backward | Assembly + F/B | GiB | Speedup | Complete |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        timings = row["timings"]
        values = [f"{timings[k]['median']:.3f}" if timings else "—" for k in ("forward_ms", "backward_ms", "total_wall_ms")]
        memory = f"{row['peak_allocated_bytes'] / 2**30:.3f}" if row["peak_allocated_bytes"] else "—"
        speedup = f"{row['speedup_vs_baseline']:.3f}x" if "speedup_vs_baseline" in row else "—"
        lines.append(f"| {row['group']} | {row['text_lengths']} | {row['variant']} | " +
                     " | ".join(values) + f" | {memory} | {speedup} | {row['complete']} |")
    lines += ["", "FA2/FA3/FA4 share the same fully packed execution. The SDPA comparison also changes padding and model layout.",
              "These fixed B2 groups do not establish an optimal token budget, full-step speedup, or training equivalence.",
              "Per-example modulations are expanded explicitly; their memory and runtime are included."]
    if result.get("error"):
        lines += ["", f"Failure during `{result.get('phase')}`: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")


def forward_metadata(policy, checks):
    """Declare the bounded forward experiment before execution."""
    return dict(mode="forward_only", experiment_version=2, variants=list(checks.VARIANTS), no_grad=True,
                full_forward_calls=len(checks.VARIANTS), backward_calls=0,
                optimizer_initializations=0, optimizer_updates=0, measured_calls=0,
                final_velocity_relative_l2=policy.numerical_tolerances["velocity_relative_l2"],
                independent_control="original singleton SDPA; the attention backend also changes")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy, training = load_probe_config(args.config)
    correctness = getattr(args, "correctness", False)
    forward_only = getattr(args, "forward_only", False)
    if correctness and forward_only:
        raise ValueError("--correctness and --forward-only are mutually exclusive")
    focused = correctness or forward_only
    if forward_only:
        from scripts.diagnostics import _flux_packing_forward as checks
        policy = checks.forward_policy(policy)
    elif correctness:
        from scripts.diagnostics import _flux_packing_correctness as checks
        policy = checks.correctness_policy(policy)
    metadata = source_metadata() if metadata is None else metadata
    hashes, data_hashes = config_hashes(args.config, policy), dataset_hashes(training)
    if metadata.get("source_sha256") != source_hashes():
        raise ValueError("uploaded source hashes differ from launch metadata")
    for key, actual in (("config_sha256", hashes), ("dataset_sha256", data_hashes)):
        if key in metadata and metadata[key] != actual:
            raise ValueError(f"uploaded {key} differs from launch metadata")
    plan = checks.build_plan(policy, training) if focused else build_plan(policy, training)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(metadata, config_sha256=hashes, dataset_sha256=data_hashes,
        config=json.loads(args.config.read_text()), training_config=json.loads(policy.training_config.read_text()),
        plan=plan, status="running",
        optimizer_updates_during_comparisons=0, checkpoint_weights_hashed=False)
    if not forward_only:
        manifest["optimizer_learning_rate_override"] = 0.0
    if forward_only:
        manifest.update(forward_metadata(policy, checks), effective_worker_seconds=policy.max_duration_seconds)
    elif correctness:
        manifest.update(mode="correctness", variants=list(checks.VARIANTS),
                        effective_worker_seconds=policy.max_duration_seconds,
                        isolated_blocks=["transformer_blocks.0", "single_transformer_blocks.0"])
    shapes._write_json(args.output_dir / "manifest.json", manifest)
    result = dict(status="error", phase="worker_start", groups=[])
    started = time.monotonic()
    try:
        returncode, timed_out = shapes.launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config),
             "--output-dir", str(args.output_dir),
             *(["--forward-only"] if forward_only else ["--correctness"] if correctness else [])],
            log_path=args.output_dir / "worker.log",
            timeout_seconds=policy.max_duration_seconds, on_progress=on_progress,
            progress_marker=PROGRESS_MARKER,
        )
        if (args.output_dir / "result.json").exists():
            result = json.loads((args.output_dir / "result.json").read_text())
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error="complete worker exceeded its configured time budget")
        elif returncode != 0:
            if result.get("status") not in ("oom", "error"):
                result["status"] = "error"
            result.setdefault("error", "worker failed; inspect worker.log")
        elif result.get("status") == "complete":
            if focused:
                checks.validate_complete(result)
            else:
                validate_complete(result, plan, policy, args.output_dir)
        elif result.get("status") == "running":
            result.update(status="error", error="worker exited with incomplete results")
    except Exception as error:
        partial = args.output_dir / "result.json"
        if partial.is_file():
            try:
                result = json.loads(partial.read_text())
            except (OSError, ValueError):
                pass  # Keep the original file; report the supervisor error below.
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = result["status"]
        rows = [] if focused else summarize(result, policy)
        shapes._write_json(args.output_dir / "summary.json", {**result, "rows": rows})
        shapes._write_json(args.output_dir / "manifest.json", manifest)
        if focused:
            checks.write_report(args.output_dir / "report.md", result)
        else:
            write_report(args.output_dir / "report.md", result, rows)
        on_progress()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--correctness", action="store_true", help="Localize differences on short_b4; no timing sweep")
    modes.add_argument("--forward-only", action="store_true", help="Compare two singletons, both packing orders and isolation; twelve forwards only")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    policy, training = load_probe_config(args.config, require_checkpoints=not args.dry_run)
    if args.forward_only:
        from scripts.diagnostics import _flux_packing_forward as checks
        policy = checks.forward_policy(policy)
    elif args.correctness:
        from scripts.diagnostics import _flux_packing_correctness as checks
        policy = checks.correctness_policy(policy)
    if args.dry_run:
        plan = checks.build_plan(policy, training) if args.correctness or args.forward_only else build_plan(policy, training)
        if args.forward_only:
            print(json.dumps(dict(forward_metadata(policy, checks), groups=plan,
                                  max_duration_seconds=policy.max_duration_seconds), indent=2))
            return 0
        if args.correctness:
            print(json.dumps(dict(mode="correctness", groups=plan, variants=list(checks.VARIANTS),
                max_duration_seconds=policy.max_duration_seconds,
                isolated_blocks=["transformer_blocks.0", "single_transformer_blocks.0"],
                measured_calls=0, optimizer_updates=0), indent=2))
            return 0
        print(json.dumps(dict(groups=plan, backends=["sdpa_b2", *policy.backends],
            cases=len(plan) * (1 + len(policy.backends)),
            measured_calls=len(plan) * (1 + len(policy.backends)) * policy.rounds * policy.measured_calls,
            max_duration_seconds=policy.max_duration_seconds), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required outside dry-run")
    result = run_worker(args) if args.worker else run_probe(args)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
