"""Controlled alignment mode for the existing FLUX shape probe and Modal launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

from diffusion_arc.model.contracts import DEFAULT_FLUX_SPATIAL_SHAPE, FlowMatchingBatch, NVARC_MAX_TOKENS
from diffusion_arc.model.flow_matching import sample_flow_matching_batch
from scripts.diagnostics import probe_flux_batch_shapes as probe


IMAGE_TOKENS = DEFAULT_FLUX_SPATIAL_SHAPE.token_count
TIMINGS = ("assembly_ms", "forward_ms", "backward_ms", "forward_backward_ms", "wall_ms", "total_wall_ms")


@dataclass(frozen=True)
class AlignmentConfig:
    training_config: Path
    alignment_multiples: tuple[int, ...]
    warmup_calls: int
    measured_calls: int
    rounds: int
    max_duration_seconds: int
    profile_groups: tuple[tuple[int, int], ...]


def load_config(path, raw, *, require_checkpoints):
    fields = {"schema_version", *AlignmentConfig.__dataclass_fields__}
    if set(raw) != fields:
        raise ValueError(f"alignment config requires exactly these fields: {sorted(fields)}")
    for name in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = raw["alignment_multiples"]
    if (
        not isinstance(values, list) or not values
        or any(type(value) is not int or value not in (8, 32, 64, 128) for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError("alignment_multiples must be a strictly increasing subset of [8, 32, 64, 128]")
    profiles = raw["profile_groups"]
    if (
        not isinstance(profiles, list)
        or any(not isinstance(pair, list) or len(pair) != 2
               or any(type(value) is not int for value in pair) for pair in profiles)
        or len({tuple(pair) for pair in profiles}) != len(profiles)
    ):
        raise ValueError("profile_groups must be unique [batch_size, baseline_text_length] pairs")
    if not isinstance(raw["training_config"], str) or not raw["training_config"]:
        raise ValueError("training_config must name a configuration file")
    policy = AlignmentConfig(
        training_config=(path.resolve().parent / raw["training_config"]).resolve(),
        alignment_multiples=tuple(values), profile_groups=tuple(map(tuple, profiles)),
        **{name: raw[name] for name in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds")},
    )
    training = probe.load_single_gpu_training_config(policy.training_config)
    probe.validate_training_layout(training, require_checkpoints=require_checkpoints)
    if training.flux_scheduling is None:
        raise ValueError("alignment probe requires an explicit FLUX scheduling catalog")
    keys = {(shape.batch_size, shape.max_tokens) for shape in training.flux_scheduling.shapes}
    if any(length <= size for size, length in keys):
        raise ValueError("baseline lengths must allow a positive, masked prefix in every row")
    if not set(policy.profile_groups) <= keys:
        raise ValueError("profile_groups must select existing FLUX catalog shapes")
    return policy, training


def build_plan(policy, training):
    """Deduplicate physical shapes within each fixed-input comparison, preserving all labels."""

    groups = []
    for shape in training.flux_scheduling.shapes:
        baseline, size = shape.max_tokens, shape.batch_size
        labels = {baseline: ["original"]}
        excluded = []
        for multiple in policy.alignment_multiples:
            for kind, offset in (("text", 0), ("joint", IMAGE_TOKENS)):
                length = ((baseline + offset + multiple - 1) // multiple) * multiple - offset
                label = f"{kind}_{multiple}"
                if length > NVARC_MAX_TOKENS:
                    excluded.append({"label": label, "text_length": length, "reason": "text_limit"})
                else:
                    labels.setdefault(length, []).append(label)
        order = sorted(labels)
        random.Random(training.seed + baseline * 16 + size).shuffle(order)
        # Spread each candidate over early and late positions across rounds.
        orders = []
        for round_index in range(policy.rounds):
            shift = round_index * len(order) // policy.rounds
            orders.append(order[shift:] + order[:shift])
        groups.append({
            "batch_size": size, "baseline_text_length": baseline,
            "baseline_catalog_forward_backward_ms": shape.forward_backward_ms,
            "real_prefix_lengths": [baseline - 1 - row for row in range(size)],
            "flow_seed": training.seed + baseline * 16 + size,
            "padded_lengths": sorted(labels), "labels": {str(k): v for k, v in labels.items()},
            "round_orders": orders, "excluded": excluded,
            "profile": (size, baseline) in policy.profile_groups,
        })
    random.Random(training.seed).shuffle(groups)
    return groups


def make_flow_batch(reserve, group, *, device):
    """Draw once per target as in dynamic B2, then reuse these tensors for the entire group."""

    generator = torch.Generator(device=device).manual_seed(group["flow_seed"])
    targets = [
        sample_flow_matching_batch(reserve[-1][0].unsqueeze(0), generator=generator)
        for _ in range(group["batch_size"])
    ]
    return FlowMatchingBatch(**{
        name: torch.cat([getattr(target, name) for target in targets])
        for name in ("noisy_latents", "timesteps", "target_velocity")
    })


def measure_call(training, models, optimizer, reserve, group, length, flow_batch, *, device, state):
    """Time the unmodified production assembly, loss, and backward; keep gradients resident."""

    optimizer.zero_grad(set_to_none=False)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    state["phase"] = "assembly"
    started = time.perf_counter()
    microbatch = probe.build_probe_microbatch(
        reserve, text_length=length, batch_size=group["batch_size"],
        prefix_lengths=group["real_prefix_lengths"],
    )
    torch.cuda.synchronize(device)
    assembled = time.perf_counter()
    state["phase"] = "forward"
    events[0].record()
    loss = probe.compute_flux_microbatch_loss(
        models.transformer, microbatch, compute_dtype=models.flux_compute_dtype, flow_batch=flow_batch,
    )
    events[1].record()
    state["phase"] = "backward"
    (loss * (group["batch_size"] / training.loop.optimizer_batch_size)).backward()
    events[2].record()
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    memory = probe._memory(device)
    # Read only the scalar loss after all timers; no hooks, mask interception, or tensor scans.
    state["phase"] = "loss_readback"
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("non-finite FLUX loss")
    return {
        "assembly_ms": (assembled - started) * 1000,
        "wall_ms": (finished - assembled) * 1000,
        "total_wall_ms": (finished - started) * 1000,
        "forward_ms": events[0].elapsed_time(events[1]),
        "backward_ms": events[1].elapsed_time(events[2]),
        "forward_backward_ms": events[0].elapsed_time(events[2]),
        "loss": loss_value, **memory,
    }


def run_group(policy, training, models, optimizer, reserve, group, *, device, state, persist):
    state.update(phase="flow_inputs", active_case={
        "batch_size": group["batch_size"], "baseline_text_length": group["baseline_text_length"],
    })
    persist()
    flow_batch = make_flow_batch(reserve, group, device=device)
    for round_index, order in enumerate(group["round_orders"]):
        for length in order:
            state["active_case"] = {
                "batch_size": group["batch_size"], "baseline_text_length": group["baseline_text_length"],
                "text_length": length, "round": round_index,
            }
            persist()
            for call in range(policy.warmup_calls + policy.measured_calls):
                state["active_case"]["call"] = call
                record = measure_call(
                    training, models, optimizer, reserve, group, length, flow_batch, device=device, state=state,
                )
                group["calls"].append({
                    "text_length": length, "round": round_index, "call": call,
                    "phase": "warmup" if call < policy.warmup_calls else "measured", **record,
                })
            persist()
    group["status"] = "complete"


def profile_group(training, models, optimizer, reserve, group, *, device, state, output_dir, persist):
    """Separate untimed traces, after every throughput measurement has finished."""

    state.update(phase="profile_flow_inputs", active_case={
        "batch_size": group["batch_size"], "baseline_text_length": group["baseline_text_length"],
    })
    persist()
    flow_batch = make_flow_batch(reserve, group, device=device)
    directory = output_dir / "traces"
    directory.mkdir(exist_ok=True)
    for length in group["padded_lengths"]:
        state.update(phase="profiling", active_case={
            "batch_size": group["batch_size"], "baseline_text_length": group["baseline_text_length"],
            "text_length": length,
        })
        persist()
        # Rewarm this shape after intervening groups without adding calls to the statistics.
        measure_call(training, models, optimizer, reserve, group, length, flow_batch, device=device, state=state)
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize(device)
        state["phase"] = "profiling"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            with torch.profiler.record_function("flux_alignment_assembly"):
                microbatch = probe.build_probe_microbatch(
                    reserve, text_length=length, batch_size=group["batch_size"],
                    prefix_lengths=group["real_prefix_lengths"],
                )
            with torch.profiler.record_function("flux_alignment_forward"):
                loss = probe.compute_flux_microbatch_loss(
                    models.transformer, microbatch, compute_dtype=models.flux_compute_dtype, flow_batch=flow_batch,
                )
            with torch.profiler.record_function("flux_alignment_backward"):
                (loss * (group["batch_size"] / training.loop.optimizer_batch_size)).backward()
            torch.cuda.synchronize(device)
        path = directory / f"batch-{group['batch_size']}-base-{group['baseline_text_length']}-text-{length}.json.gz"
        profiler.export_chrome_trace(str(path))
        group["profiles"].append({
            "text_length": length, "trace": str(path.relative_to(output_dir)),
            "attention_operators": sorted({event.key for event in profiler.key_averages()
                                           if "attention" in event.key.lower()}),
        })
        del microbatch, loss, profiler
        persist()


def run_worker(args, policy, training):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "phase": "setup", "groups": []}

    def persist():
        probe._write_json(args.output_dir / "result.json", result)

    persist()
    try:
        result["environment"] = probe._environment_metadata()
        device, models, optimizer, reserve, setup = probe.prepare_runtime(training)
        properties = torch.cuda.get_device_properties(device)
        result.update(setup, gpu={"name": properties.name, "total_memory_bytes": properties.total_memory})
        for plan in build_plan(policy, training):
            group = {**plan, "status": "running", "calls": [], "profiles": []}
            result["groups"].append(group)
            run_group(policy, training, models, optimizer, reserve, group, device=device, state=result, persist=persist)
            persist()
            print(f"{probe.PROGRESS_MARKER} {group['batch_size']} {group['baseline_text_length']} measured", flush=True)
        for group in result["groups"]:
            if group["profile"]:
                profile_group(
                    training, models, optimizer, reserve, group, device=device, state=result,
                    output_dir=args.output_dir, persist=persist,
                )
                print(f"{probe.PROGRESS_MARKER} {group['batch_size']} {group['baseline_text_length']} profiled", flush=True)
        result.update(status="complete", phase="complete", active_case=None)
    except Exception as error:
        # Stop this context after an OOM; neighboring lengths are not declared infeasible.
        result.update(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) and result["phase"] != "setup" else "error",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
    finally:
        persist()
    return result


def statistics_for(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1],
        "p95": ordered[math.ceil(len(ordered) * .95) - 1],
    }


def summarize_groups(groups, policy):
    rows = []
    for group in groups:
        measured = [call for call in group["calls"] if call["phase"] == "measured"]
        baseline_calls = [call for call in measured if call["text_length"] == group["baseline_text_length"]]
        for length in group["padded_lengths"]:
            calls = [call for call in measured if call["text_length"] == length]
            if not calls:
                continue
            complete = len(calls) == policy.rounds * policy.measured_calls
            row = {
                "status": "complete" if complete else "partial",
                "batch_size": group["batch_size"], "baseline_text_length": group["baseline_text_length"],
                "text_length": length, "joint_tokens": length + IMAGE_TOKENS,
                "labels": group["labels"][str(length)], "measured_calls": len(calls),
                "timings": {name: statistics_for([call[name] for call in calls]) for name in TIMINGS},
                "round_forward_backward_ms": {
                    str(index): statistics.median(call["forward_backward_ms"] for call in calls if call["round"] == index)
                    for index in sorted({call["round"] for call in calls})
                },
                "peak_allocated_bytes": max(call["peak_allocated_bytes"] for call in group["calls"] if call["text_length"] == length),
                "peak_reserved_bytes": max(call["peak_reserved_bytes"] for call in group["calls"] if call["text_length"] == length),
            }
            if baseline_calls:
                row["max_absolute_loss_difference_from_baseline"] = max(
                    abs(call["loss"] - baseline_calls[0]["loss"]) for call in calls
                )
            if complete and len(baseline_calls) == policy.rounds * policy.measured_calls:
                row["speedup_vs_measured_baseline"] = statistics.median(
                    call["forward_backward_ms"] for call in baseline_calls
                ) / row["timings"]["forward_backward_ms"]["median"]
            rows.append(row)
    return sorted(rows, key=lambda row: (row["batch_size"], row["baseline_text_length"], row["text_length"]))


def validate_complete(record, plan, policy, *, output_dir):
    """Validate worker output once at ingestion, never during a model call."""

    if record.get("status") != "complete" or len(record.get("groups", [])) != len(plan):
        return False
    for group, expected in zip(record["groups"], plan, strict=True):
        if group.get("status") != "complete" or any(group.get(key) != value for key, value in expected.items()):
            return False
        expected_calls = [
            (length, index, call, "warmup" if call < policy.warmup_calls else "measured")
            for index, order in enumerate(expected["round_orders"]) for length in order
            for call in range(policy.warmup_calls + policy.measured_calls)
        ]
        if [(call.get("text_length"), call.get("round"), call.get("call"), call.get("phase"))
            for call in group.get("calls", [])] != expected_calls:
            return False
        for call in group["calls"]:
            for name in (*TIMINGS, "loss", "peak_allocated_bytes", "peak_reserved_bytes"):
                value = call.get(name)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    return False
                if name in TIMINGS and value == 0:
                    return False
        profiles = group.get("profiles", [])
        if [entry.get("text_length") for entry in profiles] != (expected["padded_lengths"] if expected["profile"] else []):
            return False
        if any(not (output_dir / entry.get("trace", "")).is_file() for entry in profiles):
            return False
    return True


def write_report(path, summary):
    lines = [
        "# FLUX alignment diagnostic", "", f"Status: **{summary['status']}**. No training policy was changed.", "",
        "Times are medians of measured calls only. Speedup compares the freshly measured baseline within the same group.",
        "Partial rows have incomplete sampling. Profiler calls are excluded. Raw calls and per-round medians are in JSON.", "",
        "| B | Original text | Padded text | Joint tokens | Alignment | Fwd ms | Bwd ms | F+B ms | Assembly ms | Total wall ms | Peak allocated GiB | Speedup | Status |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["cases"]:
        timings = row["timings"]
        speedup = row.get("speedup_vs_measured_baseline")
        cells = [
            str(row["batch_size"]), str(row["baseline_text_length"]), str(row["text_length"]), str(row["joint_tokens"]),
            ", ".join(row["labels"]),
            *[f"{timings[name]['median']:.3f}" for name in ("forward_ms", "backward_ms", "forward_backward_ms", "assembly_ms", "total_wall_ms")],
            f"{row['peak_allocated_bytes'] / 2**30:.3f}", f"{speedup:.4f}x" if speedup is not None else "—", row["status"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, policy, training, *, metadata=None, on_progress=None):
    hashes = probe.config_hashes(args.config, policy)
    if metadata is None:
        identity = probe.source_metadata()
    else:
        if metadata["source_sha256"] != probe.source_hashes() or metadata["config_sha256"] != hashes:
            raise ValueError("uploaded code or configuration differs from launch metadata")
        identity = {key: metadata[key] for key in ("git_commit", "git_status", "source_sha256")}
    plan = build_plan(policy, training)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "worker"
    manifest = {
        "schema_version": 2, "mode": "flux_alignment", "status": "running", "command": sys.argv,
        "probe_config": asdict(policy), "training_config": asdict(training), "config_sha256": hashes,
        **identity, "launch_metadata": metadata, "plan": plan, "image_tokens": IMAGE_TOKENS,
        "workload": "synthetic_30x30_source_template_with_fixed_near_boundary_prefix_slices",
        "dataset_split": None, "checkpoint_weights_hashed": False,
        "optimizer_learning_rate_override": 0.0, "optimizer_steps_during_comparisons": 0,
        "training_policy_validated": False,
    }
    summary = {"schema_version": 2, "mode": "flux_alignment", "status": "running", "cases": [], "training_policy_validated": False}
    probe._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()

    def persist_progress():
        path = destination / "result.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        summary["cases"] = summarize_groups(record.get("groups", []), policy)
        summary["worker_status"] = record.get("status")
        summary["active_case"] = record.get("active_case")
        summary["worker_phase"] = record.get("phase")
        if record.get("error"):
            summary["error"] = record["error"]
        probe._write_json(args.output_dir / "summary.json", summary)
        if on_progress is not None:
            on_progress()
        return record

    try:
        persist_progress()
        command = [
            sys.executable, "-B", "-u", str(Path(probe.__file__).resolve()),
            "--config", str(args.config.resolve()), "--output-dir", str(destination.resolve()), "--alignment-worker",
        ]
        remaining = policy.max_duration_seconds - (time.monotonic() - started)
        if remaining <= 0:
            summary["status"] = "timeout"
        else:
            returncode, timed_out = probe.launch_worker(
                command, log_path=args.output_dir / "worker.log", timeout_seconds=remaining, on_progress=persist_progress,
            )
            record = persist_progress()
            summary.update(returncode=returncode, timed_out=timed_out)
            if timed_out:
                summary["status"] = "timeout"
            elif returncode == 0 and validate_complete(record, plan, policy, output_dir=destination):
                summary["status"] = "complete"
            else:
                summary.update(status="error", error=record.get("error", "invalid or incomplete worker result"))
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = summary["status"]
        probe._write_json(args.output_dir / "summary.json", summary)
        probe._write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", summary)
        if on_progress is not None:
            on_progress()
    return summary
