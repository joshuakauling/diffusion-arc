#!/usr/bin/env python3
"""Compare masked NVARC shapes and aligned neighbors with resident training memory."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.model.contracts import (
    NVARC_HIDDEN_STATE_INDICES, NVARC_MAX_TOKENS,
    validate_packed_source_sequences,
)
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    build_flux_optimizer, load_single_gpu_models, prepare_single_gpu_device,
    seed_single_gpu_process,
)
from scripts.diagnostics.probe_nvarc_batch_capacity import (
    _environment_metadata, _memory, _write_json, allocate_buffer_reserve,
    initialize_training_memory, make_sample,
)


PROGRESS_MARKER = "NVARC_GROUP_COMPLETE"


@dataclass(frozen=True)
class ProbeConfig:
    training_config: Path
    alignment_multiples: tuple[int, ...]
    warmup_calls: int
    measured_calls: int
    rounds: int
    max_duration_seconds: int
    profile_groups: tuple[tuple[int, int], ...]


def load_probe_config(path: Path, *, require_checkpoints: bool = True):
    """Validate user policy and the production configuration before loading models."""

    raw = json.loads(path.read_text())
    fields = {"schema_version", *ProbeConfig.__dataclass_fields__}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"probe config requires exactly these fields: {sorted(fields)}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("probe schema_version must be 1")
    for name in ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    multiples = raw["alignment_multiples"]
    if (
        not isinstance(multiples, list) or not multiples
        or any(type(value) is not int or value not in (8, 32, 64, 128) for value in multiples)
        or multiples != sorted(set(multiples))
    ):
        raise ValueError("alignment_multiples must be an increasing subset of 8, 32, 64, 128")
    profiles = raw["profile_groups"]
    if (
        not isinstance(profiles, list)
        or any(not isinstance(row, list) or len(row) != 2
               or any(type(value) is not int for value in row) for row in profiles)
        or len({tuple(row) for row in profiles}) != len(profiles)
    ):
        raise ValueError("profile_groups must contain unique [batch_size, baseline_tokens] pairs")
    if not isinstance(raw["training_config"], str) or not raw["training_config"]:
        raise ValueError("training_config must name a configuration file")
    policy = ProbeConfig(
        training_config=(path.resolve().parent / raw["training_config"]).resolve(),
        alignment_multiples=tuple(multiples), profile_groups=tuple(map(tuple, profiles)),
        **{name: raw[name] for name in
           ("warmup_calls", "measured_calls", "rounds", "max_duration_seconds")},
    )
    training = load_single_gpu_training_config(policy.training_config)
    if (
        training.nvarc_scheduling is None
        or training.loop.optimizer_batch_size != 48
        or training.loop.nvarc_source_batch_size != 16
        or training.loop.vae_microbatch_size != 48
        or training.loop.min_context_pairs_before_target != 2
    ):
        raise ValueError("probe requires the scheduled 16-source, 48-target B2 layout")
    if (
        training.models.nvarc_dtype != "bfloat16" or training.models.vae_dtype != "float32"
        or training.models.flux_parameter_dtype != "float32"
        or training.models.flux_compute_dtype != "bfloat16"
    ):
        raise ValueError("probe requires the benchmark FP32/BF16 precision policy")
    shapes = training.nvarc_scheduling.shapes
    if any(shape.max_tokens <= shape.batch_size for shape in shapes):
        raise ValueError("each baseline must leave room for positive, differently padded sources")
    if not set(policy.profile_groups) <= {(s.batch_size, s.max_tokens) for s in shapes}:
        raise ValueError("profile_groups must refer to baseline shapes in the training catalog")
    if require_checkpoints:
        for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"local checkpoint directory does not exist: {checkpoint}")
    return policy, training


def build_plan(policy, training) -> list[dict]:
    """Keep original batches and deduplicate rounded lengths within each comparison."""

    groups = []
    rng = random.Random(training.seed)
    shapes = list(training.nvarc_scheduling.shapes)
    rng.shuffle(shapes)
    for shape in shapes:
        base = shape.max_tokens
        rounded = {base, *(((base + k - 1) // k) * k for k in policy.alignment_multiples)}
        lengths = sorted(value for value in rounded if value <= NVARC_MAX_TOKENS)
        order = lengths.copy()
        rng.shuffle(order)
        orders = [order[index % len(order):] + order[:index % len(order)]
                  for index in range(policy.rounds)]
        groups.append({
            "batch_size": shape.batch_size, "baseline_tokens": base,
            "real_lengths": [base - 1 - row for row in range(shape.batch_size)],
            "padded_lengths": lengths, "round_orders": orders,
            "excluded_above_limit": sorted(rounded - set(lengths)),
            "profile": (shape.batch_size, base) in policy.profile_groups,
        })
    return groups


def build_group_inputs(templates, group):
    """Slice real token templates once; every candidate receives these exact CPU tensors."""

    lengths = group["real_lengths"]
    ids = torch.cat([template[:length] for template, length in zip(templates, lengths)])
    offsets = torch.tensor([0, *accumulate(lengths)], dtype=torch.int64)
    validate_packed_source_sequences(input_ids=ids, sequence_offsets=offsets)
    return ids, offsets


def collect_comparison_outputs(runner, ids, offsets, *, padded_length):
    """Run the production path unchanged and collect compact outputs outside timing."""

    lengths = (offsets[1:] - offsets[:-1]).tolist()
    output = runner(ids, offsets, padded_length=padded_length)
    expected_width = runner.model.config.hidden_size * len(NVARC_HIDDEN_STATE_INDICES)
    expected_dtype = runner.model.get_input_embeddings().weight.dtype
    states = output.sequence_hidden_states
    if len(states) != len(lengths):
        raise ValueError("NVARC must return one compact output per source")
    cpu_states = []
    for state, length in zip(states, lengths):
        if (state.shape != (length, expected_width) or state.dtype != expected_dtype
                or state.requires_grad or not bool(torch.isfinite(state).all())):
            raise ValueError("invalid compact frozen NVARC output")
        cpu_states.append(state.detach().cpu().clone())
    return tuple(cpu_states)


def numerical_difference(states, reference):
    """Report full real-position differences on CPU, outside all latency measurements."""

    maximum = squared_error = squared_reference = 0.0
    count = 0
    for actual, expected in zip(states, reference):
        actual, expected = actual.float(), expected.float()
        delta = actual - expected
        maximum = max(maximum, float(delta.abs().max()))
        squared_error += float(delta.square().double().sum())
        squared_reference += float(expected.square().double().sum())
        count += actual.numel()
    return {
        "max_abs": maximum, "rms": math.sqrt(squared_error / count),
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else None,
        "reference_is_zero": squared_reference == 0,
        "agreement_threshold_applied": False,
    }


def measure_call(runner, ids, offsets, *, padded_length, device, events):
    """Time the whole production runner, including H2D, padding, masks, and extraction."""

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    events[0].record()
    output = runner(ids, offsets, padded_length=padded_length)
    events[1].record()
    events[1].synchronize()
    wall_ms = (time.perf_counter() - started) * 1000
    gpu_span_ms = events[0].elapsed_time(events[1])
    memory = _memory(device)
    # Release the previous output before the next call, as in sequential preparation.
    del output
    return {"wall_ms": wall_ms, "gpu_span_ms": gpu_span_ms, **memory}


def profile_call(runner, ids, offsets, *, padded_length, device, path):
    """Collect one separate warm CPU/CUDA trace; profiler time is never a cost sample."""

    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profiler:
        with torch.profiler.record_function(f"probe/nvarc/{len(offsets) - 1}x{padded_length}"):
            output = runner(ids, offsets, padded_length=padded_length)
        torch.cuda.synchronize(device)
    del output
    profiler.export_chrome_trace(str(path))
    return {
        "trace": str(path), "sdpa_operators": {
            event.key: event.count for event in profiler.key_averages()
            if "scaled_dot_product" in event.key
        },
    }


def summarize_case(case, *, rounds):
    measured = [row for row in case["calls"] if row["phase"] == "measured"]
    for metric in ("wall_ms", "gpu_span_ms"):
        values = sorted(row[metric] for row in measured)
        case[metric] = {
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "round_medians": [statistics.median(
                row[metric] for row in measured if row["round"] == index
            ) for index in range(rounds)],
        }
    for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
        case[name] = max(row[name] for row in case["calls"])
    case["status"] = "success"


def run_group(policy, runner, templates, group, *, device, directory, persist):
    """Compare outputs, then interleave warmed measurements in seeded rounds."""

    ids, offsets = build_group_inputs(templates, group)
    group["input_ids_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    group["cases"] = [{"padded_length": length, "status": "pending", "calls": []}
                      for length in group["padded_lengths"]]
    reference = None
    current = None
    try:
        for current in group["cases"]:
            current.update(status="running", phase="output_comparison")
            persist()
            states = collect_comparison_outputs(
                runner, ids, offsets, padded_length=current["padded_length"],
            )
            if reference is None:
                reference = states
            current["numerical_difference"] = numerical_difference(states, reference)
            del states
            current["status"] = "checked"
            persist()
        del reference
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        cases = {case["padded_length"]: case for case in group["cases"]}
        for index, order in enumerate(group["round_orders"]):
            for length in order:
                current = cases[length]
                current.update(status="running", phase="timing")
                for call in range(policy.warmup_calls + policy.measured_calls):
                    row = measure_call(runner, ids, offsets, padded_length=length,
                                       device=device, events=events)
                    current["calls"].append({
                        "round": index, "call": call,
                        "phase": "warmup" if call < policy.warmup_calls else "measured", **row,
                    })
                    persist()
        for current in group["cases"]:
            summarize_case(current, rounds=policy.rounds)
        baseline = cases[group["baseline_tokens"]]
        for current in group["cases"]:
            current["median_paired_wall_ratio"] = statistics.median(
                a / b for a, b in zip(current["wall_ms"]["round_medians"],
                                    baseline["wall_ms"]["round_medians"])
            )
        persist()
        if group["profile"]:
            for current in group["cases"]:
                current["phase"] = "profile"
                persist()
                current["profile"] = profile_call(
                    runner, ids, offsets, padded_length=current["padded_length"], device=device,
                    path=directory / f"trace-{group['batch_size']}x{current['padded_length']}.json",
                )
                persist()
        group["status"] = "complete"
    except Exception as error:
        status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error"
        if current is not None:
            current.update(status=status, error=f"{type(error).__name__}: {error}")
        group.update(status=status, error=f"{type(error).__name__}: {error}",
                     traceback=traceback.format_exc())
        # No capacity or monotonicity inference; the supervisor preserves this partial run.
        raise
    finally:
        persist()


def prepare_runtime(training):
    device = prepare_single_gpu_device(training)
    seed_single_gpu_process(training.seed)
    models = load_single_gpu_models(training, device=device)
    if models.transformer.gradient_checkpointing:
        raise ValueError("probe requires activation checkpointing disabled")
    optimizer = build_flux_optimizer(training, transformer=models.transformer)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    initialization = initialize_training_memory(
        replace(training, scheduler=None), models, optimizer, device=device,
    )
    reserve, reserve_metadata = allocate_buffer_reserve(
        training, device=device, sequence_length=NVARC_MAX_TOKENS,
    )
    templates = []
    for row in range(max(shape.batch_size for shape in training.nvarc_scheduling.shapes)):
        sample = make_sample(models.tokenizer, seed=training.seed + row)
        if sample.prefix_token_ends[-1] != NVARC_MAX_TOKENS:
            raise ValueError("synthetic token template must contain exactly 8400 consumed tokens")
        templates.append(torch.tensor(sample.input_ids[:NVARC_MAX_TOKENS], dtype=torch.int64))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return device, models, optimizer, reserve, templates, {
        "initialization": initialization, "buffer_reserve": reserve_metadata,
        "setup_memory": _memory(device),
        "checkpoint_configs": {
            str(path): json.loads(path.read_text()) for path in (
                training.models.nvarc_checkpoint / "config.json",
                training.models.flux_checkpoint / "transformer/config.json",
                training.models.flux_checkpoint / "vae/config.json",
            ) if path.is_file()
        },
    }


def source_hashes():
    """Fingerprint uploaded code without requiring Git in the GPU container."""

    files = [Path(__file__), REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_batch_shapes_modal.py",
             REPOSITORY_ROOT / "scripts/diagnostics/probe_nvarc_batch_capacity.py",
             REPOSITORY_ROOT / "pyproject.toml", *sorted((REPOSITORY_ROOT / "src").rglob("*.py"))]
    return {str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def source_metadata():
    """Capture local checkout identity; remote runs receive it from the launcher."""

    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT,
                                    capture_output=True, text=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "source_sha256": source_hashes(),
    }


def config_hashes(path, policy):
    return {
        "probe": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training": hashlib.sha256(policy.training_config.read_bytes()).hexdigest(),
    }


def run_worker(args):
    result = {"status": "running", "phase": "setup", "groups": []}
    path = args.output_dir / "result.json"

    def persist():
        _write_json(path, result)

    persist()
    try:
        policy, training = load_probe_config(args.config)
        result.update(environment=_environment_metadata(), groups=build_plan(policy, training))
        device, models, optimizer, reserve, templates, metadata = prepare_runtime(training)
        properties = torch.cuda.get_device_properties(device)
        result.update(runtime=metadata, gpu={
            "name": properties.name, "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "visible_count": torch.cuda.device_count(), "device": str(device),
        })
        # Keep models, optimizer, gradients, and reserve alive throughout every comparison.
        result["phase"] = "groups"
        for group in result["groups"]:
            run_group(policy, models.nvarc_runner, templates, group,
                      device=device, directory=args.output_dir, persist=persist)
            print(f"{PROGRESS_MARKER} {group['batch_size']}x{group['baseline_tokens']}", flush=True)
        result["status"] = "complete"
    except Exception as error:
        result.update(
            status="oom" if result["phase"] == "groups" and isinstance(error, torch.cuda.OutOfMemoryError)
            else "error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
    finally:
        persist()
    return result


def validate_completion(result, policy, plan):
    """Reject truncated or inconsistent worker output before reporting a completed sweep."""

    groups = result.get("groups", [])
    if result.get("status") != "complete" or len(groups) != len(plan):
        return False
    for group, expected in zip(groups, plan):
        if group.get("status") != "complete" or any(group.get(key) != value for key, value in expected.items()):
            return False
        cases = group.get("cases", [])
        if [case.get("padded_length") for case in cases] != expected["padded_lengths"]:
            return False
        expected_calls = [(r, "warmup" if c < policy.warmup_calls else "measured")
                          for r in range(policy.rounds)
                          for c in range(policy.warmup_calls + policy.measured_calls)]
        for case in cases:
            calls = case.get("calls", [])
            if (case.get("status") != "success"
                    or [(c.get("round"), c.get("phase")) for c in calls] != expected_calls
                    or "numerical_difference" not in case
                    or (expected["profile"] and not case.get("profile"))):
                return False
            if any(not isinstance(c.get(metric), (float, int)) or not math.isfinite(c[metric])
                   or c[metric] <= 0 for c in calls for metric in ("wall_ms", "gpu_span_ms")):
                return False
    return True


def write_report(path, result):
    """Flatten complete and partial comparisons without turning missing timings into zero."""

    lines = [
        "# NVARC shape measurements", "", f"Sweep status: **{result['status']}**.", "",
        "Ratios compare round medians with the unchanged baseline; below 1 means lower observed latency.",
        "These isolated timings do not establish a validated training policy or statistical significance.", "",
        "| Batch | Baseline | Padding | Status | Wall median (ms) | GPU span median (ms) | Paired wall ratio | Peak allocated (GiB) |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in result.get("groups", []):
        for case in group.get("cases", []):
            def number(value):
                return "—" if value is None else f"{value:.3f}"

            peak = case.get("peak_allocated_bytes")
            values = [group.get("batch_size", "—"), group.get("baseline_tokens", "—"),
                      case.get("padded_length", "—"), case.get("status", "unknown"),
                      number(case.get("wall_ms", {}).get("median")),
                      number(case.get("gpu_span_ms", {}).get("median")),
                      number(case.get("median_paired_wall_ratio")),
                      number(peak / 2**30 if peak is not None else None)]
            lines.append("| " + " | ".join(map(str, values)) + " |")
    path.write_text("\n".join(lines) + "\n")


def launch_worker(command, *, log_path, timeout_seconds, on_progress):
    """Stream group completion to the supervisor while enforcing the GPU worker budget."""

    with log_path.open("w") as log, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        timed_out = threading.Event()

        def expire():
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(timeout_seconds, expire)
        timer.start()
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(PROGRESS_MARKER + " "):
                    print(line.strip(), flush=True)
                    on_progress()
            returncode = process.wait()
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                process.kill()
                process.wait()
    return returncode, timed_out.is_set()


def run_probe(args, *, metadata=None, on_progress: Callable[[], None] | None = None):
    policy, training = load_probe_config(args.config)
    plan = build_plan(policy, training)
    hashes = config_hashes(args.config, policy)
    if metadata is None:
        identity = source_metadata()
    else:
        if metadata.get("source_sha256") != source_hashes() or metadata.get("config_sha256") != hashes:
            raise ValueError("uploaded code or configuration differs from launch metadata")
        identity = {key: metadata[key] for key in ("git_commit", "git_status", "source_sha256")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "command": sys.argv, "probe_config": asdict(policy),
        "training_config": asdict(training), "plan": plan, **identity,
        "config_sha256": hashes, "launch_metadata": metadata,
        "workload": "fixed_near_boundary_slices_of_seeded_synthetic_ARC_sources",
        "dataset_split": None, "checkpoint_weights_hashed": False,
        "optimizer_learning_rate_override": 0.0,
        "training_policy_validated": False,
        "failure_policy": "stop on first failure; no automatic retry or capacity inference",
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--config", str(args.config.resolve()), "--output-dir", str(args.output_dir.resolve()),
               "--worker"]
    started = time.monotonic()
    result = {}

    def persist_progress():
        partial = json.loads((args.output_dir / "result.json").read_text())
        partial.update(status="running", training_policy_validated=False)
        _write_json(args.output_dir / "summary.json", partial)
        if on_progress is not None:
            on_progress()

    try:
        if on_progress is not None:
            on_progress()
        returncode, timed_out = launch_worker(
            command, log_path=args.output_dir / "worker.log",
            timeout_seconds=policy.max_duration_seconds, on_progress=persist_progress,
        )
        path = args.output_dir / "result.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        valid = returncode == 0 and validate_completion(result, policy, plan)
        result["worker_status"] = result.get("status")
        result["status"] = "timeout" if timed_out else "complete" if valid else "error"
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if not timed_out and not valid:
            result.setdefault("error", "invalid or incomplete worker result")
    except BaseException as error:
        # launch_worker has already stopped the child; preserve its last atomic record.
        partial_path = args.output_dir / "result.json"
        try:
            if partial_path.is_file():
                result = json.loads(partial_path.read_text())
        except (OSError, ValueError) as read_error:
            result["partial_result_read_error"] = f"{type(read_error).__name__}: {read_error}"
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started, training_policy_validated=False)
        manifest["status"] = result["status"]
        _write_json(args.output_dir / "summary.json", result)
        _write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", result)
        if on_progress is not None:
            on_progress()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without model loads or writes")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dry_run:
        policy, training = load_probe_config(args.config, require_checkpoints=False)
        groups = build_plan(policy, training)
        print(json.dumps({"groups": groups, "candidate_cases": sum(len(g["padded_lengths"]) for g in groups),
                          "unique_physical_shapes": len({(g["batch_size"], l)
                                                        for g in groups for l in g["padded_lengths"]})}, indent=2))
        return 0
    if args.output_dir is None:
        args.output_dir = REPOSITORY_ROOT / "artifacts/nvarc-batch-shapes" / datetime.now(
            timezone.utc,
        ).strftime("%Y%m%dT%H%M%S%fZ")
    result = run_worker(args) if args.worker else run_probe(args)
    print(f"NVARC shape probe {result['status']}: {args.output_dir}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
