#!/usr/bin/env python3
"""Measure padded FLUX forward/backward shapes with resident training memory."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
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

from diffusion_arc.model.contracts import NVARC_MAX_TOKENS
from diffusion_arc.sequence_buckets import SequenceLengthBuckets
from diffusion_arc.training.flux_input import FluxInput, NvarcSequenceState, prepare_flux_inputs_batch
from diffusion_arc.training.flux_loss import compute_flux_microbatch_loss
from diffusion_arc.training.flux_microbatch import build_flux_microbatch
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    build_flux_optimizer, load_single_gpu_models, prepare_single_gpu_device, seed_single_gpu_process,
)
from scripts.diagnostics.probe_nvarc_batch_capacity import (
    _environment_metadata, _memory, _write_json, allocate_buffer_reserve,
    initialize_training_memory, make_sample,
)


OOM_EXIT_CODE = 2
PROGRESS_MARKER = "FLUX_SHAPE_COMPLETE"


@dataclass(frozen=True)
class ProbeConfig:
    training_config: Path
    text_bucket_boundaries: tuple[int, ...]
    microbatch_sizes: tuple[int, ...]
    warmup_calls: int
    measured_calls: int
    max_duration_seconds: int


def load_probe_config(path: Path, *, require_checkpoints: bool = True):
    """Validate both user configurations before any model load or remote launch."""

    raw = json.loads(path.read_text())
    fields = {
        "schema_version", "training_config", "text_bucket_boundaries", "microbatch_sizes",
        "warmup_calls", "measured_calls", "max_duration_seconds",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"probe config requires exactly these fields: {sorted(fields)}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("probe schema_version must be 1")
    for name in ("warmup_calls", "measured_calls", "max_duration_seconds"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("text_bucket_boundaries", "microbatch_sizes"):
        values = raw[name]
        if (
            not isinstance(values, list) or not values
            or any(type(value) is not int for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"{name} must be a nonempty, strictly increasing integer list")
    if any(not 2 <= value <= NVARC_MAX_TOKENS for value in raw["text_bucket_boundaries"]):
        raise ValueError(f"text_bucket_boundaries must be between 2 and {NVARC_MAX_TOKENS}")
    if raw["microbatch_sizes"][0] != 1 or any(value not in (1, 2, 4, 8) for value in raw["microbatch_sizes"]):
        raise ValueError("microbatch_sizes must start at 1 and contain only 1, 2, 4, 8")
    if not isinstance(raw["training_config"], str) or not raw["training_config"]:
        raise ValueError("training_config must name a configuration file")
    policy = ProbeConfig(
        training_config=(path.resolve().parent / raw["training_config"]).resolve(),
        **{name: tuple(raw[name]) for name in ("text_bucket_boundaries", "microbatch_sizes")},
        **{name: raw[name] for name in ("warmup_calls", "measured_calls", "max_duration_seconds")},
    )
    training = load_single_gpu_training_config(policy.training_config)
    if (
        training.loop.optimizer_batch_size != 48
        or training.loop.nvarc_source_batch_size != 16
        or training.loop.vae_microbatch_size != 48
        or training.loop.min_context_pairs_before_target != 2
    ):
        raise ValueError("FLUX shape probe requires the 16-source, 48-target preparation layout")
    if (
        training.models.nvarc_dtype != "bfloat16" or training.models.vae_dtype != "float32"
        or training.models.flux_parameter_dtype != "float32"
        or training.models.flux_compute_dtype != "bfloat16"
    ):
        raise ValueError("FLUX shape probe requires the benchmark FP32/BF16 precision policy")
    if require_checkpoints:
        for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"local checkpoint directory does not exist: {checkpoint}")
    return policy, training


def source_hashes() -> dict:
    """Fingerprint uploaded files without requiring Git or a repository checkout."""

    sources = [
        Path(__file__), REPOSITORY_ROOT / "scripts/cloud/probe_flux_batch_shapes_modal.py",
        REPOSITORY_ROOT / "scripts/diagnostics/probe_nvarc_batch_capacity.py",
        REPOSITORY_ROOT / "pyproject.toml", *sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def source_metadata() -> dict:
    """Capture Git identity locally; remote workers verify only uploaded file hashes."""

    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "source_sha256": source_hashes(),
    }


def config_hashes(path: Path, policy: ProbeConfig) -> dict:
    return {
        "probe": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training": hashlib.sha256(policy.training_config.read_bytes()).hexdigest(),
    }


def prepare_runtime(training):
    """Initialize real frozen outputs, FP32 moments, and persistent gradients once."""

    device = prepare_single_gpu_device(training)
    seed_single_gpu_process(training.seed)
    models = load_single_gpu_models(training, device=device)
    if models.transformer.gradient_checkpointing:
        raise ValueError("this probe requires activation checkpointing disabled")
    optimizer = build_flux_optimizer(training, transformer=models.transformer)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    initialization = initialize_training_memory(
        replace(training, scheduler=None), models, optimizer, device=device,
    )
    reserve, reserve_metadata = allocate_buffer_reserve(training, device=device, sequence_length=NVARC_MAX_TOKENS)
    sample = make_sample(models.tokenizer, seed=training.seed)
    prepared = prepare_flux_inputs_batch(
        (sample,), nvarc_runner=models.nvarc_runner, vae_encoder=models.vae_encoder,
        min_context_pairs_before_target=2, vae_microbatch_size=3,
    )
    # Reuse storage already counted in the reserve for the synthetic input templates.
    if prepared[-1].prefix_end != NVARC_MAX_TOKENS:
        raise ValueError("synthetic source must cover the full 8400-token reserve")
    reserve[0].copy_(prepared[-1].nvarc_sequence_state._prefix_for(prepared[-1].prefix_end))
    reserve[-1][0].copy_(prepared[-1].clean_latent)
    del prepared
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return device, models, optimizer, reserve, {
        "initialization": initialization, "buffer_reserve": reserve_metadata,
        "setup_memory": _memory(device),
        "checkpoint_configs": {
            str(path): json.loads(path.read_text())
            for path in (
                training.models.nvarc_checkpoint / "config.json",
                training.models.flux_checkpoint / "transformer/config.json",
                training.models.flux_checkpoint / "vae/config.json",
            ) if path.is_file()
        },
    }


def build_probe_microbatch(reserve, *, text_length: int, batch_size: int):
    """Exercise production bucket padding, including a masked position in every row."""

    prefix_lengths = tuple(max(1, text_length - 1 - row) for row in range(batch_size))
    inputs = tuple(
        FluxInput(
            nvarc_sequence_state=NvarcSequenceState(
                hidden_states=reserve[0], pending_prefix_ends=(prefix,),
            ),
            prefix_end=prefix, clean_latent=reserve[-1][0],
        )
        for prefix in prefix_lengths
    )
    return build_flux_microbatch(inputs, length_buckets=SequenceLengthBuckets((text_length,)))


def run_shape(policy, training, models, optimizer, reserve, *, device, text_length, batch_size, result_path):
    """Warm one shape, then measure accumulation calls without freeing gradient storage."""

    result = {
        "status": "running", "phase": "assembly", "text_length": text_length,
        "batch_size": batch_size, "joint_tokens_per_target": text_length + 900,
        "real_prefix_lengths": [max(1, text_length - 1 - row) for row in range(batch_size)],
        "loss_weight": batch_size / training.loop.optimizer_batch_size,
        "calls": [],
    }
    _write_json(result_path, result)
    generator = torch.Generator(device=device).manual_seed(training.seed + text_length * 16 + batch_size)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    microbatch = loss = None
    try:
        for call in range(policy.warmup_calls + policy.measured_calls):
            # Zero values outside the timer, retaining the full accumulation footprint.
            optimizer.zero_grad(set_to_none=False)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["phase"] = "assembly"
            started = time.perf_counter()
            microbatch = build_probe_microbatch(reserve, text_length=text_length, batch_size=batch_size)
            torch.cuda.synchronize(device)
            assembly_seconds = time.perf_counter() - started
            result["phase"] = "forward_backward"
            started = time.perf_counter()
            events[0].record()
            loss = compute_flux_microbatch_loss(
                models.transformer, microbatch, compute_dtype=models.flux_compute_dtype, generator=generator,
            )
            if not bool(torch.isfinite(loss.detach()).item()):
                raise FloatingPointError("non-finite FLUX probe loss")
            events[1].record()
            (loss * result["loss_weight"]).backward()
            events[2].record()
            torch.cuda.synchronize(device)
            row = {
                "call": call + 1, "phase": "warmup" if call < policy.warmup_calls else "measured",
                "assembly_seconds": assembly_seconds,
                "wall_seconds": time.perf_counter() - started,
                "forward_seconds": events[0].elapsed_time(events[1]) / 1000,
                "backward_seconds": events[1].elapsed_time(events[2]) / 1000,
                "forward_backward_seconds": events[0].elapsed_time(events[2]) / 1000,
                "loss": float(loss.detach().item()), **_memory(device),
            }
            result["calls"].append(row)
            _write_json(result_path, result)
        # AdamW runs separately so its constant cost does not distort per-target timings.
        result["phase"] = "optimizer"
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        result["optimizer"] = {"seconds": time.perf_counter() - started, **_memory(device)}
        measured = result["calls"][policy.warmup_calls:]
        result.update(
            status="success", first_call=result["calls"][0],
            median_forward_seconds=statistics.median(row["forward_seconds"] for row in measured),
            median_backward_seconds=statistics.median(row["backward_seconds"] for row in measured),
            median_forward_backward_seconds=statistics.median(row["forward_backward_seconds"] for row in measured),
            median_wall_seconds=statistics.median(row["wall_seconds"] for row in measured),
        )
        result["seconds_per_target"] = result["median_forward_backward_seconds"] / batch_size
        result["targets_per_second"] = 1 / result["seconds_per_target"]
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            result[name] = max(row[name] for row in [*result["calls"], result["optimizer"]])
        total = torch.cuda.get_device_properties(device).total_memory
        result["allocated_headroom_bytes"] = total - result["peak_allocated_bytes"]
        result["reserved_headroom_bytes"] = total - result["peak_reserved_bytes"]
    except Exception as error:
        result.update(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        try:
            result["failure_memory"] = _memory(device)
        except Exception:
            pass
    finally:
        _write_json(result_path, result)
    return result


def run_worker(args) -> dict:
    """Reuse loaded models until the first OOM; never reuse that CUDA context afterward."""

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result = {"status": "running", "phase": "setup", "cases": [], "next_length_index": args.worker_start_length_index}
    _write_json(result_path, result)
    try:
        policy, training = load_probe_config(args.config)
        if not 0 <= args.worker_start_length_index < len(policy.text_bucket_boundaries):
            raise ValueError("invalid worker start index")
        result["environment"] = _environment_metadata()
        device, models, optimizer, reserve, setup = prepare_runtime(training)
        properties = torch.cuda.get_device_properties(device)
        result.update(setup, gpu={"name": properties.name, "total_memory_bytes": properties.total_memory})
        result["phase"] = "probe"
        _write_json(result_path, result)
        for index in range(args.worker_start_length_index, len(policy.text_bucket_boundaries)):
            length = policy.text_bucket_boundaries[index]
            for size_index, size in enumerate(policy.microbatch_sizes):
                record = run_shape(
                    policy, training, models, optimizer, reserve, device=device,
                    text_length=length, batch_size=size,
                    result_path=args.output_dir / f"length-{length}-batch-{size}.json",
                )
                result["cases"].append(record)
                if record["status"] == "oom":
                    result["cases"].extend({
                        "status": "skipped", "text_length": length, "batch_size": larger,
                        "reason": "larger_batch_after_oom_at_same_length",
                    } for larger in policy.microbatch_sizes[size_index + 1:])
                    result.update(status="oom", next_length_index=index + 1)
                elif record["status"] != "success":
                    result["status"] = "error"
                _write_json(result_path, result)
                print(f"{PROGRESS_MARKER} {length} {size} {record['status']}", flush=True)
                if record["status"] != "success":
                    return result
            result["next_length_index"] = index + 1
        result["status"] = "complete"
    except Exception as error:
        # A model-load or initialization OOM is not evidence about a candidate shape.
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        _write_json(result_path, result)
    return result


def launch_worker(command, *, log_path, timeout_seconds, on_progress):
    """Stream progress and enforce the remaining wall-clock budget, including setup."""

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


def _valid_worker_result(record, *, policy, start, returncode):
    """Require a complete ordered matrix segment before advancing the supervisor."""

    status = record.get("status")
    end = record.get("next_length_index")
    if status not in ("complete", "oom") or returncode != (0 if status == "complete" else OOM_EXIT_CODE):
        return False
    if type(end) is not int or not start < end <= len(policy.text_bucket_boundaries):
        return False
    if status == "complete" and end != len(policy.text_bucket_boundaries):
        return False
    expected = [(length, size) for length in policy.text_bucket_boundaries[start:end] for size in policy.microbatch_sizes]
    cases = record.get("cases", [])
    if [(row.get("text_length"), row.get("batch_size")) for row in cases] != expected:
        return False
    saw_oom = False
    for row in cases:
        if row.get("status") == "success" and not saw_oom:
            calls = row.get("calls", [])
            if [call.get("phase") for call in calls] != ["warmup"] * policy.warmup_calls + ["measured"] * policy.measured_calls:
                return False
        elif row.get("status") == "oom" and not saw_oom and row["text_length"] == policy.text_bucket_boundaries[end - 1]:
            saw_oom = True
        elif row.get("status") != "skipped" or not saw_oom:
            return False
    return saw_oom == (status == "oom")


def run_probe(args, *, metadata=None, on_progress: Callable[[], None] | None = None) -> dict:
    """Run one bounded sweep, persist each shape, and restart only after candidate OOMs."""

    policy, training = load_probe_config(args.config)
    hashes = config_hashes(args.config, policy)
    if metadata is None:
        identity = source_metadata()
    else:
        if metadata["source_sha256"] != source_hashes() or metadata["config_sha256"] != hashes:
            raise ValueError("uploaded code or configuration differs from launch metadata")
        identity = {key: metadata[key] for key in ("git_commit", "git_status", "source_sha256")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "command": sys.argv, "probe_config": asdict(policy),
        "training_config": asdict(training), "config_sha256": hashes, **identity,
        "launch_metadata": metadata, "optimizer_learning_rate_override": 0.0,
        "workload": "synthetic_30x30_source_template_with_near_boundary_prefix_slices",
        "dataset_split": None, "checkpoint_weights_hashed": False,
        "skip_assumption": "larger batches at the same length are skipped after the first OOM",
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    deadline = started + policy.max_duration_seconds
    completed_cases = []
    workers = []
    summary = {"status": "running", "cases": [], "workers": workers}
    start = 0
    try:
        while start < len(policy.text_bucket_boundaries):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                summary["status"] = "timeout"
                break
            destination = args.output_dir / f"worker-{len(workers) + 1:02d}"
            command = [
                sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                "--config", str(args.config.resolve()), "--output-dir", str(destination.resolve()),
                "--worker-start-length-index", str(start),
            ]

            def persist_progress():
                partial = json.loads((destination / "result.json").read_text())
                summary["cases"] = [*completed_cases, *partial["cases"]]
                _write_json(args.output_dir / "summary.json", summary)
                if on_progress is not None:
                    on_progress()

            returncode, timed_out = launch_worker(
                command, log_path=destination.with_suffix(".log"),
                timeout_seconds=remaining, on_progress=persist_progress,
            )
            result_path = destination / "result.json"
            record = json.loads(result_path.read_text()) if result_path.is_file() else {}
            workers.append({"result_path": str(result_path), "returncode": returncode, "timed_out": timed_out})
            summary["cases"] = [*completed_cases, *record.get("cases", [])]
            if timed_out:
                summary["status"] = "timeout"
                break
            if not _valid_worker_result(record, policy=policy, start=start, returncode=returncode):
                summary.update(status="error", error=record.get("error", "invalid or incomplete worker result"))
                break
            completed_cases.extend(record["cases"])
            start = record["next_length_index"]
        else:
            summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["largest_tested_success_by_length"] = {
            str(length): max((row["batch_size"] for row in summary["cases"] if row["text_length"] == length and row["status"] == "success"), default=None)
            for length in policy.text_bucket_boundaries
        }
        summary["training_policy_validated"] = False
        manifest["status"] = summary["status"]
        _write_json(args.output_dir / "summary.json", summary)
        _write_json(args.output_dir / "manifest.json", manifest)
        if on_progress is not None:
            on_progress()
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-start-length-index", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        args.output_dir = REPOSITORY_ROOT / "artifacts/flux-batch-shapes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.worker_start_length_index is not None:
        result = run_worker(args)
        return {"complete": 0, "oom": OOM_EXIT_CODE, "error": 1}[result["status"]]
    result = run_probe(args)
    print(f"FLUX shape probe {result['status']}: {args.output_dir}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
