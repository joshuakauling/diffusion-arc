#!/usr/bin/env python3
"""Bound NVARC batch capacity at 8,400 tokens with resident training memory."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.arc_data.augmentation import (
    ArcAugmentationKey, ArcAugmentationSpec, ArcAugmentedTrainingSample, ArcGeometry,
)
from diffusion_arc.arc_data.prompt import render_training_prompt
from diffusion_arc.arc_data.tokenization import tokenize_nvarc_training_sample
from diffusion_arc.arc_data.types import ArcPair
from diffusion_arc.model.contracts import (
    NVARC_CONTEXT_WIDTH, NVARC_MAX_TOKENS, TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT, validate_packed_source_sequences,
)
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_loop import run_single_gpu_training
from diffusion_arc.training.single_gpu_setup import (
    build_flow_generator, build_flux_optimizer, build_flux_scheduler,
    load_single_gpu_models, prepare_single_gpu_device, seed_single_gpu_process,
)


TARGETS_PER_SOURCE = 3
OOM_EXIT_CODE = 2


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--sequence-length", type=int, choices=(NVARC_MAX_TOKENS,),
        default=NVARC_MAX_TOKENS,
        help="this probe uses complete worst-case five-pair ARC sources",
    )
    parser.add_argument("--max-batch-size", type=_positive_integer, default=64)
    parser.add_argument("--repetitions", type=_positive_integer, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--candidate-batch-size", type=_positive_integer, help=argparse.SUPPRESS,
    )
    return parser


def _load_config(path: Path, *, require_checkpoints: bool = True):
    config = load_single_gpu_training_config(path)
    if (
        config.loop.optimizer_batch_size != 48
        or config.loop.min_context_pairs_before_target != 2
    ):
        raise ValueError("capacity probe requires the 48-target, two-demo layout")
    if require_checkpoints:
        for checkpoint in (config.models.nvarc_checkpoint, config.models.flux_checkpoint):
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"local checkpoint directory does not exist: {checkpoint}")
    return config


def make_sample(tokenizer, *, seed: int, grid_size: int = 30, pair_count: int = 5):
    """Render and tokenize a deterministic synthetic task through production code."""

    rng = random.Random(seed)

    def grid():
        return tuple(tuple(rng.randrange(10) for _ in range(grid_size)) for _ in range(grid_size))

    sample = ArcAugmentedTrainingSample(
        key=ArcAugmentationKey(
            task_id="00000000",
            augmentation=ArcAugmentationSpec(
                geometry=ArcGeometry.IDENTITY,
                color_mapping=tuple(range(10)),
                pair_indices=tuple(range(pair_count)),
            ),
        ),
        pairs=tuple(ArcPair(grid(), grid()) for _ in range(pair_count)),
    )
    return tokenize_nvarc_training_sample(render_training_prompt(sample), tokenizer)


def make_probe_inputs(tokenizer, *, seed: int, batch_size: int, sequence_length: int):
    """Repeat one real-token source; the unused final output never enters NVARC."""

    sample = make_sample(tokenizer, seed=seed)
    source_end = sample.prefix_token_ends[-1]
    if source_end != sequence_length:
        raise ValueError(
            f"synthetic source has {source_end} real tokens, expected {sequence_length}"
        )
    source = torch.tensor(sample.input_ids[:source_end], dtype=torch.int64)
    input_ids = source.repeat(batch_size)
    offsets = torch.arange(batch_size + 1, dtype=torch.int64) * sequence_length
    validate_packed_source_sequences(input_ids=input_ids, sequence_offsets=offsets)
    return input_ids, offsets


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def initialize_training_memory(config, models, optimizer, *, device):
    """Allocate real AdamW state and retain gradients for accumulation capacity probes."""

    sample = make_sample(models.tokenizer, seed=config.seed, grid_size=1, pair_count=2)
    gradients = {}

    def retain_gradients(_loss, _targets, _microbatches):
        # Keep the actual buffers across the training loop's end-of-step cleanup.
        gradients.update((parameter, parameter.grad) for parameter in models.transformer.parameters())

    # Apply the configured initial LR, including its zero-LR first warm-up step.
    scheduler = build_flux_scheduler(config, optimizer=optimizer)
    run_single_gpu_training(
        (sample,), nvarc_runner=models.nvarc_runner, vae_encoder=models.vae_encoder,
        transformer=models.transformer, optimizer=optimizer,
        flux_compute_dtype=models.flux_compute_dtype,
        max_optimizer_steps=1, optimizer_batch_size=1, flux_microbatch_size=1,
        min_context_pairs_before_target=1, nvarc_source_batch_size=1,
        vae_microbatch_size=1, generator=build_flow_generator(config, device=device),
        on_optimizer_step=retain_gradients,
    )
    for parameter, gradient in gradients.items():
        parameter.grad = gradient
    gradient_bytes = optimizer_state_bytes = 0
    for parameter in models.transformer.parameters():
        if not parameter.requires_grad:
            raise RuntimeError("the complete FLUX transformer must be trainable")
        if parameter.grad is None:
            raise RuntimeError("initialization did not populate every FLUX gradient")
        gradient_bytes += _tensor_bytes(parameter.grad)
        state = optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if (
                not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                or value.dtype != parameter.dtype or value.device != parameter.device
            ):
                raise RuntimeError(f"initialization did not populate AdamW {name}")
            optimizer_state_bytes += _tensor_bytes(value)
    return {
        "optimizer_steps": 1, "initialization_targets": 1,
        "grid_size": 1, "image_tokens": TARGET_LATENT_TOKEN_COUNT,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "scheduler_enabled": scheduler is not None,
        "gradient_bytes": gradient_bytes, "adamw_moment_bytes": optimizer_state_bytes,
    }


def allocate_buffer_reserve(config, *, device, sequence_length: int):
    """Keep a conservative full optimizer batch of frozen outputs resident.

    FIFO occupancy immediately before preparation is below 48 targets. Reserve
    48 targets and 16 full source states anyway, independently of candidate size.
    Also account for the previous optimizer_inputs tuple retained by the loop.
    These tensors reserve storage only; they are never passed to a model.
    """

    target_slots = config.loop.optimizer_batch_size
    source_slots = math.ceil(target_slots / TARGETS_PER_SOURCE)
    hidden_dtype = getattr(torch, config.models.nvarc_dtype)
    latent_dtype = getattr(torch, config.models.vae_dtype)
    reserve = [
        torch.zeros((sequence_length, NVARC_CONTEXT_WIDTH), device=device, dtype=hidden_dtype)
        for _ in range(source_slots)
    ]
    hidden_bytes = sum(map(_tensor_bytes, reserve))
    reserve.append(torch.zeros(
        (2 * target_slots, TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
        device=device, dtype=latent_dtype,
    ))
    return reserve, {
        "policy": "full_optimizer_batch_plus_previous_optimizer_latents",
        "fifo_source_slots": source_slots, "fifo_target_slots": target_slots,
        "previous_optimizer_target_slots": target_slots,
        "source_length": sequence_length,
        "hidden_dtype": str(hidden_dtype), "latent_dtype": str(latent_dtype),
        "hidden_bytes": hidden_bytes, "latent_bytes": _tensor_bytes(reserve[-1]),
        "total_bytes": sum(map(_tensor_bytes, reserve)),
    }


def _memory(device) -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _write_json(path: Path, value) -> None:
    # Preserve the last complete record if a worker is killed during an update.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def _environment_metadata() -> dict:
    return {
        "packages": {name: version(name) for name in (
            "torch", "transformers", "diffusers", "accelerate", "einops", "safetensors",
        )},
        "python": platform.python_version(), "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "activation_checkpointing": False, "distributed_strategy": "single_process",
    }


def run_candidate(args) -> dict:
    """Run one candidate in its own process; setup OOM is not a capacity bound."""

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "result.json"
    result = {
        "status": "running", "phase": "setup", "batch_size": args.candidate_batch_size,
        "sequence_length": args.sequence_length, "required_repetitions": args.repetitions,
        "repetitions": [], "command": sys.argv,
    }
    device = None
    _write_json(result_path, result)
    try:
        config = _load_config(args.config)
        result["config"] = asdict(config)
        result["environment"] = _environment_metadata()
        device = prepare_single_gpu_device(config)
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name, "total_memory_bytes": properties.total_memory,
            "device": str(device), "visible_count": torch.cuda.device_count(),
            "compute_capability": [properties.major, properties.minor],
        }
        result["checkpoint_configs"] = {
            str(path): json.loads(path.read_text())
            for path in (
                config.models.nvarc_checkpoint / "config.json",
                config.models.flux_checkpoint / "transformer/config.json",
                config.models.flux_checkpoint / "vae/config.json",
            ) if path.is_file()
        }
        _write_json(result_path, result)
        seed_single_gpu_process(config.seed)
        models = load_single_gpu_models(config, device=device)
        optimizer = build_flux_optimizer(config, transformer=models.transformer)
        result["initialization"] = initialize_training_memory(
            config, models, optimizer, device=device,
        )
        reserve, result["buffer_reserve"] = allocate_buffer_reserve(
            config, device=device, sequence_length=args.sequence_length,
        )
        input_ids, offsets = make_probe_inputs(
            models.tokenizer, seed=config.seed, batch_size=args.candidate_batch_size,
            sequence_length=args.sequence_length,
        )
        result["input_ids_sha256"] = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()
        result["workload"] = "seeded_synthetic_five_pair_30x30_repeated_source"
        result["dataset_split"] = None
        torch.cuda.synchronize(device)
        result["setup_memory"] = _memory(device)
        result["phase"] = "probe"
        _write_json(result_path, result)
        expected_dtype = getattr(torch, config.models.nvarc_dtype)
        for repetition in range(args.repetitions):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = models.nvarc_runner(input_ids, offsets)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            memory = _memory(device)
            if len(output.sequence_hidden_states) != args.candidate_batch_size:
                raise RuntimeError("NVARC returned the wrong number of source states")
            for state in output.sequence_hidden_states:
                if (
                    state.shape != (args.sequence_length, NVARC_CONTEXT_WIDTH)
                    or state.dtype != expected_dtype or state.device != device
                    or state.requires_grad or state.grad_fn is not None
                ):
                    raise RuntimeError("NVARC output violates its frozen source-state contract")
            del state
            output_bytes = sum(map(_tensor_bytes, output.sequence_hidden_states))
            del output  # Do not overlap the previous output with the next call.
            result["repetitions"].append({
                "repetition": repetition + 1, "seconds": seconds,
                "output_bytes": output_bytes, **memory,
            })
            _write_json(result_path, result)
        # Keep explicit owners alive through every measured call.
        result["retained_reserve_bytes"] = sum(map(_tensor_bytes, reserve))
        result["status"] = "success"
        result["peak_allocated_bytes"] = max(
            row["peak_allocated_bytes"] for row in result["repetitions"]
        )
        result["peak_reserved_bytes"] = max(
            row["peak_reserved_bytes"] for row in result["repetitions"]
        )
        result["allocated_headroom_bytes"] = (
            properties.total_memory - result["peak_allocated_bytes"]
        )
    except Exception as error:
        is_probe_oom = isinstance(error, torch.cuda.OutOfMemoryError) and result["phase"] == "probe"
        result.update(
            status="oom" if is_probe_oom else "error",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        if device is not None:
            try:
                result["failure_memory"] = _memory(device)
            except Exception as memory_error:
                result["failure_memory_error"] = f"{type(memory_error).__name__}: {memory_error}"
    finally:
        _write_json(result_path, result)
    return result


def search_capacity(evaluate: Callable[[int], dict], *, maximum: int) -> dict:
    """Double, then bisect an observed success/OOM bracket; abort other errors."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    candidates = []
    lower, upper, candidate = 0, None, 1
    while True:
        result = evaluate(candidate)
        candidates.append(result)
        status = result["status"]
        if status == "success":
            lower = candidate
        elif status == "oom":
            upper = candidate
        else:
            return {
                "status": "error", "largest_tested_success": lower or None,
                "smallest_tested_oom": upper, "boundary_known": False,
                "candidates": candidates,
            }
        if upper is not None and upper - lower == 1:
            break
        if upper is None:
            if lower == maximum:
                break
            candidate = min(2 * lower, maximum)
        else:
            candidate = (lower + upper) // 2
    return {
        "status": "complete", "largest_tested_success": lower or None,
        "smallest_tested_oom": upper, "boundary_known": upper is not None,
        "search_limit_reached": upper is None,
        "boundary_assumption": "feasibility is monotone within the searched bracket",
        "full_pipeline_capacity_validated": False, "candidates": candidates,
    }


def _source_hashes() -> dict:
    sources = [
        Path(__file__),
        REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_batch_capacity_modal.py",
        *sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
    ]
    return {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def _source_metadata() -> dict:
    def git(*arguments):
        completed = subprocess.run(
            ["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "source_sha256": _source_hashes(),
    }


def run_search(
    args, *, metadata: dict | None = None, on_candidate_end: Callable[[], None] | None = None,
) -> dict:
    """Launch sequential workers, preserving each candidate's record and stderr."""

    config = _load_config(args.config)
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    provenance = None
    if metadata is not None:
        # A remote container has the uploaded source, not the caller's Git checkout.
        provenance = {key: metadata[key] for key in ("git_commit", "git_status", "source_sha256")}
        if provenance["source_sha256"] != _source_hashes():
            raise ValueError("uploaded source differs from the launch metadata")
        if metadata["config_sha256"] != config_sha256:
            raise ValueError("uploaded config differs from the launch metadata")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if provenance is None:
        provenance = _source_metadata()
    manifest = {
        "status": "running", "command": sys.argv, "config": asdict(config),
        "config_source": json.loads(args.config.read_text()),
        "config_sha256": config_sha256,
        "sequence_length": args.sequence_length, "max_batch_size": args.max_batch_size,
        "repetitions": args.repetitions, **provenance,
        "launch_metadata": metadata,
        "checkpoint_weights_hashed": False,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    results = []

    def evaluate(batch_size):
        destination = args.output_dir / f"batch-{batch_size:04d}"
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()),
            "--config", str(args.config.resolve()),
            "--sequence-length", str(args.sequence_length),
            "--repetitions", str(args.repetitions),
            "--candidate-batch-size", str(batch_size),
            "--output-dir", str(destination.resolve()),
        ]
        print(f"Probing {batch_size} sources x {args.sequence_length} tokens", flush=True)
        with (args.output_dir / f"batch-{batch_size:04d}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        result_path = destination / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        expected_exit = {"success": 0, "oom": OOM_EXIT_CODE}.get(result.get("status"))
        if (
            expected_exit is None or completed.returncode != expected_exit
            or result.get("batch_size") != batch_size
            or (result.get("status") == "success" and len(result.get("repetitions", [])) != args.repetitions)
        ):
            result = {
                **result, "status": "error", "batch_size": batch_size,
                "worker_error": f"invalid/incomplete worker result; exit code {completed.returncode}",
            }
        result["result_path"] = str(result_path)
        results.append(result)
        _write_json(args.output_dir / "candidates.json", results)
        if on_candidate_end is not None:
            on_candidate_end()
        return result

    try:
        summary = search_capacity(evaluate, maximum=args.max_batch_size)
        manifest["status"] = summary["status"]
        _write_json(args.output_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        manifest.update(status="error", error=f"{type(error).__name__}: {error}")
        _write_json(args.output_dir / "summary.json", {
            "status": "error", "boundary_known": False, "candidates": results,
            "error": manifest["error"],
        })
        raise
    finally:
        _write_json(args.output_dir / "manifest.json", manifest)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = REPOSITORY_ROOT / "artifacts/nvarc-capacity" / run_id
    if args.candidate_batch_size is not None:
        result = run_candidate(args)
        return {"success": 0, "oom": OOM_EXIT_CODE, "error": 1}[result["status"]]
    summary = run_search(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2))
    print(f"Capacity probe results: {args.output_dir}")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
