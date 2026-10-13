#!/usr/bin/env python3
"""Find the tested FSDP2 FLUX pack limit with synthetic conditioning and AdamW."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.model.contracts import NVARC_MAX_TOKENS, TARGET_LATENT_SHAPE
from diffusion_arc.model.flux.packing import PackedFluxForward, packing_layout
from diffusion_arc.model.flux.velocity import build_flux_image_position_ids
from diffusion_arc.training.benchmark_config import load_training_benchmark_config
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.distributed import distributed_runtime
from diffusion_arc.training.distribution_strategy import build_distribution_strategy
from diffusion_arc.training.distribution_strategy.fsdp import shard_flux
from diffusion_arc.training.setup import build_optimizer, load_models, seed_process
from diffusion_arc.training.step.loss import build_velocity_loss
from diffusion_arc.training.step.lr_schedule import build_lr_scheduler


IMAGE_TOKENS = TARGET_LATENT_SHAPE[0]
FULL_SEQUENCE_TOKENS = NVARC_MAX_TOKENS + IMAGE_TOKENS
LOCAL_TRAINING_TARGETS = 24
ACCUMULATION_PACKS = 2
OOM_EXIT_CODE = 2


def validate_options(start_tokens, step_tokens, max_tokens, updates, reshard_after_forward=True):
    for name, value in (("start_tokens", start_tokens), ("step_tokens", step_tokens),
                        ("max_tokens", max_tokens), ("updates", updates)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not FULL_SEQUENCE_TOKENS <= start_tokens <= max_tokens <= LOCAL_TRAINING_TARGETS * FULL_SEQUENCE_TOKENS:
        raise ValueError("token bounds must satisfy 9300 <= start_tokens <= max_tokens <= 223200")
    if updates < 2:
        raise ValueError("updates must be at least 2 to measure initialized AdamW state")
    if type(reshard_after_forward) is not bool:
        raise ValueError("reshard_after_forward must be a boolean")


def shard_probe_flux(flux_model, *, runtime, reshard_after_forward, blocks_per_group=1):
    """Select the diagnostic block policy while preserving root/backward sharding."""
    return shard_flux(flux_model, runtime=runtime, reshard_after_forward=reshard_after_forward,
                      blocks_per_group=blocks_per_group)


def load_probe_config(path: Path, *, require_checkpoints: bool = True):
    """Read the existing FSDP recipe without constructing a dataset or tokenizer."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("config_type") == "ddp_benchmark":
        benchmark = load_training_benchmark_config(path)
        if benchmark.fsdp_gradient_sync != "last_microbatch":
            raise ValueError("capacity probe does not support benchmark fsdp_gradient_sync=per_microbatch")
        config = benchmark.training
    else:
        config = load_training_config(path, expected_type="training_run")
    recipe = config.recipe_dict()
    gpu_count = recipe["deployment"]["gpu_count"]
    if (recipe["execution"]["distributed_strategy"] != "fsdp2"
            or gpu_count not in (2, 4, 8)
            or config.loop.optimizer_batch_size != LOCAL_TRAINING_TARGETS * gpu_count
            or config.targets_per_sequence != 3
            or recipe["models"]["initialization"] != "base_checkpoint"
            or recipe["models"]["resume_from"] is not None):
        raise ValueError("probe requires fresh 2/4/8-GPU FSDP2 with 24 targets per GPU "
                         "and three targets per source")
    if require_checkpoints:
        for checkpoint in (config.models.nvarc_checkpoint, config.models.flux_checkpoint):
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"required cached checkpoint is missing: {checkpoint}")
    return config


def probe_text_lengths(joint_tokens: int) -> tuple[int, ...]:
    """Fill the budget exactly using the fewest legal text/image segments.

    Keep leading text segments at 8,400 where possible. A new segment needs its
    own 900 image tokens, so at a segment boundary the preceding text segment
    is shortened to leave at least one text token for every remaining segment.
    """
    if type(joint_tokens) is not int or joint_tokens < FULL_SEQUENCE_TOKENS:
        raise ValueError("joint_tokens must be an integer of at least 9300")
    count = math.ceil(joint_tokens / FULL_SEQUENCE_TOKENS)
    remaining = joint_tokens - count * IMAGE_TOKENS
    lengths = []
    for index in range(count):
        length = min(NVARC_MAX_TOKENS, remaining - (count - index - 1))
        lengths.append(length)
        remaining -= length
    return tuple(lengths)


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tensor_bytes(tensor):
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return local.numel() * local.element_size()


def _memory(device):
    if device.type != "cuda":
        return {}  # Explicit CPU test injection; the production worker requires CUDA.
    statistics = torch.cuda.memory_stats(device)
    return dict(allocator_retry_count=statistics.get("num_alloc_retries", 0),
                allocator_oom_count=statistics.get("num_ooms", 0),
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def allocate_source_reserve(config, models, *, device, generator):
    """Retain a full rank-local conditioning batch and an explicit latent margin."""
    source_count = LOCAL_TRAINING_TARGETS // config.targets_per_sequence
    width = models.flux_transformer.config.joint_attention_dim
    sources = tuple(torch.randn((NVARC_MAX_TOKENS, width), device=device,
                                dtype=getattr(torch, config.models.nvarc_dtype), generator=generator)
                    for _ in range(source_count))
    latents = torch.zeros((2 * LOCAL_TRAINING_TARGETS, *TARGET_LATENT_SHAPE), device=device,
                         dtype=getattr(torch, config.models.vae_dtype))
    return sources, latents, dict(source_slots=source_count, source_length=NVARC_MAX_TOKENS,
                                 target_slots=LOCAL_TRAINING_TARGETS,
                                 additional_latent_target_slots=LOCAL_TRAINING_TARGETS,
                                 hidden_bytes=sum(_tensor_bytes(t) for t in sources),
                                 latent_bytes=_tensor_bytes(latents),
                                 policy="retain_all_source_states_plus_additional_latents")


def run_probe_updates(flux_model, optimizer, *, runtime, sources, text_lengths, updates,
                      loss_function, max_grad_norm, generator, scheduler=None,
                      on_progress=lambda *_: None, on_update=lambda *_: None):
    """Accumulate two full packs per update, including production clipping.

    This repeats one stress pack rather than reproducing distinct training
    targets. All ranks have the same shapes and number of synchronized calls.
    Source buffers remain live throughout; per-call tensors are released after
    backward as in training. Update two runs with actual AdamW moments resident.
    FSDP2 reduces accumulated gradients in the final pack before clipping.
    """
    strategy = build_distribution_strategy(flux_model, compute_dtype=flux_model.compute_dtype,
                                           backend=flux_model.backend)
    targets = len(text_lengths)
    global_targets = strategy.all_reduce_target_count(ACCUMULATION_PACKS * targets)
    rows = []
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        _synchronize(runtime.device)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        started = time.perf_counter()
        total_loss = torch.zeros((), device=runtime.device)
        for pack in range(ACCUMULATION_PACKS):
            on_progress("build_pack", update)
            text = torch.cat([sources[index % len(sources)][:length]
                              for index, length in enumerate(text_lengths)])
            layout = packing_layout(text_lengths, IMAGE_TOKENS, runtime.device)
            positions = build_flux_image_position_ids(targets, runtime.device)
            latents, target = (torch.randn((targets, *TARGET_LATENT_SHAPE), device=runtime.device,
                                          dtype=torch.float32, generator=generator) for _ in range(2))
            timesteps = torch.full((targets,), .5, device=runtime.device)
            on_progress("forward_backward", update)
            is_last = pack == ACCUMULATION_PACKS - 1
            with strategy.microbatch_context(is_last=is_last):
                prediction = flux_model(latents, text, positions, timesteps, layout)
                loss = loss_function(prediction, target, ((30, 30),) * targets).loss
                weighted_loss = loss * strategy.loss_weight(targets, global_targets)
                weighted_loss.backward()
            total_loss += weighted_loss.detach()
            del weighted_loss, loss, prediction, timesteps, latents, target, positions, layout, text
        strategy.check_step_loss(total_loss)
        on_progress("clip_optimizer", update)
        norm = torch.nn.utils.clip_grad_norm_(strategy.parameters,
                    max_norm=math.inf if max_grad_norm is None else max_grad_norm,
                    error_if_nonfinite=True, foreach=False)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        _synchronize(runtime.device)
        row = dict(update=update, loss=float(total_loss.item()), gradient_norm=float(norm.item()),
                   learning_rates=learning_rates, seconds=time.perf_counter() - started,
                   accumulation_packs=ACCUMULATION_PACKS, probe_targets_per_rank=ACCUMULATION_PACKS * targets,
                   adamw_moment_bytes=sum(_tensor_bytes(state[name]) for state in optimizer.state.values()
                                         for name in ("exp_avg", "exp_avg_sq")), **_memory(runtime.device))
        rows.append(row)
        on_update(rows)
    return rows


def run_candidate(args):
    """Write rank-local failures before distributed cleanup can terminate peers."""
    rank = int(os.environ["RANK"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"rank-{rank}.json"
    result = dict(rank=rank, status="running", phase="setup", joint_tokens=args.candidate_tokens,
                  text_lengths=probe_text_lengths(args.candidate_tokens), updates=[])
    _write_json(result_path, result)
    try:
        config = load_probe_config(args.config)
        gpu_count = config.recipe["deployment"]["gpu_count"]
        result.update(fsdp_reshard_after_forward=args.reshard_after_forward,
                      gpu_count=gpu_count, fsdp_blocks_per_group=config.fsdp_blocks_per_group,
                      fsdp_root_reshard_after_forward=False, fsdp_reshard_after_backward=True,
                      fsdp_gradient_sync="last_microbatch")
        _write_json(result_path, result)
        with distributed_runtime(config) as runtime:
            try:
                if runtime.world_size != gpu_count or runtime.device.type != "cuda":
                    raise ValueError("capacity worker requires the configured number of CUDA ranks")
                seed_process(config.seed)
                models = load_models(config, device=runtime.device)
                flux_model = shard_probe_flux(PackedFluxForward(models.flux_transformer,
                                        compute_dtype=models.flux_compute_dtype,
                                        backend=models.flux_packing_backend), runtime=runtime,
                                        reshard_after_forward=args.reshard_after_forward,
                                        blocks_per_group=config.fsdp_blocks_per_group)
                optimizer = build_optimizer(config, flux_transformer=flux_model)
                scheduler = build_lr_scheduler(config, optimizer=optimizer)
                generator = torch.Generator(device=runtime.device).manual_seed(config.seed + rank)
                sources, latent_reserve, result["buffer_reserve"] = allocate_source_reserve(
                    config, models, device=runtime.device, generator=generator)
                properties = torch.cuda.get_device_properties(runtime.device)
                result["gpu"] = dict(name=properties.name, total_memory_bytes=properties.total_memory,
                                     device=str(runtime.device), threads=torch.get_num_threads())
                result["packages"] = {name: version(name) for name in
                                      ("torch", "diffusers", "transformers", "flash-attn-3")}
                result["cuda"] = torch.version.cuda
                result["setup_memory"] = _memory(runtime.device)
                result["phase"] = "probe"

                def progress(operation, update):
                    result.update(operation=operation, current_update=update)
                    _write_json(result_path, result)

                def record(rows):
                    result["updates"] = rows
                    _write_json(result_path, result)

                run_probe_updates(flux_model, optimizer, runtime=runtime, sources=sources,
                    text_lengths=result["text_lengths"], updates=args.updates,
                    loss_function=build_velocity_loss(config), max_grad_norm=config.optimizer.max_grad_norm,
                    generator=generator, scheduler=scheduler,
                    on_progress=progress, on_update=record)
                result.update(status="success", retained_latent_reserve_bytes=_tensor_bytes(latent_reserve))
                _write_json(result_path, result)
            except Exception as error:
                result.update(status="oom" if isinstance(error, torch.cuda.OutOfMemoryError)
                              and result["phase"] == "probe" else "error",
                              error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                              failure_memory=_memory(runtime.device))
                _write_json(result_path, result)
                raise
    except Exception as error:
        if result["status"] == "running":
            result.update(status="error", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            _write_json(result_path, result)
        return OOM_EXIT_CODE if result["status"] == "oom" else 1
    return 0


def collect_candidate(directory, *, tokens, updates, returncode, timed_out=False, world_size=2):
    """Accept only complete all-rank passes or an explicitly recorded probe OOM."""
    expected_ranks = set(range(world_size))
    ranks = [json.loads(path.read_text()) for rank in range(world_size)
             if (path := directory / f"rank-{rank}.json").is_file()]
    result = dict(joint_tokens=tokens, text_lengths=probe_text_lengths(tokens),
                  ranks=ranks, worker_returncode=returncode, result_directory=str(directory))
    if any(row.get("joint_tokens") != tokens or row.get("rank") not in expected_ranks for row in ranks):
        return dict(result, status="error", error="worker identity differs from the candidate")
    independent_errors = [row for row in ranks if row.get("status") == "error"
                          and row.get("error_type") not in ("DistBackendError", "DistNetworkError")]
    if (returncode or timed_out) and not independent_errors and any(row.get("status") == "oom"
                                      and row.get("phase") == "probe" for row in ranks):
        return dict(result, status="oom")
    if any(row.get("status") in ("error", "oom") for row in ranks):
        return dict(result, status="error", error="recorded worker failure; inspect rank files/log")
    if timed_out:
        return dict(result, status="time_limit")
    if (returncode == 0 and len(ranks) == world_size and {row["rank"] for row in ranks} == expected_ranks
            and all(row.get("status") == "success" and len(row.get("updates", [])) == updates
                    and [item.get("update") for item in row["updates"]] == list(range(1, updates + 1))
                    for row in ranks)):
        return dict(result, status="success",
                    peak_allocated_bytes=max(item["peak_allocated_bytes"]
                                             for row in ranks for item in row["updates"]),
                    peak_reserved_bytes=max(item["peak_reserved_bytes"]
                                            for row in ranks for item in row["updates"]))
    return dict(result, status="error", error="incomplete or failed torchrun result; inspect rank files/log")


def search_capacity(evaluate, *, start_tokens, step_tokens, max_tokens, updates=2):
    """Walk upward to the first OOM; if the starting pack OOMs, walk downward."""
    validate_options(start_tokens, step_tokens, max_tokens, updates)
    candidate, direction, largest_pass, smallest_oom = start_tokens, 1, None, None
    results = []
    status = "complete"
    while True:
        result = evaluate(candidate)
        results.append(result)
        if result["status"] == "success":
            largest_pass = candidate
            if direction < 0 or candidate == max_tokens:
                break
            candidate = min(candidate + step_tokens, max_tokens)
        elif result["status"] == "oom":
            smallest_oom = candidate
            if largest_pass is not None or candidate == FULL_SEQUENCE_TOKENS:
                break
            direction = -1
            candidate = max(FULL_SEQUENCE_TOKENS, candidate - step_tokens)
        else:
            status = result["status"] if result["status"] == "time_limit" else "error"
            break
    return dict(status=status, largest_tested_success=largest_pass, smallest_tested_oom=smallest_oom,
                boundary_known=largest_pass is not None and smallest_oom is not None,
                search_limit_reached=largest_pass == max_tokens and smallest_oom is None,
                full_pipeline_capacity_validated=False, candidates=results)


def source_metadata():
    sources = [Path(__file__), REPOSITORY_ROOT / "scripts/cloud/probe_flux_capacity_modal.py",
               *sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
               *sorted((REPOSITORY_ROOT / "configs").rglob("*.json"))]

    def git(*args):
        process = subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None

    return dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                source_sha256={str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources})


def _run_worker(command, *, log, timeout):
    environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                   "PYTORCH_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        timed_out = True
    except BaseException:
        timed_out = False
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, timed_out


def run_search(args, *, metadata=None, on_candidate_end=None, timeout_seconds=1710):
    """Isolate every candidate in fresh torchrun processes and persist progress."""
    validate_options(args.start_tokens, args.step_tokens, args.max_tokens, args.updates,
                     args.reshard_after_forward)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    config = load_probe_config(args.config)
    gpu_count = config.recipe["deployment"]["gpu_count"]
    source = source_metadata() if metadata is None else metadata
    if metadata is not None:
        if metadata["source_sha256"] != source_metadata()["source_sha256"]:
            raise ValueError("uploaded code/configs differ from launch metadata")
        if metadata["config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError("uploaded config differs from launch metadata")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", config=config.recipe_dict(), seed=config.seed, gpu_count=gpu_count,
        start_tokens=args.start_tokens, step_tokens=args.step_tokens, max_tokens=args.max_tokens,
        updates=args.updates, timeout_seconds=timeout_seconds, launch_metadata=source,
        fsdp_reshard_after_forward=args.reshard_after_forward,
        fsdp_root_reshard_after_forward=False, fsdp_reshard_after_backward=True,
        fsdp_gradient_sync="last_microbatch",
        fsdp_blocks_per_group=config.fsdp_blocks_per_group,
        workload="synthetic_long_text_packed_flux", frozen_model_weights_resident=True,
        frozen_encoder_execution_measured=False, global_training_batch_reproduced=False,
        full_pipeline_capacity_validated=False, checkpoint_weights_hashed=False)
    _write_json(args.output_dir / "manifest.json", manifest)
    deadline = time.monotonic() + timeout_seconds
    candidates = []

    def evaluate(tokens):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return dict(joint_tokens=tokens, status="time_limit", ranks=[])
        directory = args.output_dir / f"tokens-{tokens:06d}"
        directory.mkdir()
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", f"--nproc_per_node={gpu_count}", "--max_restarts=0", str(Path(__file__).resolve()),
                   "--config", str(args.config.resolve()), "--candidate-tokens", str(tokens),
                   "--updates", str(args.updates), "--output-dir", str(directory)]
        command.append("--reshard-after-forward" if args.reshard_after_forward else "--no-reshard-after-forward")
        print(f"Probing {tokens} joint tokens: text lengths {probe_text_lengths(tokens)}", flush=True)
        with (directory / "torchrun.log").open("w") as log:
            code, timed_out = _run_worker(command, log=log, timeout=remaining)
        result = collect_candidate(directory, tokens=tokens, updates=args.updates,
                                   returncode=code, timed_out=timed_out, world_size=gpu_count)
        candidates.append(result)
        _write_json(directory / "result.json", result)
        _write_json(args.output_dir / "candidates.json", candidates)
        print(f"{tokens} tokens: {result['status']}", flush=True)
        if on_candidate_end is not None:
            on_candidate_end()
        return result

    try:
        summary = search_capacity(evaluate, start_tokens=args.start_tokens, step_tokens=args.step_tokens,
                                  max_tokens=args.max_tokens, updates=args.updates)
        manifest["status"] = summary["status"]
        _write_json(args.output_dir / "summary.json", summary)
        return summary
    except BaseException as error:
        manifest.update(status="error", error=f"{type(error).__name__}: {error}")
        _write_json(args.output_dir / "summary.json", dict(status="error", candidates=candidates,
                                                         error=manifest["error"]))
        raise
    finally:
        _write_json(args.output_dir / "manifest.json", manifest)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/training/benchmarks/fsdp2_training.json")
    parser.add_argument("--start-tokens", type=int, default=12288)
    parser.add_argument("--step-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--reshard-after-forward", action=argparse.BooleanOptionalAction, default=True,
                        help="reshard FLUX blocks after forward; disable to retain gathered weights for backward")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-tokens", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_options(args.start_tokens, args.step_tokens, args.max_tokens, args.updates,
                     args.reshard_after_forward)
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = REPOSITORY_ROOT / "artifacts/flux-capacity" / run_id
    if args.candidate_tokens is not None:
        return run_candidate(args)
    summary = run_search(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2))
    print(f"Capacity probe results: {args.output_dir}")
    return 1 if summary["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
