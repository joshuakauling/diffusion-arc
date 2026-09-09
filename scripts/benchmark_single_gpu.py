#!/usr/bin/env python3
"""Benchmark the existing training loop without a profiler or remote telemetry."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.arc_data import load_arc_split_config
from diffusion_arc.arc_data._split_io import resolve_inside
from diffusion_arc.arc_data._split_layout import resolve_split_layout
from diffusion_arc.training.flux_input import PreparationMetrics
from diffusion_arc.training.flux_schedule import FluxTargetBatch
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_loop import run_single_gpu_training
from diffusion_arc.training.single_gpu_setup import (
    build_flow_generator,
    build_flux_optimizer,
    build_flux_scheduler,
    build_single_sample_loader,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


def check_inputs(config_path: Path):
    """Check required files, without generating data or tokenizing the dataset."""

    config = load_single_gpu_training_config(config_path)
    if config.benchmark is None:
        raise ValueError("benchmark entrypoint requires config.benchmark")
    split_file = config.data.split_directory / "config.json"
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Benchmark split is not prepared: {split_file}. "
            "Prepare the agreed representative data separately; no overfitting fallback."
        )
    split = load_arc_split_config(split_file)
    view = next((view for view in split.views if view.name == config.data.view_name), None)
    if view is None:
        raise ValueError(f"Unknown benchmark data view: {config.data.view_name}")
    files = [split_file] + [
        resolve_inside(config.data.split_directory, relative, name="benchmark input")
        for relative in (*dict(split.task_split_files).values(), view.output)
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark input is not prepared: {path}")
    _, source_directory = resolve_split_layout(
        config.data.split_directory, dataset=split.dataset, source_split=split.source_split,
    )
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Benchmark raw-task directory is not prepared: {source_directory}")
    return config, files


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    """Pool measured steps, using linear interpolation for the p95 quantile."""

    measured = [row for row in rows if row["phase"] == "measure"]
    if not measured:
        return {"measured_steps": 0}
    times = sorted(row["seconds"] for row in measured)
    position = 0.95 * (len(times) - 1)
    lower = int(position)
    p95 = times[lower] + (times[min(lower + 1, len(times) - 1)] - times[lower]) * (position - lower)
    seconds = sum(times)
    return {
        "measured_steps": len(measured),
        "measured_seconds": seconds,
        "median_step_seconds": statistics.median(times),
        "p95_step_seconds": p95,
        "targets_per_second": sum(row["targets"] for row in measured) / seconds,
        "completed_sources_per_second": sum(row["completed_sources"] for row in measured) / seconds,
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in measured),
    }


class _StepRecorder:
    """Keep CPU-only workload metadata; time the entire consumer-first pipeline."""

    def __init__(self, config, device, optimizer, scheduler):
        self.config, self.device = config, device
        self.optimizer, self.scheduler = optimizer, scheduler
        self.rows: list[dict] = []
        self.pending_targets: deque[tuple[int, bool]] = deque()
        self.sources: list[dict] = []
        self.nvarc_tokens = 0
        self.nvarc_batches: list[dict] = []
        self.flux_batches: tuple[FluxTargetBatch, ...] | None = None

    def flux_schedule(self, batches: tuple[FluxTargetBatch, ...]) -> None:
        """Retain the actual physical plan without recomputing the scheduler."""

        self.flux_batches = batches

    def preparation(self, metrics: PreparationMetrics) -> None:
        """Charge completed preparation to the next recorded optimizer step.

        Source metadata describes work done since the previous update, while
        target metadata follows FIFO consumption and can span updates. CPU
        prefetch alone adds neither prepared sources nor pending targets here.
        """

        # Batch indices are window-local; stored indices address this step's
        # prepared_sources, which may contain several preparation windows.
        source_offset = len(self.sources)
        for source in metrics.sources:
            self.sources.append({"key": source.key, "tokens": source.prefix_ends[-1]})
            self.pending_targets.extend(
                (length, index == len(source.prefix_ends) - 1)
                for index, length in enumerate(source.prefix_ends)
            )
        self.nvarc_tokens += sum(metrics.source_lengths)
        self.nvarc_batches.append({
            "source_lengths": list(metrics.source_lengths),
            "total_tokens": sum(metrics.source_lengths),
            "source_indices": list(range(source_offset, len(self.sources))),
        })

    def start(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def step(self, loss, target_count, microbatch_count):
        # Step callback: retain the LR used, then advance the training schedule.
        learning_rate = self.optimizer.param_groups[0]["lr"]
        if self.scheduler is not None:
            self.scheduler.step()
        # Benchmark recording: finish timing and record consumed FIFO targets.
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self.started
        consumed = [self.pending_targets.popleft() for _ in range(target_count)]
        batches = self.flux_batches
        if batches is None:
            raise RuntimeError("FLUX benchmark requires the executed schedule")
        if len(batches) != microbatch_count:
            raise RuntimeError("FLUX schedule and executed microbatch count disagree")
        padded_prefixes = [0] * target_count
        for batch in batches:
            for index in batch.target_indices:
                padded_prefixes[index] = consumed[index][0] if batch.padded_length is None else batch.padded_length
        self.rows.append({
            "step": len(self.rows) + 1,
            "phase": "warmup" if len(self.rows) < self.config.benchmark.warmup_steps else "measure",
            "seconds": seconds,
            "loss": float(loss.item()),
            "learning_rate": learning_rate,
            "targets": target_count,
            "flux_microbatches": microbatch_count,
            "flux_batches": [
                {"batch_size": len(batch.target_indices), "padded_length": batch.padded_length,
                 **({"joint_tokens": batch.joint_tokens} if batch.joint_tokens is not None else {}),
                 "target_indices": list(batch.target_indices)}
                for batch in batches
            ],
            "completed_sources": sum(last for _, last in consumed),
            "prepared_sources": self.sources,
            "nvarc_unpadded_tokens": sum(source["tokens"] for source in self.sources),
            "nvarc_processed_tokens": self.nvarc_tokens,
            "nvarc_calls": len(self.nvarc_batches),
            "nvarc_batches": self.nvarc_batches,
            "target_prefix_tokens": [length for length, _ in consumed],
            "flux_padded_prefix_tokens": padded_prefixes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        })
        if target_count != self.config.loop.optimizer_batch_size:
            raise RuntimeError("Benchmark exhausted its data with an incomplete optimizer batch")
        self.sources, self.nvarc_tokens = [], 0
        self.nvarc_batches = []
        self.flux_batches = None
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()


def benchmark_once(config_path: Path, output_directory: Path) -> None:
    """Run one fresh process; retain completed steps even after a Python/CUDA error."""

    started = time.perf_counter()
    config, data_files = check_inputs(config_path)
    output_directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "failed",
        "command": sys.argv,
        "config_source": json.loads(config_path.read_text()),
        "config": asdict(config),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_files_sha256": {
            str(path.relative_to(config.data.split_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in data_files
        },
        "packages": {name: version(name) for name in ("torch", "transformers", "diffusers", "accelerate")},
        "python": platform.python_version(),
        "host": {"platform": platform.platform(), "logical_cpu_count": os.cpu_count()},
        "cuda": torch.version.cuda,
        "execution": {
            "activation_checkpointing": False,
            "distributed_strategy": "single_process",
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "checkpoint_configs": {
            str(path): json.loads(path.read_text())
            for path in (
                config.models.nvarc_checkpoint / "config.json",
                config.models.flux_checkpoint / "transformer/config.json",
                config.models.flux_checkpoint / "vae/config.json",
            ) if path.is_file()
        },
    }
    recorder = None
    _write_json(output_directory / "manifest.json", {**manifest, "status": "running"})
    try:
        device = prepare_single_gpu_device(config)
        manifest["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "visible_count": torch.cuda.device_count(),
        }
        seed_single_gpu_process(config.seed)
        models = load_single_gpu_models(config, device=device)
        if config.flux_packing is not None:
            manifest["packages"]["flash-attn-3"] = version("flash-attn-3")
            manifest["execution"].update(flux_backend=models.flux_packing_backend.binding,
                modulation_reduction="fp32", flux_packed_token_budget=config.flux_packing.max_tokens)
        samples = build_single_sample_loader(config, tokenizer=models.tokenizer)
        optimizer = build_flux_optimizer(config, transformer=models.transformer)
        scheduler = build_flux_scheduler(config, optimizer=optimizer)
        generator = build_flow_generator(config, device=device)
        recorder = _StepRecorder(config, device, optimizer, scheduler)
        recorder.start()
        run_single_gpu_training(
            samples,
            nvarc_runner=models.nvarc_runner,
            vae_encoder=models.vae_encoder,
            transformer=models.transformer,
            optimizer=optimizer,
            flux_compute_dtype=models.flux_compute_dtype,
            **asdict(config.loop),
            flux_bucket_boundaries=config.padding.flux_bucket_boundaries,
            flux_scheduling=config.flux_scheduling,
            flux_packing=config.flux_packing,
            **({"flux_packing_backend": models.flux_packing_backend} if config.flux_packing is not None else {}),
            on_preparation=recorder.preparation,
            on_flux_schedule=recorder.flux_schedule,
            generator=generator,
            on_optimizer_step=recorder.step,
        )
        if len(recorder.rows) != config.loop.max_optimizer_steps:
            raise RuntimeError(
                f"Benchmark requires {config.loop.max_optimizer_steps} full steps; "
                f"got {len(recorder.rows)}"
            )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        rows = recorder.rows if recorder else []
        manifest["process_seconds"] = time.perf_counter() - started
        with (output_directory / "steps.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        _write_json(output_directory / "summary.json", {"status": manifest["status"], **summarize(rows)})
        _write_json(output_directory / "manifest.json", manifest)


def run_repetitions(config_path: Path, output_directory: Path, *, metadata=None, on_repetition_end=None):
    """Launch sequential fresh Python processes, never three concurrent GPU jobs."""

    config, _ = check_inputs(config_path)
    output_directory.mkdir(parents=True, exist_ok=False)
    launch = {
        "status": "running", "command": sys.argv,
        "config": asdict(config), "metadata": metadata or {},
    }
    _write_json(output_directory / "manifest.json", launch)
    all_rows = []
    started = time.perf_counter()
    try:
        for repetition in range(1, config.benchmark.repetitions + 1):
            destination = output_directory / f"repetition-{repetition}"
            print(f"Benchmark repetition {repetition}/{config.benchmark.repetitions}", flush=True)
            try:
                subprocess.run([
                    sys.executable, "-B", str(Path(__file__).resolve()),
                    "--config", str(config_path.resolve()),
                    "--output-dir", str(destination.resolve()), "--single-repetition",
                ], check=True)
            finally:
                if on_repetition_end is not None:
                    on_repetition_end()
            lines = (destination / "steps.jsonl").read_text().splitlines()
            all_rows.extend(json.loads(line) for line in lines)
        launch["status"] = "complete"
    except BaseException as error:
        launch.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        launch["wall_seconds"] = time.perf_counter() - started
        _write_json(output_directory / "summary.json", {"status": launch["status"], **summarize(all_rows)})
        _write_json(output_directory / "manifest.json", launch)


def source_revision() -> dict:
    """Record revision and dirty state; cloud callers pass this before uploading."""

    def git(*args):
        result = subprocess.run(["git", *args], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--single-repetition", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or REPOSITORY_ROOT / "artifacts/benchmarks" / run_id
    if args.single_repetition:
        benchmark_once(args.config, output)
    else:
        run_repetitions(args.config, output, metadata=source_revision())
    print(f"Benchmark results: {output}")


if __name__ == "__main__":
    main()
