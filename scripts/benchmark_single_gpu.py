#!/usr/bin/env python3
"""Benchmark the existing training loop without a profiler or remote telemetry."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import islice
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
        self.padded_tokens = 0

    def samples(self, samples):
        iterator = iter(samples)
        while batch := tuple(islice(iterator, self.config.loop.nvarc_source_batch_size)):
            lengths = []
            for sample in batch:
                prefixes = sample.prefix_token_ends[
                    self.config.loop.min_context_pairs_before_target - 1 :
                ]
                if prefixes:
                    lengths.append(prefixes[-1])
                    self.sources.append({"key": str(sample.key), "tokens": prefixes[-1]})
                    self.pending_targets.extend(
                        (length, index == len(prefixes) - 1)
                        for index, length in enumerate(prefixes)
                    )
            self.padded_tokens += max(lengths, default=0) * len(lengths)
            yield from batch

    def start(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    def step(self, loss, target_count, microbatch_count):
        learning_rate = self.optimizer.param_groups[0]["lr"]
        if self.scheduler is not None:
            self.scheduler.step()
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - self.started
        consumed = [self.pending_targets.popleft() for _ in range(target_count)]
        self.rows.append({
            "step": len(self.rows) + 1,
            "phase": "warmup" if len(self.rows) < self.config.benchmark.warmup_steps else "measure",
            "seconds": seconds,
            "loss": float(loss.item()),
            "learning_rate": learning_rate,
            "targets": target_count,
            "flux_microbatches": microbatch_count,
            "completed_sources": sum(last for _, last in consumed),
            "prepared_sources": self.sources,
            "nvarc_unpadded_tokens": sum(source["tokens"] for source in self.sources),
            "nvarc_padded_tokens": self.padded_tokens,
            "target_prefix_tokens": [length for length, _ in consumed],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        })
        if target_count != self.config.loop.optimizer_batch_size:
            raise RuntimeError("Benchmark exhausted its data with an incomplete optimizer batch")
        self.sources, self.padded_tokens = [], 0
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
        samples = build_single_sample_loader(config, tokenizer=models.tokenizer)
        optimizer = build_flux_optimizer(config, transformer=models.transformer)
        scheduler = build_flux_scheduler(config, optimizer=optimizer)
        generator = build_flow_generator(config, device=device)
        recorder = _StepRecorder(config, device, optimizer, scheduler)
        recorder.start()
        run_single_gpu_training(
            recorder.samples(samples),
            nvarc_runner=models.nvarc_runner,
            vae_encoder=models.vae_encoder,
            transformer=models.transformer,
            optimizer=optimizer,
            flux_compute_dtype=models.flux_compute_dtype,
            **asdict(config.loop),
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
