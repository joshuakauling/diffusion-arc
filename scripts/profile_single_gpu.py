#!/usr/bin/env python3
"""Profile the single-GPU B3 baseline with Torch or Nsight."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.training.profiling import (
    NSYS_WARMUP_STEPS, NSYS_ACTIVE_STEPS, nsight_capture,
)
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.benchmark_runner import run_benchmark
from diffusion_arc.training.run.experiment import BenchmarkInstrumentation
from diffusion_arc.training.profiling import torch_training_ranges


PROFILER_WAIT_STEPS = 2
PROFILER_WARMUP_STEPS = 2
PROFILER_ACTIVE_STEPS = 3
PROFILER_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/profiling"


class _ProfileExperiment(BenchmarkInstrumentation):
    """Attach collectors to the common update loop without per-step synchronization."""

    def __init__(self, config, config_path, kind):
        super().__init__(max_steps=config.loop.max_optimizer_steps, emit=lambda record: None)
        self.kind, self.config_path = kind, Path(config_path)
        self.rows, self.summary_path, self.collector = [], None, None
        self.runtime_metadata = {}

    def start(self, stack, config, models, dataset):
        device = self.runtime.device
        if self.kind == "nsys":
            output = Path(os.environ.get("DIFFUSION_ARC_PROFILER_OUTPUT_ROOT",
                                        str(PROFILER_OUTPUT_ROOT / self.config_path.stem)))
            output.mkdir(parents=True, exist_ok=True)
            self.summary_path = output / "training-summary.json"
            self.runtime_metadata = {
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "packages": {name: version(name) for name in ("transformers", "diffusers", "flash_attn_3")},
                "gpu": torch.cuda.get_device_name(device),
                "seed": config.seed, "flux_compute_dtype": str(models.flux_compute_dtype),
            }
            self.collector = stack.enter_context(nsight_capture(device))
        else:
            directory = PROFILER_OUTPUT_ROOT / self.config_path.stem
            directory.mkdir(parents=True, exist_ok=True)
            self.trace_path = directory / "trace.json"
            stack.enter_context(torch_training_ranges())
            self.collector = stack.enter_context(torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                schedule=torch.profiler.schedule(wait=PROFILER_WAIT_STEPS,
                    warmup=PROFILER_WARMUP_STEPS, active=PROFILER_ACTIVE_STEPS, repeat=1),
                on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(self.trace_path)),
                record_shapes=False, profile_memory=False, with_stack=False,
            ))

    def before_step(self):
        pass

    def completed_step(self):
        pass

    def after_step(self, run_step, batch, result, progress):
        self.rows.append(result)
        self.collector.step()

    def finish(self, progress):
        pass

    def write_summary(self, status):
        if self.summary_path is not None:
            payload = {
                "status": status, "completed_updates": len(self.rows),
                "warmup_updates": NSYS_WARMUP_STEPS,
                "recorded_updates": max(0, len(self.rows) - NSYS_WARMUP_STEPS),
                "pytorch_memory": self.collector.memory_samples if self.collector is not None else [],
                **self.runtime_metadata,
                "updates": [{"update": i + 1, "loss": result.loss,
                             "targets": result.target_count, "flux_packs": result.flux_forward_calls}
                            for i, result in enumerate(self.rows)],
            }
            self.summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def profile_from_config(config_path: str | Path, *, profiler_kind: str = "torch") -> None:
    """Profile the same model setup, batch preparation and updates as training."""
    if profiler_kind not in ("torch", "nsys"):
        raise ValueError("profiler_kind must be torch or nsys")
    config = load_training_config(config_path, expected_type="training_run")
    if config.recipe["execution"]["distributed_strategy"] != "single_process":
        raise ValueError("single-GPU profiling requires distributed_strategy=single_process")
    required = (NSYS_WARMUP_STEPS + NSYS_ACTIVE_STEPS if profiler_kind == "nsys" else
                PROFILER_WAIT_STEPS + PROFILER_WARMUP_STEPS + PROFILER_ACTIVE_STEPS)
    if config.loop.max_optimizer_steps != required:
        raise ValueError(f"{profiler_kind} profiling requires exactly {required} optimizer updates")
    experiment = _ProfileExperiment(config, config_path, profiler_kind)
    status = "failed"
    try:
        run_benchmark(config, experiment=experiment)
        status = "complete"
    finally:
        experiment.write_summary(status)
    if profiler_kind == "torch":
        print(f"Profiler trace: {experiment.trace_path}")
    else:
        print("Nsight capture complete: four warmup and three recorded optimizer updates")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="path to one strict single-GPU training JSON config",
    )
    parser.add_argument("--profiler", choices=("torch", "nsys"), default="torch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profiler == "torch":
        profile_from_config(args.config)
    else:
        profile_from_config(args.config, profiler_kind=args.profiler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
