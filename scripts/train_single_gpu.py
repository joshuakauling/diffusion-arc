#!/usr/bin/env python3
"""Run the initial offline single-GPU Diffusion-ARC training policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.training.single_gpu_config import (  # noqa: E402
    load_single_gpu_training_config,
)
from diffusion_arc.training.single_gpu_loop import (  # noqa: E402
    run_single_gpu_training,
)
from diffusion_arc.training.single_gpu_setup import (  # noqa: E402
    build_flow_generator,
    build_flux_optimizer,
    build_flux_scheduler,
    build_single_sample_loader,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


PROFILER_WAIT_STEPS = 2
PROFILER_WARMUP_STEPS = 2
PROFILER_ACTIVE_STEPS = 3
PROFILER_OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/profiling"


class _ProfiledModule(nn.Module):
    """Add one profiler range around a module without changing source modules."""

    def __init__(self, module: nn.Module, *, range_name: str) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module")
        if not isinstance(range_name, str) or not range_name:
            raise ValueError("range_name must be a non-empty string")
        self.wrapped_module = module
        self.range_name = range_name

    def forward(self, *args: object, **kwargs: object) -> object:
        # Profiler range: label this model call in the trace timeline.
        with torch.profiler.record_function(self.range_name):
            return self.wrapped_module(*args, **kwargs)


def train_from_config(config_path: str | Path) -> None:
    """Construct one local process and profile the configured training loop."""

    config = load_single_gpu_training_config(config_path)
    device = prepare_single_gpu_device(config)
    seed_single_gpu_process(config.seed)

    models = load_single_gpu_models(config, device=device)
    training_samples = build_single_sample_loader(
        config,
        tokenizer=models.tokenizer,
    )
    optimizer = build_flux_optimizer(
        config,
        transformer=models.transformer,
    )
    scheduler = build_flux_scheduler(config, optimizer=optimizer)
    flow_generator = build_flow_generator(config, device=device)
    profiled_nvarc_runner = _ProfiledModule(
        models.nvarc_runner,
        range_name="prepare/nvarc",
    )
    profiled_vae_encoder = _ProfiledModule(
        models.vae_encoder,
        range_name="prepare/vae",
    )
    profiled_transformer = _ProfiledModule(
        models.transformer,
        range_name="train/flux_forward",
    )

    trace_directory = PROFILER_OUTPUT_ROOT / Path(config_path).stem
    trace_directory.mkdir(parents=True, exist_ok=True)
    trace_path = trace_directory / "trace.json"

    def export_trace(profiler: object) -> None:
        export = getattr(profiler, "export_chrome_trace", None)
        if not callable(export):
            raise TypeError("torch profiler must expose export_chrome_trace()")
        export(str(trace_path))

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        schedule=torch.profiler.schedule(
            wait=PROFILER_WAIT_STEPS,
            warmup=PROFILER_WARMUP_STEPS,
            active=PROFILER_ACTIVE_STEPS,
            repeat=1,
        ),
        # Profiler callback: export the completed capture.
        on_trace_ready=export_trace,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        # Step callback: advance the learning-rate and profiler schedules.
        def on_optimizer_step(*_args: object) -> None:
            if scheduler is not None:
                scheduler.step()
            profiler.step()

        run_single_gpu_training(
            training_samples,
            nvarc_runner=profiled_nvarc_runner,
            vae_encoder=profiled_vae_encoder,
            transformer=profiled_transformer,
            optimizer=optimizer,
            flux_compute_dtype=models.flux_compute_dtype,
            max_optimizer_steps=config.loop.max_optimizer_steps,
            optimizer_batch_size=config.loop.optimizer_batch_size,
            flux_microbatch_size=config.loop.flux_microbatch_size,
            min_context_pairs_before_target=(
                config.loop.min_context_pairs_before_target
            ),
            nvarc_source_batch_size=config.loop.nvarc_source_batch_size,
            vae_microbatch_size=config.loop.vae_microbatch_size,
            flux_bucket_boundaries=config.padding.flux_bucket_boundaries,
            nvarc_scheduling=config.nvarc_scheduling,
            flux_scheduling=config.flux_scheduling,
            generator=flow_generator,
            on_optimizer_step=on_optimizer_step,
        )

    print(f"Profiler trace: {trace_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="path to one strict single-GPU training JSON config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_from_config(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
