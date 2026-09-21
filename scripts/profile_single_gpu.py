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
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.training.profiling import (
    NSYS_WARMUP_STEPS, NSYS_ACTIVE_STEPS, nsight_capture, training_range,
)
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.trainer import Trainer
from diffusion_arc.training.step.flow_sampling import build_flow_generator, build_flow_sampler
from diffusion_arc.training.step.loss import build_velocity_loss
from diffusion_arc.training.setup import build_optimizer, load_models, prepare_device, seed_process
from diffusion_arc.training.step.lr_schedule import build_lr_scheduler
from diffusion_arc.training.data import build_dataloader, iter_optimizer_batches


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
        with training_range(self.range_name, torch_range=True):
            return self.wrapped_module(*args, **kwargs)


def profile_from_config(config_path: str | Path, *, profiler_kind: str = "torch") -> None:
    """Construct one local process and profile the configured training loop."""

    if profiler_kind not in ("torch", "nsys"):
        raise ValueError("profiler_kind must be torch or nsys")
    config = load_training_config(config_path, expected_type="baseline")
    if profiler_kind == "nsys" and config.loop.max_optimizer_steps != NSYS_WARMUP_STEPS + NSYS_ACTIVE_STEPS:
        raise ValueError("Nsight profiling requires exactly seven optimizer updates")
    device = prepare_device(config)
    seed_process(config.seed)

    models = load_models(config, device=device)
    training_samples = build_dataloader(
        config,
        tokenizer=models.tokenizer,
    )
    optimizer = build_optimizer(
        config,
        flux_transformer=models.flux_transformer,
    )
    scheduler = build_lr_scheduler(config, optimizer=optimizer)
    flow_generator = build_flow_generator(config, device=device)
    profiled_nvarc_encoder = _ProfiledModule(
        models.nvarc_encoder,
        range_name="prepare/nvarc",
    )
    profiled_vae_encoder = _ProfiledModule(
        models.vae_encoder,
        range_name="prepare/vae",
    )

    nsight_rows = []
    nsight_summary = None
    if profiler_kind == "nsys":
        output_root = Path(os.environ.get("DIFFUSION_ARC_PROFILER_OUTPUT_ROOT",
                                         str(PROFILER_OUTPUT_ROOT / Path(config_path).stem)))
        output_root.mkdir(parents=True, exist_ok=True)
        nsight_summary = output_root / "training-summary.json"
        nsight_runtime = {
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("transformers", "diffusers", "flash_attn_3")},
            "gpu": torch.cuda.get_device_name(device),
            "seed": config.seed, "flux_compute_dtype": str(models.flux_compute_dtype),
        }
        profiler_context = nsight_capture(device)
    else:
        trace_directory = PROFILER_OUTPUT_ROOT / Path(config_path).stem
        trace_directory.mkdir(parents=True, exist_ok=True)
        trace_path = trace_directory / "trace.json"

        def export_trace(profiler: object) -> None:
            export = getattr(profiler, "export_chrome_trace", None)
            if not callable(export):
                raise TypeError("torch profiler must expose export_chrome_trace()")
            export(str(trace_path))

        profiler_context = torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            schedule=torch.profiler.schedule(
                wait=PROFILER_WAIT_STEPS, warmup=PROFILER_WARMUP_STEPS,
                active=PROFILER_ACTIVE_STEPS, repeat=1,
            ),
            on_trace_ready=export_trace,
            record_shapes=False, profile_memory=False, with_stack=False,
        )

    trainer = Trainer(
        nvarc_encoder=profiled_nvarc_encoder, vae_encoder=profiled_vae_encoder,
        flux_transformer=models.flux_transformer, optimizer=optimizer, scheduler=scheduler,
        flow_sampler=build_flow_sampler(config, generator=flow_generator),
        loss_function=build_velocity_loss(config),
        flux_compute_dtype=models.flux_compute_dtype,
        flux_packing_backend=models.flux_packing_backend,
        nvarc_max_tokens=config.nvarc_max_tokens,
        vae_microbatch_size=config.loop.vae_microbatch_size,
        flux_max_joint_tokens=config.flux_packing.max_tokens,
        flux_max_spread_tokens=config.flux_packing.max_spread_tokens,
        min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
        max_grad_norm=config.optimizer.max_grad_norm,
    )
    status = "failed"
    profiler = None
    try:
        with profiler_context as profiler:
            for sequences in iter_optimizer_batches(
                training_samples, optimizer_batch_size=config.loop.optimizer_batch_size,
                min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
                max_optimizer_steps=config.loop.max_optimizer_steps,
            ):
                result = trainer.step(sequences)
                if profiler_kind == "nsys":
                    nsight_rows.append(result)
                profiler.step()
        status = "complete"
    finally:
        if nsight_summary is not None:
            payload = {
                "status": status,
                "completed_updates": len(nsight_rows),
                "warmup_updates": NSYS_WARMUP_STEPS,
                "recorded_updates": max(0, len(nsight_rows) - NSYS_WARMUP_STEPS),
                "pytorch_memory": profiler.memory_samples if profiler is not None else [],
                **nsight_runtime,
                "updates": [{"update": i + 1,
                             "loss": result.loss,
                             "targets": result.target_count, "flux_packs": result.flux_forward_calls}
                            for i, result in enumerate(nsight_rows)],
            }
            nsight_summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    if profiler_kind == "torch":
        print(f"Profiler trace: {trace_path}")
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
