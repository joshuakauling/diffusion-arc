#!/usr/bin/env python3
"""Run the initial offline single-GPU Diffusion-ARC training policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.training.config import (  # noqa: E402
    load_single_gpu_training_config,
)
from diffusion_arc.training.single_gpu import run_single_gpu_training  # noqa: E402
from diffusion_arc.training.single_gpu_setup import (  # noqa: E402
    build_flow_generator,
    build_flux_optimizer,
    build_single_sample_loader,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


def train_from_config(config_path: str | Path) -> None:
    """Construct one local process and execute the configured training loop."""

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
    flow_generator = build_flow_generator(config, device=device)

    run_single_gpu_training(
        training_samples,
        nvarc_runner=models.nvarc_runner,
        vae_encoder=models.vae_encoder,
        transformer=models.transformer,
        optimizer=optimizer,
        flux_compute_dtype=models.flux_compute_dtype,
        max_optimizer_steps=config.loop.max_optimizer_steps,
        optimizer_batch_size=config.loop.optimizer_batch_size,
        flux_microbatch_size=config.loop.flux_microbatch_size,
        min_context_pairs_before_target=(
            config.loop.min_context_pairs_before_target
        ),
        generator=flow_generator,
    )


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
