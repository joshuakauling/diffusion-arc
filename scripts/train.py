#!/usr/bin/env python3
"""Train or resume a configured single-GPU, DDP, or FSDP2 epoch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.training.config import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="path to a training_run JSON config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config, expected_type="training_run")

    # Load the GPU runtime only after configuration validation succeeds.
    from diffusion_arc.training.runner import run_training

    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
