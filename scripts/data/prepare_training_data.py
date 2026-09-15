#!/usr/bin/env python3
"""Prepare the complete, reproducible NVARC epoch described by a run recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.data.prepare_nvarc_benchmark import prepare_nvarc_bundle, sha256_file

if TYPE_CHECKING:
    from diffusion_arc.training.config import TrainingConfig


def prepare_training_data(config: TrainingConfig, *, config_path: Path) -> dict:
    """Use all indexed tasks once; data preparation needs no model or GPU."""

    split = config.data.split_directory
    if split.parent.name != "splits":
        raise ValueError("prepared_split_directory must match <bundle>/splits/<split_name>")
    data = config.recipe["data"]
    loop = config.recipe["loop"]
    source_paths = {
        "source_archive": data["source_archive"],
        "task_index": data["task_index"],
        "official_training_keys": data["official_training_keys"],
    }
    code_paths = (
        Path(__file__),
        REPOSITORY_ROOT / "src/diffusion_arc/arc_data/augmentation.py",
        REPOSITORY_ROOT / "src/diffusion_arc/arc_data/splits/views.py",
        REPOSITORY_ROOT / "src/diffusion_arc/training/config.py",
    )
    return prepare_nvarc_bundle(
        archive_path=source_paths["source_archive"],
        index_path=source_paths["task_index"],
        official_keys_path=source_paths["official_training_keys"],
        output_directory=split.parent.parent,
        source_count=data["expected_task_count"],
        seed=config.seed,
        geometry_sampling=data["geometry_sampling"],
        require_full_pool=True,
        view_name=config.data.view_name,
        split_name=split.name,
        preparation_metadata={
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_paths": {name: str(path.resolve()) for name, path in source_paths.items()},
            "code_sha256": {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in code_paths},
            "optimizer_batch_size": config.loop.optimizer_batch_size,
            "optimizer_step_count": config.loop.max_optimizer_steps,
            "final_batch_target_count": loop["expected_final_batch_target_count"],
            "drop_last": False,
            "dataloader_shuffle": config.data.shuffle,
        },
    )


def main(argv: list[str] | None = None) -> None:
    from diffusion_arc.training.config import load_training_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_training_config(args.config, expected_type="training_run")
    manifest = prepare_training_data(config, config_path=args.config)
    print(f"Prepared {manifest['source_count']} sequences / {manifest['target_count']} targets")
    print(f"Key SHA-256: {manifest['keys_sha256']}")
    print(f"Training split: {config.data.split_directory}")


if __name__ == "__main__":
    main()
