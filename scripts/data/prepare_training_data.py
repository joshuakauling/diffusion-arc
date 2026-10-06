#!/usr/bin/env python3
"""Prepare the complete, reproducible NVARC epoch described by a run recipe."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from random import Random
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.data.prepare_nvarc_benchmark import _write_json, prepare_nvarc_bundle, sha256_file

if TYPE_CHECKING:
    from diffusion_arc.training.config import TrainingConfig


def prepare_training_data(config: TrainingConfig, *, config_path: Path) -> dict:
    """Use all indexed tasks once, reading source_archive as a tar file or extracted tree."""

    if config.recipe["data"]["source"] != "epoch_pool":
        raise ValueError("data preparation requires data.source=epoch_pool")
    split = config.data.split_directory
    if split.parent.name != "splits":
        raise ValueError("prepared_split_directory must match <bundle>/splits/<split_name>")
    data = config.recipe["data"]
    source_paths = {
        "source_archive": data["source_archive"],
        "task_index": data["task_index"],
        "official_training_keys": data["official_training_keys"],
    }
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
        preparation_metadata=_preparation_metadata(config, config_path),
    )


def _preparation_metadata(config: TrainingConfig, config_path: Path) -> dict:
    code_paths = (
        Path(__file__),
        REPOSITORY_ROOT / "src/diffusion_arc/arc_data/augmentation.py",
        REPOSITORY_ROOT / "src/diffusion_arc/arc_data/splits/views.py",
        REPOSITORY_ROOT / "src/diffusion_arc/training/config.py",
    )
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_paths": {
            name: str(config.recipe["data"][name].resolve())
            for name in ("source_archive", "task_index", "official_training_keys")
        },
        "code_sha256": {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in code_paths},
        "optimizer_batch_size": config.loop.optimizer_batch_size,
        "optimizer_step_count": config.epoch_optimizer_steps,
        "final_batch_target_count": ((config.recipe["data"]["expected_task_count"] * config.targets_per_sequence - 1)
                                     % config.loop.optimizer_batch_size + 1),
        "drop_last": False,
        "dataloader_shuffle": config.data.shuffle,
    }


def prepare_next_epoch_data(parent_config: TrainingConfig, config: TrainingConfig) -> dict:
    """Resample an already prepared full epoch from metadata, sharing its raw tasks."""
    from diffusion_arc.arc_data import (
        ARC_AUGMENTATION_GENERATOR_VERSION, ArcAugmentationSpec, ArcGeometry,
        ArcTrainingViewMode, load_arc_split_config, make_augmentation_key,
    )
    from diffusion_arc.arc_data.splits.views import _candidate_seed

    if any(item.recipe["data"]["source"] != "epoch_pool" for item in (parent_config, config)):
        raise ValueError("epoch continuation requires data.source=epoch_pool")
    parent_split = parent_config.data.split_directory
    parent_training_data_manifest_path = parent_split / "manifest.json"
    parent_training_data_manifest = json.loads(parent_training_data_manifest_path.read_text())
    split_config = load_arc_split_config(parent_split / "config.json")
    parent_view = next(view for view in split_config.views if view.name == parent_config.data.view_name)
    records = sorted(parent_training_data_manifest["sources"], key=lambda record: record["task_key"])
    count = config.recipe["data"]["expected_task_count"]
    if (parent_training_data_manifest["purpose"] != "training_epoch"
            or parent_training_data_manifest["source_count"] != count or parent_training_data_manifest["eligible_task_count"] != count
            or len(records) != count or len({record["task_key"] for record in records}) != count
            or parent_training_data_manifest["augmentation"]["generator_version"] != ARC_AUGMENTATION_GENERATOR_VERSION
            or split_config.max_pairs != 5 or split_config.max_cells != 9000
            or parent_view.mode is not ArcTrainingViewMode.SAMPLED
            or parent_view.augmentations_per_task != 1 or parent_view.exclude
            or any(record["pair_count"] < 5 for record in records)):
        raise ValueError("Reusing training data requires a prepared full-pool, five-pair epoch")
    if any(record[f"parent{parent}_arc_agi_1_split"] != "training"
           for record in records for parent in (1, 2)):
        raise ValueError("Prepared tasks must have training-only parents")
    geometry_sampling = config.recipe["data"]["geometry_sampling"]
    if geometry_sampling not in ("cycle", "uniform"):
        raise ValueError("geometry_sampling must be 'cycle' or 'uniform'")
    split = config.data.split_directory
    if split.parent.name != "splits":
        raise ValueError("prepared_split_directory must match <bundle>/splits/<split_name>")
    output_directory = split.parent.parent.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite NVARC bundle: {output_directory}")
    view = replace(parent_view, name=config.data.view_name, seed=config.seed,
                   output=Path(f"augmented_keys/{config.data.view_name}.json"),
                   geometry_sampling=geometry_sampling)
    Random(config.seed).shuffle(records)
    keys = []
    for record in records:
        rng = Random(_candidate_seed(view, record["task_key"], 0))
        indices = list(range(record["pair_count"]))
        rng.shuffle(indices)
        colors = list(range(10))
        rng.shuffle(colors)
        geometry = rng.choice(tuple(ArcGeometry)) if geometry_sampling == "uniform" else ArcGeometry.IDENTITY
        # Five pairs always fit 9000 cells, so grid contents cannot affect selection.
        augmentation = ArcAugmentationSpec(geometry, tuple(colors), tuple(indices[:5]))
        keys.append(make_augmentation_key(record["task_key"], augmentation))

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_directory.name}-", dir=output_directory.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        destination_split = bundle / "splits" / split.name
        payload = json.loads((parent_split / "config.json").read_text())
        payload["views"] = {view.name: {
            "task_split": "train", "output": str(view.output), "mode": "sampled",
            "seed": config.seed, "augmentations_per_task": 1, "exclude": [],
            **({"geometry_sampling": geometry_sampling} if geometry_sampling != "cycle" else {}),
        }}
        _write_json(destination_split / "config.json", payload)
        _write_json(destination_split / payload["task_splits"]["train"], [record["task_key"] for record in records])
        _write_json(destination_split / payload["task_splits"]["validation"], [])
        _write_json(destination_split / view.output, keys)
        (bundle / "raw").symlink_to((parent_split.parent.parent / "raw").resolve(), target_is_directory=True)
        training_data_manifest = {
            **parent_training_data_manifest, "seed": config.seed, "sources": records,
            "augmentation": {**parent_training_data_manifest["augmentation"],
                             "geometry": "uniform_D4" if geometry_sampling == "uniform" else "ID"},
            "keys_sha256": sha256_file(destination_split / view.output),
            "command": sys.argv, "python": sys.version,
            "preparation_script_sha256": sha256_file(Path(__file__)),
            "preparation": _preparation_metadata(config, config.path),
            "reused_from": {"manifest": str(parent_training_data_manifest_path.resolve()),
                            "manifest_sha256": sha256_file(parent_training_data_manifest_path)},
        }
        _write_json(destination_split / "manifest.json", training_data_manifest)
        if output_directory.exists():
            raise FileExistsError(f"NVARC bundle appeared during preparation: {output_directory}")
        bundle.rename(output_directory)
    return training_data_manifest


def main(argv: list[str] | None = None) -> None:
    from diffusion_arc.training.config import load_training_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_training_config(args.config, expected_type="training_run")
    training_data_manifest = prepare_training_data(config, config_path=args.config)
    print(f"Prepared {training_data_manifest['source_count']} sequences / {training_data_manifest['target_count']} targets")
    print(f"Key SHA-256: {training_data_manifest['keys_sha256']}")
    print(f"Training split: {config.data.split_directory}")


if __name__ == "__main__":
    main()
