from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diffusion_arc.arc_data import generate_augmented_key_files, load_arc_split_config


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def task_payload() -> dict[str, object]:
    examples = [
        {"input": [[0]], "output": [[1]]},
        {"input": [[2]], "output": [[3]]},
    ]
    return {
        "train": examples,
        "test": [{"input": [[4]], "output": [[5]]}],
    }


def split_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator_version": 1,
        "source": {"dataset": "arc-agi-2", "split": "training"},
        "palette": "greedy_rgb_maximin_v1",
        "limits": {"max_pairs": 3, "max_cells": 6, "max_tokens": 8192},
        "task_splits": {"train": "train.json", "validation": "validation.json"},
        "views": {
            "train": {
                "task_split": "train",
                "output": "augmented_keys/train.json",
                "mode": "sampled",
                "seed": 42,
                "augmentations_per_task": 4,
                "exclude": ["train_original"],
            },
            "train_original": {
                "task_split": "train",
                "output": "augmented_keys/train_original.json",
                "mode": "identity",
                "exclude": [],
            },
            "train_unseen": {
                "task_split": "train",
                "output": "augmented_keys/train_unseen.json",
                "mode": "sampled",
                "seed": 43,
                "augmentations_per_task": 3,
                "exclude": ["train", "train_original"],
            },
        },
    }


class ArcSplitGenerationTest(unittest.TestCase):
    def test_generates_reproducible_disjoint_training_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root = root / "data/splits/overfitting_run_1"
            source_root = root / "data/raw/arc-agi-2/training"
            write_json(source_root / "train-a.json", task_payload())
            write_json(source_root / "validation-a.json", task_payload())
            write_json(split_root / "config.json", split_config())
            write_json(split_root / "train.json", ["train-a"])
            write_json(split_root / "validation.json", ["validation-a"])

            generated = generate_augmented_key_files(split_root)
            repeated = generate_augmented_key_files(split_root)

            self.assertEqual(generated, repeated)
            self.assertEqual(len(generated["train"]), 4)
            self.assertEqual(
                generated["train_original"],
                ("train-a_ID_0123456789_0,1,2",),
            )
            self.assertEqual(len(generated["train_unseen"]), 3)
            self.assertTrue(
                set(generated["train"]).isdisjoint(generated["train_original"])
            )
            self.assertTrue(
                set(generated["train_unseen"]).isdisjoint(
                    set(generated["train"]) | set(generated["train_original"])
                )
            )
            self.assertEqual(
                json.loads(
                    (split_root / "augmented_keys/train_unseen.json").read_text(
                        encoding="utf-8"
                    )
                ),
                list(generated["train_unseen"]),
            )

    def test_rejects_train_validation_task_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root = root / "data/splits/overfitting_run_1"
            source_root = root / "data/raw/arc-agi-2/training"
            write_json(source_root / "shared-task.json", task_payload())
            write_json(split_root / "config.json", split_config())
            write_json(split_root / "train.json", ["shared-task"])
            write_json(split_root / "validation.json", ["shared-task"])

            with self.assertRaisesRegex(ValueError, "overlap"):
                generate_augmented_key_files(split_root)

    def test_rejects_split_directory_outside_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root = Path(temporary_directory) / "data/configurations/run"
            write_json(split_root / "config.json", split_config())

            with self.assertRaisesRegex(ValueError, "data_directory"):
                generate_augmented_key_files(split_root)

    def test_rejects_booleans_as_config_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"

            for field in ("schema_version", "generator_version"):
                payload = split_config()
                payload[field] = True
                write_json(config_path, payload)

                with self.subTest(field=field), self.assertRaises(TypeError):
                    load_arc_split_config(config_path)

    def test_rejects_the_retired_standard_rgb_palette(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            payload = split_config()
            payload["palette"] = "arc_standard_v1"
            write_json(config_path, payload)

            with self.assertRaisesRegex(ValueError, "unsupported palette"):
                load_arc_split_config(config_path)


if __name__ == "__main__":
    unittest.main()
