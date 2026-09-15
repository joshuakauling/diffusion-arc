"""Test benchmark preparation with tiny local archives, never real checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from diffusion_arc.arc_data import (
    ArcGeometry, load_training_task, parse_augmentation_key,
)
from diffusion_arc.training.data import ArcTrainingDataset
from scripts.data.prepare_nvarc_benchmark import DERIVED_GROUP, prepare_benchmark
from tests.training.test_data import make_nvarc_tokenizer, write_json


def _fixture(
    root: Path, *, pair_count: int = 30, large: bool = False, task_count: int = 12,
) -> list[dict]:
    root.mkdir()
    write_json(root / "arc_agi_1_training_keys.json", {
        "group": "arc_agi_1_training", "key_count": 400,
        "keys": [f"{index:08x}" for index in range(400)],
    })
    pairs = [{"input": [[i % 10]], "output": [[(i + 1) % 10]]} for i in range(pair_count)]
    if large:
        pairs = [{"input": [[1] * 30] * 30, "output": [[2] * 30] * 30}] * pair_count
    content = json.dumps(pairs).encode()
    records = []
    with tarfile.open(root / "nvarc-all-classified.tar.gz", "w:gz") as archive:
        for index in range(task_count):
            parent1, parent2 = f"{index:08x}", f"{index + 20:08x}"
            task_id = f"{parent1}_{parent2}"
            path = f"tasks/nvarc_training/{parent1}/{task_id}.json"
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            records.append({
                "task_key": task_id, "parent1_id": parent1, "parent2_id": parent2,
                "parent1_arc_agi_1_split": "training", "parent2_arc_agi_1_split": "training",
                "source_subset": "nvarc_training", "archive_path": path,
                "sha256": hashlib.sha256(content).hexdigest(), "pair_count": pair_count,
            })
    _write_index(root, records)
    return records


def _write_index(root: Path, records: list[dict]) -> None:
    (root / f"{DERIVED_GROUP}_tasks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


class NvarcBenchmarkPreparationTest(unittest.TestCase):
    def test_reproducible_keys_order_and_dataset_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            records = _fixture(source)
            manifests = []
            for name in ("first", "repeat", "reordered"):
                if name == "reordered":
                    _write_index(source, list(reversed(records)))
                manifests.append(prepare_benchmark(
                    source_root=source, output_directory=root / name, source_count=6,
                ))
            self.assertEqual(len({manifest["keys_sha256"] for manifest in manifests}), 1)
            manifest = manifests[0]
            self.assertEqual(manifest["source_count"], 6)
            self.assertEqual(manifest["target_count"], 18)
            self.assertEqual(manifest["eligible_task_count"], 12)
            split = root / "first/splits/train"
            serialized = json.loads((split / "augmented_keys/train.json").read_text())
            keys = [parse_augmentation_key(key) for key in serialized]
            self.assertEqual(len({key.task_id for key in keys}), 6)
            self.assertEqual(
                [key.task_id for key in keys], json.loads((split / "train.json").read_text()),
            )
            for key in keys:
                self.assertIs(key.augmentation.geometry, ArcGeometry.IDENTITY)
                self.assertEqual(len(set(key.augmentation.pair_indices)), 5)
                self.assertEqual(set(key.augmentation.color_mapping), set(range(10)))
                raw_path = root / "first/raw" / DERIVED_GROUP / "training" / f"{key.task_id}.json"
                task = load_training_task(raw_path)
                self.assertEqual([pair.input_grid[0][0] for pair in task.pairs], [i % 10 for i in range(30)])
            self.assertTrue(any(key.augmentation.color_mapping != tuple(range(10)) for key in keys))
            dataset = ArcTrainingDataset(split, "train", make_nvarc_tokenizer())
            self.assertEqual(tuple(map(str, dataset.keys)), tuple(serialized))
            for sample in dataset:
                self.assertEqual(len(sample.prefix_token_ends[1:]), 3)
                self.assertLessEqual(sample.prefix_token_ends[-1], 8400)

    def test_large_pairs_are_not_filtered_and_prefix_fits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root / "source", large=True)
            prepare_benchmark(source_root=root / "source", output_directory=root / "out", source_count=1)
            dataset = ArcTrainingDataset(root / "out/splits/train", "train", make_nvarc_tokenizer())
            sample = dataset[0]
            self.assertEqual(len(sample.key.augmentation.pair_indices), 5)
            self.assertEqual(sample.prefix_token_ends[-1], 8400)
            self.assertGreater(len(sample.input_ids), 8400)

    def test_refuses_existing_bundle_and_changed_seed_changes_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root / "source")
            first = prepare_benchmark(source_root=root / "source", output_directory=root / "out", source_count=6)
            original = (root / "out/splits/train/manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_benchmark(source_root=root / "source", output_directory=root / "out", source_count=6)
            self.assertEqual((root / "out/splits/train/manifest.json").read_bytes(), original)
            changed = prepare_benchmark(
                source_root=root / "source", output_directory=root / "changed", source_count=6, seed=168,
            )
            self.assertNotEqual(first["keys_sha256"], changed["keys_sha256"])

    def test_bad_source_records_abort_without_publishing_a_bundle(self) -> None:
        for field, value in (
            ("parent2_arc_agi_1_split", "evaluation"), ("sha256", "0" * 64),
            ("pair_count", 4), ("pair_count", 31), ("archive_path", "../../outside.json"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                records = _fixture(root / "source")
                records[0][field] = value
                _write_index(root / "source", records)
                with self.assertRaises(ValueError):
                    prepare_benchmark(
                        source_root=root / "source", output_directory=root / "out", source_count=12,
                    )
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
