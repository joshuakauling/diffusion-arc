"""Verify epoch preparation through tiny local archives and the shared dataset."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
import tempfile
from types import SimpleNamespace
import unittest

from diffusion_arc.arc_data import ArcGeometry, parse_augmentation_key
from diffusion_arc.training.data import ArcTrainingDataset
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.data import build_dataloader
from scripts.data.prepare_training_data import prepare_training_data, sha256_file
from tests.arc_data.test_nvarc_benchmark import DERIVED_GROUP, _fixture, _write_index
from tests.training.test_data import make_nvarc_tokenizer, write_json


def _config(root: Path, source: Path, *, name: str = "epoch", count: int = 26, seed: int = 167):
    target_count = count * 3
    return SimpleNamespace(
        seed=seed,
        data=SimpleNamespace(
            split_directory=root / name / "splits" / "train", view_name="train", shuffle=False,
        ),
        loop=SimpleNamespace(optimizer_batch_size=48, max_optimizer_steps=(target_count + 47) // 48),
        recipe={
            "data": {
                "source_archive": source / "nvarc-all-classified.tar.gz",
                "task_index": source / f"{DERIVED_GROUP}_tasks.jsonl",
                "official_training_keys": source / "arc_agi_1_training_keys.json",
                "expected_task_count": count,
                "geometry_sampling": "uniform",
            },
            "loop": {"expected_final_batch_target_count": (target_count - 1) % 48 + 1},
        },
    )


class TrainingDataPreparationTest(unittest.TestCase):
    def test_run_config_preparation_and_b3_loader_preserve_order_and_tail(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _fixture(source, task_count=26)
            payload = json.loads((repository / "configs/training/arc_agi_1_epoch.json").read_text())
            payload["data"].update(
                source_archive="source/nvarc-all-classified.tar.gz",
                task_index=f"source/{DERIVED_GROUP}_tasks.jsonl",
                official_training_keys="source/arc_agi_1_training_keys.json",
                prepared_split_directory="epoch/splits/train", expected_task_count=26,
                num_workers=0, prefetch_factor=None,
            )
            payload["loop"].update(max_optimizer_steps=2, expected_target_count=78,
                                   expected_final_batch_target_count=30)
            payload["scheduler"].update(warmup_steps=2, first_update_warmup_fraction=0.5)
            config_path = root / "recipe.json"
            write_json(config_path, payload)
            config = load_training_config(config_path, expected_type="training_run")
            manifest = prepare_training_data(config, config_path=config_path)
            loader = build_dataloader(config, tokenizer=make_nvarc_tokenizer())
            samples = list(loader)
            self.assertEqual([sample.key.task_id for sample in samples],
                             [record["task_key"] for record in manifest["sources"]])
            self.assertEqual([len(sample.target_grids[1:]) for sample in samples], [3] * 26)
            self.assertEqual([sum(len(sample.target_grids[1:]) for sample in samples[start:start + 16])
                              for start in range(0, len(samples), 16)], [48, 30])

    def test_full_pool_order_d4_reproducibility_and_partial_target_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            records = _fixture(source, task_count=26)
            config_path = root / "recipe.json"
            write_json(config_path, {"fixture": True})
            manifests = []
            for name in ("epoch", "repeat", "reordered"):
                if name == "reordered":
                    _write_index(source, list(reversed(records)))
                config = _config(root, source, name=name)
                manifests.append(prepare_training_data(config, config_path=config_path))
            self.assertEqual(len({manifest["keys_sha256"] for manifest in manifests}), 1)
            config = _config(root, source)
            manifest = manifests[0]
            self.assertEqual(manifest["purpose"], "training_epoch")
            self.assertEqual(manifest["source_count"], 26)
            self.assertEqual(manifest["eligible_task_count"], 26)
            self.assertEqual(manifest["target_count"], 78)
            self.assertEqual(manifest["augmentation"]["geometry"], "uniform_D4")
            self.assertEqual(manifest["preparation"]["optimizer_step_count"], 2)
            self.assertEqual(manifest["preparation"]["final_batch_target_count"], 30)
            self.assertFalse(manifest["preparation"]["dataloader_shuffle"])

            split = config.data.split_directory
            serialized = json.loads((split / "augmented_keys/train.json").read_text())
            keys = [parse_augmentation_key(value) for value in serialized]
            expected_order = sorted(record["task_key"] for record in records)
            Random(config.seed).shuffle(expected_order)
            self.assertEqual([key.task_id for key in keys], expected_order)
            self.assertEqual(len(set(expected_order)), len(records))
            self.assertGreater(len({key.augmentation.geometry for key in keys}), 1)
            self.assertTrue({key.augmentation.geometry for key in keys} <= set(ArcGeometry))
            self.assertTrue(all(len(set(key.augmentation.pair_indices)) == 5 for key in keys))
            self.assertTrue(all(set(key.augmentation.color_mapping) == set(range(10)) for key in keys))
            self.assertEqual(json.loads((split / "validation.json").read_text()), [])
            dataset = ArcTrainingDataset(split, "train", make_nvarc_tokenizer())
            target_counts = [len(sample.prefix_token_ends[1:]) for sample in dataset]
            self.assertEqual(target_counts, [3] * 26)
            self.assertEqual(sum(target_counts[16:]), 30)

    def test_explicit_source_paths_and_independent_split_and_view_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _fixture(source, task_count=26)
            config = _config(root, source)
            for role, original in tuple(config.recipe["data"].items()):
                if not isinstance(original, Path):
                    continue
                destination = root / role / "input"
                destination.parent.mkdir()
                original.rename(destination)
                config.recipe["data"][role] = destination
            config.data.split_directory = root / "epoch/splits/first_pass"
            config.data.view_name = "augmented"
            config_path = root / "recipe.json"
            write_json(config_path, {"fixture": True})
            manifest = prepare_training_data(config, config_path=config_path)
            self.assertEqual(manifest["preparation"]["config_sha256"], sha256_file(config_path))
            for role in ("source_archive", "task_index", "official_training_keys"):
                self.assertEqual(manifest["inputs_sha256"][role], sha256_file(config.recipe["data"][role]))
            dataset = ArcTrainingDataset(config.data.split_directory, "augmented", make_nvarc_tokenizer())
            self.assertEqual(len(dataset), 26)

    def test_pool_count_mismatch_and_corrupt_sources_never_publish(self) -> None:
        for error_case in ("count", "checksum", "parent"):
            with self.subTest(error_case=error_case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                records = _fixture(source, task_count=26)
                if error_case == "checksum":
                    records[0]["sha256"] = "0" * 64
                elif error_case == "parent":
                    records[0]["parent2_arc_agi_1_split"] = "evaluation"
                _write_index(source, records)
                config = _config(root, source, count=25 if error_case == "count" else 26)
                config_path = root / "recipe.json"
                write_json(config_path, {"fixture": True})
                with self.assertRaises(ValueError):
                    prepare_training_data(config, config_path=config_path)
                self.assertFalse((root / "epoch").exists())

    def test_existing_bundle_is_unchanged_and_new_seed_changes_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _fixture(source, task_count=26)
            config_path = root / "recipe.json"
            write_json(config_path, {"fixture": True})
            config = _config(root, source)
            first = prepare_training_data(config, config_path=config_path)
            original_manifest = (config.data.split_directory / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_training_data(config, config_path=config_path)
            self.assertEqual((config.data.split_directory / "manifest.json").read_bytes(), original_manifest)
            changed = prepare_training_data(_config(root, source, name="changed", seed=168), config_path=config_path)
            self.assertNotEqual(first["keys_sha256"], changed["keys_sha256"])


if __name__ == "__main__":
    unittest.main()
