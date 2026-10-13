"""Prepared token lengths retain exact source identities without startup scans."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from diffusion_arc.training.source_lengths import (
    load_source_token_lengths, source_lengths_path, write_source_token_lengths,
)
from scripts.data import prepare_training_token_lengths as preparation


@dataclass(frozen=True)
class SourceKey:
    index: int

    @property
    def augmentation(self):
        return SimpleNamespace(pair_indices=(0, 1, 2, 3, 4))

    def __str__(self):
        return f"source-{self.index}"


class PreparedSources:
    def __init__(self, directory, count=3):
        self.split_directory = Path(directory)
        self.view_name = "train"
        self.keys = tuple(SourceKey(index) for index in range(count))
        self.fingerprint = "a" * 64
        self.max_tokens = 8400
        self.visited = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.visited.append(index)
        return SimpleNamespace(prefix_token_ends=(1, 4, 7, 10))


class SourceLengthTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = PreparedSources(self.root)
        self.lengths = ((4, 7, 10), (5, 12, 8400), (8, 15, 32))

    def _write(self):
        return write_source_token_lengths(
            self.dataset, self.lengths, min_context_pairs_before_target=2,
            provenance={"method": "test"},
        )

    def _load(self):
        return load_source_token_lengths(self.dataset, min_context_pairs_before_target=2)

    def test_roundtrip_reads_metadata_without_reading_any_source(self):
        path = self._write()
        self.assertEqual(path, self.root / "train.token_lengths.json")
        self.assertEqual(source_lengths_path(self.dataset), path)
        self.assertEqual(self._load(), self.lengths)
        self.assertEqual(self.dataset.visited, [])
        self.assertEqual(json.loads(path.read_text())["provenance"], {"method": "test"})

    def test_missing_metadata_requires_explicit_preparation(self):
        with self.assertRaisesRegex(FileNotFoundError, "prepare_training_token_lengths.py"):
            self._load()
        self.assertEqual(self.dataset.visited, [])

    def test_rejects_changed_dataset_view_selection_and_key_order(self):
        path = self._write()
        original = json.loads(path.read_text())
        for field, value in (
            ("schema_version", 2), ("schema_version", True),
            ("data_fingerprint", "b" * 64), ("view_name", "validation"),
            ("min_context_pairs_before_target", 1),
            ("min_context_pairs_before_target", 2.0),
            ("source_keys", list(reversed(original["source_keys"]))),
        ):
            with self.subTest(field=field, value=value):
                payload = deepcopy(original)
                payload[field] = value
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "Stale or mismatched"):
                    self._load()
        self.assertEqual(self.dataset.visited, [])

    def test_rejects_missing_targets_invalid_lengths_and_excessive_prefixes(self):
        path = self._write()
        original = json.loads(path.read_text())
        for lengths in (None, [], [[1, 2, 3]] * 2,
                        [[1, 2]] * 3, [[0, 2, 3]] * 3,
                        [[1, 1, 3]] * 3, [[3, 2, 4]] * 3,
                        [[True, 2, 3]] * 3, [[1, 2.0, 3]] * 3,
                        [[1, 2, 8401]] * 3):
            with self.subTest(lengths=lengths):
                payload = deepcopy(original)
                payload["source_prefix_token_lengths"] = lengths
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    self._load()

    def _steps(self):
        # Deliberately reordered rank ownership; metadata remains in dataset order.
        row = {
            "event": "step",
            "ranks": [
                {"source_indices": [2, 0], "source_keys": ["source-2", "source-0"],
                 "target_prefix_tokens": [8, 15, 32, 4, 7, 10], "targets": 6},
                {"source_indices": [1], "source_keys": ["source-1"],
                 "target_prefix_tokens": [5, 12, 8400], "targets": 3},
            ],
        }
        manifest = {
            "data_fingerprint": self.dataset.fingerprint,
            "effective_training": {"loop": {"min_context_pairs_before_target": 2}},
            "source_identity": {"source_sha256": "producer-code"},
        }
        return row, manifest

    def _save_steps(self, row, manifest):
        path = self.root / "steps.jsonl"
        path.write_text(json.dumps(row) + "\n")
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        return path

    def test_reuses_complete_logged_view_and_records_producer_identity(self):
        row, manifest = self._steps()
        steps = self._save_steps(row, manifest)
        lengths, provenance = preparation.source_prefix_lengths_from_steps(
            self.dataset, steps, min_context_pairs_before_target=2,
        )
        self.assertEqual(lengths, self.lengths)
        self.assertEqual(provenance["steps_sha256"], preparation._sha256(steps))
        self.assertEqual(provenance["benchmark_source_identity"], manifest["source_identity"])
        self.assertEqual(self.dataset.visited, [])

    def test_rejects_stale_incomplete_duplicated_and_misaligned_logged_sources(self):
        original, original_manifest = self._steps()
        variants = []
        row = deepcopy(original)
        row["ranks"].pop()
        variants.append((row, original_manifest, "cover every prepared source"))
        row = deepcopy(original)
        row["ranks"].append(deepcopy(row["ranks"][0]))
        variants.append((row, original_manifest, "appears more than once"))
        for index in (-1, 3, True):
            row = deepcopy(original)
            row["ranks"][1]["source_indices"] = [index]
            variants.append((row, original_manifest, "outside the prepared view"))
        row = deepcopy(original)
        row["ranks"][0]["source_keys"][0] = "wrong-source"
        variants.append((row, original_manifest, "key differs"))
        row = deepcopy(original)
        row["ranks"][0]["target_prefix_tokens"].pop()
        variants.append((row, original_manifest, "prefix count differs"))
        manifest = deepcopy(original_manifest)
        manifest["data_fingerprint"] = "wrong-data"
        variants.append((original, manifest, "fingerprint differs"))
        manifest = deepcopy(original_manifest)
        manifest["effective_training"]["loop"]["min_context_pairs_before_target"] = 1
        variants.append((original, manifest, "target selection differs"))
        for row, manifest, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                preparation.source_prefix_lengths_from_steps(
                    self.dataset, self._save_steps(row, manifest), min_context_pairs_before_target=2,
                )

    def test_explicit_preparation_reuses_logs_without_loading_tokenizer(self):
        row, manifest = self._steps()
        steps = self._save_steps(row, manifest)
        config = preparation.REPOSITORY_ROOT / "configs/training/benchmarks/b9_fsdp2_12k_no_forward_reshard.json"
        with patch.object(preparation, "ArcTrainingDataset", return_value=self.dataset), \
             patch.object(preparation.AutoTokenizer, "from_pretrained") as tokenizer:
            destination = preparation.prepare_training_token_lengths(config, steps_jsonl=steps)
        tokenizer.assert_not_called()
        self.assertEqual(destination, source_lengths_path(self.dataset))
        self.assertEqual(self._load(), self.lengths)
        self.assertEqual(self.dataset.visited, [])
        provenance = json.loads(destination.read_text())["provenance"]
        self.assertEqual(provenance["config_sha256"], preparation._sha256(config))

    def test_rejects_malformed_step_and_manifest_objects(self):
        row, manifest = self._steps()
        cases = [([], manifest, "step records"), (row, [], "manifest must be a JSON object")]
        malformed = deepcopy(row)
        malformed["ranks"][0] = []
        cases.append((malformed, manifest, "ranks must contain JSON objects"))
        malformed_manifest = deepcopy(manifest)
        malformed_manifest["effective_training"] = []
        cases.append((row, malformed_manifest, "missing its target selection"))
        malformed_manifest = deepcopy(manifest)
        malformed_manifest["effective_training"]["loop"]["min_context_pairs_before_target"] = 2.0
        cases.append((row, malformed_manifest, "target selection differs"))
        for step, producer, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                preparation.source_prefix_lengths_from_steps(
                    self.dataset, self._save_steps(step, producer), min_context_pairs_before_target=2,
                )

    def test_explicit_cpu_preparation_uses_only_cached_tokenizer_and_selects_suffix_targets(self):
        config = preparation.REPOSITORY_ROOT / "configs/training/arc_agi_1_epoch.json"
        tokenizer = Mock()
        with patch.object(preparation, "ArcTrainingDataset", return_value=self.dataset) as dataset, \
             patch.object(preparation.AutoTokenizer, "from_pretrained", return_value=tokenizer) as load:
            destination = preparation.prepare_training_token_lengths(config)
        self.assertEqual(load.call_args.kwargs, {"local_files_only": True, "trust_remote_code": False})
        self.assertEqual(tokenizer.model_max_length, preparation.NVARC_TOKENIZER_MAX_TOKENS)
        self.assertIs(dataset.call_args.kwargs["tokenizer"], tokenizer)
        self.assertEqual(self.dataset.visited, [0, 1, 2])
        self.assertEqual(self._load(), ((4, 7, 10),) * 3)
        self.assertEqual(json.loads(destination.read_text())["provenance"]["method"], "cpu_tokenization")

    def test_cpu_preparation_accepts_tokenizer_metadata_without_model_weights(self):
        config = preparation.REPOSITORY_ROOT / "configs/training/arc_agi_1_epoch.json"
        tokenizer_directory = self.root / "cached-tokenizer"
        with patch.object(preparation, "ArcTrainingDataset", return_value=self.dataset), \
                patch.object(preparation.AutoTokenizer, "from_pretrained", return_value=Mock()) as load:
            destination = preparation.prepare_training_token_lengths(config,
                tokenizer_directory=tokenizer_directory)
        self.assertEqual(load.call_args.args, (tokenizer_directory.resolve(),))
        self.assertEqual(load.call_args.kwargs, {"local_files_only": True, "trust_remote_code": False})
        provenance = json.loads(destination.read_text())["provenance"]
        self.assertEqual(provenance["tokenizer_directory"], str(tokenizer_directory.resolve()))
        self.assertNotEqual(provenance["nvarc_checkpoint"], provenance["tokenizer_directory"])


if __name__ == "__main__":
    unittest.main()
