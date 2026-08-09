from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diffusion_arc.dataset import (
    load_arc_dataset,
    load_dataset_manifest,
    sha256_file,
    summarize_samples,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = REPOSITORY_ROOT / "configs/datasets/arc_agi_2_initial.json"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _task(*, include_test_output: bool = False):
    test_example = {"input": [[3]]}
    if include_test_output:
        test_example["output"] = [[9]]
    return {
        "train": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[2]], "output": [[1]]},
        ],
        "test": [test_example],
    }


def _manifest_payload(
    challenges: Path,
    solutions: Path,
    *,
    task_id: str = "abc123ef",
    test_index: int = 0,
):
    return {
        "schema_version": 1,
        "name": "fixture",
        "benchmark": "arc-fixture",
        "source_split": "training",
        "source": {
            "challenges_file": challenges.name,
            "challenges_sha256": sha256_file(challenges),
            "solutions_file": solutions.name,
            "solutions_sha256": sha256_file(solutions),
        },
        "samples": [
            {
                "task_id": task_id,
                "test_index": test_index,
                "label": "fixture_rule",
            }
        ],
        "subset_prefix_lengths": {"smoke_1": 1},
        "selection_constraints": {
            "test_examples_per_task": 1,
            "min_train_examples": 2,
            "max_train_examples": 2,
            "max_prompt_characters": 1000,
            "max_grid_dimension": 2,
        },
    }


class DatasetManifestTest(unittest.TestCase):
    def test_canonical_manifest_has_confirmed_nested_keys(self):
        manifest = load_dataset_manifest(CANONICAL_MANIFEST)

        self.assertEqual(
            [sample.task_id for sample in manifest.samples],
            [
                "3c9b0459",
                "0d3d703e",
                "8be77c9e",
                "c9e6f938",
                "00576224",
                "0520fde7",
                "88a62173",
                "d631b094",
            ],
        )
        self.assertEqual(
            dict(manifest.subset_prefix_lengths),
            {"smoke_1": 1, "tiny_4": 4, "tiny_8": 8},
        )
        self.assertEqual(
            manifest.specs_for_subset("tiny_4"),
            manifest.samples[:4],
        )

    def test_rejects_duplicate_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenges = root / "challenges.json"
            solutions = root / "solutions.json"
            manifest_path = root / "manifest.json"
            _write_json(challenges, {"abc123ef": _task()})
            _write_json(solutions, {"abc123ef": [[[9]]]})
            payload = _manifest_payload(challenges, solutions)
            payload["samples"].append(dict(payload["samples"][0]))
            payload["subset_prefix_lengths"] = {"all": 2}
            _write_json(manifest_path, payload)

            with self.assertRaisesRegex(ValueError, "Duplicate manifest sample"):
                load_dataset_manifest(manifest_path)


class ArcDatasetTest(unittest.TestCase):
    def _fixture(self, root: Path, *, task=None, solutions_payload=None):
        challenges = root / "challenges.json"
        solutions = root / "solutions.json"
        manifest = root / "manifest.json"
        _write_json(challenges, {"abc123ef": task or _task()})
        _write_json(
            solutions,
            solutions_payload if solutions_payload is not None else {"abc123ef": [[[9]]]},
        )
        _write_json(manifest, _manifest_payload(challenges, solutions))
        return manifest, challenges, solutions

    def test_loads_target_separately_and_summarizes_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, challenges, solutions = self._fixture(Path(directory))

            samples = load_arc_dataset(
                manifest,
                challenges_path=challenges,
                solutions_path=solutions,
                subset="smoke_1",
            )
            summary = summarize_samples(samples)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].target_grid, ((9,),))
        self.assertNotIn("9", samples[0].prompt)
        self.assertTrue(samples[0].prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(samples[0].test_input_shape, (1, 1))
        self.assertEqual(samples[0].target_shape, (1, 1))
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["samples"][0]["task_id"], "abc123ef")

    def test_rejects_inline_test_output(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, challenges, solutions = self._fixture(
                Path(directory),
                task=_task(include_test_output=True),
            )

            with self.assertRaisesRegex(ValueError, "must not contain output"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )

    def test_rejects_missing_solution(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, challenges, solutions = self._fixture(
                Path(directory),
                solutions_payload={},
            )

            with self.assertRaisesRegex(KeyError, "absent from the solution source"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )

    def test_rejects_unknown_task_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, challenges, solutions = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["task_id"] = "deadbeef"
            _write_json(manifest, payload)

            with self.assertRaisesRegex(KeyError, "absent from the challenge source"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )

    def test_rejects_out_of_range_test_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, challenges, solutions = self._fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["samples"][0]["test_index"] = 1
            _write_json(manifest, payload)

            with self.assertRaisesRegex(IndexError, "test index 1"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )

    def test_rejects_invalid_grid(self):
        invalid_task = _task()
        invalid_task["train"][0]["input"] = [[10]]
        with tempfile.TemporaryDirectory() as directory:
            manifest, challenges, solutions = self._fixture(
                Path(directory),
                task=invalid_task,
            )

            with self.assertRaisesRegex(ValueError, "must be in \\[0, 9\\]"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )

    def test_rejects_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, challenges, solutions = self._fixture(Path(directory))
            _write_json(challenges, {"abc123ef": _task(), "deadbeef": _task()})

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_arc_dataset(
                    manifest,
                    challenges_path=challenges,
                    solutions_path=solutions,
                    subset="smoke_1",
                )


if __name__ == "__main__":
    unittest.main()
