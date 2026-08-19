"""Tests for the manual TrainingCollator inspection script."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.inspect_training_collator import inspect_collator


def _write_task(path: Path, values: tuple[int, ...]) -> None:
    pairs = [
        {"input": [[value]], "output": [[(value + 1) % 10]]}
        for value in values
    ]
    path.write_text(
        json.dumps({"train": pairs[:-1], "test": pairs[-1:]}),
        encoding="utf-8",
    )


class InspectTrainingCollatorTests(unittest.TestCase):
    def test_reports_valid_multi_source_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_path = directory / "first-task.json"
            second_path = directory / "second-task.json"
            _write_task(first_path, (0, 1, 2, 3))
            _write_task(second_path, (4, 5, 6, 7))
            args = argparse.Namespace(
                task_paths=[first_path, second_path],
                max_pairs=4,
                max_targets_per_source=2,
                max_cells=120,
                max_tokens=8192,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                batch = inspect_collator(args)

        self.assertEqual(batch.tensors.sequence_offsets.shape, (3,))
        self.assertEqual(batch.tensors.target_sequence_indices.tolist(), [0, 0, 1, 1])
        self.assertIn("target_images: shape=(4, 3, 480, 480)", output.getvalue())
        self.assertIn("available_targets: 3", output.getvalue())
        self.assertIn("selected_targets: 2", output.getvalue())
        self.assertIn("selected_candidate_positions: (1, 2)", output.getvalue())
        self.assertIn("contract_validation: valid", output.getvalue())

    def test_accepts_fewer_targets_than_the_configured_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_path = Path(temporary_directory) / "short-task.json"
            _write_task(task_path, (0, 1))
            args = argparse.Namespace(
                task_paths=[task_path],
                max_pairs=6,
                max_targets_per_source=2,
                max_cells=120,
                max_tokens=8192,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                batch = inspect_collator(args)

        self.assertEqual(batch.tensors.target_sequence_indices.tolist(), [0])
        self.assertIn("available_targets: 1", output.getvalue())
        self.assertIn("selected_targets: 1", output.getvalue())

    def test_no_target_limit_selects_all_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_path = Path(temporary_directory) / "all-targets.json"
            _write_task(task_path, (0, 1, 2, 3))
            args = argparse.Namespace(
                task_paths=[task_path],
                max_pairs=4,
                max_targets_per_source=None,
                max_cells=120,
                max_tokens=8192,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                batch = inspect_collator(args)

        self.assertEqual(batch.tensors.target_sequence_indices.tolist(), [0, 0, 0])
        self.assertIn("max_targets_per_source: all", output.getvalue())


if __name__ == "__main__":
    unittest.main()
