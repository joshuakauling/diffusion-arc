from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.print_arc_prompt import load_arc_task, main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/print_arc_prompt.py"


def _task() -> dict:
    return {
        "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6]]}],
        "test": [{"input": [[7, 8]], "output": [[9, 9, 9]]}],
    }


def _write_task(data_root: Path, split: str, task_id: str) -> None:
    split_root = data_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / f"{task_id}.json").write_text(
        json.dumps(_task()),
        encoding="utf-8",
    )


class PrintArcPromptTests(unittest.TestCase):
    def test_loads_ids_from_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            _write_task(data_root, "training", "abc123ef")
            _write_task(data_root, "evaluation", "deadbeef")

            training_task, training_split = load_arc_task(
                "ABC123EF",
                data_root=data_root,
            )
            evaluation_task, evaluation_split = load_arc_task(
                "deadbeef",
                data_root=data_root,
            )

        self.assertEqual(training_split, "training")
        self.assertEqual(evaluation_split, "evaluation")
        self.assertEqual(training_task, _task())
        self.assertEqual(evaluation_task, _task())

    def test_cli_switches_between_compact_and_space_separated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            _write_task(data_root, "training", "abc123ef")
            base_command = [
                sys.executable,
                str(SCRIPT_PATH),
                "abc123ef",
                "--data-root",
                str(data_root),
            ]

            compact = subprocess.run(
                base_command,
                check=True,
                capture_output=True,
                text=True,
            )
            spaced = subprocess.run(
                [*base_command, "--space-separated"],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("Training example 1 input:\n12\n34", compact.stdout)
        self.assertIn("Test input 1:\n78", compact.stdout)
        self.assertIn("Training example 1 input:\n1 2\n3 4", spaced.stdout)
        self.assertIn("Test input 1:\n7 8", spaced.stdout)
        self.assertNotIn("999", compact.stdout)
        self.assertNotIn("9 9 9", spaced.stdout)
        self.assertEqual(compact.stderr, "")
        self.assertEqual(spaced.stderr, "")

    def test_cli_requests_task_id_when_positional_argument_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            _write_task(data_root, "training", "abc123ef")
            with (
                mock.patch("builtins.input", return_value="abc123ef") as mocked_input,
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                exit_code = main(["--data-root", str(data_root)])

        self.assertEqual(exit_code, 0)
        mocked_input.assert_called_once_with("ARC task ID: ")
        self.assertIn("Test input 1:\n78", stdout.getvalue())

    def test_rejects_missing_and_duplicate_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            with self.assertRaises(FileNotFoundError):
                load_arc_task("abc123ef", data_root=data_root)

            _write_task(data_root, "training", "abc123ef")
            _write_task(data_root, "evaluation", "abc123ef")
            with self.assertRaises(ValueError):
                load_arc_task("abc123ef", data_root=data_root)


if __name__ == "__main__":
    unittest.main()
