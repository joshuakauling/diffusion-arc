from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.milestone1_smoke_test import (
    _tokenize_exact,
    hash_nvarc_directory,
    load_arc_example,
)
from diffusion_arc.serialization import serialize_arc_task


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        self.text = text
        self.kwargs = kwargs
        return {
            "input_ids": torch.tensor([[11, 0, 15]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
        }


class SmokeHelperTest(unittest.TestCase):
    def test_loads_solution_separately_from_prompt(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            challenges = root / "challenges.json"
            solutions = root / "solutions.json"
            challenges.write_text(json.dumps({"abc": task}), encoding="utf-8")
            solutions.write_text(json.dumps({"abc": [[[9, 9]]]}), encoding="utf-8")

            loaded, target, task_id = load_arc_example(
                challenges,
                solutions,
                task_id="abc",
                test_index=0,
            )

        self.assertEqual(task_id, "abc")
        self.assertEqual(target, [[9, 9]])
        self.assertEqual(loaded, task)
        self.assertNotIn("99", serialize_arc_task(loaded))

    def test_exact_tokenization_disables_padding_and_truncation(self):
        tokenizer = FakeTokenizer()

        input_ids, attention_mask = _tokenize_exact(tokenizer, "prompt")

        self.assertEqual(input_ids.shape, (1, 3))
        self.assertTrue(bool((attention_mask == 1).all()))
        self.assertFalse(tokenizer.kwargs["padding"])
        self.assertFalse(tokenizer.kwargs["truncation"])
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])

    def test_nvarc_manifest_hash_is_stable_and_covers_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "config.json").write_text('{"hidden_size": 2560}', encoding="utf-8")
            (root / "vocab.json").write_text('{"0": 0}', encoding="utf-8")

            first = hash_nvarc_directory(root)
            second = hash_nvarc_directory(root)

        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            {entry["path"] for entry in first["files"]},
            {"config.json", "model.safetensors", "vocab.json"},
        )


if __name__ == "__main__":
    unittest.main()
