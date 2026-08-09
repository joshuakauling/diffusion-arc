from __future__ import annotations

import unittest

import torch

from diffusion_arc.positions import (
    flatten_latents,
    make_image_ids,
    make_text_ids,
    unflatten_latents,
)
from diffusion_arc.serialization import (
    serialize_arc_task,
    serialize_arc_task_for_llm,
    serialize_grid,
)


class SerializationTests(unittest.TestCase):
    def test_exact_nvarc_prefix_and_no_test_output_leakage(self) -> None:
        task = {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5], [6]],
                }
            ],
            "test": [
                {
                    "input": [[7, 8]],
                    # This field may exist in an evaluation fixture but must
                    # never enter frozen NVARC conditioning.
                    "output": [[9, 9, 9]],
                }
            ],
        }

        serialized = serialize_arc_task(task, test_index=0)

        self.assertEqual(
            serialized,
            "<|im_start|>user\n12\n34<|im_end|>"
            "<|im_start|>assistant\n5\n6<|im_end|>"
            "<|im_start|>user\n78<|im_end|>"
            "<|im_start|>assistant\n",
        )
        self.assertNotIn("999", serialized)

    def test_selected_test_input_only(self) -> None:
        task = {
            "train": [],
            "test": [{"input": [[1]]}, {"input": [[2]], "output": [[3]]}],
        }
        self.assertEqual(
            serialize_arc_task(task, 1),
            "<|im_start|>user\n2<|im_end|><|im_start|>assistant\n",
        )

    def test_rejects_non_rectangular_or_out_of_palette_grid(self) -> None:
        with self.assertRaises(ValueError):
            serialize_grid([[1, 2], [3]])
        with self.assertRaises(ValueError):
            serialize_grid([[10]])
        with self.assertRaises(TypeError):
            serialize_grid([[True]])

    def test_provider_neutral_prompt_includes_all_tests_without_outputs(self) -> None:
        task = {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6]]}],
            "test": [
                {"input": [[0, 1]], "output": [[9, 9, 9]]},
                {"input": [[2], [3]], "output": [[8, 8, 8]]},
            ],
        }

        compact = serialize_arc_task_for_llm(task)
        spaced = serialize_arc_task_for_llm(task, space_separated=True)

        self.assertIn("Training example 1 input:\n12\n34", compact)
        self.assertIn("Training example 1 output:\n56", compact)
        self.assertIn("Test input 1:\n01", compact)
        self.assertIn("Test input 2:\n2\n3", compact)
        self.assertIn("Training example 1 input:\n1 2\n3 4", spaced)
        self.assertIn("Test input 1:\n0 1", spaced)
        for forbidden in ("999", "9 9 9", "888", "8 8 8"):
            self.assertNotIn(forbidden, compact)
            self.assertNotIn(forbidden, spaced)

    def test_provider_neutral_prompt_rejects_an_empty_test_set(self) -> None:
        with self.assertRaises(ValueError):
            serialize_arc_task_for_llm({"train": [], "test": []})


class PositionTests(unittest.TestCase):
    def test_text_ids_match_bfl_axis_convention(self) -> None:
        ids = make_text_ids(batch=2, length=3)
        self.assertEqual(ids.shape, (2, 3, 4))
        self.assertEqual(ids.dtype, torch.int64)
        expected = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
        torch.testing.assert_close(ids[0], expected)
        torch.testing.assert_close(ids[1], expected)

    def test_image_ids_are_row_major_and_match_flattening(self) -> None:
        ids = make_image_ids(batch=1, height=2, width=3)
        expected = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 2, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 2, 0],
            ]
        )
        torch.testing.assert_close(ids[0], expected)

        latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        tokens = flatten_latents(latents)
        torch.testing.assert_close(tokens[0, :, 0], torch.arange(6.0))
        torch.testing.assert_close(unflatten_latents(tokens, 2, 3), latents)

    def test_unflatten_rejects_wrong_token_count(self) -> None:
        with self.assertRaises(ValueError):
            unflatten_latents(torch.zeros(1, 5, 4), height=2, width=3)


if __name__ == "__main__":
    unittest.main()
