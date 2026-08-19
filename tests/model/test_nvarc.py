"""Tests for target-blind batched NVARC execution."""

from __future__ import annotations

import inspect
import types
import unittest

import torch
from torch import Tensor, nn

from diffusion_arc.model import (
    NVARC_CONTEXT_WIDTH,
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_PAD_TOKEN_ID,
    NvarcBatchRunner,
    PackedNvarcHiddenStates,
)


class FakeNvarc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(16, 1)
        self.forward_calls = 0
        self.seen_training_modes: list[bool] = []
        self.seen_grad_modes: list[bool] = []
        self.last_input_ids: Tensor | None = None
        self.last_attention_mask: Tensor | None = None
        self.last_output_hidden_states: bool | None = None
        self.last_use_cache: bool | None = None
        self.last_return_dict: bool | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        return_dict: bool,
    ) -> types.SimpleNamespace:
        self.forward_calls += 1
        self.seen_training_modes.append(self.training)
        self.seen_grad_modes.append(torch.is_grad_enabled())
        self.last_input_ids = input_ids.detach().cpu().clone()
        self.last_attention_mask = attention_mask.detach().cpu().clone()
        self.last_output_hidden_states = output_hidden_states
        self.last_use_cache = use_cache
        self.last_return_dict = return_dict

        batch_size, sequence_length = input_ids.shape
        token_values = input_ids.to(torch.float32).unsqueeze(-1)
        hidden_states = tuple(
            (token_values + float(index)).expand(
                batch_size,
                sequence_length,
                NVARC_HIDDEN_SIZE,
            )
            for index in range(max(NVARC_HIDDEN_STATE_INDICES) + 1)
        )
        return types.SimpleNamespace(hidden_states=hidden_states)


class MissingInputEmbeddingsNvarc(nn.Module):
    pass


class NvarcBatchRunnerTests(unittest.TestCase):
    def test_forward_interface_is_target_blind(self) -> None:
        parameter_names = tuple(inspect.signature(NvarcBatchRunner.forward).parameters)

        self.assertEqual(
            parameter_names,
            ("self", "input_ids", "sequence_offsets"),
        )

    def test_batches_variable_lengths_and_returns_compact_hidden_states(self) -> None:
        model = FakeNvarc()
        runner = NvarcBatchRunner(model)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        sequence_offsets = torch.tensor([0, 2, 6], dtype=torch.int64)

        result = runner(input_ids, sequence_offsets)

        self.assertIsInstance(result, PackedNvarcHiddenStates)
        self.assertIs(result.sequence_offsets, sequence_offsets)
        self.assertEqual(result.hidden_states.shape, (6, NVARC_CONTEXT_WIDTH))
        self.assertEqual(model.forward_calls, 1)
        torch.testing.assert_close(
            model.last_input_ids,
            torch.tensor(
                [
                    [1, 2, NVARC_PAD_TOKEN_ID, NVARC_PAD_TOKEN_ID],
                    [3, 4, 5, 6],
                ],
                dtype=torch.int64,
            ),
        )
        torch.testing.assert_close(
            model.last_attention_mask,
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                ]
            ),
        )
        self.assertTrue(model.last_output_hidden_states)
        self.assertFalse(model.last_use_cache)
        self.assertTrue(model.last_return_dict)

        expected_tokens = input_ids.to(torch.float32)
        for chunk_index, hidden_state_index in enumerate(NVARC_HIDDEN_STATE_INDICES):
            chunk_start = chunk_index * NVARC_HIDDEN_SIZE
            torch.testing.assert_close(
                result.hidden_states[:, chunk_start],
                expected_tokens + float(hidden_state_index),
            )

    def test_freezes_model_and_keeps_eval_mode(self) -> None:
        model = FakeNvarc().train()
        runner = NvarcBatchRunner(model)

        self.assertFalse(runner.training)
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

        runner.train(True)
        self.assertFalse(runner.training)
        self.assertFalse(model.training)

        result = runner(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
        )

        self.assertEqual(model.seen_training_modes, [False])
        self.assertEqual(model.seen_grad_modes, [False])
        self.assertFalse(result.hidden_states.requires_grad)
        self.assertIsNone(result.hidden_states.grad_fn)

    def test_fails_clearly_without_input_embedding_interface(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "input embeddings"):
            NvarcBatchRunner(MissingInputEmbeddingsNvarc())(
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([0, 2], dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
