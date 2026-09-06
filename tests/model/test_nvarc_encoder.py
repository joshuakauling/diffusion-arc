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
    NvarcBatchOutput,
    NvarcBatchRunner,
)
from diffusion_arc.sequence_buckets import SequenceLengthBuckets


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
    def test_bucket_padding_keeps_masks_and_compact_source_outputs(self) -> None:
        model = FakeNvarc()
        runner = NvarcBatchRunner(model, length_buckets=SequenceLengthBuckets((8, 16)))
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        offsets = torch.tensor([0, 2, 6])
        result = runner(ids, offsets)
        self.assertEqual(tuple(model.last_input_ids.shape), (2, 8))
        torch.testing.assert_close(model.last_attention_mask, torch.tensor([
            [True, True, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False],
        ]))
        self.assertTrue(torch.all(model.last_input_ids[~model.last_attention_mask] == NVARC_PAD_TOKEN_ID))
        reference = NvarcBatchRunner(FakeNvarc())(ids, offsets)
        for actual, expected in zip(result.sequence_hidden_states, reference.sequence_hidden_states):
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.untyped_storage().nbytes(), actual.numel() * actual.element_size())
            self.assertFalse(actual.requires_grad)
        self.assertNotEqual(result.sequence_hidden_states[0].data_ptr(),
                            result.sequence_hidden_states[1].data_ptr())

    def test_bucketed_qwen_matches_separate_exact_length_encodings(self) -> None:
        from transformers import Qwen3Config, Qwen3Model

        torch.manual_seed(11)
        config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=4, max_position_embeddings=32)
        config._attn_implementation = "sdpa"
        model = Qwen3Model(config)
        runner = NvarcBatchRunner(model, length_buckets=SequenceLengthBuckets((8,)))
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        padded = runner(ids, torch.tensor([0, 2, 6]))
        exact_runner = NvarcBatchRunner(model)
        for index, tokens in enumerate((ids[:2], ids[2:])):
            exact = exact_runner(tokens, torch.tensor([0, len(tokens)]))
            torch.testing.assert_close(padded.sequence_hidden_states[index],
                                       exact.sequence_hidden_states[0], atol=3e-6, rtol=3e-6)

    def test_bucket_overflow_fails_before_model_execution(self) -> None:
        model = FakeNvarc()
        runner = NvarcBatchRunner(model, length_buckets=SequenceLengthBuckets((4,)))
        with self.assertRaisesRegex(ValueError, "exceeds the largest bucket"):
            runner(torch.ones(5, dtype=torch.int64), torch.tensor([0, 5]))
        self.assertEqual(model.forward_calls, 0)

    def test_forward_interface_is_target_blind(self) -> None:
        parameter_names = tuple(inspect.signature(NvarcBatchRunner.forward).parameters)

        self.assertEqual(
            parameter_names,
            ("self", "input_ids", "sequence_offsets"),
        )

    def test_batches_variable_lengths_and_returns_independent_sequence_states(
        self,
    ) -> None:
        model = FakeNvarc()
        runner = NvarcBatchRunner(model)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        sequence_offsets = torch.tensor([0, 2, 6], dtype=torch.int64)

        result = runner(input_ids, sequence_offsets)

        self.assertIsInstance(result, NvarcBatchOutput)
        self.assertEqual(len(result.sequence_hidden_states), 2)
        self.assertEqual(
            [tuple(states.shape) for states in result.sequence_hidden_states],
            [(2, NVARC_CONTEXT_WIDTH), (4, NVARC_CONTEXT_WIDTH)],
        )
        self.assertNotEqual(
            result.sequence_hidden_states[0].untyped_storage().data_ptr(),
            result.sequence_hidden_states[1].untyped_storage().data_ptr(),
        )
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

        expected_sources = (input_ids[:2], input_ids[2:])
        for sequence_states, expected_tokens in zip(
            result.sequence_hidden_states,
            expected_sources,
            strict=True,
        ):
            for chunk_index, hidden_state_index in enumerate(
                NVARC_HIDDEN_STATE_INDICES
            ):
                chunk_start = chunk_index * NVARC_HIDDEN_SIZE
                torch.testing.assert_close(
                    sequence_states[:, chunk_start],
                    expected_tokens.to(torch.float32) + float(hidden_state_index),
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
        self.assertTrue(
            all(
                not hidden_states.requires_grad and hidden_states.grad_fn is None
                for hidden_states in result.sequence_hidden_states
            )
        )

    def test_fails_clearly_without_input_embedding_interface(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "input embeddings"):
            NvarcBatchRunner(MissingInputEmbeddingsNvarc())(
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([0, 2], dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
