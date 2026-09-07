"""Tests for physical FLUX microbatch construction and state ownership."""

from __future__ import annotations

from dataclasses import replace
import gc
import unittest
import weakref

import torch

from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)
from diffusion_arc.training import (
    FluxInput,
    FluxMicrobatch,
    NvarcSequenceState,
    build_flux_microbatch,
)
from diffusion_arc.sequence_buckets import SequenceLengthBuckets


def _flux_inputs() -> tuple[FluxInput, ...]:
    prefix_ends = (2, 4, 6)
    sequence_state = NvarcSequenceState(
        hidden_states=torch.arange(24, dtype=torch.float32).reshape(6, 4),
        pending_prefix_ends=prefix_ends,
    )
    return tuple(
        FluxInput(
            nvarc_sequence_state=sequence_state,
            prefix_end=prefix_end,
            clean_latent=torch.full(
                (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                float(target_position + 1),
            ),
        )
        for target_position, prefix_end in enumerate(prefix_ends)
    )


class FluxMicrobatchTests(unittest.TestCase):
    def test_scheduled_padding_preserves_reordered_rows_and_releases_shared_owner(self) -> None:
        inputs = _flux_inputs()
        state = inputs[0].nvarc_sequence_state
        microbatch = build_flux_microbatch((inputs[2], inputs[0]), padded_length=9)
        self.assertEqual(microbatch.text_inputs.encoder_hidden_states.shape, (2, 9, 4))
        self.assertEqual(microbatch.text_inputs.text_attention_mask.sum(dim=1).tolist(), [6, 2])
        self.assertEqual(microbatch.clean_latents[:, 0, 0].tolist(), [3, 1])
        self.assertEqual(state.pending_prefix_ends, frozenset({4}))
        mask = microbatch.text_inputs.text_attention_mask
        self.assertTrue(torch.all(microbatch.text_inputs.encoder_hidden_states[~mask] == 0))
        self.assertEqual(microbatch.text_inputs.text_position_ids[0, :, 3].tolist(), list(range(9)))
        build_flux_microbatch((inputs[1],), padded_length=9)
        self.assertTrue(state.is_released)

    def test_bucketed_microbatch_releases_states_after_copying_real_prefixes(self) -> None:
        inputs = _flux_inputs()
        state = inputs[0].nvarc_sequence_state
        microbatch = build_flux_microbatch(inputs, length_buckets=SequenceLengthBuckets((8,)))
        self.assertEqual(microbatch.text_inputs.encoder_hidden_states.shape, (3, 8, 4))
        self.assertEqual(microbatch.text_inputs.text_attention_mask.sum(dim=1).tolist(), [2, 4, 6])
        self.assertTrue(state.is_released)
        self.assertEqual(state.allocated_bytes, 0)
        self.assertEqual(microbatch.clean_latents[:, 0, 0].tolist(), [1, 2, 3])
        torch.testing.assert_close(microbatch.text_inputs.encoder_hidden_states[0, :2],
                                   torch.arange(8, dtype=torch.float32).reshape(2, 4))

    def test_bucket_overflow_keeps_all_prefixes_pending(self) -> None:
        inputs = _flux_inputs()
        state = inputs[0].nvarc_sequence_state
        with self.assertRaisesRegex(ValueError, "exceeds the largest bucket"):
            build_flux_microbatch(inputs, length_buckets=SequenceLengthBuckets((4,)))
        self.assertEqual(state.pending_prefix_ends, frozenset({2, 4, 6}))
        self.assertFalse(state.is_released)

    def test_sequence_owner_rejects_autograd_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "autograd graph"):
            NvarcSequenceState(
                hidden_states=torch.ones((2, 4), requires_grad=True),
                pending_prefix_ends=(2,),
            )

    def test_builds_physical_batch_and_releases_only_after_last_input(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state

        first_microbatch = build_flux_microbatch(inputs[:2])

        self.assertIsInstance(first_microbatch, FluxMicrobatch)
        self.assertEqual(first_microbatch.target_count, 2)
        self.assertEqual(
            first_microbatch.text_inputs.encoder_hidden_states.shape,
            (2, 4, 4),
        )
        torch.testing.assert_close(
            first_microbatch.text_inputs.text_attention_mask,
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                ]
            ),
        )
        self.assertEqual(
            first_microbatch.clean_latents.shape,
            (2, TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
        )
        self.assertEqual(first_microbatch.image_position_ids.shape, (2, 900, 4))
        self.assertEqual(first_microbatch.image_position_ids.dtype, torch.int64)
        torch.testing.assert_close(
            first_microbatch.image_position_ids[0, (0, 1, 29, 30, 899), :],
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 29, 0],
                    [0, 1, 0, 0],
                    [0, 29, 29, 0],
                ],
                dtype=torch.int64,
            ),
        )
        torch.testing.assert_close(
            first_microbatch.image_position_ids[0],
            first_microbatch.image_position_ids[1],
        )
        self.assertEqual(sequence_state.pending_prefix_ends, frozenset({6}))
        self.assertFalse(sequence_state.is_released)

        retained_text = first_microbatch.text_inputs.encoder_hidden_states.clone()
        second_microbatch = build_flux_microbatch(inputs[2:])

        self.assertEqual(sequence_state.pending_prefix_ends, frozenset())
        self.assertTrue(sequence_state.is_released)
        self.assertEqual(sequence_state.allocated_bytes, 0)
        torch.testing.assert_close(
            first_microbatch.text_inputs.encoder_hidden_states,
            retained_text,
        )

        with self.assertRaisesRegex(RuntimeError, "not pending"):
            build_flux_microbatch(inputs[2:])

    def test_failed_tensor_build_does_not_consume_any_input(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        pending_before = sequence_state.pending_prefix_ends
        incompatible = replace(
            inputs[1],
            clean_latent=torch.zeros(
                (TARGET_LATENT_TOKEN_COUNT - 1, TARGET_LATENT_CHANNELS)
            ),
        )

        with self.assertRaises(RuntimeError):
            build_flux_microbatch((inputs[0], incompatible))

        self.assertEqual(sequence_state.pending_prefix_ends, pending_before)
        self.assertFalse(sequence_state.is_released)
        self.assertGreater(sequence_state.allocated_bytes, 0)

    def test_releases_independent_sequence_owners_separately(self) -> None:
        first_source_inputs = _flux_inputs()
        second_source_inputs = _flux_inputs()
        first_state = first_source_inputs[0].nvarc_sequence_state
        second_state = second_source_inputs[0].nvarc_sequence_state

        microbatch = build_flux_microbatch(
            (*first_source_inputs, second_source_inputs[0])
        )

        self.assertEqual(microbatch.target_count, 4)
        self.assertTrue(first_state.is_released)
        self.assertEqual(first_state.pending_prefix_ends, frozenset())
        self.assertFalse(second_state.is_released)
        self.assertEqual(
            second_state.pending_prefix_ends,
            frozenset({4, 6}),
        )

    def test_finished_microbatch_retains_neither_source_tensor_nor_owner(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        state_reference = weakref.ref(sequence_state)
        hidden_states = sequence_state._hidden_states
        self.assertIsNotNone(hidden_states)
        hidden_state_reference = weakref.ref(hidden_states)
        del hidden_states

        microbatch = build_flux_microbatch(inputs)
        self.assertTrue(sequence_state.is_released)
        gc.collect()
        self.assertIsNone(hidden_state_reference())

        del inputs
        del sequence_state
        gc.collect()

        self.assertIsNone(state_reference())
        self.assertEqual(microbatch.target_count, 3)

    def test_rejects_empty_duplicate_and_unknown_inputs_without_consuming(self) -> None:
        inputs = _flux_inputs()
        sequence_state = inputs[0].nvarc_sequence_state
        pending_before = sequence_state.pending_prefix_ends

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_flux_microbatch(())
        with self.assertRaisesRegex(ValueError, "prefix ends must be unique"):
            build_flux_microbatch((inputs[0], inputs[0]))

        unknown = replace(inputs[0], prefix_end=5)
        with self.assertRaisesRegex(RuntimeError, "not pending"):
            build_flux_microbatch((unknown,))

        self.assertEqual(sequence_state.pending_prefix_ends, pending_before)
        self.assertFalse(sequence_state.is_released)


if __name__ == "__main__":
    unittest.main()
