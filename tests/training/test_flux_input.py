"""Tests for frozen preparation of individual FLUX training inputs."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_arc.arc_data.augmentation import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcGeometry,
)
from diffusion_arc.arc_data.canvas import render_grid_canvas
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model import (
    NvarcBatchOutput,
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)
from diffusion_arc.training.flux_input import (
    FluxInput,
    NvarcSequenceState,
    prepare_flux_inputs,
)


class RecordingNvarcRunner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        sequence_offsets: torch.Tensor,
    ) -> NvarcBatchOutput:
        self.calls.append((input_ids.clone(), sequence_offsets.clone()))
        hidden_states = torch.arange(
            input_ids.numel() * 8,
            dtype=torch.float32,
        ).reshape(input_ids.numel(), 8)
        return NvarcBatchOutput(
            sequence_hidden_states=(hidden_states,),
        )


class RecordingVaeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, target_images: torch.Tensor) -> torch.Tensor:
        self.calls.append(target_images.clone())
        target_count = target_images.shape[0]
        clean_latents = torch.stack(
            tuple(
                torch.full(
                    (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                    float(target_index + 1),
                )
                for target_index in range(target_count)
            )
        )
        return clean_latents


def _sequence() -> ArcTokenizedTrainingSample:
    key = ArcAugmentationKey(
        task_id="abc123ef",
        augmentation=ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=(4, 1, 3, 2),
        ),
    )
    return ArcTokenizedTrainingSample(
        key=key,
        input_ids=(14, 11, 10, 0, 15, 14, 12, 10, 7, 15),
        prefix_token_ends=(2, 5, 8),
        target_grids=(((2,),), ((7,),), ((9,),)),
    )


class FluxInputPreparationTests(unittest.TestCase):
    def test_prepares_all_targets_with_one_nvarc_and_one_vae_forward(self) -> None:
        sequence = _sequence()
        nvarc = RecordingNvarcRunner()
        vae = RecordingVaeEncoder()

        inputs = prepare_flux_inputs(
            sequence,
            nvarc_runner=nvarc,  # type: ignore[arg-type]
            vae_encoder=vae,  # type: ignore[arg-type]
        )

        self.assertEqual(len(nvarc.calls), 1)
        self.assertEqual(nvarc.calls[0][0].tolist(), list(sequence.input_ids[:8]))
        self.assertEqual(nvarc.calls[0][1].tolist(), [0, 8])
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(vae.calls[0].shape, (3, 3, 480, 480))
        torch.testing.assert_close(
            vae.calls[0][1],
            render_grid_canvas(sequence.target_grids[1]),
        )

        self.assertTrue(all(isinstance(item, FluxInput) for item in inputs))
        self.assertEqual([item.prefix_end for item in inputs], [2, 5, 8])
        self.assertEqual(
            tuple(FluxInput.__dataclass_fields__),
            ("nvarc_sequence_state", "prefix_end", "clean_latent"),
        )
        self.assertTrue(
            all(
                item.nvarc_sequence_state is inputs[0].nvarc_sequence_state
                for item in inputs
            )
        )
        self.assertIsInstance(inputs[0].nvarc_sequence_state, NvarcSequenceState)
        self.assertEqual(inputs[0].nvarc_sequence_state.sequence_length, 8)
        self.assertEqual(
            inputs[0].nvarc_sequence_state.pending_prefix_ends,
            frozenset({2, 5, 8}),
        )
        self.assertFalse(inputs[0].nvarc_sequence_state.is_released)
        self.assertGreater(inputs[0].nvarc_sequence_state.allocated_bytes, 0)
        self.assertEqual(
            [float(item.clean_latent[0, 0]) for item in inputs],
            [1.0, 2.0, 3.0],
        )

    def test_applies_minimum_context_before_frozen_model_work(self) -> None:
        sequence = _sequence()
        nvarc = RecordingNvarcRunner()
        vae = RecordingVaeEncoder()

        inputs = prepare_flux_inputs(
            sequence,
            nvarc_runner=nvarc,  # type: ignore[arg-type]
            vae_encoder=vae,  # type: ignore[arg-type]
            min_context_pairs_before_target=2,
        )

        self.assertEqual([item.prefix_end for item in inputs], [5, 8])
        self.assertEqual(
            [float(item.clean_latent[0, 0]) for item in inputs],
            [1.0, 2.0],
        )
        self.assertEqual(len(nvarc.calls), 1)
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(vae.calls[0].shape[0], 2)
        torch.testing.assert_close(
            vae.calls[0][0],
            render_grid_canvas(sequence.target_grids[1]),
        )

    def test_skips_ineligible_sequence_without_model_calls(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = RecordingVaeEncoder()

        inputs = prepare_flux_inputs(
            _sequence(),
            nvarc_runner=nvarc,  # type: ignore[arg-type]
            vae_encoder=vae,  # type: ignore[arg-type]
            min_context_pairs_before_target=4,
        )

        self.assertEqual(inputs, ())
        self.assertEqual(nvarc.calls, [])
        self.assertEqual(vae.calls, [])

    def test_rejects_invalid_minimum_context(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_flux_inputs(
                    _sequence(),
                    nvarc_runner=RecordingNvarcRunner(),  # type: ignore[arg-type]
                    vae_encoder=RecordingVaeEncoder(),  # type: ignore[arg-type]
                    min_context_pairs_before_target=value,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
