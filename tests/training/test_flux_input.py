"""Tests for frozen preparation of individual FLUX training inputs."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

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
    prepare_flux_inputs_batch,
)
from diffusion_arc.training.nvarc_schedule import NvarcBatchShape, NvarcSchedulingPolicy


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
        offsets = sequence_offsets.tolist()
        return NvarcBatchOutput(
            sequence_hidden_states=tuple(
                hidden_states[start:end].clone()
                for start, end in zip(offsets, offsets[1:])
            ),
        )


class RecordingVaeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []
        self.next_target_value = 1

    def forward(self, target_images: torch.Tensor) -> torch.Tensor:
        self.calls.append(target_images.clone())
        target_count = target_images.shape[0]
        clean_latents = torch.stack(
            tuple(
                torch.full(
                    (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                    float(self.next_target_value + target_index),
                )
                for target_index in range(target_count)
            )
        )
        self.next_target_value += target_count
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
    def test_scheduling_restores_hidden_states_targets_and_observes_actual_calls(self) -> None:
        sequences = tuple(replace(
            _sequence(), key=replace(_sequence().key, task_id=f"{index:08x}"),
            input_ids=(index,) * length,
            prefix_token_ends=(1, 2, length),
            target_grids=(((index,),), ((index,),), ((index,),)),
        ) for index, length in enumerate((3, 9, 4, 10, 5)))
        ineligible = replace(_sequence(), prefix_token_ends=(2,), target_grids=(((0,),),))
        policy = NvarcSchedulingPolicy((NvarcBatchShape(2, 5), NvarcBatchShape(2, 10)), 0)
        calls, events, reports = [], [], []

        class Nvarc(nn.Module):
            def forward(self, ids, offsets, *, padded_length):
                offsets = offsets.tolist()
                calls.append((tuple(int(ids[start]) for start in offsets[:-1]), padded_length))
                events.append("nvarc")
                return NvarcBatchOutput(tuple(ids[start:end, None].float().clone()
                                              for start, end in zip(offsets, offsets[1:])))

        class Vae(nn.Module):
            def forward(self, images):
                events.append(("vae", len(images)))
                return images.flatten(1).unsqueeze(-1)

        with patch("diffusion_arc.training.flux_input._render_valid_grid_canvas",
                   side_effect=lambda grid: torch.tensor(grid, dtype=torch.float32)):
            inputs = prepare_flux_inputs_batch(
                (*sequences[:2], ineligible, *sequences[2:]),
                nvarc_runner=Nvarc(), vae_encoder=Vae(), min_context_pairs_before_target=2,
                vae_microbatch_size=6, nvarc_scheduling=policy, on_preparation=reports.append,
            )
        self.assertEqual(events, ["nvarc"] * 3 + [("vae", 6), ("vae", 4)])
        self.assertEqual([int(item.clean_latent[0, 0]) for item in inputs],
                         [index for index in range(5) for _ in range(2)])
        for index, item in enumerate(inputs):
            state = item.nvarc_sequence_state._prefix_for(item.prefix_end)
            self.assertTrue(torch.all(state == index // 2))
            self.assertFalse(state.requires_grad)
        report, = reports
        self.assertEqual([source.key for source in report.sources], [str(s.key) for s in sequences])
        self.assertEqual([(batch.source_indices, batch.padded_length) for batch in report.nvarc_batches], calls)
        self.assertEqual(sum(len(batch.source_indices) == 1 for batch in report.nvarc_batches), 1)
        self.assertEqual(sorted(index for batch in report.nvarc_batches for index in batch.source_indices), list(range(5)))

    def test_scheduling_skips_empty_eligible_window_without_work_or_report(self) -> None:
        nvarc, vae, reports = RecordingNvarcRunner(), RecordingVaeEncoder(), []
        result = prepare_flux_inputs_batch(
            (_sequence(),) * 3, nvarc_runner=nvarc, vae_encoder=vae,
            min_context_pairs_before_target=4,
            nvarc_scheduling=NvarcSchedulingPolicy((NvarcBatchShape(2, 10),), 0),
            on_preparation=reports.append,
        )
        self.assertEqual((result, nvarc.calls, vae.calls, reports), ((), [], [], []))

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

    def test_batches_eligible_sources_and_chunks_vae_targets_in_source_order(
        self,
    ) -> None:
        first = _sequence()
        second = replace(
            _sequence(),
            input_ids=(1, 2, 3, 4, 5, 6, 7),
            prefix_token_ends=(3, 6),
            target_grids=(((4,),), ((5,),)),
        )
        ineligible = replace(
            _sequence(),
            input_ids=(8, 9, 10),
            prefix_token_ends=(2,),
            target_grids=(((6,),),),
        )
        nvarc = RecordingNvarcRunner()
        vae = RecordingVaeEncoder()

        inputs = prepare_flux_inputs_batch(
            (first, ineligible, second),
            nvarc_runner=nvarc,  # type: ignore[arg-type]
            vae_encoder=vae,  # type: ignore[arg-type]
            min_context_pairs_before_target=2,
            vae_microbatch_size=2,
        )

        self.assertEqual(len(nvarc.calls), 1)
        packed_ids, offsets = nvarc.calls[0]
        self.assertEqual(
            packed_ids.tolist(),
            [*first.input_ids[:8], *second.input_ids[:6]],
        )
        self.assertEqual(offsets.tolist(), [0, 8, 14])
        self.assertEqual([call.shape[0] for call in vae.calls], [2, 1])
        torch.testing.assert_close(
            vae.calls[0][0],
            render_grid_canvas(first.target_grids[1]),
        )
        torch.testing.assert_close(
            vae.calls[1][0],
            render_grid_canvas(second.target_grids[1]),
        )

        self.assertEqual([item.prefix_end for item in inputs], [5, 8, 6])
        self.assertEqual(
            [float(item.clean_latent[0, 0]) for item in inputs],
            [1.0, 2.0, 3.0],
        )
        self.assertIs(inputs[0].nvarc_sequence_state, inputs[1].nvarc_sequence_state)
        self.assertIsNot(
            inputs[0].nvarc_sequence_state,
            inputs[2].nvarc_sequence_state,
        )
        self.assertEqual(inputs[0].nvarc_sequence_state.sequence_length, 8)
        self.assertEqual(inputs[2].nvarc_sequence_state.sequence_length, 6)

    def test_batched_preparation_rejects_empty_sources_and_invalid_vae_size(
        self,
    ) -> None:
        nvarc = RecordingNvarcRunner()
        vae = RecordingVaeEncoder()

        with self.assertRaisesRegex(ValueError, "sequences"):
            prepare_flux_inputs_batch(
                (),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
            )
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "vae_microbatch_size",
            ):
                prepare_flux_inputs_batch(
                    (_sequence(),),
                    nvarc_runner=nvarc,  # type: ignore[arg-type]
                    vae_encoder=vae,  # type: ignore[arg-type]
                    vae_microbatch_size=value,  # type: ignore[arg-type]
                )

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
