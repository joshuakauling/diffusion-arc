"""Tests for target-aware FLUX text-input preparation."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import unittest

import torch

from diffusion_arc.model import (
    NVARC_CONTEXT_WIDTH,
    FluxTextInputs,
    PackedNvarcHiddenStates,
    build_flux_joint_attention_mask,
    prepare_flux_text_inputs,
)


def _packed_hidden_states(
    source_lengths: tuple[int, ...],
    *,
    random: bool = False,
) -> tuple[PackedNvarcHiddenStates, tuple[torch.Tensor, ...]]:
    offsets = [0]
    sources = []
    for source_length in source_lengths:
        if random:
            source = torch.randn(source_length, NVARC_CONTEXT_WIDTH)
        else:
            token_values = torch.arange(
                offsets[-1] + 10,
                offsets[-1] + source_length + 10,
                dtype=torch.float32,
            )
            source = token_values.unsqueeze(1).expand(-1, NVARC_CONTEXT_WIDTH)
        sources.append(source)
        offsets.append(offsets[-1] + source_length)

    return (
        PackedNvarcHiddenStates(
            hidden_states=torch.cat(sources),
            sequence_offsets=torch.tensor(offsets, dtype=torch.int64),
        ),
        tuple(sources),
    )


def _tiny_flux_model():
    from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel

    return Flux2Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=NVARC_CONTEXT_WIDTH,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    ).eval()


class FluxTextInputTests(unittest.TestCase):
    def test_builds_exact_prefixes_with_dynamic_right_padding(self) -> None:
        packed, _ = _packed_hidden_states((5, 4))

        result = prepare_flux_text_inputs(
            packed,
            target_sequence_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
            target_prefix_ends=torch.tensor([2, 5, 3], dtype=torch.int64),
        )

        self.assertIsInstance(result, FluxTextInputs)
        self.assertEqual(
            result.encoder_hidden_states.shape,
            (3, 5, NVARC_CONTEXT_WIDTH),
        )
        torch.testing.assert_close(
            result.encoder_hidden_states[:, :, 0],
            torch.tensor(
                [
                    [10, 11, 0, 0, 0],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 0, 0],
                ],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(
            result.text_attention_mask,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, True, True, True],
                    [True, True, True, False, False],
                ]
            ),
        )
        self.assertEqual(result.text_attention_mask.dtype, torch.bool)
        self.assertEqual(result.text_position_ids.shape, (3, 5, 4))
        self.assertEqual(result.text_position_ids.dtype, torch.int64)
        torch.testing.assert_close(
            result.text_position_ids[:, :, :3],
            torch.zeros((3, 5, 3), dtype=torch.int64),
        )
        torch.testing.assert_close(
            result.text_position_ids[:, :, 3],
            torch.arange(5, dtype=torch.int64).expand(3, -1),
        )

    def test_future_source_states_cannot_enter_an_earlier_prefix(self) -> None:
        packed, _ = _packed_hidden_states((5, 4))
        changed_hidden_states = packed.hidden_states.clone()
        changed_hidden_states[2:5] = 1000.0
        changed_hidden_states[8:9] = -1000.0
        changed = PackedNvarcHiddenStates(
            hidden_states=changed_hidden_states,
            sequence_offsets=packed.sequence_offsets,
        )
        target_sequence_indices = torch.tensor([0, 1], dtype=torch.int64)
        target_prefix_ends = torch.tensor([2, 3], dtype=torch.int64)

        original_inputs = prepare_flux_text_inputs(
            packed,
            target_sequence_indices,
            target_prefix_ends,
        )
        changed_inputs = prepare_flux_text_inputs(
            changed,
            target_sequence_indices,
            target_prefix_ends,
        )

        torch.testing.assert_close(
            original_inputs.encoder_hidden_states,
            changed_inputs.encoder_hidden_states,
        )

    def test_builds_transformer_ready_joint_key_mask(self) -> None:
        text_mask = torch.tensor(
            [[True, True, False], [True, True, True]],
            dtype=torch.bool,
        )

        result = build_flux_joint_attention_mask(text_mask, image_token_count=3)

        self.assertEqual(result.shape, (2, 1, 1, 6))
        self.assertEqual(result.dtype, torch.bool)
        torch.testing.assert_close(
            result[:, 0, 0],
            torch.tensor(
                [
                    [True, True, False, True, True, True],
                    [True, True, True, True, True, True],
                ]
            ),
        )

    def test_output_contract_is_frozen(self) -> None:
        result = FluxTextInputs(
            encoder_hidden_states=torch.zeros((1, 1, NVARC_CONTEXT_WIDTH)),
            text_position_ids=torch.zeros((1, 1, 4), dtype=torch.int64),
            text_attention_mask=torch.ones((1, 1), dtype=torch.bool),
        )

        with self.assertRaises(FrozenInstanceError):
            result.text_position_ids = torch.ones((1, 1, 4))  # type: ignore[misc]

    def test_masked_batch_matches_unpadded_outputs_and_gradients(self) -> None:
        torch.manual_seed(7)
        packed, sources = _packed_hidden_states((3, 5), random=True)
        text_inputs = prepare_flux_text_inputs(
            packed,
            target_sequence_indices=torch.tensor([0, 1], dtype=torch.int64),
            target_prefix_ends=torch.tensor([3, 5], dtype=torch.int64),
        )
        image_token_count = 3
        image_states = torch.randn(2, image_token_count, 8)
        image_ids = torch.zeros((2, image_token_count, 4), dtype=torch.int64)
        image_ids[:, :, 2] = torch.arange(image_token_count)
        timesteps = torch.tensor([0.25, 0.75])
        joint_mask = build_flux_joint_attention_mask(
            text_inputs.text_attention_mask,
            image_token_count,
        )
        model = _tiny_flux_model()

        with torch.no_grad():
            batched_output = model(
                hidden_states=image_states,
                encoder_hidden_states=text_inputs.encoder_hidden_states,
                timestep=timesteps,
                img_ids=image_ids,
                txt_ids=text_inputs.text_position_ids,
                guidance=None,
                joint_attention_kwargs={"attention_mask": joint_mask},
            ).sample
            reference_output = torch.cat(
                [
                    model(
                        hidden_states=image_states[index : index + 1],
                        encoder_hidden_states=source.unsqueeze(0),
                        timestep=timesteps[index : index + 1],
                        img_ids=image_ids[index : index + 1],
                        txt_ids=text_inputs.text_position_ids[
                            index : index + 1, : source.shape[0]
                        ],
                        guidance=None,
                    ).sample
                    for index, source in enumerate(sources)
                ]
            )

            poisoned_context = text_inputs.encoder_hidden_states.clone()
            poisoned_context[~text_inputs.text_attention_mask] = 1000.0
            poisoned_output = model(
                hidden_states=image_states,
                encoder_hidden_states=poisoned_context,
                timestep=timesteps,
                img_ids=image_ids,
                txt_ids=text_inputs.text_position_ids,
                guidance=None,
                joint_attention_kwargs={"attention_mask": joint_mask},
            ).sample
            unmasked_output = model(
                hidden_states=image_states,
                encoder_hidden_states=text_inputs.encoder_hidden_states,
                timestep=timesteps,
                img_ids=image_ids,
                txt_ids=text_inputs.text_position_ids,
                guidance=None,
            ).sample

        torch.testing.assert_close(
            batched_output,
            reference_output,
            atol=3e-6,
            rtol=3e-6,
        )
        torch.testing.assert_close(batched_output, poisoned_output, atol=0, rtol=0)
        self.assertGreater(
            float((batched_output[0] - unmasked_output[0]).abs().max()),
            1e-5,
        )

        batched_model = copy.deepcopy(model)
        reference_model = copy.deepcopy(model)
        batched_loss = (
            batched_model(
                hidden_states=image_states,
                encoder_hidden_states=text_inputs.encoder_hidden_states,
                timestep=timesteps,
                img_ids=image_ids,
                txt_ids=text_inputs.text_position_ids,
                guidance=None,
                joint_attention_kwargs={"attention_mask": joint_mask},
            )
            .sample.square()
            .flatten(1)
            .mean(dim=1)
            .mean()
        )
        batched_loss.backward()

        reference_losses = []
        for index, source in enumerate(sources):
            reference_prediction = reference_model(
                hidden_states=image_states[index : index + 1],
                encoder_hidden_states=source.unsqueeze(0),
                timestep=timesteps[index : index + 1],
                img_ids=image_ids[index : index + 1],
                txt_ids=text_inputs.text_position_ids[
                    index : index + 1, : source.shape[0]
                ],
                guidance=None,
            ).sample
            reference_losses.append(reference_prediction.square().mean())
        reference_loss = torch.stack(reference_losses).mean()
        reference_loss.backward()

        torch.testing.assert_close(batched_loss, reference_loss, atol=3e-6, rtol=3e-6)
        for batched_parameter, reference_parameter in zip(
            batched_model.parameters(),
            reference_model.parameters(),
            strict=True,
        ):
            if batched_parameter.grad is None:
                self.assertIsNone(reference_parameter.grad)
                continue
            torch.testing.assert_close(
                batched_parameter.grad,
                reference_parameter.grad,
                atol=3e-6,
                rtol=3e-6,
            )


if __name__ == "__main__":
    unittest.main()
