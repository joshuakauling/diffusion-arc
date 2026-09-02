"""Tests for deterministic flow probes used by the overfitting pilot."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
    FluxTextInputs,
)
from diffusion_arc.training import (
    FluxMicrobatch,
    evaluate_fixed_flow_probe,
    prepare_flow_probe_microbatch,
)


class FakeVaeEncoder(nn.Module):
    def forward(self, target_images: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            (
                target_images.shape[0],
                TARGET_LATENT_TOKEN_COUNT,
                TARGET_LATENT_CHANNELS,
            ),
            dtype=torch.float32,
        )


class ScalarTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def _text_inputs(batch_size: int) -> FluxTextInputs:
    return FluxTextInputs(
        encoder_hidden_states=torch.zeros((batch_size, 2, 4)),
        text_position_ids=torch.zeros((batch_size, 2, 4), dtype=torch.int64),
        text_attention_mask=torch.ones((batch_size, 2), dtype=torch.bool),
    )


def _microbatch(target_count: int) -> FluxMicrobatch:
    return FluxMicrobatch(
        text_inputs=_text_inputs(target_count),
        clean_latents=torch.zeros(
            (target_count, TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS)
        ),
        image_position_ids=torch.zeros(
            (target_count, TARGET_LATENT_TOKEN_COUNT, 4),
            dtype=torch.int64,
        ),
    )


class FixedFlowProbeTests(unittest.TestCase):
    def test_prepares_scoring_targets_separately_from_text_conditioning(self) -> None:
        result = prepare_flow_probe_microbatch(
            _text_inputs(2),
            (((1, 2),), ((3,), (4,))),
            vae_encoder=FakeVaeEncoder(),  # type: ignore[arg-type]
        )

        self.assertEqual(result.target_count, 2)
        self.assertEqual(
            result.clean_latents.shape,
            (2, TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
        )
        self.assertEqual(
            result.image_position_ids.shape,
            (2, TARGET_LATENT_TOKEN_COUNT, 4),
        )

        with self.assertRaisesRegex(ValueError, "batch sizes"):
            prepare_flow_probe_microbatch(
                _text_inputs(2),
                (((1,),),),
                vae_encoder=FakeVaeEncoder(),  # type: ignore[arg-type]
            )

    def test_fixed_seed_repeats_weighted_loss_and_restores_mode(self) -> None:
        transformer = ScalarTransformer().train()

        def sampled_loss(
            model,
            microbatch,
            *,
            compute_dtype,
            generator,
        ):
            self.assertIs(model, transformer)
            self.assertEqual(compute_dtype, torch.bfloat16)
            return torch.rand((), generator=generator) + microbatch.target_count

        with patch(
            "diffusion_arc.training.flow_probes.compute_flux_microbatch_loss",
            side_effect=sampled_loss,
        ):
            first = evaluate_fixed_flow_probe(
                transformer,
                (_microbatch(1), _microbatch(3)),
                compute_dtype=torch.bfloat16,
                seed=167,
            )
            second = evaluate_fixed_flow_probe(
                transformer,
                (_microbatch(1), _microbatch(3)),
                compute_dtype=torch.bfloat16,
                seed=167,
            )

        self.assertEqual(first, second)
        self.assertEqual(first.target_count, 4)
        self.assertEqual(first.microbatch_count, 2)
        self.assertTrue(transformer.training)


if __name__ == "__main__":
    unittest.main()
