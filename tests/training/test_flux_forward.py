"""Tests for one complete FLUX microbatch loss computation."""

from __future__ import annotations

import unittest

import torch

from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from diffusion_arc.training import (
    FluxInput,
    FluxMicrobatch,
    NvarcSequenceState,
    build_flux_microbatch,
    compute_flux_microbatch_loss,
)


def _tiny_transformer() -> Flux2Transformer2DModel:
    return Flux2Transformer2DModel(
        in_channels=TARGET_LATENT_CHANNELS,
        out_channels=TARGET_LATENT_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )


def _microbatch() -> FluxMicrobatch:
    prefix_ends = (2, 4)
    sequence_state = NvarcSequenceState(
        hidden_states=torch.randn(4, 16),
        pending_prefix_ends=prefix_ends,
    )
    inputs = tuple(
        FluxInput(
            nvarc_sequence_state=sequence_state,
            prefix_end=prefix_end,
            clean_latent=torch.randn(
                TARGET_LATENT_TOKEN_COUNT,
                TARGET_LATENT_CHANNELS,
            ),
        )
        for prefix_end in prefix_ends
    )
    return build_flux_microbatch(inputs)


class FluxForwardTests(unittest.TestCase):
    def test_computes_finite_loss_and_transformer_gradients(self) -> None:
        torch.manual_seed(7)
        transformer = _tiny_transformer()
        microbatch = _microbatch()
        generator = torch.Generator().manual_seed(11)

        loss = compute_flux_microbatch_loss(
            transformer,
            microbatch,
            generator=generator,
        )

        self.assertEqual(loss.shape, torch.Size())
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        gradients = tuple(
            parameter.grad
            for parameter in transformer.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
