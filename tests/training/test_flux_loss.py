"""Tests for one complete FLUX microbatch loss computation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

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


def _microbatch(*, text_dtype: torch.dtype = torch.float32) -> FluxMicrobatch:
    prefix_ends = (2, 4)
    sequence_state = NvarcSequenceState(
        hidden_states=torch.randn(4, 16, dtype=text_dtype),
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


class _DtypeRecordingTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(
            TARGET_LATENT_CHANNELS,
            TARGET_LATENT_CHANNELS,
            bias=False,
            dtype=torch.float32,
        )
        self.context_projection = torch.nn.Linear(
            16,
            TARGET_LATENT_CHANNELS,
            bias=False,
            dtype=torch.float32,
        )
        self.hidden_dtype: torch.dtype | None = None
        self.text_dtype: torch.dtype | None = None
        self.output_dtype: torch.dtype | None = None
        self.text_output_dtype: torch.dtype | None = None
        self.autocast_enabled = False
        self.timestep_device: torch.device | None = None
        self.mask_device: torch.device | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        joint_attention_kwargs: dict,
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.hidden_dtype = hidden_states.dtype
        self.text_dtype = encoder_hidden_states.dtype
        self.autocast_enabled = torch.is_autocast_enabled(hidden_states.device.type)
        self.timestep_device = timestep.device
        self.mask_device = joint_attention_kwargs["attention_mask"].device
        context = self.context_projection(encoder_hidden_states)
        sample = self.projection(hidden_states) + context.mean(dim=1, keepdim=True)
        self.output_dtype = sample.dtype
        self.text_output_dtype = context.dtype
        return SimpleNamespace(sample=sample)


class FluxForwardTests(unittest.TestCase):
    def test_computes_finite_loss_and_transformer_gradients(self) -> None:
        torch.manual_seed(7)
        transformer = _tiny_transformer()
        microbatch = _microbatch()
        generator = torch.Generator().manual_seed(11)

        loss = compute_flux_microbatch_loss(
            transformer,
            microbatch,
            compute_dtype=torch.bfloat16,
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
        self.assertTrue(all(gradient.dtype == torch.float32 for gradient in gradients))

    def test_preserves_inputs_and_autocasts_transformer_compute(self) -> None:
        transformer = _DtypeRecordingTransformer()
        microbatch = _microbatch(text_dtype=torch.bfloat16)
        self.assertEqual(microbatch.clean_latents.dtype, torch.float32)
        self.assertEqual(
            microbatch.text_inputs.encoder_hidden_states.dtype,
            torch.bfloat16,
        )

        loss = compute_flux_microbatch_loss(
            transformer,
            microbatch,
            compute_dtype=torch.bfloat16,
            generator=torch.Generator().manual_seed(19),
        )

        self.assertEqual(transformer.projection.weight.dtype, torch.float32)
        self.assertEqual(transformer.context_projection.weight.dtype, torch.float32)
        self.assertEqual(transformer.hidden_dtype, torch.float32)
        self.assertEqual(transformer.text_dtype, torch.bfloat16)
        self.assertTrue(transformer.autocast_enabled)
        self.assertEqual(transformer.output_dtype, torch.bfloat16)
        self.assertEqual(transformer.text_output_dtype, torch.bfloat16)
        self.assertEqual(
            transformer.timestep_device,
            transformer.projection.weight.device,
        )
        self.assertEqual(
            transformer.mask_device,
            transformer.projection.weight.device,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        self.assertEqual(transformer.projection.weight.grad.dtype, torch.float32)
        self.assertEqual(
            transformer.context_projection.weight.grad.dtype,
            torch.float32,
        )

    def test_rejects_bfloat16_persistent_transformer_parameters(self) -> None:
        transformer = torch.nn.Linear(
            TARGET_LATENT_CHANNELS,
            TARGET_LATENT_CHANNELS,
            bias=False,
            dtype=torch.bfloat16,
        )

        with self.assertRaisesRegex(RuntimeError, "must use float32"):
            compute_flux_microbatch_loss(
                transformer,
                _microbatch(text_dtype=torch.bfloat16),
                compute_dtype=torch.bfloat16,
                generator=torch.Generator().manual_seed(23),
            )


if __name__ == "__main__":
    unittest.main()
