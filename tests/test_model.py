from __future__ import annotations

import types
import unittest

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diffusion_arc.nvarc_conditioning import NVARCConditioner
from diffusion_arc.grid_codec import render_arc_canvas
from diffusion_arc.model import (
    ARC_IMAGE_TOKENS,
    ARC_LATENT_CHANNELS,
    DiffusionARCModel,
)


class FakeNVARC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = 0

    def get_input_embeddings(self):
        return self

    @property
    def weight(self):
        return self.scale

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        self.calls += 1
        self.assert_contract(
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        batch, length = input_ids.shape
        hidden_states = tuple(
            torch.full(
                (batch, length, 2560),
                float(index),
                device=input_ids.device,
                dtype=self.scale.dtype,
            )
            * self.scale
            for index in range(28)
        )
        return types.SimpleNamespace(hidden_states=hidden_states)

    @staticmethod
    def assert_contract(*, output_hidden_states, use_cache, return_dict):
        assert output_hidden_states
        assert not use_cache
        assert return_dict


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode(self, images: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool2d(images.mean(dim=1, keepdim=True), (30, 30))
        return pooled.expand(images.shape[0], ARC_LATENT_CHANNELS, 30, 30) * self.scale

    def decode(self, latents: Tensor) -> Tensor:
        images = render_arc_canvas([[[0, 1], [2, 3]]])
        return images.to(device=latents.device, dtype=latents.dtype).expand(
            latents.shape[0], -1, -1, -1
        )


class FakeFlux(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def forward(self, *, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        self.calls += 1
        self.last_shapes = {
            "x": tuple(x.shape),
            "x_ids": tuple(x_ids.shape),
            "timesteps": tuple(timesteps.shape),
            "ctx": tuple(ctx.shape),
            "ctx_ids": tuple(ctx_ids.shape),
        }
        if guidance is not None:
            raise AssertionError("The initial path must not pass guidance.")
        return x * self.scale


def make_model() -> DiffusionARCModel:
    conditioner = NVARCConditioner(FakeNVARC())
    return DiffusionARCModel(
        conditioner=conditioner,
        flux=FakeFlux(),
        vae=FakeVAE(),
    )


class DiffusionARCModelTest(unittest.TestCase):
    def setUp(self):
        self.input_ids = torch.tensor([[11, 0, 15]], dtype=torch.int64)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.target = [[[0, 1], [2, 3]]]

    def test_forward_backward_reaches_flux_only(self):
        model = make_model().train()
        noise = torch.zeros((1, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS))
        timesteps = torch.tensor([0.5])

        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            target_grids=self.target,
            noise=noise,
            timesteps=timesteps,
            return_details=True,
        )
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.predicted_velocity.shape, noise.shape)
        self.assertEqual(model.flux.last_shapes["ctx"], (1, 3, 7680))
        self.assertEqual(model.flux.last_shapes["x_ids"], (1, 900, 4))
        self.assertEqual(model.flux.last_shapes["ctx_ids"], (1, 3, 4))
        self.assertIsNotNone(model.flux.scale.grad)
        self.assertTrue(torch.isfinite(model.flux.scale.grad))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.nvarc.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.vae.parameters())
        )

    def test_train_keeps_nvarc_and_vae_in_eval_mode(self):
        model = make_model()

        model.train()

        self.assertTrue(model.flux.training)
        self.assertFalse(model.conditioner.training)
        self.assertFalse(model.nvarc.training)
        self.assertFalse(model.vae.training)
        model.assert_parameter_boundaries()

    def test_generate_conditions_once_and_strictly_decodes(self):
        model = make_model().eval()
        noise = torch.zeros((1, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS))

        output = model.generate(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            num_steps=2,
            noise=noise,
        )

        self.assertEqual(model.nvarc.calls, 1)
        self.assertEqual(model.flux.calls, 2)
        self.assertEqual(output.latents.shape, noise.shape)
        self.assertEqual(output.images.shape, (1, 3, 480, 480))
        self.assertTrue(output.decodes[0].valid)
        torch.testing.assert_close(
            output.decodes[0].grid,
            torch.tensor([[0, 1], [2, 3]]),
        )

    def test_predict_velocity_rejects_integer_timesteps(self):
        model = make_model()
        latents = torch.zeros((1, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS))
        context = torch.zeros((1, 2, 7680))

        with self.assertRaisesRegex(TypeError, "floating-point"):
            model.predict_velocity(
                latents=latents,
                timesteps=torch.tensor([1]),
                context=context,
            )

    def test_predict_velocity_rejects_non_integer_position_ids(self):
        model = make_model()
        latents = torch.zeros((1, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS))
        context = torch.zeros((1, 2, 7680))

        with self.assertRaisesRegex(TypeError, "torch.int64"):
            model.predict_velocity(
                latents=latents,
                timesteps=torch.tensor([0.5]),
                context=context,
                image_ids=torch.zeros((1, ARC_IMAGE_TOKENS, 4)),
                text_ids=torch.zeros((1, 2, 4)),
            )


if __name__ == "__main__":
    unittest.main()
