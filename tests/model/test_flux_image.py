"""Tests for frozen FLUX.2 VAE target-image encoding."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from diffusion_arc.model import (
    TARGET_IMAGE_SHAPE,
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_HEIGHT,
    TARGET_LATENT_SHAPE,
    TARGET_LATENT_TOKEN_COUNT,
    TARGET_LATENT_WIDTH,
    EncodedTargetLatents,
    FluxVaeEncoder,
)


class _FakePosterior:
    def __init__(self, vae: FakeFluxVae, batch_size: int) -> None:
        self._vae = vae
        self._batch_size = batch_size

    def mode(self) -> Tensor:
        self._vae.posterior_mode_calls += 1
        return self._vae.encoded_latents[: self._batch_size]


class FakeFluxVae(nn.Module):
    def __init__(
        self,
        *,
        batch_size: int = 2,
        dtype: torch.dtype = torch.float32,
        batch_norm_eps: float = 0.25,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.bn = nn.BatchNorm2d(
            TARGET_LATENT_CHANNELS,
            eps=batch_norm_eps,
            affine=False,
        ).to(dtype=dtype)
        self.config = SimpleNamespace(batch_norm_eps=batch_norm_eps)

        values = torch.arange(
            batch_size * 32 * 60 * 60,
            dtype=dtype,
        ).reshape(batch_size, 32, 60, 60)
        self.register_buffer("encoded_latents", values.div(1000.0))
        with torch.no_grad():
            self.bn.running_mean.copy_(
                torch.linspace(-0.5, 0.5, TARGET_LATENT_CHANNELS, dtype=dtype)
            )
            self.bn.running_var.copy_(
                torch.linspace(0.5, 2.0, TARGET_LATENT_CHANNELS, dtype=dtype)
            )

        self.encode_calls = 0
        self.posterior_mode_calls = 0
        self.seen_training_modes: list[bool] = []
        self.seen_grad_modes: list[bool] = []
        self.last_images: Tensor | None = None

    def encode(self, images: Tensor) -> SimpleNamespace:
        self.encode_calls += 1
        self.seen_training_modes.append(self.training)
        self.seen_grad_modes.append(torch.is_grad_enabled())
        self.last_images = images.detach().clone()
        return SimpleNamespace(
            latent_dist=_FakePosterior(self, images.shape[0]),
        )


def _target_images(
    batch_size: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return torch.zeros((batch_size, *TARGET_IMAGE_SHAPE), dtype=dtype)


def _reference_clean_latents(vae: FakeFluxVae, batch_size: int) -> Tensor:
    latents = vae.encoded_latents[:batch_size]
    packed = torch.stack(
        tuple(
            latents[:, channel, row_offset::2, column_offset::2]
            for channel in range(32)
            for row_offset in range(2)
            for column_offset in range(2)
        ),
        dim=1,
    )
    mean = vae.bn.running_mean.reshape(1, -1, 1, 1)
    std = torch.sqrt(
        vae.bn.running_var.reshape(1, -1, 1, 1)
        + vae.config.batch_norm_eps
    )
    normalized = (packed - mean) / std
    return normalized.reshape(
        batch_size,
        TARGET_LATENT_CHANNELS,
        TARGET_LATENT_TOKEN_COUNT,
    ).transpose(1, 2).contiguous()


class FluxVaeEncoderTests(unittest.TestCase):
    def test_fixed_target_latent_contract_constants(self) -> None:
        self.assertEqual(TARGET_LATENT_CHANNELS, 128)
        self.assertEqual(TARGET_LATENT_HEIGHT, 30)
        self.assertEqual(TARGET_LATENT_WIDTH, 30)
        self.assertEqual(TARGET_LATENT_TOKEN_COUNT, 900)
        self.assertEqual(TARGET_LATENT_SHAPE, (900, 128))

    def test_patches_normalizes_flattens_and_builds_position_ids(self) -> None:
        vae = FakeFluxVae(batch_size=2)
        encoder = FluxVaeEncoder(vae)

        result = encoder(_target_images(batch_size=2))

        self.assertIsInstance(result, EncodedTargetLatents)
        self.assertEqual(result.clean_latents.shape, (2, *TARGET_LATENT_SHAPE))
        torch.testing.assert_close(
            result.clean_latents,
            _reference_clean_latents(vae, batch_size=2),
        )
        self.assertEqual(result.image_position_ids.shape, (2, 900, 4))
        self.assertEqual(result.image_position_ids.dtype, torch.int64)
        torch.testing.assert_close(
            result.image_position_ids[0, (0, 1, 29, 30, 899), :],
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
            result.image_position_ids[0],
            result.image_position_ids[1],
        )

    def test_freezes_vae_and_uses_posterior_mode_without_grad(self) -> None:
        vae = FakeFluxVae().train()
        encoder = FluxVaeEncoder(vae)

        self.assertFalse(encoder.training)
        self.assertFalse(vae.training)
        self.assertFalse(any(parameter.requires_grad for parameter in vae.parameters()))

        encoder.train(True)
        result = encoder(_target_images())

        self.assertFalse(encoder.training)
        self.assertFalse(vae.training)
        self.assertEqual(vae.encode_calls, 1)
        self.assertEqual(vae.posterior_mode_calls, 1)
        self.assertEqual(vae.seen_training_modes, [False])
        self.assertEqual(vae.seen_grad_modes, [False])
        self.assertFalse(result.clean_latents.requires_grad)
        self.assertIsNone(result.clean_latents.grad_fn)

    def test_casts_images_to_vae_dtype(self) -> None:
        vae = FakeFluxVae(dtype=torch.float64)
        encoder = FluxVaeEncoder(vae)

        result = encoder(_target_images(dtype=torch.float32))

        self.assertIsNotNone(vae.last_images)
        self.assertEqual(vae.last_images.dtype, torch.float64)
        self.assertEqual(result.clean_latents.dtype, torch.float64)

    def test_latents_support_downstream_parameter_gradients(self) -> None:
        vae = FakeFluxVae()
        result = FluxVaeEncoder(vae)(_target_images())
        projection = nn.Linear(TARGET_LATENT_CHANNELS, 1)

        projection(result.clean_latents).square().mean().backward()

        self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))

    def test_output_contract_is_frozen(self) -> None:
        result = EncodedTargetLatents(
            clean_latents=torch.zeros((1, *TARGET_LATENT_SHAPE)),
            image_position_ids=torch.zeros((1, 900, 4), dtype=torch.int64),
        )

        with self.assertRaises(FrozenInstanceError):
            result.clean_latents = torch.ones(1)  # type: ignore[misc]

    def test_rejects_incompatible_vae_interface(self) -> None:
        with self.assertRaises(TypeError):
            FluxVaeEncoder(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "encode"):
            FluxVaeEncoder(nn.Identity())


if __name__ == "__main__":
    unittest.main()
