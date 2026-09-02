"""Tests for frozen FLUX.2 VAE encoding and decoding."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import Tensor, nn

from diffusion_arc.model import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    TARGET_IMAGE_SHAPE,
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_HEIGHT,
    TARGET_LATENT_SHAPE,
    TARGET_LATENT_TOKEN_COUNT,
    TARGET_LATENT_WIDTH,
    FluxVaeDecoder,
    FluxVaeEncoder,
    FluxSpatialShape,
)


COMPACT_SPATIAL_SHAPE = FluxSpatialShape(height=3, width=3)


class _FakePosterior:
    def __init__(self, vae: FakeFluxVae, latents: Tensor) -> None:
        self._vae = vae
        self._latents = latents

    def mode(self) -> Tensor:
        self._vae.posterior_mode_calls += 1
        return self._latents


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

        with torch.no_grad():
            self.bn.running_mean.copy_(
                torch.linspace(-0.5, 0.5, TARGET_LATENT_CHANNELS, dtype=dtype)
            )
            self.bn.running_var.copy_(
                torch.linspace(0.5, 2.0, TARGET_LATENT_CHANNELS, dtype=dtype)
            )

        self.encode_calls = 0
        self.decode_calls = 0
        self.posterior_mode_calls = 0
        self.seen_training_modes: list[bool] = []
        self.seen_grad_modes: list[bool] = []
        self.decode_training_modes: list[bool] = []
        self.decode_grad_modes: list[bool] = []
        self.last_images: Tensor | None = None
        self.last_encoded_latents: Tensor | None = None
        self.last_decode_latents: Tensor | None = None

    def encode(self, images: Tensor) -> SimpleNamespace:
        self.encode_calls += 1
        self.seen_training_modes.append(self.training)
        self.seen_grad_modes.append(torch.is_grad_enabled())
        self.last_images = images.detach().clone()
        latent_height = images.shape[2] // 8
        latent_width = images.shape[3] // 8
        values = torch.arange(
            images.shape[0] * 32 * latent_height * latent_width,
            dtype=images.dtype,
            device=images.device,
        ).reshape(images.shape[0], 32, latent_height, latent_width)
        self.last_encoded_latents = values.div(1000.0)
        return SimpleNamespace(
            latent_dist=_FakePosterior(self, self.last_encoded_latents),
        )

    def decode(self, latents: Tensor) -> SimpleNamespace:
        self.decode_calls += 1
        self.decode_training_modes.append(self.training)
        self.decode_grad_modes.append(torch.is_grad_enabled())
        self.last_decode_latents = latents.detach().clone()
        values = latents.mean(dim=(1, 2, 3), keepdim=True)
        images = values.expand(
            latents.shape[0],
            3,
            latents.shape[2] * 8,
            latents.shape[3] * 8,
        )
        return SimpleNamespace(sample=images)


def _target_images(
    batch_size: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> Tensor:
    return torch.zeros(
        (
            batch_size,
            3,
            spatial_shape.image_height,
            spatial_shape.image_width,
        ),
        dtype=dtype,
    )


def _reference_clean_latents(vae: FakeFluxVae, batch_size: int) -> Tensor:
    latents = vae.last_encoded_latents
    if latents is None:
        raise AssertionError("fake VAE has not encoded images")
    latents = latents[:batch_size]
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
        packed.shape[2] * packed.shape[3],
    ).transpose(1, 2).contiguous()


def _reference_vae_decode_latents(
    vae: FakeFluxVae,
    packed_latents: Tensor,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> Tensor:
    packed_grid = packed_latents.to(dtype=vae.anchor.dtype).transpose(1, 2).reshape(
        packed_latents.shape[0],
        TARGET_LATENT_CHANNELS,
        spatial_shape.height,
        spatial_shape.width,
    )
    mean = vae.bn.running_mean.reshape(1, -1, 1, 1)
    std = torch.sqrt(
        vae.bn.running_var.reshape(1, -1, 1, 1)
        + vae.config.batch_norm_eps
    )
    denormalized = packed_grid * std + mean
    unpatched = denormalized.reshape(
        packed_latents.shape[0],
        32,
        2,
        2,
        spatial_shape.height,
        spatial_shape.width,
    ).permute(0, 1, 4, 2, 5, 3)
    return unpatched.reshape(
        packed_latents.shape[0],
        32,
        spatial_shape.height * 2,
        spatial_shape.width * 2,
    )


class FluxVaeEncoderTests(unittest.TestCase):
    def test_fixed_target_latent_contract_constants(self) -> None:
        self.assertEqual(TARGET_LATENT_CHANNELS, 128)
        self.assertEqual(TARGET_LATENT_HEIGHT, 30)
        self.assertEqual(TARGET_LATENT_WIDTH, 30)
        self.assertEqual(TARGET_LATENT_TOKEN_COUNT, 900)
        self.assertEqual(TARGET_LATENT_SHAPE, (900, 128))

    def test_patches_normalizes_and_flattens_latents(self) -> None:
        vae = FakeFluxVae(batch_size=2)
        encoder = FluxVaeEncoder(vae)

        result = encoder(_target_images(batch_size=2))

        self.assertIsInstance(result, Tensor)
        self.assertEqual(result.shape, (2, *TARGET_LATENT_SHAPE))
        torch.testing.assert_close(
            result,
            _reference_clean_latents(vae, batch_size=2),
        )

    def test_encodes_explicit_compact_spatial_shape(self) -> None:
        vae = FakeFluxVae(batch_size=2)
        encoder = FluxVaeEncoder(vae)

        result = encoder(
            _target_images(batch_size=2, spatial_shape=COMPACT_SPATIAL_SHAPE),
            spatial_shape=COMPACT_SPATIAL_SHAPE,
        )

        self.assertEqual(result.shape, (2, 9, TARGET_LATENT_CHANNELS))
        torch.testing.assert_close(
            result,
            _reference_clean_latents(vae, batch_size=2),
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
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)

    def test_casts_images_to_vae_dtype(self) -> None:
        vae = FakeFluxVae(dtype=torch.float64)
        encoder = FluxVaeEncoder(vae)

        result = encoder(_target_images(dtype=torch.float32))

        self.assertIsNotNone(vae.last_images)
        self.assertEqual(vae.last_images.dtype, torch.float64)
        self.assertEqual(result.dtype, torch.float64)

    def test_latents_support_downstream_parameter_gradients(self) -> None:
        vae = FakeFluxVae()
        result = FluxVaeEncoder(vae)(_target_images())
        projection = nn.Linear(TARGET_LATENT_CHANNELS, 1)

        projection(result).square().mean().backward()

        self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in vae.parameters()))

    def test_rejects_incompatible_vae_interface(self) -> None:
        with self.assertRaises(TypeError):
            FluxVaeEncoder(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "encode"):
            FluxVaeEncoder(nn.Identity())


class FluxVaeDecoderTests(unittest.TestCase):
    def test_denormalizes_unpatches_and_decodes_once(self) -> None:
        vae = FakeFluxVae(batch_size=1, dtype=torch.float64)
        decoder = FluxVaeDecoder(vae)
        packed_latents = torch.linspace(
            -1.0,
            1.0,
            steps=TARGET_LATENT_TOKEN_COUNT * TARGET_LATENT_CHANNELS,
            dtype=torch.float32,
        ).reshape(1, *TARGET_LATENT_SHAPE)

        result = decoder(packed_latents)

        self.assertEqual(vae.decode_calls, 1)
        self.assertIsNotNone(vae.last_decode_latents)
        torch.testing.assert_close(
            vae.last_decode_latents,
            _reference_vae_decode_latents(vae, packed_latents),
        )
        self.assertEqual(result.shape, (1, *TARGET_IMAGE_SHAPE))
        self.assertEqual(result.dtype, torch.float64)

    def test_decodes_explicit_compact_spatial_shape(self) -> None:
        vae = FakeFluxVae(batch_size=1, dtype=torch.float64)
        decoder = FluxVaeDecoder(vae)
        packed_latents = torch.linspace(
            -1.0,
            1.0,
            steps=COMPACT_SPATIAL_SHAPE.token_count * TARGET_LATENT_CHANNELS,
            dtype=torch.float32,
        ).reshape(1, COMPACT_SPATIAL_SHAPE.token_count, TARGET_LATENT_CHANNELS)

        result = decoder(
            packed_latents,
            spatial_shape=COMPACT_SPATIAL_SHAPE,
        )

        torch.testing.assert_close(
            vae.last_decode_latents,
            _reference_vae_decode_latents(
                vae,
                packed_latents,
                COMPACT_SPATIAL_SHAPE,
            ),
        )
        self.assertEqual(result.shape, (1, 3, 48, 48))

    def test_freezes_vae_decodes_without_grad_and_stays_in_eval_mode(self) -> None:
        vae = FakeFluxVae().train()
        decoder = FluxVaeDecoder(vae)
        packed_latents = torch.zeros(
            (1, *TARGET_LATENT_SHAPE),
            requires_grad=True,
        )

        decoder.train(True)
        result = decoder(packed_latents)

        self.assertFalse(decoder.training)
        self.assertFalse(vae.training)
        self.assertFalse(any(parameter.requires_grad for parameter in vae.parameters()))
        self.assertEqual(vae.decode_training_modes, [False])
        self.assertEqual(vae.decode_grad_modes, [False])
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)

    def test_rejects_invalid_latents_and_decoder_interfaces(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "decode"):
            FluxVaeDecoder(nn.Identity())
        with self.assertRaisesRegex(ValueError, "packed_latents"):
            FluxVaeDecoder(FakeFluxVae())(
                torch.zeros((1, TARGET_LATENT_TOKEN_COUNT - 1, TARGET_LATENT_CHANNELS))
            )


if __name__ == "__main__":
    unittest.main()
