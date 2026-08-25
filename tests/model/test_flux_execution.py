"""Tests for the shared FLUX velocity execution boundary."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_HEIGHT,
    TARGET_LATENT_SHAPE,
    TARGET_LATENT_TOKEN_COUNT,
    TARGET_LATENT_WIDTH,
    FluxSpatialShape,
    FluxTextInputs,
    build_flux_image_position_ids,
    predict_flux_velocity,
)


class RecordingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(
            TARGET_LATENT_CHANNELS,
            TARGET_LATENT_CHANNELS,
            bias=False,
            dtype=torch.float32,
        )
        self.hidden_dtype: torch.dtype | None = None
        self.autocast_enabled = False
        self.timesteps: torch.Tensor | None = None
        self.image_ids: torch.Tensor | None = None
        self.text_ids: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        joint_attention_kwargs: dict[str, torch.Tensor],
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.hidden_dtype = hidden_states.dtype
        self.autocast_enabled = torch.is_autocast_enabled(hidden_states.device.type)
        self.timesteps = timestep.detach().clone()
        self.image_ids = img_ids.detach().clone()
        self.text_ids = txt_ids.detach().clone()
        self.attention_mask = joint_attention_kwargs["attention_mask"].detach().clone()
        return SimpleNamespace(sample=self.projection(hidden_states))


def _text_inputs(batch_size: int = 2) -> FluxTextInputs:
    text_length = 3
    mask = torch.tensor(
        ((True, True, False), (True, True, True)),
        dtype=torch.bool,
    )[:batch_size]
    position_ids = torch.zeros((batch_size, text_length, 4), dtype=torch.int64)
    position_ids[:, :, 3] = torch.arange(text_length)
    return FluxTextInputs(
        encoder_hidden_states=torch.randn(batch_size, text_length, 5),
        text_position_ids=position_ids,
        text_attention_mask=mask,
    )


class FluxExecutionTests(unittest.TestCase):
    def test_builds_fixed_arc_image_coordinates(self) -> None:
        position_ids = build_flux_image_position_ids(2, torch.device("cpu"))

        self.assertEqual(position_ids.shape, (2, TARGET_LATENT_TOKEN_COUNT, 4))
        self.assertEqual(position_ids.dtype, torch.int64)
        torch.testing.assert_close(position_ids[0], position_ids[1])
        self.assertEqual(tuple(position_ids[0, 0].tolist()), (0, 0, 0, 0))
        self.assertEqual(
            tuple(position_ids[0, TARGET_LATENT_WIDTH - 1].tolist()),
            (0, 0, TARGET_LATENT_WIDTH - 1, 0),
        )
        self.assertEqual(
            tuple(position_ids[0, -1].tolist()),
            (0, TARGET_LATENT_HEIGHT - 1, TARGET_LATENT_WIDTH - 1, 0),
        )

    def test_runs_autocast_velocity_with_joint_attention_mask(self) -> None:
        transformer = RecordingTransformer()
        latents = torch.randn(2, *TARGET_LATENT_SHAPE)
        text_inputs = _text_inputs()
        image_ids = build_flux_image_position_ids(2, torch.device("cpu"))
        timesteps = torch.tensor((1.0, 0.25), dtype=torch.float32)

        result = predict_flux_velocity(
            transformer,
            latents,
            text_inputs,
            image_ids,
            timesteps,
            compute_dtype=torch.bfloat16,
        )

        self.assertEqual(result.shape, latents.shape)
        self.assertEqual(transformer.hidden_dtype, torch.float32)
        self.assertTrue(transformer.autocast_enabled)
        torch.testing.assert_close(transformer.timesteps, timesteps)
        torch.testing.assert_close(transformer.image_ids, image_ids)
        torch.testing.assert_close(
            transformer.text_ids,
            text_inputs.text_position_ids,
        )
        self.assertEqual(
            transformer.attention_mask.shape,
            (2, 1, 1, 3 + TARGET_LATENT_TOKEN_COUNT),
        )
        torch.testing.assert_close(
            transformer.attention_mask[:, 0, 0, :3],
            text_inputs.text_attention_mask,
        )
        self.assertTrue(
            bool(transformer.attention_mask[:, 0, 0, 3:].all().item())
        )

    def test_runs_compact_nine_token_grid(self) -> None:
        transformer = RecordingTransformer()
        spatial_shape = FluxSpatialShape(height=3, width=3)
        latents = torch.randn(2, spatial_shape.token_count, TARGET_LATENT_CHANNELS)
        image_ids = build_flux_image_position_ids(
            2,
            torch.device("cpu"),
            spatial_shape=spatial_shape,
        )

        result = predict_flux_velocity(
            transformer,
            latents,
            _text_inputs(),
            image_ids,
            torch.tensor((1.0, 0.25)),
            compute_dtype=torch.bfloat16,
        )

        self.assertEqual(result.shape, latents.shape)
        self.assertEqual(image_ids.shape, (2, 9, 4))
        self.assertEqual(tuple(image_ids[0, -1].tolist()), (0, 2, 2, 0))
        self.assertEqual(transformer.attention_mask.shape, (2, 1, 1, 12))

    def test_rejects_wrong_image_token_shape_before_transformer_work(self) -> None:
        transformer = RecordingTransformer()
        with self.assertRaisesRegex(ValueError, "image_position_ids"):
            predict_flux_velocity(
                transformer,
                torch.zeros((1, *TARGET_LATENT_SHAPE)),
                _text_inputs(batch_size=1),
                torch.zeros(
                    (1, TARGET_LATENT_TOKEN_COUNT - 1, 4),
                    dtype=torch.int64,
                ),
                torch.ones(1),
                compute_dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
