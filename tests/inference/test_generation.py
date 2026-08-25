"""Tests for reusable Euler generation and exact-grid diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from diffusion_arc.arc_data import (
    CANVAS_SIZE,
    PAD_CLASS,
    decode_arc_grids,
    render_grid_canvas,
)
from diffusion_arc.inference import (
    diagnose_exact_grids,
    generate_arc_batch,
    linear_timestep_schedule,
)
from diffusion_arc.model import TARGET_LATENT_SHAPE, FluxTextInputs


class ConstantVelocityTransformer(nn.Module):
    def __init__(self, velocity: float = 2.0, *, raise_on_call: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.velocity = velocity
        self.raise_on_call = raise_on_call
        self.timesteps: list[torch.Tensor] = []
        self.training_modes: list[bool] = []
        self.grad_modes: list[bool] = []

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.timesteps.append(timestep.detach().clone())
        self.training_modes.append(self.training)
        self.grad_modes.append(torch.is_grad_enabled())
        if self.raise_on_call:
            raise RuntimeError("intentional transformer failure")
        sample = torch.ones_like(hidden_states) * self.velocity + self.anchor * 0
        return SimpleNamespace(sample=sample)


class PaletteImageDecoder(nn.Module):
    def __init__(self, grid: tuple[tuple[int, ...], ...]) -> None:
        super().__init__()
        self.register_buffer("image", render_grid_canvas(grid))
        self.calls = 0
        self.grad_modes: list[bool] = []
        self.last_latents: torch.Tensor | None = None

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        self.last_latents = latents.detach().clone()
        return self.image.unsqueeze(0).expand(latents.shape[0], -1, -1, -1)


def _text_inputs(batch_size: int = 1) -> FluxTextInputs:
    text_length = 2
    position_ids = torch.zeros((batch_size, text_length, 4), dtype=torch.int64)
    position_ids[:, :, 3] = torch.arange(text_length)
    return FluxTextInputs(
        encoder_hidden_states=torch.zeros((batch_size, text_length, 4)),
        text_position_ids=position_ids,
        text_attention_mask=torch.ones((batch_size, text_length), dtype=torch.bool),
    )


def _class_canvas(grid: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    canvas = torch.full(
        (CANVAS_SIZE, CANVAS_SIZE),
        PAD_CLASS,
        dtype=torch.int64,
    )
    values = torch.tensor(grid, dtype=torch.int64)
    canvas[: values.shape[0], : values.shape[1]] = values
    return canvas


class GenerationTests(unittest.TestCase):
    def test_euler_integrates_seeded_noise_and_decodes_once(self) -> None:
        transformer = ConstantVelocityTransformer(velocity=2.0).train()
        decoder = PaletteImageDecoder(((1, 2, 3), (4, 5, 6)))
        schedule = torch.tensor((1.0, 0.75, 0.0), dtype=torch.float32)
        generator = torch.Generator().manual_seed(13)
        expected_initial = torch.randn(
            (1, *TARGET_LATENT_SHAPE),
            generator=torch.Generator().manual_seed(13),
        )

        result = generate_arc_batch(
            transformer,
            decoder,
            _text_inputs(),
            timestep_schedule=schedule,
            compute_dtype=torch.bfloat16,
            generator=generator,
        )

        torch.testing.assert_close(result.final_latents, expected_initial - 2.0)
        self.assertEqual(len(transformer.timesteps), 2)
        torch.testing.assert_close(transformer.timesteps[0], torch.tensor((1.0,)))
        torch.testing.assert_close(transformer.timesteps[1], torch.tensor((0.75,)))
        self.assertEqual(transformer.training_modes, [False, False])
        self.assertEqual(transformer.grad_modes, [False, False])
        self.assertTrue(transformer.training)

        self.assertEqual(decoder.calls, 1)
        self.assertEqual(decoder.grad_modes, [False])
        torch.testing.assert_close(decoder.last_latents, expected_initial - 2.0)
        self.assertEqual(result.rgb_images.dtype, torch.uint8)
        self.assertEqual(result.rgb_images.device, torch.device("cpu"))
        self.assertEqual(result.rgb_images.shape, (1, 3, 480, 480))
        self.assertFalse(torch.is_inference(result.final_latents))
        self.assertFalse(torch.is_inference(result.rgb_images))
        self.assertFalse(torch.is_inference(result.class_canvases))
        self.assertEqual(result.class_canvases.shape, (1, 30, 30))
        self.assertTrue(result.predictions[0].valid)
        self.assertEqual(result.predictions[0].grid, ((1, 2, 3), (4, 5, 6)))

    def test_identical_generation_seeds_are_reproducible(self) -> None:
        transformer = ConstantVelocityTransformer(velocity=0.0)
        decoder = PaletteImageDecoder(((0,),))
        arguments = {
            "timestep_schedule": linear_timestep_schedule(2),
            "compute_dtype": torch.bfloat16,
        }

        first = generate_arc_batch(
            transformer,
            decoder,
            _text_inputs(),
            generator=torch.Generator().manual_seed(167),
            **arguments,
        )
        second = generate_arc_batch(
            transformer,
            decoder,
            _text_inputs(),
            generator=torch.Generator().manual_seed(167),
            **arguments,
        )

        torch.testing.assert_close(first.final_latents, second.final_latents)
        torch.testing.assert_close(first.rgb_images, second.rgb_images)
        self.assertEqual(first.predictions[0].grid, second.predictions[0].grid)

    def test_requires_strict_one_to_zero_schedule(self) -> None:
        torch.testing.assert_close(
            linear_timestep_schedule(4),
            torch.tensor((1.0, 0.75, 0.5, 0.25, 0.0)),
        )
        transformer = ConstantVelocityTransformer()
        decoder = PaletteImageDecoder(((0,),))
        invalid_schedules = (
            (0.9, 0.0),
            (1.0, 0.1),
            (1.0,),
            (1.0, 0.5, 0.5, 0.0),
            (1.0, float("nan"), 0.0),
            torch.tensor(((1.0, 0.0),)),
        )

        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                generate_arc_batch(
                    transformer,
                    decoder,
                    _text_inputs(),
                    timestep_schedule=schedule,
                    compute_dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(1),
                )

        self.assertEqual(transformer.timesteps, [])
        self.assertEqual(decoder.calls, 0)

    def test_restores_training_mode_after_generation_failure(self) -> None:
        transformer = ConstantVelocityTransformer(raise_on_call=True).train()

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            generate_arc_batch(
                transformer,
                PaletteImageDecoder(((0,),)),
                _text_inputs(),
                timestep_schedule=(1.0, 0.0),
                compute_dtype=torch.bfloat16,
                generator=torch.Generator().manual_seed(3),
            )

        self.assertTrue(transformer.training)

    def test_exact_grid_diagnostics_distinguish_all_failure_modes(self) -> None:
        exact = _class_canvas(((1, 2), (3, 4)))
        wrong_cell = _class_canvas(((1, 2), (3, 0)))
        wrong_shape = _class_canvas(((1, 2),))
        all_pad = torch.full_like(exact, PAD_CLASS)
        predictions = decode_arc_grids(
            torch.stack((exact, wrong_cell, wrong_shape, all_pad))
        )
        expected = ((1, 2), (3, 4))

        diagnostics = diagnose_exact_grids(
            predictions,
            (expected, expected, expected, expected),
        )

        self.assertTrue(diagnostics[0].exact_grid)
        self.assertEqual(diagnostics[0].cell_accuracy, 1.0)
        self.assertTrue(diagnostics[1].shape_match)
        self.assertEqual(diagnostics[1].cell_accuracy, 0.75)
        self.assertFalse(diagnostics[1].exact_grid)
        self.assertEqual(diagnostics[2].predicted_shape, (1, 2))
        self.assertFalse(diagnostics[2].shape_match)
        self.assertIsNone(diagnostics[2].cell_accuracy)
        self.assertFalse(diagnostics[3].prediction_valid)
        self.assertEqual(diagnostics[3].invalid_reason, "all_pad")
        self.assertIsNone(diagnostics[3].predicted_shape)
        self.assertFalse(diagnostics[3].exact_grid)


if __name__ == "__main__":
    unittest.main()
