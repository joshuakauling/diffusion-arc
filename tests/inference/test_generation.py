"""Tests for target-free padded and packed Euler generation."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
import unittest

import torch
from torch import nn

from diffusion_arc.arc_data import (
    render_grid_canvas,
)
from diffusion_arc.inference import generation
from diffusion_arc.inference.conditioning import PackedGenerationInput, sample_conditioning_noise
from diffusion_arc.inference import (
    continue_arc_batch,
    generate_arc_batch,
    linear_timestep_schedule,
)
from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_SHAPE,
    FluxSpatialShape,
    FluxTextInputs,
)


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
        self.grid = grid
        self.calls = 0
        self.grad_modes: list[bool] = []
        self.last_latents: torch.Tensor | None = None

    def forward(
        self,
        latents: torch.Tensor,
        *,
        spatial_shape: FluxSpatialShape,
    ) -> torch.Tensor:
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        self.last_latents = latents.detach().clone()
        image = render_grid_canvas(
            self.grid,
            canvas_size=spatial_shape.height,
        ).to(device=latents.device)
        return image.unsqueeze(0).expand(latents.shape[0], -1, -1, -1)


def _text_inputs(batch_size: int = 1) -> FluxTextInputs:
    text_length = 2
    position_ids = torch.zeros((batch_size, text_length, 4), dtype=torch.int64)
    position_ids[:, :, 3] = torch.arange(text_length)
    return FluxTextInputs(
        encoder_hidden_states=torch.zeros((batch_size, text_length, 4)),
        text_position_ids=position_ids,
        text_attention_mask=torch.ones((batch_size, text_length), dtype=torch.bool),
    )


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

    def test_continues_from_explicit_latents_at_partial_timestep(self) -> None:
        transformer = ConstantVelocityTransformer(velocity=2.0).train()
        decoder = PaletteImageDecoder(((1, 2), (3, 4)))
        initial_latents = torch.full(
            (2, *TARGET_LATENT_SHAPE),
            3.0,
            dtype=torch.float32,
        )

        result = continue_arc_batch(
            transformer,
            decoder,
            _text_inputs(batch_size=2),
            initial_latents=initial_latents,
            timestep_schedule=(0.5, 0.25, 0.0),
            compute_dtype=torch.bfloat16,
        )

        torch.testing.assert_close(
            result.final_latents,
            torch.full_like(result.final_latents, 2.0),
        )
        torch.testing.assert_close(
            result.timestep_schedule,
            torch.tensor((0.5, 0.25, 0.0)),
        )
        self.assertEqual(len(transformer.timesteps), 2)
        torch.testing.assert_close(
            transformer.timesteps[0],
            torch.tensor((0.5, 0.5)),
        )
        self.assertTrue(transformer.training)
        self.assertEqual(decoder.calls, 1)

    def test_partial_generation_validates_latents_and_schedule(self) -> None:
        transformer = ConstantVelocityTransformer()
        decoder = PaletteImageDecoder(((0,),))
        valid_latents = torch.zeros((1, *TARGET_LATENT_SHAPE))
        invalid_cases = (
            {"initial_latents": valid_latents, "timestep_schedule": (0.0, 0.0)},
            {"initial_latents": valid_latents, "timestep_schedule": (1.1, 0.0)},
            {"initial_latents": valid_latents, "timestep_schedule": (0.5, 0.1)},
            {
                "initial_latents": torch.zeros((2, *TARGET_LATENT_SHAPE)),
                "timestep_schedule": (0.5, 0.0),
            },
            {
                "initial_latents": torch.zeros(
                    (1, TARGET_LATENT_SHAPE[0], TARGET_LATENT_CHANNELS - 1)
                ),
                "timestep_schedule": (0.5, 0.0),
            },
        )

        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                continue_arc_batch(
                    transformer,
                    decoder,
                    _text_inputs(),
                    compute_dtype=torch.bfloat16,
                    **case,
                )

        with self.assertRaises(TypeError):
            continue_arc_batch(
                transformer,
                decoder,
                _text_inputs(),
                initial_latents=torch.zeros(
                    (1, *TARGET_LATENT_SHAPE), dtype=torch.int64
                ),
                timestep_schedule=(0.5, 0.0),
                compute_dtype=torch.bfloat16,
            )

    def test_generates_known_full_compact_grid_without_pad_inference(self) -> None:
        spatial_shape = FluxSpatialShape(height=3, width=3)
        grid = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        transformer = ConstantVelocityTransformer(velocity=0.0)
        decoder = PaletteImageDecoder(grid)

        result = generate_arc_batch(
            transformer,
            decoder,
            _text_inputs(),
            timestep_schedule=(1.0, 0.0),
            compute_dtype=torch.bfloat16,
            generator=torch.Generator().manual_seed(17),
            spatial_shape=spatial_shape,
            infer_padding_boundary=False,
        )

        self.assertEqual(
            result.final_latents.shape,
            (1, 9, TARGET_LATENT_CHANNELS),
        )
        self.assertEqual(result.rgb_images.shape, (1, 3, 48, 48))
        self.assertEqual(result.class_canvases.shape, (1, 3, 3))
        self.assertTrue(result.predictions[0].valid)
        self.assertEqual(result.predictions[0].grid, grid)

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


def _packed_input(index=0, *, task_id=None, query_index=0, length=2, value=0.0):
    return PackedGenerationInput(
        task_id=task_id or f"task-{index}", query_index=query_index,
        text=torch.full((length, 7680), value, dtype=torch.bfloat16),
    )


class _PackedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))


class _PackedDecoder(nn.Module):
    def __init__(self, grid=((0, 0, 0),) * 3):
        super().__init__()
        self.image = render_grid_canvas(grid)
        self.states = []

    def forward(self, latents, *, spatial_shape):
        assert not torch.is_grad_enabled()
        assert (spatial_shape.height, spatial_shape.width) == (30, 30)
        self.states.append(latents.detach().clone())
        return self.image.unsqueeze(0).expand(latents.shape[0], -1, -1, -1)


class PackedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.transformer = _PackedTransformer().train()
        self.arguments = {"compute_dtype": torch.float32, "backend": object()}

    def generate(self, inputs, packs, decoder=None, **overrides):
        arguments = {"steps": 4, "seeds": (50, 51), **self.arguments, **overrides}
        return generation.generate_packed_arc_samples(
            self.transformer, decoder or _PackedDecoder(), inputs, packs, **arguments,
        )

    def test_real_packed_transformer_matches_singleton_euler_trajectories(self):
        from diffusion_arc.inference.generation import continue_arc_batch, linear_timestep_schedule
        from diffusion_arc.model.flux.conditioning import pad_flux_text_prefixes
        from tests.model.test_flux_packing import SegmentedReference, tiny_model

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        model = tiny_model(in_channels=128, out_channels=128, joint_attention_dim=7680)
        targets = tuple(replace(_packed_input(i, length=i + 2), text=_packed_input(i, length=i + 2).text.float())
                        for i in range(2))
        decoder = _PackedDecoder()
        generation.generate_packed_arc_samples(model, decoder, targets, ((0, 1),), steps=2,
            seeds=(50, 51), compute_dtype=torch.float32, backend=SegmentedReference())
        for index, target in enumerate(targets):
            for sample_index, seed in enumerate((50, 51)):
                initial = sample_conditioning_noise(target, seed, ("generation", sample_index), "cpu").unsqueeze(0)
                reference = continue_arc_batch(model, _PackedDecoder(), pad_flux_text_prefixes((target.text,)),
                    initial_latents=initial, timestep_schedule=linear_timestep_schedule(2),
                    compute_dtype=torch.float32)
                torch.testing.assert_close(decoder.states[sample_index][index], reference.final_latents[0],
                                           atol=2e-5, rtol=2e-5)

    def test_euler_uses_fp32_state_negative_deltas_and_fixed_independent_noise(self):
        target = _packed_input()
        decoder = _PackedDecoder()
        initial = []
        times_seen = []

        def predict(transformer, state, text, positions, times, layout, **_):
            self.assertFalse(transformer.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(state.dtype, torch.float32)
            self.assertEqual(state.shape, (1, 900, 128))
            times_seen.append(times[0].item())
            if times[0] == 1:
                initial.append(state.clone())
            return torch.full_like(state, 2.0, dtype=torch.bfloat16)

        rng = torch.random.get_rng_state().clone()
        with patch.object(generation, "predict_packed_flux_velocity", side_effect=predict):
            result = self.generate((target,), ((0,),), decoder)
        self.assertEqual(times_seen, [1., .75, .5, .25] * 2)
        for original, final in zip(initial, decoder.states, strict=True):
            torch.testing.assert_close(final, original - 2.0)
        self.assertFalse(torch.equal(initial[0], initial[1]))
        self.assertTrue(self.transformer.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        self.assertTrue(all(prediction.valid for prediction in result[0]))

    def test_generates_without_any_supervision_fields(self):
        with patch.object(generation, "predict_packed_flux_velocity", side_effect=lambda _, x, *a, **k: x * 0):
            result = self.generate((_packed_input(),), ((0,),), steps=1, seeds=(50,))
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0].grid, ((0, 0, 0),) * 3)


if __name__ == "__main__":
    unittest.main()
