"""Tests for straight-path flow-matching sampling and loss."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

import torch

from diffusion_arc.model import (
    TARGET_LATENT_SHAPE,
    FlowMatchingBatch,
    flow_matching_loss,
    sample_flow_matching_batch,
)


def _clean_latents(
    target_count: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = torch.linspace(
        -0.75,
        0.75,
        steps=TARGET_LATENT_SHAPE[0] * TARGET_LATENT_SHAPE[1],
        dtype=dtype,
    ).reshape(1, *TARGET_LATENT_SHAPE)
    return values.expand(target_count, -1, -1).clone()


class FlowMatchingTests(unittest.TestCase):
    def test_samples_independent_times_and_builds_straight_path(self) -> None:
        clean_latents = _clean_latents()

        result = sample_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(41),
        )

        self.assertIsInstance(result, FlowMatchingBatch)
        self.assertEqual(result.noisy_latents.shape, (2, *TARGET_LATENT_SHAPE))
        self.assertEqual(result.timesteps.shape, (2,))
        self.assertEqual(result.target_velocity.shape, (2, *TARGET_LATENT_SHAPE))
        self.assertTrue(
            bool(torch.all((result.timesteps >= 0) & (result.timesteps < 1)))
        )
        self.assertNotEqual(result.timesteps[0].item(), result.timesteps[1].item())

        noise = result.target_velocity + clean_latents
        self.assertFalse(torch.equal(noise[0], noise[1]))
        broadcast_timesteps = result.timesteps.reshape(2, 1, 1)
        torch.testing.assert_close(
            result.noisy_latents,
            (1 - broadcast_timesteps) * clean_latents
            + broadcast_timesteps * noise,
        )
        torch.testing.assert_close(result.target_velocity, noise - clean_latents)

    def test_seeded_sampling_is_reproducible(self) -> None:
        clean_latents = _clean_latents()

        first = sample_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(167),
        )
        second = sample_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(167),
        )

        torch.testing.assert_close(first.timesteps, second.timesteps)
        torch.testing.assert_close(first.noisy_latents, second.noisy_latents)
        torch.testing.assert_close(first.target_velocity, second.target_velocity)

    def test_latent_dtype_is_preserved_and_timesteps_are_float32(self) -> None:
        clean_latents = _clean_latents(dtype=torch.float64)

        result = sample_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertEqual(result.noisy_latents.dtype, torch.float64)
        self.assertEqual(result.target_velocity.dtype, torch.float64)
        self.assertEqual(result.timesteps.dtype, torch.float32)
        self.assertEqual(result.noisy_latents.device, clean_latents.device)
        self.assertEqual(result.target_velocity.device, clean_latents.device)
        self.assertEqual(result.timesteps.device, clean_latents.device)

    def test_rejects_invalid_clean_latent_contract(self) -> None:
        invalid_latents = (
            torch.zeros((1, TARGET_LATENT_SHAPE[0], TARGET_LATENT_SHAPE[1] - 1)),
            torch.zeros((1, 0, TARGET_LATENT_SHAPE[1])),
            torch.zeros((0, *TARGET_LATENT_SHAPE)),
            torch.zeros((1, *TARGET_LATENT_SHAPE), dtype=torch.int64),
        )

        for clean_latents in invalid_latents:
            with self.subTest(shape=tuple(clean_latents.shape), dtype=clean_latents.dtype):
                expected_error = (
                    TypeError if clean_latents.dtype == torch.int64 else ValueError
                )
                with self.assertRaises(expected_error):
                    sample_flow_matching_batch(clean_latents)

        with self.assertRaises(TypeError):
            sample_flow_matching_batch(  # type: ignore[arg-type]
                _clean_latents(1),
                generator=object(),
            )

    def test_compact_token_grid_uses_the_same_flow_objective(self) -> None:
        clean_latents = torch.randn(2, 9, TARGET_LATENT_SHAPE[1])

        result = sample_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(23),
        )
        loss = flow_matching_loss(result.target_velocity, result.target_velocity)

        self.assertEqual(result.noisy_latents.shape, clean_latents.shape)
        self.assertEqual(result.target_velocity.shape, clean_latents.shape)
        self.assertEqual(loss.item(), 0.0)

    def test_output_contract_is_frozen(self) -> None:
        result = sample_flow_matching_batch(
            _clean_latents(1),
            generator=torch.Generator().manual_seed(5),
        )

        with self.assertRaises(FrozenInstanceError):
            result.timesteps = torch.ones(1)  # type: ignore[misc]

    def test_loss_is_unweighted_float32_mse_and_backpropagates(self) -> None:
        predicted_velocity = torch.empty(
            (2, *TARGET_LATENT_SHAPE),
            dtype=torch.bfloat16,
        )
        predicted_velocity[0].fill_(1)
        predicted_velocity[1].fill_(3)
        predicted_velocity.requires_grad_()
        target_velocity = torch.zeros(
            (2, *TARGET_LATENT_SHAPE),
            dtype=torch.float32,
        )

        loss = flow_matching_loss(predicted_velocity, target_velocity)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 5.0)
        loss.backward()
        self.assertIsNotNone(predicted_velocity.grad)
        self.assertTrue(bool(torch.isfinite(predicted_velocity.grad).all()))

    def test_loss_rejects_mismatched_target_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            flow_matching_loss(
                torch.zeros((2, *TARGET_LATENT_SHAPE)),
                torch.zeros((1, *TARGET_LATENT_SHAPE)),
            )


if __name__ == "__main__":
    unittest.main()
