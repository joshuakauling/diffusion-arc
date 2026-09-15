"""Tests for flow sampling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.model import TARGET_LATENT_SHAPE
from diffusion_arc.training.step.flow_sampling import (
    EndpointFlowPolicy,
    FlowMatchingBatch,
    sample_endpoint_flow_batches,
    sample_flow_matching_batch,
    sample_stratified_flow_matching_batch,
)
from diffusion_arc.training.step.loss import flow_matching_loss


class EndpointFlowSamplingTests(unittest.TestCase):
    def test_full_and_tail_batches_have_exact_quota_and_correct_flow(self):
        policy = EndpointFlowPolicy(48, 4, 30, 3)
        for count, quota in ((48, 4), (30, 3)):
            with self.subTest(count=count):
                clean = tuple(torch.full((2, 128), index / 100) for index in range(count))
                flows = sample_endpoint_flow_batches(clean, endpoint_count=policy.endpoint_count(count),
                                                     generator=torch.Generator().manual_seed(167))
                times = torch.cat([flow.timesteps for flow in flows])
                self.assertEqual(int((times == 1).sum()), quota)
                self.assertTrue(bool(((times > 0) & (times <= 1)).all()))
                noises = []
                for latent, flow in zip(clean, flows):
                    noise = flow.target_velocity[0] + latent
                    noises.append(noise)
                    torch.testing.assert_close(flow.noisy_latents[0],
                                               (1 - flow.timesteps) * latent + flow.timesteps * noise)
                    if flow.timesteps.item() == 1:
                        torch.testing.assert_close(flow.noisy_latents[0], noise)
                    self.assertEqual(flow.noisy_latents.dtype, torch.float32)
                self.assertFalse(torch.equal(noises[0], noises[1]))
                self.assertGreater(len(set(times[times != 1].tolist())), 1)

    def test_uniform_lattice_excludes_both_endpoints_exactly(self):
        with patch("diffusion_arc.training.step.flow_sampling.torch.randint",
                   return_value=torch.tensor([1, 2**24 - 1])):
            flows = sample_endpoint_flow_batches((torch.zeros(1, 128),) * 2, endpoint_count=0,
                                                 generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(torch.cat([f.timesteps for f in flows]),
                                   torch.tensor([2**-24, 1 - 2**-24]), rtol=0, atol=0)

    def test_seed_restore_replays_times_noise_and_rng_position(self):
        clean = tuple(torch.zeros(2, 128) for _ in range(48))
        generator = torch.Generator().manual_seed(167)
        state = generator.get_state()
        first = sample_endpoint_flow_batches(clean, endpoint_count=4, generator=generator)
        after_first = generator.get_state()
        generator.set_state(state)
        second = sample_endpoint_flow_batches(clean, endpoint_count=4, generator=generator)
        for a, b in zip(first, second):
            torch.testing.assert_close(a.timesteps, b.timesteps, rtol=0, atol=0)
            torch.testing.assert_close(a.noisy_latents, b.noisy_latents, rtol=0, atol=0)
            torch.testing.assert_close(a.target_velocity, b.target_velocity, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), after_first, rtol=0, atol=0)
        third = sample_endpoint_flow_batches(clean, endpoint_count=4, generator=generator)
        self.assertFalse(torch.equal(torch.cat([f.timesteps for f in first]), torch.cat([f.timesteps for f in third])))

    def test_policy_and_sampler_reject_invalid_counts_and_precision(self):
        for args in ((0, 0, 0, 0), (48, 49, 30, 3), (48, 4, 30, 31), (48, 4, 49, 3),
                     (48, 4, 48, 3), (True, 0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                EndpointFlowPolicy(*args)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            EndpointFlowPolicy(48, 4, 30, 3).endpoint_count(29)
        for clean, quota in (((), 0), ((torch.zeros(2, 128),), 2),
                             ((torch.zeros(2, 128, dtype=torch.bfloat16),), 1),
                             ((torch.zeros(2, 128),), True)):
            with self.subTest(quota=quota), self.assertRaises(ValueError):
                sample_endpoint_flow_batches(clean, endpoint_count=quota)


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

    def test_stratified_sampling_covers_every_batch_interval(self) -> None:
        clean_latents = _clean_latents(8)

        result = sample_stratified_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(167),
        )

        ordered = result.timesteps.sort().values
        lower = torch.arange(8, dtype=torch.float32) / 8
        upper = torch.arange(1, 9, dtype=torch.float32) / 8
        self.assertTrue(bool(torch.all(ordered >= lower)))
        self.assertTrue(bool(torch.all(ordered < upper)))

        noise = result.target_velocity + clean_latents
        self.assertFalse(torch.equal(noise[0], noise[1]))
        broadcast_timesteps = result.timesteps.reshape(8, 1, 1)
        torch.testing.assert_close(
            result.noisy_latents,
            (1 - broadcast_timesteps) * clean_latents
            + broadcast_timesteps * noise,
        )

    def test_stratified_sampling_is_seeded_and_handles_one_exposure(self) -> None:
        clean_latents = _clean_latents(8)

        first = sample_stratified_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(72),
        )
        second = sample_stratified_flow_matching_batch(
            clean_latents,
            generator=torch.Generator().manual_seed(72),
        )
        single = sample_stratified_flow_matching_batch(
            _clean_latents(1),
            generator=torch.Generator().manual_seed(72),
        )

        torch.testing.assert_close(first.timesteps, second.timesteps)
        torch.testing.assert_close(first.noisy_latents, second.noisy_latents)
        torch.testing.assert_close(first.target_velocity, second.target_velocity)
        self.assertEqual(single.timesteps.shape, (1,))
        self.assertGreaterEqual(single.timesteps.item(), 0.0)
        self.assertLess(single.timesteps.item(), 1.0)

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


if __name__ == "__main__":
    unittest.main()
