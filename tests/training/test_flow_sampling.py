"""Tests for flow sampling."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.model import TARGET_LATENT_SHAPE
from diffusion_arc.training.step.flow_sampling import (
    FlowTimeQuotaPolicy,
    QuotaFlowSampler,
    FlowMatchingBatch,
    build_flow_sampler,
    sample_quota_flow_batches,
    sample_flow_matching_batch,
    sample_stratified_flow_matching_batch,
    select_flow_batch,
)
from diffusion_arc.training.step.loss import flow_matching_loss


class QuotaFlowSamplingTests(unittest.TestCase):
    def test_full_and_tail_batches_have_exact_quota_and_correct_flow(self):
        policy = FlowTimeQuotaPolicy(48, 4, 30, 3)
        for count, quota in ((48, 4), (30, 3)):
            with self.subTest(count=count):
                clean = tuple(torch.full((2, 128), index / 100) for index in range(count))
                flows = sample_quota_flow_batches(clean, endpoint_count=policy.endpoint_count(count),
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
            flows = sample_quota_flow_batches((torch.zeros(1, 128),) * 2, endpoint_count=0,
                                                 generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(torch.cat([f.timesteps for f in flows]),
                                   torch.tensor([2**-24, 1 - 2**-24]), rtol=0, atol=0)

    def test_seed_restore_replays_times_noise_and_rng_position(self):
        clean = tuple(torch.zeros(2, 128) for _ in range(48))
        generator = torch.Generator().manual_seed(167)
        state = generator.get_state()
        first = sample_quota_flow_batches(clean, endpoint_count=4, generator=generator)
        after_first = generator.get_state()
        generator.set_state(state)
        second = sample_quota_flow_batches(clean, endpoint_count=4, generator=generator)
        for a, b in zip(first, second):
            torch.testing.assert_close(a.timesteps, b.timesteps, rtol=0, atol=0)
            torch.testing.assert_close(a.noisy_latents, b.noisy_latents, rtol=0, atol=0)
            torch.testing.assert_close(a.target_velocity, b.target_velocity, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), after_first, rtol=0, atol=0)
        third = sample_quota_flow_batches(clean, endpoint_count=4, generator=generator)
        self.assertFalse(torch.equal(torch.cat([f.timesteps for f in first]), torch.cat([f.timesteps for f in third])))

    def test_policy_and_sampler_reject_invalid_counts_and_precision(self):
        for args in ((0, 0, 0, 0), (48, 49, 30, 3), (48, 4, 30, 31), (48, 4, 49, 3),
                     (48, 4, 48, 3), (True, 0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FlowTimeQuotaPolicy(*args)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            FlowTimeQuotaPolicy(48, 4, 30, 3).endpoint_count(29)
        for clean, quota in (((), 0), ((torch.zeros(2, 128),), 2),
                             ((torch.zeros(2, 128, dtype=torch.bfloat16),), 1),
                             ((torch.zeros(2, 128),), True)):
            with self.subTest(quota=quota), self.assertRaises(ValueError):
                sample_quota_flow_batches(clean, endpoint_count=quota)


class HighNoiseFlowSamplingTests(unittest.TestCase):
    def test_legacy_stream_preserves_recorded_main_draws_and_generator_position(self):
        # Recorded from main 5f42682, before sharing quota draws with DDP.
        generator = torch.Generator().manual_seed(169)
        flows = sample_quota_flow_batches(
            (torch.zeros(2, 128),) * 3, endpoint_count=1, high_noise_count=1, generator=generator,
        )
        torch.testing.assert_close(torch.cat([flow.timesteps for flow in flows]),
                                   torch.tensor([1.0, 0.5035141706466675, 0.967369556427002]),
                                   atol=0, rtol=0)
        expected = torch.tensor([
            [-0.19766007363796234, -0.12827102839946747, 1.310320496559143, -0.18620900809764862],
            [-1.6903228759765625, -0.505885124206543, 0.27265220880508423, -1.2397191524505615],
            [0.6370355486869812, 0.4552030861377716, 0.4129663109779358, 0.9227776527404785],
        ])
        torch.testing.assert_close(torch.stack([flow.target_velocity.flatten()[:4] for flow in flows]),
                                   expected, atol=0, rtol=0)
        self.assertEqual(torch.randint(0, 2**24, (3,), generator=generator).tolist(),
                         [444962, 14300187, 4768179])

    def test_full_and_tail_quotas_assign_three_groups_before_packing(self):
        policy = FlowTimeQuotaPolicy(48, 24, 30, 15, 12, 8)
        randint = torch.randint

        def draw_times(low, high, shape, **kwargs):
            # Make the broad uniform group identifiable even though its support
            # normally overlaps with the dedicated high-noise group.
            if low == 1 and high == 2**24:
                return torch.full(shape, 2**22, dtype=torch.int64, device=kwargs["device"])
            return randint(low, high, shape, **kwargs)

        for count, endpoint, high_noise, uniform in ((48, 24, 12, 12), (30, 15, 8, 7)):
            with self.subTest(count=count):
                clean = tuple(torch.full((2, 128), index / 100) for index in range(count))
                sampler = QuotaFlowSampler(torch.Generator().manual_seed(169), policy)
                with patch("diffusion_arc.training.step.flow_sampling.torch.randint", side_effect=draw_times):
                    flows = sampler(clean)
                times = torch.cat([flow.timesteps for flow in flows])
                self.assertEqual(int((times == 1).sum()), endpoint)
                self.assertEqual(int(((times > 0.8) & (times < 1)).sum()), high_noise)
                self.assertEqual(int((times == 0.25).sum()), uniform)
                self.assertFalse(bool((times[:endpoint] == 1).all()))
                for latent, flow in zip(clean, flows):
                    noise = flow.target_velocity[0] + latent
                    torch.testing.assert_close(
                        flow.noisy_latents[0], (1 - flow.timesteps) * latent + flow.timesteps * noise,
                    )
                    self.assertEqual(flow.noisy_latents.dtype, torch.float32)

                # A different physical order selects the same logical draws.
                indices = (count - 1, 0, count // 2)
                packed = select_flow_batch(flows, indices)
                torch.testing.assert_close(packed.timesteps, times[list(indices)], rtol=0, atol=0)
                for slot, index in enumerate(indices):
                    torch.testing.assert_close(packed.noisy_latents[slot], flows[index].noisy_latents[0])

    def test_high_noise_interval_excludes_both_boundaries_in_float32(self):
        def edge_draws(low, high, shape, **kwargs):
            return torch.tensor([low, high - 1], device=kwargs["device"])

        with patch("diffusion_arc.training.step.flow_sampling.torch.randint", side_effect=edge_draws):
            flows = sample_quota_flow_batches(
                (torch.zeros(1, 128),) * 2, endpoint_count=0, high_noise_count=2,
                generator=torch.Generator().manual_seed(9),
            )
        times = torch.cat([flow.timesteps for flow in flows]).sort().values
        expected = torch.stack((
            torch.nextafter(torch.tensor(0.8), torch.tensor(1.0)),
            torch.nextafter(torch.tensor(1.0), torch.tensor(0.8)),
        ))
        torch.testing.assert_close(times, expected, rtol=0, atol=0)
        self.assertTrue(bool(((times > 0.8) & (times < 1)).all()))

    def test_restoring_generator_replays_mixture_and_independent_noise(self):
        generator = torch.Generator().manual_seed(169)
        sampler = QuotaFlowSampler(generator, FlowTimeQuotaPolicy(48, 24, 30, 15, 12, 8))
        clean = (torch.zeros(2, 128),) * 48
        state = generator.get_state()
        first = sampler(clean)
        after_first = generator.get_state()
        generator.set_state(state)
        replay = sampler(clean)
        for a, b in zip(first, replay):
            torch.testing.assert_close(a.timesteps, b.timesteps, rtol=0, atol=0)
            torch.testing.assert_close(a.noisy_latents, b.noisy_latents, rtol=0, atol=0)
            torch.testing.assert_close(a.target_velocity, b.target_velocity, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), after_first, rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0].target_velocity, first[1].target_velocity))
        next_times = torch.cat([flow.timesteps for flow in sampler(clean)])
        first_times = torch.cat([flow.timesteps for flow in first])
        self.assertFalse(torch.equal(first_times == 1, next_times == 1))

    def test_quota_policy_rejects_overlap_invalid_bounds_and_inconsistent_tail(self):
        for kwargs in (
            {"high_noise_targets_per_full_batch": 25},
            {"high_noise_targets_in_final_batch": 16},
            {"high_noise_targets_per_full_batch": True},
            {"high_noise_min": -0.1},
            {"high_noise_max": 1.1},
            {"high_noise_min": 1.0},
            {"high_noise_min": float("nan")},
            {"high_noise_max": float("inf")},
            {"high_noise_min": 1.0 - 2**-24},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FlowTimeQuotaPolicy(48, 24, 30, 15, **kwargs)
        with self.assertRaisesRegex(ValueError, "full-batch high-noise"):
            FlowTimeQuotaPolicy(48, 24, 48, 24, 12, 8)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            FlowTimeQuotaPolicy(48, 24, 30, 15, 12, 8).high_noise_count(29)
        for count in (-1, True, 2):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "high_noise_count"):
                sample_quota_flow_batches((torch.zeros(2, 128),) * 2,
                                             endpoint_count=1, high_noise_count=count)

    def test_build_sampler_resolves_new_and_existing_recipe(self):
        for high_noise in (False, True):
            with self.subTest(high_noise=high_noise):
                flow = {
                    "rng_scheme": "legacy_stream_v1",
                    "time_distribution": "uniform_with_fixed_endpoint_count",
                    "endpoint_targets_per_full_batch": 4,
                    "endpoint_targets_in_final_batch": 3,
                }
                if high_noise:
                    flow.update(
                        time_distribution="uniform_with_fixed_endpoint_and_high_noise_counts",
                        endpoint_targets_per_full_batch=24, endpoint_targets_in_final_batch=15,
                        high_noise_targets_per_full_batch=12, high_noise_targets_in_final_batch=8,
                        high_noise_min=0.8, high_noise_max=1.0,
                    )
                config = SimpleNamespace(
                    config_type="recipe", recipe={"flow": flow},
                    loop=SimpleNamespace(optimizer_batch_size=48), final_batch_target_count=30,
                )
                sampler = build_flow_sampler(config, generator=torch.Generator().manual_seed(169))
                self.assertIsInstance(sampler, QuotaFlowSampler)
                self.assertEqual(sampler.policy.endpoint_count(48), 24 if high_noise else 4)
                self.assertEqual(sampler.policy.high_noise_count(48), 12 if high_noise else 0)
                self.assertEqual(sampler.policy.high_noise_count(30), 8 if high_noise else 0)

    def test_indexed_recipe_cannot_silently_use_the_legacy_stream(self):
        config = SimpleNamespace(config_type="training_run", recipe={"flow": {
            "rng_scheme": "global_step_target_v1",
        }})
        with self.assertRaisesRegex(ValueError, "global batch plan"):
            build_flow_sampler(config, generator=torch.Generator())


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
