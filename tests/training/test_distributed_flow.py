"""Global endpoint quotas and exact flow assignment across rank and pack layouts."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.training.step.distributed_flow import DistributedFlowSampler, build_distributed_flow_sampler


class DistributedFlowTests(unittest.TestCase):
    def test_existing_ddp_seed_version_preserves_recorded_endpoint_draws(self):
        # Recorded from the original DDP implementation at 62674fd on CPU.
        flows = DistributedFlowSampler(167, 2, 3, (0, 1, 2), 1)((torch.zeros(900, 128),) * 3)
        torch.testing.assert_close(torch.cat([flow.timesteps for flow in flows]),
                                   torch.tensor([1.0, 0.2697577476501465, 0.959400475025177]),
                                   atol=0, rtol=0)
        expected_noise = torch.tensor([
            [1.1000956296920776, -0.435962438583374, -1.0297881364822388, -0.05705995485186577],
            [-0.7274459004402161, -0.6112457513809204, -2.322434425354004, -1.3640118837356567],
            [-1.4439936876296997, 0.29676762223243713, -1.9178637266159058, 0.9092144966125488],
        ])
        torch.testing.assert_close(torch.stack([flow.target_velocity.flatten()[:4] for flow in flows]),
                                   expected_noise, atol=0, rtol=0)

    def test_high_noise_full_and_tail_draws_match_native_single_after_partition(self):
        config = SimpleNamespace(
            seed=169, loop=SimpleNamespace(optimizer_batch_size=6), final_batch_target_count=3,
            recipe={"flow": {
                "time_distribution": "uniform_with_fixed_endpoint_and_high_noise_counts",
                "endpoint_targets_per_full_batch": 2, "endpoint_targets_in_final_batch": 1,
                "high_noise_targets_per_full_batch": 3, "high_noise_targets_in_final_batch": 1,
                "high_noise_min": 0.8, "high_noise_max": 1.0,
            }},
        )
        randint = torch.randint

        def draw_times(low, high, shape, **kwargs):
            if low == 1 and high == 2**24:
                return torch.full(shape, 2**22, dtype=torch.int64, device=kwargs["device"])
            return randint(low, high, shape, **kwargs)

        for count, endpoints, high_noise, partitions in (
            (6, 2, 3, ((0, 3), (4, 1, 2, 5))), (3, 1, 1, ((2,), (1, 0))),
        ):
            with self.subTest(count=count):
                clean = tuple(torch.full((900, 128), i / 100) for i in range(count))
                native = build_distributed_flow_sampler(
                    config, step=7, global_target_count=count, target_indices=tuple(range(count)),
                )
                self.assertEqual((native.endpoint_count, native.high_noise_count), (endpoints, high_noise))
                with patch("diffusion_arc.training.step.flow_sampling.torch.randint", side_effect=draw_times):
                    reference = native(clean)
                    times = torch.cat([flow.timesteps for flow in reference])
                    self.assertEqual(int((times == 1).sum()), endpoints)
                    self.assertEqual(int(((times > 0.8) & (times < 1)).sum()), high_noise)
                    self.assertEqual(int((times == 0.25).sum()), count - endpoints - high_noise)
                    for indices in partitions:
                        sampler = build_distributed_flow_sampler(
                            config, step=7, global_target_count=count, target_indices=indices,
                        )
                        for index, actual in zip(indices, sampler(tuple(clean[i] for i in indices))):
                            for name in ("noisy_latents", "timesteps", "target_velocity"):
                                torch.testing.assert_close(getattr(actual, name), getattr(reference[index], name),
                                                           atol=0, rtol=0)

    def test_observer_receives_actual_draws_without_changing_sampling(self):
        clean = tuple(torch.full((900, 128), i / 10) for i in range(3))
        sampler = DistributedFlowSampler(167, 0, 3, (0, 1, 2), 1)
        captured = []
        observed = replace(sampler, observe_draw=lambda i, t, n: captured.append((i, t.clone(), n.clone())))(clean)
        for expected, actual, (index, time, noise) in zip(sampler(clean), observed, captured, strict=True):
            torch.testing.assert_close(expected.target_velocity, actual.target_velocity, atol=0, rtol=0)
            torch.testing.assert_close(actual.target_velocity, (noise - clean[index]).unsqueeze(0), atol=0, rtol=0)
            torch.testing.assert_close(actual.timesteps, time, atol=0, rtol=0)

    def test_global_draws_survive_partition_reordering_and_unrelated_rng_draws(self):
        clean = tuple(torch.full((900, 128), i / 10) for i in range(7))
        sampler = DistributedFlowSampler(167, 2, 7, tuple(range(7)), 3)
        before = torch.random.get_rng_state().clone()
        reference = sampler(clean)
        torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
        self.assertEqual(sum(float(flow.timesteps[0]) == 1 for flow in reference), 3)
        self.assertTrue(all(0 < float(flow.timesteps[0]) <= 1 for flow in reference))
        for indices in ((6, 0, 4), (1, 2, 3, 5), (2,)):
            actual = replace(sampler, target_indices=indices)(tuple(clean[i] for i in indices))
            for index, flow in zip(indices, actual):
                expected = reference[index]
                for name in ("noisy_latents", "timesteps", "target_velocity"):
                    torch.testing.assert_close(getattr(flow, name), getattr(expected, name), atol=0, rtol=0)
                noise = flow.target_velocity + clean[index]
                time = flow.timesteps
                torch.testing.assert_close(flow.noisy_latents, (1 - time) * clean[index] + time * noise)
        different_step = replace(sampler, step=3)(clean)
        different_seed = replace(sampler, seed=168)(clean)
        self.assertFalse(torch.equal(reference[0].target_velocity, different_step[0].target_velocity))
        self.assertFalse(torch.equal(reference[0].target_velocity, different_seed[0].target_velocity))
        self.assertFalse(torch.equal(reference[0].target_velocity, reference[1].target_velocity))

    def test_rejects_invalid_plan_and_encoding_count(self):
        sampler = DistributedFlowSampler(167, 0, 4, (0, 1), 1)
        for settings in (dict(seed=-1), dict(step=-1), dict(global_target_count=0),
                         dict(endpoint_count=5), dict(target_indices=()),
                         dict(high_noise_count=4), dict(high_noise_count=-1), dict(high_noise_count=True),
                         dict(high_noise_min=1.0), dict(high_noise_max=float("nan")),
                         dict(target_indices=(0, 0)), dict(target_indices=(4,)),
                         dict(target_indices=(True,))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(sampler, **settings)
        with self.assertRaisesRegex(ValueError, "encoded local target count"):
            sampler((torch.zeros(900, 128),))
        for dtype in (torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float32"):
                sampler((torch.zeros(900, 128, dtype=dtype),) * 2)


if __name__ == "__main__":
    unittest.main()
