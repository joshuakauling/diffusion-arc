"""Global endpoint quotas and exact flow assignment across rank and pack layouts."""

from dataclasses import replace
import unittest

import torch

from diffusion_arc.training.step.distributed_flow import DistributedFlowSampler


class DistributedFlowTests(unittest.TestCase):
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
