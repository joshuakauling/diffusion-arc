from __future__ import annotations

import unittest

import torch

from diffusion_arc.flow import flow_matching_loss, make_flow_batch
from diffusion_arc.sampling import (
    bfl_time_schedule,
    compute_empirical_mu,
    euler_integrate,
)


class FlowTests(unittest.TestCase):
    def test_straight_path_and_velocity_target(self) -> None:
        clean = torch.tensor([[[1.0, -1.0], [2.0, 0.0]]])
        noise = torch.tensor([[[3.0, 1.0], [0.0, 4.0]]])
        timesteps = torch.tensor([0.25])

        batch = make_flow_batch(clean, timesteps=timesteps, noise=noise)

        torch.testing.assert_close(
            batch.noisy_latents,
            0.75 * clean + 0.25 * noise,
        )
        torch.testing.assert_close(batch.target_velocity, noise - clean)
        self.assertIs(batch.clean_latents, clean)
        self.assertIs(batch.noise, noise)

    def test_seeded_sampling_is_deterministic(self) -> None:
        clean = torch.zeros(2, 3, 4)
        first = make_flow_batch(
            clean,
            generator=torch.Generator().manual_seed(123),
        )
        second = make_flow_batch(
            clean,
            generator=torch.Generator().manual_seed(123),
        )
        torch.testing.assert_close(first.timesteps, second.timesteps)
        torch.testing.assert_close(first.noise, second.noise)
        torch.testing.assert_close(first.noisy_latents, second.noisy_latents)

    def test_loss_is_float32_finite_and_backpropagates(self) -> None:
        prediction = torch.ones(1, 2, 3, dtype=torch.float16, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = flow_matching_loss(prediction, target)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_rejects_invalid_time(self) -> None:
        with self.assertRaises(ValueError):
            make_flow_batch(
                torch.zeros(1, 2, 3),
                timesteps=torch.tensor([1.1]),
            )


class SamplingTests(unittest.TestCase):
    def test_schedule_matches_pinned_bfl_reference_values(self) -> None:
        self.assertAlmostEqual(compute_empirical_mu(900, 50), 1.6889832340736843)
        schedule = bfl_time_schedule(num_steps=50, image_seq_len=900)
        self.assertEqual(schedule.shape, (51,))
        self.assertEqual(schedule[0].item(), 1.0)
        self.assertEqual(schedule[-1].item(), 0.0)
        self.assertTrue(bool((schedule[:-1] > schedule[1:]).all()))
        self.assertAlmostEqual(schedule[1].item(), 0.9962446093559265)
        self.assertAlmostEqual(schedule[25].item(), 0.8440904021263123)
        self.assertAlmostEqual(schedule[-2].item(), 0.09949600696563721)

    def test_backward_euler_constant_velocity(self) -> None:
        initial = torch.zeros(2, 3, 4)
        observed_times: list[torch.Tensor] = []

        def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            observed_times.append(time.clone())
            return torch.full_like(state, 2.0)

        result = euler_integrate(initial, velocity, [1.0, 0.5, 0.0])
        torch.testing.assert_close(result, torch.full_like(initial, -2.0))
        self.assertEqual(len(observed_times), 2)
        torch.testing.assert_close(observed_times[0], torch.ones(2))
        torch.testing.assert_close(observed_times[1], torch.full((2,), 0.5))

    def test_euler_rejects_non_backward_schedule(self) -> None:
        with self.assertRaises(ValueError):
            euler_integrate(
                torch.zeros(1, 2, 3),
                lambda state, _time: state,
                [1.0, 0.5, 0.75, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
