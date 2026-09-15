"""Optimizer-used rates and continuation of linear warmup without a restart."""

import copy
import unittest

import torch

from diffusion_arc.training.step.lr_schedule import LinearWarmupConstantLR


class LinearWarmupConstantTests(unittest.TestCase):
    def test_first_fiftieth_and_later_updates_use_the_agreed_rates(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=50)
        rates = []
        for _ in range(60):
            rates.append(optimizer.param_groups[0]["lr"])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(rates[0], 2e-7)
        self.assertAlmostEqual(rates[48], 9.8e-6)
        self.assertEqual(rates[49:], [1e-5] * 11)
        self.assertEqual(scheduler.last_epoch, 60)

    def test_restoring_optimizer_and_scheduler_continues_warmup_exactly(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        scheduler = LinearWarmupConstantLR(optimizer, warmup_steps=50)

        def advance(value, opt, sched, count):
            rates = []
            for _ in range(count):
                rates.append(opt.param_groups[0]["lr"])
                value.grad = value.detach().clone() * 0.3
                opt.step()
                sched.step()
            return rates

        advance(parameter, optimizer, scheduler, 17)
        saved_parameter = parameter.detach().clone()
        optimizer_state, scheduler_state = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())
        expected = advance(parameter, optimizer, scheduler, 40)
        restored_parameter = torch.nn.Parameter(saved_parameter)
        restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1e-5)
        restored_scheduler = LinearWarmupConstantLR(restored_optimizer, warmup_steps=50)
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        actual = advance(restored_parameter, restored_optimizer, restored_scheduler, 40)
        self.assertEqual(actual, expected)
        torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
        self.assertEqual(restored_scheduler.last_epoch, 57)

    def test_rejects_invalid_warmup(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1e-5)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LinearWarmupConstantLR(optimizer, warmup_steps=value)


if __name__ == "__main__":
    unittest.main()
