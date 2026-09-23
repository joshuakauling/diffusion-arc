"""Torchrun device ownership and cleanup on normal or exceptional exit."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.training.distributed import _torchrun_ranks, distributed_runtime


class DistributedRuntimeTests(unittest.TestCase):
    def test_requires_valid_torchrun_environment_and_visible_devices(self):
        valid = dict(RANK="1", LOCAL_RANK="1", WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
        for environment in ({}, dict(valid, RANK="two"), dict(valid, RANK="2"),
                            dict(valid, LOCAL_RANK="2"), dict(valid, LOCAL_WORLD_SIZE="3")):
            with self.subTest(environment=environment), patch.dict("os.environ", environment, clear=True), \
                    patch("torch.cuda.device_count", return_value=2), self.assertRaises(ValueError):
                _torchrun_ranks()
        with patch.dict("os.environ", valid, clear=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "more local processes"):
                _torchrun_ranks()

    def test_selects_local_device_before_init_and_cleans_up_after_error(self):
        from dataclasses import dataclass

        @dataclass
        class Config:
            device: str = "cuda:0"

        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        for rank, world_size, fail in ((0, 1, False), (1, 2, False), (1, 2, True)):
            torch.set_num_threads(2)
            events = []
            with ExitStack() as stack:
                stack.enter_context(patch("diffusion_arc.training.distributed._torchrun_ranks", return_value=(rank, rank, world_size)))
                stack.enter_context(patch("torch.distributed.is_initialized", return_value=False))
                def prepare(config):
                    self.assertEqual(config.device, f"cuda:{rank}")
                    self.assertEqual(torch.get_num_threads(), 8)
                    events.append("device")
                    return torch.device(config.device)
                stack.enter_context(patch("diffusion_arc.training.distributed.prepare_device", side_effect=prepare))
                initialize = stack.enter_context(patch("torch.distributed.init_process_group", side_effect=lambda *a, **k: events.append("init")))
                destroy = stack.enter_context(patch("torch.distributed.destroy_process_group", side_effect=lambda: events.append("destroy")))
                try:
                    with distributed_runtime(Config()) as runtime:
                        self.assertEqual((runtime.rank, runtime.local_rank, runtime.world_size), (rank, rank, world_size))
                        events.append("body")
                        if fail:
                            raise LookupError("training failed")
                except LookupError:
                    self.assertTrue(fail)
                self.assertEqual(events, ["device", "init", "body", "destroy"])
                self.assertEqual(initialize.call_args.args, ("nccl",))
                self.assertEqual(initialize.call_args.kwargs["device_id"], torch.device(f"cuda:{rank}"))
                destroy.assert_called_once()

    def test_does_not_take_over_an_existing_process_group(self):
        with patch("torch.distributed.is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "own its process group"):
                with distributed_runtime(SimpleNamespace()):
                    self.fail("unexpected initialized context")


if __name__ == "__main__":
    unittest.main()
