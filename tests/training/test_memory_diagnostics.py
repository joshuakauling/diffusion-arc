"""OOM evidence survives error handling without CUDA work or DDP coordination."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.training.run.memory_diagnostics import CudaMemoryHistory


class CudaMemoryHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.events = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("torch.cuda.memory.get_allocator_backend", return_value="native"))
        self.record = stack.enter_context(patch("torch.cuda.memory._record_memory_history",
                                               side_effect=lambda **kw: self.events.append(("history", kw))))
        self.dump = stack.enter_context(patch("torch.cuda.memory._dump_snapshot", side_effect=self.save_snapshot))
        self.stats = stack.enter_context(patch("torch.cuda.memory_stats", return_value=dict.fromkeys((
            "allocated_bytes.all.current", "reserved_bytes.all.current", "active_bytes.all.current",
            "inactive_split_bytes.all.current", "allocated_bytes.all.peak", "num_alloc_retries", "num_ooms",
        ), 1)))
        stack.enter_context(patch("torch.cuda.mem_get_info", return_value=(300, 1000)))
        for forbidden in ("torch.cuda.synchronize", "torch.cuda.empty_cache", "torch.distributed.all_reduce",
                          "torch.distributed.barrier"):
            stack.enter_context(patch(forbidden, side_effect=AssertionError(f"unexpected {forbidden}")))
        stack.enter_context(patch("diffusion_arc.training.run.memory_diagnostics.print"))

    def save_snapshot(self, filename):
        self.events.append(("dump", None))
        Path(filename).write_bytes(pickle.dumps({"device_traces": [[{"action": "oom", "size": 430}]]}))

    def test_oom_snapshot_precedes_distributed_cleanup_and_preserves_error(self):
        @contextmanager
        def distributed_scope():
            try:
                yield
            finally:
                self.events.append(("distributed_cleanup", None))

        original = torch.OutOfMemoryError("failed request")
        with self.assertRaises(torch.OutOfMemoryError) as caught:
            with distributed_scope(), CudaMemoryHistory(self.directory, 1, torch.device("cuda:1")):
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual([event for event, _ in self.events], ["history", "dump", "history", "distributed_cleanup"])
        self.assertEqual(self.events[0][1]["max_entries"], 100_000)
        self.assertEqual(self.events[0][1]["stacks"], "all")
        self.assertEqual(self.events[2][1]["enabled"], None)
        snapshot = self.directory / "rank-1-oom.pickle"
        self.assertEqual(pickle.loads(snapshot.read_bytes())["device_traces"][0][0]["action"], "oom")
        self.assertFalse(snapshot.with_suffix(".pickle.partial").exists())
        metadata = json.loads(snapshot.with_suffix(".json").read_text())
        self.assertEqual(metadata["reason"], "oom")
        self.assertEqual(metadata["error_type"], type(original).__name__)
        self.assertEqual(metadata["message"], "failed request")
        self.assertEqual(metadata["snapshot_bytes"], snapshot.stat().st_size)

    def test_success_still_saves_a_reference_snapshot(self):
        with CudaMemoryHistory(self.directory, 0, torch.device("cuda:0")):
            pass
        self.assertTrue((self.directory / "rank-0-complete.pickle").exists())
        self.assertFalse((self.directory / "rank-0-oom.pickle").exists())
        self.assertEqual(self.record.call_args.kwargs["enabled"], None)

    def test_snapshot_failure_never_replaces_the_original_oom(self):
        self.dump.side_effect = OSError("disk full")
        original = torch.OutOfMemoryError("original OOM")
        with self.assertRaises(torch.OutOfMemoryError) as caught:
            with CudaMemoryHistory(self.directory, 0, torch.device("cuda:0")):
                raise original
        self.assertIs(caught.exception, original)
        self.assertIn("disk full", original.__notes__[0])
        self.assertEqual(self.record.call_args.kwargs["enabled"], None)
        metadata = json.loads((self.directory / "rank-0-oom.json").read_text())
        self.assertIn("disk full", metadata["snapshot_error"])

    def test_stats_failure_does_not_lose_the_snapshot(self):
        self.stats.side_effect = RuntimeError("device unavailable")
        with CudaMemoryHistory(self.directory, 0, torch.device("cuda:0")):
            pass
        self.assertTrue((self.directory / "rank-0-complete.pickle").exists())
        metadata = json.loads((self.directory / "rank-0-complete.json").read_text())
        self.assertIn("device unavailable", metadata["memory_stats_error"])

    def test_metadata_and_disable_failures_preserve_original_exception(self):
        self.record.side_effect = [None, RuntimeError("disable failed")]
        original = torch.OutOfMemoryError("original OOM")
        with patch.object(Path, "write_text", side_effect=OSError("metadata failed")):
            with self.assertRaises(torch.OutOfMemoryError) as caught:
                with CudaMemoryHistory(self.directory, 0, torch.device("cuda:0")):
                    raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(len(original.__notes__), 2)
        self.assertTrue((self.directory / "rank-0-oom.pickle").exists())

    def test_failed_capture_on_success_is_not_silently_accepted(self):
        self.dump.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            with CudaMemoryHistory(self.directory, 0, torch.device("cuda:0")):
                pass
        self.assertEqual(self.record.call_args.kwargs["enabled"], None)

    def test_cpu_is_rejected_before_recording(self):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            CudaMemoryHistory(self.directory, 0, torch.device("cpu"))
        self.record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
