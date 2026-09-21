"""CPU checks for exact Nsight capture boundaries and cleanup."""

from contextlib import ExitStack
from dataclasses import replace
import json
import gzip
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import profile_single_gpu
import torch

from diffusion_arc.training.profiling import nsight_capture, torch_capture, training_range

from tests.training.test_profile_single_gpu_script import _AcceptAnyModule, _config
from diffusion_arc.training.setup import TrainingModels


class TorchCaptureTests(unittest.TestCase):
    def test_only_update_eleven_is_exported_and_annotations_are_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            with torch_capture(Path(directory), 1, torch.device("cpu"), warmup_steps=10) as profiler:
                for step in range(1, 12):
                    with training_range(f"optimizer_step/{step}"):
                        value = torch.ones(2, requires_grad=True)
                        value.square().sum().backward()
                    profiler.step()
            with gzip.open(Path(directory) / "rank-1.trace.json.gz", "rt") as handle:
                trace = json.load(handle)
            names = {event["name"] for event in trace["traceEvents"]}
            self.assertEqual({name for name in names if name.startswith("optimizer_step/")}, {"optimizer_step/11"})
            self.assertIn("aten::square", names)
            self.assertEqual(trace["training_capture"]["rank"], 1)
            with patch("torch.profiler.record_function") as record:
                with training_range("outside_profile"):
                    pass
                record.assert_not_called()
            with self.assertRaises(FileExistsError):
                with torch_capture(Path(directory), 1, torch.device("cpu"), warmup_steps=10):
                    pass

    def test_incomplete_capture_and_failure_restore_annotation_state(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, "training failed" if fail else "capture window"):
                    with torch_capture(Path(directory), 0, torch.device("cpu"), warmup_steps=2) as profiler:
                        torch.ones(2).sum()
                        profiler.step()
                        if fail:
                            raise RuntimeError("training failed")
                with patch("torch.profiler.record_function") as record:
                    with training_range("outside_profile"):
                        pass
                    record.assert_not_called()


class NsightCaptureTests(unittest.TestCase):
    def _cuda(self, stack):
        events = []
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated",
                     "max_memory_reserved", "reset_peak_memory_stats"):
            stack.enter_context(patch(f"torch.cuda.{name}", return_value=0))
        stack.enter_context(patch("torch.cuda.memory_stats", return_value={}))
        for name, label in (("synchronize", "sync"), ("profiler.start", "start"),
                            ("profiler.stop", "stop"), ("nvtx.range_push", "push"),
                            ("nvtx.range_pop", "pop")):
            stack.enter_context(patch(f"torch.cuda.{name}",
                side_effect=lambda *args, label=label: events.append((label, *args))))
        return events

    def test_four_warmup_then_three_complete_updates(self):
        with ExitStack() as stack:
            events = self._cuda(stack)
            with nsight_capture("cuda:0") as capture:
                for update in range(1, 8):
                    events.append(("work", update))
                    capture.step()
            self.assertEqual(events, [
                ("work", 1), ("work", 2), ("work", 3), ("work", 4),
                ("sync", "cuda:0"), ("start",), ("push", "optimizer_step/5"),
                ("work", 5), ("pop",), ("push", "optimizer_step/6"),
                ("work", 6), ("pop",), ("push", "optimizer_step/7"),
                ("work", 7), ("sync", "cuda:0"), ("pop",), ("stop",),
            ])

    def test_allocator_peaks_are_reset_each_update_without_extra_gpu_waits(self):
        with ExitStack() as stack:
            events = self._cuda(stack)
            allocated = stack.enter_context(patch("torch.cuda.memory_allocated", side_effect=range(8)))
            reserved = stack.enter_context(patch("torch.cuda.memory_reserved", return_value=100))
            peak = stack.enter_context(patch("torch.cuda.max_memory_allocated", side_effect=range(10, 18)))
            stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=120))
            stack.enter_context(patch("torch.cuda.memory_stats", return_value={"num_alloc_retries": 2, "num_ooms": 0}))
            reset = stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
            with nsight_capture("cuda:0") as capture:
                for _ in range(7):
                    capture.step()
            self.assertEqual(capture.memory_samples[-1], {
                "completed_updates": 7, "allocated_bytes": 7, "reserved_bytes": 100,
                "peak_allocated_bytes": 17, "peak_reserved_bytes": 120,
                "allocation_retries": 2, "oom_count": 0,
            })
            self.assertEqual(reset.call_count, 8)
            self.assertEqual(sum(event[0] == "sync" for event in events), 2)

    def test_exception_stops_capture_and_restores_disabled_nvtx(self):
        with ExitStack() as stack:
            events = self._cuda(stack)
            with self.assertRaisesRegex(ValueError, "training failed"):
                with nsight_capture("cuda:0") as capture:
                    for _ in range(4):
                        capture.step()
                    with training_range("train/flux_backward"):
                        raise ValueError("training failed")
            self.assertEqual(events[-3:], [("sync", "cuda:0"), ("pop",), ("stop",)])
            count = len(events)
            with training_range("inactive"):
                pass
            self.assertEqual(len(events), count)

    def test_short_stream_is_not_a_successful_profile(self):
        for updates in (3, 5):
            with self.subTest(updates=updates), ExitStack() as stack:
                events = self._cuda(stack)
                with self.assertRaisesRegex(RuntimeError, "seven completed"):
                    with nsight_capture("cuda:0") as capture:
                        for _ in range(updates):
                            capture.step()
                self.assertEqual(sum(e[0] == "stop" for e in events), int(updates >= 4))

    def test_gpu_sync_failure_still_stops_capture(self):
        with ExitStack() as stack:
            events = self._cuda(stack)
            with self.assertRaisesRegex(RuntimeError, "GPU error"):
                with nsight_capture("cuda:0") as capture:
                    for _ in range(4):
                        capture.step()
                    with patch("torch.cuda.synchronize", side_effect=RuntimeError("GPU error")):
                        capture.close()
            self.assertEqual(events[-1], ("stop",))

    def test_wrong_step_limit_fails_before_device_or_model_loading(self):
        config = _config(Path("/tmp"))
        config = replace(config, loop=replace(config.loop, max_optimizer_steps=60))
        with patch.object(profile_single_gpu, "load_training_config", return_value=config), \
             patch.object(profile_single_gpu, "prepare_device") as prepare:
            with self.assertRaisesRegex(ValueError, "exactly seven"):
                profile_single_gpu.profile_from_config("unused.json", profiler_kind="nsys")
            prepare.assert_not_called()

    def test_nsys_entrypoint_shares_training_without_torch_profiler(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            config = _config(root)
            models = TrainingModels(vae_decoder=object(), flux_packing_backend=object(),tokenizer=object(), nvarc_encoder=_AcceptAnyModule(),
                vae_encoder=_AcceptAnyModule(), flux_transformer=_AcceptAnyModule(),
                flux_compute_dtype=torch.bfloat16)
            events = self._cuda(stack)
            for name, result in {
                "load_training_config": config,
                "prepare_device": torch.device("cuda:0"),
                "seed_process": None, "load_models": models,
                "build_dataloader": [SimpleNamespace(target_grids=(None,) * 6, key=f"s{i}") for i in range(7)],
                "build_optimizer": object(),
                "build_lr_scheduler": None, "build_flow_generator": object(),
                "build_flow_sampler": object(), "build_velocity_loss": object(),
                "version": "test-version",
            }.items():
                stack.enter_context(patch.object(profile_single_gpu, name, return_value=result))
            stack.enter_context(patch.dict("os.environ", {"DIFFUSION_ARC_PROFILER_OUTPUT_ROOT": str(root)}))
            stack.enter_context(patch("torch.cuda.get_device_name", return_value="test GPU"))
            torch_profiler = stack.enter_context(patch("torch.profiler.profile"))
            def make_trainer(*args, **kwargs):
                def step(sequences):
                    with training_range("train/adamw"):
                        pass
                    return SimpleNamespace(loss=0.5, target_count=5, flux_forward_calls=2)
                return SimpleNamespace(step=step)
            stack.enter_context(patch.object(profile_single_gpu, "Trainer", side_effect=make_trainer))
            profile_single_gpu.main(["--config", str(root / "run.json"), "--profiler", "nsys"])
            torch_profiler.assert_not_called()
            summary = json.loads((root / "training-summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_updates"], 7)
            self.assertEqual(summary["recorded_updates"], 3)
            self.assertEqual(len(summary["pytorch_memory"]), 8)
            self.assertEqual(summary["pytorch_memory"][-1]["completed_updates"], 7)
            self.assertEqual(summary["updates"][-1], {"update": 7, "loss": 0.5, "targets": 5, "flux_packs": 2})
            self.assertEqual(sum(e[0] == "start" for e in events), 1)
            self.assertFalse((root / "trace.json").exists())
