"""Single-GPU collectors wrap the shared training loop without extra updates."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from scripts import profile_single_gpu as profile
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.profiling import _TORCH_ENABLED


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/training/profiling/single_gpu_b3_torch_profile.json"


class ProfileSingleGpuScriptTests(unittest.TestCase):
    def test_delegates_unchanged_config_and_advances_collector_once_per_update(self):
        config = load_training_config(CONFIG)
        profiler = Mock()
        context = MagicMock()
        context.__enter__.return_value = profiler
        def shared(received, *, experiment):
            self.assertIs(received, config)
            self.assertEqual(experiment.max_steps, 7)
            experiment.runtime = SimpleNamespace(device=torch.device("cuda:0"))
            with ExitStack() as stack:
                experiment.start(stack, config, None, None)
                self.assertTrue(_TORCH_ENABLED.get())
                for step in range(7):
                    experiment.before_step()
                    experiment.completed_step()
                    experiment.after_step(step + 1, None, SimpleNamespace(loss=0.5), None)
                experiment.finish(None)
            self.assertFalse(_TORCH_ENABLED.get())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(profile, "load_training_config", return_value=config) as load, \
                patch.object(profile, "run_benchmark", side_effect=shared) as run, \
                patch.object(profile, "PROFILER_OUTPUT_ROOT", Path(directory)), \
                patch.object(torch.profiler, "profile", return_value=context) as collect, \
                patch.object(torch.cuda, "synchronize", side_effect=AssertionError("unexpected synchronization")):
            profile.profile_from_config(CONFIG)
            collect.call_args.kwargs["on_trace_ready"](profiler)
            profiler.export_chrome_trace.assert_called_once_with(str(Path(directory) / CONFIG.stem / "trace.json"))
        load.assert_called_once_with(CONFIG, expected_type="training_run")
        run.assert_called_once()
        self.assertEqual(profiler.step.call_count, 7)

    def test_nsight_failure_preserves_completed_update_summary(self):
        config = load_training_config(CONFIG)
        collector = SimpleNamespace(step=Mock(), memory_samples=[{"allocated_bytes": 42}])
        context = MagicMock()
        context.__enter__.return_value = collector
        def shared(received, *, experiment):
            experiment.runtime = SimpleNamespace(device=torch.device("cuda:0"))
            with ExitStack() as stack:
                experiment.start(stack, config, SimpleNamespace(flux_compute_dtype=torch.bfloat16), None)
                for step in range(2):
                    experiment.after_step(step + 1, None,
                        SimpleNamespace(loss=0.5, target_count=48, flux_forward_calls=8), None)
                raise RuntimeError("training failed")
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {"DIFFUSION_ARC_PROFILER_OUTPUT_ROOT": directory}), \
                patch.object(profile, "run_benchmark", side_effect=shared), \
                patch.object(profile, "version", return_value="test"), \
                patch.object(torch.cuda, "get_device_name", return_value="test GPU"), \
                patch.object(profile, "nsight_capture", return_value=context), \
                self.assertRaisesRegex(RuntimeError, "training failed"):
            try:
                profile.profile_from_config(CONFIG, profiler_kind="nsys")
            finally:
                summary = json.loads((Path(directory) / "training-summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["completed_updates"], 2)
                self.assertEqual(summary["recorded_updates"], 0)
                self.assertEqual(summary["pytorch_memory"], collector.memory_samples)
        self.assertEqual(collector.step.call_count, 2)
        context.__exit__.assert_called_once()

    def test_invalid_capture_length_fails_before_training(self):
        config = load_training_config(CONFIG)
        config = replace(config, loop=replace(config.loop, max_optimizer_steps=6))
        with patch.object(profile, "load_training_config", return_value=config), \
                patch.object(profile, "run_benchmark") as run:
            for kind in ("torch", "nsys"):
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "exactly 7"):
                    profile.profile_from_config(CONFIG, profiler_kind=kind)
        run.assert_not_called()

    def test_cli_accepts_only_the_required_config_option(self):
        with patch.object(profile, "profile_from_config") as train:
            self.assertEqual(profile.main(["--config", "run.json"]), 0)
        train.assert_called_once_with(Path("run.json"))
        with self.assertRaises(SystemExit), patch("sys.stderr", Mock()):
            profile.main([])


if __name__ == "__main__":
    unittest.main()
