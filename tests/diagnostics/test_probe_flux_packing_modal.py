"""Modal launcher checks with no image build, network or GPU allocation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_flux_packing as probe
from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App
from tests.diagnostics.test_probe_nvarc_packing_modal import Image

CONFIG = "configs/diagnostics/flux_packing.json"
TRAINING_CONFIG = "configs/training/benchmarks/single_gpu_b2.json"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.image = Image()
        self.volume = SimpleNamespace(commit=Mock())
        self.lookup = Mock(return_value=self.volume)
        class RecordingApp(_App):
            def function(self, **options):
                decorate = super().function(**options)
                def record(function):
                    if not hasattr(self, "options_by_function"):
                        self.options_by_function = {}
                    self.options_by_function[function.__name__] = options
                    return decorate(function)
                return record
        fake_modal = SimpleNamespace(App=RecordingApp, is_local=lambda: True, __version__="test",
            Volume=SimpleNamespace(from_name=self.lookup),
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)))
        spec = importlib.util.spec_from_file_location("_flux_packing_modal_test",
            probe.REPOSITORY_ROOT / "scripts/cloud/probe_flux_packing_modal.py")
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}):
            spec.loader.exec_module(self.launcher)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)

    def layout(self, directory):
        directory = directory.resolve()
        repository, volume = directory / "repository", directory / "volume"
        for relative in (CONFIG, TRAINING_CONFIG):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((probe.REPOSITORY_ROOT / relative).read_text())
        for name in ("nvarc-qwen3-4b-grids15-sft139", "flux2-klein-base-4b"):
            (volume / name).mkdir(parents=True)
        return repository, volume

    def test_pinned_cpu_image_keeps_backward_varlen_and_hdim128(self):
        self.lookup.assert_called_once_with("diffusion-arc-models", create_if_missing=False)
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["timeout"], 2700)
        for key in ("BACKWARD", "VARLEN", "HDIM128"):
            self.assertEqual(self.image.variables[f"FLASH_ATTENTION_DISABLE_{key}"], "FALSE")
        fa3_commands = self.image.build_commands[1][0]
        self.assertTrue(any("#subdirectory=hopper" in c and self.launcher.FA3_REVISION in c for c in fa3_commands))
        self.assertTrue(any("assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_BACKWARD']" in c for c in fa3_commands))
        self.assertTrue(all(kwargs.get("gpu") is None for _, kwargs in self.image.build_commands))
        self.assertEqual([p.name for p in self.image.directories], ["src", "scripts", "configs", "data"])
        self.assertFalse(self.launcher._ignore_data(Path("benchmarks/single_gpu_b0/splits/train/config.json")))
        self.assertTrue(self.launcher._ignore_data(Path("unrelated_dataset/private.json")))

    def test_local_validation_hashes_configs_and_data_before_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                 patch.object(probe, "source_metadata", return_value={"source_sha256": {"x": "y"}}), \
                 patch.object(probe, "dataset_hashes", return_value={"data": "digest"}):
                self.launcher.main()
            call = self.launcher.probe_flux_packing.remote.call_args.args
            self.assertEqual(call[0], CONFIG)
            self.assertEqual(call[1]["dataset_sha256"], {"data": "digest"})
            self.assertEqual(set(call[1]["config_sha256"]), {"probe", "training"})
            self.assertFalse((repository / "checkpoints").exists())

    def test_excessive_budget_and_invalid_path_fail_before_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            raw = json.loads((repository / CONFIG).read_text())
            raw["max_duration_seconds"] = 2700
            (repository / CONFIG).write_text(json.dumps(raw))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                 self.assertRaisesRegex(ValueError, "120 seconds"):
                self.launcher.main()
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        self.launcher.probe_flux_packing.remote.assert_not_called()

    def test_remote_preserves_results_and_reports_failure_phase(self):
        for status in ("complete", "timeout", "oom"):
            with tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()
                def run(args, *, metadata, on_progress):
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-packing"))
                    self.assertEqual(metadata["modal_resources"]["gpu"], "H200")
                    on_progress()
                    return dict(status=status, phase="long_b2/fa4/backward", error="original error")
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     patch.object(probe, "run_probe", side_effect=run):
                    if status == "complete":
                        self.launcher.probe_flux_packing(CONFIG, {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, f"{status} during long_b2/fa4/backward"):
                            self.launcher.probe_flux_packing(CONFIG, {})
                self.assertEqual(self.volume.commit.call_count, 2)

    def test_existing_checkpoint_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            (repository / "checkpoints").mkdir()
            marker = repository / "checkpoints/user-file"
            marker.write_text("preserve")
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                 patch.object(self.launcher, "VOLUME_ROOT", volume), \
                 self.assertRaisesRegex(RuntimeError, "checkpoint path already exists"):
                self.launcher.probe_flux_packing(CONFIG, {})
            self.assertEqual(marker.read_text(), "preserve")

    def test_correctness_dispatch_has_separate_ten_minute_gpu_limit(self):
        options = self.launcher.app.options_by_function["probe_flux_packing_correctness"]
        self.assertEqual(options["timeout"], 600)
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                 patch.object(probe, "source_metadata", return_value={}), \
                 patch.object(probe, "dataset_hashes", return_value={}):
                self.launcher.main(correctness=True)
        self.launcher.probe_flux_packing.remote.assert_not_called()
        self.launcher.probe_flux_packing_correctness.remote.assert_called_once()

    def test_completed_correctness_report_can_contain_failed_numerics(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            def run(args, *, metadata, on_progress):
                self.assertTrue(args.correctness)
                self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-packing-correctness"))
                self.assertEqual(metadata["modal_resources"]["timeout"], 600)
                return {"status": "complete", "numerical_passed": False}
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                 patch.object(self.launcher, "VOLUME_ROOT", volume), \
                 patch.object(probe, "run_probe", side_effect=run):
                self.launcher.probe_flux_packing_correctness(CONFIG, {})

    def test_forward_dispatch_uses_one_h200_with_ten_minute_limit(self):
        options = self.launcher.app.options_by_function["probe_flux_packing_forward"]
        self.assertEqual(options["timeout"], 600)
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                 patch.object(probe, "source_metadata", return_value={}), \
                 patch.object(probe, "dataset_hashes", return_value={}):
                self.launcher.main(forward_only=True)
                _, policy, _ = self.launcher._validate_layout(repository, CONFIG,
                    checkpoint_root=repository / "checkpoints", require_checkpoints=False, forward_only=True)
        self.assertEqual(policy.max_duration_seconds, 480)
        self.launcher.probe_flux_packing.remote.assert_not_called()
        self.launcher.probe_flux_packing_correctness.remote.assert_not_called()
        self.launcher.probe_flux_packing_forward.remote.assert_called_once()

    def test_forward_results_are_separate_and_failures_preserve_their_phase(self):
        for status in ("complete", "timeout", "oom"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()
                def run(args, *, metadata, on_progress):
                    self.assertTrue(args.forward_only)
                    self.assertFalse(args.correctness)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-packing-forward"))
                    self.assertEqual(metadata["modal_resources"]["timeout"], 600)
                    on_progress()
                    return dict(status=status, phase="fa3_ab/forward", numerical_passed=False,
                                error="original error")
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     patch.object(probe, "run_probe", side_effect=run) as launched:
                    if status == "complete":
                        self.launcher.probe_flux_packing_forward(CONFIG, {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, f"{status} during fa3_ab/forward"):
                            self.launcher.probe_flux_packing_forward(CONFIG, {})
                launched.assert_called_once()
                self.assertEqual(self.volume.commit.call_count, 2)

    def test_conflicting_modes_fail_before_remote_or_checkpoint_link(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.launcher.main(correctness=True, forward_only=True)
        self.launcher.probe_flux_packing.remote.assert_not_called()
        self.launcher.probe_flux_packing_correctness.remote.assert_not_called()
        self.launcher.probe_flux_packing_forward.remote.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                 patch.object(self.launcher, "VOLUME_ROOT", volume), \
                 self.assertRaisesRegex(ValueError, "mutually exclusive"):
                self.launcher._run_remote(CONFIG, {}, correctness=True, forward_only=True)
            self.assertFalse((repository / "checkpoints").exists())
