"""Offline checks of the FLUX probe launcher; Modal execution is always mocked."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_flux_batch_shapes as probe
from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App, _Image


CONFIG = "configs/diagnostics/flux_batch_shapes.json"
TRAINING_CONFIG = "configs/training/benchmarks/single_gpu_b2_preparation.json"


class _ProbeImage(_Image):
    def __init__(self):
        super().__init__()
        self.files = []

    def add_local_file(self, path, **_kwargs):
        self.files.append(Path(path))
        return self


class FluxProbeModalTests(unittest.TestCase):
    def setUp(self):
        self.image = _ProbeImage()
        self.volume = SimpleNamespace(commit=Mock())
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test-modal",
            Volume=SimpleNamespace(from_name=Mock(return_value=self.volume)),
            Image=SimpleNamespace(debian_slim=Mock(return_value=self.image)),
        )
        spec = importlib.util.spec_from_file_location(
            "_flux_shape_modal_test", probe.REPOSITORY_ROOT / "scripts/cloud/probe_flux_batch_shapes_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}):
            spec.loader.exec_module(self.launcher)
        self.path_patch = patch.object(sys, "path", list(sys.path))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def layout(self, root):
        repository, volume = root.resolve() / "repository", root.resolve() / "volume"
        for relative in (CONFIG, TRAINING_CONFIG):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((probe.REPOSITORY_ROOT / relative).read_text())
        for name in ("nvarc-qwen3-4b-grids15-sft139", "flux2-klein-base-4b"):
            (volume / name).mkdir(parents=True)
        return repository, volume

    def test_single_h200_no_retries_bounded_timeout_and_no_dataset_or_trace_upload(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["timeout"], 1800)
        self.assertEqual([path.name for path in self.image.directories], ["src", "scripts", "configs"])
        self.assertEqual([path.name for path in self.image.files], ["pyproject.toml"])

    def test_local_launch_checks_both_configs_without_checkpoint_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _volume = self.layout(Path(directory))
            with (
                patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository),
                patch.object(probe, "source_metadata", return_value={"git_commit": "head", "git_status": " M work", "source_sha256": {}}),
            ):
                self.launcher.main()
            call = self.launcher.probe_flux_batch_shapes.remote.call_args.args
            self.assertEqual(call[0], CONFIG)
            self.assertEqual(call[1]["git_commit"], "head")
            self.assertEqual(set(call[1]["config_sha256"]), {"probe", "training"})
            self.assertEqual(call[1]["image_id"], "test-image")
            self.assertFalse((repository / "checkpoints").exists())
            self.assertFalse((repository / "data").exists())

    def test_invalid_path_and_excessive_budget_fail_before_remote_call(self):
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        with tempfile.TemporaryDirectory() as directory:
            repository, _volume = self.layout(Path(directory))
            raw = json.loads((repository / CONFIG).read_text())
            raw["max_duration_seconds"] = 1800
            (repository / CONFIG).write_text(json.dumps(raw))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaisesRegex(ValueError, "60 seconds"):
                self.launcher.main()
        self.launcher.probe_flux_batch_shapes.remote.assert_not_called()

    def test_remote_persists_progress_and_propagates_incomplete_sweeps(self):
        for status in ("complete", "timeout", "error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()

                def run(args, *, metadata, on_progress):
                    self.assertEqual(args.config, repository / CONFIG)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-batch-shapes"))
                    self.assertEqual(metadata["modal_resources"]["gpu"], "H200")
                    self.assertIs(on_progress, self.volume.commit)
                    on_progress()
                    return {"status": status}

                with (
                    patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository),
                    patch.object(self.launcher, "VOLUME_ROOT", volume),
                    patch.object(probe, "run_probe", side_effect=run) as run_mock,
                ):
                    if status == "complete":
                        result = self.launcher.probe_flux_batch_shapes(CONFIG, {})
                        self.assertTrue(Path(result).is_relative_to(volume))
                    else:
                        with self.assertRaisesRegex(RuntimeError, status):
                            self.launcher.probe_flux_batch_shapes(CONFIG, {})
                run_mock.assert_called_once()
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_unexpected_checkpoint_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            (repository / "checkpoints").mkdir()
            marker = repository / "checkpoints/user-file"
            marker.write_text("preserve")
            with (
                patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository),
                patch.object(self.launcher, "VOLUME_ROOT", volume),
                patch.object(probe, "run_probe") as run,
                self.assertRaisesRegex(RuntimeError, "checkpoint path already exists"),
            ):
                self.launcher.probe_flux_batch_shapes(CONFIG, {})
            self.assertEqual(marker.read_text(), "preserve")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
