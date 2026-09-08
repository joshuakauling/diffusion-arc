"""Offline checks of NVARC Modal launch, cached checkpoint paths, and persistence."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_nvarc_batch_shapes as probe
from tests.diagnostics.test_probe_flux_batch_shapes_modal import _ProbeImage
from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App


CONFIG = "configs/diagnostics/nvarc_batch_shapes.json"
TRAINING_CONFIG = "configs/training/benchmarks/single_gpu_b2.json"


class NvarcShapeModalTests(unittest.TestCase):
    def setUp(self):
        self.image = _ProbeImage()
        self.volume = SimpleNamespace(commit=Mock())
        self.volume_lookup = Mock(return_value=self.volume)
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test-modal",
            Volume=SimpleNamespace(from_name=self.volume_lookup),
            Image=SimpleNamespace(debian_slim=Mock(return_value=self.image)),
        )
        spec = importlib.util.spec_from_file_location(
            "_nvarc_shape_modal_test", probe.REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_batch_shapes_modal.py",
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

    def test_existing_volume_one_h200_no_retries_and_only_code_uploads(self):
        options = self.launcher.app.function_options
        for name, value in (("gpu", "H200"), ("cpu", 8.0), ("memory", 64 * 1024),
                            ("max_containers", 1), ("retries", 0), ("timeout", 1800)):
            self.assertEqual(options[name], value)
        self.volume_lookup.assert_called_once_with("diffusion-arc-models", create_if_missing=False)
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual([path.name for path in self.image.directories], ["src", "scripts", "configs"])
        self.assertEqual([path.name for path in self.image.files], ["pyproject.toml"])

    def test_local_launch_without_models_forwards_both_hashes_and_local_git(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _volume = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), patch.object(
                probe, "source_metadata", return_value={
                    "git_commit": "local-head", "git_status": " M work", "source_sha256": {"code": "hash"},
                },
            ):
                self.launcher.main()
            call = self.launcher.probe_nvarc_batch_shapes.remote.call_args.args
            self.assertEqual(call[0], CONFIG)
            self.assertEqual(call[1]["git_commit"], "local-head")
            self.assertEqual(call[1]["git_status"], " M work")
            self.assertEqual(set(call[1]["config_sha256"]), {"probe", "training"})
            self.assertEqual(call[1]["image_id"], "test-image")
            self.assertEqual(call[1]["modal_version"], "test-modal")
            self.assertFalse((repository / "checkpoints").exists())
            self.assertFalse((repository / "data").exists())

    def test_excess_budget_and_paths_fail_before_gpu_call(self):
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        for field, value in (("max_duration_seconds", 1800), ("training_config", "../../outside.json")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                repository, _volume = self.layout(Path(directory))
                if field == "training_config":
                    (repository / "outside.json").write_text((repository / TRAINING_CONFIG).read_text())
                raw = json.loads((repository / CONFIG).read_text())
                raw[field] = value
                (repository / CONFIG).write_text(json.dumps(raw))
                with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(ValueError):
                    self.launcher.main()
        self.launcher.probe_nvarc_batch_shapes.remote.assert_not_called()

    def test_checkpoint_escape_fails_locally_and_missing_cache_fails_remotely(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                self.launcher, "VOLUME_ROOT", volume,
            ), patch.object(probe, "run_probe") as run:
                (volume / "nvarc-qwen3-4b-grids15-sft139").rmdir()
                with self.assertRaises(FileNotFoundError):
                    self.launcher.probe_nvarc_batch_shapes(CONFIG, {})
                run.assert_not_called()
            raw = json.loads((repository / TRAINING_CONFIG).read_text())
            raw["models"]["nvarc_checkpoint"] = "/outside/models"
            (repository / TRAINING_CONFIG).write_text(json.dumps(raw))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(ValueError):
                self.launcher.main()
        self.launcher.probe_nvarc_batch_shapes.remote.assert_not_called()

    def test_volume_commits_on_progress_completion_and_probe_failures(self):
        for status in ("complete", "timeout", "error", "exception"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()

                def run(args, *, metadata, on_progress):
                    self.assertEqual(args.config, repository / CONFIG)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/nvarc-batch-shapes"))
                    self.assertEqual(metadata["modal_resources"]["gpu"], "H200")
                    self.assertEqual(metadata["git_commit"], "local-head")
                    self.assertIs(on_progress, self.volume.commit)
                    on_progress()
                    if status == "exception":
                        raise ValueError("test failure")
                    return {"status": status}

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                    self.launcher, "VOLUME_ROOT", volume,
                ), patch.object(probe, "run_probe", side_effect=run) as run_mock:
                    metadata = {"git_commit": "local-head"}
                    if status == "complete":
                        result = self.launcher.probe_nvarc_batch_shapes(CONFIG, metadata)
                        self.assertTrue(Path(result).is_relative_to(volume))
                    elif status == "exception":
                        with self.assertRaisesRegex(ValueError, "test failure"):
                            self.launcher.probe_nvarc_batch_shapes(CONFIG, metadata)
                    else:
                        with self.assertRaisesRegex(RuntimeError, status):
                            self.launcher.probe_nvarc_batch_shapes(CONFIG, metadata)
                run_mock.assert_called_once()
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_worker_oom_reports_original_cause_and_failed_shape(self):
        cause = "OutOfMemoryError: CUDA out of memory. Tried to allocate 410.00 MiB."
        result = {
            "status": "error", "worker_status": "oom", "error": cause,
            "groups": [{"batch_size": 8, "cases": [
                {"padded_length": 2702, "status": "checked"},
                {"padded_length": 2752, "status": "oom", "phase": "output_comparison"},
                {"padded_length": 2816, "status": "pending"},
            ]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                self.launcher, "VOLUME_ROOT", volume,
            ), patch.object(probe, "run_probe", return_value=result):
                with self.assertRaises(RuntimeError) as raised:
                    self.launcher.probe_nvarc_batch_shapes(CONFIG, {})
            message = str(raised.exception)
            self.assertIn(cause, message)
            self.assertIn("8x2752 (output_comparison)", message)
            self.assertIn(f"partial results: {volume}/results/nvarc-batch-shapes/", message)
            self.volume.commit.assert_called_once()

    def test_existing_directory_and_wrong_symlink_are_preserved(self):
        for kind in ("directory", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                link = repository / "checkpoints"
                if kind == "directory":
                    link.mkdir()
                    (link / "user-file").write_text("preserve")
                else:
                    link.symlink_to(repository / "elsewhere", target_is_directory=True)
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                    self.launcher, "VOLUME_ROOT", volume,
                ), patch.object(probe, "run_probe") as run, self.assertRaises(RuntimeError):
                    self.launcher.probe_nvarc_batch_shapes(CONFIG, {})
                run.assert_not_called()
                if kind == "directory":
                    self.assertEqual((link / "user-file").read_text(), "preserve")
                else:
                    self.assertEqual(link.readlink(), repository / "elsewhere")


if __name__ == "__main__":
    unittest.main()
