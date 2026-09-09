"""Offline checks of the capacity launcher with Modal calls replaced by fakes."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_nvarc_batch_capacity as probe


class _Image:
    object_id = "test-image"

    def __init__(self):
        self.directories = []

    def __getattr__(self, name):
        if name in {"entrypoint", "apt_install", "pip_install", "env", "run_commands"}:
            return lambda *args, **kwargs: self
        raise AttributeError(name)

    def pip_install_from_pyproject(self, *_args):
        return self

    def workdir(self, *_args):
        return self

    def add_local_dir(self, path, **_kwargs):
        self.directories.append(Path(path))
        return self


class _App:
    def __init__(self, _name):
        self.function_options = None

    def function(self, **options):
        self.function_options = options

        def decorate(function):
            function.remote = Mock(return_value="remote-results")
            return function

        return decorate

    def local_entrypoint(self):
        return lambda function: function


class CapacityModalTests(unittest.TestCase):
    def setUp(self):
        self.image = _Image()
        self.volume = SimpleNamespace(commit=Mock())
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test-modal",
            Volume=SimpleNamespace(from_name=Mock(return_value=self.volume)),
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)),
        )
        spec = importlib.util.spec_from_file_location(
            "_capacity_modal_test_module",
            probe.REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_batch_capacity_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}):
            spec.loader.exec_module(self.launcher)
        # Launcher calls must not leave temporary test roots on the import path.
        self.path_patch = patch.object(sys, "path", list(sys.path))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def _remote_layout(self, root):
        root = root.resolve()
        repository = root / "repo"
        volume = root / "volume"
        for name in ("nvarc-qwen3-4b-grids15-sft139", "flux2-klein-base-4b"):
            (volume / name).mkdir(parents=True)
        relative = "configs/training/benchmarks/single_gpu_b3.json"
        config_path = repository / relative
        config_path.parent.mkdir(parents=True)
        config_path.write_text((probe.REPOSITORY_ROOT / relative).read_text())
        return repository, volume, relative

    def test_declares_one_bounded_h200_function_and_uploads_only_code_and_configs(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["cpu"], 8.0)
        self.assertEqual(options["memory"], 64 * 1024)
        self.assertEqual(options["timeout"], 1800)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual([path.name for path in self.image.directories], ["src", "scripts", "configs"])

    def test_local_entrypoint_forwards_options_and_local_provenance(self):
        metadata = {"git_commit": "local-head", "git_status": " M work", "source_sha256": {}}
        config_path = probe.REPOSITORY_ROOT / "configs/training/benchmarks/single_gpu_b3.json"
        # Local checkpoint files are unnecessary: they live on the remote Volume.
        with patch.object(probe, "_source_metadata", return_value=metadata):
            self.launcher.main(
                "configs/training/benchmarks/single_gpu_b3.json",
                sequence_length=8400, max_batch_size=7, repetitions=2,
            )
        call = self.launcher.probe_nvarc_batch_capacity.remote.call_args.args
        self.assertEqual(call[:4], ("configs/training/benchmarks/single_gpu_b3.json", 8400, 7, 2))
        self.assertEqual(call[4]["git_commit"], "local-head")
        self.assertEqual(call[4]["git_status"], " M work")
        self.assertEqual(call[4]["modal_version"], "test-modal")
        self.assertEqual(call[4]["image_id"], "test-image")
        self.assertEqual(call[4]["config_sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())

    def test_local_preflight_needs_neither_checkpoint_files_nor_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            relative = "configs/training/benchmarks/single_gpu_b3.json"
            config_path = root / relative
            config_path.parent.mkdir(parents=True)
            config_path.write_text((probe.REPOSITORY_ROOT / relative).read_text())
            with (
                patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", root),
                patch.object(probe, "_source_metadata", return_value={}),
            ):
                self.launcher.main(relative)
            self.launcher.probe_nvarc_batch_capacity.remote.assert_called_once()
            self.assertFalse((root / "checkpoints").exists())
            self.assertFalse((root / "data").exists())

    def test_invalid_options_or_config_fail_before_requesting_gpu(self):
        for options in (
            {"sequence_length": 8192}, {"max_batch_size": 0}, {"repetitions": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.launcher.main("configs/training/benchmarks/single_gpu_b3.json", **options)
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        with self.assertRaises(FileNotFoundError):
            self.launcher.main("configs/missing-capacity-config.json")
        self.launcher.probe_nvarc_batch_capacity.remote.assert_not_called()

    def test_remote_search_passes_persistence_hook_and_commits_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume, relative = self._remote_layout(Path(directory))
            metadata = {"git_commit": "local-head"}

            def search(args, *, metadata, on_candidate_end):
                self.assertEqual(args.config, repository / relative)
                self.assertTrue(args.output_dir.is_relative_to(volume / "results/nvarc-capacity"))
                self.assertEqual((args.sequence_length, args.max_batch_size, args.repetitions), (8400, 7, 2))
                self.assertEqual(metadata["git_commit"], "local-head")
                self.assertEqual(metadata["modal_resources"]["timeout"], 1800)
                self.assertIs(on_candidate_end, self.volume.commit)
                on_candidate_end()
                return {"status": "complete"}

            with (
                patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository),
                patch.object(self.launcher, "VOLUME_ROOT", volume),
                patch.object(probe, "run_search", side_effect=search) as run,
            ):
                result = self.launcher.probe_nvarc_batch_capacity(relative, 8400, 7, 2, metadata)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(self.volume.commit.call_count, 2)
            self.assertTrue(Path(result).is_relative_to(volume / "results/nvarc-capacity"))
            self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_remote_failure_and_exception_commit_and_propagate(self):
        for raises in (False, True):
            with self.subTest(raises=raises), tempfile.TemporaryDirectory() as directory:
                self.volume.commit.reset_mock()
                repository, volume, relative = self._remote_layout(Path(directory))

                def search(_args, **kwargs):
                    kwargs["on_candidate_end"]()
                    if raises:
                        raise RuntimeError("worker exception")
                    return {"status": "error"}

                with (
                    patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository),
                    patch.object(self.launcher, "VOLUME_ROOT", volume),
                    patch.object(probe, "run_search", side_effect=search),
                    self.assertRaisesRegex(RuntimeError, "worker exception" if raises else "search failed"),
                ):
                    self.launcher.probe_nvarc_batch_capacity(relative, 8400, 64, 3, {})
                self.assertEqual(self.volume.commit.call_count, 2)

    def test_remote_rejects_unexpected_checkpoint_layout_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume, relative = self._remote_layout(Path(directory))
            (repository / "checkpoints").mkdir()
            marker = repository / "checkpoints/user-file"
            marker.write_text("preserve")
            with (
                patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository),
                patch.object(self.launcher, "VOLUME_ROOT", volume),
                patch.object(probe, "run_search") as run,
                self.assertRaisesRegex(RuntimeError, "checkpoint path already exists"),
            ):
                self.launcher.probe_nvarc_batch_capacity(relative, 8400, 64, 3, {})
            run.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
