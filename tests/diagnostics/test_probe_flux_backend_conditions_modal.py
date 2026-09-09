"""Check the conditions launcher without builds, network, or GPU execution."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/diagnostics/flux_backend_conditions.json"
PROBE_MODULE = "scripts.diagnostics.probe_flux_backend_conditions"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.image = SimpleNamespace(object_id="existing-pinned-image")
        self.volume = SimpleNamespace(commit=Mock())
        self.infrastructure = SimpleNamespace(
            DATA_ROOT=Path("benchmarks/single_gpu_b0"), LOCAL_REPOSITORY_ROOT=REPOSITORY_ROOT,
            REMOTE_REPOSITORY_ROOT=Path("/opt/diffusion-arc"),
            VOLUME_ROOT=Path("/mnt/diffusion-arc-models"), image=self.image, model_volume=self.volume,
        )
        fake_modal = SimpleNamespace(App=_App, __version__="test-modal", is_local=lambda: True)
        spec = importlib.util.spec_from_file_location("_flux_backend_conditions_modal_test",
            REPOSITORY_ROOT / "scripts/cloud/probe_flux_backend_conditions_modal.py")
        self.launcher = importlib.util.module_from_spec(spec)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)
        with patch.dict(sys.modules, {"modal": fake_modal,
                "scripts.cloud.probe_flux_packing_modal": self.infrastructure}):
            spec.loader.exec_module(self.launcher)
        self.probe = SimpleNamespace(
            source_metadata=Mock(return_value={"source_sha256": {"script": "digest"}}),
            config_hashes=Mock(return_value={"probe": "config-digest"}),
            dataset_hashes=Mock(return_value={"fixture": "fixture-digest"}),
            run_probe=Mock(), load_probe_config=Mock(),
        )
        modules = patch.dict(sys.modules, {PROBE_MODULE: self.probe})
        modules.start()
        self.addCleanup(modules.stop)

    def test_reuses_image_and_volume_with_one_bounded_h200(self):
        options = self.launcher.app.function_options
        self.assertIs(options["image"], self.image)
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual({key: options[key] for key in ("gpu", "cpu", "memory", "timeout")},
                         {"gpu": "H200", "cpu": 8.0, "memory": 65536, "timeout": 1920})
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)

    def test_remote_import_handles_relocated_launcher(self):
        namespace = {"__file__": "/root/probe_flux_backend_conditions_modal.py",
                     "__name__": "_remote_backend_conditions_launcher"}
        fake_modal = SimpleNamespace(App=_App, __version__="test-modal", is_local=lambda: False)
        source = (REPOSITORY_ROOT / "scripts/cloud/probe_flux_backend_conditions_modal.py").read_text()
        with patch.dict(sys.modules, {"modal": fake_modal,
                "scripts.cloud.probe_flux_packing_modal": self.infrastructure}):
            exec(compile(source, namespace["__file__"], "exec"), namespace)
        self.assertEqual(namespace["REPOSITORY_ROOT"], Path("/opt/diffusion-arc"))

    def test_local_preflight_and_provenance_are_complete_before_gpu_request(self):
        policy = object()
        path = REPOSITORY_ROOT / CONFIG
        with patch.object(self.launcher, "_validate_layout", return_value=(path, policy)) as validate:
            self.launcher.main()
        validate.assert_called_once_with(REPOSITORY_ROOT, CONFIG,
            checkpoint_root=REPOSITORY_ROOT / "checkpoints", require_checkpoints=False)
        self.probe.config_hashes.assert_called_once_with(path, policy)
        self.probe.dataset_hashes.assert_called_once_with(policy)
        relative, metadata = self.launcher.probe_flux_backend_conditions.remote.call_args.args
        self.assertEqual(relative, CONFIG)
        self.assertEqual(metadata["dataset_sha256"], {"fixture": "fixture-digest"})
        self.assertEqual(metadata["config_sha256"], {"probe": "config-digest"})
        self.assertEqual(metadata["source_sha256"], {"script": "digest"})
        self.assertEqual(metadata["image_id"], "existing-pinned-image")
        self.assertEqual(metadata["modal_version"], "test-modal")

    def test_config_fixture_checkpoint_and_budget_bounds_precede_gpu_request(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            config = repository / CONFIG
            fixture = repository / "configs/diagnostics/flux_backend_inputs.json"
            checkpoints = tuple(repository / "checkpoints" / name for name in ("base", "nvarc", "overfit"))
            for invalid in ("config", "fixture", "checkpoint", "budget", "zero_budget", "bool_budget"):
                policy = SimpleNamespace(config_paths=(config,), fixture_paths=(fixture,),
                    checkpoint_paths=checkpoints, max_duration_seconds=1800)
                outside = repository / "outside"
                if invalid == "config":
                    policy.config_paths = (outside,)
                elif invalid == "fixture":
                    policy.fixture_paths = (outside,)
                elif invalid == "checkpoint":
                    policy.checkpoint_paths = (*checkpoints[:-1], outside)
                else:
                    policy.max_duration_seconds = {"budget": 1801, "zero_budget": 0, "bool_budget": True}[invalid]
                self.probe.load_probe_config.return_value = policy
                with self.subTest(invalid=invalid), \
                     patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                     self.assertRaises(ValueError):
                    self.launcher.main()
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(ValueError):
                self.launcher.main("../outside.json")
        self.launcher.probe_flux_backend_conditions.remote.assert_not_called()

    def test_local_checkpoint_files_are_optional_but_remote_directories_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            config = repository / CONFIG
            checkpoint = repository / "checkpoints/base"
            policy = SimpleNamespace(config_paths=(config,), fixture_paths=(),
                                     checkpoint_paths=(checkpoint,), max_duration_seconds=1800)
            self.probe.load_probe_config.return_value = policy
            self.assertEqual(self.launcher._validate_layout(repository, CONFIG,
                checkpoint_root=repository / "checkpoints", require_checkpoints=False), (config, policy))
            self.assertFalse(checkpoint.exists())
            with self.assertRaises(FileNotFoundError):
                self.launcher._validate_layout(repository, CONFIG,
                    checkpoint_root=repository / "checkpoints", require_checkpoints=True)
            checkpoint.mkdir(parents=True)
            self.assertEqual(self.launcher._validate_layout(repository, CONFIG,
                checkpoint_root=repository / "checkpoints", require_checkpoints=True), (config, policy))
            self.probe.load_probe_config.assert_called_with(config, require_checkpoints=True)

    def test_remote_persists_progress_success_and_failure(self):
        for status in ("complete", "timeout", "oom", "exception"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = Path(directory).resolve() / "repository", Path(directory).resolve() / "volume"
                repository.mkdir()
                volume.mkdir()
                self.volume.commit.reset_mock()

                def run(args, *, metadata, on_progress):
                    self.assertEqual(args.config, repository / CONFIG)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-backend-conditions"))
                    self.assertEqual(metadata["modal_resources"]["timeout"], 1920)
                    self.assertEqual(metadata["git_commit"], "local-head")
                    on_progress()
                    if status == "exception":
                        raise RuntimeError("worker exception")
                    return dict(status=status, phase="overfit_seen/fa3", numerical_passed=False,
                                error="original error")

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     patch.object(self.launcher, "_validate_layout",
                                  return_value=(repository / CONFIG, object())) as validate, \
                     patch.object(self.probe, "run_probe", side_effect=run) as worker:
                    if status == "complete":
                        result = self.launcher.probe_flux_backend_conditions(CONFIG, {"git_commit": "local-head"})
                        self.assertTrue(Path(result).is_relative_to(volume / "results/flux-backend-conditions"))
                    else:
                        message = "worker exception" if status == "exception" else f"{status} during overfit_seen/fa3"
                        with self.assertRaisesRegex(RuntimeError, message):
                            self.launcher.probe_flux_backend_conditions(CONFIG, {"git_commit": "local-head"})
                validate.assert_called_once_with(repository, CONFIG,
                    checkpoint_root=volume, require_checkpoints=True)
                worker.assert_called_once()
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_existing_checkpoint_paths_are_preserved(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as directory:
                repository, volume = Path(directory).resolve() / "repository", Path(directory).resolve() / "volume"
                repository.mkdir()
                volume.mkdir()
                destination = repository / ("user-checkpoints" if symlink else "checkpoints")
                destination.mkdir()
                marker = destination / "user-file"
                marker.write_text("preserve")
                if symlink:
                    (repository / "checkpoints").symlink_to(destination, target_is_directory=True)
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     self.assertRaisesRegex(RuntimeError, "checkpoint"):
                    self.launcher.probe_flux_backend_conditions(CONFIG, {})
                self.assertEqual(marker.read_text(), "preserve")
                self.probe.run_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
