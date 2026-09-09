"""Validate the BFL comparison launcher without builds, network or GPU calls."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/diagnostics/flux_bfl.json"
INPUT_CONFIG = "configs/diagnostics/flux_packing.json"
TRAINING_CONFIG = "configs/training/benchmarks/single_gpu_b2.json"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        from scripts.diagnostics import probe_flux_bfl

        self.probe = probe_flux_bfl
        self.image = SimpleNamespace(object_id="existing-pinned-image")
        self.volume = SimpleNamespace(commit=Mock())
        infrastructure = SimpleNamespace(
            DATA_ROOT=Path("benchmarks/single_gpu_b0"), LOCAL_REPOSITORY_ROOT=REPOSITORY_ROOT,
            REMOTE_REPOSITORY_ROOT=Path("/opt/diffusion-arc"),
            VOLUME_ROOT=Path("/mnt/diffusion-arc-models"), image=self.image, model_volume=self.volume,
        )
        fake_modal = SimpleNamespace(App=_App, __version__="test-modal", is_local=lambda: True)
        spec = importlib.util.spec_from_file_location("_flux_bfl_modal_test",
            REPOSITORY_ROOT / "scripts/cloud/probe_flux_bfl_modal.py")
        self.launcher = importlib.util.module_from_spec(spec)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)
        with patch.dict(sys.modules, {"modal": fake_modal,
                "scripts.cloud.probe_flux_packing_modal": infrastructure}):
            spec.loader.exec_module(self.launcher)

    def layout(self, directory):
        repository, volume = directory.resolve() / "repository", directory.resolve() / "volume"
        for relative in (CONFIG, INPUT_CONFIG, TRAINING_CONFIG):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((REPOSITORY_ROOT / relative).read_text())
        for name in ("nvarc-qwen3-4b-grids15-sft139", "flux2-klein-base-4b"):
            (volume / name).mkdir(parents=True)
        return repository, volume

    def test_reuses_image_and_volume_with_one_bounded_h200(self):
        options = self.launcher.app.function_options
        self.assertIs(options["image"], self.image)
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.assertEqual({key: options[key] for key in ("gpu", "cpu", "memory", "timeout")},
                         {"gpu": "H200", "cpu": 8.0, "memory": 65536, "timeout": 600})
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["max_containers"], 1)

    def test_remote_import_uses_mounted_repository_when_modal_relocates_launcher(self):
        namespace = {"__file__": "/root/probe_flux_bfl_modal.py", "__name__": "_remote_bfl_launcher"}
        infrastructure = SimpleNamespace(
            DATA_ROOT=Path("benchmarks/single_gpu_b0"), LOCAL_REPOSITORY_ROOT=Path("/opt/diffusion-arc"),
            REMOTE_REPOSITORY_ROOT=Path("/opt/diffusion-arc"),
            VOLUME_ROOT=Path("/mnt/diffusion-arc-models"), image=self.image, model_volume=self.volume)
        fake_modal = SimpleNamespace(App=_App, __version__="test-modal", is_local=lambda: False)
        source = (REPOSITORY_ROOT / "scripts/cloud/probe_flux_bfl_modal.py").read_text()
        with patch.dict(sys.modules, {"modal": fake_modal,
                "scripts.cloud.probe_flux_packing_modal": infrastructure}):
            exec(compile(source, namespace["__file__"], "exec"), namespace)
        self.assertEqual(namespace["REPOSITORY_ROOT"], Path("/opt/diffusion-arc"))

    def test_local_validation_hashes_configs_and_data_before_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                 patch.object(self.probe, "source_metadata", return_value={"source_sha256": {"x": "y"}}), \
                 patch.object(self.probe, "dataset_hashes", return_value={"data": "digest"}):
                self.launcher.main()
            relative, metadata = self.launcher.probe_flux_bfl.remote.call_args.args
            self.assertEqual(relative, CONFIG)
            self.assertEqual(metadata["dataset_sha256"], {"data": "digest"})
            self.assertEqual(metadata["config_sha256"], self.probe.config_hashes(
                repository / CONFIG,
                self.probe.load_probe_config(repository / CONFIG, require_checkpoints=False)[0]))
            self.assertEqual(metadata["source_sha256"], {"x": "y"})
            self.assertEqual(metadata["image_id"], "existing-pinned-image")
            self.assertFalse((repository / "checkpoints").exists())

    def test_paths_dataset_and_worker_budget_are_checked_before_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            valid_policy, valid_training = self.probe.load_probe_config(
                repository / CONFIG, require_checkpoints=False)
            outside = repository / "outside.json"
            for invalid in ("input_config", "training_config", "checkpoint", "dataset", "budget"):
                policy = SimpleNamespace(input_config=valid_policy.input_config,
                    training_config=valid_policy.training_config, max_duration_seconds=480)
                training = SimpleNamespace(models=SimpleNamespace(
                    nvarc_checkpoint=valid_training.models.nvarc_checkpoint,
                    flux_checkpoint=valid_training.models.flux_checkpoint),
                    data=SimpleNamespace(split_directory=valid_training.data.split_directory))
                if invalid in ("input_config", "training_config"):
                    setattr(policy, invalid, outside)
                elif invalid == "checkpoint":
                    training.models.flux_checkpoint = outside
                elif invalid == "dataset":
                    training.data.split_directory = repository / "data/private/splits/train"
                else:
                    policy.max_duration_seconds = 481
                with self.subTest(invalid=invalid), \
                     patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                     patch.object(self.probe, "load_probe_config", return_value=(policy, training)), \
                     self.assertRaises(ValueError):
                    self.launcher.main()
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        self.launcher.probe_flux_bfl.remote.assert_not_called()

    def test_remote_persists_partial_results_and_preserves_failure_phase(self):
        for status in ("complete", "timeout", "oom", "exception"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()

                def run(args, *, metadata, on_progress):
                    self.assertEqual(args.config, repository / CONFIG)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-bfl"))
                    self.assertEqual(metadata["modal_resources"]["timeout"], 600)
                    self.assertEqual(metadata["git_commit"], "local-head")
                    on_progress()
                    if status == "exception":
                        raise RuntimeError("worker exception")
                    return dict(status=status, phase="fp32/bfl_a", numerical_passed=False,
                                error="original error")

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     patch.object(self.probe, "run_probe", side_effect=run) as worker:
                    if status == "complete":
                        result = self.launcher.probe_flux_bfl(CONFIG, {"git_commit": "local-head"})
                        self.assertTrue(Path(result).is_relative_to(volume / "results/flux-bfl"))
                    else:
                        message = "worker exception" if status == "exception" else f"{status} during fp32/bfl_a"
                        with self.assertRaisesRegex(RuntimeError, message):
                            self.launcher.probe_flux_bfl(CONFIG, {"git_commit": "local-head"})
                worker.assert_called_once()
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual((repository / "checkpoints").resolve(), volume)

    def test_unexpected_checkpoint_paths_are_preserved(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                destination = repository / "user-checkpoints" if symlink else repository / "checkpoints"
                destination.mkdir()
                marker = destination / "user-file"
                marker.write_text("preserve")
                if symlink:
                    (repository / "checkpoints").symlink_to(destination, target_is_directory=True)
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                     patch.object(self.launcher, "VOLUME_ROOT", volume), \
                     patch.object(self.probe, "run_probe") as worker, \
                     self.assertRaisesRegex(RuntimeError, "checkpoint"):
                    self.launcher.probe_flux_bfl(CONFIG, {})
                self.assertEqual(marker.read_text(), "preserve")
                worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
