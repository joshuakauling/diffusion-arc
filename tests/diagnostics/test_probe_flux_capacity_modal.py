"""Offline checks of the bounded synthetic FLUX/FSDP2 Modal launcher."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_flux_capacity as probe
from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App, _Image


ROOT = Path(__file__).resolve().parents[2]


class FluxCapacityModalTests(unittest.TestCase):
    def setUp(self):
        self.image = _Image()
        self.volume = SimpleNamespace(commit=Mock())
        self.volume_factory = Mock(return_value=self.volume)
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test-modal",
            Volume=SimpleNamespace(from_name=self.volume_factory),
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)),
        )
        spec = importlib.util.spec_from_file_location(
            "_flux_capacity_modal_test", ROOT / "scripts/cloud/probe_flux_capacity_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}), patch.object(sys, "path", list(sys.path)):
            spec.loader.exec_module(self.launcher)
        self.path_patch = patch.object(sys, "path", list(sys.path))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def make_repository(self, root):
        repository = Path(root).resolve()
        config_path = repository / self.launcher.CONFIG
        config_path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / self.launcher.CONFIG, config_path)
        shutil.copyfile(ROOT / "configs/training/arc_agi_1_epoch.json",
                        repository / "configs/training/arc_agi_1_epoch.json")
        return repository

    def remote_layout(self, directory):
        root = Path(directory).resolve()
        repository = self.make_repository(root / "repo")
        volume = root / "volume"
        for name in ("nvarc-qwen3-4b-grids15-sft139", "flux2-klein-base-4b"):
            (volume / name).mkdir(parents=True)
        return repository, volume

    def test_fixed_two_h200_budget_cached_volume_and_code_only_upload(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200:2")
        self.assertEqual(options["cpu"], (16.0, 16.0))
        self.assertEqual(options["memory"], (128 * 1024, 128 * 1024))
        self.assertEqual(options["timeout"], 1800)
        self.assertEqual((options["max_containers"], options["retries"]), (1, 0))
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.volume_factory.assert_called_once_with("diffusion-arc-models", create_if_missing=False)
        self.assertEqual([path.relative_to(ROOT).as_posix() for path in self.image.directories],
                         ["src", "scripts", "configs"])

    def test_defaults_forward_provenance_without_local_weights_or_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            identity = dict(source_sha256="source-hash", git_commit="head", git_status=" M work")
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                    patch.object(probe, "source_metadata", return_value=identity), \
                    patch.object(probe, "load_probe_config", wraps=probe.load_probe_config) as load:
                self.launcher.main()
            load.assert_called_once_with(repository / self.launcher.CONFIG, require_checkpoints=False)
            arguments = self.launcher.probe_flux_capacity.remote.call_args.args
            self.assertEqual(arguments[:5], (self.launcher.CONFIG, 12288, 1024, 49152, 2))
            self.assertTrue(arguments[6])
            self.assertEqual(len(arguments), 7)
            metadata = arguments[5]
            self.assertEqual({key: metadata[key] for key in identity}, identity)
            self.assertEqual(metadata["command"], sys.argv)
            self.assertEqual(metadata["modal_version"], "test-modal")
            self.assertEqual(metadata["image_id"], "test-image")
            self.assertEqual(metadata["config_sha256"],
                             hashlib.sha256((repository / self.launcher.CONFIG).read_bytes()).hexdigest())
            self.assertFalse((repository / "checkpoints").exists())
            self.assertFalse((repository / "data").exists())

    def test_invalid_options_or_config_reject_before_gpu_dispatch(self):
        for options in (dict(start_tokens=0), dict(step_tokens=0), dict(max_tokens=12287), dict(updates=0),
                        dict(reshard_after_forward="false")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.launcher.main(**options)
        with self.assertRaises(ValueError):
            self.launcher.main(config="../outside.json")
        with self.assertRaises(FileNotFoundError):
            self.launcher.main(config="configs/missing.json")
        with tempfile.TemporaryDirectory() as directory:
            repository = self.make_repository(directory)
            config_path = repository / self.launcher.CONFIG
            payload = json.loads(config_path.read_text())
            for strategy, gpu_count in (("ddp", 2), ("fsdp2", 1)):
                payload["execution"].update(distributed_strategy=strategy, gpu_count=gpu_count)
                config_path.write_text(json.dumps(payload))
                with self.subTest(strategy=strategy, gpu_count=gpu_count), \
                        patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), \
                        self.assertRaises(ValueError):
                    self.launcher.main()
        self.launcher.probe_flux_capacity.remote.assert_not_called()

    def test_no_forward_reshard_option_is_forwarded_to_the_remote_search(self):
        with patch.object(probe, "source_metadata", return_value={}):
            self.launcher.main(reshard_after_forward=False)
        arguments = self.launcher.probe_flux_capacity.remote.call_args.args
        self.assertEqual(arguments[:5], (self.launcher.CONFIG, 12288, 1024, 49152, 2))
        self.assertFalse(arguments[6])
        self.assertEqual(len(arguments), 7)

    def test_both_model_paths_must_remain_in_the_cached_checkpoint_mount(self):
        for model_path in ("nvarc_checkpoint", "flux_checkpoint"):
            with self.subTest(model_path=model_path), tempfile.TemporaryDirectory() as directory:
                repository = self.make_repository(Path(directory) / "repo")
                recipe_path = repository / "configs/training/arc_agi_1_epoch.json"
                recipe = json.loads(recipe_path.read_text())
                recipe["models"][model_path] = str(Path(directory).resolve() / "outside-models" / model_path)
                recipe_path.write_text(json.dumps(recipe))
                with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(ValueError):
                    self.launcher.main()
        self.launcher.probe_flux_capacity.remote.assert_not_called()

    def test_complete_and_time_limited_search_keep_candidates_and_commit_finally(self):
        for status in ("complete", "time_limit"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                self.volume.commit.reset_mock()
                repository, volume = self.remote_layout(directory)

                def search(args, *, metadata, on_candidate_end, timeout_seconds):
                    self.assertEqual(args.config, repository / self.launcher.CONFIG)
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/flux-capacity"))
                    self.assertEqual((args.start_tokens, args.step_tokens, args.max_tokens, args.updates),
                                     (15360, 1024, 17408, 2))
                    self.assertFalse(args.reshard_after_forward)
                    self.assertEqual(timeout_seconds, 1710)
                    self.assertEqual(metadata["git_commit"], "head")
                    self.assertEqual(metadata["modal_resources"], self.launcher.RESOURCES)
                    self.assertEqual(metadata["fa3_revision"], self.launcher.FA3_REVISION)
                    self.assertIs(on_candidate_end, self.volume.commit)
                    args.output_dir.mkdir(parents=True)
                    (args.output_dir / "candidate.json").write_text('{"status":"ok"}')
                    on_candidate_end()
                    return dict(status=status)

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(probe, "load_probe_config", wraps=probe.load_probe_config) as load, \
                        patch.object(probe, "run_search", side_effect=search) as run:
                    result = self.launcher.probe_flux_capacity(
                        self.launcher.CONFIG, 15360, 1024, 17408, 2, dict(git_commit="head"),
                        reshard_after_forward=False,
                    )
                run.assert_called_once()
                load.assert_called_once_with(repository / self.launcher.CONFIG, require_checkpoints=True)
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual((repository / "checkpoints").resolve(), volume)
                self.assertTrue((Path(result) / "candidate.json").is_file())
                self.assertFalse((repository / "data").exists())

    def test_search_error_and_worker_exception_commit_partial_results_and_propagate(self):
        for raises in (False, True):
            with self.subTest(raises=raises), tempfile.TemporaryDirectory() as directory:
                self.volume.commit.reset_mock()
                repository, volume = self.remote_layout(directory)

                def search(args, **kwargs):
                    args.output_dir.mkdir(parents=True)
                    (args.output_dir / "partial.json").write_text('{"status":"partial"}')
                    kwargs["on_candidate_end"]()
                    if raises:
                        raise RuntimeError("worker exception")
                    return dict(status="error")

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(probe, "run_search", side_effect=search), \
                        self.assertRaisesRegex(RuntimeError, "worker exception" if raises else "search failed"):
                    self.launcher.probe_flux_capacity(self.launcher.CONFIG, 12288, 1024, 49152, 2, {})
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertEqual(len(list((volume / "results/flux-capacity").glob("*/partial.json"))), 1)

    def test_unexpected_checkpoint_directory_or_symlink_is_preserved(self):
        for wrong_symlink in (False, True):
            with self.subTest(wrong_symlink=wrong_symlink), tempfile.TemporaryDirectory() as directory:
                self.volume.commit.reset_mock()
                repository, volume = self.remote_layout(directory)
                link = repository / "checkpoints"
                if wrong_symlink:
                    outside = Path(directory).resolve() / "other-models"
                    outside.mkdir()
                    link.symlink_to(outside, target_is_directory=True)
                else:
                    link.mkdir()
                marker = link / "user-file"
                marker.write_text("preserve")
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(probe, "run_search") as run, \
                        self.assertRaisesRegex(RuntimeError, "unexpected checkpoint symlink" if wrong_symlink
                                               else "checkpoint path already exists"):
                    self.launcher.probe_flux_capacity(self.launcher.CONFIG, 12288, 1024, 49152, 2, {})
                run.assert_not_called()
                self.assertEqual(marker.read_text(), "preserve")
                self.volume.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
