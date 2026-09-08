"""Validate the launcher without building an image or contacting Modal."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.diagnostics import probe_nvarc_packing as probe
from tests.diagnostics.test_probe_flux_batch_shapes_modal import _ProbeImage
from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App

CONFIG = "configs/diagnostics/nvarc_packing.json"
TRAINING_CONFIG = "configs/training/benchmarks/single_gpu_b2.json"


class Image(_ProbeImage):
    def __init__(self):
        super().__init__()
        self.build_commands = []
        self.build_environments = []
        self.variables = {}
        self.packages = []

    def entrypoint(self, args):
        return self

    def apt_install(self, *packages):
        return self

    def pip_install(self, *packages):
        self.packages.extend(packages)
        return self

    def env(self, variables):
        self.variables.update(variables)
        return self

    def run_commands(self, *commands, **kwargs):
        self.build_commands.append((commands, kwargs))
        self.build_environments.append(dict(self.variables))
        return self


class PackingModalTests(unittest.TestCase):
    def setUp(self):
        self.image = Image()
        self.volume = SimpleNamespace(commit=Mock())
        self.volume_lookup = Mock(return_value=self.volume)
        fake_modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test-modal",
            Volume=SimpleNamespace(from_name=self.volume_lookup),
            Image=SimpleNamespace(from_registry=Mock(return_value=self.image)),
        )
        spec = importlib.util.spec_from_file_location(
            "_packing_modal_test", probe.REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_packing_modal.py",
        )
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake_modal}):
            spec.loader.exec_module(self.launcher)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)

    def layout(self, root):
        repository, volume = root.resolve() / "repository", root.resolve() / "volume"
        for relative in (CONFIG, TRAINING_CONFIG):
            path = repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((probe.REPOSITORY_ROOT / relative).read_text())
        (volume / "nvarc-qwen3-4b-grids15-sft139").mkdir(parents=True)
        return repository, volume

    def test_pinned_cpu_build_and_gpu_budget_existing_volume_no_retries(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["timeout"], 900)
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["volumes"], {str(self.launcher.VOLUME_ROOT): self.volume})
        self.volume_lookup.assert_called_once_with("diffusion-arc-models", create_if_missing=False)
        commands, kwargs = self.image.build_commands[0]
        self.assertIsNone(kwargs.get("gpu"))
        self.assertTrue(any("--no-build-isolation --no-deps flash-attn==2.8.3.post1" in c
                            for c in commands))
        self.assertEqual(self.image.build_environments[0]["CC"], "/usr/bin/gcc")
        self.assertEqual(self.image.build_environments[0]["CXX"], "/usr/bin/g++")
        self.assertEqual(commands[:2], ("/usr/bin/gcc --version", "/usr/bin/g++ --version"))
        self.assertEqual(self.image.variables["FLASH_ATTENTION_FORCE_BUILD"], "TRUE")
        self.assertEqual(self.image.variables["FLASH_ATTN_CUDA_ARCHS"], "90")
        self.assertEqual([p.name for p in self.image.directories], ["src", "scripts", "configs", "data"])
        self.assertEqual([p.name for p in self.image.files], ["pyproject.toml"])
        fa3_commands, _ = self.image.build_commands[1]
        self.assertTrue(any(self.launcher.FA3_REVISION + "#subdirectory=hopper" in c for c in fa3_commands))
        self.assertEqual(self.image.variables["FLASH_ATTENTION_DISABLE_BACKWARD"], "TRUE")
        self.assertNotIn("FLASH_ATTENTION_DISABLE_HDIM128", self.image.variables)
        self.assertTrue(all(kwargs.get("gpu") is None for _, kwargs in self.image.build_commands))
        self.assertIn("flash-attn-4[cu13]==4.0.0b29", self.image.packages)
        self.assertIn("nvidia-cudnn-frontend==1.13.0", self.image.packages)

    def test_local_launch_needs_no_models_and_forwards_dataset_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), patch.object(
                probe, "source_metadata", return_value={"git_commit": "head", "source_sha256": {}},
            ), patch.object(probe, "dataset_hashes", return_value={"bundle": "hash"}):
                self.launcher.main()
            args = self.launcher.probe_nvarc_packing.remote.call_args.args
            self.assertEqual(args[0], CONFIG)
            self.assertEqual(args[1]["git_commit"], "head")
            self.assertEqual(set(args[1]["config_sha256"]), {"probe", "training"})
            self.assertEqual(args[1]["modal_version"], "test-modal")
            self.assertEqual(args[1]["dataset_sha256"], {"bundle": "hash"})
            self.assertFalse((repository / "checkpoints").exists())

    def test_only_existing_benchmark_bundle_is_uploaded(self):
        for path in ("benchmarks", "benchmarks/single_gpu_b0", "benchmarks/single_gpu_b0/splits/train/config.json"):
            self.assertFalse(self.launcher._ignore_data(Path(path)))
        for path in ("unrelated", "benchmarks/other_run", "private.json"):
            self.assertTrue(self.launcher._ignore_data(Path(path)))

    def test_missing_dataset_fails_before_remote_call(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = self.layout(Path(directory))
            with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(FileNotFoundError):
                self.launcher.main()
        self.launcher.probe_nvarc_packing.remote.assert_not_called()

    def test_invalid_layout_or_timeout_fails_before_remote_call(self):
        with self.assertRaises(ValueError):
            self.launcher.main("../outside.json")
        for change in ("timeout", "checkpoint"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                repository, _ = self.layout(Path(directory))
                path = repository / (CONFIG if change == "timeout" else TRAINING_CONFIG)
                raw = json.loads(path.read_text())
                if change == "timeout":
                    raw["max_duration_seconds"] = 900
                else:
                    raw["models"]["nvarc_checkpoint"] = "/outside/model"
                path.write_text(json.dumps(raw))
                with patch.object(self.launcher, "LOCAL_REPOSITORY_ROOT", repository), self.assertRaises(ValueError):
                    self.launcher.main()
        self.launcher.probe_nvarc_packing.remote.assert_not_called()

    def test_remote_requires_only_nvarc_and_commits_complete_and_failed_runs(self):
        for status in ("complete", "error", "timeout"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                repository, volume = self.layout(Path(directory))
                self.volume.commit.reset_mock()

                def run(args, *, metadata, on_progress):
                    self.assertTrue(args.output_dir.is_relative_to(volume / "results/nvarc-packing"))
                    self.assertEqual(metadata["modal_resources"]["gpu"], "H200")
                    on_progress()
                    return {"status": status, "phase": "longest_source/flash_packed", "error": "original cause"}

                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                    self.launcher, "VOLUME_ROOT", volume,
                ), patch.object(probe, "run_probe", side_effect=run):
                    if status == "complete":
                        self.launcher.probe_nvarc_packing(CONFIG, {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, "original cause"):
                            self.launcher.probe_nvarc_packing(CONFIG, {})
                self.assertEqual(self.volume.commit.call_count, 2)
                self.assertFalse((volume / "flux2-klein-base-4b").exists())

    def test_existing_checkpoint_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = self.layout(Path(directory))
            (repository / "checkpoints").mkdir()
            marker = repository / "checkpoints/user-file"
            marker.write_text("preserve")
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), patch.object(
                self.launcher, "VOLUME_ROOT", volume,
            ), patch.object(probe, "run_probe") as run, self.assertRaises(RuntimeError):
                self.launcher.probe_nvarc_packing(CONFIG, {})
            run.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
