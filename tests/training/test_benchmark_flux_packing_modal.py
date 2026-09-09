"""Offline checks of the integrated smoke launcher's bounds and failure paths."""

import asyncio
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/training/benchmarks/single_gpu_b3_smoke.json"
IMAGE_ID = "im-R5CncDboPfMQryWnq0UZn4"
MODULE = "scripts.benchmark_single_gpu"
RUN_ID = "20260909T010203123456Z"


class _Image:
    object_id = IMAGE_ID

    def __init__(self):
        self.mounts = []

    def add_local_dir(self, path, **options):
        self.mounts.append((path, options))
        return self

    add_local_file = add_local_dir


class _App:
    app_id = "ap-smoke-test"

    def __init__(self, name):
        self.name = name

    def function(self, **options):
        self.function_options = options

        def decorate(function):
            function.spawn = SimpleNamespace(aio=AsyncMock())
            return function

        return decorate

    def local_entrypoint(self):
        return lambda function: function


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.image = _Image()
        self.volume = SimpleNamespace(commit=SimpleNamespace(aio=AsyncMock()))
        modal = SimpleNamespace(App=_App, is_local=lambda: True, __version__="test",
            Image=SimpleNamespace(from_id=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=Mock(return_value=self.volume)))
        source = ROOT / "scripts/cloud/benchmark_flux_packing_modal.py"
        spec = importlib.util.spec_from_file_location("_packed_smoke_launcher_test", source)
        self.launcher = importlib.util.module_from_spec(spec)
        paths = patch.object(sys, "path", list(sys.path))
        paths.start()
        self.addCleanup(paths.stop)
        with patch.dict(sys.modules, {"modal": modal}):
            spec.loader.exec_module(self.launcher)
        self.modal = modal

    def test_cached_image_mounts_and_fixed_resource_budget(self):
        self.modal.Image.from_id.assert_called_once_with(IMAGE_ID)
        self.assertEqual([str(path.relative_to(ROOT)) for path, _ in self.image.mounts],
                         ["src", "scripts", "configs", "pyproject.toml", "data/benchmarks/single_gpu_b0"])
        self.assertTrue(all(options["copy"] is False for _, options in self.image.mounts))
        self.modal.Volume.from_name.assert_called_once_with("diffusion-arc-models",
            environment_name="diffusion-arc", create_if_missing=False)
        options = self.launcher.app.function_options
        self.assertEqual((options["gpu"], options["cpu"], options["memory"]),
                         ("H200", (8.0, 8.0), (65536, 65536)))
        self.assertEqual((options["timeout"], options["startup_timeout"]), (390, 60))
        self.assertEqual((options["max_containers"], options["min_containers"], options["buffer_containers"]), (1, 0, 0))
        self.assertEqual(options["retries"], 0)
        self.assertIs(options["nonpreemptible"], False)
        self.assertNotIn("region", options)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(options["env"]["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual((self.launcher.MAX_WORKER_SECONDS, self.launcher.FINAL_COMMIT_SECONDS), (330, 60))
        self.assertEqual((self.launcher.LIFECYCLE_SECONDS, self.launcher.CLEANUP_SECONDS), (480, 30))
        self.assertLessEqual(options["startup_timeout"] + options["timeout"] + options["scaledown_window"], 480)
        self.assertAlmostEqual(self.launcher.BUDGET["maximum_list_price_estimate_usd"], 0.7237824)

    @staticmethod
    def _config(repository):
        return SimpleNamespace(data=SimpleNamespace(split_directory=repository / "data/benchmarks/single_gpu_b0/splits/train"),
            loop=SimpleNamespace(max_optimizer_steps=3, optimizer_batch_size=48),
            benchmark=SimpleNamespace(warmup_steps=1, measured_steps=2, repetitions=1),
            flux_packing=SimpleNamespace(max_tokens=12_288), optimizer=SimpleNamespace(learning_rate=1e-5),
            models=SimpleNamespace(nvarc_checkpoint=repository / "checkpoints/nvarc",
                                   flux_checkpoint=repository / "checkpoints/flux"))

    def test_preflight_restricts_config_data_workload_and_positive_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            variants = [("loop", "max_optimizer_steps", 4), ("loop", "optimizer_batch_size", 96),
                        ("benchmark", "warmup_steps", 2), ("benchmark", "measured_steps", 3),
                        ("benchmark", "repetitions", 2), ("flux_packing", "max_tokens", 13_312),
                        ("optimizer", "learning_rate", 0), ("data", "split_directory", repository / "other"),
                        ("models", "flux_checkpoint", Path("/outside-models"))]
            for section, key, value in variants:
                config = self._config(repository)
                setattr(getattr(config, section), key, value)
                with self.subTest(section=section, key=key), patch.dict(sys.modules,
                        {MODULE: SimpleNamespace(check_inputs=Mock(return_value=(config, [])))}):
                    with self.assertRaises(ValueError):
                        self.launcher._validate_layout(repository, CONFIG,
                            checkpoint_root=repository / "checkpoints", require_checkpoints=False)
            loader = Mock(return_value=(self._config(repository), []))
            with patch.dict(sys.modules, {MODULE: SimpleNamespace(check_inputs=loader)}):
                with self.assertRaisesRegex(ValueError, "requires"):
                    self.launcher._validate_layout(repository, "../other.json",
                        checkpoint_root=repository / "checkpoints", require_checkpoints=False)
                loader.assert_not_called()
                with self.assertRaisesRegex(FileNotFoundError, "cached checkpoint"):
                    self.launcher._validate_layout(repository, CONFIG,
                        checkpoint_root=repository / "checkpoints", require_checkpoints=True)
                self.assertEqual(self.launcher._validate_layout(repository, CONFIG,
                    checkpoint_root=repository / "checkpoints", require_checkpoints=False), repository / CONFIG)

    def test_identity_covers_source_config_and_data_with_portable_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            files = {"src/example.py": "source", "scripts/run.py": "entrypoint",
                     "pyproject.toml": "project", CONFIG: "config",
                     "data/benchmarks/single_gpu_b0/raw/task.json": "data"}
            for relative, contents in files.items():
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            before = self.launcher._identity(repository)
            self.assertEqual(set(before["source_sha256"]), {"src/example.py", "scripts/run.py", "pyproject.toml"})
            (repository / "src/example.py").write_text("changed")
            after = self.launcher._identity(repository)
            self.assertNotEqual(before["source_sha256"], after["source_sha256"])
            self.assertEqual(before["dataset_sha256"], after["dataset_sha256"])

    def test_worker_calls_productive_benchmark_once_without_config_overrides(self):
        runner = Mock()
        with patch.dict(sys.modules, {MODULE: SimpleNamespace(benchmark_once=runner)}), \
                patch.object(sys, "argv", ["worker", "/config", "/output"]), \
                patch("signal.signal") as register:
            exec(self.launcher.WORKER_CODE, {})
        runner.assert_called_once_with(Path("/config"), Path("/output"))
        handler = register.call_args.args[1]
        with self.assertRaises(TimeoutError):
            handler(signal.SIGTERM, None)

    def test_worker_failure_and_timeout_preserve_logs_and_bound_termination(self):
        for waits, signals, expected in (([7], [], RuntimeError),
                ([subprocess.TimeoutExpired("worker", 320), 1], [signal.SIGTERM], TimeoutError),
                ([subprocess.TimeoutExpired("worker", 320), subprocess.TimeoutExpired("worker", 5), -9],
                 [signal.SIGTERM, signal.SIGKILL], TimeoutError)):
            with self.subTest(signals=signals), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                process = SimpleNamespace(pid=123, wait=Mock(side_effect=waits))
                with patch.object(self.launcher.subprocess, "Popen", return_value=process) as spawn, \
                        patch.object(self.launcher.os, "killpg") as kill:
                    with self.assertRaises(expected):
                        self.launcher._run_worker(ROOT, ROOT / CONFIG, destination)
                self.assertTrue((destination / "worker.log").is_file())
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                self.assertEqual(spawn.call_args.args[0][-1], str(destination / "repetition-1"))
                process.wait.assert_any_call(timeout=320)
                self.assertEqual(kill.call_args_list, [call(123, sig) for sig in signals])

    def test_remote_keeps_worker_artifacts_and_commits_once_on_success_or_error(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                repository, volume = Path(directory) / "repo", Path(directory) / "volume"
                repository.mkdir()
                volume.mkdir()
                identity = {"source_sha256": {"source": "sha"}, "config_sha256": {}, "dataset_sha256": {}}
                metadata = {**identity, "image_id": IMAGE_ID, "run_id": RUN_ID}

                def worker(repo, config, destination):
                    self.assertEqual((repo, config), (repository, repository / CONFIG))
                    repetition = destination / "repetition-1"
                    repetition.mkdir()
                    (repetition / "steps.jsonl").write_text('{"step": 1}\n')
                    (repetition / "summary.json").write_text(json.dumps(
                        {"status": "failed" if failed else "complete", "measured_steps": 2}))
                    if failed:
                        raise RuntimeError("worker failure")

                self.volume.commit.aio.reset_mock()
                with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                        patch.object(self.launcher, "VOLUME_ROOT", volume), \
                        patch.object(self.launcher, "_identity", return_value=identity), \
                        patch.object(self.launcher, "_validate_layout", return_value=repository / CONFIG), \
                        patch.object(self.launcher, "_run_worker", side_effect=worker):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, "worker failure"):
                            self.launcher.benchmark_flux_packing(CONFIG, metadata)
                    else:
                        self.launcher.benchmark_flux_packing(CONFIG, metadata)
                output = volume / self.launcher.RESULTS / RUN_ID
                self.assertTrue((output / "repetition-1/steps.jsonl").is_file())
                self.assertEqual(json.loads((output / "manifest.json").read_text())["status"],
                                 "failed" if failed else "complete")
                self.volume.commit.aio.assert_awaited_once()

    def test_remote_rejects_changed_identity_and_duplicate_run_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, volume = Path(directory) / "repo", Path(directory) / "volume"
            repository.mkdir()
            (volume / self.launcher.RESULTS / RUN_ID).mkdir(parents=True)
            identity = {"source_sha256": {"file": "sha"}}
            valid = {**identity, "image_id": IMAGE_ID, "run_id": RUN_ID}
            variants = [{**valid, "image_id": "im-other"}, {**valid, "run_id": "../escape"},
                        {**valid, "source_sha256": {"file": "other"}}]
            with patch.object(self.launcher, "REMOTE_REPOSITORY_ROOT", repository), \
                    patch.object(self.launcher, "VOLUME_ROOT", volume), \
                    patch.object(self.launcher, "_identity", return_value=identity), \
                    patch.object(self.launcher, "_validate_layout", return_value=repository / CONFIG), \
                    patch.object(self.launcher, "_run_worker") as worker:
                for metadata in variants:
                    with self.assertRaises(ValueError):
                        self.launcher.benchmark_flux_packing(CONFIG, metadata)
                self.assertFalse((repository / "checkpoints").exists())
                with self.assertRaisesRegex(RuntimeError, "refusing to repeat"):
                    self.launcher.benchmark_flux_packing(CONFIG, valid)
            worker.assert_not_called()

    def test_lifecycle_timeout_cancels_the_single_call_and_stops_its_app(self):
        remote = SimpleNamespace(get=SimpleNamespace(aio=AsyncMock(side_effect=TimeoutError("deadline"))),
                                 cancel=SimpleNamespace(aio=AsyncMock()))
        self.launcher.benchmark_flux_packing.spawn.aio.return_value = remote
        with patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        self.launcher.benchmark_flux_packing.spawn.aio.assert_awaited_once()
        remote.cancel.aio.assert_awaited_once_with(terminate_containers=True)
        stop.assert_awaited_once_with("ap-one")

    def test_hanging_submission_and_final_commit_have_deadlines(self):
        async def hang(*args):
            await asyncio.Event().wait()

        self.launcher.benchmark_flux_packing.spawn.aio.side_effect = hang
        with patch.object(self.launcher, "LIFECYCLE_SECONDS", 0.01), \
                patch.object(self.launcher, "CLEANUP_SECONDS", 0), \
                patch.object(self.launcher, "_stop_app", new_callable=AsyncMock) as stop:
            with self.assertRaises(TimeoutError):
                asyncio.run(self.launcher._bounded_remote_call(CONFIG, {"modal_app_id": "ap-one"}))
        stop.assert_awaited_once_with("ap-one")
        self.volume.commit.aio.side_effect = hang
        with patch.object(self.launcher, "FINAL_COMMIT_SECONDS", 0.01):
            with self.assertRaises(TimeoutError):
                asyncio.run(self.launcher._commit_final())


if __name__ == "__main__":
    unittest.main()
