"""Offline checks of validation resource limits and worker cancellation."""

import asyncio
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.training.test_benchmark_flux_packing_modal import _App, _Image


class ValidationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.image = _Image()
        self.image.object_id = "im-IGaCiJ89DjpJQfQE43QTjh"
        modal = SimpleNamespace(
            App=_App, is_local=lambda: True, __version__="test",
            Image=SimpleNamespace(from_id=Mock(return_value=self.image)),
            Volume=SimpleNamespace(from_name=Mock(return_value=SimpleNamespace(commit=SimpleNamespace(aio=AsyncMock())))),
        )
        path = Path(__file__).resolve().parents[2] / "scripts/cloud/benchmark_validation_modal.py"
        spec = importlib.util.spec_from_file_location("_validation_launcher_test", path)
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": modal}):
            spec.loader.exec_module(self.launcher)

    def test_resources_and_mounts_bound_one_offline_h200_without_training_data(self):
        options = self.launcher.app.function_options
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["cpu"], (8., 8.))
        self.assertEqual(options["memory"], (65536, 65536))
        self.assertEqual((options["max_containers"], options["min_containers"], options["buffer_containers"]), (1, 0, 0))
        self.assertEqual(options["retries"], 0)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertLessEqual(options["startup_timeout"] + options["timeout"] + options["scaledown_window"],
                             self.launcher.LIFECYCLE_SECONDS)
        self.assertLessEqual(self.launcher.WORKER_SECONDS + 60, options["timeout"])
        self.assertLess(self.launcher.BUDGET["maximum_resource_list_price_estimate_usd"], 1.82)
        mounted = [str(path.relative_to(self.launcher.LOCAL_ROOT)) for path, _ in self.image.mounts]
        self.assertEqual(mounted, ["src", "scripts", "configs", self.launcher.BUNDLE, "pyproject.toml"])

    def test_worker_timeout_terminates_then_kills_and_does_not_retry(self):
        process = Mock(pid=123)
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 990),
                                    subprocess.TimeoutExpired("worker", 5), -9]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(self.launcher.subprocess, "Popen", return_value=process) as popen, \
                patch.object(self.launcher.os, "killpg") as kill:
            with self.assertRaisesRegex(TimeoutError, "worker budget"):
                self.launcher.run_worker(Path("config.json"), Path(directory))
        self.assertEqual(popen.call_count, 1)
        self.assertEqual([call.args for call in kill.call_args_list], [(123, signal.SIGTERM), (123, signal.SIGKILL)])

    def test_remote_failure_cancels_container_and_stops_only_own_app(self):
        remote = SimpleNamespace(get=SimpleNamespace(aio=AsyncMock(side_effect=RuntimeError("failed"))),
                                 cancel=SimpleNamespace(aio=AsyncMock()))
        self.launcher.benchmark.spawn.aio.return_value = remote
        process = SimpleNamespace(wait=AsyncMock(return_value=0))
        with patch.object(self.launcher.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)) as stop:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                asyncio.run(self.launcher.bounded_call({"modal_app_id": "ap-own-validation"}))
        remote.cancel.aio.assert_awaited_once_with(terminate_containers=True)
        self.assertIn("ap-own-validation", stop.call_args.args)
        self.assertIn("--env", stop.call_args.args)


if __name__ == "__main__":
    unittest.main()
