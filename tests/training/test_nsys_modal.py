"""Offline Nsight launcher contract and artifact failure checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.training.test_training_modal import _App, RecordingImage


class NsightModalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        image, self.volume = RecordingImage(), Mock()
        fake = SimpleNamespace(App=_App, is_local=lambda: True,
                               Image=SimpleNamespace(from_registry=Mock(return_value=image)),
                               Volume=SimpleNamespace(from_name=Mock(return_value=self.volume)))
        spec = importlib.util.spec_from_file_location("nsys_launcher", self.root / "scripts/cloud/profile_single_gpu_nsys_modal.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": fake}), patch.object(sys, "path", list(sys.path)):
            spec.loader.exec_module(self.module)

    def test_config_and_command(self):
        m = self.module
        config = m.validate_config(self.root, m.DEFAULT_CONFIG)
        for value in ("../README.md", str(config)):
            with self.assertRaises(ValueError):
                m.validate_config(self.root, value)
        with self.assertRaises(ValueError):
            m.validate_config(self.root, "configs/training/benchmarks/single_gpu_b3.json")
        command = m.profile_command(config, Path("/tmp/output"))
        self.assertIn(str(m.REMOTE_REPOSITORY_ROOT / "scripts/profile_single_gpu.py"), command)
        self.assertIn("--capture-range=cudaProfilerApi", command)
        self.assertIn("--kill=none", command)
        for option in ("--trace=cuda,nvtx", "--sample=none",
                       "--cpuctxsw=none", "--python-sampling=false",
                       "--cuda-memory-usage=false",
                       "--gpu-metrics-devices=cuda-visible", "--gpu-metrics-frequency=10000"):
            self.assertIn(option, command)
        self.assertFalse(any(option.startswith("--osrt-") for option in command))
        evidence = m._capture_evidence("")
        self.assertEqual(evidence["requested"], ["cuda_api_and_kernels", "nvtx", "gpu_metrics"])
        self.assertTrue({"python_sampling", "python_gil", "os_runtime", "file_access",
                         "cuda_memory_usage"}.issubset(evidence["disabled"]))
        self.assertEqual(command[-2:], ["--profiler", "nsys"])
        self.assertFalse(m._ignore_data(Path("benchmarks/single_gpu_b0/splits/train")))
        self.assertTrue(m._ignore_data(Path("raw")))

    def test_preflight_rejects_mutated_canonical_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / self.module.DEFAULT_CONFIG
            config.parent.mkdir(parents=True)
            original = json.loads((self.root / self.module.DEFAULT_CONFIG).read_text())
            for section, key, value in (("loop", "max_steps", 8),
                                        ("models", "nvarc_checkpoint", "/tmp/checkpoint"),
                                        ("models", "nvarc_dtype", "float32"),
                                        ("data", "num_workers", 4)):
                payload = json.loads(json.dumps(original))
                payload[section][key] = value
                config.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    self.module.validate_config(root, self.module.DEFAULT_CONFIG)
            config.write_text(json.dumps(original))
            with self.assertRaises(FileNotFoundError):
                self.module.validate_config(root, self.module.DEFAULT_CONFIG)
            original["execution"]["nvarc_max_tokens"] = 8400
            config.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "seven-update B3"):
                self.module.validate_config(root, self.module.DEFAULT_CONFIG)

    def test_persists_success_missing_report_and_process_failure(self):
        m = self.module
        for outcome in ("success", "missing", "failure", "timeout", "incomplete"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                process = Mock(pid=123)
                def launch(*args, **kwargs):
                    if outcome in ("success", "incomplete"):
                        (output / "trace.nsys-rep").write_bytes(b"report")
                        (output / "training-summary.json").write_text(json.dumps({"status": "complete" if outcome == "success" else "failed", "completed_updates": 7, "recorded_updates": 3}))
                    return process
                process.wait.return_value = 1 if outcome == "failure" else 0
                if outcome == "timeout":
                    process.wait.side_effect = [subprocess.TimeoutExpired("nsys", m.RUN_TIMEOUT_SECONDS), 0]
                with patch.object(m, "REMOTE_REPOSITORY_ROOT", self.root), patch.object(m, "_capture", return_value="metadata"), patch.object(m, "_preflight", return_value=[{"returncode": 0}]), patch.object(m.subprocess, "Popen", side_effect=launch), patch.object(m.os, "killpg") as kill:
                    if outcome == "success":
                        self.assertEqual(m._run_profile(self.root / m.DEFAULT_CONFIG, output), str(output / "trace.nsys-rep"))
                    else:
                        with self.assertRaises((RuntimeError, subprocess.SubprocessError)):
                            m._run_profile(self.root / m.DEFAULT_CONFIG, output)
                    if outcome == "timeout":
                        kill.assert_called_once()
                summary = json.loads((output / "run-summary.json").read_text())
                self.assertEqual(summary["status"], "capture_complete_unverified" if outcome == "success" else "failed")
                self.assertTrue((output / "config.json").is_file())
                self.assertTrue((output / "run.log").is_file())
                self.assertTrue(summary["source_sha256"])
        self.assertEqual(self.volume.commit.call_count, 5)

    def test_success_does_not_verify_missing_collectors(self):
        for log in ("", "WARNING: GPU metrics unavailable due to permissions\n"):
            evidence = self.module._capture_evidence(log)
            self.assertEqual(evidence["verified"], [])
            self.assertEqual(evidence["status"], "requires_report_review")
            self.assertEqual(bool(evidence["diagnostic_messages"]), bool(log))

    def test_failed_capability_preflight_prevents_training(self):
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(m, "REMOTE_REPOSITORY_ROOT", self.root), \
                    patch.object(m, "_preflight", return_value=[{"returncode": 1}]), \
                    patch.object(m.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    m._run_profile(self.root / m.DEFAULT_CONFIG, output)
                launch.assert_not_called()
            metadata = json.loads((output / "run-summary.json").read_text())
            self.assertEqual(metadata["status"], "failed")

    def test_preflight_preserves_stdout_stderr_and_failures(self):
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = SimpleNamespace(stdout="capabilities\n", stderr="permission denied\n", returncode=1)
            with patch.object(m.subprocess, "run", return_value=result) as run:
                diagnostics = m._preflight(output)
            self.assertEqual(run.call_count, 3)
            for entry in diagnostics:
                self.assertEqual(entry["returncode"], 1)
                self.assertEqual((output / entry["log"]).read_text(), "capabilities\npermission denied\n")

    def test_disabled_cpu_checks_are_recorded_without_blocking_training(self):
        m = self.module
        report = (
            "- Root privilege: enabled\n"
            "- Linux perf_event_open syscall available: Fail\n"
            "- Sampling trigger event available: Fail\n"
            "- CPU Profiling Environment (process-tree): Fail\n"
            "- CPU Profiling Environment (system-wide): Fail\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = SimpleNamespace(stdout=report, stderr="", returncode=0)
            process = Mock(pid=123)
            process.wait.return_value = 0
            def launch(*args, **kwargs):
                (output / "trace.nsys-rep").write_bytes(b"report")
                (output / "training-summary.json").write_text(json.dumps({
                    "status": "complete", "completed_updates": 7, "recorded_updates": 3,
                }))
                return process
            with patch.object(m, "REMOTE_REPOSITORY_ROOT", self.root), \
                    patch.object(m.subprocess, "run", return_value=result), \
                    patch.object(m, "_capture", return_value="metadata"), \
                    patch.object(m.subprocess, "Popen", side_effect=launch) as start:
                m._run_profile(self.root / m.DEFAULT_CONFIG, output)
                start.assert_called_once()
            metadata = json.loads((output / "run-summary.json").read_text())
            self.assertEqual(metadata["preflight"][0]["returncode"], 0)
            self.assertEqual(len(metadata["preflight"][0]["disabled_cpu_failures"]), 3)
            self.assertEqual(metadata["preflight"][1]["disabled_cpu_failures"], [])
            self.assertEqual(metadata["status"], "capture_complete_unverified")
            self.assertNotIn("cpu_sampling", metadata["collection"]["requested"])
            self.assertIn("cpu_sampling", metadata["collection"]["disabled"])
            self.assertIn("cpu_context_switches", metadata["collection"]["disabled"])
