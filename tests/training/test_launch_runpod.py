"""Local archive and detached-process checks; no SSH server, network or GPU needed."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud import launch_runpod as launcher


def pod_environment():
    """Stand in for a Linux Pod environment during the real local subprocess smoke test."""
    import os
    return os.environ.copy()


class LocalLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pod sources ' ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.root = self.directory / "repository"
        self.workspace = self.directory / "workspace"
        self.root.mkdir()
        self.workspace.mkdir()
        self.write("README.md", "test project")
        self.write("pyproject.toml", "[project]\nname='fixture'\n")
        self.write("src/diffusion_arc/__init__.py", 'VALUE = "original"\n')
        self.write("src/deleted.py", "old\n")
        self.write("scripts/cloud/launch_runpod.py", Path(launcher.__file__).read_text())
        self.write("scripts/cloud/train_runpod.py", "\n".join((
            "import os, sys, time", "from diffusion_arc import VALUE",
            "print(VALUE, flush=True)", "print('stderr marker', file=sys.stderr, flush=True)",
            "time.sleep(0.2)", "sys.exit(int(os.environ.get('TEST_POD_EXIT', '0')))",
        )))
        self.write("scripts/train_single_gpu.py", "# fixture\n")
        self.write("configs/recipe.json", "{}")
        self.write(".gitignore", ".env\nartifacts/\n")
        for command in (("init", "-q"), ("add", "."),
                        ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                         "commit", "-qm", "fixture")):
            subprocess.run(["git", *command], cwd=self.root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def receive(self, content, job_id="test-job", *, digest=None, check=True):
        # Use the actual remote bootstrap in a separate Python process; replace SSH with a shell.
        transport = [sys.executable, "-c", "import os, sys; os.execv('/bin/sh', ['sh', '-c', sys.argv[-1]])"]
        return launcher.remote_python(
            transport, sys.executable, (pod_environment, launcher.receive_and_start),
            "receive_and_start(*sys.argv[1:])",
            [digest or hashlib.sha256(content).hexdigest(), str(self.workspace),
             json.dumps({"job_id": job_id, "arguments": ["run", "--run-name", "example"]})],
            input=content, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, timeout=10,
        )

    def completed(self, job_id):
        status = self.workspace / "jobs" / job_id / "status.json"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if status.exists() and "exit_code" in (value := json.loads(status.read_text())):
                return value
            time.sleep(0.05)
        self.fail("Local detached worker did not finish")

    def test_snapshot_contains_current_working_tree_without_credentials_or_data(self):
        self.write("src/diffusion_arc/__init__.py", 'VALUE = "edited"\n')
        self.write("src/untracked.py", "new = True\n")
        (self.root / "src/deleted.py").unlink()
        self.write(".env", "TOKEN=never-upload-this")
        self.write("data/task.json", "private task")
        self.write("src/model.safetensors", "weights")
        self.write("src/__pycache__/cached.py", "cache")
        first, manifest = launcher.source_snapshot(self.root)
        repeated, _ = launcher.source_snapshot(self.root)
        self.assertEqual(first, repeated)
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
            self.assertEqual(archive.extractfile("src/diffusion_arc/__init__.py").read(), b'VALUE = "edited"\n')
            names = set(archive.getnames())
        self.assertIn("src/untracked.py", names)
        self.assertTrue(manifest["working_tree_dirty"])
        self.assertFalse(names & {".env", "data/task.json", "src/model.safetensors", "src/deleted.py", "src/__pycache__/cached.py"})
        self.assertNotIn(b"never-upload-this", gzip.decompress(first))

    def test_source_symlinks_are_rejected(self):
        outside = self.directory / "outside"
        outside.write_text("private")
        (self.root / "src/link.py").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            launcher.source_snapshot(self.root)

    def test_ssh_parser_accepts_both_transports_and_rejects_shell_options(self):
        key = self.directory / "key with spaces"
        key.write_text("not-a-real-key")
        args = launcher.ssh_arguments(f"ssh root@192.0.2.1 -p 22001 -i {str(key)!r}")
        self.assertEqual(args[-1], "root@192.0.2.1")
        self.assertIn(str(key), args)
        self.assertIn("-T", args)
        proxy = launcher.ssh_arguments(f"ssh pod-id@ssh.runpod.io -i {str(key)!r}")
        self.assertIn("-tt", proxy)
        self.assertIn("EscapeChar=none", proxy)
        self.assertEqual(proxy[-1], "pod-id@ssh.runpod.io")
        for command in ("ssh", "ssh host ; echo leak", "ssh -o ProxyCommand=evil host",
                        "ssh host -p 0", "ssh host -p 99999", "ssh -p", "scp host"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                launcher.ssh_arguments(command)

    @contextmanager
    def terminal_transport(self, *, crlf=False):
        """Replace only SSH with a real local PTY, preserving echo, line limits and subprocesses."""
        popen = subprocess.Popen
        read = os.read
        output_fds = set()

        def start(command, **kwargs):
            if command[0] != "fixture-ssh":
                return popen(command, **kwargs)
            master, slave = pty.openpty()
            try:
                process = popen(["/bin/bash", "--noprofile", "--norc", "-i"],
                                stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
            except BaseException:
                os.close(master)
                raise
            finally:
                os.close(slave)
            process.stdin = os.fdopen(master, "wb", buffering=0)
            process.stdout = os.fdopen(os.dup(master), "rb", buffering=0)
            output_fds.add(process.stdout.fileno())
            return process

        def terminal_read(fd, size):
            content = read(fd, size)
            # The real proxy applies CRLF even after the inner terminal enters raw mode.
            if crlf and fd in output_fds:
                content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            return content

        with patch.object(launcher.subprocess, "Popen", side_effect=start), \
                patch.object(launcher.os, "read", side_effect=terminal_read):
            yield

    def test_proxy_streams_binary_input_without_echo_and_returns_remote_exit_code(self):
        content = bytes(range(256)) * 4096
        argument = "a 'quoted' path with $HOME and `uname`"
        program = ("import hashlib, sys\n"
                   "print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())\n"
                   "print(sys.argv[1])\n"
                   "print('stderr marker', file=sys.stderr)\n"
                   "raise SystemExit(7)\n")
        with self.terminal_transport(crlf=True):
            result = launcher.proxy_python(
                ["fixture-ssh", "pod@ssh.runpod.io"], sys.executable, program, [argument],
                input=content, stdout=subprocess.PIPE, timeout=10,
            )
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout.decode(), hashlib.sha256(content).hexdigest()
                         + "\n" + argument + "\nstderr marker\n")

    def test_proxy_upload_detaches_and_log_reconnection_uses_same_transport(self):
        content, _ = launcher.source_snapshot(self.root)
        ssh = ["fixture-ssh", "pod@ssh.runpod.io"]
        with self.terminal_transport():
            result = launcher.remote_python(
                ssh, sys.executable, (pod_environment, launcher.receive_and_start),
                "receive_and_start(*sys.argv[1:])",
                [hashlib.sha256(content).hexdigest(), str(self.workspace),
                 json.dumps({"job_id": "proxy-job", "arguments": ["run"]})],
                input=content, stdout=subprocess.PIPE, check=True, timeout=10,
            )
            receipt = json.loads(result.stdout)
            self.assertEqual(self.completed("proxy-job")["exit_code"], 0)
            result = launcher.remote_python(
                ssh, sys.executable, (launcher.follow_remote_job,),
                "raise SystemExit(follow_remote_job(sys.argv[1]))", [receipt["job_directory"]],
                stdout=subprocess.PIPE, timeout=10,
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, (Path(receipt["job_directory"]) / "console.log").read_bytes())

    def test_proxy_remote_error_is_checked_before_any_worker_starts(self):
        with self.terminal_transport(), self.assertRaises(subprocess.CalledProcessError) as raised:
            launcher.remote_python(
                ["fixture-ssh", "pod@ssh.runpod.io"], sys.executable,
                (pod_environment, launcher.receive_and_start), "receive_and_start(*sys.argv[1:])",
                ["0" * 64, str(self.workspace), "{}"], input=b"corrupt archive",
                stdout=subprocess.PIPE, check=True, timeout=10,
            )
        self.assertIn(b"checksum", raised.exception.output)
        self.assertFalse((self.workspace / "jobs").exists())

    def test_proxy_reports_stopped_pod_and_bounds_startup_wait(self):
        with self.assertRaisesRegex(RuntimeError, "container not found"):
            launcher.proxy_python(
                [sys.executable, "-c", "print('container not found')"], sys.executable,
                "print('must not execute')", [], stdout=subprocess.PIPE, timeout=2,
            )
        with self.assertRaisesRegex(RuntimeError, "timed out while opening the proxy shell"):
            launcher.proxy_python(
                [sys.executable, "-c", "import time; time.sleep(60)"], sys.executable,
                "print('must not execute')", [], stdout=subprocess.PIPE, timeout=0.2,
            )

    def test_proxy_streams_logs_and_rejects_disconnect_without_completion_marker(self):
        stream = io.BytesIO()
        with self.terminal_transport():
            result = launcher.proxy_python(
                ["fixture-ssh", "pod@ssh.runpod.io"], sys.executable,
                "print('live log')", [], stdout=stream, timeout=10,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(stream.getvalue(), b"live log\n")
            with self.assertRaisesRegex(RuntimeError, "disconnected before reporting completion"):
                launcher.proxy_python(
                    ["fixture-ssh", "pod@ssh.runpod.io"], sys.executable,
                    "import os; os._exit(0)", [], stdout=subprocess.PIPE, timeout=10,
                )

    def test_real_detached_worker_uses_uploaded_code_and_records_both_streams_and_exit_code(self):
        self.write("src/diffusion_arc/__init__.py", 'VALUE = "uploaded working tree"\n')
        content, _ = launcher.source_snapshot(self.root)
        with patch.dict(os.environ, {"PYTHONPATH": "/opt/stale-image-code", "TEST_POD_EXIT": "7"}):
            receipt = json.loads(self.receive(content).stdout)
        # The bootstrap process has exited; its detached worker still completes independently.
        self.assertEqual(self.completed("test-job")["exit_code"], 7)
        job = Path(receipt["job_directory"])
        log = (job / "console.log").read_text()
        self.assertIn("uploaded working tree", log)
        self.assertIn("stderr marker", log)
        stream = SimpleNamespace(buffer=io.BytesIO())
        with patch.object(sys, "stdout", stream):
            self.assertEqual(launcher.follow_remote_job(str(job)), 7)
        self.assertEqual(stream.buffer.getvalue(), log.encode())

    def test_new_snapshot_preserves_previous_code_and_existing_snapshot_rejects_extra_files(self):
        original, _ = launcher.source_snapshot(self.root)
        first = json.loads(self.receive(original, "first").stdout)
        self.write("src/diffusion_arc/__init__.py", 'VALUE = "second snapshot"\n')
        changed, _ = launcher.source_snapshot(self.root)
        second = json.loads(self.receive(changed, "second").stdout)
        self.assertNotEqual(first["source_directory"], second["source_directory"])
        old = Path(first["source_directory"])
        self.assertEqual((old / "src/diffusion_arc/__init__.py").read_text(), 'VALUE = "original"\n')
        self.completed("first")
        self.completed("second")
        (old / "src/unexpected.py").write_text("unexpected = True")
        result = self.receive(original, "third", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"inventory was changed", result.stderr)
        self.assertFalse((self.workspace / "jobs/third").exists())

    def test_log_follower_handles_completion_between_status_read_and_pid_check(self):
        job = self.workspace / "race-job"
        job.mkdir()
        (job / "worker.json").write_text('{"pid": 12345}')
        (job / "console.log").write_bytes(b"started\n")
        (job / "status.json").write_text('{"state": "running"}')

        def complete(*_args):
            (job / "status.json").write_text('{"state": "succeeded", "exit_code": 0}')
            with (job / "console.log").open("ab") as log:
                log.write(b"finished\n")
            raise ProcessLookupError

        stream = SimpleNamespace(buffer=io.BytesIO())
        with patch.object(os, "kill", side_effect=complete), patch.object(sys, "stdout", stream):
            self.assertEqual(launcher.follow_remote_job(str(job)), 0)
        self.assertEqual(stream.buffer.getvalue(), b"started\nfinished\n")

    def test_invalid_transfer_and_path_traversal_never_start_a_worker(self):
        content, _ = launcher.source_snapshot(self.root)
        result = self.receive(content, digest="0" * 64, check=False)
        self.assertIn(b"checksum", result.stderr)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("../../escaped.py")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        result = self.receive(buffer.getvalue(), check=False)
        self.assertIn(b"unsafe", result.stderr)
        self.assertFalse((self.workspace / "jobs").exists())
        self.assertFalse((self.workspace / "escaped.py").exists())

    def test_missing_remote_secrets_prevent_upload_and_dry_run_never_connects(self):
        arguments = ["run", "--ssh", "ssh root@192.0.2.1", "--run-name", "test",
                     "--config", str(self.root / "configs/recipe.json")]
        with patch.object(launcher, "ROOT", self.root), patch.object(launcher, "remote_python") as remote:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main([*arguments, "--dry-run"]), 0)
            remote.assert_not_called()
            self.assertFalse((self.root / "artifacts").exists())
            remote.side_effect = subprocess.CalledProcessError(1, "preflight")
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.main(arguments)
            self.assertEqual(remote.call_count, 1)
            self.assertFalse((self.root / "artifacts").exists())

    def test_pod_environment_whitelists_only_needed_variables(self):
        raw = b"HF_TOKEN=pod-secret\0WANDB_API_KEY=wandb\0UNRELATED_SECRET=hidden\0LD_LIBRARY_PATH=/cuda/lib\0"
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "read_bytes", return_value=raw):
            environment = launcher.pod_environment()
        self.assertEqual(environment["HF_TOKEN"], "pod-secret")
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/cuda/lib")
        self.assertNotIn("UNRELATED_SECRET", environment)
        with patch.object(launcher, "pod_environment", return_value={"HF_TOKEN": "x"}):
            with self.assertRaisesRegex(ValueError, "WANDB_API_KEY, KAGGLE_API_TOKEN"):
                launcher.remote_preflight("run", str(self.workspace))

    def test_imported_launcher_reports_captured_upload_failure_without_following_logs(self):
        arguments = ["continue", "--ssh", "ssh root@192.0.2.1", "--run-name", "epoch4",
                     "--from-run", "epoch3", "--config", str(self.root / "configs/recipe.json")]
        error = subprocess.CalledProcessError(
            1, ["ssh"], output=b"ValueError: Existing source snapshot file inventory was changed\n")
        stderr = io.StringIO()
        with patch.object(launcher, "ROOT", self.root), \
                patch.object(launcher, "remote_python", side_effect=[None, error]) as remote, \
                redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            self.assertEqual(launcher.main(arguments), 1)
        self.assertIn("Existing source snapshot file inventory was changed", stderr.getvalue())
        self.assertIn("RunPod source upload/start failed (exit 1)", stderr.getvalue())
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(list((self.root / "artifacts").rglob("remote.json")), [])


if __name__ == "__main__":
    unittest.main()
