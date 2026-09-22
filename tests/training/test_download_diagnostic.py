"""Verified result downloads over direct SSH and the interactive RunPod proxy."""

import base64
from contextlib import redirect_stdout
import fcntl
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.cloud import diagnose_runpod
from scripts.cloud import download_diagnostic as download
from scripts.cloud import launch_runpod as transport
from tests.training import test_launch_runpod as transport_tests


class DiagnosticDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="diagnostic backup ' ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.run_name = "b3-partial-step1812"
        self.results = self.workspace / "diagnostics" / self.run_name
        (self.results / "comparisons").mkdir(parents=True)
        self.png = bytes(range(256)) * 100 + b"\x00\r\n\n\x03\xff"
        (self.results / "report.json").write_text(json.dumps({"status": "succeeded"}))
        (self.results / "records.jsonl").write_text('{"exact_grid":false}\n')
        (self.results / "comparisons/example.png").write_bytes(self.png)
        job = self.workspace / "jobs" / "the-job"
        job.mkdir(parents=True)
        (job / "request.json").write_text(json.dumps({"arguments": ["diagnose", "--run-name", self.run_name]}))
        (job / "status.json").write_text(json.dumps({"state": "succeeded", "exit_code": 0}))
        (job / "console.log").write_bytes(b"complete\n")
        other = self.workspace / "jobs" / "other-job"
        other.mkdir()
        (other / "request.json").write_text(json.dumps({"arguments": ["diagnose", "--run-name", self.run_name + "-other"]}))
        (other / "console.log").write_text("unrelated")
        self.destination = self.root / "downloaded"

    def payload(self, partial="0"):
        stream = io.StringIO()
        with redirect_stdout(stream):
            download.export_diagnostic(str(self.workspace), self.run_name, str(download.MAX_RESULT_BYTES), partial)
        return stream.getvalue().encode()

    def verify_result(self, content):
        manifest = download.save_download(content, self.destination, self.run_name)
        self.assertEqual(manifest["diagnostic_status"], "succeeded")
        self.assertEqual((self.destination / "comparisons/example.png").read_bytes(), self.png)
        self.assertEqual((self.destination / "logs/the-job/console.log").read_bytes(), b"complete\n")
        self.assertNotIn("logs/other-job/console.log", manifest["files"])
        self.assertEqual(json.loads((self.destination / "download.json").read_text()), manifest)
        for path, record in manifest["files"].items():
            self.assertEqual(hashlib.sha256((self.destination / path).read_bytes()).hexdigest(), record["sha256"])

    def test_direct_transport_restores_binary_results_and_matching_logs(self):
        shell = [sys.executable, "-c", "import os, sys; os.execv('/bin/sh', ['sh', '-c', sys.argv[-1]])"]
        result = transport.remote_python(shell, sys.executable, (download.export_diagnostic,),
            "export_diagnostic(*sys.argv[1:])", [str(self.workspace), self.run_name, str(download.MAX_RESULT_BYTES)],
            stdout=subprocess.PIPE, check=True, timeout=10)
        self.verify_result(result.stdout)

    def test_proxy_download_preserves_binary_bytes_across_crlf_translation(self):
        with transport_tests.LocalLaunchTests.terminal_transport(self, crlf=True):
            result = transport.remote_python(["fixture-ssh", "pod@ssh.runpod.io"], sys.executable,
                (download.export_diagnostic,), "export_diagnostic(*sys.argv[1:])",
                [str(self.workspace), self.run_name, str(download.MAX_RESULT_BYTES)],
                stdout=subprocess.PIPE, check=True, timeout=10)
        self.verify_result(result.stdout)

    def test_running_diagnostic_prevents_inconsistent_snapshot(self):
        with (self.workspace / ".diffusion-arc.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "still using this workspace"):
                self.payload()

    def test_partial_snapshot_preserves_complete_records_while_job_holds_lock(self):
        (self.results / "report.json").write_text(json.dumps({"status": "running"}))
        growing = b'{"exact_grid":false}\n{"unfinished":'
        (self.results / "records.jsonl").write_bytes(growing)
        (self.results / "report.json.tmp").write_text('{"unfinished":')
        with (self.workspace / ".diffusion-arc.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = download.save_download(self.payload("1"), self.destination, self.run_name)
            with self.assertRaisesRegex(RuntimeError, "still using this workspace"):
                self.payload()
        self.assertTrue(manifest["partial_snapshot"])
        self.assertEqual(manifest["diagnostic_status"], "running")
        self.assertEqual((self.destination / "records.jsonl").read_bytes(), b'{"exact_grid":false}\n')
        self.assertEqual((self.results / "records.jsonl").read_bytes(), growing)
        self.assertNotIn("report.json.tmp", manifest["files"])
        self.assertNotIn("comparisons/example.png", manifest["files"])
        self.assertEqual((self.destination / "logs/the-job/console.log").read_bytes(), b"complete\n")

    def test_partial_cli_uses_new_directory_and_sends_flag_without_relaunch(self):
        stdout = io.StringIO()
        args = ["download", "--ssh", "ssh root@192.0.2.1", "--run-name", self.run_name, "--partial"]
        with patch.object(transport, "remote_python") as remote, redirect_stdout(stdout):
            self.assertEqual(diagnose_runpod.main([*args, "--dry-run"]), 0)
            remote.assert_not_called()
        plan = json.loads(stdout.getvalue())
        self.assertTrue(plan["partial_snapshot"])
        self.assertTrue(Path(plan["local_directory"]).name.startswith(self.run_name + "-partial-"))
        with patch.object(transport, "remote_python", return_value=subprocess.CompletedProcess([], 0, self.payload("1"))) as remote, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(diagnose_runpod.main([*args, "--output", str(self.destination)]), 0)
        self.assertEqual(remote.call_args.args[4][-1], "1")
        self.assertTrue(json.loads((self.destination / "download.json").read_text())["partial_snapshot"])

    def test_remote_symlinks_and_oversized_backups_fail_before_output(self):
        with self.assertRaisesRegex(ValueError, "size limit"), redirect_stdout(io.StringIO()):
            download.export_diagnostic(str(self.workspace), self.run_name, "10")
        (self.results / "linked.json").symlink_to(self.results / "report.json")
        with self.assertRaisesRegex(ValueError, "non-regular"):
            self.payload()

    def test_corruption_publishes_no_partial_backup_and_can_be_retried(self):
        original = self.payload()
        payload = json.loads(original)
        payload["manifest"]["files"]["records.jsonl"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            download.save_download(json.dumps(payload).encode(), self.destination, self.run_name)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.destination.parent.glob(".diagnostic-download-*")))
        payload = json.loads(original)
        payload["manifest"]["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "archive failed"):
            download.save_download(json.dumps(payload).encode(), self.destination, self.run_name)
        self.verify_result(original)
        with self.assertRaises(FileExistsError):
            download.save_download(original, self.destination, self.run_name)

    def test_unsafe_archive_paths_and_symlinks_are_never_extracted(self):
        for name, mode in (("../outside.txt", stat.S_IFREG), ("/absolute.txt", stat.S_IFREG),
                           ("linked.txt", stat.S_IFLNK)):
            with self.subTest(name=name):
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    member = zipfile.ZipInfo(name)
                    member.external_attr = (mode | 0o644) << 16
                    archive.writestr(member, b"x")
                content = buffer.getvalue()
                manifest = {"schema_version": 1, "run_name": self.run_name,
                    "archive_sha256": hashlib.sha256(content).hexdigest(),
                    "files": {name: {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}}}
                payload = json.dumps({"manifest": manifest, "archive_base64": base64.b64encode(content).decode()}).encode()
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    download.save_download(payload, self.destination, self.run_name)
                self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / "outside.txt").exists())

    def test_download_subcommand_does_not_need_secrets_or_launch_a_job(self):
        payload = self.payload()
        arguments = ["download", "--ssh", "ssh root@192.0.2.1", "--run-name", self.run_name,
                     "--output", str(self.destination)]
        with patch.object(transport, "remote_python", return_value=subprocess.CompletedProcess([], 0, payload)) as remote, \
                patch.object(transport, "source_snapshot", side_effect=AssertionError("launch attempted")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(diagnose_runpod.main([*arguments, "--dry-run"]), 0)
            remote.assert_not_called()
            self.assertFalse(self.destination.exists())
            self.assertEqual(diagnose_runpod.main(arguments), 0)
        self.assertEqual(remote.call_count, 1)
        self.assertEqual((self.destination / "comparisons/example.png").read_bytes(), self.png)

    def test_remote_failure_leaves_local_results_unpublished(self):
        with patch.object(transport, "remote_python", return_value=subprocess.CompletedProcess([], 1, b"busy")), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Remote download failed"):
            download.main(["--ssh", "ssh root@192.0.2.1", "--run-name", self.run_name, "--output", str(self.destination)])
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
