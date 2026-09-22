#!/usr/bin/env python3
"""Download verified final results or a partial snapshot of a running diagnostic."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cloud import launch_runpod as transport


MAX_RESULT_BYTES = 128 * 1024**2


def export_diagnostic(workspace: str, run_name: str, maximum: str, partial: str = "0") -> None:
    """Export final artifacts under the lock, or completed text records while busy."""
    import base64
    from datetime import datetime, timezone
    import fcntl
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import re
    import sys
    import zipfile

    workspace = Path(workspace)
    maximum = int(maximum)
    if partial not in ("0", "1"):
        raise ValueError("Invalid partial snapshot flag")
    partial = partial == "1"
    if (not workspace.is_absolute() or ".." in workspace.parts
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name)):
        raise ValueError("Invalid remote workspace or run name")
    root = workspace / "diagnostics" / run_name
    if not root.is_dir() or root.is_symlink() or root.parent.is_symlink():
        raise ValueError(f"Diagnostic result directory is missing or linked: {root}")
    with (workspace / ".diffusion-arc.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not partial:
                raise RuntimeError("A launcher is still using this workspace; use --partial to save completed records now") from None
        paths = {}
        for path in sorted(root.rglob("*")):
            # Atomic JSON writes use temporary files; PNGs may still be mid-write.
            if partial and path.suffix not in (".json", ".jsonl"):
                continue
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f"Diagnostic contains a non-regular entry: {path}")
            if path.is_file():
                paths[path.relative_to(root).as_posix()] = path
        if not paths:
            raise ValueError("Diagnostic result directory is empty")
        if "logs" in {Path(name).parts[0] for name in paths} or "download.json" in paths:
            raise ValueError("Diagnostic results use a reserved backup filename")
        # Job receipts select logs by the exact command and run name, never by a name prefix.
        jobs = workspace / "jobs"
        for request_path in sorted(jobs.glob("*/request.json")):
            if request_path.is_symlink() or request_path.parent.is_symlink() or jobs.is_symlink():
                raise ValueError("Job receipts must not be symlinks")
            request = json.loads(request_path.read_text())
            arguments = request.get("arguments", [])
            if not isinstance(arguments, list) or arguments[:1] != ["diagnose"] or "--run-name" not in arguments:
                continue
            index = arguments.index("--run-name") + 1
            if index >= len(arguments) or arguments[index] != run_name:
                continue
            for name in ("request.json", "status.json", "worker.json", "console.log"):
                path = request_path.parent / name
                if path.is_symlink():
                    raise ValueError(f"Job artifact must not be a symlink: {path}")
                if path.is_file():
                    paths[f"logs/{request_path.parent.name}/{name}"] = path
        if len(paths) > 4096 or sum(path.stat().st_size for path in paths.values()) > maximum:
            raise ValueError("Diagnostic backup exceeds the file-count or 128 MiB size limit")
        inventory, total, buffer = {}, 0, io.BytesIO()
        report = {}
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in paths.items():
                with path.open("rb") as handle:
                    size = os.fstat(handle.fileno()).st_size
                    if total + size > maximum:
                        raise ValueError("Diagnostic files grew beyond the backup size limit")
                    content = handle.read(size)
                if partial and name.endswith(".jsonl"):
                    # Retain complete flushed records if a write crosses the snapshot boundary.
                    content = content[:content.rfind(b"\n") + 1]
                total += len(content)
                if total > maximum:
                    raise ValueError("Diagnostic files grew beyond the backup size limit")
                inventory[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                archive.writestr(name, content)
                if name == "report.json":
                    report = json.loads(content)
        content = buffer.getvalue()
        manifest = {
            "schema_version": 1, "run_name": run_name, "remote_directory": str(root),
            "diagnostic_status": report.get("status", "unknown"),
            "partial_snapshot": partial,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": hashlib.sha256(content).hexdigest(), "files": inventory,
        }
        # The proxy can translate LF to CRLF; base64 preserves PNG and archive bytes.
        sys.stdout.write(json.dumps({"manifest": manifest, "archive_base64": base64.b64encode(content).decode()}) + "\n")
        sys.stdout.flush()


def save_download(payload: bytes, destination: Path, run_name: str) -> dict:
    """Publish a local directory only after every archived file passes verification."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Local destination already exists; choose another --output: {destination}")
    value = json.loads(payload)
    manifest = value["manifest"]
    if manifest.get("schema_version") != 1 or manifest.get("run_name") != run_name:
        raise ValueError("Backup manifest describes another run or schema")
    content = base64.b64decode(value["archive_base64"], validate=True)
    if len(content) > MAX_RESULT_BYTES + 1024**2 or hashlib.sha256(content).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Downloaded archive failed its checksum or size check")
    inventory = manifest["files"]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if (not members or len(members) > 4096 or len(names) != len(set(names))
                or set(names) != set(inventory) or "download.json" in names
                or sum(member.file_size for member in members) > MAX_RESULT_BYTES):
            raise ValueError("Archive differs from its file inventory or exceeds the backup limit")
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (path.is_absolute() or any(part in ("", ".", "..") for part in member.filename.split("/"))
                    or "\\" in member.filename or member.is_dir()
                    or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Archive contains an unsafe path or non-regular entry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".diagnostic-download-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "results"
            staged.mkdir()
            for member in members:
                data = archive.read(member)
                expected = inventory[member.filename]
                if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Downloaded file checksum mismatch: {member.filename}")
                target = staged / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            (staged / "download.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Local destination was created during download: {destination}")
            staged.rename(destination)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", required=True, help="Copied SSH command, as used for launching")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--python", default="/opt/venv/bin/python", help="Interpreter on the Pod")
    parser.add_argument("--output", type=Path, help="New local destination directory")
    parser.add_argument("--partial", action="store_true", help="Save completed records and logs during a run; omit images")
    parser.add_argument("--dry-run", action="store_true", help="Show paths without SSH or writes")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name):
        parser.error("invalid run name")
    if any(not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
           for path in (args.workspace, args.python)):
        parser.error("remote workspace and Python paths must be absolute without '..'")
    ssh = transport.ssh_arguments(args.ssh)
    directory = args.run_name
    if args.partial:
        directory += "-partial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (args.output or ROOT / "artifacts/diagnostics" / directory).expanduser().absolute()
    if args.dry_run:
        print(json.dumps({"remote_directory": str(PurePosixPath(args.workspace) / "diagnostics" / args.run_name),
                          "local_directory": str(destination), "include_job_logs": True,
                          "partial_snapshot": args.partial}, indent=2))
        return 0
    if destination.exists() or destination.is_symlink():
        parser.error(f"local destination exists; choose another --output: {destination}")
    print(f"Downloading diagnostic {args.run_name} and matching job logs...", flush=True)
    result = transport.remote_python(
        ssh, args.python, (export_diagnostic,), "export_diagnostic(*sys.argv[1:])",
        [args.workspace, args.run_name, str(MAX_RESULT_BYTES), "1" if args.partial else "0"],
        stdout=subprocess.PIPE, timeout=300,
    )
    if result.returncode:
        if result.stdout:
            print(result.stdout.decode(errors="replace"), file=sys.stderr)
        raise RuntimeError(f"Remote download failed (exit {result.returncode}); no local backup was published")
    manifest = save_download(result.stdout, destination, args.run_name)
    print(f"Download verified: {len(manifest['files'])} files saved to {destination}", flush=True)
    print(f"Diagnostic status: {manifest['diagnostic_status']}. Backup receipt: {destination / 'download.json'}", flush=True)
    if args.partial:
        print("Partial snapshot saved. Images are omitted; run download without --partial after completion.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"Diagnostic download: {error}", file=sys.stderr)
        raise SystemExit(1)
