#!/usr/bin/env python3
"""Upload local code and start a job on an existing Pod via direct SSH or RunPod's proxy."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import select
import shlex
import subprocess
import sys
import tarfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
MAX_SOURCE_BYTES = 64 * 1024**2


def source_snapshot(root: Path) -> tuple[bytes, dict]:
    """Capture tracked/modified/untracked source files, excluding ignored files and artifacts."""
    root = root.resolve()
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
    ).decode().split("\0")
    files, buffer, total = {}, io.BytesIO(), 0
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(set(names) - {""}):
            relative = PurePosixPath(name)
            allowed = (name in ("README.md", "pyproject.toml")
                       or (relative.parts[0] in ("src", "scripts") and relative.suffix == ".py")
                       or (relative.parts[0] == "configs" and relative.suffix == ".json"))
            if not allowed or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            path = root / name
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
                raise ValueError(f"Source snapshot cannot contain symlinks: {name}")
            if not path.is_file():
                continue  # Locally deleted tracked files stay deleted in the snapshot.
            if path.stat().st_size > 8 * 1024**2:
                raise ValueError(f"Unexpectedly large source file: {name}")
            content = path.read_bytes()
            total += len(content)
            if total > MAX_SOURCE_BYTES:
                raise ValueError("Source snapshot exceeds 64 MiB")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
            files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        required = {"pyproject.toml", "README.md", "scripts/train.py", "scripts/train_single_gpu.py",
                    "scripts/cloud/train_runpod.py", "scripts/cloud/launch_runpod.py"}
        if missing := required - files.keys():
            raise ValueError(f"Source snapshot is incomplete: {sorted(missing)}")
        source_snapshot_manifest = {
            "schema_version": 1, "files": files,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        }
        content = json.dumps(source_snapshot_manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("_source.json")
        info.size, info.mode = len(content), 0o644
        archive.addfile(info, io.BytesIO(content))
    return gzip.compress(buffer.getvalue(), mtime=0), source_snapshot_manifest


def ssh_arguments(command: str) -> list[str]:
    """Parse a copied SSH command and select a terminal for RunPod's proxy."""
    parts = shlex.split(command)
    if not parts or parts.pop(0) != "ssh":
        raise ValueError("--ssh must be the connection command beginning with ssh")
    options, target = [], None
    while parts:
        item = parts.pop(0)
        if item in ("-p", "-i"):
            if not parts:
                raise ValueError(f"Missing SSH argument for {item}")
            value = parts.pop(0)
            if item == "-p":
                if not value.isdecimal() or not 1 <= int(value) <= 65535:
                    raise ValueError("SSH port must be between 1 and 65535")
            else:
                key = Path(value).expanduser().resolve()
                if not key.is_file():
                    raise ValueError(f"SSH identity file is missing: {key}")
                value = str(key)
            options.extend((item, value))
        elif target is None and re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9.:-]*", item):
            target = item
        else:
            raise ValueError("Use a plain SSH target with optional -p and -i; shell commands are not accepted")
    if target is None:
        raise ValueError("SSH command is missing its destination")
    proxy = target.rsplit("@", 1)[-1].lower() == "ssh.runpod.io"
    return ["ssh", "-tt" if proxy else "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "EscapeChar=none",
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            *options, target]


def pod_environment() -> dict[str, str]:
    """Recover selected Pod variables that an sshd login may omit; never print or save them."""
    import os
    from pathlib import Path

    environment = os.environ.copy()
    names = {
        "HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN", "HF_OUTPUT_REPO",
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_BASE_URL", "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "LD_LIBRARY_PATH",
    }
    try:
        for entry in Path("/proc/1/environ").read_bytes().split(b"\0"):
            name, separator, value = entry.partition(b"=")
            key = name.decode()
            if separator and key in names and key not in environment:
                environment[key] = value.decode()
    except OSError as error:
        raise RuntimeError("Cannot read Pod environment; use the Pod's root SSH account") from error
    return environment


def remote_preflight(command: str, workspace: str) -> None:
    """Check remote secrets and the mounted workspace before sending any project files."""
    from pathlib import Path

    environment = pod_environment()
    names = {"run": ("HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN"),
             "finetune": ("HF_TOKEN", "WANDB_API_KEY", "KAGGLE_API_TOKEN"),
             "continue": ("HF_TOKEN", "WANDB_API_KEY"), "upload": ("HF_TOKEN",),
             "diagnose": ("HF_TOKEN",)}[command]
    missing = [name for name in names if not environment.get(name, "").strip()
               or environment[name].strip().startswith("{{")]
    if missing:
        raise ValueError("Missing Pod secrets: " + ", ".join(missing))
    if not Path(workspace).is_dir():
        raise ValueError(f"Workspace disk is not mounted: {workspace}")
    print("SSH, Pod secrets and workspace are available", flush=True)


def receive_and_start(digest: str, workspace: str, request_json: str) -> None:
    """Verify a streamed source archive, publish it atomically, and detach its worker."""
    import hashlib
    import io
    import json
    import os
    from pathlib import Path, PurePosixPath
    import subprocess
    import sys
    import tarfile
    import tempfile

    content = sys.stdin.buffer.read(64 * 1024**2 + 1)
    if len(content) > 64 * 1024**2 or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Transferred source archive failed its checksum/size check")
    code = Path(workspace) / "code"
    code.mkdir(exist_ok=True)
    destination = code / digest
    with tempfile.TemporaryDirectory(prefix=".upload-", dir=code) as temporary:
        staged = Path(temporary) / "source"
        staged.mkdir()
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
            seen, total = set(), 0
            for member in archive:
                path = PurePosixPath(member.name)
                total += member.size
                if (not member.isfile() or path.is_absolute() or ".." in path.parts
                        or member.name in seen or total > 64 * 1024**2):
                    raise ValueError("Source archive has unsafe or duplicate members")
                seen.add(member.name)
                target = staged / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    output.write(source.read())
        source_snapshot_manifest = json.loads((staged / "_source.json").read_text())
        if seen != set(source_snapshot_manifest["files"]) | {"_source.json"}:
            raise ValueError("Source archive differs from its file inventory")
        for name, item in source_snapshot_manifest["files"].items():
            if hashlib.sha256((staged / name).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Source file checksum mismatch: {name}")
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("Source snapshot directory cannot be a symlink")
            actual = {str(path.relative_to(destination)) for path in destination.rglob("*")
                      if path.is_file() or path.is_symlink()}
            if actual != seen:
                raise ValueError("Existing source snapshot file inventory was changed")
            for name in seen:
                target = destination / name
                if (target.is_symlink() or not target.is_file()
                        or not target.resolve().is_relative_to(destination)
                        or target.read_bytes() != (staged / name).read_bytes()):
                    raise ValueError(f"Existing source snapshot was changed: {name}")
        else:
            staged.rename(destination)

    request = json.loads(request_json)
    job = Path(workspace) / "jobs" / request["job_id"]
    job.mkdir(parents=True, exist_ok=False)
    request.update(source_directory=str(destination), source_sha256=digest)
    (job / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    environment = pod_environment()
    environment.update(PYTHONPATH=f"{destination / 'src'}:{destination}", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1")
    with (job / "console.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-c", "from scripts.cloud.launch_runpod import run_remote_job; "
             "import sys; raise SystemExit(run_remote_job(sys.argv[1]))", str(job)],
            cwd=destination, env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
    (job / "worker.json").write_text(json.dumps({"pid": process.pid}) + "\n")
    print(json.dumps({"job_id": request["job_id"], "job_directory": str(job),
                      "source_directory": str(destination), "pid": process.pid}), flush=True)


def run_remote_job(job_directory: str) -> int:
    """Supervise the existing Pod entrypoint and publish its final exit status."""
    from datetime import datetime, timezone

    job = Path(job_directory)
    request = json.loads((job / "request.json").read_text())
    source = Path(request["source_directory"])
    status = {"state": "running", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}

    def save_status():
        temporary = job / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(job / "status.json")

    save_status()
    try:
        arguments = request["arguments"]
        entrypoint = "scripts/cloud/train_runpod.py"
        if arguments[0] == "continue":
            entrypoint, arguments = "scripts/cloud/continue_runpod.py", arguments[1:]
        elif arguments[0] == "diagnose":
            entrypoint, arguments = "scripts/cloud/diagnose_runpod.py", arguments[1:]
        elif arguments[0] == "finetune":
            entrypoint, arguments = "scripts/cloud/finetune_runpod.py", arguments[1:]
        command = [sys.executable, str(source / entrypoint), *arguments]
        result = subprocess.run(command, cwd=source, check=False)
        code = result.returncode if result.returncode >= 0 else 128 - result.returncode
    except Exception as error:
        print(f"Pod launcher failed: {type(error).__name__}: {error}", flush=True)
        code = 1
    status.update(state="succeeded" if code == 0 else "failed", exit_code=code,
                  finished_at=datetime.now(timezone.utc).isoformat())
    save_status()
    return code


def follow_remote_job(job_directory: str) -> int:
    """Stream a detached worker's log until completion, reporting a lost worker explicitly."""
    import json
    import os
    from pathlib import Path
    import sys
    import time

    job = Path(job_directory)
    worker = json.loads((job / "worker.json").read_text())
    with (job / "console.log").open("rb") as log:
        while True:
            chunk = log.read(65536)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                continue
            status = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
            if "exit_code" in status:
                sys.stdout.buffer.write(log.read())
                sys.stdout.buffer.flush()
                return status["exit_code"]
            try:
                os.kill(worker["pid"], 0)
                # A PID can be reused after a Pod restart; require our unique job argument.
                if Path("/proc").is_dir():
                    command_line = Path(f"/proc/{worker['pid']}/cmdline").read_bytes().split(b"\0")
                    if str(job).encode() not in command_line:
                        raise ProcessLookupError
            except (ProcessLookupError, FileNotFoundError):
                # Completion can occur between the status read and the process check.
                status = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
                if "exit_code" in status:
                    sys.stdout.buffer.write(log.read())
                    sys.stdout.buffer.flush()
                    return status["exit_code"]
                print("Worker exited without a final status; inspect the Pod and its checkpoints", flush=True)
                return 1
            time.sleep(0.5)


def proxy_session(marker: str) -> None:
    """Execute one framed Python request inside the proxy's terminal, without echoing source."""
    import base64
    import io
    import json
    import os
    import select
    import sys
    import termios
    import traceback
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    wire = sys.stdout.buffer

    def read_exact(size):
        chunks = []
        while size:
            if not select.select([fd], [], [], 60)[0]:
                raise TimeoutError("Timed out receiving source through the RunPod proxy")
            chunk = os.read(fd, min(size, 65536))
            if not chunk:
                raise EOFError("RunPod proxy source transfer was interrupted")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    code = 1
    try:
        # Raw input avoids terminal line limits and control-character processing.
        tty.setraw(fd)
        wire.write(f"\n{marker}:INPUT\n".encode())
        wire.flush()
        size = int(read_exact(8), 16)
        if size > 96 * 1024**2:
            raise ValueError("RunPod proxy request exceeds 96 MiB")
        request = json.loads(read_exact(size))
        sys.argv = ["-c", *request["arguments"]]
        sys.stdin = io.TextIOWrapper(io.BytesIO(base64.b64decode(request["input"], validate=True)))
        try:
            exec(compile(request["program"], "<runpod-launcher>", "exec"), {"__name__": "__main__"})
            code = 0
        except SystemExit as error:
            code = error.code if isinstance(error.code, int) else 0 if error.code is None else 1
            if isinstance(error.code, str):
                print(error.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        wire.write(f"\n{marker}:EXIT:{code}\n".encode())
        wire.flush()
        termios.tcsetattr(fd, termios.TCSANOW, previous)
    raise SystemExit(code)


def proxy_python(ssh: list[str], python: str, program: str, arguments: list[str], *,
                 input: bytes = b"", stdout=None, check=False, timeout=None):
    """Run Python over an interactive-only SSH connection; terminal output combines both streams."""
    marker = "ARC_" + uuid.uuid4().hex
    payload = json.dumps({"program": program, "arguments": arguments,
                          "input": base64.b64encode(input).decode()}).encode()
    deadline = time.monotonic() + timeout if timeout is not None else None
    startup_deadline = min(deadline, time.monotonic() + 45) if deadline else time.monotonic() + 45
    buffer = b""
    captured = bytearray()
    phase = "opening the proxy shell"
    with subprocess.Popen(ssh, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as process:
        os.set_blocking(process.stdin.fileno(), False)

        def read_chunk(until):
            remaining = None if until is None else max(0, until - time.monotonic())
            if not select.select([process.stdout], [], [], remaining)[0]:
                raise RuntimeError(f"RunPod proxy timed out while {phase}; no job start was confirmed"
                                   if phase != "receiving remote output" else
                                   "RunPod proxy timed out receiving output; a started job may still be running")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                detail = buffer[-2000:].decode(errors="replace").strip()
                raise RuntimeError(f"RunPod proxy disconnected before reporting completion. {detail}")
            return chunk

        def send(content):
            remaining = memoryview(content)
            until = min(deadline, time.monotonic() + 60) if deadline else time.monotonic() + 60
            while remaining:
                if not select.select([], [process.stdin], [], max(0, until - time.monotonic()))[1]:
                    raise subprocess.TimeoutExpired(ssh, timeout or 60)
                remaining = remaining[os.write(process.stdin.fileno(), remaining[:65536]):]

        def wait_for(token):
            nonlocal buffer
            while token not in buffer:
                # RunPod can translate LF to CRLF outside the Pod's raw terminal.
                buffer = (buffer + read_chunk(startup_deadline)).replace(b"\r\n", b"\n")
                if len(buffer) > 1024**2:
                    raise RuntimeError("Unexpected output while opening the RunPod proxy terminal")
            buffer = buffer.split(token, 1)[1]

        def emit(content):
            if stdout == subprocess.PIPE:
                captured.extend(content)
            elif stdout != subprocess.DEVNULL:
                stream = sys.stdout.buffer if stdout is None else stdout
                stream.write(content)
                stream.flush()

        try:
            # Enter raw mode with a small command before sending code beyond terminal line limits.
            send(f"stty -echo; printf '\\n{marker}:SHELL\\n'\n".encode())
            wait_for(f"\n{marker}:SHELL\n".encode())
            phase = "starting Python in the proxy terminal"
            stub = ("import sys,termios,tty\nsaved=termios.tcgetattr(0)\n"
                    "try:\n tty.setraw(0)\n"
                    f" print('\\n{marker}:BOOT',flush=True)\n"
                    " size=int(sys.stdin.buffer.read(8),16)\n"
                    " exec(sys.stdin.buffer.read(size))\n"
                    "finally:\n termios.tcsetattr(0,termios.TCSANOW,saved)\n")
            send(("exec " + shlex.join([python, "-u", "-c", stub]) + "\n").encode())
            wait_for(f"\n{marker}:BOOT\n".encode())
            phase = "starting the source receiver"
            bootstrap = (inspect.getsource(proxy_session) + f"\nproxy_session({marker!r})\n").encode()
            send(f"{len(bootstrap):08x}".encode() + bootstrap)
            wait_for(f"\n{marker}:INPUT\n".encode())
            send(f"{len(payload):08x}".encode() + payload)
            phase = "receiving remote output"
            ending = f"\n{marker}:EXIT:".encode()
            while True:
                position = buffer.find(ending)
                if position >= 0:
                    emit(buffer[:position])
                    buffer = buffer[position + len(ending):]
                    while b"\n" not in buffer:
                        buffer = (buffer + read_chunk(deadline)).replace(b"\r\n", b"\n")
                    code = int(buffer.split(b"\n", 1)[0])
                    break
                # Keep a partial end marker across SSH reads while streaming all other output.
                if len(buffer) > len(ending):
                    emit(buffer[:-len(ending)])
                    buffer = buffer[-len(ending):]
                buffer = (buffer + read_chunk(deadline)).replace(b"\r\n", b"\n")
            result = subprocess.CompletedProcess(ssh, code, bytes(captured) if stdout == subprocess.PIPE else None)
            if check:
                result.check_returncode()
            return result
        finally:
            # Only the local transport is closed; a started detached worker stays on the Pod.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def remote_python(ssh: list[str], python: str, functions: tuple, invocation: str,
                  arguments: list[str], **kwargs):
    program = "\n\n".join(inspect.getsource(function) for function in functions)
    program = "from __future__ import annotations\n" + program + "\nimport sys\n" + invocation
    if ssh[-1].rsplit("@", 1)[-1].lower() == "ssh.runpod.io":
        return proxy_python(ssh, python, program, arguments, **kwargs)
    return subprocess.run([*ssh, shlex.join([python, "-c", program, *arguments])], **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "continue", "finetune", "upload", "diagnose", "logs"))
    parser.add_argument("--ssh", required=True, help="Copied RunPod SSH command (direct TCP or ssh.runpod.io)")
    parser.add_argument("--run-name")
    parser.add_argument("--job-id", help="Job printed by a previous launch; required for logs")
    parser.add_argument("--workspace", default="/workspace", help="Mounted directory on the Pod")
    parser.add_argument("--python", default="/opt/venv/bin/python", help="Prepared interpreter on the Pod")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", help="Absolute checkpoint path on the Pod")
    parser.add_argument("--from-run", help="Completed local run supplying the next epoch's full checkpoint")
    parser.add_argument("--augmentation-seed", type=int, help="Fresh data seed for the next epoch")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Return after starting; otherwise follow logs")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source/command locally; no SSH or writes")
    for option in ("output-repo", "wandb-project", "wandb-entity"):
        parser.add_argument(f"--{option}")
    args = parser.parse_args(argv)
    if args.config is None:
        recipe = "diagnostics/partial_denoising_b3.json" if args.command == "diagnose" else "training/arc_agi_1_epoch.json"
        if args.command == "finetune":
            recipe = "training/arc_agi_1_epoch3_high_noise.json"
        args.config = ROOT / "configs" / recipe
    ssh = ssh_arguments(args.ssh)
    for value in (args.workspace, args.python, args.resume):
        if value is not None and (not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
            parser.error("Remote workspace, Python and resume paths must be absolute without '..'")
    if args.command == "logs":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.job_id or ""):
            parser.error("logs requires --job-id from a previous launch")
        job = str(PurePosixPath(args.workspace) / "jobs" / args.job_id)
        if args.dry_run:
            print(json.dumps({"command": "logs", "job_directory": job}, indent=2))
            return 0
        return remote_python(ssh, args.python, (follow_remote_job,),
                             "raise SystemExit(follow_remote_job(sys.argv[1]))", [job]).returncode
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_name or ""):
        parser.error("requires --run-name with letters, digits, underscores or hyphens")
    if args.command not in ("run", "continue", "finetune") and (args.prepare_only or args.resume):
        parser.error("--prepare-only and --resume apply only to run/continue/finetune")
    if args.command == "continue":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.from_run or ""):
            parser.error("continue requires --from-run naming a completed run")
        if args.from_run == args.run_name:
            parser.error("continue requires a new local --run-name; W&B keeps the previous run")
        if args.wandb_project or args.wandb_entity:
            parser.error("continue inherits the checkpoint's W&B project and entity")
    elif args.from_run is not None or args.augmentation_seed is not None:
        parser.error("--from-run and --augmentation-seed apply only to continue")
    if args.command == "diagnose":
        if args.output_repo or args.wandb_project or args.wandb_entity:
            parser.error("diagnose writes local Pod artifacts; upload and W&B options do not apply")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.cloud.diagnose_runpod import load_config
        diagnostic = load_config(args.config)
    if args.command == "finetune":
        sys.path.insert(0, str(ROOT / "src"))
        from diffusion_arc.training.config import load_training_config
        recipe_config = load_training_config(args.config, expected_type="training_run")
        if recipe_config.recipe["models"]["initialization"] != "transformer_weights":
            parser.error("finetune requires a transformer_weights initialization config")
    content, source_snapshot_manifest = source_snapshot(ROOT)
    try:
        relative_config = args.config.resolve().relative_to(ROOT)
    except ValueError:
        parser.error("--config must be inside this repository")
    if relative_config.as_posix() not in source_snapshot_manifest["files"]:
        parser.error("--config must be a non-ignored JSON file inside this repository's configs directory")
    if args.command == "diagnose":
        dependency = (args.config.resolve().parent / diagnostic["training_config"]).resolve().relative_to(ROOT)
        if dependency.as_posix() not in source_snapshot_manifest["files"]:
            parser.error("diagnostic training_config must be included in the source snapshot")
    digest = hashlib.sha256(content).hexdigest()
    arguments = [args.command, "--run-name", args.run_name, "--workspace", args.workspace]
    if args.command in ("run", "diagnose", "finetune"):
        arguments.extend(("--config", str(relative_config)))
    for option in ("resume", "output_repo", "wandb_project", "wandb_entity", "from_run", "augmentation_seed"):
        value = getattr(args, option)
        if value is not None:
            arguments.extend(("--" + option.replace("_", "-"), str(value)))
    if args.prepare_only:
        arguments.append("--prepare-only")
    job_id = args.run_name + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    request = {"job_id": job_id, "arguments": arguments}
    plan = {**request, "source_sha256": digest, "source_bytes": len(content),
            "source_file_count": len(source_snapshot_manifest["files"]), "ssh": ssh, "workspace": args.workspace}
    if args.command == "diagnose":
        plan["diagnostic"] = diagnostic
    if args.command == "finetune":
        plan["initialization"] = recipe_config.recipe_dict()["models"]
        plan["flow"] = recipe_config.recipe_dict()["flow"]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    remote_python(ssh, args.python, (pod_environment, remote_preflight),
                  "remote_preflight(*sys.argv[1:])", [args.command, args.workspace], check=True)
    artifacts = ROOT / "artifacts/runpod-launches" / job_id
    artifacts.mkdir(parents=True)
    (artifacts / "source.tar.gz").write_bytes(content)
    (artifacts / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Job {job_id}: uploading {len(source_snapshot_manifest['files'])} source files ({len(content) / 1024**2:.2f} MiB)", flush=True)
    try:
        result = remote_python(ssh, args.python, (pod_environment, receive_and_start),
                               "receive_and_start(*sys.argv[1:])", [digest, args.workspace, json.dumps(request)],
                               input=content, stdout=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as error:
        # Continuation and finetuning import main, bypassing this module's CLI handler.
        if error.output:
            print(error.output.decode(errors="replace") if isinstance(error.output, bytes) else error.output,
                  file=sys.stderr)
        print(f"RunPod source upload/start failed (exit {error.returncode}).", file=sys.stderr)
        return 1
    receipt = json.loads(result.stdout)
    (artifacts / "remote.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Started job {job_id}; local source snapshot: {artifacts}", flush=True)
    print("The job continues if this terminal disconnects. Reconnect with:", flush=True)
    print(shlex.join([sys.executable, str(Path(__file__).resolve()), "logs", "--ssh", args.ssh,
                      "--workspace", args.workspace, "--python", args.python, "--job-id", job_id]), flush=True)
    if args.detach:
        return 0
    return remote_python(ssh, args.python, (follow_remote_job,),
                         "raise SystemExit(follow_remote_job(sys.argv[1]))", [receipt["job_directory"]]).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        if error.output:
            print(error.output.decode(errors="replace") if isinstance(error.output, bytes) else error.output,
                  file=sys.stderr)
        print(f"RunPod command failed (exit {error.returncode}).", file=sys.stderr)
        raise SystemExit(1)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"RunPod launcher: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nLocal connection closed; a job already started on the Pod keeps running.", file=sys.stderr)
        raise SystemExit(130)
