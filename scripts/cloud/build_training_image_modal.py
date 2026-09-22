#!/usr/bin/env python3
"""Build the RunPod Docker image in a temporary Modal CPU VM, or test Docker alone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FILES = {"Dockerfile", ".dockerignore", "scripts/cloud/build_image_models.py"}
RESOURCES = {
    "smoke": {"cpu": (1.0, 1.0), "memory": 2048, "timeout": 5 * 60},
    "build": {"cpu": (4.0, 4.0), "memory": 32 * 1024, "timeout": 2 * 60 * 60},
}
SMOKE_DOCKERFILE = (
    'FROM busybox:1.37.0\n'
    'RUN test "$(uname -m)" = "x86_64"\n'
    'CMD ["echo", "Modal Docker smoke test passed"]\n'
)
WAIT_FOR_DOCKER = """\
for attempt in $(seq 1 120); do
    if docker info >/dev/null 2>&1; then exit 0; fi
    sleep 1
done
echo 'Docker did not become ready within 120 seconds' >&2
exit 1
"""
REGISTRY_LOGIN = """\
printf '%s' "$REGISTRY_PASSWORD" | docker login "$1" \
    --username "$REGISTRY_USERNAME" --password-stdin
"""


def registry_for_image(image: str) -> str:
    """Require an explicit registry and version tag before allocating a VM."""
    match = re.fullmatch(
        r"(?P<registry>[a-z0-9][a-z0-9.-]*(?::[0-9]+)?)/"
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
        r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})", image,
    )
    if match is None or "." not in match["registry"]:
        raise ValueError("--image must include a registry and tag, e.g. ghcr.io/owner/repo:run-v1")
    if match["tag"] in {"latest", "buildcache"}:
        raise ValueError("choose a new version tag; latest and buildcache are reserved")
    return match["registry"]


def build_context(root: Path) -> tuple[bytes, dict]:
    """Archive the working tree selected by .dockerignore, without following symlinks."""
    from modal.file_pattern_matcher import FilePatternMatcher

    root = root.resolve()
    ignored = FilePatternMatcher.from_file(root / ".dockerignore")
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
    ).decode().split("\0")
    files = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(set(paths) - {""}):
            relative = Path(name)
            if ignored(relative):
                continue
            path = root / relative
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
                raise ValueError(f"build context must not contain symlinks: {relative}")
            if not path.is_file():
                continue
            content = path.read_bytes()
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(content)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            archive.addfile(info, io.BytesIO(content))
            files[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    if missing := CONTEXT_FILES - files.keys():
        raise ValueError(f"build context is missing required files: {sorted(missing)}")
    if extra := files.keys() - CONTEXT_FILES:
        raise ValueError(f"runtime/model image must not include application files: {sorted(extra)}")
    data = gzip.compress(buffer.getvalue(), mtime=0)
    return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "files": files}


def model_credentials() -> dict[str, str]:
    """Read standard local credentials for transient BuildKit secrets, never build arguments."""
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise ValueError("A local Hugging Face token is required to build the model image")
    credentials = {"HF_TOKEN": token}
    if api_token := os.environ.get("KAGGLE_API_TOKEN"):
        credentials["KAGGLE_API_TOKEN"] = api_token
    else:
        username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
        if not (username and key):
            path = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
            legacy = json.loads(path.read_text()) if path.is_file() else {}
            username, key = legacy.get("username"), legacy.get("key")
        if not all(isinstance(value, str) and value for value in (username, key)):
            raise ValueError("Local Kaggle credentials are required to build the model image")
        credentials.update(KAGGLE_USERNAME=username, KAGGLE_KEY=key)
    return credentials


def run_command(sandbox, *command: str, log, timeout: int, secrets=()) -> None:
    """Follow a remote log file; process completion never depends on Modal's log stream EOF."""
    remote_log = f"/tmp/image-build-{uuid4().hex}.log"
    process = sandbox.exec("sh", "-c", 'logfile=$1\nshift\nexec "$@" > "$logfile" 2>&1',
                           "build-command", remote_log, *command, timeout=timeout, secrets=secrets, bufsize=1)
    offset = 0

    def forward_output():
        nonlocal offset
        try:
            output = sandbox.filesystem.read_text(remote_log)
        except FileNotFoundError:
            return  # The shell has not opened its log yet.
        fresh = output[offset:]
        print(fresh, end="", flush=True)
        log.write(fresh)
        log.flush()
        offset = len(output)

    while process.poll() is None:
        forward_output()
        time.sleep(2)
    process.wait()
    forward_output()
    if process.returncode != 0:
        raise RuntimeError(f"{command[0]} failed with exit code {process.returncode}; see build.log")


def execute(plan: dict, context: bytes | None, destination: Path) -> None:
    """Run Docker in a VM and terminate it on success, failure or keyboard interruption."""
    import modal

    secret = None
    model_secret = None
    secret_arguments = []
    if plan["command"] == "build":
        secret = modal.Secret.from_name(
            plan["registry_secret"], required_keys=["REGISTRY_USERNAME", "REGISTRY_PASSWORD"],
        )
        secret.hydrate()
        credentials = model_credentials()
        model_secret = modal.Secret.from_dict(credentials)
        for name in sorted(credentials):
            secret_arguments.extend(("--secret", f"id={name.lower()},env={name}"))
        del credentials
    # This small outer image only supplies Docker; our Dockerfile builds the training image.
    builder = (
        modal.Image.from_registry("ubuntu:24.04")
        .env({"DEBIAN_FRONTEND": "noninteractive"})
        .apt_install("docker.io", "docker-buildx", "ca-certificates")
        .run_commands("mkdir -p /build/context /build/smoke")
    )
    app = modal.App.lookup("diffusion-arc-image-builder", create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "/usr/bin/dockerd", "--host=unix:///var/run/docker.sock",
            app=app, image=builder, experimental_options={"vm_runtime": True},
            env={"DOCKER_CONFIG": "/tmp/registry-auth"}, **RESOURCES[plan["command"]],
        )
    try:
        plan.update(sandbox_id=sandbox.object_id, status="running", modal_version=modal.__version__)
        (destination / "build.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Modal sandbox: {sandbox.object_id}", flush=True)
        with (destination / "build.log").open("w", buffering=1) as log:
            def run(*command, timeout=300, secrets=()):
                run_command(sandbox, *command, log=log, timeout=timeout, secrets=secrets)

            run("sh", "-ec", WAIT_FOR_DOCKER, timeout=130)
            if secret is not None:
                run("sh", "-ec", REGISTRY_LOGIN, "registry-login", plan["registry"], secrets=[secret])
            run("sh", "-ec", 'test "$(uname -m)" = x86_64 && docker version && docker buildx version')
            sandbox.filesystem.write_text(SMOKE_DOCKERFILE, "/build/smoke/Dockerfile")
            run("docker", "buildx", "build", "--platform", "linux/amd64", "--progress", "plain",
                "--load", "--tag", "diffusion-arc-smoke", "/build/smoke")
            run("docker", "run", "--rm", "diffusion-arc-smoke")
            if plan["command"] == "build":
                sandbox.filesystem.write_bytes(context, "/build/context.tar.gz")
                run("tar", "-xzf", "/build/context.tar.gz", "-C", "/build/context")
                run("sh", "-ec", "df -h /; test $(df -Pk / | awk 'NR == 2 {print $4}') -ge 104857600")
                cache = plan["image"].rsplit(":", 1)[0] + ":models-buildcache"
                run("docker", "buildx", "create", "--name", "training-builder", "--driver",
                    "docker-container", "--use")
                run(
                    "docker", "buildx", "build", "--platform", "linux/amd64", "--progress", "plain",
                    "--tag", plan["image"], "--push", "--metadata-file", "/build/metadata.json",
                    "--cache-from", f"type=registry,ref={cache}",
                    "--cache-to", f"type=registry,ref={cache},mode=min",
                    "--label", f"org.opencontainers.image.revision={plan['source_revision']}",
                    "--label", f"io.diffusion-arc.context-sha256={plan['context']['sha256']}",
                    *secret_arguments, "/build/context", timeout=RESOURCES["build"]["timeout"],
                    secrets=[model_secret],
                )
                metadata = json.loads(sandbox.filesystem.read_text("/build/metadata.json"))
                digest = metadata["containerimage.digest"]
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise ValueError("Docker returned an invalid image digest")
                reference = plan["image"].rsplit(":", 1)[0] + "@" + digest
                run("docker", "buildx", "imagetools", "inspect", reference)
                for name, options in (("registry-index.json", ("--raw",)),
                                      ("registry-image.json", ("--format", "{{json .Image}}"))):
                    run("sh", "-ec", 'destination=$1\nshift\nexec docker buildx imagetools inspect "$@" > "$destination"',
                        "registry-inspection", "/build/" + name, reference, *options)
                    payload = json.loads(sandbox.filesystem.read_text("/build/" + name))
                    (destination / name).write_text(json.dumps(payload, indent=2) + "\n")
                plan.update(image_digest=digest, runpod_image=reference)
                (destination / "docker-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                run("docker", "buildx", "build", "--platform", "linux/amd64", "--progress", "plain",
                    "--target", "bundle-manifest", "--output", "type=local,dest=/build/model-manifest",
                    *secret_arguments, "/build/context", secrets=[model_secret])
                manifest = json.loads(sandbox.filesystem.read_text("/build/model-manifest/manifest.json"))
                (destination / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                plan["model_bytes"] = sum(item["bytes"] for component in ("flux", "nvarc")
                                          for item in manifest[component]["files"].values())
                print(f"RunPod Container Image: {reference}", flush=True)
        plan["status"] = "succeeded"
    except BaseException:
        plan["status"] = "failed"
        raise
    finally:
        try:
            sandbox.terminate(wait=True)
        finally:
            plan["finished_at"] = datetime.now(timezone.utc).isoformat()
            (destination / "build.json").write_text(json.dumps(plan, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "build"))
    parser.add_argument("--image", help="destination registry/repository:version for a full build")
    parser.add_argument("--registry-secret", default="diffusion-arc-registry",
                        help="existing Modal Secret with REGISTRY_USERNAME and REGISTRY_PASSWORD")
    parser.add_argument("--dry-run", action="store_true", help="inspect inputs locally; make no Modal calls")
    args = parser.parse_args(argv)
    if args.command == "build" and not args.image:
        parser.error("build requires --image")
    if args.command == "smoke" and args.image:
        parser.error("smoke does not publish an image; omit --image")
    registry = registry_for_image(args.image) if args.image else None
    context, context_manifest = build_context(ROOT) if args.command == "build" else (None, None)
    plan = {
        "command": args.command, "resources": RESOURCES[args.command], "image": args.image,
        "registry": registry, "registry_secret": args.registry_secret if registry else None,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "context": context_manifest, "status": "prepared",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    destination = ROOT / "artifacts/image-builds" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    if context is not None:
        (destination / "context.tar.gz").write_bytes(context)
    (destination / "build.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Local build artifacts: {destination}", flush=True)
    execute(plan, context, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
