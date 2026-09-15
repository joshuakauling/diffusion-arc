#!/usr/bin/env python3
"""Run one validation timing benchmark on a bounded, cached Modal H200."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import modal


REMOTE_ROOT = Path("/opt/diffusion-arc")
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
ENVIRONMENT = "diffusion-arc"
CACHED_IMAGE_ID = "im-IGaCiJ89DjpJQfQE43QTjh"
CONFIG = "configs/validation/benchmark_b3.json"
BUNDLE = "data/benchmarks/arc_agi_1_validation"
LIFECYCLE_SECONDS = 1200
WORKER_SECONDS = 1000
RESOURCES = {"gpu": "H200", "cpu": (8.0, 8.0), "memory": (65536, 65536),
             "timeout": 1100, "startup_timeout": 60}
BUDGET = {
    "lifecycle_seconds": LIFECYCLE_SECONDS,
    "resource_rate_usd_per_second": .001261 + 8 * .0000131 + 64 * .00000222,
    "price_source": "https://modal.com/pricing",
    "price_checked": "2026-09-11",
    "basis": "One H200, at most 8 CPU cores and 64 GiB RAM; no premium placement or retries",
}
BUDGET["maximum_resource_list_price_estimate_usd"] = LIFECYCLE_SECONDS * BUDGET["resource_rate_usd_per_second"]

app = modal.App("diffusion-arc-validation-benchmark")
model_volume = modal.Volume.from_name("diffusion-arc-models", environment_name=ENVIRONMENT,
                                     create_if_missing=False)
image = modal.Image.from_id(CACHED_IMAGE_ID)
for directory in ("src", "scripts", "configs", BUNDLE):
    image = image.add_local_dir(LOCAL_ROOT / directory, remote_path=str(REMOTE_ROOT / directory),
                               copy=False, ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(LOCAL_ROOT / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"), copy=False)


def validate_layout(repository, *, remote):
    sys.path.insert(0, str(repository))
    from scripts.benchmark_validation import load_config, validate_bundle

    config = repository / CONFIG
    _, runtime, bundle = load_config(config)
    if bundle != repository / BUNDLE:
        raise ValueError("only the agreed validation bundle may be mounted")
    validate_bundle(bundle)
    for path in (runtime.models.nvarc_checkpoint, runtime.models.flux_checkpoint):
        root = VOLUME_ROOT if remote else repository / "checkpoints"
        path.resolve().relative_to(root.resolve())
        if remote and not path.is_dir():
            raise FileNotFoundError(f"required cached model is missing: {path}")
    return config


def identity(repository):
    paths = [p for name in ("src", "scripts", "configs", BUNDLE)
             for p in (repository / name).rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
    paths.append(repository / "pyproject.toml")
    return {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


WORKER_CODE = """
from pathlib import Path
import signal
import sys
from scripts.benchmark_validation import run
def interrupted(signum, frame):
    raise TimeoutError('validation worker reached its deadline')
signal.signal(signal.SIGTERM, interrupted)
run(Path(sys.argv[1]), Path(sys.argv[2]))
"""


def run_worker(config, destination):
    with (destination / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", "-c", WORKER_CODE,
                                    str(config), str(destination / "measurement")],
                                   cwd=REMOTE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=WORKER_SECONDS - 10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise TimeoutError("validation measurement exceeded its worker budget") from None
        if code:
            raise RuntimeError(f"validation worker exited with code {code}; see worker.log")


@app.function(image=image, **RESOURCES, max_containers=1, min_containers=0,
              buffer_containers=0, scaledown_window=2, retries=0, nonpreemptible=False,
              env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
              volumes={str(VOLUME_ROOT): model_volume})
def benchmark(metadata: dict):
    if metadata.get("image_id") != CACHED_IMAGE_ID or metadata.get("source_identity") != identity(REMOTE_ROOT):
        raise ValueError("remote source or cached image differs from the submitted benchmark")
    run_id = metadata.get("run_id", "")
    if not isinstance(run_id, str) or re.fullmatch(r"\d{8}T\d{12}Z", run_id) is None:
        raise ValueError("invalid run identity")
    link = REMOTE_ROOT / "checkpoints"
    if link.is_symlink():
        if link.resolve() != VOLUME_ROOT:
            raise RuntimeError("unexpected checkpoint symlink")
    elif link.exists():
        raise RuntimeError("checkpoint path already exists")
    else:
        link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    config = validate_layout(REMOTE_ROOT, remote=True)
    destination = VOLUME_ROOT / "results/validation-benchmark" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "metadata": metadata, "resources": RESOURCES, "budget": BUDGET}
    started = time.monotonic()
    try:
        run_worker(config, destination)
        report = json.loads((destination / "measurement/report.json").read_text())
        if report["status"] != "complete" or report["summary"]["measured_repetitions"] != 2:
            raise RuntimeError("worker did not complete both measured validation rounds")
        manifest["status"] = "complete"
        return {"destination": str(destination), "summary": report["summary"], "workload": report["workload"]}
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["worker_wall_seconds"] = time.monotonic() - started
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        async def commit():
            await asyncio.wait_for(model_volume.commit.aio(), timeout=60)
        asyncio.run(commit())
        print(f"Validation artifacts: {destination}", flush=True)


async def bounded_call(metadata):
    call = None
    async def invoke():
        nonlocal call
        call = await benchmark.spawn.aio(metadata)
        return await call.get.aio()
    try:
        return await asyncio.wait_for(invoke(), timeout=LIFECYCLE_SECONDS - 30)
    except BaseException:
        if call is not None:
            try:
                await asyncio.wait_for(call.cancel.aio(terminate_containers=True), timeout=10)
            except BaseException as error:
                print(f"Cancellation failed; stopping this app: {error}", flush=True)
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "modal", "app", "stop",
            metadata["modal_app_id"], "--env", ENVIRONMENT, "--yes")
        try:
            await asyncio.wait_for(process.wait(), timeout=15)
        except BaseException:
            process.kill()
            raise
        raise


@app.local_entrypoint()
def main():
    validate_layout(LOCAL_ROOT, remote=False)
    if image.object_id != CACHED_IMAGE_ID:
        raise RuntimeError("source mounts unexpectedly changed the cached image")
    metadata = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                "image_id": CACHED_IMAGE_ID, "source_identity": identity(LOCAL_ROOT),
                "modal_app_id": app.app_id, "modal_version": modal.__version__, "command": sys.argv}
    print(json.dumps({"run_id": metadata["run_id"], "budget": BUDGET}), flush=True)
    print(json.dumps(asyncio.run(bounded_call(metadata)), indent=2))
