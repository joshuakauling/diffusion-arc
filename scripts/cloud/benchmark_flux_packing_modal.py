#!/usr/bin/env python3
"""Run the integrated packed training smoke test on one bounded cached H200."""

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

REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
ENVIRONMENT = "diffusion-arc"
CACHED_IMAGE_ID = "im-R5CncDboPfMQryWnq0UZn4"
CONFIG = "configs/training/benchmarks/single_gpu_b3_smoke.json"
DATASET = "data/benchmarks/single_gpu_b0"
RESULTS = "results/flux-packing-training-smoke"
LIFECYCLE_SECONDS = 480
CLEANUP_SECONDS = 30
MAX_WORKER_SECONDS = 330
FINAL_COMMIT_SECONDS = 60
RESOURCES = {"gpu": "H200", "cpu": (8.0, 8.0), "memory": (65536, 65536),
             "timeout": 390, "startup_timeout": 60}
PRICE_USD_PER_SECOND = 0.001261 + 8 * 0.0000131 + 64 * 0.00000222
BUDGET = {"lifecycle_seconds": LIFECYCLE_SECONDS,
          "resource_rate_usd_per_second": PRICE_USD_PER_SECOND,
          "maximum_list_price_estimate_usd": PRICE_USD_PER_SECOND * LIFECYCLE_SECONDS,
          "price_basis": "H200, eight CPU cores and 64 GiB RAM, both requested and limited; no premium placement"}

app = modal.App("diffusion-arc-flux-packing-training-smoke")
model_volume = modal.Volume.from_name("diffusion-arc-models", environment_name=ENVIRONMENT,
                                     create_if_missing=False)
image = modal.Image.from_id(CACHED_IMAGE_ID)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / directory,
        remote_path=str(REMOTE_REPOSITORY_ROOT / directory), copy=False,
        ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(LOCAL_REPOSITORY_ROOT / "pyproject.toml",
    remote_path=str(REMOTE_REPOSITORY_ROOT / "pyproject.toml"), copy=False)
image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / DATASET,
    remote_path=str(REMOTE_REPOSITORY_ROOT / DATASET), copy=False)


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints):
    from scripts.benchmark_single_gpu import check_inputs

    repository = repository.resolve()
    path = (repository / config).resolve()
    if path != repository / CONFIG:
        raise ValueError(f"this bounded smoke launcher requires {CONFIG}")
    training, _ = check_inputs(path)
    if training.data.split_directory.resolve() != (repository / DATASET / "splits/train").resolve():
        raise ValueError("smoke test requires the existing B0 data bundle")
    if (training.loop.max_optimizer_steps, training.loop.optimizer_batch_size) != (3, 48):
        raise ValueError("smoke test requires three optimizer steps of 48 targets")
    benchmark = training.benchmark
    if (benchmark.warmup_steps, benchmark.measured_steps, benchmark.repetitions) != (1, 2, 1):
        raise ValueError("smoke test requires one warmup step, two measured steps and one repetition")
    if training.flux_packing is None or training.flux_packing.max_tokens != 12_288:
        raise ValueError("smoke test requires packed FA3 with a 12288-token budget")
    if training.optimizer.learning_rate <= 0:
        raise ValueError("smoke test requires actual updates with a positive configured learning rate")
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.resolve().relative_to(checkpoint_root.resolve())
        if require_checkpoints and not checkpoint.is_dir():
            raise FileNotFoundError(f"cached checkpoint directory does not exist: {checkpoint}")
    return path


def _identity(repository):
    """Hash the mounted source, exact config and complete prepared B0 bundle."""
    def hashes(paths):
        return {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(paths)}

    sources = [path for directory in ("src", "scripts")
               for path in (repository / directory).rglob("*.py")]
    sources.append(repository / "pyproject.toml")
    data = [path for path in (repository / DATASET).rglob("*") if path.is_file()]
    return {"source_sha256": hashes(sources), "config_sha256": hashes([repository / CONFIG]),
            "dataset_sha256": hashes(data)}


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


WORKER_CODE = """import signal
import sys
from pathlib import Path
from scripts.benchmark_single_gpu import benchmark_once

def interrupted(signum, frame):
    raise TimeoutError("training smoke worker interrupted at its deadline")

signal.signal(signal.SIGTERM, interrupted)
benchmark_once(Path(sys.argv[1]), Path(sys.argv[2]))
"""


def _run_worker(repository, path, destination):
    """Allow finally-based partial saves before enforcing the hard worker bound."""
    command = [sys.executable, "-B", "-u", "-c", WORKER_CODE, str(path), str(destination / "repetition-1")]
    with (destination / "worker.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=MAX_WORKER_SECONDS - 10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            raise TimeoutError("training smoke worker exceeded its runtime budget") from None
        if code:
            raise RuntimeError(f"training smoke worker exited with code {code}; see worker.log")


async def _commit_final():
    await asyncio.wait_for(model_volume.commit.aio(), timeout=FINAL_COMMIT_SECONDS)


@app.function(image=image, **RESOURCES, max_containers=1, min_containers=0,
              buffer_containers=0, scaledown_window=2, retries=0, nonpreemptible=False,
              env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
              volumes={str(VOLUME_ROOT): model_volume})
def benchmark_flux_packing(config: str, metadata: dict) -> str:
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    if metadata.get("image_id") != CACHED_IMAGE_ID:
        raise ValueError("launch image must match the cached image")
    run_id = metadata.get("run_id", "")
    if not isinstance(run_id, str) or re.fullmatch(r"\d{8}T\d{12}Z", run_id) is None:
        raise ValueError("run_id must be the launch-generated UTC timestamp")
    for name, hashes in _identity(REMOTE_REPOSITORY_ROOT).items():
        if metadata.get(name) != hashes:
            raise ValueError(f"remote {name} differs from the submitted content")
    link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if link.is_symlink():
        if link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {link}")
    elif link.exists():
        raise RuntimeError(f"checkpoint path already exists: {link}")
    else:
        link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path = _validate_layout(REMOTE_REPOSITORY_ROOT, config,
        checkpoint_root=VOLUME_ROOT, require_checkpoints=True)
    destination = VOLUME_ROOT / RESULTS / run_id
    if destination.exists():
        raise RuntimeError(f"run directory already exists; refusing to repeat training: {destination}")
    destination.mkdir(parents=True)
    manifest = {"status": "running", "metadata": metadata, "modal_resources": RESOURCES, "budget": BUDGET}
    _write_json(destination / "manifest.json", manifest)
    started = time.monotonic()
    try:
        _run_worker(REMOTE_REPOSITORY_ROOT, path, destination)
        summary = json.loads((destination / "repetition-1/summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("measured_steps") != 2:
            raise RuntimeError("worker did not complete the two measured optimizer steps")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["worker_wall_seconds"] = time.monotonic() - started
        _write_json(destination / "manifest.json", manifest)
        print(f"Saving training smoke results: {destination}", flush=True)
        asyncio.run(_commit_final())
    return str(destination)


async def _stop_app(app_id):
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "modal", "app", "stop",
        app_id, "--env", ENVIRONMENT, "--yes", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=15)
    except BaseException:
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=5)
        raise
    if process.returncode:
        raise RuntimeError(f"could not stop Modal app {app_id}: {output.decode(errors='replace')}")


async def _bounded_remote_call(config, metadata):
    call = None
    deadline = time.monotonic() + LIFECYCLE_SECONDS - CLEANUP_SECONDS

    async def invoke():
        nonlocal call
        call = await benchmark_flux_packing.spawn.aio(config, metadata)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("training smoke lifecycle deadline elapsed during submission")
        return await call.get.aio(timeout=remaining)

    try:
        return await asyncio.wait_for(invoke(), timeout=LIFECYCLE_SECONDS - CLEANUP_SECONDS)
    except BaseException:
        if call is not None:
            try:
                await asyncio.wait_for(call.cancel.aio(terminate_containers=True), timeout=10)
            except BaseException as error:
                print(f"Call cancellation failed; stopping this app: {error}", flush=True)
        try:
            await _stop_app(metadata["modal_app_id"])
        except BaseException as error:
            print(f"App stop failed: {error}; provider startup/function limits remain active", flush=True)
        raise


@app.local_entrypoint()
def main(config: str = CONFIG):
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    path = _validate_layout(LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False)
    if image.object_id != CACHED_IMAGE_ID:
        raise RuntimeError("source mounts unexpectedly changed the cached image ID")
    metadata = {**_identity(LOCAL_REPOSITORY_ROOT), "image_id": CACHED_IMAGE_ID,
        "modal_version": modal.__version__, "modal_app_id": app.app_id,
        "modal_environment": ENVIRONMENT, "command": sys.argv,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"), "budget": BUDGET}
    print(f"Submitting one training smoke run: {metadata['run_id']}; maximum resource list-price estimate "
          f"${BUDGET['maximum_list_price_estimate_usd']:.4f}", flush=True)
    result = asyncio.run(_bounded_remote_call(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata))
    print(f"Training smoke complete: {result}")
