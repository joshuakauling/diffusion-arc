#!/usr/bin/env python3
"""Run one FA3 modulation-backward experiment within the approved one-dollar envelope."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import modal

REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
ENVIRONMENT = "diffusion-arc"
CACHED_IMAGE_ID = "im-R5CncDboPfMQryWnq0UZn4"
LIFECYCLE_SECONDS = 600
CLEANUP_SECONDS = 30
FINAL_COMMIT_SECONDS = 60
MAX_WORKER_SECONDS = 420
# Function time includes preflight and final persistence. At its deadline it may
# interrupt the final commit; startup + function + scaledown still fit the cap.
RESOURCES = {"gpu": "H200", "cpu": (8.0, 8.0), "memory": (65536, 65536),
             "timeout": 480, "startup_timeout": 90}
PRICE_USD_PER_SECOND = 0.001261 + 8 * 0.0000131 + 64 * 0.00000222
BUDGET = {"planned_ceiling_usd": 1.0,
          "resource_rate_usd_per_second": PRICE_USD_PER_SECOND,
          "lifecycle_seconds": LIFECYCLE_SECONDS,
          "maximum_list_price_estimate_usd": PRICE_USD_PER_SECOND * LIFECYCLE_SECONDS,
          "price_basis": "H200, eight requested/limited CPU cores, 64 GiB requested/limited RAM; no premium placement",
          "cleanup_reserve_seconds": CLEANUP_SECONDS, "final_commit_seconds": FINAL_COMMIT_SECONDS}

app = modal.App("diffusion-arc-flux-modulation-backward")
model_volume = modal.Volume.from_name("diffusion-arc-models", environment_name=ENVIRONMENT,
                                     create_if_missing=False)

# Mount current source onto the exact existing image; none of these calls builds a layer.
image = modal.Image.from_id(CACHED_IMAGE_ID)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / directory,
        remote_path=str(REMOTE_REPOSITORY_ROOT / directory), copy=False,
        ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(LOCAL_REPOSITORY_ROOT / "pyproject.toml",
    remote_path=str(REMOTE_REPOSITORY_ROOT / "pyproject.toml"), copy=False)
image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / "data/benchmarks/single_gpu_b0",
    remote_path=str(REMOTE_REPOSITORY_ROOT / "data/benchmarks/single_gpu_b0"), copy=False)


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints):
    from scripts.diagnostics.probe_flux_modulation_backward import load_probe_config

    repository = repository.resolve()
    config_root = (repository / "configs").resolve()
    path = (repository / config).resolve()
    path.relative_to(config_root)
    policy = load_probe_config(path, require_checkpoints=require_checkpoints)
    for source in policy.config_paths:
        source.resolve().relative_to(config_root)
    for checkpoint in policy.checkpoint_paths:
        checkpoint.resolve().relative_to(checkpoint_root.resolve())
        if require_checkpoints and not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if policy.training.data.split_directory != repository / "data/benchmarks/single_gpu_b0/splits/train":
        raise ValueError("diagnostic must use the existing B0 benchmark bundle")
    if type(policy.max_duration_seconds) is not int or not 1 <= policy.max_duration_seconds <= MAX_WORKER_SECONDS:
        raise ValueError(f"worker budget must be in [1, {MAX_WORKER_SECONDS}] seconds")
    return path, policy


async def _commit_final():
    """Bound the single explicit commit independently of the experiment worker."""
    await asyncio.wait_for(model_volume.commit.aio(), timeout=FINAL_COMMIT_SECONDS)


@app.function(image=image, **RESOURCES, max_containers=1, min_containers=0,
              buffer_containers=0, scaledown_window=2, retries=0, nonpreemptible=False,
              env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_modulation_backward(config: str, metadata: dict) -> str:
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_modulation_backward import run_probe

    if metadata.get("image_id") != CACHED_IMAGE_ID:
        raise ValueError("launch image must match the approved cached image")
    run_id = metadata.get("run_id", "")
    if not isinstance(run_id, str) or re.fullmatch(r"\d{8}T\d{12}Z", run_id) is None:
        raise ValueError("run_id must be the launch-generated UTC timestamp")
    link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if link.is_symlink():
        if link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {link}")
    elif link.exists():
        raise RuntimeError(f"checkpoint path already exists: {link}")
    else:
        link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path, _ = _validate_layout(REMOTE_REPOSITORY_ROOT, config,
        checkpoint_root=VOLUME_ROOT, require_checkpoints=True)
    destination = VOLUME_ROOT / "results/flux-modulation-backward" / run_id
    # The same submitted call retains its ID if the provider reschedules it.
    if destination.exists():
        raise RuntimeError(f"run directory already exists; refusing to repeat the experiment: {destination}")
    print(f"FLUX modulation-backward results directory: {destination}", flush=True)
    try:
        # Local files and Modal's background persistence retain progress; never block a worker
        # progress event on a synchronous volume commit.
        result = run_probe(SimpleNamespace(config=path, output_dir=destination),
            metadata={**metadata, "modal_resources": RESOURCES, "budget": BUDGET},
            on_progress=lambda: None)
        if result["status"] != "complete":
            raise RuntimeError(f"FLUX modulation backward {result['status']} during {result.get('phase')}: "
                               f"{result.get('error')}; partial results: {destination}")
    finally:
        print(f"Saving FLUX modulation-backward results: {destination}", flush=True)
        asyncio.run(_commit_final())
    return str(destination)


async def _stop_app(app_id):
    """Use the public CLI to stop exactly this app if a call is cancelled or fails."""
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", "modal", "app", "stop",
        app_id, "--env", ENVIRONMENT, "--yes", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=20)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"could not stop Modal app {app_id}: {output.decode(errors='replace')}")


async def _bounded_remote_call(config, metadata):
    """One submission, with a local deadline covering queueing, startup and final save."""
    call = None
    deadline = time.monotonic() + LIFECYCLE_SECONDS - CLEANUP_SECONDS

    async def invoke():
        nonlocal call
        call = await probe_flux_modulation_backward.spawn.aio(config, metadata)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("modulation-backward lifecycle deadline elapsed during submission")
        return await call.get.aio(timeout=remaining)

    try:
        return await asyncio.wait_for(invoke(), timeout=LIFECYCLE_SECONDS - CLEANUP_SECONDS)
    except BaseException:
        if call is not None:
            try:
                await asyncio.wait_for(call.cancel.aio(terminate_containers=True), timeout=10)
            except BaseException as error:
                print(f"Call cancellation failed; stopping this app: {type(error).__name__}: {error}", flush=True)
        try:
            await _stop_app(metadata["modal_app_id"])
        except BaseException as error:
            print(f"App stop failed: {type(error).__name__}: {error}; provider startup/function limits remain active",
                  flush=True)
        raise


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/flux_modulation_backward.json"):
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_modulation_backward import config_hashes, dataset_hashes, source_metadata

    path, policy = _validate_layout(LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False)
    if image.object_id != CACHED_IMAGE_ID:
        raise RuntimeError("source mounts unexpectedly changed the approved image ID")
    metadata = {**source_metadata(), "config_sha256": config_hashes(path, policy),
        "dataset_sha256": dataset_hashes(policy), "command": sys.argv,
        "modal_version": modal.__version__, "image_id": CACHED_IMAGE_ID,
        "modal_app_id": app.app_id, "modal_environment": ENVIRONMENT,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"), "budget": BUDGET}
    print(f"Submitting one bounded run: {metadata['run_id']}; maximum resource list-price estimate "
          f"${BUDGET['maximum_list_price_estimate_usd']:.4f}", flush=True)
    result = asyncio.run(_bounded_remote_call(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata))
    print(f"Modal FLUX modulation-backward probe complete: {result}")
