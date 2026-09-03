#!/usr/bin/env python3
"""Run the single-GPU Torch profiler workload on one Modal H200."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2]
    if modal.is_local()
    else REMOTE_REPOSITORY_ROOT
)
VOLUME_NAME = "diffusion-arc-models"
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")

app = modal.App("diffusion-arc-torch-profile")
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(str(LOCAL_REPOSITORY_ROOT / "pyproject.toml"))
    .workdir(REMOTE_REPOSITORY_ROOT)
    .add_local_dir(
        LOCAL_REPOSITORY_ROOT / "src",
        remote_path=str(REMOTE_REPOSITORY_ROOT / "src"),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_REPOSITORY_ROOT / "scripts",
        remote_path=str(REMOTE_REPOSITORY_ROOT / "scripts"),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_REPOSITORY_ROOT / "configs",
        remote_path=str(REMOTE_REPOSITORY_ROOT / "configs"),
    )
    .add_local_dir(
        LOCAL_REPOSITORY_ROOT / "data" / "splits",
        remote_path=str(REMOTE_REPOSITORY_ROOT / "data" / "splits"),
    )
    .add_local_dir(
        LOCAL_REPOSITORY_ROOT / "data" / "raw" / "arc-agi-2" / "training",
        remote_path=str(
            REMOTE_REPOSITORY_ROOT / "data" / "raw" / "arc-agi-2" / "training"
        ),
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,
    memory=64 * 1024,
    timeout=15 * 60,
    max_containers=1,
    retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def profile_single_gpu(config: str) -> str:
    """Run one profiler config and persist its trace on the mounted Volume."""

    import sys

    remote_config = (REMOTE_REPOSITORY_ROOT / config).resolve()
    try:
        remote_config.relative_to(REMOTE_REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("config must be a repository-relative path") from error
    if not remote_config.is_file():
        raise FileNotFoundError(f"training config does not exist: {remote_config}")

    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT:
            raise RuntimeError(
                f"unexpected checkpoint symlink target: {checkpoint_link.resolve()}"
            )
    elif checkpoint_link.exists():
        raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)

    if str(REMOTE_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))

    from scripts import train_single_gpu

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    profiler_output_root = VOLUME_ROOT / "results" / "torch-profiler" / run_id
    train_single_gpu.PROFILER_OUTPUT_ROOT = profiler_output_root
    train_single_gpu.train_from_config(remote_config)

    trace_path = profiler_output_root / remote_config.stem / "trace.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"profiler trace was not created: {trace_path}")
    model_volume.commit()

    print(f"Persistent profiler trace: {trace_path}")
    return str(trace_path)


@app.local_entrypoint()
def main(config: str) -> None:
    """Start the remote profiling Function from ``modal run``."""

    trace_path = profile_single_gpu.remote(config)
    print(f"Modal profiler run complete: {trace_path}")
