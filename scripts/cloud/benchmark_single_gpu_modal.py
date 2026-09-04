#!/usr/bin/env python3
"""Run three sequential benchmark repetitions in one detached Modal Function."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
)
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
DATA_ROOT = Path("benchmarks/single_gpu_b0")
RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 2 * 60 * 60}


def _ignore_data(path: Path) -> bool:
    """Upload only the future B0 data bundle, not archives or overfitting data."""

    return not (path.is_relative_to(DATA_ROOT) or DATA_ROOT.is_relative_to(path))


app = modal.App("diffusion-arc-benchmark")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject(str(LOCAL_REPOSITORY_ROOT / "pyproject.toml"))
    .workdir(REMOTE_REPOSITORY_ROOT)
)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(
        LOCAL_REPOSITORY_ROOT / directory,
        remote_path=str(REMOTE_REPOSITORY_ROOT / directory),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
image = image.add_local_dir(
    LOCAL_REPOSITORY_ROOT / "data",
    remote_path=str(REMOTE_REPOSITORY_ROOT / "data"),
    ignore=_ignore_data,
)


@app.function(
    image=image,
    **RESOURCES,
    max_containers=1,
    retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def benchmark_single_gpu(config: str, metadata: dict) -> str:
    """Keep all repetitions inside one Function so detach covers the full run."""

    remote_config = (REMOTE_REPOSITORY_ROOT / config).resolve()
    remote_config.relative_to(REMOTE_REPOSITORY_ROOT)
    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT:
            raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
    elif checkpoint_link.exists():
        raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)

    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.benchmark_single_gpu import run_repetitions

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/benchmarks" / run_id
    try:
        run_repetitions(
            remote_config, destination,
            metadata={**metadata, "modal_resources": RESOURCES},
            on_repetition_end=model_volume.commit,
        )
    finally:
        model_volume.commit()
        print(f"Persistent benchmark results: {destination}")
    return str(destination)


@app.local_entrypoint()
def main(config: str) -> None:
    """Refuse a missing data bundle before invoking the GPU Function."""

    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.benchmark_single_gpu import check_inputs, source_revision

    config_path = (LOCAL_REPOSITORY_ROOT / config).resolve()
    relative_config = config_path.relative_to(LOCAL_REPOSITORY_ROOT)
    training_config, _ = check_inputs(config_path)
    expected_split = LOCAL_REPOSITORY_ROOT / "data" / DATA_ROOT / "splits/train"
    if training_config.data.split_directory != expected_split:
        raise ValueError(f"This launcher uploads only the B0 bundle at {expected_split}")
    metadata = {
        **source_revision(),
        "command": sys.argv,
        "modal_version": modal.__version__,
        "image_id": image.object_id,
    }
    result = benchmark_single_gpu.remote(str(relative_config), metadata)
    print(f"Modal benchmark complete: {result}")
