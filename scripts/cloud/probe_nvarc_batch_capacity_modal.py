#!/usr/bin/env python3
"""Run the complete NVARC capacity search in one detached Modal H200 Function."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
)
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 30 * 60}

app = modal.App("diffusion-arc-nvarc-capacity")
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


def _validate_options(sequence_length: int, max_batch_size: int, repetitions: int) -> None:
    if isinstance(sequence_length, bool) or sequence_length != 8400:
        raise ValueError("sequence_length must be 8400")
    for name, value in (("max_batch_size", max_batch_size), ("repetitions", repetitions)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


@app.function(
    image=image, **RESOURCES, max_containers=1, retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def probe_nvarc_batch_capacity(
    config: str, sequence_length: int, max_batch_size: int, repetitions: int, metadata: dict,
) -> str:
    """Persist each completed candidate and propagate failed searches to Modal."""

    _validate_options(sequence_length, max_batch_size, repetitions)
    remote_config = (REMOTE_REPOSITORY_ROOT / config).resolve()
    remote_config.relative_to(REMOTE_REPOSITORY_ROOT / "configs")
    if not remote_config.is_file():
        raise FileNotFoundError(f"probe config does not exist: {remote_config}")
    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
    elif checkpoint_link.exists():
        raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)

    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_batch_capacity import _load_config, run_search

    config_values = _load_config(remote_config)
    for checkpoint in (config_values.models.nvarc_checkpoint, config_values.models.flux_checkpoint):
        checkpoint.relative_to(VOLUME_ROOT.resolve())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/nvarc-capacity" / run_id
    args = SimpleNamespace(
        config=remote_config, output_dir=destination, sequence_length=sequence_length,
        max_batch_size=max_batch_size, repetitions=repetitions,
    )
    try:
        summary = run_search(
            args, metadata={**metadata, "modal_resources": RESOURCES},
            on_candidate_end=model_volume.commit,
        )
        if summary["status"] != "complete":
            raise RuntimeError(f"NVARC capacity search failed; inspect {destination}")
    finally:
        model_volume.commit()
        print(f"NVARC capacity results directory: {destination}")
    return str(destination)


@app.local_entrypoint()
def main(
    config: str, sequence_length: int = 8400, max_batch_size: int = 64, repetitions: int = 3,
) -> None:
    """Validate the upload and record local provenance before requesting a GPU."""

    _validate_options(sequence_length, max_batch_size, repetitions)
    config_path = (LOCAL_REPOSITORY_ROOT / config).resolve()
    config_path.relative_to(LOCAL_REPOSITORY_ROOT / "configs")
    relative_config = config_path.relative_to(LOCAL_REPOSITORY_ROOT)
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_batch_capacity import _load_config, _source_metadata

    config_values = _load_config(config_path, require_checkpoints=False)
    for checkpoint in (config_values.models.nvarc_checkpoint, config_values.models.flux_checkpoint):
        checkpoint.relative_to((LOCAL_REPOSITORY_ROOT / "checkpoints").resolve())
    metadata = {
        **_source_metadata(), "command": sys.argv,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "modal_version": modal.__version__, "image_id": image.object_id,
    }
    result = probe_nvarc_batch_capacity.remote(
        str(relative_config), sequence_length, max_batch_size, repetitions, metadata,
    )
    print(f"Modal NVARC capacity probe complete: {result}")
