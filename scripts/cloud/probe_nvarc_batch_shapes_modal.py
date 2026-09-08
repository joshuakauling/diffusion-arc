#!/usr/bin/env python3
"""Run the masked NVARC alignment probe on one H200 with cached checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 30 * 60}

app = modal.App("diffusion-arc-nvarc-shapes")
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
image = image.add_local_file(
    LOCAL_REPOSITORY_ROOT / "pyproject.toml", remote_path=str(REMOTE_REPOSITORY_ROOT / "pyproject.toml"),
)


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints):
    from scripts.diagnostics.probe_nvarc_batch_shapes import load_probe_config

    path = (repository / config).resolve()
    path.relative_to(repository / "configs")
    policy, training = load_probe_config(path, require_checkpoints=require_checkpoints)
    policy.training_config.relative_to(repository / "configs")
    if policy.max_duration_seconds > RESOURCES["timeout"] - 60:
        raise ValueError("probe time budget must leave at least 60 seconds before the Modal timeout")
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.relative_to(checkpoint_root.resolve())
    return path, policy


@app.function(
    image=image, **RESOURCES, max_containers=1, retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def probe_nvarc_batch_shapes(config: str, metadata: dict) -> str:
    """Persist the sweep and propagate failures while retaining its partial artifacts."""

    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_batch_shapes import run_probe

    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
    elif checkpoint_link.exists():
        raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path, _policy = _validate_layout(
        REMOTE_REPOSITORY_ROOT, config, checkpoint_root=VOLUME_ROOT, require_checkpoints=True,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/nvarc-batch-shapes" / run_id
    try:
        result = run_probe(
            SimpleNamespace(config=path, output_dir=destination),
            metadata={**metadata, "modal_resources": RESOURCES}, on_progress=model_volume.commit,
        )
        if result["status"] != "complete":
            failed_case = next((
                f" at {group['batch_size']}x{case['padded_length']} ({case['phase']})"
                for group in result.get("groups", []) for case in group.get("cases", [])
                if case.get("status") in ("oom", "error")
            ), "")
            cause = result.get("error", "no worker error recorded; see worker.log")
            raise RuntimeError(
                f"NVARC shape probe {result['status']}{failed_case}: {cause}; "
                f"partial results: {destination}"
            )
    finally:
        print(f"NVARC shape results directory: {destination}")
        model_volume.commit()
    return str(destination)


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/nvarc_batch_shapes.json") -> None:
    """Validate local inputs and provenance before calling the GPU function."""

    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_batch_shapes import config_hashes, source_metadata

    path, policy = _validate_layout(
        LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False,
    )
    metadata = {
        **source_metadata(), "config_sha256": config_hashes(path, policy),
        "command": sys.argv, "modal_version": modal.__version__, "image_id": image.object_id,
    }
    result = probe_nvarc_batch_shapes.remote(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata)
    print(f"Modal NVARC shape probe complete: {result}")
