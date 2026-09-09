#!/usr/bin/env python3
"""Launch the independent BFL/Diffusers forward diagnostic on one H200."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/diffusion-arc")
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cloud.probe_flux_packing_modal import (
    DATA_ROOT,
    LOCAL_REPOSITORY_ROOT,
    REMOTE_REPOSITORY_ROOT,
    VOLUME_ROOT,
    image,
    model_volume,
)

RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 600}
app = modal.App("diffusion-arc-flux-bfl")


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints):
    from scripts.diagnostics.probe_flux_bfl import load_probe_config

    path = (repository / config).resolve()
    path.relative_to(repository / "configs")
    policy, training = load_probe_config(path, require_checkpoints=require_checkpoints)
    for source in (policy.input_config, policy.training_config):
        source.resolve().relative_to(repository / "configs")
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.resolve().relative_to(checkpoint_root.resolve())
    if training.data.split_directory != repository / "data" / DATA_ROOT / "splits/train":
        raise ValueError("diagnostic must use the existing B0 benchmark bundle")
    if policy.max_duration_seconds > RESOURCES["timeout"] - 120:
        raise ValueError("worker budget must leave at least 120 seconds before the Modal timeout")
    return path, policy, training


@app.function(image=image, **RESOURCES, max_containers=1, retries=0,
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_bfl(config: str, metadata: dict) -> str:
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_bfl import run_probe

    link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if link.is_symlink():
        if link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {link}")
    elif link.exists():
        raise RuntimeError(f"checkpoint path already exists: {link}")
    else:
        link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path, _, _ = _validate_layout(REMOTE_REPOSITORY_ROOT, config,
        checkpoint_root=VOLUME_ROOT, require_checkpoints=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/flux-bfl" / run_id
    try:
        result = run_probe(SimpleNamespace(config=path, output_dir=destination),
            metadata={**metadata, "modal_resources": RESOURCES},
            on_progress=model_volume.commit)
        if result["status"] != "complete":
            raise RuntimeError(f"FLUX BFL comparison {result['status']} during {result.get('phase')}: "
                               f"{result.get('error')}; partial results: {destination}")
    finally:
        model_volume.commit()
        print(f"FLUX BFL results directory: {destination}")
    return str(destination)


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/flux_bfl.json"):
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_bfl import config_hashes, dataset_hashes, source_metadata

    path, policy, training = _validate_layout(LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False)
    metadata = {**source_metadata(), "config_sha256": config_hashes(path, policy),
                "dataset_sha256": dataset_hashes(training), "command": sys.argv,
                "modal_version": modal.__version__, "image_id": image.object_id}
    result = probe_flux_bfl.remote(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata)
    print(f"Modal FLUX BFL probe complete: {result}")
