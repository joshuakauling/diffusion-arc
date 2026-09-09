#!/usr/bin/env python3
"""Launch the bounded input-distribution attention diagnostic on one H200."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import modal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/diffusion-arc")
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cloud.probe_flux_packing_modal import (
    LOCAL_REPOSITORY_ROOT,
    REMOTE_REPOSITORY_ROOT,
    VOLUME_ROOT,
    image,
    model_volume,
)

RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 1920}
app = modal.App("diffusion-arc-flux-backend-conditions")


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints):
    from scripts.diagnostics.probe_flux_backend_conditions import load_probe_config

    config_root = (repository / "configs").resolve()
    path = (repository / config).resolve()
    path.relative_to(config_root)
    policy = load_probe_config(path, require_checkpoints=require_checkpoints)
    # The existing image uploads configs, including the compact input fixture.
    for source in (*policy.config_paths, *policy.fixture_paths):
        source.resolve().relative_to(config_root)
    for checkpoint in policy.checkpoint_paths:
        checkpoint.resolve().relative_to(checkpoint_root.resolve())
        if require_checkpoints and not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if (type(policy.max_duration_seconds) is not int
            or not 1 <= policy.max_duration_seconds <= RESOURCES["timeout"] - 120):
        raise ValueError("worker budget must be positive and leave 120 seconds before the Modal timeout")
    return path, policy


@app.function(image=image, **RESOURCES, max_containers=1, retries=0,
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_backend_conditions(config: str, metadata: dict) -> str:
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_backend_conditions import run_probe

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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/flux-backend-conditions" / run_id
    try:
        result = run_probe(SimpleNamespace(config=path, output_dir=destination),
            metadata={**metadata, "modal_resources": RESOURCES},
            on_progress=model_volume.commit)
        if result["status"] != "complete":
            raise RuntimeError(f"FLUX backend conditions {result['status']} during {result.get('phase')}: "
                               f"{result.get('error')}; partial results: {destination}")
    finally:
        model_volume.commit()
        print(f"FLUX backend conditions results directory: {destination}")
    return str(destination)


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/flux_backend_conditions.json"):
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_backend_conditions import (
        config_hashes, dataset_hashes, source_metadata,
    )

    path, policy = _validate_layout(LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False)
    metadata = {**source_metadata(), "config_sha256": config_hashes(path, policy),
                "dataset_sha256": dataset_hashes(policy), "command": sys.argv,
                "modal_version": modal.__version__, "image_id": image.object_id}
    result = probe_flux_backend_conditions.remote(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata)
    print(f"Modal FLUX backend conditions probe complete: {result}")
