#!/usr/bin/env python3
"""Run a synthetic FLUX/FSDP2 capacity search in one detached two-H200 function."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
CONFIG = "configs/training/benchmarks/fsdp2_training.json"
RESOURCES = dict(gpu="H200:2", cpu=(16.0, 16.0), memory=(128 * 1024, 128 * 1024), timeout=1800)

sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
from scripts.cloud._training_image import FA3_REVISION, training_image

app = modal.App("diffusion-arc-flux-capacity")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)
image = training_image(LOCAL_REPOSITORY_ROOT).workdir(REMOTE_REPOSITORY_ROOT)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(
        LOCAL_REPOSITORY_ROOT / directory, remote_path=str(REMOTE_REPOSITORY_ROOT / directory),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )


@app.function(
    image=image, **RESOURCES, max_containers=1, retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def probe_flux_capacity(
    config: str, start_tokens: int, step_tokens: int, max_tokens: int, updates: int, metadata: dict,
    reshard_after_forward: bool = True,
) -> str:
    """Persist completed candidates and retain legitimate time-limited partial results."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/flux-capacity" / run_id
    try:
        sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
        from scripts.diagnostics.probe_flux_capacity import load_probe_config, run_search, validate_options

        validate_options(start_tokens, step_tokens, max_tokens, updates, reshard_after_forward)
        remote_config = (REMOTE_REPOSITORY_ROOT / config).resolve()
        remote_config.relative_to(REMOTE_REPOSITORY_ROOT / "configs")
        checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != VOLUME_ROOT.resolve():
                raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
        elif checkpoint_link.exists():
            raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
        else:
            checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)
        training = load_probe_config(remote_config, require_checkpoints=True)
        for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
            checkpoint.relative_to(VOLUME_ROOT.resolve())
        args = SimpleNamespace(config=remote_config, output_dir=destination, start_tokens=start_tokens,
                               step_tokens=step_tokens, max_tokens=max_tokens, updates=updates,
                               reshard_after_forward=reshard_after_forward)
        summary = run_search(
            args, metadata={**metadata, "modal_resources": RESOURCES, "fa3_revision": FA3_REVISION},
            on_candidate_end=model_volume.commit, timeout_seconds=1710,
        )
        if summary["status"] == "error":
            raise RuntimeError(f"FLUX capacity search failed; inspect {destination}")
    finally:
        model_volume.commit()
        print(f"FLUX capacity results directory: {destination}", flush=True)
    return str(destination)


@app.local_entrypoint()
def main(
    config: str = CONFIG, start_tokens: int = 12288, step_tokens: int = 1024,
    max_tokens: int = 49152, updates: int = 2,
    reshard_after_forward: bool = True,
) -> None:
    """Validate configuration without datasets or local model weights before allocation."""
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_capacity import load_probe_config, source_metadata, validate_options

    validate_options(start_tokens, step_tokens, max_tokens, updates, reshard_after_forward)
    config_path = (LOCAL_REPOSITORY_ROOT / config).resolve()
    config_path.relative_to(LOCAL_REPOSITORY_ROOT / "configs")
    training = load_probe_config(config_path, require_checkpoints=False)
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.relative_to((LOCAL_REPOSITORY_ROOT / "checkpoints").resolve())
    metadata = {
        **source_metadata(), "command": sys.argv,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "modal_version": modal.__version__, "image_id": image.object_id,
    }
    result = probe_flux_capacity.remote(
        str(config_path.relative_to(LOCAL_REPOSITORY_ROOT)), start_tokens, step_tokens, max_tokens, updates, metadata,
        reshard_after_forward,
    )
    print(f"Modal FLUX capacity probe complete: {result}")
