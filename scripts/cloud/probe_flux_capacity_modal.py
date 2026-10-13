#!/usr/bin/env python3
"""Run a bounded synthetic FLUX/FSDP2 capacity search on 2, 4 or 8 H200s."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import modal


REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_NAME = os.environ.get("DIFFUSION_ARC_MODAL_VOLUME", "diffusion-arc-training-v2")
VOLUME_ROOT = Path(os.environ.get("DIFFUSION_ARC_MODAL_VOLUME_ROOT", "/workspace"))
MODEL_SUBDIRECTORY = Path(os.environ.get("DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY", "models"))
if (not VOLUME_NAME.strip() or not VOLUME_ROOT.is_absolute() or ".." in VOLUME_ROOT.parts
        or MODEL_SUBDIRECTORY.is_absolute() or ".." in MODEL_SUBDIRECTORY.parts):
    raise ValueError("Modal storage requires an existing volume, absolute mount and contained model directory")
CONFIG = "configs/training/benchmarks/fsdp2_training.json"


def resources(gpus: int, time_limit_seconds: int = 600) -> dict:
    """Retain the per-GPU CPU/RAM ratio and bound the search plus cleanup."""
    if type(gpus) is not int or gpus not in (2, 4, 8):
        raise ValueError("--gpus must be 2, 4 or 8")
    if type(time_limit_seconds) is not int or not 30 <= time_limit_seconds <= 1800:
        raise ValueError("--time-limit-seconds must be an integer between 30 and 1800")
    return dict(gpu=f"H200:{gpus}", cpu=(8.0 * gpus,) * 2,
                memory=(64 * 1024 * gpus,) * 2, timeout=time_limit_seconds + 90)


RESOURCES = resources(2)

sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
from scripts.cloud._training_image import FA3_REVISION, training_image

app = modal.App("diffusion-arc-flux-capacity")
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = training_image(LOCAL_REPOSITORY_ROOT).env({
    "DIFFUSION_ARC_MODAL_VOLUME": VOLUME_NAME,
    "DIFFUSION_ARC_MODAL_VOLUME_ROOT": str(VOLUME_ROOT),
    "DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY": MODEL_SUBDIRECTORY.as_posix(),
}).workdir(REMOTE_REPOSITORY_ROOT)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(
        LOCAL_REPOSITORY_ROOT / directory, remote_path=str(REMOTE_REPOSITORY_ROOT / directory),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )


@app.function(
    image=image, **RESOURCES, max_containers=1, retries=0, scaledown_window=2,
    volumes={str(VOLUME_ROOT): model_volume},
)
def probe_flux_capacity(
    config: str, start_tokens: int, step_tokens: int, max_tokens: int, updates: int, metadata: dict,
    reshard_after_forward: bool = True,
    gpus: int = 2, time_limit_seconds: int = 600,
) -> str:
    """Persist completed candidates and retain legitimate time-limited partial results."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = VOLUME_ROOT / "results/flux-capacity" / run_id
    try:
        sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
        from scripts.diagnostics.probe_flux_capacity import load_probe_config, run_search, validate_options

        validate_options(start_tokens, step_tokens, max_tokens, updates, reshard_after_forward)
        allocation = resources(gpus, time_limit_seconds)
        remote_config = (REMOTE_REPOSITORY_ROOT / config).resolve()
        remote_config.relative_to(REMOTE_REPOSITORY_ROOT / "configs")
        checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
        model_root = VOLUME_ROOT / MODEL_SUBDIRECTORY
        if checkpoint_link.is_symlink():
            if checkpoint_link.resolve() != model_root.resolve():
                raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
        elif checkpoint_link.exists():
            raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
        else:
            checkpoint_link.symlink_to(model_root, target_is_directory=True)
        training = load_probe_config(remote_config, require_checkpoints=True)
        if training.recipe["deployment"]["gpu_count"] != gpus:
            raise ValueError("--gpus must match the configuration's execution.gpu_count")
        for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
            checkpoint.relative_to(VOLUME_ROOT.resolve())
        args = SimpleNamespace(config=remote_config, output_dir=destination, start_tokens=start_tokens,
                               step_tokens=step_tokens, max_tokens=max_tokens, updates=updates,
                               reshard_after_forward=reshard_after_forward)
        summary = run_search(
            args, metadata={**metadata, "modal_resources": allocation, "fa3_revision": FA3_REVISION,
                            "model_volume": dict(name=VOLUME_NAME, mount_path=str(VOLUME_ROOT),
                                                 model_subdirectory=MODEL_SUBDIRECTORY.as_posix())},
            on_candidate_end=model_volume.commit, timeout_seconds=time_limit_seconds,
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
    gpus: int = 2, time_limit_seconds: int = 600,
) -> None:
    """Validate configuration without datasets or local model weights before allocation."""
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_capacity import load_probe_config, source_metadata, validate_options

    validate_options(start_tokens, step_tokens, max_tokens, updates, reshard_after_forward)
    allocation = resources(gpus, time_limit_seconds)
    config_path = (LOCAL_REPOSITORY_ROOT / config).resolve()
    config_path.relative_to(LOCAL_REPOSITORY_ROOT / "configs")
    training = load_probe_config(config_path, require_checkpoints=False)
    if training.recipe["deployment"]["gpu_count"] != gpus:
        raise ValueError("--gpus must match the configuration's execution.gpu_count")
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.relative_to((LOCAL_REPOSITORY_ROOT / "checkpoints").resolve())
    metadata = {
        **source_metadata(), "command": sys.argv,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "modal_version": modal.__version__, "image_id": image.object_id,
    }
    result = probe_flux_capacity.with_options(**allocation).remote(
        str(config_path.relative_to(LOCAL_REPOSITORY_ROOT)), start_tokens, step_tokens, max_tokens, updates, metadata,
        reshard_after_forward, gpus, time_limit_seconds,
    )
    print(f"Modal FLUX capacity probe complete: {result}")
