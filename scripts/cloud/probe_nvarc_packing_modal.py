#!/usr/bin/env python3
"""Run the standalone NVARC packing comparison on one H200."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import modal

REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 15 * 60}
FLASH_ATTN_VERSION = "2.8.3.post1"
FA3_REVISION = "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820"
FA4_VERSION = "4.0.0b29"
CUDNN_FRONTEND_VERSION = "1.13.0"
DATA_ROOT = Path("benchmarks/single_gpu_b0")
CUDA_IMAGE = (
    "nvidia/cuda:13.0.2-devel-ubuntu24.04"
    "@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)

app = modal.App("diffusion-arc-nvarc-packing")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)

# Compile against the pinned torch/CUDA ABI during the CPU image build, never
# inside the paid H200 function. Force compilation to avoid a mismatched wheel.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install_from_pyproject(str(LOCAL_REPOSITORY_ROOT / "pyproject.toml"))
    .pip_install("ninja==1.11.1.4", "packaging==25.0", "setuptools==80.9.0", "wheel==0.45.1")
    # Override Python's compiler defaults for setuptools and NVCC's host code.
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++",
          "FLASH_ATTENTION_FORCE_BUILD": "TRUE", "FLASH_ATTN_CUDA_ARCHS": "90",
          "MAX_JOBS": "4", "NVCC_THREADS": "2"})
    .run_commands(
        "/usr/bin/gcc --version",
        "/usr/bin/g++ --version",
        "python -c \"import torch; assert torch.__version__.split('+')[0] == '2.13.0'; "
        "assert torch.version.cuda == '13.0', torch.version.cuda\"",
        f"python -m pip install --no-build-isolation --no-deps flash-attn=={FLASH_ATTN_VERSION}",
        "python -c \"import torch, flash_attn, flash_attn_2_cuda; "
        f"assert flash_attn.__version__ == '{FLASH_ATTN_VERSION}'\"",
    )
    # NVARC needs BF16 forward with head_dim=128. Keep GQA, causal and Varlen
    # support; omit unused dtypes, dimensions and backward from the CPU build.
    .env({f"FLASH_ATTENTION_DISABLE_{feature}": "TRUE" for feature in (
        "BACKWARD", "FP16", "FP8", "HDIM64", "HDIM96", "HDIM192", "HDIM256",
        "HDIMDIFF64", "HDIMDIFF192", "SM80", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
    )})
    .run_commands(
        "python -m pip install --no-build-isolation --no-deps "
        f"'git+https://github.com/Dao-AILab/flash-attention.git@{FA3_REVISION}#subdirectory=hopper'",
        "python -c \"import torch, flash_attn_3._C; "
        "from flash_attn_3.flash_attn_interface import flash_attn_varlen_func\"",
    )
    .pip_install(
        "torch==2.13.0", f"flash-attn-4[cu13]=={FA4_VERSION}",
        "nvidia-cutlass-dsl[cu13]==4.6.2", "quack-kernels==0.6.4",
        "apache-tvm-ffi==0.1.13.post3", "torch-c-dlpack-ext==0.1.5",
        f"nvidia-cudnn-frontend=={CUDNN_FRONTEND_VERSION}",
    )
    .run_commands(
        "python -m pip check",
        "python -c \"import torch, cudnn; "
        "from flash_attn.flash_attn_interface import flash_attn_varlen_func as fa2; "
        "from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3; "
        "from flash_attn.cute.interface import flash_attn_varlen_func as fa4; "
        "assert torch.__version__.split('+')[0] == '2.13.0'; "
        "assert torch.version.cuda == '13.0'; "
        "assert cudnn.backend_version() == torch.backends.cudnn.version()\"",
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .workdir(REMOTE_REPOSITORY_ROOT)
)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(
        LOCAL_REPOSITORY_ROOT / directory, remote_path=str(REMOTE_REPOSITORY_ROOT / directory),
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
image = image.add_local_file(
    LOCAL_REPOSITORY_ROOT / "pyproject.toml", remote_path=str(REMOTE_REPOSITORY_ROOT / "pyproject.toml"),
)


def _ignore_data(path: Path) -> bool:
    return not (path.is_relative_to(DATA_ROOT) or DATA_ROOT.is_relative_to(path))


image = image.add_local_dir(
    LOCAL_REPOSITORY_ROOT / "data", remote_path=str(REMOTE_REPOSITORY_ROOT / "data"),
    ignore=_ignore_data,
)


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoint):
    from scripts.diagnostics.probe_nvarc_packing import load_probe_config

    path = (repository / config).resolve()
    path.relative_to(repository / "configs")
    policy, training = load_probe_config(path, require_checkpoint=require_checkpoint)
    policy.training_config.relative_to(repository / "configs")
    training.models.nvarc_checkpoint.relative_to(checkpoint_root.resolve())
    if training.data.split_directory != repository / "data" / DATA_ROOT / "splits/train":
        raise ValueError("this launcher uploads only the existing B0 benchmark dataset")
    if policy.max_duration_seconds > RESOURCES["timeout"] - 60:
        raise ValueError("worker budget must leave at least 60 seconds before the Modal timeout")
    return path, policy


@app.function(
    image=image, **RESOURCES, max_containers=1, retries=0,
    volumes={str(VOLUME_ROOT): model_volume},
)
def probe_nvarc_packing(config: str, metadata: dict) -> str:
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_packing import run_probe

    checkpoint_link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if checkpoint_link.is_symlink():
        if checkpoint_link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {checkpoint_link}")
    elif checkpoint_link.exists():
        raise RuntimeError(f"checkpoint path already exists: {checkpoint_link}")
    else:
        checkpoint_link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path, _ = _validate_layout(
        REMOTE_REPOSITORY_ROOT, config, checkpoint_root=VOLUME_ROOT, require_checkpoint=True,
    )
    destination = VOLUME_ROOT / "results/nvarc-packing" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ",
    )
    try:
        result = run_probe(
            SimpleNamespace(config=path, output_dir=destination), on_progress=model_volume.commit,
            metadata={**metadata, "modal_resources": RESOURCES, "cuda_image": CUDA_IMAGE,
                      "flash_attn_build_version": FLASH_ATTN_VERSION,
                      "fa3_revision": FA3_REVISION, "fa4_version": FA4_VERSION,
                      "cudnn_frontend_version": CUDNN_FRONTEND_VERSION},
        )
        if result["status"] != "complete":
            raise RuntimeError(
                f"NVARC packing {result['status']} during {result.get('phase')}: "
                f"{result.get('error', 'see worker.log')}; partial results: {destination}"
            )
    finally:
        print(f"NVARC packing results directory: {destination}")
        model_volume.commit()
    return str(destination)


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/nvarc_packing.json") -> None:
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_nvarc_packing import (
        config_hashes, dataset_hashes, load_probe_config, source_metadata,
    )

    path, policy = _validate_layout(
        LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoint=False,
    )
    metadata = {
        **source_metadata(), "config_sha256": config_hashes(path, policy),
        "command": sys.argv, "modal_version": modal.__version__, "image_id": image.object_id,
        "dataset_sha256": dataset_hashes(load_probe_config(path, require_checkpoint=False)[1]),
    }
    result = probe_nvarc_packing.remote(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata)
    print(f"Modal NVARC packing probe complete: {result}")
