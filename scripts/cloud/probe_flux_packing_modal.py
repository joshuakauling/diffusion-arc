#!/usr/bin/env python3
"""Launch the standalone FLUX attention/packing diagnostic on one H200."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import modal

REMOTE_REPOSITORY_ROOT = Path("/opt/diffusion-arc")
LOCAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPOSITORY_ROOT
VOLUME_ROOT = Path("/mnt/diffusion-arc-models")
RESOURCES = {"gpu": "H200", "cpu": 8.0, "memory": 64 * 1024, "timeout": 45 * 60}
CORRECTNESS_RESOURCES = {**RESOURCES, "timeout": 10 * 60}
FORWARD_RESOURCES = {**RESOURCES, "timeout": 10 * 60}
FLASH_ATTN_VERSION = "2.8.3.post1"
FA3_REVISION = "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820"
FA4_VERSION = "4.0.0b29"
DATA_ROOT = Path("benchmarks/single_gpu_b0")
CUDA_IMAGE = (
    "nvidia/cuda:13.0.2-devel-ubuntu24.04"
    "@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)

app = modal.App("diffusion-arc-flux-packing")
model_volume = modal.Volume.from_name("diffusion-arc-models", create_if_missing=False)

# CPU image builds only. Unlike the frozen NVARC image, retain BF16 backward.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install_from_pyproject(str(LOCAL_REPOSITORY_ROOT / "pyproject.toml"))
    .pip_install("ninja==1.11.1.4", "packaging==25.0", "setuptools==80.9.0", "wheel==0.45.1")
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++",
          "FLASH_ATTENTION_FORCE_BUILD": "TRUE", "FLASH_ATTN_CUDA_ARCHS": "90",
          "FLASH_ATTENTION_DISABLE_BACKWARD": "FALSE",
          "FLASH_ATTENTION_DISABLE_VARLEN": "FALSE", "FLASH_ATTENTION_DISABLE_HDIM128": "FALSE",
          "MAX_JOBS": "4", "NVCC_THREADS": "2"})
    .run_commands(
        "python -c \"import torch; assert torch.__version__.split('+')[0] == '2.13.0'; "
        "assert torch.version.cuda == '13.0'\"",
        f"python -m pip install --no-build-isolation --no-deps flash-attn=={FLASH_ATTN_VERSION}",
        "python -c \"import torch, flash_attn, flash_attn_2_cuda; "
        f"assert flash_attn.__version__ == '{FLASH_ATTN_VERSION}'\"",
    )
    .env({f"FLASH_ATTENTION_DISABLE_{feature}": "TRUE" for feature in (
        "FP16", "FP8", "HDIM64", "HDIM96", "HDIM192", "HDIM256", "HDIMDIFF64", "HDIMDIFF192",
        "SM80", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
    )})
    .run_commands(
        "python -m pip install --no-build-isolation --no-deps "
        f"'git+https://github.com/Dao-AILab/flash-attention.git@{FA3_REVISION}#subdirectory=hopper'",
        "python -c \"import torch, flash_attn_3._C; from flash_attn_config import CONFIG; "
        "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_BACKWARD']; "
        "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_VARLEN']; "
        "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_HDIM128']\"",
    )
    .pip_install(
        "torch==2.13.0", f"flash-attn-4[cu13]=={FA4_VERSION}",
        "nvidia-cutlass-dsl[cu13]==4.6.2", "quack-kernels==0.6.4",
        "apache-tvm-ffi==0.1.13.post3", "torch-c-dlpack-ext==0.1.5",
    )
    .run_commands(
        "python -m pip check",
        "python -c \"import torch; "
        "from flash_attn.flash_attn_interface import flash_attn_varlen_func as fa2; "
        "from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3; "
        "from flash_attn.cute.interface import flash_attn_varlen_func as fa4; "
        "assert torch.__version__.split('+')[0] == '2.13.0'; assert torch.version.cuda == '13.0'\"",
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    .workdir(REMOTE_REPOSITORY_ROOT)
)
for directory in ("src", "scripts", "configs"):
    image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / directory,
        remote_path=str(REMOTE_REPOSITORY_ROOT / directory), ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(LOCAL_REPOSITORY_ROOT / "pyproject.toml",
                             remote_path=str(REMOTE_REPOSITORY_ROOT / "pyproject.toml"))


def _ignore_data(path):
    return not (path.is_relative_to(DATA_ROOT) or DATA_ROOT.is_relative_to(path))


image = image.add_local_dir(LOCAL_REPOSITORY_ROOT / "data",
    remote_path=str(REMOTE_REPOSITORY_ROOT / "data"), ignore=_ignore_data)


def _mode_resources(correctness, forward_only):
    if correctness and forward_only:
        raise ValueError("--correctness and --forward-only are mutually exclusive")
    return FORWARD_RESOURCES if forward_only else CORRECTNESS_RESOURCES if correctness else RESOURCES


def _validate_layout(repository, config, *, checkpoint_root, require_checkpoints,
                     correctness=False, forward_only=False):
    from scripts.diagnostics.probe_flux_packing import load_probe_config
    resources = _mode_resources(correctness, forward_only)
    path = (repository / config).resolve()
    path.relative_to(repository / "configs")
    policy, training = load_probe_config(path, require_checkpoints=require_checkpoints)
    if forward_only:
        from scripts.diagnostics._flux_packing_forward import forward_policy
        policy = forward_policy(policy)
    elif correctness:
        from scripts.diagnostics._flux_packing_correctness import correctness_policy
        policy = correctness_policy(policy)
    policy.training_config.relative_to(repository / "configs")
    for checkpoint in (training.models.nvarc_checkpoint, training.models.flux_checkpoint):
        checkpoint.relative_to(checkpoint_root.resolve())
    if training.data.split_directory != repository / "data" / DATA_ROOT / "splits/train":
        raise ValueError("diagnostic must use the existing B0 benchmark bundle")
    if policy.max_duration_seconds > resources["timeout"] - 120:
        raise ValueError("worker budget must leave at least 120 seconds before the Modal timeout")
    return path, policy, training


def _run_remote(config: str, metadata: dict, *, correctness=False, forward_only=False) -> str:
    resources = _mode_resources(correctness, forward_only)
    sys.path.insert(0, str(REMOTE_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_packing import run_probe
    link = REMOTE_REPOSITORY_ROOT / "checkpoints"
    if link.is_symlink():
        if link.resolve() != VOLUME_ROOT.resolve():
            raise RuntimeError(f"unexpected checkpoint symlink: {link}")
    elif link.exists():
        raise RuntimeError(f"checkpoint path already exists: {link}")
    else:
        link.symlink_to(VOLUME_ROOT, target_is_directory=True)
    path, _, _ = _validate_layout(REMOTE_REPOSITORY_ROOT, config,
        checkpoint_root=VOLUME_ROOT, require_checkpoints=True,
        correctness=correctness, forward_only=forward_only)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_root = ("results/flux-packing-forward" if forward_only else
                   "results/flux-packing-correctness" if correctness else "results/flux-packing")
    destination = VOLUME_ROOT / result_root / run_id
    try:
        result = run_probe(SimpleNamespace(config=path, output_dir=destination,
                                          correctness=correctness, forward_only=forward_only),
            metadata={**metadata, "modal_resources": resources},
            on_progress=model_volume.commit)
        if result["status"] != "complete":
            raise RuntimeError(f"FLUX packing {result['status']} during {result.get('phase')}: "
                               f"{result.get('error')}; partial results: {destination}")
    finally:
        model_volume.commit()
        print(f"FLUX packing results directory: {destination}")
    return str(destination)


@app.function(image=image, **FORWARD_RESOURCES, max_containers=1, retries=0,
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_packing_forward(config: str, metadata: dict) -> str:
    return _run_remote(config, metadata, forward_only=True)


@app.function(image=image, **CORRECTNESS_RESOURCES, max_containers=1, retries=0,
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_packing_correctness(config: str, metadata: dict) -> str:
    return _run_remote(config, metadata, correctness=True)


@app.function(image=image, **RESOURCES, max_containers=1, retries=0,
              volumes={str(VOLUME_ROOT): model_volume})
def probe_flux_packing(config: str, metadata: dict) -> str:
    return _run_remote(config, metadata)


@app.local_entrypoint()
def main(config: str = "configs/diagnostics/flux_packing.json", correctness: bool = False,
         forward_only: bool = False):
    sys.path.insert(0, str(LOCAL_REPOSITORY_ROOT))
    from scripts.diagnostics.probe_flux_packing import config_hashes, dataset_hashes, source_metadata
    path, policy, training = _validate_layout(LOCAL_REPOSITORY_ROOT, config,
        checkpoint_root=LOCAL_REPOSITORY_ROOT / "checkpoints", require_checkpoints=False,
        correctness=correctness, forward_only=forward_only)
    metadata = {**source_metadata(), "config_sha256": config_hashes(path, policy),
                "dataset_sha256": dataset_hashes(training), "command": sys.argv,
                "modal_version": modal.__version__, "image_id": image.object_id}
    function = (probe_flux_packing_forward if forward_only else
                probe_flux_packing_correctness if correctness else probe_flux_packing)
    result = function.remote(str(path.relative_to(LOCAL_REPOSITORY_ROOT)), metadata)
    print(f"Modal FLUX packing probe complete: {result}")
