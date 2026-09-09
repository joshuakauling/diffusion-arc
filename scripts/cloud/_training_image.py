"""Build the pinned Hopper/BF16 FA3 runtime before allocating a training GPU."""

from pathlib import Path

FA3_REVISION = "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820"
CUDA_IMAGE = (
    "nvidia/cuda:13.0.2-devel-ubuntu24.04"
    "@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
)


def training_image(repository: Path):
    """Compile BF16/head-128 FA3 for frozen NVARC and trainable packed FLUX."""
    import modal

    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
        .entrypoint([])
        .apt_install("build-essential", "git")
        .pip_install_from_pyproject(str(repository / "pyproject.toml"))
        .pip_install("ninja==1.11.1.4", "packaging==25.0", "setuptools==80.9.0", "wheel==0.45.1")
        .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++",
              "FLASH_ATTENTION_FORCE_BUILD": "TRUE", "FLASH_ATTN_CUDA_ARCHS": "90",
              "FLASH_ATTENTION_DISABLE_BACKWARD": "FALSE",
              "FLASH_ATTENTION_DISABLE_VARLEN": "FALSE",
              "FLASH_ATTENTION_DISABLE_HDIM128": "FALSE",
              "MAX_JOBS": "4", "NVCC_THREADS": "2"})
        .env({f"FLASH_ATTENTION_DISABLE_{feature}": "TRUE" for feature in (
            "FP16", "FP8", "HDIM64", "HDIM96", "HDIM192", "HDIM256",
            "HDIMDIFF64", "HDIMDIFF192", "SM80", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
        )})
        .run_commands(
            "python -c \"import torch; assert torch.__version__.split('+')[0] == '2.13.0'; "
            "assert torch.version.cuda == '13.0'\"",
            "python -m pip install --no-build-isolation --no-deps "
            f"'git+https://github.com/Dao-AILab/flash-attention.git@{FA3_REVISION}#subdirectory=hopper'",
            "python -c \"import torch, flash_attn_3._C; "
            "from flash_attn_3.flash_attn_interface import flash_attn_varlen_func; "
            "from flash_attn_config import CONFIG; "
            "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_BACKWARD']; "
            "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_VARLEN']; "
            "assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_HDIM128']\"",
        )
        .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    )
