"""Strict local-only loading for NVARC and official BFL FLUX.2 components."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ._vendor.bfl_flux2.autoencoder import AutoEncoder, AutoEncoderParams
from ._vendor.bfl_flux2.model import Flux2, Klein4BParams


OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256 = (
    "9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92"
)
OFFICIAL_FLUX2_VAE_SHA256 = (
    "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3"
)
# Published NVARC qwen3_configs vocabulary and added-token contract.
NVARC_TOKEN_IDS = {
    **{str(value): value for value in range(10)},
    "Ċ": 10,
    "user": 11,
    "assistant": 12,
    "<|endoftext|>": 13,
    "<|im_start|>": 14,
    "<|im_end|>": 15,
}
NVARC_PROBE_PROMPT = (
    "<|im_start|>user\n0<|im_end|><|im_start|>assistant\n"
)
NVARC_PROBE_IDS = (14, 11, 10, 0, 15, 14, 12, 10)


def _require_local_path(path: str | Path, *, directory: bool) -> Path:
    resolved = Path(path).expanduser().resolve()
    predicate = resolved.is_dir if directory else resolved.is_file
    expected = "directory" if directory else "file"
    if not predicate():
        raise FileNotFoundError(f"Expected a local {expected}: {resolved}")
    return resolved


def sha256_file(path: str | Path) -> str:
    """Hash one local file without loading it into memory."""

    resolved = _require_local_path(path, directory=False)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    normalized = expected_sha256.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters.")
    actual = sha256_file(path)
    if actual != normalized:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {path}: expected {normalized}, got {actual}."
        )


def _load_safetensors_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Loading BFL weights requires the pinned safetensors dependency."
        ) from error
    return load_file(str(path), device=str(device))


def _assert_floating_parameter_dtype(
    module: nn.Module,
    expected_dtype: torch.dtype,
    *,
    name: str,
) -> None:
    actual = {
        parameter.dtype
        for parameter in module.parameters()
        if torch.is_floating_point(parameter)
    }
    if not actual:
        raise RuntimeError(f"{name} exposes no floating-point parameters.")
    if actual != {expected_dtype}:
        formatted = ", ".join(sorted(str(dtype) for dtype in actual))
        raise RuntimeError(
            f"{name} checkpoint parameter dtype must be {expected_dtype}; got {formatted}. "
            "The loader does not silently cast checkpoint weights."
        )


def load_bfl_flux2_klein_4b_local(
    weight_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    expected_sha256: str | None = OFFICIAL_FLUX2_KLEIN_BASE_4B_SHA256,
) -> Flux2:
    """Load BFL Klein Base 4B locally and require the expected checkpoint dtype."""

    path = _require_local_path(weight_path, directory=False)
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors FLUX weight file, got {path}.")
    _verify_expected_sha256(path, expected_sha256)
    target_device = torch.device(device)

    with torch.device("meta"):
        model = Flux2(Klein4BParams())
    state = _load_safetensors_state(path, target_device)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.to(device=target_device)

    if model.txt_in.in_features != 7680 or model.txt_in.out_features != 3072:
        raise RuntimeError(
            "Loaded FLUX transformer does not expose the required 7680 -> 3072 text input."
        )
    _assert_floating_parameter_dtype(model, dtype, name="FLUX")
    model.requires_grad_(True)
    model.train()
    return model


def load_bfl_autoencoder_local(
    weight_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    expected_sha256: str | None = OFFICIAL_FLUX2_VAE_SHA256,
) -> AutoEncoder:
    """Load the official BFL FLUX.2 autoencoder strictly locally and frozen."""

    path = _require_local_path(weight_path, directory=False)
    if path.suffix != ".safetensors":
        raise ValueError(f"Expected a .safetensors VAE weight file, got {path}.")
    _verify_expected_sha256(path, expected_sha256)
    target_device = torch.device(device)

    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    state = _load_safetensors_state(path, target_device)
    autoencoder.load_state_dict(state, strict=True, assign=True)
    del state
    autoencoder = autoencoder.to(device=target_device)
    _assert_floating_parameter_dtype(autoencoder, dtype, name="VAE")
    autoencoder.requires_grad_(False)
    autoencoder.eval()
    return autoencoder


def verify_nvarc_checkpoint_contract(
    model: nn.Module,
    tokenizer: Any,
    *,
    expected_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Verify the released 16-token NVARC checkpoint interface."""

    for token, expected_id in NVARC_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise RuntimeError(
                f"NVARC token {token!r} maps to {actual}, expected {expected_id}."
            )
    if getattr(tokenizer, "pad_token_id", None) != 13:
        raise RuntimeError(
            f"NVARC pad_token_id must equal 13, got "
            f"{getattr(tokenizer, 'pad_token_id', None)}."
        )
    if getattr(tokenizer, "eos_token_id", None) != 15:
        raise RuntimeError(
            f"NVARC eos_token_id must equal 15, got "
            f"{getattr(tokenizer, 'eos_token_id', None)}."
        )
    tokenizer_size = len(tokenizer)
    if tokenizer_size != 16:
        raise RuntimeError(f"NVARC tokenizer must contain 16 IDs, got {tokenizer_size}.")
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max_length is None or int(tokenizer_max_length) < 8192:
        raise RuntimeError(
            "NVARC tokenizer model_max_length must support at least 8192 tokens, "
            f"got {tokenizer_max_length}."
        )
    encoded_probe = tokenizer.encode(NVARC_PROBE_PROMPT, add_special_tokens=False)
    if isinstance(encoded_probe, torch.Tensor):
        encoded_probe = encoded_probe.tolist()
    encoded_probe = tuple(int(token_id) for token_id in encoded_probe)
    if encoded_probe != NVARC_PROBE_IDS:
        raise RuntimeError(
            "NVARC tokenizer does not reproduce the canonical ARC prompt IDs: "
            f"expected {NVARC_PROBE_IDS}, got {encoded_probe}."
        )
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if unknown_id is not None and int(unknown_id) in encoded_probe:
        raise RuntimeError("The canonical NVARC prompt unexpectedly contains an unknown token.")

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_rows = int(input_embeddings.weight.shape[0])
    output_rows = int(output_embeddings.weight.shape[0])
    if input_rows != 16 or output_rows != 16:
        raise RuntimeError(
            f"NVARC embedding/output rows must both equal 16, got {input_rows}/{output_rows}."
        )

    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if hidden_size != 2560:
        raise RuntimeError(f"NVARC hidden size must equal 2560, got {hidden_size}.")
    if num_hidden_layers is None or int(num_hidden_layers) != 36:
        raise RuntimeError(
            f"NVARC must expose exactly 36 hidden layers, got {num_hidden_layers}."
        )
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None or int(max_positions) < 8192:
        raise RuntimeError(
            "NVARC max_position_embeddings must support at least 8192 tokens, "
            f"got {max_positions}."
        )
    _assert_floating_parameter_dtype(model, expected_dtype, name="NVARC")

    return {
        "input_embedding_rows": input_rows,
        "output_embedding_rows": output_rows,
        "hidden_size": int(hidden_size),
        "num_hidden_layers": int(num_hidden_layers),
        "max_position_embeddings": int(max_positions),
        "tokenizer_size": tokenizer_size,
        "tokenizer_model_max_length": int(tokenizer_max_length),
        "probe_prompt_ids": list(encoded_probe),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
    }


def load_nvarc_local(
    model_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load NVARC from an explicit local Transformers directory without network access."""

    path = _require_local_path(model_path, directory=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Loading NVARC requires the pinned transformers dependency."
        ) from error

    target_device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=dtype,
        device_map={"": str(target_device)},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    contract = verify_nvarc_checkpoint_contract(model, tokenizer, expected_dtype=dtype)
    return model, tokenizer, contract
