"""Runtime construction for the initial offline single-GPU training process."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from diffusion_arc._constants import NVARC_TOKENIZER_MAX_TOKENS
from diffusion_arc.model.contracts import (
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_MAX_TOKENS,
    NVARC_PAD_TOKEN_ID,
    NVARC_VOCAB_SIZE,
    TARGET_LATENT_CHANNELS,
)
from diffusion_arc.model.flux_vae import FluxVaeDecoder, FluxVaeEncoder
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
from diffusion_arc.sequence_buckets import SequenceLengthBuckets

from .dataset import ArcTrainingDataset
from .single_gpu_config import SingleGpuTrainingConfig


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class SingleGpuModels:
    """Loaded production models with explicit frozen and trainable ownership."""

    tokenizer: PreTrainedTokenizerBase
    nvarc_runner: NvarcBatchRunner
    vae_decoder: FluxVaeDecoder
    vae_encoder: FluxVaeEncoder
    transformer: Flux2Transformer2DModel
    flux_compute_dtype: torch.dtype


def prepare_single_gpu_device(config: SingleGpuTrainingConfig) -> torch.device:
    """Select the configured CUDA device and verify required BF16 support."""

    if not torch.cuda.is_available():
        raise RuntimeError("single-GPU training requires an available CUDA device")

    requested_device = torch.device(config.device)
    if requested_device.index is not None:
        torch.cuda.set_device(requested_device)
    device = torch.device("cuda", torch.cuda.current_device())
    if (
        "bfloat16"
        in {
            config.models.nvarc_dtype,
            config.models.vae_dtype,
            config.models.flux_compute_dtype,
        }
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the configured CUDA device does not support bfloat16")
    return device


def seed_single_gpu_process(seed: int) -> None:
    """Seed process-level Python, CPU, and CUDA random-number generators."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_single_gpu_models(
    config: SingleGpuTrainingConfig,
    *,
    device: torch.device,
    transformer_checkpoint: str | Path | None = None,
) -> SingleGpuModels:
    """Load and validate all models from local checkpoint directories only.

    ``transformer_checkpoint`` selects a standalone FLUX transformer saved with
    ``save_pretrained`` while the configured base checkpoint continues to own
    the frozen VAE. Omitting it preserves the training-time checkpoint layout.
    """

    if not config.models.nvarc_checkpoint.is_dir():
        raise FileNotFoundError(
            f"NVARC checkpoint directory does not exist: "
            f"{config.models.nvarc_checkpoint}"
        )
    if not config.models.flux_checkpoint.is_dir():
        raise FileNotFoundError(
            f"FLUX checkpoint directory does not exist: "
            f"{config.models.flux_checkpoint}"
        )
    if transformer_checkpoint is None:
        transformer_source = config.models.flux_checkpoint
        transformer_subfolder = "transformer"
    else:
        transformer_source = Path(transformer_checkpoint)
        if not transformer_source.is_dir():
            raise FileNotFoundError(
                "standalone FLUX transformer checkpoint directory does not exist: "
                f"{transformer_source}"
            )
        transformer_subfolder = None

    nvarc_dtype = _TORCH_DTYPES[config.models.nvarc_dtype]
    vae_dtype = _TORCH_DTYPES[config.models.vae_dtype]
    flux_parameter_dtype = _TORCH_DTYPES[config.models.flux_parameter_dtype]
    flux_compute_dtype = _TORCH_DTYPES[config.models.flux_compute_dtype]
    device_map = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(
        config.models.nvarc_checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    nvarc_model = AutoModelForCausalLM.from_pretrained(
        config.models.nvarc_checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=nvarc_dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    _validate_nvarc_checkpoint(
        nvarc_model,
        tokenizer,
        expected_device=device,
        expected_dtype=nvarc_dtype,
    )
    tokenizer.model_max_length = NVARC_TOKENIZER_MAX_TOKENS
    nvarc_runner = NvarcBatchRunner(
        nvarc_model,
        length_buckets=SequenceLengthBuckets(config.padding.nvarc_bucket_boundaries),
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        config.models.flux_checkpoint,
        subfolder="vae",
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=vae_dtype,
        device_map=device_map,
    )
    _validate_parameter_placement(
        vae,
        name="FLUX VAE",
        expected_device=device,
        expected_dtype=vae_dtype,
    )
    vae_encoder = FluxVaeEncoder(vae)
    vae_decoder = FluxVaeDecoder(vae)

    transformer_load_arguments = {
        "local_files_only": True,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": flux_parameter_dtype,
        "device_map": device_map,
    }
    if transformer_subfolder is not None:
        transformer_load_arguments["subfolder"] = transformer_subfolder
    transformer = Flux2Transformer2DModel.from_pretrained(
        transformer_source,
        **transformer_load_arguments,
    )
    _validate_flux_checkpoint(transformer)
    _validate_parameter_placement(
        transformer,
        name="FLUX transformer",
        expected_device=device,
        expected_dtype=flux_parameter_dtype,
    )
    transformer.requires_grad_(True)

    if any(parameter.requires_grad for parameter in nvarc_runner.parameters()):
        raise RuntimeError("NVARC must remain frozen")
    if any(parameter.requires_grad for parameter in vae_encoder.parameters()):
        raise RuntimeError("the FLUX VAE must remain frozen")
    if any(parameter.requires_grad for parameter in vae_decoder.parameters()):
        raise RuntimeError("the FLUX VAE must remain frozen")
    if not all(parameter.requires_grad for parameter in transformer.parameters()):
        raise RuntimeError("every FLUX transformer parameter must remain trainable")

    return SingleGpuModels(
        tokenizer=tokenizer,
        nvarc_runner=nvarc_runner,
        vae_decoder=vae_decoder,
        vae_encoder=vae_encoder,
        transformer=transformer,
        flux_compute_dtype=flux_compute_dtype,
    )


def build_single_sample_loader(
    config: SingleGpuTrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> DataLoader:
    """Yield individual CPU samples in seeded sampler order.

    Optional workers prefetch ``num_workers * prefetch_factor`` sample tasks,
    including data loading and tokenization. This does not enlarge the GPU
    preparation window: the training loop groups samples only as it consumes
    them. The loader's random generator is independent of flow-noise sampling.
    """

    dataset = ArcTrainingDataset(
        split_directory=config.data.split_directory,
        view_name=config.data.view_name,
        tokenizer=tokenizer,
    )
    data_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    worker_arguments = {}
    if config.data.num_workers:
        # Models can already own CUDA contexts. Workers must start fresh and
        # return samples in sampler order, regardless of tokenization latency.
        worker_arguments = {
            "multiprocessing_context": "spawn",
            "prefetch_factor": config.data.prefetch_factor,
            "in_order": True,
        }
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        generator=data_generator,
        **worker_arguments,
    )


def build_flux_optimizer(
    config: SingleGpuTrainingConfig,
    *,
    transformer: nn.Module,
) -> AdamW:
    """Build AdamW over the complete trainable FLUX transformer only."""

    return AdamW(
        transformer.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.epsilon,
        weight_decay=config.optimizer.weight_decay,
        foreach=False,
    )


def build_flow_generator(
    config: SingleGpuTrainingConfig,
    *,
    device: torch.device,
) -> torch.Generator:
    """Build a device-local flow generator independent of DataLoader draws."""

    return torch.Generator(device=device).manual_seed(config.seed)


def build_flux_scheduler(
    config: SingleGpuTrainingConfig,
    *,
    optimizer: AdamW,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Build an optional schedule advanced once after each optimizer update."""

    if config.scheduler is None:
        return None
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=config.scheduler.total_steps,
    )


def _validate_nvarc_checkpoint(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    config = getattr(model, "config", None)
    expected_config = {
        "vocab_size": NVARC_VOCAB_SIZE,
        "hidden_size": NVARC_HIDDEN_SIZE,
    }
    for field, expected in expected_config.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise RuntimeError(
                f"NVARC config {field} must be {expected}, received {actual!r}"
            )
    attention_backend = getattr(config, "_attn_implementation", None)
    if attention_backend != "sdpa":
        raise RuntimeError(
            "NVARC attention implementation must resolve to 'sdpa', "
            f"received {attention_backend!r}"
        )

    hidden_layer_count = getattr(config, "num_hidden_layers", None)
    if hidden_layer_count != 36:
        raise RuntimeError(
            f"NVARC must expose 36 layers, received {hidden_layer_count!r}"
        )
    max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if (
        not isinstance(max_position_embeddings, int)
        or max_position_embeddings < NVARC_MAX_TOKENS
    ):
        raise RuntimeError(
            f"NVARC must support at least {NVARC_MAX_TOKENS} positions"
        )

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        raise RuntimeError("NVARC must expose get_input_embeddings()")
    embedding_weight = getattr(get_input_embeddings(), "weight", None)
    if not isinstance(embedding_weight, torch.Tensor):
        raise RuntimeError("NVARC input embeddings must expose a tensor weight")
    if embedding_weight.shape[0] != NVARC_VOCAB_SIZE:
        raise RuntimeError(
            f"NVARC input embeddings must contain {NVARC_VOCAB_SIZE} rows"
        )
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_output_embeddings):
        raise RuntimeError("NVARC must expose get_output_embeddings()")
    output_weight = getattr(get_output_embeddings(), "weight", None)
    if not isinstance(output_weight, torch.Tensor):
        raise RuntimeError("NVARC output embeddings must expose a tensor weight")
    if output_weight.shape[0] != NVARC_VOCAB_SIZE:
        raise RuntimeError(
            f"NVARC output embeddings must contain {NVARC_VOCAB_SIZE} rows"
        )

    vocabulary = tokenizer.get_vocab()
    expected_tokens = {
        **{str(value): value for value in range(10)},
        "Ċ": 10,
        "user": 11,
        "assistant": 12,
        "<|endoftext|>": NVARC_PAD_TOKEN_ID,
        "<|im_start|>": 14,
        "<|im_end|>": 15,
    }
    mismatches = {
        token: (expected, vocabulary.get(token))
        for token, expected in expected_tokens.items()
        if vocabulary.get(token) != expected
    }
    if mismatches:
        raise RuntimeError(f"NVARC tokenizer contract differs: {mismatches!r}")
    if len(tokenizer) != NVARC_VOCAB_SIZE:
        raise RuntimeError(
            f"NVARC tokenizer must contain {NVARC_VOCAB_SIZE} tokens"
        )
    if tokenizer.pad_token_id != NVARC_PAD_TOKEN_ID:
        raise RuntimeError(
            f"NVARC pad token ID must be {NVARC_PAD_TOKEN_ID}"
        )
    if tokenizer.eos_token_id != 15:
        raise RuntimeError("NVARC end token ID must be 15")
    canonical_probe = "<|im_start|>user\n0<|im_end|><|im_start|>assistant\n"
    expected_probe_ids = [14, 11, 10, 0, 15, 14, 12, 10]
    probe_ids = tokenizer.encode(canonical_probe, add_special_tokens=False)
    if probe_ids != expected_probe_ids:
        raise RuntimeError(
            "NVARC tokenizer canonical probe differs: "
            f"expected {expected_probe_ids!r}, received {probe_ids!r}"
        )

    _validate_parameter_placement(
        model,
        name="NVARC",
        expected_device=expected_device,
        expected_dtype=expected_dtype,
    )


def _validate_flux_checkpoint(transformer: Flux2Transformer2DModel) -> None:
    config = transformer.config
    expected_config = {
        "in_channels": TARGET_LATENT_CHANNELS,
        "joint_attention_dim": 7680,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
        "num_layers": 5,
        "num_single_layers": 20,
        "guidance_embeds": False,
    }
    for field, expected in expected_config.items():
        actual = getattr(config, field, None)
        if actual != expected:
            raise RuntimeError(
                f"FLUX transformer config {field} must be {expected!r}, "
                f"received {actual!r}"
            )
    if tuple(config.axes_dims_rope) != (32, 32, 32, 32):
        raise RuntimeError("FLUX transformer must use four 32-wide RoPE axes")
    if transformer.out_channels != TARGET_LATENT_CHANNELS:
        raise RuntimeError(
            f"FLUX transformer output width must be {TARGET_LATENT_CHANNELS}"
        )


def _validate_parameter_placement(
    model: nn.Module,
    *,
    name: str,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    parameter_count = 0
    devices: set[torch.device] = set()
    dtypes: set[torch.dtype] = set()
    for parameter in model.parameters():
        parameter_count += 1
        devices.add(parameter.device)
        dtypes.add(parameter.dtype)
    if parameter_count == 0:
        raise RuntimeError(f"{name} must expose parameters")
    if devices != {expected_device}:
        raise RuntimeError(
            f"{name} must be entirely on {expected_device}, "
            f"received {tuple(map(str, devices))!r}"
        )
    if dtypes != {expected_dtype}:
        raise RuntimeError(
            f"{name} must use only {expected_dtype}, "
            f"received {tuple(map(str, dtypes))!r}"
        )


__all__ = [
    "SingleGpuModels",
    "build_flow_generator",
    "build_flux_optimizer",
    "build_flux_scheduler",
    "build_single_sample_loader",
    "load_single_gpu_models",
    "prepare_single_gpu_device",
    "seed_single_gpu_process",
]
