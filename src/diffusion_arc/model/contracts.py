"""Tensor contracts shared by model execution during training and inference.

The validators inspect tensor values and are intended to run on CPU before a
batch is transferred to an accelerator. Running them on accelerator tensors can
introduce device synchronization.

The flattened token representation records logical source boundaries without
prescribing an attention implementation. A runner must split the sources or
convert them to a correctly masked padded or variable-length execution layout.
Task identities and augmentation metadata remain in a runner-side sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
import torch

from diffusion_arc._constants import (
    NVARC_CONTEXT_WIDTH,
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_MAX_TOKENS,
    NVARC_PAD_TOKEN_ID,
    NVARC_VOCAB_SIZE,
)

TARGET_IMAGE_CHANNELS = 3
TARGET_IMAGE_HEIGHT = 480
TARGET_IMAGE_WIDTH = 480
TARGET_IMAGE_SHAPE = (
    TARGET_IMAGE_CHANNELS,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_WIDTH,
)
TARGET_LATENT_CHANNELS = 128
TARGET_LATENT_HEIGHT = TARGET_IMAGE_HEIGHT // 16
TARGET_LATENT_WIDTH = TARGET_IMAGE_WIDTH // 16
TARGET_LATENT_TOKEN_COUNT = TARGET_LATENT_HEIGHT * TARGET_LATENT_WIDTH
TARGET_LATENT_SHAPE = (
    TARGET_LATENT_TOKEN_COUNT,
    TARGET_LATENT_CHANNELS,
)


@dataclass(frozen=True)
class FluxSpatialShape:
    """Explicit packed FLUX image-token geometry for one experiment.

    The primary ARC path remains the fixed ``30x30`` token grid.  Diagnostic
    experiments may pass another explicitly validated geometry without
    changing the default contract used by training and inference.
    """

    height: int
    width: int

    def __post_init__(self) -> None:
        for name, value in (("height", self.height), ("width", self.width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def token_count(self) -> int:
        """Return the number of packed image tokens."""

        return self.height * self.width

    @property
    def image_height(self) -> int:
        """Return the corresponding VAE image height in pixels."""

        return self.height * 16

    @property
    def image_width(self) -> int:
        """Return the corresponding VAE image width in pixels."""

        return self.width * 16


DEFAULT_FLUX_SPATIAL_SHAPE = FluxSpatialShape(
    height=TARGET_LATENT_HEIGHT,
    width=TARGET_LATENT_WIDTH,
)


@dataclass(frozen=True)
class NvarcBatchOutput:
    """Independent unpadded hidden-state tensors for one NVARC batch.

    ``sequence_hidden_states`` follows the input source order and contains one
    owning ``[L_s, 7680]`` tensor per complete source sequence. Different
    sequence tensors do not share storage, so their lifetimes can be managed
    independently after the physical NVARC batch has completed.
    """

    sequence_hidden_states: tuple[Tensor, ...]


@dataclass(frozen=True)
class FluxTextInputs:
    """Dynamically padded per-target text inputs for a FLUX microbatch.

    ``encoder_hidden_states`` contains exact NVARC conditioning prefixes in
    target-exposure order, right-padded to the longest prefix in the physical
    microbatch. ``text_attention_mask`` marks only real prefix positions.
    ``text_position_ids`` uses FLUX's four-axis ``(0, 0, 0, l)`` text
    coordinates.
    """

    encoder_hidden_states: Tensor
    text_position_ids: Tensor
    text_attention_mask: Tensor


def validate_packed_source_sequences(
    *,
    input_ids: Tensor,
    sequence_offsets: Tensor,
) -> tuple[int, ...]:
    """Validate packed NVARC sources and return their individual lengths."""

    _validate_index_tensor(input_ids, name="input_ids")
    if input_ids.numel() == 0:
        raise ValueError("input_ids must contain at least one source sequence")
    if torch.any((input_ids < 0) | (input_ids >= NVARC_VOCAB_SIZE)).item():
        raise ValueError(f"input_ids values must lie in [0, {NVARC_VOCAB_SIZE})")

    _validate_index_tensor(sequence_offsets, name="sequence_offsets")
    if sequence_offsets.numel() < 2:
        raise ValueError("sequence_offsets must contain at least a start and end offset")

    offsets = sequence_offsets.tolist()
    if offsets[0] != 0:
        raise ValueError("sequence_offsets must start at zero")
    if offsets[-1] != input_ids.numel():
        raise ValueError("the last sequence offset must equal the input_ids length")

    source_lengths = tuple(end - start for start, end in zip(offsets, offsets[1:]))
    if any(length <= 0 for length in source_lengths):
        raise ValueError("sequence_offsets must be strictly increasing")
    if any(length > NVARC_MAX_TOKENS for length in source_lengths):
        raise ValueError(
            f"each source sequence must contain at most {NVARC_MAX_TOKENS} tokens"
        )
    return source_lengths


def _validate_index_tensor(
    value: Tensor,
    *,
    name: str,
    expected_length: int | None = None,
) -> None:
    _require_tensor(value, name=name)
    if value.dtype != torch.int64:
        raise TypeError(f"{name} must have dtype torch.int64")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_length is not None and value.numel() != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")


def _require_tensor(value: object, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def validate_target_latents(latents: Tensor, *, name: str) -> None:
    """Validate the narrow transformer-facing target-latent interface."""

    if not isinstance(latents, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if latents.ndim != 3 or latents.shape[2] != TARGET_LATENT_CHANNELS:
        raise ValueError(
            f"{name} must have shape [K, T, {TARGET_LATENT_CHANNELS}]"
        )
    if latents.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one target exposure")
    if latents.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one image token")
    if not latents.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype")


__all__ = [
    "validate_target_latents",
    "NVARC_CONTEXT_WIDTH",
    "NVARC_HIDDEN_SIZE",
    "NVARC_HIDDEN_STATE_INDICES",
    "NVARC_MAX_TOKENS",
    "NVARC_PAD_TOKEN_ID",
    "NVARC_VOCAB_SIZE",
    "TARGET_IMAGE_CHANNELS",
    "TARGET_IMAGE_HEIGHT",
    "TARGET_IMAGE_SHAPE",
    "TARGET_IMAGE_WIDTH",
    "TARGET_LATENT_CHANNELS",
    "TARGET_LATENT_HEIGHT",
    "TARGET_LATENT_SHAPE",
    "TARGET_LATENT_TOKEN_COUNT",
    "TARGET_LATENT_WIDTH",
    "DEFAULT_FLUX_SPATIAL_SHAPE",
    "FluxSpatialShape",
    "FluxTextInputs",
    "NvarcBatchOutput",
    "validate_packed_source_sequences",
]
