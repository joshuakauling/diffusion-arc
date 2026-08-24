"""Host-side tensor contracts at the boundary between ARC data and model code.

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

import torch
from torch import Tensor

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
class PackedTrainingBatch:
    """Packed source sequences and their aligned training target exposures.

    ``sequence_offsets`` delimits ``S`` unpadded source sequences inside the
    flat token tensor. Each of the ``K`` target exposures refers to one source
    and stores an end-exclusive prefix boundary relative to that source. The
    final prefix may equal its source length when the unused terminal target
    output has been omitted.
    """

    input_ids: Tensor
    sequence_offsets: Tensor
    target_sequence_indices: Tensor
    target_prefix_ends: Tensor
    target_images: Tensor


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
class FlowMatchingBatch:
    """One independently sampled straight-path state per target exposure.

    ``noisy_latents`` contains the interpolated transformer input ``z_t``.
    ``timesteps`` contains one normalized flow time per target exposure.
    ``target_velocity`` is the straight-path supervision ``epsilon - z_0``.
    """

    noisy_latents: Tensor
    timesteps: Tensor
    target_velocity: Tensor


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


def validate_packed_training_batch(batch: PackedTrainingBatch) -> None:
    """Raise when a packed training batch violates the public tensor contract."""

    source_lengths = validate_packed_source_sequences(
        input_ids=batch.input_ids,
        sequence_offsets=batch.sequence_offsets,
    )
    target_count = _validate_target_images(batch.target_images)

    _validate_index_tensor(
        batch.target_sequence_indices,
        name="target_sequence_indices",
        expected_length=target_count,
    )
    _validate_index_tensor(
        batch.target_prefix_ends,
        name="target_prefix_ends",
        expected_length=target_count,
    )

    target_sequence_indices = batch.target_sequence_indices.tolist()
    target_prefix_ends = batch.target_prefix_ends.tolist()
    source_count = len(source_lengths)

    previous_source_index = -1
    previous_prefix_end = 0
    seen_source_indices: set[int] = set()

    for target_index, (source_index, prefix_end) in enumerate(
        zip(target_sequence_indices, target_prefix_ends, strict=True)
    ):
        if not 0 <= source_index < source_count:
            raise ValueError(
                "target_sequence_indices contains an out-of-range source index "
                f"at target {target_index}"
            )
        if source_index < previous_source_index:
            raise ValueError("target_sequence_indices must be ordered by source")

        if source_index != previous_source_index:
            previous_prefix_end = 0

        if prefix_end <= previous_prefix_end:
            raise ValueError(
                "target_prefix_ends must be positive and strictly increasing "
                "within each source"
            )
        if prefix_end > source_lengths[source_index]:
            raise ValueError(
                "target_prefix_ends must not lie beyond their source sequence"
            )

        seen_source_indices.add(source_index)
        previous_source_index = source_index
        previous_prefix_end = prefix_end

    if seen_source_indices != set(range(source_count)):
        raise ValueError("every source sequence must have at least one target exposure")


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


def _validate_target_images(target_images: Tensor) -> int:
    _require_tensor(target_images, name="target_images")
    if target_images.dtype != torch.float32:
        raise TypeError("target_images must have dtype torch.float32")
    if target_images.ndim != 4 or tuple(target_images.shape[1:]) != TARGET_IMAGE_SHAPE:
        raise ValueError(
            "target_images must have shape "
            f"[K, {TARGET_IMAGE_CHANNELS}, {TARGET_IMAGE_HEIGHT}, {TARGET_IMAGE_WIDTH}]"
        )

    target_count = target_images.shape[0]
    if target_count == 0:
        raise ValueError("a training batch must contain at least one target exposure")
    if not torch.isfinite(target_images).all().item():
        raise ValueError("target_images must contain only finite values")
    if torch.any((target_images < -1.0) | (target_images > 1.0)).item():
        raise ValueError("target_images values must lie in [-1, 1]")
    return target_count


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


__all__ = [
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
    "FlowMatchingBatch",
    "FluxTextInputs",
    "NvarcBatchOutput",
    "PackedTrainingBatch",
    "validate_packed_source_sequences",
    "validate_packed_training_batch",
]
