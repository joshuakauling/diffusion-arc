"""Public model-layer contracts for Diffusion-ARC."""

from .contracts import (
    NVARC_MAX_TOKENS,
    NVARC_VOCAB_SIZE,
    TARGET_IMAGE_CHANNELS,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_SHAPE,
    TARGET_IMAGE_WIDTH,
    PackedTrainingBatch,
    validate_packed_training_batch,
)

__all__ = [
    "NVARC_MAX_TOKENS",
    "NVARC_VOCAB_SIZE",
    "TARGET_IMAGE_CHANNELS",
    "TARGET_IMAGE_HEIGHT",
    "TARGET_IMAGE_SHAPE",
    "TARGET_IMAGE_WIDTH",
    "PackedTrainingBatch",
    "validate_packed_training_batch",
]
