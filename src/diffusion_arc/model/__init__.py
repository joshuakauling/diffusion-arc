"""Public model-layer contracts for Diffusion-ARC."""

from .contracts import (
    NVARC_CONTEXT_WIDTH,
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_MAX_TOKENS,
    NVARC_PAD_TOKEN_ID,
    NVARC_VOCAB_SIZE,
    TARGET_IMAGE_CHANNELS,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_SHAPE,
    TARGET_IMAGE_WIDTH,
    PackedNvarcHiddenStates,
    PackedTrainingBatch,
    validate_packed_source_sequences,
    validate_packed_training_batch,
)
from .nvarc import NvarcBatchRunner

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
    "NvarcBatchRunner",
    "PackedNvarcHiddenStates",
    "PackedTrainingBatch",
    "validate_packed_source_sequences",
    "validate_packed_training_batch",
]
