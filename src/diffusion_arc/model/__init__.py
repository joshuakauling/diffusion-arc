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
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_HEIGHT,
    TARGET_LATENT_SHAPE,
    TARGET_LATENT_TOKEN_COUNT,
    TARGET_LATENT_WIDTH,
    FlowMatchingBatch,
    FluxTextInputs,
    NvarcBatchOutput,
    PackedTrainingBatch,
    validate_packed_source_sequences,
    validate_packed_training_batch,
)
from .flow import flow_matching_loss, sample_flow_matching_batch
from .flux_image import FluxVaeEncoder
from .flux_text import build_flux_joint_attention_mask, pad_flux_text_prefixes
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
    "TARGET_LATENT_CHANNELS",
    "TARGET_LATENT_HEIGHT",
    "TARGET_LATENT_SHAPE",
    "TARGET_LATENT_TOKEN_COUNT",
    "TARGET_LATENT_WIDTH",
    "FlowMatchingBatch",
    "FluxTextInputs",
    "FluxVaeEncoder",
    "NvarcBatchOutput",
    "NvarcBatchRunner",
    "PackedTrainingBatch",
    "build_flux_joint_attention_mask",
    "flow_matching_loss",
    "pad_flux_text_prefixes",
    "sample_flow_matching_batch",
    "validate_packed_source_sequences",
    "validate_packed_training_batch",
]
