"""Public model-layer contracts for Diffusion-ARC."""

from .contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
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
    FluxSpatialShape,
    FluxTextInputs,
    NvarcBatchOutput,
    validate_packed_source_sequences,
)

from .flux_conditioning import (
    build_flux_joint_attention_mask,
    pad_flux_text_prefixes,
)
from .flux_vae import FluxVaeDecoder, FluxVaeEncoder
from .flux_velocity import (
    build_flux_image_position_ids,
    flux_transformer_device,
    predict_flux_velocity,
)
from .nvarc_encoder import NvarcEncoder

__all__ = [
    "DEFAULT_FLUX_SPATIAL_SHAPE",
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
    "FluxSpatialShape",
    "FluxTextInputs",
    "FluxVaeDecoder",
    "FluxVaeEncoder",
    "NvarcBatchOutput",
    "NvarcEncoder",
    "build_flux_joint_attention_mask",
    "build_flux_image_position_ids",
    "flux_transformer_device",
    "pad_flux_text_prefixes",
    "predict_flux_velocity",
    "validate_packed_source_sequences",
]
