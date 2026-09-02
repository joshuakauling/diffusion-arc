"""Build physical FLUX microbatches from prepared logical inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from diffusion_arc.model.contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    FluxSpatialShape,
    FluxTextInputs,
)
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids

from .flux_input import FluxInput, NvarcSequenceState


@dataclass(frozen=True)
class FluxMicrobatch:
    """Physical deterministic inputs for one later FLUX execution.

    Flow noise and timesteps are intentionally absent and are sampled only when
    this microbatch enters the trainable FLUX step.
    """

    text_inputs: FluxTextInputs
    clean_latents: Tensor
    image_position_ids: Tensor

    @property
    def target_count(self) -> int:
        """Return the number of logical FLUX inputs in this microbatch."""

        return self.clean_latents.shape[0]


def build_flux_microbatch(
    flux_inputs: Sequence[FluxInput],
    *,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> FluxMicrobatch:
    """Copy logical inputs into one physical batch and release consumed states.

    Every validation and tensor allocation completes before any pending prefix
    is removed. A failed build therefore leaves all source states
    available for diagnosis or retry.
    """

    inputs = tuple(flux_inputs)
    if not inputs:
        raise ValueError("flux_inputs must not be empty")
    if not isinstance(spatial_shape, FluxSpatialShape):
        raise TypeError("spatial_shape must be a FluxSpatialShape")

    pending_by_state: dict[NvarcSequenceState, list[int]] = {}
    for flux_input in inputs:
        pending_by_state.setdefault(flux_input.nvarc_sequence_state, []).append(
            flux_input.prefix_end
        )

    for sequence_state, prefix_ends in pending_by_state.items():
        sequence_state._validate_pending_prefixes(prefix_ends)

    hidden_state_prefixes = tuple(
        flux_input.nvarc_sequence_state._prefix_for(flux_input.prefix_end)
        for flux_input in inputs
    )
    text_inputs = pad_flux_text_prefixes(hidden_state_prefixes)
    clean_latents = torch.stack(
        tuple(flux_input.clean_latent for flux_input in inputs)
    )
    if clean_latents.ndim != 3 or clean_latents.shape[1] != spatial_shape.token_count:
        raise ValueError(
            "clean latent token count does not match the requested spatial shape"
        )
    image_position_ids = build_flux_image_position_ids(
        len(inputs),
        clean_latents.device,
        spatial_shape=spatial_shape,
    )
    microbatch = FluxMicrobatch(
        text_inputs=text_inputs,
        clean_latents=clean_latents,
        image_position_ids=image_position_ids,
    )

    for sequence_state, prefix_ends in pending_by_state.items():
        sequence_state._commit_prefixes(prefix_ends)

    return microbatch


__all__ = ["FluxMicrobatch", "build_flux_microbatch"]
