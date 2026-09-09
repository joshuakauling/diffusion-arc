"""Build physical FLUX microbatches from prepared logical inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from diffusion_arc.model.contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    FluxSpatialShape,
    FluxTextInputs,
)
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids
from diffusion_arc.sequence_buckets import SequenceLengthBuckets

from .flux_input import FluxInput, NvarcSequenceState

if TYPE_CHECKING:
    from diffusion_arc.model.flux_packing import PackingLayout


@dataclass(frozen=True)
class FluxMicrobatch:
    """Physical deterministic inputs for one later FLUX execution.

    Flow noise and timesteps are supplied separately by the trainable FLUX step.
    """

    text_inputs: FluxTextInputs
    clean_latents: Tensor
    image_position_ids: Tensor

    @property
    def target_count(self) -> int:
        """Return the number of logical FLUX inputs in this microbatch."""

        return self.clean_latents.shape[0]


@dataclass(frozen=True)
class PackedFluxMicrobatch:
    """Valid text tokens and independent attention segments for one physical call."""

    text: Tensor
    layout: PackingLayout
    clean_latents: Tensor
    image_position_ids: Tensor

    @property
    def target_count(self) -> int:
        return self.clean_latents.shape[0]


def build_flux_microbatch(
    flux_inputs: Sequence[FluxInput],
    *,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
    length_buckets: SequenceLengthBuckets = SequenceLengthBuckets(),
    padded_length: int | None = None,
    max_joint_tokens: int | None = None,
) -> FluxMicrobatch | PackedFluxMicrobatch:
    """Copy logical inputs into one physical batch and release consumed states.

    Every validation and tensor allocation completes before any pending prefix
    is removed. A failed build therefore leaves all source states
    available for diagnosis or retry. Once copying succeeds, owners with no
    remaining prefixes release their full-source tensor references; this batch
    retains its own conditioning tensors for the later FLUX forward/backward.
    """

    inputs = tuple(flux_inputs)
    if not inputs:
        raise ValueError("flux_inputs must not be empty")
    if not isinstance(spatial_shape, FluxSpatialShape):
        raise TypeError("spatial_shape must be a FluxSpatialShape")
    if max_joint_tokens is not None:
        if type(max_joint_tokens) is not int or max_joint_tokens <= 0:
            raise ValueError("max_joint_tokens must be a positive integer")
        if padded_length is not None or length_buckets != SequenceLengthBuckets():
            raise ValueError("packed microbatches cannot request text padding")
        if sum(item.prefix_end + spatial_shape.token_count for item in inputs) > max_joint_tokens:
            raise ValueError("packed microbatch exceeds the joint token budget")

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
    if max_joint_tokens is None:
        text_inputs = pad_flux_text_prefixes(
            hidden_state_prefixes, length_buckets=length_buckets, padded_length=padded_length,
        )
    else:
        from diffusion_arc.model.flux_packing import packing_layout

        text = torch.cat(hidden_state_prefixes)
        layout = packing_layout(tuple(item.prefix_end for item in inputs), spatial_shape.token_count, text.device)
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
    if max_joint_tokens is None:
        microbatch = FluxMicrobatch(text_inputs, clean_latents, image_position_ids)
    else:
        if clean_latents.device != text.device:
            raise ValueError("packed text and latents must use the same device")
        microbatch = PackedFluxMicrobatch(text, layout, clean_latents, image_position_ids)

    for sequence_state, prefix_ends in pending_by_state.items():
        sequence_state._commit_prefixes(prefix_ends)

    return microbatch


__all__ = ["FluxMicrobatch", "PackedFluxMicrobatch", "build_flux_microbatch"]
