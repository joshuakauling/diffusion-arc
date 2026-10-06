"""Build packed FLUX inputs and execute one forward with the selected objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diffusion_arc.model.contracts import DEFAULT_FLUX_SPATIAL_SHAPE, FluxSpatialShape, TARGET_LATENT_SHAPE
from diffusion_arc.model.flux.packing import PackingLayout, packing_layout
from diffusion_arc.model.flux.velocity import build_flux_image_position_ids

from .encoding import FluxInput, NvarcSequenceState
from .flow_sampling import FlowMatchingBatch
from .loss import LossResult, VelocityLoss
from ..profiling import training_range


@dataclass(frozen=True)
class FluxMicrobatch:
    """Text conditioning, attention layout and image positions for one FLUX call.

    Image latents and flow times are supplied separately through FlowMatchingBatch.
    """

    text: Tensor
    layout: PackingLayout
    image_position_ids: Tensor

    @property
    def target_count(self) -> int:
        return len(self.layout.text_lengths)


def build_flux_microbatch(
    flux_inputs: Sequence[FluxInput],
    *,
    max_joint_tokens: int,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> FluxMicrobatch:
    """Copy target conditioning, committing source release only after success.

    Pending-prefix checks protect changing ownership. All allocations complete
    before any prefix is consumed, so a failed build leaves its sources intact.
    """
    inputs = tuple(flux_inputs)
    if not inputs:
        raise ValueError("flux_inputs must not be empty")
    if sum(item.prefix_end + spatial_shape.token_count for item in inputs) > max_joint_tokens:
        raise ValueError("packed microbatch exceeds the joint token budget")

    pending_by_state: dict[NvarcSequenceState, list[int]] = {}
    for item in inputs:
        pending_by_state.setdefault(item.nvarc_sequence_state, []).append(item.prefix_end)
    for state, prefix_ends in pending_by_state.items():
        state._validate_pending_prefixes(prefix_ends)

    prefixes = tuple(item.nvarc_sequence_state._prefix_for(item.prefix_end) for item in inputs)
    text = torch.cat(prefixes)
    layout = packing_layout(tuple(item.prefix_end for item in inputs), spatial_shape.token_count, text.device)
    positions = build_flux_image_position_ids(len(inputs), text.device, spatial_shape=spatial_shape)
    microbatch = FluxMicrobatch(text, layout, positions)
    for state, prefix_ends in pending_by_state.items():
        state._commit_prefixes(prefix_ends)
    return microbatch


def compute_flux_microbatch_loss(
    inputs: Sequence[FluxInput],
    flow_batch: FlowMatchingBatch,
    *,
    transformer: nn.Module,
    max_joint_tokens: int,
    loss_function: VelocityLoss,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> LossResult:
    """Build conditioning, run FLUX with the supplied flow batch and compute loss.

    ``transformer`` is a PackedFluxForward module, optionally wrapped in DDP.
    The loss receives the predicted velocity, target velocity and grid shapes.
    Grid shapes are never passed to FLUX. Sampling, optimizer-batch weighting,
    backward and logging are handled by the caller.
    """
    microbatch = build_flux_microbatch(
        inputs, max_joint_tokens=max_joint_tokens, spatial_shape=spatial_shape,
    )
    with training_range("train/flux_forward", torch_range=True):
        prediction = transformer(
            flow_batch.noisy_latents, microbatch.text, microbatch.image_position_ids,
            flow_batch.timesteps, microbatch.layout,
        )
    return loss_function(
        prediction, flow_batch.target_velocity.to(device=prediction.device),
        tuple(item.grid_shape for item in inputs),
    )


def compute_fsdp_dummy_loss(transformer: nn.Module, *, device: torch.device) -> Tensor:
    """Participate in every FLUX block with one zero-weight synthetic target.

    No source cache, target supervision or RNG is consumed. The ordinary graph
    reaches every trainable block so FSDP executes the same collectives as peers.
    """
    latents = torch.zeros((1, *TARGET_LATENT_SHAPE), device=device, dtype=torch.float32)
    text = torch.zeros((1, transformer.transformer.config.joint_attention_dim),
                       device=device, dtype=transformer.compute_dtype)
    positions = build_flux_image_position_ids(1, device)
    layout = packing_layout((1,), TARGET_LATENT_SHAPE[0], device)
    timesteps = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    with training_range("train/flux_forward", torch_range=True):
        prediction = transformer(latents, text, positions, timesteps, layout)
    return prediction.float().mean() * 0.0


__all__ = ["FluxMicrobatch", "build_flux_microbatch", "compute_flux_microbatch_loss", "compute_fsdp_dummy_loss"]
