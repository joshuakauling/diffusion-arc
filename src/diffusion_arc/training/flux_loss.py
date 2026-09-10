"""Execute one prepared FLUX microbatch and compute its flow loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion_arc.model.flow_matching import (
    flow_matching_loss,
    sample_flow_matching_batch,
)
from diffusion_arc.model.contracts import FlowMatchingBatch
from diffusion_arc.model.flux_velocity import predict_flux_velocity

from .profiling import training_range
from .flux_microbatch import FluxMicrobatch, PackedFluxMicrobatch


def compute_flux_microbatch_loss(
    transformer: nn.Module,
    microbatch: FluxMicrobatch | PackedFluxMicrobatch,
    *,
    compute_dtype: torch.dtype,
    generator: torch.Generator | None = None,
    flow_batch: FlowMatchingBatch | None = None,
    packing_backend: object | None = None,
) -> Tensor:
    """Return the FP32 mean loss, optionally using already assigned flow inputs."""

    if flow_batch is None:
        flow_batch = sample_flow_matching_batch(
            microbatch.clean_latents,
            generator=generator,
        )
    if isinstance(microbatch, PackedFluxMicrobatch):
        from diffusion_arc.model.flux_packing import predict_packed_flux_velocity

        if packing_backend is None:
            raise ValueError("packed FLUX loss requires the preflighted FA3 backend")
        with training_range("train/flux_forward", torch_range=True):
            predicted_velocity = predict_packed_flux_velocity(
                transformer, flow_batch.noisy_latents, microbatch.text, microbatch.image_position_ids,
                flow_batch.timesteps, microbatch.layout, compute_dtype=compute_dtype, backend=packing_backend,
            )
    else:
        if packing_backend is not None:
            raise ValueError("FA3 packing backend requires a packed microbatch")
        predicted_velocity = predict_flux_velocity(
            transformer, flow_batch.noisy_latents, microbatch.text_inputs,
            microbatch.image_position_ids, flow_batch.timesteps, compute_dtype=compute_dtype,
        )
    return flow_matching_loss(
        predicted_velocity,
        flow_batch.target_velocity.to(device=predicted_velocity.device),
    )


__all__ = ["compute_flux_microbatch_loss"]
