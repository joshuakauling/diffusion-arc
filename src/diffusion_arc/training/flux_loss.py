"""Execute one prepared FLUX microbatch and compute its flow loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion_arc.model.flow_matching import (
    flow_matching_loss,
    sample_flow_matching_batch,
)
from diffusion_arc.model.flux_velocity import predict_flux_velocity

from .flux_microbatch import FluxMicrobatch


def compute_flux_microbatch_loss(
    transformer: nn.Module,
    microbatch: FluxMicrobatch,
    *,
    compute_dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Run one BF16-autocast FLUX forward and return its FP32 mean loss."""

    flow_batch = sample_flow_matching_batch(
        microbatch.clean_latents,
        generator=generator,
    )
    predicted_velocity = predict_flux_velocity(
        transformer,
        flow_batch.noisy_latents,
        microbatch.text_inputs,
        microbatch.image_position_ids,
        flow_batch.timesteps,
        compute_dtype=compute_dtype,
    )
    return flow_matching_loss(
        predicted_velocity,
        flow_batch.target_velocity.to(device=predicted_velocity.device),
    )


__all__ = ["compute_flux_microbatch_loss"]
