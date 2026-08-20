"""Straight-path flow-matching sampling and velocity loss."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import FlowMatchingBatch, TARGET_LATENT_SHAPE


def _validate_target_latents(latents: Tensor, *, name: str) -> None:
    """Validate the narrow transformer-facing target-latent interface."""

    if not isinstance(latents, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if latents.ndim != 3 or tuple(latents.shape[1:]) != TARGET_LATENT_SHAPE:
        raise ValueError(
            f"{name} must have shape [K, {TARGET_LATENT_SHAPE[0]}, "
            f"{TARGET_LATENT_SHAPE[1]}]"
        )
    if latents.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one target exposure")
    if not latents.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype")


def sample_flow_matching_batch(
    clean_latents: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    """Sample uniform flow times and Gaussian noise for clean target latents.

    Each target exposure receives an independent scalar ``t`` and an
    independent standard-Gaussian tensor ``epsilon``. The returned state uses
    ``z_t = (1 - t) * z_0 + t * epsilon`` and the supervision target is
    ``epsilon - z_0``.
    """

    _validate_target_latents(clean_latents, name="clean_latents")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    target_count = clean_latents.shape[0]
    timesteps = torch.rand(
        (target_count,),
        device=clean_latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    broadcast_timesteps = timesteps.to(dtype=clean_latents.dtype).reshape(
        target_count,
        1,
        1,
    )
    noisy_latents = (
        (1 - broadcast_timesteps) * clean_latents
        + broadcast_timesteps * noise
    )
    target_velocity = noise - clean_latents

    return FlowMatchingBatch(
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        target_velocity=target_velocity,
    )


def flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
) -> Tensor:
    """Return the unweighted mean squared velocity error in float32."""

    _validate_target_latents(predicted_velocity, name="predicted_velocity")
    _validate_target_latents(target_velocity, name="target_velocity")
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "predicted_velocity and target_velocity must have identical shapes"
        )
    if predicted_velocity.device != target_velocity.device:
        raise ValueError(
            "predicted_velocity and target_velocity must use the same device"
        )

    return F.mse_loss(
        predicted_velocity.float(),
        target_velocity.float(),
        reduction="mean",
    )


__all__ = ["flow_matching_loss", "sample_flow_matching_batch"]
