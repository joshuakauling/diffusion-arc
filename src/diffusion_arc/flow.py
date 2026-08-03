"""Straight-path flow-matching corruption and objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowBatch:
    """One sampled straight-path flow training batch."""

    clean_latents: Tensor
    noisy_latents: Tensor
    noise: Tensor
    timesteps: Tensor
    target_velocity: Tensor


def _validate_latents(latents: Tensor, *, name: str) -> None:
    if not isinstance(latents, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if latents.ndim != 3:
        raise ValueError(f"{name} must have shape [B, N, C]")
    if any(size <= 0 for size in latents.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if not torch.is_floating_point(latents):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(latents).all():
        raise ValueError(f"{name} contains non-finite values")


def make_flow_batch(
    clean: Tensor,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowBatch:
    """Construct ``z_t=(1-t)z_0+t*epsilon`` and ``epsilon-z_0``.

    ``clean`` may use any token and channel dimensions for tiny contract tests;
    the composed ARC model is responsible for enforcing ``[B, 900, 128]``.
    Explicit noise and timesteps make this helper fully deterministic.
    """

    _validate_latents(clean, name="clean")
    batch = clean.shape[0]

    if timesteps is None:
        timesteps = torch.rand(
            (batch,),
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    else:
        if not isinstance(timesteps, Tensor):
            raise TypeError("timesteps must be a torch.Tensor")
        if timesteps.shape != (batch,):
            raise ValueError(
                f"timesteps must have shape [{batch}], got {tuple(timesteps.shape)}"
            )
        if not torch.is_floating_point(timesteps):
            raise TypeError("timesteps must use a floating-point dtype")
        if timesteps.device != clean.device:
            raise ValueError("timesteps and clean must be on the same device")
        timesteps = timesteps.to(dtype=clean.dtype)

    if not torch.isfinite(timesteps).all():
        raise ValueError("timesteps contain non-finite values")
    if bool(((timesteps < 0) | (timesteps > 1)).any()):
        raise ValueError("timesteps must lie in [0, 1]")

    if noise is None:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
    else:
        _validate_latents(noise, name="noise")
        if noise.shape != clean.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} must equal clean shape "
                f"{tuple(clean.shape)}"
            )
        if noise.device != clean.device:
            raise ValueError("noise and clean must be on the same device")
        if noise.dtype != clean.dtype:
            raise TypeError("noise and clean must use the same dtype")

    broadcast_time = timesteps.reshape(batch, 1, 1)
    noisy = (1 - broadcast_time) * clean + broadcast_time * noise
    target = noise - clean
    if not torch.isfinite(noisy).all() or not torch.isfinite(target).all():
        raise FloatingPointError("flow corruption produced non-finite values")

    return FlowBatch(
        clean_latents=clean,
        noisy_latents=noisy,
        noise=noise,
        timesteps=timesteps,
        target_velocity=target,
    )


def flow_matching_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Return mean squared velocity error, accumulated in float32."""

    _validate_latents(prediction, name="prediction")
    _validate_latents(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
    if not torch.isfinite(loss):
        raise FloatingPointError("flow matching loss is non-finite")
    return loss
