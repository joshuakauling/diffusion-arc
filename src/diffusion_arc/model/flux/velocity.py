"""Shared FLUX velocity execution for training and generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    TARGET_LATENT_CHANNELS,
    FluxSpatialShape,
    FluxTextInputs,
)
from .conditioning import build_flux_joint_attention_mask


def flux_transformer_device(transformer: nn.Module) -> torch.device:
    """Return the device holding the persistent FP32 FLUX parameters."""

    if not isinstance(transformer, nn.Module):
        raise TypeError("transformer must be a torch.nn.Module")
    try:
        parameter = next(transformer.parameters())
    except StopIteration as error:
        raise RuntimeError("FLUX transformer must expose parameters") from error
    if parameter.dtype != torch.float32:
        raise RuntimeError("persistent FLUX parameters must use float32")
    return parameter.device


def build_flux_image_position_ids(
    batch_size: int,
    device: torch.device,
    *,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
) -> Tensor:
    """Build ``(0, h, w, 0)`` coordinates for an explicit image-token grid."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if not isinstance(spatial_shape, FluxSpatialShape):
        raise TypeError("spatial_shape must be a FluxSpatialShape")

    position_ids = torch.zeros(
        (spatial_shape.height, spatial_shape.width, 4),
        dtype=torch.int64,
        device=device,
    )
    position_ids[:, :, 1] = torch.arange(
        spatial_shape.height,
        device=device,
    ).unsqueeze(1)
    position_ids[:, :, 2] = torch.arange(
        spatial_shape.width,
        device=device,
    ).unsqueeze(0)
    return position_ids.reshape(1, spatial_shape.token_count, 4).expand(
        batch_size,
        -1,
        -1,
    )


def predict_flux_velocity(
    transformer: nn.Module,
    latents: Tensor,
    text_inputs: FluxTextInputs,
    image_position_ids: Tensor,
    timesteps: Tensor,
    *,
    compute_dtype: torch.dtype,
) -> Tensor:
    """Run one FLUX velocity prediction through the shared model interface."""

    batch_size = _validate_execution_inputs(
        latents=latents,
        text_inputs=text_inputs,
        image_position_ids=image_position_ids,
        timesteps=timesteps,
    )
    if (
        not isinstance(compute_dtype, torch.dtype)
        or not compute_dtype.is_floating_point
    ):
        raise TypeError("compute_dtype must be a floating-point torch.dtype")

    transformer_device = flux_transformer_device(transformer)
    joint_attention_mask = build_flux_joint_attention_mask(
        text_inputs.text_attention_mask,
        image_token_count=latents.shape[1],
    )
    with torch.autocast(
        device_type=transformer_device.type,
        dtype=compute_dtype,
    ):
        output = transformer(
            hidden_states=latents.to(device=transformer_device),
            encoder_hidden_states=text_inputs.encoder_hidden_states.to(
                device=transformer_device
            ),
            timestep=timesteps.to(device=transformer_device),
            img_ids=image_position_ids.to(device=transformer_device),
            txt_ids=text_inputs.text_position_ids.to(device=transformer_device),
            guidance=None,
            joint_attention_kwargs={
                "attention_mask": joint_attention_mask.to(device=transformer_device)
            },
            return_dict=True,
        )

    predicted_velocity = getattr(output, "sample", None)
    if not isinstance(predicted_velocity, Tensor):
        raise RuntimeError("FLUX transformer output must expose a tensor sample")
    expected_shape = tuple(latents.shape)
    if tuple(predicted_velocity.shape) != expected_shape:
        raise RuntimeError(
            f"FLUX velocity must have shape {expected_shape}, "
            f"received {tuple(predicted_velocity.shape)}"
        )
    if not predicted_velocity.dtype.is_floating_point:
        raise RuntimeError("FLUX velocity must use a floating-point dtype")
    return predicted_velocity


def _validate_execution_inputs(
    *,
    latents: Tensor,
    text_inputs: FluxTextInputs,
    image_position_ids: Tensor,
    timesteps: Tensor,
) -> int:
    if not isinstance(latents, Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim != 3 or latents.shape[2] != TARGET_LATENT_CHANNELS:
        raise ValueError(
            f"latents must have shape [B, T, {TARGET_LATENT_CHANNELS}]"
        )
    if latents.shape[0] == 0:
        raise ValueError("latents must contain at least one sample")
    if latents.shape[1] == 0:
        raise ValueError("latents must contain at least one image token")
    if not latents.dtype.is_floating_point:
        raise TypeError("latents must use a floating-point dtype")

    if not isinstance(text_inputs, FluxTextInputs):
        raise TypeError("text_inputs must be a FluxTextInputs instance")
    batch_size = latents.shape[0]
    hidden_states = text_inputs.encoder_hidden_states
    text_position_ids = text_inputs.text_position_ids
    text_attention_mask = text_inputs.text_attention_mask
    if hidden_states.ndim != 3 or hidden_states.shape[0] != batch_size:
        raise ValueError("encoder_hidden_states must have shape [B, L, D]")
    text_length = hidden_states.shape[1]
    if text_length == 0:
        raise ValueError("encoder_hidden_states must contain at least one token")
    if tuple(text_position_ids.shape) != (batch_size, text_length, 4):
        raise ValueError("text_position_ids must have shape [B, L, 4]")
    if text_position_ids.dtype != torch.int64:
        raise TypeError("text_position_ids must have dtype torch.int64")
    if tuple(text_attention_mask.shape) != (batch_size, text_length):
        raise ValueError("text_attention_mask must have shape [B, L]")
    if text_attention_mask.dtype != torch.bool:
        raise TypeError("text_attention_mask must have dtype torch.bool")

    expected_image_ids_shape = (batch_size, latents.shape[1], 4)
    if tuple(image_position_ids.shape) != expected_image_ids_shape:
        raise ValueError(
            f"image_position_ids must have shape {expected_image_ids_shape}"
        )
    if image_position_ids.dtype != torch.int64:
        raise TypeError("image_position_ids must have dtype torch.int64")
    if tuple(timesteps.shape) != (batch_size,):
        raise ValueError("timesteps must have shape [B]")
    if not timesteps.dtype.is_floating_point:
        raise TypeError("timesteps must use a floating-point dtype")
    return batch_size


__all__ = [
    "build_flux_image_position_ids",
    "flux_transformer_device",
    "predict_flux_velocity",
]
