"""Composed frozen-NVARC, trainable-FLUX, frozen-VAE ARC model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .nvarc_conditioning import NVARCConditioner
from .flow import flow_matching_loss, make_flow_batch
from .grid_codec import GridDecodeResult, decode_arc_image, render_arc_canvas
from .positions import (
    flatten_latents,
    make_image_ids,
    make_text_ids,
    unflatten_latents,
)
from .sampling import bfl_time_schedule, euler_integrate


ARC_LATENT_CHANNELS = 128
ARC_LATENT_HEIGHT = 30
ARC_LATENT_WIDTH = 30
ARC_IMAGE_TOKENS = ARC_LATENT_HEIGHT * ARC_LATENT_WIDTH


@dataclass(frozen=True)
class DiffusionARCOutput:
    """Training output, with large diagnostic tensors returned only on request."""

    loss: Tensor
    timesteps: Tensor
    predicted_velocity: Tensor | None = None
    target_velocity: Tensor | None = None
    noisy_latents: Tensor | None = None
    clean_latents: Tensor | None = None


@dataclass(frozen=True)
class DiffusionARCGenerationOutput:
    """Generated packed latents, decoded images, and strict ARC decodes."""

    latents: Tensor
    images: Tensor
    decodes: tuple[GridDecodeResult, ...]
    timesteps: Tensor


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        if torch.is_floating_point(parameter):
            return parameter.device, parameter.dtype
    for buffer in module.buffers():
        if torch.is_floating_point(buffer):
            return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _require_packed_arc_latents(latents: Tensor, *, name: str) -> None:
    if not isinstance(latents, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if latents.ndim != 3:
        raise RuntimeError(
            f"{name} must have shape [B, {ARC_IMAGE_TOKENS}, "
            f"{ARC_LATENT_CHANNELS}], got {tuple(latents.shape)}."
        )
    expected = (latents.shape[0], ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS)
    if tuple(latents.shape) != expected:
        raise RuntimeError(
            f"{name} must have shape [B, {ARC_IMAGE_TOKENS}, "
            f"{ARC_LATENT_CHANNELS}], got {tuple(latents.shape)}."
        )
    if not torch.is_floating_point(latents):
        raise TypeError(f"{name} must use a floating-point dtype.")
    if not bool(torch.isfinite(latents).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values.")


class DiffusionARCModel(nn.Module):
    """The complete initial flow-only Diffusion-ARC runtime model.

    The module owns all three components while permitting explicit placement
    on separate devices. Calling ``model.to(...)`` is therefore intentionally
    unnecessary; place NVARC, FLUX, and the VAE before construction.
    """

    def __init__(
        self,
        *,
        conditioner: NVARCConditioner,
        flux: nn.Module,
        vae: nn.Module,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.flux = flux
        self.vae = vae

        self.conditioner.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.flux.requires_grad_(True)
        self.conditioner.eval()
        self.vae.eval()

    @property
    def nvarc(self) -> nn.Module:
        """Expose the frozen NVARC model owned by the conditioner."""

        return self.conditioner.model

    def train(self, mode: bool = True) -> DiffusionARCModel:
        """Train FLUX only while keeping NVARC and the VAE deterministic."""

        super().train(mode)
        self.flux.train(mode)
        self.conditioner.eval()
        self.vae.eval()
        return self

    def assert_parameter_boundaries(self) -> None:
        if any(parameter.requires_grad for parameter in self.conditioner.parameters()):
            raise RuntimeError("Every NVARC conditioner parameter must be frozen.")
        if any(parameter.requires_grad for parameter in self.vae.parameters()):
            raise RuntimeError("Every VAE parameter must be frozen.")
        flux_parameters = tuple(self.flux.parameters())
        if not flux_parameters:
            raise RuntimeError("FLUX must expose trainable parameters.")
        if not all(parameter.requires_grad for parameter in flux_parameters):
            raise RuntimeError("Every FLUX transformer parameter must be trainable.")
        if self.conditioner.training or self.conditioner.model.training:
            raise RuntimeError("NVARC must remain in evaluation mode.")
        if self.vae.training:
            raise RuntimeError("The VAE must remain in evaluation mode.")

    def encode_conditioning(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        context = self.conditioner(input_ids, attention_mask)
        flux_device, flux_dtype = _module_device_dtype(self.flux)
        context = context.to(device=flux_device, dtype=flux_dtype)
        expected = (input_ids.shape[0], input_ids.shape[1], 7680)
        if tuple(context.shape) != expected:
            raise RuntimeError(
                f"FLUX conditioning must have shape {expected}, got {tuple(context.shape)}."
            )
        if not bool(torch.isfinite(context).all().item()):
            raise FloatingPointError("FLUX conditioning contains non-finite values.")
        return context

    def encode_targets(self, target_grids: Any) -> Tensor:
        images = render_arc_canvas(target_grids)
        vae_device, vae_dtype = _module_device_dtype(self.vae)
        images = images.to(device=vae_device, dtype=vae_dtype)
        with torch.no_grad():
            latent_grid = self.vae.encode(images)
        expected_grid = (
            images.shape[0],
            ARC_LATENT_CHANNELS,
            ARC_LATENT_HEIGHT,
            ARC_LATENT_WIDTH,
        )
        if tuple(latent_grid.shape) != expected_grid:
            raise RuntimeError(
                f"VAE target latents must have shape {expected_grid}, "
                f"got {tuple(latent_grid.shape)}."
            )
        if not bool(torch.isfinite(latent_grid).all().item()):
            raise FloatingPointError("VAE target latents contain non-finite values.")

        packed = flatten_latents(latent_grid)
        flux_device, flux_dtype = _module_device_dtype(self.flux)
        packed = packed.to(device=flux_device, dtype=flux_dtype)
        _require_packed_arc_latents(packed, name="clean_latents")
        return packed

    def predict_velocity(
        self,
        *,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        image_ids: Tensor | None = None,
        text_ids: Tensor | None = None,
    ) -> Tensor:
        _require_packed_arc_latents(latents, name="latents")
        batch = latents.shape[0]
        if timesteps.shape != (batch,):
            raise ValueError(
                f"timesteps must have shape [{batch}], got {tuple(timesteps.shape)}."
            )
        if timesteps.device != latents.device:
            raise ValueError("timesteps and latents must be on the same device.")
        if not torch.is_floating_point(timesteps):
            raise TypeError("timesteps must use a floating-point dtype.")
        if not bool(torch.isfinite(timesteps).all().item()):
            raise ValueError("timesteps contain non-finite values.")
        if bool(((timesteps < 0) | (timesteps > 1)).any().item()):
            raise ValueError("timesteps must lie in [0, 1].")
        timesteps = timesteps.to(dtype=latents.dtype)
        if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != 7680:
            raise ValueError(
                f"context must have shape [B, L, 7680], got {tuple(context.shape)}."
            )
        if not 1 <= context.shape[1] <= 8192:
            raise ValueError(
                f"context length must lie in [1, 8192], got {context.shape[1]}."
            )
        if context.device != latents.device or context.dtype != latents.dtype:
            raise ValueError("context must match the FLUX latent device and dtype.")

        if image_ids is None:
            image_ids = make_image_ids(
                batch,
                ARC_LATENT_HEIGHT,
                ARC_LATENT_WIDTH,
                device=latents.device,
            )
        if text_ids is None:
            text_ids = make_text_ids(batch, context.shape[1], device=latents.device)
        if image_ids.shape != (batch, ARC_IMAGE_TOKENS, 4):
            raise ValueError(f"image_ids has unexpected shape {tuple(image_ids.shape)}.")
        if text_ids.shape != (batch, context.shape[1], 4):
            raise ValueError(f"text_ids has unexpected shape {tuple(text_ids.shape)}.")
        if image_ids.device != latents.device or text_ids.device != latents.device:
            raise ValueError("image_ids and text_ids must be on the FLUX latent device.")
        if image_ids.dtype != torch.int64 or text_ids.dtype != torch.int64:
            raise TypeError("image_ids and text_ids must use torch.int64.")

        velocity = self.flux(
            x=latents,
            x_ids=image_ids,
            timesteps=timesteps,
            ctx=context,
            ctx_ids=text_ids,
            guidance=None,
        )
        if tuple(velocity.shape) != tuple(latents.shape):
            raise RuntimeError(
                f"FLUX velocity shape {tuple(velocity.shape)} does not match "
                f"latents {tuple(latents.shape)}."
            )
        if not bool(torch.isfinite(velocity).all().item()):
            raise FloatingPointError("FLUX velocity contains non-finite values.")
        return velocity

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_grids: Any,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_details: bool = False,
    ) -> DiffusionARCOutput:
        self.assert_parameter_boundaries()
        context = self.encode_conditioning(input_ids, attention_mask)
        clean = self.encode_targets(target_grids)
        if context.shape[0] != clean.shape[0]:
            raise ValueError(
                f"Conditioning batch {context.shape[0]} does not match target "
                f"batch {clean.shape[0]}."
            )

        if timesteps is not None:
            timesteps = timesteps.to(device=clean.device)
        if noise is not None:
            noise = noise.to(device=clean.device)
        flow = make_flow_batch(
            clean,
            timesteps=timesteps,
            noise=noise,
            generator=generator,
        )
        predicted = self.predict_velocity(
            latents=flow.noisy_latents,
            timesteps=flow.timesteps,
            context=context,
        )
        loss = flow_matching_loss(predicted, flow.target_velocity)

        return DiffusionARCOutput(
            loss=loss,
            timesteps=flow.timesteps,
            predicted_velocity=predicted if return_details else None,
            target_velocity=flow.target_velocity if return_details else None,
            noisy_latents=flow.noisy_latents if return_details else None,
            clean_latents=flow.clean_latents if return_details else None,
        )

    def decode_latents(
        self, latents: Tensor
    ) -> tuple[Tensor, tuple[GridDecodeResult, ...]]:
        _require_packed_arc_latents(latents, name="latents")
        latent_grid = unflatten_latents(
            latents,
            ARC_LATENT_HEIGHT,
            ARC_LATENT_WIDTH,
        )
        vae_device, vae_dtype = _module_device_dtype(self.vae)
        latent_grid = latent_grid.to(device=vae_device, dtype=vae_dtype)
        with torch.no_grad():
            images = self.vae.decode(latent_grid)
        expected = (latents.shape[0], 3, 480, 480)
        if tuple(images.shape) != expected:
            raise RuntimeError(
                f"VAE output must have shape {expected}, got {tuple(images.shape)}."
            )
        if not bool(torch.isfinite(images).all().item()):
            raise FloatingPointError("VAE output contains non-finite values.")
        decodes = tuple(decode_arc_image(image) for image in images)
        return images, decodes

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        num_steps: int = 50,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DiffusionARCGenerationOutput:
        context = self.encode_conditioning(input_ids, attention_mask)
        flux_device, flux_dtype = _module_device_dtype(self.flux)
        batch = context.shape[0]
        if noise is None:
            latents = torch.randn(
                (batch, ARC_IMAGE_TOKENS, ARC_LATENT_CHANNELS),
                device=flux_device,
                dtype=flux_dtype,
                generator=generator,
            )
        else:
            latents = noise.to(device=flux_device, dtype=flux_dtype)
        _require_packed_arc_latents(latents, name="initial_noise")

        image_ids = make_image_ids(
            batch,
            ARC_LATENT_HEIGHT,
            ARC_LATENT_WIDTH,
            device=flux_device,
        )
        text_ids = make_text_ids(batch, context.shape[1], device=flux_device)
        schedule = bfl_time_schedule(
            num_steps,
            ARC_IMAGE_TOKENS,
            device=flux_device,
            dtype=torch.float32,
        )

        def velocity_fn(state: Tensor, time_batch: Tensor) -> Tensor:
            return self.predict_velocity(
                latents=state,
                timesteps=time_batch,
                context=context,
                image_ids=image_ids,
                text_ids=text_ids,
            )

        latents = euler_integrate(latents, velocity_fn, schedule)
        images, decodes = self.decode_latents(latents)
        return DiffusionARCGenerationOutput(
            latents=latents,
            images=images,
            decodes=decodes,
            timesteps=schedule,
        )
