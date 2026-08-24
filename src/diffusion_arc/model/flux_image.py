"""Frozen FLUX.2 VAE encoding for fixed ARC target canvases."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import (
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)


def _module_input_device_and_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return the device and floating dtype expected by a model's parameters."""

    try:
        parameter = next(module.parameters())
    except StopIteration as error:
        raise RuntimeError("FLUX VAE must expose at least one parameter") from error
    if not parameter.dtype.is_floating_point:
        raise RuntimeError("FLUX VAE parameters must use a floating-point dtype")
    return parameter.device, parameter.dtype


def _validate_vae_interface(vae: nn.Module) -> None:
    """Validate stable FLUX.2 VAE structure once at module construction."""

    if not callable(getattr(vae, "encode", None)):
        raise RuntimeError("FLUX VAE must expose an encode() method")

    batch_norm = getattr(vae, "bn", None)
    running_mean = getattr(batch_norm, "running_mean", None)
    running_var = getattr(batch_norm, "running_var", None)
    if not isinstance(running_mean, Tensor) or not isinstance(running_var, Tensor):
        raise RuntimeError("FLUX VAE must expose BatchNorm running statistics")
    expected_shape = (TARGET_LATENT_CHANNELS,)
    if tuple(running_mean.shape) != expected_shape or tuple(running_var.shape) != expected_shape:
        raise RuntimeError(
            "FLUX VAE BatchNorm statistics must have shape "
            f"{expected_shape}"
        )

    config = getattr(vae, "config", None)
    batch_norm_eps = getattr(config, "batch_norm_eps", None)
    if (
        isinstance(batch_norm_eps, bool)
        or not isinstance(batch_norm_eps, (int, float))
        or batch_norm_eps <= 0
    ):
        raise RuntimeError("FLUX VAE config must expose a positive batch_norm_eps")

    _module_input_device_and_dtype(vae)


def _patchify_latents(latents: Tensor) -> Tensor:
    """Move each spatial 2x2 VAE patch into the channel dimension."""

    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        channels,
        height // 2,
        2,
        width // 2,
        2,
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(
        batch_size,
        channels * 4,
        height // 2,
        width // 2,
    )


class FluxVaeEncoder(nn.Module):
    """Encode target canvases through a permanently frozen FLUX.2 VAE."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        if not isinstance(vae, nn.Module):
            raise TypeError("vae must be a torch.nn.Module")
        _validate_vae_interface(vae)

        self.vae = vae
        self.vae.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> FluxVaeEncoder:
        """Keep both the encoder and its frozen VAE in evaluation mode."""

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        super().train(False)
        self.vae.eval()
        return self

    def forward(self, target_images: Tensor) -> Tensor:
        """Encode trusted target canvases into clean packed FLUX latents."""

        device, dtype = _module_input_device_and_dtype(self.vae)
        images = target_images.to(device=device, dtype=dtype)

        with torch.no_grad():
            encoder_output = self.vae.encode(images)
            latent_distribution = getattr(encoder_output, "latent_dist", None)
            posterior_mode = getattr(latent_distribution, "mode", None)
            if not callable(posterior_mode):
                raise RuntimeError(
                    "FLUX VAE encode() output must expose latent_dist.mode()"
                )
            vae_latents = posterior_mode()
            packed_latents = _patchify_latents(vae_latents)

            batch_norm = self.vae.bn
            running_mean = batch_norm.running_mean.reshape(1, -1, 1, 1).to(
                device=packed_latents.device,
                dtype=packed_latents.dtype,
            )
            running_std = torch.sqrt(
                batch_norm.running_var.reshape(1, -1, 1, 1)
                + self.vae.config.batch_norm_eps
            ).to(
                device=packed_latents.device,
                dtype=packed_latents.dtype,
            )
            normalized_latents = (packed_latents - running_mean) / running_std
            clean_latents = normalized_latents.reshape(
                images.shape[0],
                TARGET_LATENT_CHANNELS,
                TARGET_LATENT_TOKEN_COUNT,
            ).transpose(1, 2).contiguous()

        return clean_latents


__all__ = ["FluxVaeEncoder"]
