"""Frozen FLUX.2 VAE encoding for fixed ARC target canvases."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    TARGET_IMAGE_CHANNELS,
    TARGET_LATENT_CHANNELS,
    FluxSpatialShape,
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


def _validate_vae_normalization_interface(vae: nn.Module) -> None:
    """Validate shared FLUX.2 VAE normalization state once at construction."""

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


def _validate_vae_interface(vae: nn.Module) -> None:
    """Validate the stable FLUX.2 VAE encoder interface."""

    if not callable(getattr(vae, "encode", None)):
        raise RuntimeError("FLUX VAE must expose an encode() method")
    _validate_vae_normalization_interface(vae)


def _validate_vae_decoder_interface(vae: nn.Module) -> None:
    """Validate the stable FLUX.2 VAE decoder interface."""

    if not callable(getattr(vae, "decode", None)):
        raise RuntimeError("FLUX VAE must expose a decode() method")
    _validate_vae_normalization_interface(vae)


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


def _unpatchify_latents(latents: Tensor) -> Tensor:
    """Move packed channel groups back into spatial 2x2 VAE patches."""

    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        channels // 4,
        2,
        2,
        height,
        width,
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(
        batch_size,
        channels // 4,
        height * 2,
        width * 2,
    )


def _normalization_statistics(
    vae: nn.Module,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return broadcast FLUX VAE BatchNorm statistics on a tensor's placement."""

    running_mean = vae.bn.running_mean.reshape(1, -1, 1, 1).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    running_std = torch.sqrt(
        vae.bn.running_var.reshape(1, -1, 1, 1)
        + vae.config.batch_norm_eps
    ).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    return running_mean, running_std


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

    def forward(
        self,
        target_images: Tensor,
        *,
        spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
    ) -> Tensor:
        """Encode trusted target canvases into clean packed FLUX latents."""

        if not isinstance(target_images, Tensor):
            raise TypeError("target_images must be a torch.Tensor")
        if not isinstance(spatial_shape, FluxSpatialShape):
            raise TypeError("spatial_shape must be a FluxSpatialShape")
        expected_image_shape = (
            TARGET_IMAGE_CHANNELS,
            spatial_shape.image_height,
            spatial_shape.image_width,
        )
        if (
            target_images.ndim != 4
            or tuple(target_images.shape[1:]) != expected_image_shape
        ):
            raise ValueError(
                f"target_images must have shape [B, {expected_image_shape[0]}, "
                f"{expected_image_shape[1]}, {expected_image_shape[2]}]"
            )
        if target_images.shape[0] == 0:
            raise ValueError("target_images must contain at least one image")
        if not target_images.dtype.is_floating_point:
            raise TypeError("target_images must use a floating-point dtype")

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
            expected_packed_shape = (
                images.shape[0],
                TARGET_LATENT_CHANNELS,
                spatial_shape.height,
                spatial_shape.width,
            )
            if tuple(packed_latents.shape) != expected_packed_shape:
                raise RuntimeError(
                    "packed VAE latents must have shape "
                    f"{expected_packed_shape}, received {tuple(packed_latents.shape)}"
                )

            running_mean, running_std = _normalization_statistics(
                self.vae,
                packed_latents,
            )
            normalized_latents = (packed_latents - running_mean) / running_std
            clean_latents = normalized_latents.reshape(
                images.shape[0],
                TARGET_LATENT_CHANNELS,
                spatial_shape.token_count,
            ).transpose(1, 2).contiguous()

        return clean_latents


class FluxVaeDecoder(nn.Module):
    """Decode packed FLUX states through a permanently frozen FLUX.2 VAE."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        if not isinstance(vae, nn.Module):
            raise TypeError("vae must be a torch.nn.Module")
        _validate_vae_decoder_interface(vae)

        self.vae = vae
        self.vae.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> FluxVaeDecoder:
        """Keep both the decoder and its frozen VAE in evaluation mode."""

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        super().train(False)
        self.vae.eval()
        return self

    def forward(
        self,
        packed_latents: Tensor,
        *,
        spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
    ) -> Tensor:
        """Decode trusted packed FLUX latents into normalized RGB images."""

        if not isinstance(packed_latents, Tensor):
            raise TypeError("packed_latents must be a torch.Tensor")
        if not isinstance(spatial_shape, FluxSpatialShape):
            raise TypeError("spatial_shape must be a FluxSpatialShape")
        if (
            packed_latents.ndim != 3
            or tuple(packed_latents.shape[1:])
            != (spatial_shape.token_count, TARGET_LATENT_CHANNELS)
        ):
            raise ValueError(
                f"packed_latents must have shape [B, {spatial_shape.token_count}, "
                f"{TARGET_LATENT_CHANNELS}]"
            )
        if packed_latents.shape[0] == 0:
            raise ValueError("packed_latents must contain at least one sample")
        if not packed_latents.dtype.is_floating_point:
            raise TypeError("packed_latents must use a floating-point dtype")

        device, dtype = _module_input_device_and_dtype(self.vae)
        packed_grid = packed_latents.to(device=device, dtype=dtype).transpose(1, 2)
        packed_grid = packed_grid.reshape(
            packed_latents.shape[0],
            TARGET_LATENT_CHANNELS,
            spatial_shape.height,
            spatial_shape.width,
        )

        with torch.no_grad():
            running_mean, running_std = _normalization_statistics(
                self.vae,
                packed_grid,
            )
            vae_latents = _unpatchify_latents(
                packed_grid * running_std + running_mean
            )
            decoder_output = self.vae.decode(vae_latents)
            images = getattr(decoder_output, "sample", None)

        if not isinstance(images, Tensor):
            raise RuntimeError("FLUX VAE decode() output must expose a tensor sample")
        expected_shape = (
            packed_latents.shape[0],
            TARGET_IMAGE_CHANNELS,
            spatial_shape.image_height,
            spatial_shape.image_width,
        )
        if tuple(images.shape) != expected_shape:
            raise RuntimeError(
                f"decoded images must have shape {expected_shape}, "
                f"received {tuple(images.shape)}"
            )
        if not images.dtype.is_floating_point:
            raise RuntimeError("decoded images must use a floating-point dtype")
        return images


__all__ = ["FluxVaeDecoder", "FluxVaeEncoder"]
