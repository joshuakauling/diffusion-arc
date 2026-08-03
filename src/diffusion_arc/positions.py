"""FLUX.2 token layout and four-axis position IDs."""

from __future__ import annotations

import torch
from torch import Tensor


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def make_text_ids(
    batch: int,
    length: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return BFL-compatible text IDs ``(0, 0, 0, l)``.

    The returned tensor has shape ``[batch, length, 4]`` and dtype
    ``torch.int64``.
    """

    batch = _positive_integer(batch, name="batch")
    length = _positive_integer(length, name="length")
    ids = torch.zeros((length, 4), device=device, dtype=torch.int64)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.int64)
    return ids.unsqueeze(0).expand(batch, -1, -1).clone()


def make_image_ids(
    batch: int,
    height: int,
    width: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return BFL-compatible image IDs ``(0, h, w, 0)``.

    Coordinates use row-major token order (width changes fastest), matching
    :func:`flatten_latents`.
    """

    batch = _positive_integer(batch, name="batch")
    height = _positive_integer(height, name="height")
    width = _positive_integer(width, name="width")

    rows = torch.arange(height, device=device, dtype=torch.int64)
    columns = torch.arange(width, device=device, dtype=torch.int64)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    ids = torch.zeros((height * width, 4), device=device, dtype=torch.int64)
    ids[:, 1] = row_grid.reshape(-1)
    ids[:, 2] = column_grid.reshape(-1)
    return ids.unsqueeze(0).expand(batch, -1, -1).clone()


def flatten_latents(latents: Tensor) -> Tensor:
    """Convert ``[B, C, H, W]`` latents to row-major ``[B, H*W, C]``."""

    if not isinstance(latents, Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim != 4:
        raise ValueError(
            f"latents must have shape [B, C, H, W], got {tuple(latents.shape)}"
        )
    if any(size <= 0 for size in latents.shape):
        raise ValueError("all latent dimensions must be positive")
    if not torch.is_floating_point(latents):
        raise TypeError("latents must use a floating-point dtype")
    if not torch.isfinite(latents).all():
        raise ValueError("latents contain non-finite values")
    return latents.permute(0, 2, 3, 1).reshape(
        latents.shape[0], latents.shape[2] * latents.shape[3], latents.shape[1]
    )


def unflatten_latents(tokens: Tensor, height: int, width: int) -> Tensor:
    """Convert row-major ``[B, H*W, C]`` tokens to ``[B, C, H, W]``."""

    if not isinstance(tokens, Tensor):
        raise TypeError("tokens must be a torch.Tensor")
    if tokens.ndim != 3:
        raise ValueError(
            f"tokens must have shape [B, H*W, C], got {tuple(tokens.shape)}"
        )
    height = _positive_integer(height, name="height")
    width = _positive_integer(width, name="width")
    if tokens.shape[1] != height * width:
        raise ValueError(
            f"token count {tokens.shape[1]} does not equal height*width "
            f"({height * width})"
        )
    if tokens.shape[0] <= 0 or tokens.shape[2] <= 0:
        raise ValueError("batch and channel dimensions must be positive")
    if not torch.is_floating_point(tokens):
        raise TypeError("tokens must use a floating-point dtype")
    if not torch.isfinite(tokens).all():
        raise ValueError("tokens contain non-finite values")
    return tokens.reshape(tokens.shape[0], height, width, tokens.shape[2]).permute(
        0, 3, 1, 2
    )
