"""Fixed ARC canvas rendering and strict deterministic decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


CANVAS_SIZE = 30
CELL_SIZE = 16
PAD_CLASS = 10

# Common ARC visualization colors followed by the white outside sentinel.
ARC_PALETTE_RGB = torch.tensor(
    [
        (0x00, 0x00, 0x00),
        (0x00, 0x74, 0xD9),
        (0xFF, 0x41, 0x36),
        (0x2E, 0xCC, 0x40),
        (0xFF, 0xDC, 0x00),
        (0xAA, 0xAA, 0xAA),
        (0xF0, 0x12, 0xBE),
        (0xFF, 0x85, 0x1B),
        (0x7F, 0xDB, 0xFF),
        (0x87, 0x0C, 0x25),
        (0xFF, 0xFF, 0xFF),
    ],
    dtype=torch.uint8,
)


@dataclass(frozen=True)
class GridDecodeResult:
    """Result of deterministic PAD-boundary decoding."""

    valid: bool
    grid: Tensor | None
    canvas: Tensor
    height: int | None
    width: int | None
    reason: str | None


def _require_cell_size(cell_size: int) -> None:
    if isinstance(cell_size, bool) or not isinstance(cell_size, int):
        raise TypeError("cell_size must be an integer")
    if cell_size != CELL_SIZE:
        raise ValueError(
            f"the initial architecture requires cell_size={CELL_SIZE}, got {cell_size}"
        )


def _validate_grid_tensor(grid: Tensor, *, name: str, allow_pad: bool) -> Tensor:
    if grid.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W]")
    height, width = grid.shape
    if not 1 <= height <= CANVAS_SIZE or not 1 <= width <= CANVAS_SIZE:
        raise ValueError(f"{name} dimensions must each lie in [1, {CANVAS_SIZE}]")
    if (
        grid.dtype == torch.bool
        or torch.is_floating_point(grid)
        or torch.is_complex(grid)
    ):
        raise TypeError(f"{name} must use an integer dtype")
    minimum = int(grid.min().item())
    maximum = int(grid.max().item())
    upper = PAD_CLASS if allow_pad else PAD_CLASS - 1
    if minimum < 0 or maximum > upper:
        raise ValueError(f"{name} values must lie in [0, {upper}]")
    return grid.to(dtype=torch.int64)


def _coerce_grids(grids: Any) -> list[Tensor]:
    if isinstance(grids, Tensor):
        if grids.ndim == 2:
            return [grids]
        if grids.ndim == 3:
            return list(grids.unbind(0))
        raise ValueError("grids tensor must have shape [H, W] or [B, H, W]")

    if not isinstance(grids, Sequence) or isinstance(grids, (str, bytes)):
        raise TypeError("grids must be a grid or a sequence of grids")
    try:
        tensor = torch.as_tensor(grids)
    except (TypeError, ValueError):
        tensor = None
    if tensor is not None:
        if tensor.ndim == 2:
            return [tensor]
        if tensor.ndim == 3:
            return list(tensor.unbind(0))
    if len(grids) == 0:
        raise ValueError("grids must not be empty")
    return [torch.as_tensor(grid) for grid in grids]


def pad_arc_grids(grids: Any) -> Tensor:
    """Top-left align ARC grids on a white-sentinel class canvas.

    Returns integer class IDs with shape ``[B, 30, 30]``.
    """

    items = _coerce_grids(grids)
    if not items:
        raise ValueError("grids must contain at least one grid")
    checked = [
        _validate_grid_tensor(grid, name=f"grids[{index}]", allow_pad=False)
        for index, grid in enumerate(items)
    ]
    device = checked[0].device
    if any(grid.device != device for grid in checked):
        raise ValueError("all grids must be on the same device")
    canvas = torch.full(
        (len(checked), CANVAS_SIZE, CANVAS_SIZE),
        PAD_CLASS,
        device=device,
        dtype=torch.int64,
    )
    for index, grid in enumerate(checked):
        height, width = grid.shape
        canvas[index, :height, :width] = grid
    return canvas


def render_arc_canvas(grids: Any, cell_size: int = CELL_SIZE) -> Tensor:
    """Render ARC grids as normalized VAE inputs ``[B, 3, 480, 480]``."""

    _require_cell_size(cell_size)
    canvas = pad_arc_grids(grids)
    palette = ARC_PALETTE_RGB.to(device=canvas.device)
    rgb = palette[canvas]
    rgb = rgb.repeat_interleave(cell_size, dim=1).repeat_interleave(
        cell_size, dim=2
    )
    image = rgb.permute(0, 3, 1, 2).to(dtype=torch.float32)
    image = image.div(127.5).sub(1.0)
    expected = (canvas.shape[0], 3, CANVAS_SIZE * cell_size, CANVAS_SIZE * cell_size)
    if image.shape != expected:
        raise AssertionError(f"rendered image has shape {tuple(image.shape)}, expected {expected}")
    if not torch.isfinite(image).all():
        raise FloatingPointError("rendered image contains non-finite values")
    return image


def decode_arc_canvas(canvas: Tensor) -> GridDecodeResult:
    """Infer a grid solely from a predicted 30x30 class canvas.

    All-PAD predictions and any PAD cell inside the inferred top-left extent are
    returned as invalid.  No dimensions or colors are repaired.
    """

    if not isinstance(canvas, Tensor):
        raise TypeError("canvas must be a torch.Tensor")
    if canvas.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(
            f"canvas must have shape [{CANVAS_SIZE}, {CANVAS_SIZE}], "
            f"got {tuple(canvas.shape)}"
        )
    canvas = _validate_grid_tensor(canvas, name="canvas", allow_pad=True)
    non_pad = canvas != PAD_CLASS
    if not bool(non_pad.any()):
        return GridDecodeResult(
            valid=False,
            grid=None,
            canvas=canvas,
            height=None,
            width=None,
            reason="all_pad",
        )

    occupied_rows = torch.nonzero(non_pad.any(dim=1), as_tuple=False).flatten()
    occupied_columns = torch.nonzero(non_pad.any(dim=0), as_tuple=False).flatten()
    height = int(occupied_rows[-1].item()) + 1
    width = int(occupied_columns[-1].item()) + 1
    region = canvas[:height, :width]
    if bool((region == PAD_CLASS).any()):
        return GridDecodeResult(
            valid=False,
            grid=None,
            canvas=canvas,
            height=height,
            width=width,
            reason="pad_inside_extent",
        )

    return GridDecodeResult(
        valid=True,
        grid=region.clone(),
        canvas=canvas,
        height=height,
        width=width,
        reason=None,
    )


def classify_arc_cells(image: Tensor, cell_size: int = CELL_SIZE) -> Tensor:
    """Classify each mean RGB cell against ten ARC colors plus white PAD.

    ``image`` must be one VAE-decoded image in ``[-1, 1]`` layout
    ``[3, 480, 480]``.  Values outside the nominal range are allowed, but every
    value must be finite; nearest-prototype classification remains explicit.
    """

    _require_cell_size(cell_size)
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("decode one image at a time; batch dimension must be 1")
        image = image[0]
    expected = (3, CANVAS_SIZE * cell_size, CANVAS_SIZE * cell_size)
    if image.shape != expected:
        raise ValueError(f"image must have shape {expected}, got {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError("image must use a floating-point dtype normalized for the VAE")
    if not torch.isfinite(image).all():
        raise ValueError("image contains non-finite values")

    cell_means = (
        image.float()
        .reshape(3, CANVAS_SIZE, cell_size, CANVAS_SIZE, cell_size)
        .mean(dim=(2, 4))
        .permute(1, 2, 0)
    )
    prototypes = (
        ARC_PALETTE_RGB.to(device=image.device, dtype=torch.float32)
        .div(127.5)
        .sub(1.0)
    )
    squared_distances = (
        cell_means.unsqueeze(-2) - prototypes.view(1, 1, -1, 3)
    ).square().sum(dim=-1)
    canvas = squared_distances.argmin(dim=-1)
    if canvas.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise AssertionError("classified canvas has an unexpected shape")
    return canvas


def decode_arc_image(
    image: Tensor,
    cell_size: int = CELL_SIZE,
) -> GridDecodeResult:
    """Classify and structurally decode one VAE output image."""

    return decode_arc_canvas(classify_arc_cells(image, cell_size=cell_size))
