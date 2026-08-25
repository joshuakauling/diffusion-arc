"""Deterministic RGB-to-grid decoding for generated ARC canvases."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .canvas import CANVAS_SIZE, CELL_SIZE
from .palette import PAD_CLASS, TARGET_PALETTE_RGB
from .types import Grid


@dataclass(frozen=True)
class ArcGridPrediction:
    """One strict PAD-delimited grid prediction and its classified canvas."""

    class_canvas: Tensor
    grid: Grid | None
    height: int | None
    width: int | None
    valid: bool
    invalid_reason: str | None


def classify_arc_cells(rgb_images: Tensor) -> Tensor:
    """Classify uint8 RGB cell means against the ten ARC colors and PAD."""

    if not isinstance(rgb_images, Tensor):
        raise TypeError("rgb_images must be a torch.Tensor")
    image_size = CANVAS_SIZE * CELL_SIZE
    expected_suffix = (3, image_size, image_size)
    if rgb_images.ndim != 4 or tuple(rgb_images.shape[1:]) != expected_suffix:
        raise ValueError(
            f"rgb_images must have shape [B, 3, {image_size}, {image_size}]"
        )
    if rgb_images.shape[0] == 0:
        raise ValueError("rgb_images must contain at least one image")
    if rgb_images.dtype != torch.uint8:
        raise TypeError("rgb_images must have dtype torch.uint8")

    cell_means = rgb_images.to(dtype=torch.float32).reshape(
        rgb_images.shape[0],
        3,
        CANVAS_SIZE,
        CELL_SIZE,
        CANVAS_SIZE,
        CELL_SIZE,
    ).mean(dim=(3, 5)).permute(0, 2, 3, 1)
    prototypes = TARGET_PALETTE_RGB.to(
        device=rgb_images.device,
        dtype=torch.float32,
    )
    squared_distances = (
        cell_means.unsqueeze(3) - prototypes.reshape(1, 1, 1, -1, 3)
    ).square().sum(dim=-1)
    return squared_distances.argmin(dim=-1)


def decode_arc_grid(class_canvas: Tensor) -> ArcGridPrediction:
    """Infer one top-left grid solely from its classified PAD boundary."""

    _validate_class_canvas(class_canvas)
    cpu_canvas = class_canvas.detach().to(device="cpu", dtype=torch.int64).clone()
    non_pad_positions = torch.nonzero(cpu_canvas != PAD_CLASS, as_tuple=False)
    if non_pad_positions.numel() == 0:
        return ArcGridPrediction(
            class_canvas=cpu_canvas,
            grid=None,
            height=None,
            width=None,
            valid=False,
            invalid_reason="all_pad",
        )

    height = int(non_pad_positions[:, 0].max().item()) + 1
    width = int(non_pad_positions[:, 1].max().item()) + 1
    inferred_rectangle = cpu_canvas[:height, :width]
    if bool((inferred_rectangle == PAD_CLASS).any().item()):
        return ArcGridPrediction(
            class_canvas=cpu_canvas,
            grid=None,
            height=height,
            width=width,
            valid=False,
            invalid_reason="pad_inside_extent",
        )

    grid = tuple(
        tuple(int(value) for value in row)
        for row in inferred_rectangle.tolist()
    )
    return ArcGridPrediction(
        class_canvas=cpu_canvas,
        grid=grid,
        height=height,
        width=width,
        valid=True,
        invalid_reason=None,
    )


def decode_arc_grids(class_canvases: Tensor) -> tuple[ArcGridPrediction, ...]:
    """Decode a non-empty batch of classified 30x30 ARC canvases."""

    if not isinstance(class_canvases, Tensor):
        raise TypeError("class_canvases must be a torch.Tensor")
    if class_canvases.ndim != 3 or tuple(class_canvases.shape[1:]) != (
        CANVAS_SIZE,
        CANVAS_SIZE,
    ):
        raise ValueError("class_canvases must have shape [B, 30, 30]")
    if class_canvases.shape[0] == 0:
        raise ValueError("class_canvases must contain at least one canvas")
    return tuple(decode_arc_grid(canvas) for canvas in class_canvases)


def _validate_class_canvas(class_canvas: Tensor) -> None:
    if not isinstance(class_canvas, Tensor):
        raise TypeError("class_canvas must be a torch.Tensor")
    if tuple(class_canvas.shape) != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError("class_canvas must have shape [30, 30]")
    if class_canvas.dtype != torch.int64:
        raise TypeError("class_canvas must have dtype torch.int64")
    if bool(((class_canvas < 0) | (class_canvas > PAD_CLASS)).any().item()):
        raise ValueError(f"class_canvas values must lie in [0, {PAD_CLASS}]")


__all__ = [
    "ArcGridPrediction",
    "classify_arc_cells",
    "decode_arc_grid",
    "decode_arc_grids",
]
