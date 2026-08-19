"""Render ARC target grids as fixed RGB canvases for the frozen VAE."""

from __future__ import annotations

import torch
from torch import Tensor

from ._grid import ARC_GRID_SIZE, validate_arc_grid
from .prompt import ArcRenderedTrainingSample
from .types import Grid


CANVAS_SIZE = ARC_GRID_SIZE
CELL_SIZE = 16
PAD_CLASS = 10

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


def render_grid_canvas(grid: Grid) -> Tensor:
    """Render one top-left-aligned ARC grid as ``[3, 480, 480]`` in ``[-1, 1]``."""

    validate_arc_grid(grid)
    class_canvas = torch.full(
        (CANVAS_SIZE, CANVAS_SIZE),
        PAD_CLASS,
        dtype=torch.int64,
    )
    grid_tensor = torch.tensor(grid, dtype=torch.int64)
    height, width = grid_tensor.shape
    class_canvas[:height, :width] = grid_tensor

    rgb = ARC_PALETTE_RGB[class_canvas]
    rgb = rgb.repeat_interleave(CELL_SIZE, dim=0).repeat_interleave(
        CELL_SIZE,
        dim=1,
    )
    return rgb.permute(2, 0, 1).to(dtype=torch.float32).div(127.5).sub(1.0)


def render_training_target_canvases(
    sample: ArcRenderedTrainingSample,
) -> Tensor:
    """Render the training targets in the same order as their prompt cuts."""

    if not isinstance(sample, ArcRenderedTrainingSample):
        raise TypeError("sample must be an ArcRenderedTrainingSample")
    if not sample.targets:
        raise ValueError("a rendered training sample must contain at least one target")

    return torch.stack(
        tuple(render_grid_canvas(target.target_grid) for target in sample.targets)
    )


__all__ = [
    "ARC_PALETTE_RGB",
    "CANVAS_SIZE",
    "CELL_SIZE",
    "PAD_CLASS",
    "render_grid_canvas",
    "render_training_target_canvases",
]
