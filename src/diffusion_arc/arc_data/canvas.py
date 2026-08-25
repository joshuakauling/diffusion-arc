"""Render ARC target grids as fixed RGB canvases for the frozen VAE."""

from __future__ import annotations

import torch
from torch import Tensor

from ._grid import ARC_GRID_SIZE, validate_arc_grid
from .palette import PAD_CLASS, TARGET_PALETTE_RGB
from .prompt import ArcRenderedTrainingSample
from .types import Grid


CANVAS_SIZE = ARC_GRID_SIZE
CELL_SIZE = 16


def render_grid_canvas(
    grid: Grid,
    *,
    canvas_size: int = CANVAS_SIZE,
) -> Tensor:
    """Render one top-left-aligned ARC grid on an explicit square canvas."""

    validate_arc_grid(grid)
    return _render_valid_grid_canvas(grid, canvas_size=canvas_size)


def _render_valid_grid_canvas(
    grid: Grid,
    *,
    canvas_size: int = CANVAS_SIZE,
) -> Tensor:
    """Render a grid whose ARC shape and palette were validated at ingestion."""

    if (
        isinstance(canvas_size, bool)
        or not isinstance(canvas_size, int)
        or canvas_size <= 0
        or canvas_size > CANVAS_SIZE
    ):
        raise ValueError(f"canvas_size must lie in [1, {CANVAS_SIZE}]")

    height = len(grid)
    width = len(grid[0])
    if height > canvas_size or width > canvas_size:
        raise ValueError(
            f"grid shape {(height, width)} exceeds canvas size {canvas_size}"
        )

    class_canvas = torch.full(
        (canvas_size, canvas_size),
        PAD_CLASS,
        dtype=torch.int64,
    )
    grid_tensor = torch.tensor(grid, dtype=torch.int64)
    class_canvas[:height, :width] = grid_tensor

    rgb = TARGET_PALETTE_RGB[class_canvas]
    rgb = rgb.repeat_interleave(CELL_SIZE, dim=0).repeat_interleave(
        CELL_SIZE,
        dim=1,
    )
    return rgb.permute(2, 0, 1).to(dtype=torch.float32).div(127.5).sub(1.0)


def render_training_target_canvases(
    sample: ArcRenderedTrainingSample,
    *,
    canvas_size: int = CANVAS_SIZE,
) -> Tensor:
    """Render the training targets in the same order as their prompt cuts."""

    if not sample.targets:
        raise ValueError("a rendered training sample must contain at least one target")

    return torch.stack(
        tuple(
            _render_valid_grid_canvas(
                target.target_grid,
                canvas_size=canvas_size,
            )
            for target in sample.targets
        )
    )


__all__ = [
    "CANVAS_SIZE",
    "CELL_SIZE",
    "render_grid_canvas",
    "render_training_target_canvases",
]
