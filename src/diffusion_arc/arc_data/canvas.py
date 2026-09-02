"""Convert between ARC grids and fixed RGB canvases."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._grid import ARC_GRID_SIZE, validate_arc_grid
from .palette import PAD_CLASS, TARGET_PALETTE_RGB
from .prompt import ArcRenderedTrainingSample
from .types import Grid


CANVAS_SIZE = ARC_GRID_SIZE
CELL_SIZE = 16


@dataclass(frozen=True)
class ArcGridPrediction:
    """One strict PAD-delimited grid prediction and its classified canvas."""

    class_canvas: Tensor
    grid: Grid | None
    height: int | None
    width: int | None
    valid: bool
    invalid_reason: str | None


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


def classify_arc_cells(
    rgb_images: Tensor,
    *,
    canvas_size: int = CANVAS_SIZE,
    cell_size: int = CELL_SIZE,
    include_pad: bool = True,
) -> Tensor:
    """Classify uint8 RGB cell means against the ten ARC colors and PAD."""

    if not isinstance(rgb_images, Tensor):
        raise TypeError("rgb_images must be a torch.Tensor")
    _validate_decode_geometry(canvas_size=canvas_size, cell_size=cell_size)
    if not isinstance(include_pad, bool):
        raise TypeError("include_pad must be a boolean")
    image_size = canvas_size * cell_size
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
        canvas_size,
        cell_size,
        canvas_size,
        cell_size,
    ).mean(dim=(3, 5)).permute(0, 2, 3, 1)
    prototype_count = PAD_CLASS + 1 if include_pad else PAD_CLASS
    prototypes = TARGET_PALETTE_RGB[:prototype_count].to(
        device=rgb_images.device,
        dtype=torch.float32,
    )
    squared_distances = (
        cell_means.unsqueeze(3) - prototypes.reshape(1, 1, 1, -1, 3)
    ).square().sum(dim=-1)
    return squared_distances.argmin(dim=-1)


def decode_arc_grid(
    class_canvas: Tensor,
    *,
    canvas_size: int = CANVAS_SIZE,
) -> ArcGridPrediction:
    """Infer one top-left grid solely from its classified PAD boundary."""

    _validate_class_canvas(class_canvas, canvas_size=canvas_size)
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


def decode_arc_grids(
    class_canvases: Tensor,
    *,
    canvas_size: int = CANVAS_SIZE,
) -> tuple[ArcGridPrediction, ...]:
    """Decode a non-empty batch using strict PAD-boundary inference."""

    if not isinstance(class_canvases, Tensor):
        raise TypeError("class_canvases must be a torch.Tensor")
    if class_canvases.ndim != 3 or tuple(class_canvases.shape[1:]) != (
        canvas_size,
        canvas_size,
    ):
        raise ValueError(
            f"class_canvases must have shape [B, {canvas_size}, {canvas_size}]"
        )
    if class_canvases.shape[0] == 0:
        raise ValueError("class_canvases must contain at least one canvas")
    return tuple(
        decode_arc_grid(canvas, canvas_size=canvas_size)
        for canvas in class_canvases
    )


def decode_fixed_arc_grid(
    class_canvas: Tensor,
    *,
    canvas_size: int,
) -> ArcGridPrediction:
    """Decode a known full square grid without consulting a PAD boundary."""

    _validate_class_canvas(class_canvas, canvas_size=canvas_size)
    cpu_canvas = class_canvas.detach().to(device="cpu", dtype=torch.int64).clone()
    if bool((cpu_canvas == PAD_CLASS).any().item()):
        return ArcGridPrediction(
            class_canvas=cpu_canvas,
            grid=None,
            height=canvas_size,
            width=canvas_size,
            valid=False,
            invalid_reason="pad_in_fixed_grid",
        )
    grid = tuple(tuple(int(value) for value in row) for row in cpu_canvas.tolist())
    return ArcGridPrediction(
        class_canvas=cpu_canvas,
        grid=grid,
        height=canvas_size,
        width=canvas_size,
        valid=True,
        invalid_reason=None,
    )


def decode_fixed_arc_grids(
    class_canvases: Tensor,
    *,
    canvas_size: int,
) -> tuple[ArcGridPrediction, ...]:
    """Decode a non-empty batch of known full square ARC grids."""

    if not isinstance(class_canvases, Tensor):
        raise TypeError("class_canvases must be a torch.Tensor")
    if class_canvases.ndim != 3 or tuple(class_canvases.shape[1:]) != (
        canvas_size,
        canvas_size,
    ):
        raise ValueError(
            f"class_canvases must have shape [B, {canvas_size}, {canvas_size}]"
        )
    if class_canvases.shape[0] == 0:
        raise ValueError("class_canvases must contain at least one canvas")
    return tuple(
        decode_fixed_arc_grid(canvas, canvas_size=canvas_size)
        for canvas in class_canvases
    )


def _validate_decode_geometry(*, canvas_size: int, cell_size: int) -> None:
    for name, value in (("canvas_size", canvas_size), ("cell_size", cell_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _validate_class_canvas(
    class_canvas: Tensor,
    *,
    canvas_size: int,
) -> None:
    if not isinstance(class_canvas, Tensor):
        raise TypeError("class_canvas must be a torch.Tensor")
    _validate_decode_geometry(canvas_size=canvas_size, cell_size=CELL_SIZE)
    if tuple(class_canvas.shape) != (canvas_size, canvas_size):
        raise ValueError(
            f"class_canvas must have shape [{canvas_size}, {canvas_size}]"
        )
    if class_canvas.dtype != torch.int64:
        raise TypeError("class_canvas must have dtype torch.int64")
    if bool(((class_canvas < 0) | (class_canvas > PAD_CLASS)).any().item()):
        raise ValueError(f"class_canvas values must lie in [0, {PAD_CLASS}]")


__all__ = [
    "ArcGridPrediction",
    "CANVAS_SIZE",
    "CELL_SIZE",
    "classify_arc_cells",
    "decode_arc_grid",
    "decode_arc_grids",
    "decode_fixed_arc_grid",
    "decode_fixed_arc_grids",
    "render_grid_canvas",
    "render_training_target_canvases",
]
