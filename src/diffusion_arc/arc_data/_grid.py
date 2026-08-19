"""Shared validation for in-memory ARC grids."""

from __future__ import annotations

from .types import Grid


ARC_GRID_SIZE = 30


def validate_arc_grid(grid: Grid) -> None:
    """Raise when an in-memory grid violates the ARC grid contract."""

    if not grid or any(not row for row in grid):
        raise ValueError("grid must contain at least one non-empty row")
    if len(grid) > ARC_GRID_SIZE:
        raise ValueError(f"grid height must not exceed {ARC_GRID_SIZE}")

    width = len(grid[0])
    if width > ARC_GRID_SIZE:
        raise ValueError(f"grid width must not exceed {ARC_GRID_SIZE}")
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must all have the same length")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 9
        for row in grid
        for value in row
    ):
        raise ValueError("grid values must be integers from 0 through 9")


__all__ = ["ARC_GRID_SIZE", "validate_arc_grid"]
