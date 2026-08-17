"""Sampling, identities, and application of ARC data augmentations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from random import Random

from .types import ArcEvaluationTask, ArcPair, ArcTrainingTask, Grid


_COLOR_MAPPING_PATTERN = re.compile(r"[0-9]{10}")
_PAIR_INDICES_PATTERN = re.compile(r"[0-9]+(?:,[0-9]+)*")


class ArcGeometry(str, Enum):
    """The eight symmetries of a square ARC grid."""

    IDENTITY = "ID"
    ROTATE_90 = "R90"
    ROTATE_180 = "R180"
    ROTATE_270 = "R270"
    FLIP_HORIZONTAL = "FH"
    FLIP_VERTICAL = "FV"
    FLIP_MAIN_DIAGONAL = "FD1"
    FLIP_ANTI_DIAGONAL = "FD2"


@dataclass(frozen=True)
class ArcAugmentationSpec:
    """Task-independent choices that define one augmented ARC view."""

    geometry: ArcGeometry
    color_mapping: tuple[int, ...]
    pair_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ArcGeometry):
            raise TypeError("geometry must be an ArcGeometry")

        if not isinstance(self.color_mapping, tuple):
            raise TypeError("color_mapping must be a tuple")
        if len(self.color_mapping) != 10:
            raise ValueError("color_mapping must contain exactly 10 colors")
        if any(
            not isinstance(color, int) or isinstance(color, bool)
            for color in self.color_mapping
        ):
            raise TypeError("color_mapping entries must be integers")
        if set(self.color_mapping) != set(range(10)):
            raise ValueError(
                "color_mapping must be a permutation of colors 0 through 9"
            )

        if not isinstance(self.pair_indices, tuple):
            raise TypeError("pair_indices must be a tuple")
        if not self.pair_indices:
            raise ValueError("pair_indices must contain at least one index")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in self.pair_indices
        ):
            raise TypeError("pair_indices entries must be integers")
        if any(index < 0 for index in self.pair_indices):
            raise ValueError("pair_indices entries must be non-negative")
        if len(set(self.pair_indices)) != len(self.pair_indices):
            raise ValueError("pair_indices entries must be unique")


@dataclass(frozen=True)
class ArcAugmentationKey:
    """The globally stable identity of one augmented ARC task view."""

    task_id: str
    augmentation: ArcAugmentationSpec

    def __post_init__(self) -> None:
        _validate_task_id(self.task_id)
        if not isinstance(self.augmentation, ArcAugmentationSpec):
            raise TypeError("augmentation must be an ArcAugmentationSpec")

    def __str__(self) -> str:
        return format_augmentation_key(self)


@dataclass(frozen=True)
class ArcAugmentedTrainingSample:
    """One ordered and transformed training task augmentation."""

    key: ArcAugmentationKey
    pairs: tuple[ArcPair, ...]


def make_augmentation_key(
    task_id: str,
    augmentation: ArcAugmentationSpec,
) -> str:
    """Build the canonical serialized key for an augmentation."""

    return format_augmentation_key(
        ArcAugmentationKey(task_id=task_id, augmentation=augmentation)
    )


def format_augmentation_key(key: ArcAugmentationKey) -> str:
    """Serialize an augmentation key in its canonical, reversible form."""

    if not isinstance(key, ArcAugmentationKey):
        raise TypeError("key must be an ArcAugmentationKey")

    augmentation = key.augmentation
    color_mapping = "".join(str(color) for color in augmentation.color_mapping)
    pair_indices = ",".join(str(index) for index in augmentation.pair_indices)
    return f"{key.task_id}_{augmentation.geometry.value}_{color_mapping}_{pair_indices}"


def parse_augmentation_key(value: str) -> ArcAugmentationKey:
    """Parse a canonical serialized augmentation key."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")

    parts = value.split("_")
    if len(parts) != 4:
        raise ValueError("augmentation key must contain exactly four fields")

    task_id, geometry_value, color_mapping_value, pair_indices_value = parts
    _validate_task_id(task_id)

    try:
        geometry = ArcGeometry(geometry_value)
    except ValueError as error:
        raise ValueError(f"unknown ARC geometry: {geometry_value!r}") from error

    if _COLOR_MAPPING_PATTERN.fullmatch(color_mapping_value) is None:
        raise ValueError("color mapping must contain exactly 10 ASCII digits")
    color_mapping = tuple(int(color) for color in color_mapping_value)

    if _PAIR_INDICES_PATTERN.fullmatch(pair_indices_value) is None:
        raise ValueError("pair indices must be comma-separated non-negative integers")
    pair_index_parts = pair_indices_value.split(",")
    if any(len(index) > 1 and index.startswith("0") for index in pair_index_parts):
        raise ValueError("pair indices must use canonical decimal notation")
    pair_indices = tuple(int(index) for index in pair_index_parts)

    return ArcAugmentationKey(
        task_id=task_id,
        augmentation=ArcAugmentationSpec(
            geometry=geometry,
            color_mapping=color_mapping,
            pair_indices=pair_indices,
        ),
    )


def validate_augmentation_for_task(
    key: ArcAugmentationKey,
    task: ArcTrainingTask | ArcEvaluationTask,
) -> None:
    """Ensure that an augmentation references pairs available in its task."""

    if not isinstance(key, ArcAugmentationKey):
        raise TypeError("key must be an ArcAugmentationKey")
    if not isinstance(task, (ArcTrainingTask, ArcEvaluationTask)):
        raise TypeError("task must be an ArcTrainingTask or ArcEvaluationTask")
    if key.task_id != task.task_id:
        raise ValueError(
            f"augmentation task_id {key.task_id!r} does not match task {task.task_id!r}"
        )

    if isinstance(task, ArcTrainingTask):
        pair_count = len(task.pairs)
    else:
        pair_count = len(task.demonstrations)

    invalid_indices = tuple(
        index for index in key.augmentation.pair_indices if index >= pair_count
    )
    if invalid_indices:
        raise ValueError(
            f"pair indices {invalid_indices!r} are out of range for {pair_count} pairs"
        )


def sample_training_augmentation(
    task: ArcTrainingTask,
    *,
    augmentation_index: int,
    max_pairs: int,
    max_cells: int,
    rng: Random,
) -> ArcAugmentationSpec:
    """Sample one bounded, reproducible augmentation of a training task."""

    _validate_training_sampling_limits(task, max_pairs=max_pairs, max_cells=max_cells)
    if augmentation_index < 0:
        raise ValueError("augmentation_index must be non-negative")

    candidate_indices = list(range(len(task.pairs)))
    rng.shuffle(candidate_indices)
    selected_indices = _select_pair_indices(
        task,
        candidate_indices=candidate_indices,
        max_pairs=max_pairs,
        max_cells=max_cells,
    )

    color_mapping = list(range(10))
    rng.shuffle(color_mapping)
    geometries = tuple(ArcGeometry)

    return ArcAugmentationSpec(
        geometry=geometries[augmentation_index % len(geometries)],
        color_mapping=tuple(color_mapping),
        pair_indices=tuple(selected_indices),
    )


def make_identity_training_augmentation(
    task: ArcTrainingTask,
    *,
    max_pairs: int,
    max_cells: int,
) -> ArcAugmentationSpec:
    """Build the untransformed, original-order view that fits the limits."""

    _validate_training_sampling_limits(task, max_pairs=max_pairs, max_cells=max_cells)
    selected_indices = _select_pair_indices(
        task,
        candidate_indices=range(len(task.pairs)),
        max_pairs=max_pairs,
        max_cells=max_cells,
    )
    return ArcAugmentationSpec(
        geometry=ArcGeometry.IDENTITY,
        color_mapping=tuple(range(10)),
        pair_indices=tuple(selected_indices),
    )


def augment_grid(grid: Grid, augmentation: ArcAugmentationSpec) -> Grid:
    """Apply an augmentation's color mapping and geometry to one grid."""

    if not isinstance(augmentation, ArcAugmentationSpec):
        raise TypeError("augmentation must be an ArcAugmentationSpec")
    _validate_grid(grid)

    transformed = tuple(
        tuple(augmentation.color_mapping[value] for value in row) for row in grid
    )

    if augmentation.geometry is ArcGeometry.IDENTITY:
        return transformed
    if augmentation.geometry is ArcGeometry.ROTATE_90:
        return _rotate_clockwise(transformed)
    if augmentation.geometry is ArcGeometry.ROTATE_180:
        return _rotate_clockwise(_rotate_clockwise(transformed))
    if augmentation.geometry is ArcGeometry.ROTATE_270:
        return _rotate_clockwise(
            _rotate_clockwise(_rotate_clockwise(transformed))
        )
    if augmentation.geometry is ArcGeometry.FLIP_HORIZONTAL:
        return tuple(reversed(transformed))
    if augmentation.geometry is ArcGeometry.FLIP_VERTICAL:
        return tuple(tuple(reversed(row)) for row in transformed)

    transposed = tuple(tuple(row) for row in zip(*transformed))
    if augmentation.geometry is ArcGeometry.FLIP_MAIN_DIAGONAL:
        return transposed
    if augmentation.geometry is ArcGeometry.FLIP_ANTI_DIAGONAL:
        return tuple(tuple(reversed(row)) for row in reversed(transposed))

    raise AssertionError(f"unhandled ARC geometry: {augmentation.geometry!r}")


def apply_training_augmentation(
    task: ArcTrainingTask,
    augmentation: ArcAugmentationSpec,
) -> ArcAugmentedTrainingSample:
    """Select, order, and transform the pairs of one training augmentation."""

    if not isinstance(task, ArcTrainingTask):
        raise TypeError("task must be an ArcTrainingTask")

    key = ArcAugmentationKey(task_id=task.task_id, augmentation=augmentation)
    validate_augmentation_for_task(key, task)
    if len(augmentation.pair_indices) < 2:
        raise ValueError("a training augmentation must contain at least two pairs")

    pairs = tuple(
        ArcPair(
            input_grid=augment_grid(task.pairs[index].input_grid, augmentation),
            output_grid=augment_grid(task.pairs[index].output_grid, augmentation),
        )
        for index in augmentation.pair_indices
    )
    return ArcAugmentedTrainingSample(key=key, pairs=pairs)


def _rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*reversed(grid)))


def _grid_cell_count(grid: Grid) -> int:
    return sum(len(row) for row in grid)


def _validate_training_sampling_limits(
    task: ArcTrainingTask,
    *,
    max_pairs: int,
    max_cells: int,
) -> None:
    if not isinstance(task, ArcTrainingTask):
        raise TypeError("task must be an ArcTrainingTask")
    if max_pairs < 2:
        raise ValueError("max_pairs must allow at least two pairs")
    if max_cells <= 0:
        raise ValueError("max_cells must be positive")


def _select_pair_indices(
    task: ArcTrainingTask,
    *,
    candidate_indices: range | list[int],
    max_pairs: int,
    max_cells: int,
) -> list[int]:
    selected_indices: list[int] = []
    selected_cells = 0
    for pair_index in candidate_indices:
        if len(selected_indices) == max_pairs:
            break

        pair = task.pairs[pair_index]
        pair_cells = _grid_cell_count(pair.input_grid) + _grid_cell_count(
            pair.output_grid
        )
        if selected_cells + pair_cells > max_cells:
            continue

        selected_indices.append(pair_index)
        selected_cells += pair_cells

    if len(selected_indices) < 2:
        raise ValueError(
            "fewer than two task pairs fit within the configured sampling limits"
        )
    return selected_indices


def _validate_grid(grid: Grid) -> None:
    if not grid or any(not row for row in grid):
        raise ValueError("grid must contain at least one non-empty row")
    width = len(grid[0])
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


def _validate_task_id(task_id: str) -> None:
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")
    if not task_id:
        raise ValueError("task_id must not be empty")
    if "_" in task_id:
        raise ValueError("task_id must not contain the key separator '_'")


__all__ = [
    "ArcAugmentationKey",
    "ArcAugmentationSpec",
    "ArcAugmentedTrainingSample",
    "ArcGeometry",
    "apply_training_augmentation",
    "augment_grid",
    "format_augmentation_key",
    "make_augmentation_key",
    "make_identity_training_augmentation",
    "parse_augmentation_key",
    "sample_training_augmentation",
    "validate_augmentation_for_task",
]
