"""Strict loading of one-file-per-task ARC datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._grid import ARC_GRID_SIZE
from .types import (
    ArcEvaluationTargets,
    ArcEvaluationTask,
    ArcPair,
    ArcQuery,
    ArcTarget,
    ArcTrainingTask,
    Grid,
)


def load_training_task(path: str | Path) -> ArcTrainingTask:
    """Load all labelled ``train`` and ``test`` pairs as one training task."""

    task_path, payload = _load_task_file(path)
    train = _require_examples(payload, "train", task_path)
    test = _require_examples(payload, "test", task_path)
    pairs = tuple(
        _load_pair(example, name=f"{split}[{index}]")
        for split, examples in (("train", train), ("test", test))
        for index, example in enumerate(examples)
    )
    return ArcTrainingTask(task_id=task_path.stem, pairs=pairs)


def load_evaluation_task(path: str | Path) -> ArcEvaluationTask:
    """Load demonstrations and query inputs without exposing query outputs."""

    task_path, payload = _load_task_file(path)
    train = _require_examples(payload, "train", task_path)
    test = _require_examples(payload, "test", task_path)
    demonstrations = tuple(
        _load_pair(example, name=f"train[{index}]")
        for index, example in enumerate(train)
    )
    queries = tuple(
        ArcQuery(
            query_index=index,
            input_grid=_load_grid_field(example, "input", name=f"test[{index}]"),
        )
        for index, example in enumerate(test)
    )
    return ArcEvaluationTask(
        task_id=task_path.stem,
        demonstrations=demonstrations,
        queries=queries,
    )


def load_evaluation_targets(path: str | Path) -> ArcEvaluationTargets:
    """Load held-out outputs separately from the model-facing evaluation task."""

    task_path, payload = _load_task_file(path)
    test = _require_examples(payload, "test", task_path)
    targets = tuple(
        ArcTarget(
            query_index=index,
            output_grid=_load_grid_field(example, "output", name=f"test[{index}]"),
        )
        for index, example in enumerate(test)
    )
    return ArcEvaluationTargets(task_id=task_path.stem, targets=targets)


def align_evaluation_targets(
    task: ArcEvaluationTask,
    targets: ArcEvaluationTargets,
) -> tuple[ArcTarget, ...]:
    """Validate and order held-out targets to match an evaluation task's queries."""

    if task.task_id != targets.task_id:
        raise ValueError(
            f"task IDs do not match: {task.task_id!r} != {targets.task_id!r}"
        )

    query_indices = tuple(query.query_index for query in task.queries)
    target_indices = tuple(target.query_index for target in targets.targets)
    if len(set(query_indices)) != len(query_indices):
        raise ValueError("evaluation task contains duplicate query indices")
    if len(set(target_indices)) != len(target_indices):
        raise ValueError("evaluation targets contain duplicate query indices")

    query_index_set = set(query_indices)
    target_index_set = set(target_indices)
    if query_index_set != target_index_set:
        missing = sorted(query_index_set - target_index_set)
        unexpected = sorted(target_index_set - query_index_set)
        raise ValueError(
            "evaluation target indices do not match queries: "
            f"missing={missing}, unexpected={unexpected}"
        )

    targets_by_index = {
        target.query_index: target for target in targets.targets
    }
    return tuple(targets_by_index[index] for index in query_indices)


def _load_task_file(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"task file must contain a JSON object: {task_path}")
    return task_path, payload


def _require_examples(
    payload: Mapping[str, Any],
    field: str,
    task_path: Path,
) -> list[Any]:
    examples = payload.get(field)
    if not isinstance(examples, list):
        raise TypeError(f"{task_path}: {field!r} must be a list")
    return examples


def _load_pair(example: Any, *, name: str) -> ArcPair:
    return ArcPair(
        input_grid=_load_grid_field(example, "input", name=name),
        output_grid=_load_grid_field(example, "output", name=name),
    )


def _load_grid_field(example: Any, field: str, *, name: str) -> Grid:
    if not isinstance(example, Mapping):
        raise TypeError(f"{name} must be an object")
    if field not in example:
        raise ValueError(f"{name} must contain {field!r}")
    return _load_grid(example[field], name=f"{name}.{field}")


def _load_grid(value: Any, *, name: str) -> Grid:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a non-empty list of rows")
    if len(value) > ARC_GRID_SIZE:
        raise ValueError(f"{name} height must not exceed {ARC_GRID_SIZE}")
    if any(not isinstance(row, list) or not row for row in value):
        raise TypeError(f"{name} rows must be non-empty lists")

    width = len(value[0])
    if width > ARC_GRID_SIZE:
        raise ValueError(f"{name} width must not exceed {ARC_GRID_SIZE}")
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular")
    if any(
        not isinstance(cell, int)
        or isinstance(cell, bool)
        or not 0 <= cell <= 9
        for row in value
        for cell in row
    ):
        raise ValueError(f"{name} cells must be integers from 0 through 9")
    return tuple(tuple(row) for row in value)


__all__ = [
    "align_evaluation_targets",
    "load_evaluation_targets",
    "load_evaluation_task",
    "load_training_task",
]
