"""Immutable, model-independent types for ARC tasks and grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


Grid: TypeAlias = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ArcPair:
    """One labelled ARC input/output pair."""

    input_grid: Grid
    output_grid: Grid


@dataclass(frozen=True)
class ArcTrainingTask:
    """All labelled pairs belonging to one task in the dataset training split."""

    task_id: str
    pairs: tuple[ArcPair, ...]


@dataclass(frozen=True)
class ArcQuery:
    """One evaluation input without access to its target output."""

    query_index: int
    input_grid: Grid


@dataclass(frozen=True)
class ArcEvaluationTask:
    """Leakage-safe evaluation input containing demonstrations and queries only."""

    task_id: str
    demonstrations: tuple[ArcPair, ...]
    queries: tuple[ArcQuery, ...]


@dataclass(frozen=True)
class ArcTarget:
    """The held-out output associated with one evaluation query."""

    query_index: int
    output_grid: Grid


@dataclass(frozen=True)
class ArcEvaluationTargets:
    """Targets kept outside the model-facing evaluation task."""

    task_id: str
    targets: tuple[ArcTarget, ...]


__all__ = [
    "ArcEvaluationTargets",
    "ArcEvaluationTask",
    "ArcPair",
    "ArcQuery",
    "ArcTarget",
    "ArcTrainingTask",
    "Grid",
]
