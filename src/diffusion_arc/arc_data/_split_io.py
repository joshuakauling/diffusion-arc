"""Shared loading and path checks for configured ARC task splits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_task_splits(
    split_root: Path,
    task_split_files: tuple[tuple[str, Path], ...],
) -> dict[str, tuple[str, ...]]:
    """Load validated task IDs for every configured split."""

    task_ids_by_split: dict[str, tuple[str, ...]] = {}
    for split_name, relative_path in task_split_files:
        path = resolve_inside(
            split_root,
            relative_path,
            name=f"task_splits.{split_name}",
        )
        task_ids = load_string_array(
            path,
            name=str(path),
            allow_empty=split_name == "validation",
        )
        for task_id in task_ids:
            if "_" in task_id:
                raise ValueError(f"task ID must not contain '_': {task_id!r}")
            require_path_component(task_id, name=f"task ID in {path}")
        task_ids_by_split[split_name] = task_ids
    return task_ids_by_split


def validate_task_split_boundary(
    task_ids_by_split: Mapping[str, tuple[str, ...]],
) -> None:
    """Reject task IDs shared by training and validation."""

    overlap = set(task_ids_by_split["train"]) & set(task_ids_by_split["validation"])
    if overlap:
        raise ValueError(
            f"train and validation task splits overlap: {tuple(sorted(overlap))!r}"
        )


def load_string_array(
    path: Path,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Load a JSON array of unique non-empty strings."""

    return require_string_array(
        load_json(path),
        name=name,
        allow_empty=allow_empty,
    )


def require_string_array(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Validate a decoded JSON string array."""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{name} entries must be non-empty strings")

    strings = tuple(value)
    if len(set(strings)) != len(strings):
        raise ValueError(f"{name} must not contain duplicates")
    return strings


def require_path_component(value: Any, *, name: str) -> str:
    """Validate a non-empty string containing exactly one path component."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be one path component")
    return value


def resolve_inside(root: Path, relative_path: Path, *, name: str) -> Path:
    """Resolve a path while requiring it to remain below ``root``."""

    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} must resolve inside the split directory")
    return resolved


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "load_json",
    "load_string_array",
    "load_task_splits",
    "require_path_component",
    "require_string_array",
    "resolve_inside",
    "validate_task_split_boundary",
]
