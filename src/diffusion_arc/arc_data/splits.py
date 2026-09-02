"""Reproducible generation of augmented-key files for ARC training views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any

from ._split_io import (
    load_json,
    load_task_splits,
    require_path_component,
    require_string_array,
    resolve_inside,
    validate_task_split_boundary,
)
from ._split_layout import resolve_split_layout
from .augmentation import (
    make_augmentation_key,
    make_identity_training_augmentation,
    sample_training_augmentation,
)
from .palette import TARGET_PALETTE_ID
from .task_loading import load_training_task
from .types import ArcTrainingTask


ARC_SPLIT_SCHEMA_VERSION = 1
ARC_AUGMENTATION_GENERATOR_VERSION = 1


class ArcTrainingViewMode(str, Enum):
    """Supported ways to construct a training view."""

    IDENTITY = "identity"
    SAMPLED = "sampled"


@dataclass(frozen=True)
class ArcTrainingViewConfig:
    """Configuration for one generated augmented-key file."""

    name: str
    task_split: str
    output: Path
    mode: ArcTrainingViewMode
    exclude: tuple[str, ...]
    seed: int | None
    augmentations_per_task: int


@dataclass(frozen=True)
class ArcSplitConfig:
    """Validated configuration for one task split and its training views."""

    dataset: str
    source_split: str
    palette: str
    max_pairs: int
    max_cells: int
    max_tokens: int
    task_split_files: tuple[tuple[str, Path], ...]
    views: tuple[ArcTrainingViewConfig, ...]


def load_arc_split_config(path: str | Path) -> ArcSplitConfig:
    """Load and strictly validate one ``config.json`` split definition."""

    config_path = Path(path)
    payload = _require_mapping(load_json(config_path), name="config")
    _require_exact_fields(
        payload,
        {
            "schema_version",
            "generator_version",
            "source",
            "palette",
            "limits",
            "task_splits",
            "views",
        },
        name="config",
    )
    schema_version = _require_int(
        payload["schema_version"],
        name="schema_version",
    )
    if schema_version != ARC_SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version: {schema_version!r}"
        )
    generator_version = _require_int(
        payload["generator_version"],
        name="generator_version",
    )
    if generator_version != ARC_AUGMENTATION_GENERATOR_VERSION:
        raise ValueError(
            f"unsupported generator_version: {generator_version!r}"
        )

    source = _require_mapping(payload["source"], name="source")
    _require_exact_fields(source, {"dataset", "split"}, name="source")
    dataset = require_path_component(source["dataset"], name="source.dataset")
    source_split = require_path_component(source["split"], name="source.split")

    palette = _require_string(payload["palette"], name="palette")
    if palette != TARGET_PALETTE_ID:
        raise ValueError(f"unsupported palette: {palette!r}")

    limits = _require_mapping(payload["limits"], name="limits")
    _require_exact_fields(
        limits,
        {"max_pairs", "max_cells", "max_tokens"},
        name="limits",
    )
    max_pairs = _require_positive_int(limits["max_pairs"], name="limits.max_pairs")
    if max_pairs < 2:
        raise ValueError("limits.max_pairs must allow at least two pairs")
    max_cells = _require_positive_int(limits["max_cells"], name="limits.max_cells")
    max_tokens = _require_positive_int(limits["max_tokens"], name="limits.max_tokens")

    task_splits_payload = _require_mapping(payload["task_splits"], name="task_splits")
    _require_exact_fields(
        task_splits_payload,
        {"train", "validation"},
        name="task_splits",
    )
    task_split_files = tuple(
        (
            name,
            _require_relative_path(value, name=f"task_splits.{name}"),
        )
        for name, value in task_splits_payload.items()
    )

    views_payload = _require_mapping(payload["views"], name="views")
    if not views_payload:
        raise ValueError("views must contain at least one training view")
    view_names = tuple(views_payload)
    views = tuple(
        _load_view_config(name, value, known_view_names=set(view_names))
        for name, value in views_payload.items()
    )
    output_paths = tuple(view.output for view in views)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("views must use distinct output paths")

    return ArcSplitConfig(
        dataset=dataset,
        source_split=source_split,
        palette=palette,
        max_pairs=max_pairs,
        max_cells=max_cells,
        max_tokens=max_tokens,
        task_split_files=task_split_files,
        views=views,
    )


def generate_augmented_key_files(
    split_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Generate every configured training view and write its JSON key array."""

    unresolved_split_root = Path(split_directory).resolve()
    config = load_arc_split_config(unresolved_split_root / "config.json")
    split_root, source_root = resolve_split_layout(
        unresolved_split_root,
        dataset=config.dataset,
        source_split=config.source_split,
    )
    task_ids_by_split = load_task_splits(split_root, config.task_split_files)
    validate_task_split_boundary(task_ids_by_split)

    tasks = _load_source_tasks(source_root, task_ids_by_split)

    views_by_name = {view.name: view for view in config.views}
    generated: dict[str, tuple[str, ...]] = {}
    active: set[str] = set()

    def generate_view(view_name: str) -> tuple[str, ...]:
        if view_name in generated:
            return generated[view_name]
        if view_name in active:
            raise ValueError("views contain a cyclic exclusion dependency")

        active.add(view_name)
        view = views_by_name[view_name]
        excluded_keys = {
            key
            for excluded_view in view.exclude
            for key in generate_view(excluded_view)
        }
        keys = _generate_view_keys(
            view,
            tasks=tasks,
            task_ids=task_ids_by_split[view.task_split],
            excluded_keys=excluded_keys,
            config=config,
        )
        active.remove(view_name)
        generated[view_name] = keys
        return keys

    for view in config.views:
        generate_view(view.name)

    serialized_outputs: dict[Path, str] = {}
    for view in config.views:
        output_path = resolve_inside(split_root, view.output, name=f"views.{view.name}.output")
        serialized_outputs[output_path] = json.dumps(
            list(generated[view.name]),
            indent=2,
        ) + "\n"

    for output_path, serialized in serialized_outputs.items():
        if output_path.exists() and output_path.read_text(encoding="utf-8") != serialized:
            if not overwrite:
                raise FileExistsError(
                    f"refusing to replace existing generated file: {output_path}"
                )

    for output_path, serialized in serialized_outputs.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")

    return {view.name: generated[view.name] for view in config.views}


def _load_view_config(
    name: Any,
    value: Any,
    *,
    known_view_names: set[str],
) -> ArcTrainingViewConfig:
    view_name = _require_string(name, name="view name")
    payload = _require_mapping(value, name=f"views.{view_name}")
    common_fields = {"task_split", "output", "mode", "exclude"}
    mode_value = payload.get("mode")
    try:
        mode = ArcTrainingViewMode(mode_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"views.{view_name}.mode must be 'identity' or 'sampled'"
        ) from error

    expected_fields = common_fields
    if mode is ArcTrainingViewMode.SAMPLED:
        expected_fields = common_fields | {"seed", "augmentations_per_task"}
    _require_exact_fields(payload, expected_fields, name=f"views.{view_name}")

    task_split = _require_string(
        payload["task_split"],
        name=f"views.{view_name}.task_split",
    )
    if task_split != "train":
        raise ValueError(f"training view {view_name!r} must use the 'train' task split")

    exclude = require_string_array(
        payload["exclude"],
        name=f"views.{view_name}.exclude",
        allow_empty=True,
    )
    if view_name in exclude:
        raise ValueError(f"view {view_name!r} cannot exclude itself")
    unknown_exclusions = tuple(
        excluded for excluded in exclude if excluded not in known_view_names
    )
    if unknown_exclusions:
        raise ValueError(
            f"view {view_name!r} excludes unknown views: {unknown_exclusions!r}"
        )

    if mode is ArcTrainingViewMode.SAMPLED:
        seed = _require_int(payload["seed"], name=f"views.{view_name}.seed")
        augmentations_per_task = _require_positive_int(
            payload["augmentations_per_task"],
            name=f"views.{view_name}.augmentations_per_task",
        )
    else:
        seed = None
        augmentations_per_task = 1

    return ArcTrainingViewConfig(
        name=view_name,
        task_split=task_split,
        output=_require_relative_path(
            payload["output"],
            name=f"views.{view_name}.output",
        ),
        mode=mode,
        exclude=exclude,
        seed=seed,
        augmentations_per_task=augmentations_per_task,
    )


def _load_source_tasks(
    source_root: Path,
    task_ids_by_split: Mapping[str, tuple[str, ...]],
) -> dict[str, ArcTrainingTask]:
    all_task_ids = tuple(
        task_id
        for split_name in ("train", "validation")
        for task_id in task_ids_by_split[split_name]
    )
    missing = tuple(
        task_id for task_id in all_task_ids if not (source_root / f"{task_id}.json").is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"source task files do not exist under {source_root}: {missing!r}"
        )
    return {
        task_id: load_training_task(source_root / f"{task_id}.json")
        for task_id in task_ids_by_split["train"]
    }


def _generate_view_keys(
    view: ArcTrainingViewConfig,
    *,
    tasks: Mapping[str, ArcTrainingTask],
    task_ids: tuple[str, ...],
    excluded_keys: set[str],
    config: ArcSplitConfig,
) -> tuple[str, ...]:
    generated: list[str] = []
    generated_set: set[str] = set()

    for task_id in task_ids:
        task = tasks[task_id]
        if view.mode is ArcTrainingViewMode.IDENTITY:
            augmentation = make_identity_training_augmentation(
                task,
                max_pairs=config.max_pairs,
                max_cells=config.max_cells,
            )
            key = make_augmentation_key(task_id, augmentation)
            if key in excluded_keys:
                raise ValueError(
                    f"identity key for task {task_id!r} is excluded from view {view.name!r}"
                )
            generated.append(key)
            generated_set.add(key)
            continue

        accepted = 0
        candidate_index = 0
        max_candidates = max(1000, view.augmentations_per_task * 100)
        while accepted < view.augmentations_per_task:
            if candidate_index == max_candidates:
                raise ValueError(
                    f"could not generate enough unique keys for task {task_id!r} "
                    f"in view {view.name!r}"
                )
            rng = Random(_candidate_seed(view, task_id, candidate_index))
            augmentation = sample_training_augmentation(
                task,
                augmentation_index=candidate_index,
                max_pairs=config.max_pairs,
                max_cells=config.max_cells,
                rng=rng,
            )
            key = make_augmentation_key(task_id, augmentation)
            candidate_index += 1
            if key in excluded_keys or key in generated_set:
                continue
            generated.append(key)
            generated_set.add(key)
            accepted += 1

    return tuple(generated)


def _candidate_seed(
    view: ArcTrainingViewConfig,
    task_id: str,
    candidate_index: int,
) -> int:
    material = (
        f"{ARC_AUGMENTATION_GENERATOR_VERSION}:"
        f"{view.name}:{view.seed}:{task_id}:{candidate_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], byteorder="big")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise ValueError(f"{name} fields differ: missing={missing!r}, extra={extra!r}")


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_relative_path(value: Any, *, name: str) -> Path:
    path = Path(_require_string(value, name=name))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path inside the split directory")
    return path


def _require_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    integer = _require_int(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


__all__ = [
    "ARC_AUGMENTATION_GENERATOR_VERSION",
    "ARC_SPLIT_SCHEMA_VERSION",
    "ArcSplitConfig",
    "ArcTrainingViewConfig",
    "ArcTrainingViewMode",
    "generate_augmented_key_files",
    "load_arc_split_config",
]
