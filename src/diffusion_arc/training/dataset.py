"""Config-driven PyTorch dataset for reproducible ARC training views."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from diffusion_arc.arc_data import (
    ArcSplitConfig,
    ArcTokenizedTrainingSample,
    ArcTrainingViewConfig,
    apply_training_augmentation,
    load_arc_split_config,
    load_training_task,
    parse_augmentation_key,
    render_training_prompt,
    tokenize_nvarc_training_sample,
    validate_augmentation_for_task,
)
from diffusion_arc.arc_data._layout import resolve_split_layout


class ArcTrainingDataset(Dataset[ArcTokenizedTrainingSample]):
    """Materialize one configured augmentation key into one tokenized sample."""

    def __init__(
        self,
        split_directory: str | Path,
        view_name: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        if not isinstance(view_name, str) or not view_name:
            raise TypeError("view_name must be a non-empty string")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("tokenizer must be a Hugging Face tokenizer")

        unresolved_split_root = Path(split_directory).resolve()
        config = load_arc_split_config(unresolved_split_root / "config.json")
        split_root, source_directory = resolve_split_layout(
            unresolved_split_root,
            dataset=config.dataset,
            source_split=config.source_split,
        )
        view = _select_view(config, view_name)
        task_ids_by_split = _load_task_splits(split_root, config)
        _validate_task_split_boundary(task_ids_by_split)

        key_path = _resolve_inside(
            split_root,
            view.output,
            name=f"views.{view.name}.output",
        )
        serialized_keys = _load_string_array(
            key_path,
            name=str(key_path),
            allow_empty=False,
        )
        keys = tuple(parse_augmentation_key(value) for value in serialized_keys)

        allowed_task_ids = set(task_ids_by_split[view.task_split])
        outside_split = tuple(
            dict.fromkeys(
                key.task_id for key in keys if key.task_id not in allowed_task_ids
            )
        )
        if outside_split:
            raise ValueError(
                f"view {view.name!r} contains task IDs outside its "
                f"{view.task_split!r} task split: {outside_split!r}"
            )

        referenced_task_ids = tuple(dict.fromkeys(key.task_id for key in keys))
        missing_task_ids = tuple(
            task_id
            for task_id in referenced_task_ids
            if not (source_directory / f"{task_id}.json").is_file()
        )
        if missing_task_ids:
            raise FileNotFoundError(
                f"source task files do not exist under {source_directory}: "
                f"{missing_task_ids!r}"
            )

        tasks = {
            task_id: load_training_task(source_directory / f"{task_id}.json")
            for task_id in referenced_task_ids
        }
        for key in keys:
            task = tasks[key.task_id]
            validate_augmentation_for_task(key, task)
            if len(key.augmentation.pair_indices) < 2:
                raise ValueError(
                    f"training key {key!s} must reference at least two pairs"
                )

        self.split_directory = split_root
        self.view_name = view.name
        self.source_directory = source_directory
        self.max_tokens = config.max_tokens
        self.keys = keys
        self._tokenizer = tokenizer
        self._tasks = tasks

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> ArcTokenizedTrainingSample:
        key = self.keys[index]
        task = self._tasks[key.task_id]
        augmented = apply_training_augmentation(task, key.augmentation)
        rendered = render_training_prompt(augmented)
        return tokenize_nvarc_training_sample(
            rendered,
            self._tokenizer,
            max_tokens=self.max_tokens,
        )


def _select_view(config: ArcSplitConfig, view_name: str) -> ArcTrainingViewConfig:
    views_by_name = {view.name: view for view in config.views}
    try:
        return views_by_name[view_name]
    except KeyError as error:
        raise ValueError(
            f"unknown training view {view_name!r}; "
            f"available views: {tuple(views_by_name)!r}"
        ) from error


def _load_task_splits(
    split_root: Path,
    config: ArcSplitConfig,
) -> dict[str, tuple[str, ...]]:
    task_ids_by_split: dict[str, tuple[str, ...]] = {}
    for split_name, relative_path in config.task_split_files:
        path = _resolve_inside(
            split_root,
            relative_path,
            name=f"task_splits.{split_name}",
        )
        task_ids = _load_string_array(
            path,
            name=str(path),
            allow_empty=split_name == "validation",
        )
        for task_id in task_ids:
            if "_" in task_id:
                raise ValueError(f"task ID must not contain '_': {task_id!r}")
            _require_path_component(task_id, name=f"task ID in {path}")
        task_ids_by_split[split_name] = task_ids
    return task_ids_by_split


def _validate_task_split_boundary(
    task_ids_by_split: Mapping[str, tuple[str, ...]],
) -> None:
    overlap = set(task_ids_by_split["train"]) & set(
        task_ids_by_split["validation"]
    )
    if overlap:
        raise ValueError(
            f"train and validation task splits overlap: {tuple(sorted(overlap))!r}"
        )


def _load_string_array(
    path: Path,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"{name} must be a JSON array")
    if not payload and not allow_empty:
        raise ValueError(f"{name} must not be empty")

    strings: list[str] = []
    for item in payload:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} entries must be non-empty strings")
        strings.append(item)
    if len(set(strings)) != len(strings):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(strings)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_path_component(value: str, *, name: str) -> None:
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be one path component")


def _resolve_inside(root: Path, relative_path: Path, *, name: str) -> Path:
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} must resolve inside the split directory")
    return resolved


__all__ = ["ArcTrainingDataset"]
