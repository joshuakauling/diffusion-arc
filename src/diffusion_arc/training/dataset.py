"""Config-driven PyTorch dataset for reproducible ARC training views."""

from __future__ import annotations

from pathlib import Path

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
from diffusion_arc.arc_data._split_io import (
    load_string_array,
    load_task_splits,
    resolve_inside,
    validate_task_split_boundary,
)
from diffusion_arc.arc_data._split_layout import resolve_split_layout


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

        unresolved_split_root = Path(split_directory).resolve()
        config = load_arc_split_config(unresolved_split_root / "config.json")
        split_root, source_directory = resolve_split_layout(
            unresolved_split_root,
            dataset=config.dataset,
            source_split=config.source_split,
        )
        view = _select_view(config, view_name)
        task_ids_by_split = load_task_splits(split_root, config.task_split_files)
        validate_task_split_boundary(task_ids_by_split)

        key_path = resolve_inside(
            split_root,
            view.output,
            name=f"views.{view.name}.output",
        )
        serialized_keys = load_string_array(
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

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> ArcTokenizedTrainingSample:
        key = self.keys[index]
        task = load_training_task(self.source_directory / f"{key.task_id}.json")
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


__all__ = ["ArcTrainingDataset"]
