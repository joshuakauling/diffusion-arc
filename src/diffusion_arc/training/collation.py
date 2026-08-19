"""CPU collation from tokenized ARC samples to model training contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from diffusion_arc.arc_data.augmentation import ArcAugmentationKey
from diffusion_arc.arc_data.canvas import render_grid_canvas
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.contracts import (
    NVARC_MAX_TOKENS,
    NVARC_VOCAB_SIZE,
    PackedTrainingBatch,
)


@dataclass(frozen=True)
class TrainingBatchMetadata:
    """Non-model identities aligned with the sources and target exposures."""

    source_keys: tuple[ArcAugmentationKey, ...]
    target_pair_indices: tuple[int, ...]


@dataclass(frozen=True)
class CollatedTrainingBatch:
    """Model tensors assembled with their aligned CPU-side audit metadata."""

    tensors: PackedTrainingBatch
    metadata: TrainingBatchMetadata


@dataclass(frozen=True)
class TrainingCollator:
    """Build one host-side batch from suffix targets of tokenized ARC samples."""

    max_targets_per_source: int | None

    def __post_init__(self) -> None:
        if self.max_targets_per_source is None:
            return
        if isinstance(self.max_targets_per_source, bool) or not isinstance(
            self.max_targets_per_source, int
        ):
            raise TypeError("max_targets_per_source must be an integer or None")
        if self.max_targets_per_source <= 0:
            raise ValueError("max_targets_per_source must be positive")

    def __call__(
        self,
        samples: Sequence[ArcTokenizedTrainingSample],
    ) -> CollatedTrainingBatch:
        """Select suffix targets and collate without padding or device moves."""

        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a sequence of ArcTokenizedTrainingSample")
        if not samples:
            raise ValueError("samples must contain at least one training sample")

        input_ids: list[int] = []
        sequence_offsets = [0]
        target_sequence_indices: list[int] = []
        target_prefix_ends: list[int] = []
        target_images: list[torch.Tensor] = []
        source_keys: list[ArcAugmentationKey] = []
        target_pair_indices: list[int] = []

        for source_index, sample in enumerate(samples):
            self._validate_sample(sample, source_index=source_index)

            available_target_count = len(sample.target_grids)
            if self.max_targets_per_source is None:
                selected_target_start = 0
            else:
                selected_target_start = max(
                    0,
                    available_target_count - self.max_targets_per_source,
                )

            selected_prefix_ends = sample.prefix_token_ends[selected_target_start:]
            selected_target_grids = sample.target_grids[selected_target_start:]
            available_pair_indices = sample.key.augmentation.pair_indices[1:]
            selected_pair_indices = available_pair_indices[selected_target_start:]
            selected_target_count = len(selected_target_grids)

            source_input_end = selected_prefix_ends[-1]
            if source_input_end > NVARC_MAX_TOKENS:
                raise ValueError(
                    f"samples[{source_index}] selected source prefix exceeds "
                    f"{NVARC_MAX_TOKENS} tokens"
                )
            input_ids.extend(sample.input_ids[:source_input_end])
            sequence_offsets.append(len(input_ids))
            source_keys.append(sample.key)

            target_sequence_indices.extend(
                [source_index] * selected_target_count
            )
            target_prefix_ends.extend(selected_prefix_ends)
            target_pair_indices.extend(selected_pair_indices)
            target_images.extend(
                render_grid_canvas(target_grid)
                for target_grid in selected_target_grids
            )

        tensors = PackedTrainingBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.int64),
            sequence_offsets=torch.tensor(sequence_offsets, dtype=torch.int64),
            target_sequence_indices=torch.tensor(
                target_sequence_indices,
                dtype=torch.int64,
            ),
            target_prefix_ends=torch.tensor(
                target_prefix_ends,
                dtype=torch.int64,
            ),
            target_images=torch.stack(target_images),
        )

        return CollatedTrainingBatch(
            tensors=tensors,
            metadata=TrainingBatchMetadata(
                source_keys=tuple(source_keys),
                target_pair_indices=tuple(target_pair_indices),
            ),
        )

    def _validate_sample(
        self,
        sample: ArcTokenizedTrainingSample,
        *,
        source_index: int,
    ) -> None:
        if not isinstance(sample, ArcTokenizedTrainingSample):
            raise TypeError(
                f"samples[{source_index}] must be an ArcTokenizedTrainingSample"
            )
        if not isinstance(sample.key, ArcAugmentationKey):
            raise TypeError(f"samples[{source_index}].key must be an ArcAugmentationKey")

        _validate_token_id_tuple(
            sample.input_ids,
            name=f"samples[{source_index}].input_ids",
        )
        _validate_integer_tuple(
            sample.prefix_token_ends,
            name=f"samples[{source_index}].prefix_token_ends",
        )
        if not isinstance(sample.target_grids, tuple):
            raise TypeError(f"samples[{source_index}].target_grids must be a tuple")

        prefix_count = len(sample.prefix_token_ends)
        target_count = len(sample.target_grids)
        if prefix_count != target_count:
            raise ValueError(
                f"samples[{source_index}] has {prefix_count} prefix ends but "
                f"{target_count} target grids"
            )
        if target_count == 0:
            raise ValueError(f"samples[{source_index}] must contain at least one target")

        previous_prefix_end = 0
        for prefix_end in sample.prefix_token_ends:
            if prefix_end <= previous_prefix_end:
                raise ValueError(
                    f"samples[{source_index}].prefix_token_ends must be positive "
                    "and strictly increasing"
                )
            if prefix_end >= len(sample.input_ids):
                raise ValueError(
                    f"samples[{source_index}].prefix_token_ends must lie before "
                    "the end of the full teacher-forced input"
                )
            previous_prefix_end = prefix_end

        pair_indices = sample.key.augmentation.pair_indices
        expected_pair_count = target_count + 1
        if len(pair_indices) != expected_pair_count:
            raise ValueError(
                f"samples[{source_index}] augmentation must identify exactly "
                f"{expected_pair_count} source pairs, got {len(pair_indices)}"
            )


def _validate_integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise TypeError(f"{name} entries must be integers")
    return value


def _validate_token_id_tuple(value: object, *, name: str) -> None:
    token_ids = _validate_integer_tuple(value, name=name)
    if any(
        token_id < 0 or token_id >= NVARC_VOCAB_SIZE
        for token_id in token_ids
    ):
        raise ValueError(
            f"{name} entries must lie in [0, {NVARC_VOCAB_SIZE})"
        )


__all__ = [
    "CollatedTrainingBatch",
    "TrainingBatchMetadata",
    "TrainingCollator",
]
