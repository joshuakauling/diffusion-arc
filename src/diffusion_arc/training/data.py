"""CPU dataset, collation, and DataLoader construction for ARC training."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from torch import Tensor
import torch
from torch.utils.data import DataLoader, Dataset
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
from diffusion_arc.arc_data.augmentation import ArcAugmentationKey
from diffusion_arc.arc_data.canvas import _render_valid_grid_canvas
from diffusion_arc.model.contracts import (
    TARGET_IMAGE_CHANNELS,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_SHAPE,
    TARGET_IMAGE_WIDTH,
    _require_tensor,
    _validate_index_tensor,
    validate_packed_source_sequences,
)

from .config import TrainingConfig


class ArcTrainingDataset(Dataset[ArcTokenizedTrainingSample]):
    """Materialize configured keys and fingerprint their validated source data.

    ``fingerprint`` identifies the ordered view and source contents at setup;
    training source files must remain unchanged while the loader is consumed.
    """

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
        # Bind resumes to the ordered view and every pair in its source tasks.
        # Tasks are already loaded for preflight; hashing adds no source reads.
        identity = {
            "dataset": config.dataset,
            "source_split": config.source_split,
            "palette": config.palette,
            "max_tokens": config.max_tokens,
            "view_name": view.name,
            "keys": serialized_keys,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for task_id in sorted(tasks):
            digest.update(b"\n")
            digest.update(
                json.dumps(asdict(tasks[task_id]), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            )
        self.fingerprint = digest.hexdigest()

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


@dataclass(frozen=True)
class PackedTrainingBatch:
    """Packed source sequences and their aligned training target exposures.

    ``sequence_offsets`` delimits ``S`` unpadded source sequences inside the
    flat token tensor. Each of the ``K`` target exposures refers to one source
    and stores an end-exclusive prefix boundary relative to that source. The
    final prefix may equal its source length when the unused terminal target
    output has been omitted.
    """

    input_ids: Tensor
    sequence_offsets: Tensor
    target_sequence_indices: Tensor
    target_prefix_ends: Tensor
    target_images: Tensor


def validate_packed_training_batch(batch: PackedTrainingBatch) -> None:
    """Raise when a packed training batch violates the public tensor contract."""

    source_lengths = validate_packed_source_sequences(
        input_ids=batch.input_ids,
        sequence_offsets=batch.sequence_offsets,
    )
    target_count = _validate_target_images(batch.target_images)

    _validate_index_tensor(
        batch.target_sequence_indices,
        name="target_sequence_indices",
        expected_length=target_count,
    )
    _validate_index_tensor(
        batch.target_prefix_ends,
        name="target_prefix_ends",
        expected_length=target_count,
    )

    target_sequence_indices = batch.target_sequence_indices.tolist()
    target_prefix_ends = batch.target_prefix_ends.tolist()
    source_count = len(source_lengths)

    previous_source_index = -1
    previous_prefix_end = 0
    seen_source_indices: set[int] = set()

    for target_index, (source_index, prefix_end) in enumerate(
        zip(target_sequence_indices, target_prefix_ends, strict=True)
    ):
        if not 0 <= source_index < source_count:
            raise ValueError(
                "target_sequence_indices contains an out-of-range source index "
                f"at target {target_index}"
            )
        if source_index < previous_source_index:
            raise ValueError("target_sequence_indices must be ordered by source")

        if source_index != previous_source_index:
            previous_prefix_end = 0

        if prefix_end <= previous_prefix_end:
            raise ValueError(
                "target_prefix_ends must be positive and strictly increasing "
                "within each source"
            )
        if prefix_end > source_lengths[source_index]:
            raise ValueError(
                "target_prefix_ends must not lie beyond their source sequence"
            )

        seen_source_indices.add(source_index)
        previous_source_index = source_index
        previous_prefix_end = prefix_end

    if seen_source_indices != set(range(source_count)):
        raise ValueError("every source sequence must have at least one target exposure")


def _validate_target_images(target_images: Tensor) -> int:
    _require_tensor(target_images, name="target_images")
    if target_images.dtype != torch.float32:
        raise TypeError("target_images must have dtype torch.float32")
    if target_images.ndim != 4 or tuple(target_images.shape[1:]) != TARGET_IMAGE_SHAPE:
        raise ValueError(
            "target_images must have shape "
            f"[K, {TARGET_IMAGE_CHANNELS}, {TARGET_IMAGE_HEIGHT}, {TARGET_IMAGE_WIDTH}]"
        )

    target_count = target_images.shape[0]
    if target_count == 0:
        raise ValueError("a training batch must contain at least one target exposure")
    if not torch.isfinite(target_images).all().item():
        raise ValueError("target_images must contain only finite values")
    if torch.any((target_images < -1.0) | (target_images > 1.0)).item():
        raise ValueError("target_images values must lie in [-1, 1]")
    return target_count


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
            input_ids.extend(sample.input_ids[:source_input_end])
            sequence_offsets.append(len(input_ids))
            source_keys.append(sample.key)

            target_sequence_indices.extend(
                [source_index] * selected_target_count
            )
            target_prefix_ends.extend(selected_prefix_ends)
            target_pair_indices.extend(selected_pair_indices)
            target_images.extend(
                _render_valid_grid_canvas(target_grid)
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


def iter_optimizer_batches(
    samples: Iterable[ArcTokenizedTrainingSample],
    *,
    optimizer_batch_size: int,
    min_context_pairs_before_target: int,
    max_optimizer_steps: int | None = None,
) -> Iterator[tuple[ArcTokenizedTrainingSample, ...]]:
    """Group whole CPU sequences by their actual eligible target count.

    Preserve source order and return a final partial batch on exhaustion. A
    sequence that would overshoot the target budget raises instead of splitting
    its targets. The step limit stops consumption before reading another sample.
    """

    for name, value in (
        ("optimizer_batch_size", optimizer_batch_size),
        ("min_context_pairs_before_target", min_context_pairs_before_target),
        ("max_optimizer_steps", max_optimizer_steps),
    ):
        if value is None and name == "max_optimizer_steps":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    batch: list[ArcTokenizedTrainingSample] = []
    target_count = completed_steps = 0
    for sample in samples:
        eligible = len(sample.target_grids) - (min_context_pairs_before_target - 1)
        if eligible <= 0:
            continue
        if target_count + eligible > optimizer_batch_size:
            raise ValueError(
                f"whole sequence {sample.key!s} would exceed optimizer_batch_size="
                f"{optimizer_batch_size}: {target_count} pending + {eligible} eligible targets"
            )
        batch.append(sample)
        target_count += eligible
        if target_count == optimizer_batch_size:
            yield tuple(batch)
            completed_steps += 1
            if completed_steps == max_optimizer_steps:
                return
            batch = []
            target_count = 0
    if batch:
        yield tuple(batch)


def build_dataloader(
    config: TrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    start_index: int = 0,
) -> DataLoader:
    """Yield individual CPU samples in seeded sampler order.

    Optional workers prefetch ``num_workers * prefetch_factor`` sample tasks,
    including data loading and tokenization. This does not enlarge the GPU
    optimizer batch: the caller groups samples only as it consumes them. The
    loader's random generator is independent of flow-noise sampling. Resumes
    skip consumed source indices before loading or tokenizing their samples;
    this cursor requires the configured, unshuffled order.
    """

    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if start_index and config.data.shuffle:
        raise ValueError("start_index requires shuffle=False")
    dataset = ArcTrainingDataset(
        split_directory=config.data.split_directory,
        view_name=config.data.view_name,
        tokenizer=tokenizer,
    )
    if start_index > len(dataset):
        raise ValueError("start_index must not exceed the dataset length")
    sampler_arguments = {"sampler": range(start_index, len(dataset))} if start_index else {}
    data_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    worker_arguments = {}
    if config.data.num_workers:
        # Models can already own CUDA contexts. Workers must start fresh and
        # return samples in sampler order, regardless of tokenization latency.
        worker_arguments = {
            "multiprocessing_context": "spawn",
            "prefetch_factor": config.data.prefetch_factor,
            "in_order": True,
        }
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers,
        generator=data_generator,
        **sampler_arguments,
        **worker_arguments,
    )


__all__ = ["ArcTrainingDataset", "PackedTrainingBatch", "validate_packed_training_batch",
           "TrainingBatchMetadata", "CollatedTrainingBatch", "TrainingCollator", "build_dataloader",
           "iter_optimizer_batches"]
