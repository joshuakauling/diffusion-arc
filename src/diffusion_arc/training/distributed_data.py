"""Plan global optimizer batches before assigning whole sources to DDP ranks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .config import TrainingConfig
from .data import ArcTrainingDataset


@dataclass(frozen=True)
class RankBatchPlan:
    """One rank's ordered sources and targets in a global optimizer batch.

    ``epoch_batch_index`` is zero-based and retains the original batch identity
    when replayed for diagnostics. Source indices address the complete dataset; target
    indices address the eligible targets in this global batch, before packing.
    Every rank receives at least one whole source, without padding or dropping.
    """

    epoch_batch_index: int
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    global_target_count: int


def plan_distributed_batches(
    target_counts: Sequence[int],
    *,
    optimizer_batch_size: int,
    rank: int,
    world_size: int,
) -> tuple[RankBatchPlan, ...]:
    """Preserve ordered, whole-source batch boundaries across world sizes.

    Like ``iter_optimizer_batches``, a source must fit the current target
    budget; overshoot is an error, and a final partial batch is retained.
    Sources must expose positive target counts. Each global batch is split
    contiguously by source count, with remainder sources on earlier ranks.
    Preflight rejects batches too small to keep every rank active.
    """

    for name, value in (("optimizer_batch_size", optimizer_batch_size),
                        ("world_size", world_size), ("rank", rank)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if optimizer_batch_size <= 0:
        raise ValueError("optimizer_batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    counts = tuple(target_counts)
    if not counts:
        raise ValueError("target_counts must contain at least one source")
    for index, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"target_counts[{index}] must be a positive integer")
        if count <= 0:
            raise ValueError(f"target_counts[{index}] must be positive")

    plans: list[RankBatchPlan] = []

    def append_batch(start: int, end: int, global_count: int) -> None:
        source_count = end - start
        if source_count < world_size:
            raise ValueError(
                f"global batch {len(plans)} has {source_count} sources for "
                f"world_size={world_size}; each DDP rank requires a source"
            )
        quotient, remainder = divmod(source_count, world_size)
        local_start = start + rank * quotient + min(rank, remainder)
        local_end = local_start + quotient + int(rank < remainder)
        target_start = sum(counts[start:local_start])
        target_end = target_start + sum(counts[local_start:local_end])
        plans.append(RankBatchPlan(
            epoch_batch_index=len(plans),
            source_indices=tuple(range(local_start, local_end)),
            target_indices=tuple(range(target_start, target_end)),
            global_target_count=global_count,
        ))

    start = pending = 0
    for index, count in enumerate(counts):
        if pending + count > optimizer_batch_size:
            raise ValueError(
                f"whole source {index} would exceed optimizer_batch_size="
                f"{optimizer_batch_size}: {pending} pending + {count} eligible targets"
            )
        pending += count
        if pending == optimizer_batch_size:
            append_batch(start, index + 1, pending)
            start, pending = index + 1, 0
    if pending:
        append_batch(start, len(counts), pending)
    return tuple(plans)


def build_distributed_dataloader(
    config: TrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    rank: int,
    world_size: int,
    expected_source_count: int | None = None,
    start_step: int = 0,
    diagnostic_batch_number: int | None = None,
    diagnostic_repeats: int = 3,
) -> tuple[DataLoader, tuple[RankBatchPlan, ...]]:
    """Preflight the global view, then materialize only this rank's sources.

    The prepared recipe already defines the global source order. The returned
    loader yields individual CPU sequences in plan order; the runner consumes
    exactly ``len(plan.source_indices)`` sequences for each optimizer step.
    Workers use an independent RNG and never initialize a CUDA context.
    start_step skips completed epoch-relative updates without renumbering the
    remaining plans or reading their source payloads. ``diagnostic_batch_number``
    selects a one-based batch number to replay with its original flow RNG identity.
    """

    if config.config_type != "training_run" or config.recipe is None:
        raise ValueError("distributed data loading requires a training_run recipe")
    if config.data.shuffle:
        raise ValueError("distributed batch planning requires shuffle=False")
    if type(start_step) is not int or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if diagnostic_batch_number is not None and start_step:
        raise ValueError("diagnostic_batch_number cannot be combined with a resume start_step")
    dataset = ArcTrainingDataset(
        split_directory=config.data.split_directory,
        view_name=config.data.view_name,
        tokenizer=tokenizer,
    )
    expected_sources = (config.recipe["data"]["expected_task_count"]
                        if expected_source_count is None else expected_source_count)
    if expected_sources is not None and (type(expected_sources) is not int or expected_sources <= 0):
        raise ValueError("expected_source_count must be a positive integer")
    keys = dataset.keys
    if expected_sources is not None and (len(keys) != expected_sources or len({key.task_id for key in keys}) != expected_sources):
        raise ValueError(f"training view must contain exactly {expected_sources} distinct tasks")
    if any(len(key.augmentation.pair_indices) != config.recipe["data"]["pairs_per_sequence"]
           for key in keys):
        raise ValueError("training view pair counts do not match the run recipe")
    plans = plan_distributed_batches(
        tuple(len(key.augmentation.pair_indices) - config.loop.min_context_pairs_before_target
              for key in keys),
        optimizer_batch_size=config.loop.optimizer_batch_size,
        rank=rank,
        world_size=world_size,
    )
    if start_step > len(plans):
        raise ValueError("start_step exceeds the epoch's optimizer batch count")
    if diagnostic_batch_number is not None:
        if type(diagnostic_batch_number) is not int or not 1 <= diagnostic_batch_number <= len(plans):
            raise ValueError("diagnostic_batch_number must select an existing one-based global batch")
        if type(diagnostic_repeats) is not int or diagnostic_repeats <= 0:
            raise ValueError("diagnostic_repeats must be positive")
        # Preserve the original batch/target identities for flow RNG; load only this batch.
        plans = (plans[diagnostic_batch_number - 1],) * diagnostic_repeats
    else:
        plans = plans[start_step:]
    local_indices = tuple(index for plan in plans for index in plan.source_indices)
    data_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    worker_arguments = {}
    if config.data.num_workers:
        worker_arguments = {
            "multiprocessing_context": "spawn",
            "prefetch_factor": config.data.prefetch_factor,
            "in_order": True,
        }
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        sampler=local_indices,
        num_workers=config.data.num_workers,
        generator=data_generator,
        **worker_arguments,
    )
    return loader, plans


__all__ = ["RankBatchPlan", "plan_distributed_batches", "build_distributed_dataloader"]
