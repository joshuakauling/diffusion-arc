"""Plan global optimizer batches before assigning whole sources to GPU ranks."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from diffusion_arc.model.contracts import TARGET_LATENT_TOKEN_COUNT

from .config import TrainingConfig
from .data import ArcTrainingDataset
from .source_lengths import load_source_token_lengths, source_lengths_path
from . import batch_packing
from .step.flux_batching import FluxMicrobatchPlan, PrecomputedFluxPlan


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
    flux_plan: PrecomputedFluxPlan | None = None


def plan_distributed_batches(
    target_counts: Sequence[int],
    *,
    optimizer_batch_size: int,
    rank: int,
    world_size: int,
    source_joint_tokens: Sequence[int] | None = None,
) -> tuple[RankBatchPlan, ...]:
    """Preserve ordered, whole-source batch boundaries across world sizes.

    Like ``iter_optimizer_batches``, a source must fit the current target
    budget; overshoot is an error, and a final partial batch is retained.
    Sources must expose positive target counts. By default each global batch is
    split contiguously by source count. With prepared token costs, assign the
    heaviest whole source to the least loaded rank, then restore local source order.
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
    costs = None if source_joint_tokens is None else tuple(source_joint_tokens)
    if costs is not None:
        if len(costs) != len(counts):
            raise ValueError("source_joint_tokens must match target_counts")
        if any(type(cost) is not int or cost <= 0 for cost in costs):
            raise ValueError("source_joint_tokens must contain positive integers")

    plans: list[RankBatchPlan] = []

    def append_batch(start: int, end: int, global_count: int) -> None:
        source_count = end - start
        if source_count < world_size:
            raise ValueError(
                f"global batch {len(plans)} has {source_count} sources for "
                f"world_size={world_size}; each rank requires a source"
            )
        if costs is None:
            quotient, remainder = divmod(source_count, world_size)
            local_start = start + rank * quotient + min(rank, remainder)
            local_end = local_start + quotient + int(rank < remainder)
            local_sources = tuple(range(local_start, local_end))
        else:
            rank_sources = [[] for _ in range(world_size)]
            rank_tokens = [0] * world_size
            for source in sorted(range(start, end), key=lambda index: (-costs[index], index)):
                owner = min(range(world_size), key=lambda index: (rank_tokens[index], len(rank_sources[index]), index))
                rank_sources[owner].append(source)
                rank_tokens[owner] += costs[source]
            local_sources = tuple(sorted(rank_sources[rank]))
        # Target IDs always refer to their original global positions, never rank order.
        targets_by_source, offset = {}, 0
        for source in range(start, end):
            targets_by_source[source] = range(offset, offset + counts[source])
            offset += counts[source]
        plans.append(RankBatchPlan(
            epoch_batch_index=len(plans),
            source_indices=local_sources,
            target_indices=tuple(target for source in local_sources for target in targets_by_source[source]),
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


def _validate_aligned_batch(aligned):
    identity = aligned[0].epoch_batch_index
    if any(plan.epoch_batch_index != identity or
           plan.global_target_count != aligned[0].global_target_count for plan in aligned):
        raise ValueError("rank batch plans must refer to the same global batch")


def _packing_sources(aligned, source_keys, source_prefix_tokens):
    _validate_aligned_batch(aligned)
    source_indices = tuple(sorted(index for plan in aligned for index in plan.source_indices))
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("global batch sources must belong to exactly one rank")
    sources = tuple(batch_packing.Source(str(source_keys[index]),
        tuple(prefix + TARGET_LATENT_TOKEN_COUNT for prefix in source_prefix_tokens[index]))
        for index in source_indices)
    target_count = sum(len(source.target_tokens) for source in sources)
    if (target_count != aligned[0].global_target_count or
            sorted(index for plan in aligned for index in plan.target_indices) != list(range(target_count))):
        raise ValueError("global batch target IDs must cover the prepared source targets")
    return source_indices, sources


def _translate_packing(aligned, source_indices, sources, packing, max_tokens):
    """Convert the shared CPU result to rank-local IDs for Trainer execution."""
    batch_packing.validate_plan(sources, packing, max_tokens)
    collective_calls = batch_packing.plan_score(packing)[0]
    target_offsets, offset = [], 0
    for source in sources:
        target_offsets.append(tuple(range(offset, offset + len(source.target_tokens))))
        offset += len(source.target_tokens)
    rank_plans = []
    for rank, owners in enumerate(packing.rank_sources):
        target_indices = tuple(index for source in owners for index in target_offsets[source])
        local_positions = {index: position for position, index in enumerate(target_indices)}
        microbatches = tuple(FluxMicrobatchPlan(
            tuple(local_positions[index] for index in pack.target_ids), pack.joint_tokens,
        ) for pack in packing.rank_packs[rank])
        prefixes = tuple(cost - TARGET_LATENT_TOKEN_COUNT
                         for source in owners for cost in sources[source].target_tokens)
        rank_plans.append(replace(aligned[rank],
            source_indices=tuple(source_indices[source] for source in owners),
            target_indices=target_indices,
            flux_plan=PrecomputedFluxPlan(microbatches, prefixes, collective_calls)))
    return tuple(rank_plans)


def precompute_distributed_packing(
    batch_plans_by_rank, *, source_keys, source_prefix_tokens, max_tokens,
):
    """Optimize selected global batches before execution; reuse replayed plans."""
    world_size = len(batch_plans_by_rank)
    if not world_size or len(source_keys) != len(source_prefix_tokens):
        raise ValueError("precomputed packing requires ranks and matching source metadata")
    if len({len(plans) for plans in batch_plans_by_rank}) != 1:
        raise ValueError("rank batch plans must have matching lengths")
    optimized, cached = [[] for _ in range(world_size)], {}
    budget = batch_packing.SearchBudget()
    for aligned in zip(*batch_plans_by_rank, strict=True):
        _validate_aligned_batch(aligned)
        identity = aligned[0].epoch_batch_index
        if identity not in cached:
            source_indices, sources = _packing_sources(aligned, source_keys, source_prefix_tokens)
            packing = batch_packing.plan_batch(sources, world_size, max_tokens, "joint", budget)
            cached[identity] = _translate_packing(aligned, source_indices, sources, packing, max_tokens)
        for rank, plan in enumerate(cached[identity]):
            optimized[rank].append(plan)
    plans = tuple(tuple(rank_plans) for rank_plans in optimized)
    encoded = json.dumps([[asdict(plan) for plan in rank_plans] for rank_plans in plans],
                         sort_keys=True, separators=(",", ":")).encode()
    metadata = dict(algorithm="joint_dp_v1", world_size=world_size,
        max_joint_tokens=max_tokens, search_budget=asdict(budget),
        unique_batches=len(cached), scheduled_batches=len(plans[0]),
        plan_sha256=hashlib.sha256(encoded).hexdigest(),
        planner_implementation_sha256=hashlib.sha256(Path(batch_packing.__file__).read_bytes()).hexdigest())
    return plans, metadata


class BackgroundBatchPlans:
    """Share a bounded CPU search queue between the sampler and training loop.

    First requests proceed in order; cached plans can be read by both consumers.
    DataLoader prefetch may run ahead of training without blocking on its cursor.
    Training releases completed plans explicitly. Only token metadata reaches
    the spawned worker; no Dataset, tokenizer, CUDA tensor or collective does.
    """

    def __init__(self, batch_plans_by_rank, *, rank, source_keys,
                 source_prefix_tokens, max_tokens, prefetch_batches=4):
        self.world_size = len(batch_plans_by_rank)
        if not self.world_size or not 0 <= rank < self.world_size:
            raise ValueError("background packing requires a valid rank and world size")
        if len({len(plans) for plans in batch_plans_by_rank}) != 1:
            raise ValueError("rank batch plans must have matching lengths")
        if len(source_keys) != len(source_prefix_tokens):
            raise ValueError("background packing requires matching source metadata")
        if type(prefetch_batches) is not int or prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be a positive integer")
        self.rank, self.max_tokens, self.prefetch_batches = rank, max_tokens, prefetch_batches
        self._aligned = tuple(zip(*batch_plans_by_rank, strict=True))
        self.source_keys, self.source_prefix_tokens = source_keys, source_prefix_tokens
        self.budget = batch_packing.SearchBudget()
        self._executor = None
        self._pending, self._ready = {}, {}
        self._next_requested, self._closed = 0, False
        self._set_last_positions()
        implementation = hashlib.sha256()
        for path in (Path(batch_packing.__file__), Path(__file__)):
            implementation.update(path.read_bytes())
        # This contract remains unchanged across resume cursors and timing windows.
        self.metadata = dict(algorithm="joint_dp_v1", world_size=self.world_size,
            max_joint_tokens=max_tokens, search_budget=asdict(self.budget),
            prefetch_batches=prefetch_batches, worker_start_method="spawn",
            implementation_sha256=implementation.hexdigest())

    def _set_last_positions(self):
        self._last_positions = {aligned[0].epoch_batch_index: index
                                for index, aligned in enumerate(self._aligned)}

    @property
    def original_plans(self):
        """Cheap batch boundaries for preflight; accessing them starts no searches."""
        return tuple(aligned[self.rank] for aligned in self._aligned)

    def __len__(self):
        return len(self._aligned)

    def truncate(self, count):
        if self._executor is not None:
            raise RuntimeError("truncate background plans before starting the worker")
        self._aligned = self._aligned[:count]
        self._set_last_positions()

    def _fill_window(self, start):
        for index in range(start, min(start + self.prefetch_batches, len(self))):
            aligned = self._aligned[index]
            identity = aligned[0].epoch_batch_index
            if identity in self._pending or identity in self._ready:
                continue
            source_indices, sources = _packing_sources(aligned, self.source_keys, self.source_prefix_tokens)
            future = self._executor.submit(batch_packing.plan_batch, sources,
                self.world_size, self.max_tokens, "joint", self.budget)
            self._pending[identity] = (future, aligned, source_indices, sources)

    def __getitem__(self, index):
        if self._closed:
            raise RuntimeError("background batch plans are closed")
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        identity = self._aligned[index][0].epoch_batch_index
        if index > self._next_requested or (index < self._next_requested and identity not in self._ready):
            raise RuntimeError("request background batch plans in order; retain them until training completes")
        try:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"))
                self._fill_window(0)
            if identity not in self._ready:
                future, aligned, source_indices, sources = self._pending.pop(identity)
                self._ready[identity] = _translate_packing(aligned, source_indices, sources,
                    future.result(), self.max_tokens)
            if index == self._next_requested:
                self._next_requested += 1
                self._fill_window(self._next_requested)
            return self._ready[identity][self.rank]
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def release(self, index):
        """Discard a plan once training has consumed all its replayed occurrences."""
        identity = self._aligned[index][0].epoch_batch_index
        if self._last_positions[identity] == index:
            self._ready.pop(identity, None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for future, *_ in self._pending.values():
            future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._ready.clear()


class _PlannedSourceSampler:
    """Request each rank's source indices only after its CPU plan is available."""

    def __init__(self, plans):
        self.plans = plans

    def __iter__(self):
        for plan in self.plans:
            yield from plan.source_indices


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
    precompute_packing: bool = False,
    max_batches: int | None = None,
) -> tuple[DataLoader, tuple[RankBatchPlan, ...] | BackgroundBatchPlans]:
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
    if type(precompute_packing) is not bool:
        raise ValueError("precompute_packing must be a boolean")
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError("max_batches must be a positive integer")
    if precompute_packing and (config.rank_batch_assignment != "token_balanced_v1" or
            config.flux_packing.algorithm != "legacy_bfd_v1"):
        raise ValueError("precomputed packing requires prepared token balancing and legacy BFD")
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
    source_costs = None
    if config.rank_batch_assignment in ("token_balanced_v1", "joint_packing_v1"):
        prefixes = load_source_token_lengths(dataset,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target)
        source_costs = tuple(sum(length + TARGET_LATENT_TOKEN_COUNT for length in source) for source in prefixes)
        dataset.eligible_prefix_token_lengths = prefixes
        dataset.lengths_min_context_pairs = config.loop.min_context_pairs_before_target
        dataset.rank_assignment_metadata = dict(algorithm=config.rank_batch_assignment,
            source_token_lengths_sha256=hashlib.sha256(source_lengths_path(dataset).read_bytes()).hexdigest())
    plans = plan_distributed_batches(
        tuple(len(key.augmentation.pair_indices) - config.loop.min_context_pairs_before_target
              for key in keys),
        optimizer_batch_size=config.loop.optimizer_batch_size,
        rank=rank,
        world_size=world_size,
        source_joint_tokens=source_costs,
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
    if max_batches is not None:
        plans = plans[:max_batches]
    background_packing = config.rank_batch_assignment == "joint_packing_v1"
    if precompute_packing or background_packing:
        started = time.perf_counter()
        all_rank_plans = []
        for owner in range(world_size):
            original = plan_distributed_batches(
                tuple(len(prefix) for prefix in prefixes),
                optimizer_batch_size=config.loop.optimizer_batch_size,
                rank=owner, world_size=world_size, source_joint_tokens=source_costs)
            all_rank_plans.append(tuple(original[plan.epoch_batch_index] for plan in plans))
        if background_packing:
            plans = BackgroundBatchPlans(tuple(all_rank_plans), rank=rank,
                source_keys=keys, source_prefix_tokens=prefixes, max_tokens=config.flux_packing.max_tokens)
            dataset.rank_assignment_metadata["background_packing"] = plans.metadata
        else:
            prepared, metadata = precompute_distributed_packing(tuple(all_rank_plans),
                source_keys=keys, source_prefix_tokens=prefixes, max_tokens=config.flux_packing.max_tokens)
            plans = prepared[rank]
            dataset.rank_assignment_metadata["precomputed_packing"] = metadata
            dataset.precomputed_packing_plans = prepared
            dataset.precomputed_packing_seconds = time.perf_counter() - started
    local_indices = (_PlannedSourceSampler(plans) if background_packing else
                     tuple(index for plan in plans for index in plan.source_indices))
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


__all__ = ["RankBatchPlan", "BackgroundBatchPlans", "plan_distributed_batches",
           "precompute_distributed_packing", "build_distributed_dataloader"]
