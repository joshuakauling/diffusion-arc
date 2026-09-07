"""Deterministic NVARC packing inside one fixed source window."""

from __future__ import annotations

from dataclasses import dataclass

from diffusion_arc._constants import NVARC_MAX_TOKENS


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, order=True)
class NvarcBatchShape:
    """One permitted physical shape, not a cross product of two catalogs."""

    batch_size: int
    max_tokens: int

    def __post_init__(self) -> None:
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.max_tokens, "max_tokens")
        if self.batch_size not in (2, 4, 8):
            raise ValueError("regular NVARC batch_size must be 2, 4, or 8")
        if self.max_tokens > NVARC_MAX_TOKENS:
            raise ValueError(f"max_tokens must not exceed {NVARC_MAX_TOKENS}")


@dataclass(frozen=True)
class NvarcSchedulingPolicy:
    """Validated shapes and a heuristic trade-off between padding and calls.

    Plans minimize ``sum(batch_size * padded_length) + call_overhead_tokens *
    number_of_calls``. The per-call weight is expressed in token positions,
    not measured time; increasing it favors fewer calls at the cost of padding.
    Odd windows admit one singleton at its real length. See
    ``docs/performance/b2-preparation-policy.md`` for the catalog rationale.
    """

    shapes: tuple[NvarcBatchShape, ...]
    call_overhead_tokens: int

    def __post_init__(self) -> None:
        shapes = tuple(self.shapes)
        if not shapes or any(not isinstance(shape, NvarcBatchShape) for shape in shapes):
            raise ValueError("shapes must contain NvarcBatchShape objects")
        if len(set(shapes)) != len(shapes):
            raise ValueError("shapes must not contain duplicates")
        if (
            isinstance(self.call_overhead_tokens, bool)
            or not isinstance(self.call_overhead_tokens, int)
            or self.call_overhead_tokens < 0
        ):
            raise ValueError("call_overhead_tokens must be a nonnegative integer")
        max_tokens = max(shape.max_tokens for shape in shapes)
        if NvarcBatchShape(2, max_tokens) not in shapes:
            raise ValueError("the catalog must include a pair at its largest token boundary")
        object.__setattr__(self, "shapes", tuple(sorted(shapes)))


@dataclass(frozen=True)
class NvarcSourceBatch:
    """Execution plan for one physical NVARC call, without owning tensors.

    ``source_indices`` refer to the original window after eligibility filtering,
    before length sorting. ``padded_length`` is the exact execution length,
    including any padding; an odd-window singleton uses its real source length.
    """

    source_indices: tuple[int, ...]
    padded_length: int


def schedule_nvarc_sources(
    source_lengths: tuple[int, ...],
    policy: NvarcSchedulingPolicy,
) -> tuple[NvarcSourceBatch, ...]:
    """Minimize padded positions plus a fixed cost per NVARC call.

    Sort stably by descending length, then partition contiguous groups using
    dynamic programming. Ties prefer fewer calls, then the first canonical
    traversal (ascending batch size and boundary). Odd windows admit exactly
    one singleton at its real source length; even windows never use it. Sources
    never cross the window boundary and no dummy sources are introduced.
    """

    max_tokens = max(shape.max_tokens for shape in policy.shapes)
    for length in source_lengths:
        _positive_integer(length, "source length")
        if length > max_tokens:
            raise ValueError("source length exceeds the NVARC scheduling catalog")
    order = tuple(sorted(range(len(source_lengths)), key=lambda index: -source_lengths[index]))
    count = len(order)
    needs_singleton = count % 2
    # Each state keeps the cheapest plan for a prefix of the sorted sources.
    # Track singleton use separately so an odd window consumes exactly one.
    best = {(0, 0): (0, 0, ())}
    for start in range(count):
        for used_singleton in range(needs_singleton + 1):
            previous = best.get((start, used_singleton))
            if previous is None:
                continue
            candidates = [(shape.batch_size, shape.max_tokens) for shape in policy.shapes]
            if needs_singleton and not used_singleton:
                candidates.insert(0, (1, source_lengths[order[start]]))
            for batch_size, padded_length in candidates:
                end = start + batch_size
                # Descending order makes the first source the group's longest.
                if end > count or source_lengths[order[start]] > padded_length:
                    continue
                tail_used = used_singleton + (batch_size == 1)
                cost = previous[0] + batch_size * padded_length + policy.call_overhead_tokens
                calls = previous[1] + 1
                current = best.get((end, tail_used))
                if current is None or (cost, calls) < current[:2]:
                    batch = NvarcSourceBatch(order[start:end], padded_length)
                    best[end, tail_used] = (cost, calls, (*previous[2], batch))
    return best[count, needs_singleton][2]


__all__ = [
    "NvarcBatchShape",
    "NvarcSchedulingPolicy",
    "NvarcSourceBatch",
    "schedule_nvarc_sources",
]
