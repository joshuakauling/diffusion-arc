"""Pack one optimizer window under a joint text/image token budget."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from diffusion_arc._constants import NVARC_MAX_TOKENS


# The fixed 30 x 30 target canvas contributes one image token per ARC cell.
_FLUX_IMAGE_TOKENS = 900
_INT32_MAX = (1 << 31) - 1
# Bound CPU planning work deterministically, without a wall-clock/RNG dependency.
_SEARCH_NODES = 20_000
_IMPROVEMENT_PASSES = 16


@dataclass(frozen=True)
class FluxPackingPolicy:
    """Versioned planner, hard call budget and optional source-release spread."""

    max_tokens: int = 12_288
    max_spread_tokens: int = 1_000
    algorithm: str = "balanced_release_v1"

    def __post_init__(self) -> None:
        if self.algorithm not in ("legacy_bfd_v1", "balanced_release_v1"):
            raise ValueError("algorithm must be legacy_bfd_v1 or balanced_release_v1")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        minimum = NVARC_MAX_TOKENS + _FLUX_IMAGE_TOKENS
        if not minimum <= self.max_tokens <= _INT32_MAX:
            raise ValueError(f"max_tokens must lie in [{minimum}, {_INT32_MAX}]")
        if (type(self.max_spread_tokens) is not int
                or not 0 <= self.max_spread_tokens <= _INT32_MAX):
            raise ValueError(f"max_spread_tokens must be an integer in [0, {_INT32_MAX}]")


@dataclass(frozen=True)
class FluxTargetBatch:
    """Physical call with indices into the original FIFO optimizer batch."""

    target_indices: tuple[int, ...]
    joint_tokens: int


def plan_flux_batches(
    prefix_lengths: tuple[int, ...],
    policy: FluxPackingPolicy,
    *,
    source_ids: tuple[int, ...] | None = None,
    source_bytes: tuple[int, ...] | None = None,
) -> tuple[FluxTargetBatch, ...]:
    """Use the selected deterministic planner without splitting or dropping targets.

    legacy_bfd_v1 preserves the original best-fit-decreasing call and target order.
    balanced_release_v1 prefers few calls, balanced loads, then earlier release.
    Each target costs its prefix plus 900 image tokens. Bounded search tries to
    reduce the best-fit-decreasing call count, then moves/swaps balance that
    fixed count. With source metadata, further moves/swaps and pack reordering
    minimize sum(source_bytes * last_pack_index) within max_spread_tokens.
    A largest pack always runs first, including the balanced fallback when the
    spread bound cannot be met. Targets are never split or dropped.

    Source IDs index source_bytes; omit both to obtain the balanced reference.
    Search is deterministic but bounded: neither minimum call count nor global
    optimality of balancing/release is guaranteed for arbitrary inputs.
    """

    if not isinstance(policy, FluxPackingPolicy):
        raise ValueError("policy must be a FluxPackingPolicy")
    for length in prefix_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= NVARC_MAX_TOKENS:
            raise ValueError(f"prefix lengths must be integers in [1, {NVARC_MAX_TOKENS}]")

    if (source_ids is None) != (source_bytes is None):
        raise ValueError("source_ids and source_bytes must be supplied together")
    if source_ids is not None:
        if (len(source_ids) != len(prefix_lengths)
                or any(type(i) is not int or not 0 <= i < len(source_bytes) for i in source_ids)
                or set(source_ids) != set(range(len(source_bytes)))):
            raise ValueError("source_ids must map every target to a represented source")
        if any(type(size) is not int or size <= 0 for size in source_bytes):
            raise ValueError("source_bytes must contain positive integers")
    if not prefix_lengths:
        return ()

    costs = tuple(length + _FLUX_IMAGE_TOKENS for length in prefix_lengths)
    if policy.algorithm == "legacy_bfd_v1":
        packs = _best_fit_decreasing(costs, policy.max_tokens)
        return tuple(FluxTargetBatch(tuple(pack), sum(costs[i] for i in pack)) for pack in packs)
    packs = _few_packs(costs, policy.max_tokens)
    # Least-loaded placement often avoids the almost empty tail left by BFD.
    balanced = [[] for _ in packs]
    loads = [0 for _ in packs]
    for index in sorted(range(len(costs)), key=lambda i: -costs[i]):
        slot = min(range(len(loads)), key=loads.__getitem__)
        if loads[slot] + costs[index] > policy.max_tokens:
            break
        balanced[slot].append(index)
        loads[slot] += costs[index]
    else:
        if _balance_score(balanced, costs) < _balance_score(packs, costs):
            packs = balanced
    packs = _improve(packs, costs, policy, lambda p: _balance_score(p, costs))
    if source_ids is not None and _balance_score(packs, costs)[0] <= policy.max_spread_tokens:
        def release_score(candidate):
            last = [0] * len(source_bytes)
            for position, pack in enumerate(candidate):
                for index in pack:
                    last[source_ids[index]] = position
            return (sum(size * end for size, end in zip(source_bytes, last)),
                    *_balance_score(candidate, costs))

        packs = _improve(packs, costs, policy, release_score, release=True)
    return tuple(FluxTargetBatch(tuple(pack), sum(costs[i] for i in pack)) for pack in packs)


def _best_fit_decreasing(costs, budget):
    """Preserve the original stable target ordering and earliest-pack tie break."""
    packs, remaining = [], []
    order = sorted(range(len(costs)), key=lambda i: -costs[i])
    for index in order:
        cost = costs[index]
        best = min(
            (pack_index for pack_index, free in enumerate(remaining) if free >= cost),
            key=lambda pack_index: (remaining[pack_index], pack_index),
            default=None,
        )
        if best is None:
            packs.append([index])
            remaining.append(budget - cost)
        else:
            packs[best].append(index)
            remaining[best] -= cost
    return packs


def _few_packs(costs, budget):
    packs = _best_fit_decreasing(costs, budget)
    order = sorted(range(len(costs)), key=lambda i: -costs[i])

    # Try smaller counts from the volume lower bound, sharing one search budget.
    nodes_left = _SEARCH_NODES

    def place(position):
        nonlocal nodes_left
        if position == len(order):
            return True
        if nodes_left <= 0:
            return False
        nodes_left -= 1
        index = order[position]
        seen_loads = set()
        for slot in sorted(range(len(trial)), key=lambda i: -loads[i]):
            load = loads[slot]
            if load in seen_loads or load + costs[index] > budget:
                continue
            seen_loads.add(load)
            trial[slot].append(index)
            loads[slot] += costs[index]
            if place(position + 1):
                return True
            trial[slot].pop()
            loads[slot] -= costs[index]
        return False

    lower_bound = (sum(costs) + budget - 1) // budget
    for count in range(lower_bound, len(packs)):
        trial, loads = [[] for _ in range(count)], [0] * count
        if place(0):
            return trial
        if nodes_left <= 0:
            break
    return packs


def _balance_score(packs, costs):
    loads = [sum(costs[i] for i in pack) for pack in packs]
    return max(loads) - min(loads), sum(load * load for load in loads)


def _largest_first(packs, costs):
    first = max(range(len(packs)), key=lambda p: sum(costs[i] for i in packs[p]))
    return [packs[first], *packs[:first], *packs[first + 1:]]


def _neighbors(packs, costs, policy, *, release):
    """Keep the call count fixed; reject illegal loads before constructing plans."""
    loads = [sum(costs[i] for i in pack) for pack in packs]
    for a, b in combinations(range(len(packs)), 2):
        # None denotes a one-way move. Never empty a physical call.
        for left in (*packs[a], None):
            for right in (*packs[b], None):
                if left is None and right is None:
                    continue
                if (left is None and len(packs[b]) == 1
                        or right is None and len(packs[a]) == 1):
                    continue
                delta = (costs[right] if right is not None else 0) - (costs[left] if left is not None else 0)
                changed = loads.copy()
                changed[a] += delta
                changed[b] -= delta
                if max(changed) > policy.max_tokens:
                    continue
                if release and max(changed) - min(changed) > policy.max_spread_tokens:
                    continue
                candidate = list(packs)
                candidate[a] = [i for i in packs[a] if i != left] + ([] if right is None else [right])
                candidate[b] = [i for i in packs[b] if i != right] + ([] if left is None else [left])
                yield _largest_first(candidate, costs)
        if release:
            candidate = list(packs)
            candidate[a], candidate[b] = candidate[b], candidate[a]
            yield _largest_first(candidate, costs)


def _improve(packs, costs, policy, score, *, release=False):
    packs = _largest_first(packs, costs)
    current_score = score(packs)
    for _ in range(_IMPROVEMENT_PASSES):
        best, best_score = packs, current_score
        for candidate in _neighbors(packs, costs, policy, release=release):
            candidate_score = score(candidate)
            if candidate_score < best_score:
                best, best_score = candidate, candidate_score
        if best_score == current_score:
            break
        packs, current_score = best, best_score
    return packs


__all__ = [
    "FluxPackingPolicy", "FluxTargetBatch", "plan_flux_batches",
]
