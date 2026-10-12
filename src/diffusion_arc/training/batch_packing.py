"""Bounded CPU assignment and packing for complete optimizer batches.

Inputs contain only prepared token costs. Targets remain indivisible and every
source stays on one rank. Plans minimize collective slots, then aligned token
work; the score does not estimate CUDA execution time.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations
import math


@dataclass(frozen=True)
class Source:
    key: str
    target_tokens: tuple[int, ...]


@dataclass(frozen=True)
class Pack:
    target_ids: tuple[int, ...]
    joint_tokens: int


@dataclass(frozen=True)
class PackingPlan:
    rank_sources: tuple[tuple[int, ...], ...]
    rank_packs: tuple[tuple[Pack, ...], ...]


@dataclass(frozen=True)
class SearchBudget:
    source_pool_size: int = 16
    source_candidates: int = 16
    source_passes: int = 2
    pack_passes: int = 3
    completion_nodes: int = 2_000
    completion_finalists: int = 2

    def __post_init__(self):
        if type(self.source_pool_size) is not int or not 2 <= self.source_pool_size <= 20:
            raise ValueError("source_pool_size must be an integer in [2, 20]")
        for name in ("source_candidates", "completion_finalists"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("source_passes", "pack_passes", "completion_nodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def _target_layout(sources):
    costs, source_targets = [], []
    for source in sources:
        source_targets.append(tuple(range(len(costs), len(costs) + len(source.target_tokens))))
        costs.extend(source.target_tokens)
    return tuple(costs), tuple(source_targets)


def _sort_packs(packs):
    return tuple(sorted(packs, key=lambda p: -p.joint_tokens))


def _make_pack(ids, costs):
    return Pack(tuple(sorted(ids)), sum(costs[i] for i in ids))


def _bfd(ids, costs, capacity):
    packs, loads = [], []
    for index in sorted(ids, key=lambda i: (-costs[i], i)):
        eligible = [j for j, load in enumerate(loads) if load + costs[index] <= capacity]
        slot = min(eligible, key=lambda j: (-loads[j], j), default=None)
        if slot is None:
            packs.append([index])
            loads.append(costs[index])
        else:
            packs[slot].append(index)
            loads[slot] += costs[index]
    return _sort_packs(_make_pack(p, costs) for p in packs)


def _replace_rank(plan, rank, packs):
    ranks = list(plan.rank_packs)
    ranks[rank] = _sort_packs(packs)
    return PackingPlan(plan.rank_sources, tuple(ranks))


def _profiles(plan):
    return tuple(tuple(p.joint_tokens for p in packs) for packs in plan.rank_packs)


def plan_score(plan):
    """Prefer fewer collective slots, then smaller slot maxima and variance.

    Dummy slots contribute zero logical workload. This is a token-work proxy,
    not a prediction of attention time or FSDP communication duration.
    """
    profiles = _profiles(plan)
    count, ranks = max(map(len, profiles)), len(profiles)
    critical = variance = 0
    for slot in range(count):
        loads = [p[slot] if slot < len(p) else 0 for p in profiles]
        critical += max(loads)
        variance += ranks * sum(x * x for x in loads) - sum(loads) ** 2
    return count, critical, variance


def validate_plan(sources, plan, max_tokens):
    """Independently check coverage, whole-source ownership and physical loads."""
    costs, source_targets = _target_layout(sources)
    if len(plan.rank_sources) != len(plan.rank_packs) or not plan.rank_sources:
        raise ValueError("rank source/pack counts must agree and be nonempty")
    seen_sources, seen_targets = [], []
    for source_ids, packs in zip(plan.rank_sources, plan.rank_packs, strict=True):
        if not source_ids or not packs:
            raise ValueError("every rank must contain sources and packs")
        if any(type(s) is not int or not 0 <= s < len(sources) for s in source_ids):
            raise ValueError("invalid source index")
        seen_sources.extend(source_ids)
        local = {i for s in source_ids for i in source_targets[s]}
        previous = max_tokens
        for pack in packs:
            if (not pack.target_ids or any(type(i) is not int or i not in local for i in pack.target_ids)):
                raise ValueError("pack targets must belong to this rank's sources")
            actual = sum(costs[i] for i in pack.target_ids)
            if type(pack.joint_tokens) is not int or actual != pack.joint_tokens:
                raise ValueError("pack token total does not match its targets")
            if not 0 < actual <= max_tokens:
                raise ValueError("pack exceeds the token capacity")
            if actual > previous:
                raise ValueError("packs must be in descending token order")
            previous = actual
            seen_targets.extend(pack.target_ids)
    if sorted(seen_sources) != list(range(len(sources))):
        raise ValueError("sources must appear exactly once")
    if sorted(seen_targets) != list(range(len(costs))):
        raise ValueError("targets must appear exactly once")


def _baseline(sources, ranks, costs, source_targets, capacity):
    owners, loads = [[] for _ in range(ranks)], [0] * ranks
    source_costs = [sum(s.target_tokens) for s in sources]
    for source in sorted(range(len(sources)), key=lambda s: (-source_costs[s], s)):
        rank = min(range(ranks), key=lambda r: (loads[r], len(owners[r]), r))
        owners[rank].append(source)
        loads[rank] += source_costs[source]
    owners = tuple(tuple(sorted(o)) for o in owners)
    packs = tuple(_bfd([i for s in owner for i in source_targets[s]], costs, capacity) for owner in owners)
    return PackingPlan(owners, packs)


def _fit_bins(ids, costs, capacity, count, nodes):
    """Try one smaller bin count with a deterministic branch-node limit."""
    if sum(costs[i] for i in ids) > count * capacity or count <= 0 or not nodes:
        return None
    order = sorted(ids, key=lambda i: (-costs[i], i))
    # A balanced start plus exact two-bin repair often avoids branching entirely.
    balanced, balanced_loads = [[] for _ in range(count)], [0] * count
    for index in order:
        slot = min(range(count), key=lambda j: (balanced_loads[j], j))
        balanced[slot].append(index)
        balanced_loads[slot] += costs[index]
    for a, b in combinations(range(count), 2):
        if max(balanced_loads[a], balanced_loads[b]) <= capacity:
            continue
        total = balanced_loads[a] + balanced_loads[b]
        if total > 2 * capacity:
            continue
        union = tuple(sorted(balanced[a] + balanced[b]))
        histories = _reachable_sums(union, costs, capacity)
        low, high = max(1, total - capacity), min(capacity, total - 1)
        choices = _near_sums(histories[-1], (total // 2, total // 2 + 1), low, high)
        if choices:
            value = min(choices, key=lambda x: (abs(2 * x - total), x))
            selected = _reconstruct_subset(union, costs, histories, value)
            balanced[a], balanced[b] = sorted(selected), sorted(set(union) - selected)
            balanced_loads[a], balanced_loads[b] = value, total - value
    if all(0 < load <= capacity for load in balanced_loads):
        return _sort_packs(_make_pack(p, costs) for p in balanced)
    trial, loads = [[] for _ in range(count)], [0] * count

    def place(position):
        nonlocal nodes
        if position == len(order):
            return True
        if nodes <= 0:
            return False
        nodes -= 1
        index, seen = order[position], set()
        for slot in sorted(range(count), key=lambda j: (-loads[j], j)):
            load = loads[slot]
            if load in seen or load + costs[index] > capacity:
                continue
            seen.add(load)
            trial[slot].append(index)
            loads[slot] += costs[index]
            if place(position + 1):
                return True
            trial[slot].pop()
            loads[slot] -= costs[index]
        return False

    if place(0):
        return _sort_packs(_make_pack(p, costs) for p in trial if p)
    return None


def _repair_count(plan, costs, capacity, goal, nodes, ranks=None):
    """Repair all violating ranks together, preserving intermediate progress."""
    changed = plan
    for rank in (range(len(plan.rank_packs)) if ranks is None else ranks):
        packs = changed.rank_packs[rank]
        if len(packs) > goal:
            fitted = _fit_bins([i for p in packs for i in p.target_ids], costs, capacity, goal, nodes)
            if fitted is not None:
                changed = _replace_rank(changed, rank, fitted)
    return changed


def _reachable_sums(ids, costs, capacity):
    """Bitset DP; histories reconstruct one exact target subset per sum."""
    mask, histories = (1 << (capacity + 1)) - 1, [1]
    for index in ids:
        bits = histories[-1]
        histories.append((bits | bits << costs[index]) & mask)
    return histories


def _near_sums(bits, points, low, high):
    bits &= ((1 << (high + 1)) - 1) ^ ((1 << low) - 1)
    if not bits:
        return []
    found = set()
    # Distinct profile breakpoints often collapse to the same feasible bound.
    for point in {max(low, min(high, int(point))) for point in points}:
        below = bits & ((1 << (point + 1)) - 1)
        if below:
            found.add(below.bit_length() - 1)
        above = bits >> point
        if above:
            found.add(point + (above & -above).bit_length() - 1)
    return sorted(found)


def _reconstruct_subset(ids, costs, histories, total):
    selected = []
    for position in range(len(ids) - 1, -1, -1):
        if not (histories[position] >> total) & 1:
            index = ids[position]
            selected.append(index)
            total -= costs[index]
    if total:
        raise RuntimeError("unreachable subset sum")
    return set(selected)


def _other_rank_loads(plan, rank):
    others = [p for r, p in enumerate(_profiles(plan)) if r != rank]
    count = max(map(len, others), default=0)
    maximum, sums, squares = [], [], []
    for slot in range(max(count, len(plan.rank_packs[rank]))):
        loads = [p[slot] if slot < len(p) else 0 for p in others]
        maximum.append(max(loads, default=0))
        sums.append(sum(loads))
        squares.append(sum(x * x for x in loads))
    return count, maximum, sums, squares


def _rank_score(loads, ranks, other):
    count, maximum, sums, squares = other
    slots = max(count, len(loads))
    critical = variance = 0
    for j in range(slots):
        value = loads[j] if j < len(loads) else 0
        critical += max(value, maximum[j])
        variance += ranks * (value * value + squares[j]) - (value + sums[j]) ** 2
    return slots, critical, variance


def _repair_pack_pairs(plan, costs, capacity, passes, ranks=None):
    """Repartition two local packs via DP against the complete slot envelope.

    Slot-max cost is piecewise linear in a subset's token sum. Query reachable
    sums around its ordering/max breakpoints and the variance minima, instead
    of iterating over every sum. All targets remain on their original rank.
    """
    ranks = tuple(range(len(plan.rank_packs)) if ranks is None else ranks)
    for _ in range(passes):
        before = plan_score(plan)
        for rank in ranks:
            other = _other_rank_loads(plan, rank)
            # Pair count is fixed for this sweep; call-count repair runs separately.
            for a, b in combinations(range(len(plan.rank_packs[rank])), 2):
                packs = plan.rank_packs[rank]
                ids = tuple(sorted(packs[a].target_ids + packs[b].target_ids))
                total = packs[a].joint_tokens + packs[b].joint_tokens
                low, high = max((total + 1) // 2, total - capacity, 1), min(capacity, total - 1)
                remaining = [p.joint_tokens for j, p in enumerate(packs) if j not in (a, b)]
                points = {low, high, packs[a].joint_tokens, total // 2}
                for value in remaining + other[1]:
                    points.update((value, total - value))
                if len(plan.rank_packs) > 1:
                    denominator = 2 * (len(plan.rank_packs) - 1)
                    for x in other[2]:
                        for y in other[2]:
                            vertex = ((len(plan.rank_packs) - 1) * total + x - y) // denominator
                            points.update((vertex, vertex + 1))
                histories = _reachable_sums(ids, costs, capacity)
                best_score, best_sum = plan_score(plan), None
                for value in _near_sums(histories[-1], points, low, high):
                    loads = sorted(remaining + [value, total - value], reverse=True)
                    score = _rank_score(loads, len(plan.rank_packs), other)
                    if score < best_score:
                        best_score, best_sum = score, value
                if best_sum is not None:
                    selected = _reconstruct_subset(ids, costs, histories, best_sum)
                    new = [p for j, p in enumerate(packs) if j not in (a, b)]
                    new.extend((_make_pack(selected, costs), _make_pack(set(ids) - selected, costs)))
                    plan = _replace_rank(plan, rank, new)
        if plan_score(plan) == before:
            break
    return plan


def _local(plan, costs, capacity, budget):
    best = plan
    lower = max(math.ceil(sum(p.joint_tokens for p in packs) / capacity) for packs in plan.rank_packs)
    for goal in range(plan_score(plan)[0] - 1, lower - 1, -1):
        # Commit the whole attempt after every rank has had its completion search.
        trial = _repair_count(best, costs, capacity, goal, budget.completion_nodes)
        if plan_score(trial) < plan_score(best):
            best = trial
    return _repair_pack_pairs(best, costs, capacity, budget.pack_passes)


def _subset_list(source_ids, source_costs):
    result = [(0, 0)]
    for bit, source in enumerate(source_ids):
        result += [(value + source_costs[source], mask | (1 << bit)) for value, mask in result]
    return result


def _source_pool(plan, a, b, source_costs, size, sweep):
    """Select a fixed-size diverse pool; every other source stays on its rank."""
    quotas = [min(len(plan.rank_sources[a]), size // 2), min(len(plan.rank_sources[b]), size - size // 2)]
    for index, rank in enumerate((a, b)):
        quotas[index] = min(len(plan.rank_sources[rank]), quotas[index] + size - sum(quotas))
    selected = []
    for rank, quota in zip((a, b), quotas, strict=True):
        order = sorted(plan.rank_sources[rank], key=lambda s: (-source_costs[s], s))
        # Alternating large/small sources keeps swaps useful on larger local batches.
        diverse = [v for pair in zip(order[: (len(order) + 1) // 2], reversed(order[(len(order) + 1) // 2 :])) for v in pair]
        if len(order) % 2:
            diverse.append(order[len(order) // 2])
        offset = (sweep * quota) % len(diverse)
        selected.extend((diverse + diverse)[offset : offset + quota])
    return tuple(selected)


def _source_candidates(plan, a, b, source_costs, pool_size, candidate_count, sweep):
    pool = _source_pool(plan, a, b, source_costs, pool_size, sweep)
    pool_set = set(pool)
    fixed_a = tuple(s for s in plan.rank_sources[a] if s not in pool_set)
    fixed_b = tuple(s for s in plan.rank_sources[b] if s not in pool_set)
    cost_a, cost_b = sum(source_costs[s] for s in fixed_a), sum(source_costs[s] for s in fixed_b)
    pool_total = sum(source_costs[s] for s in pool)
    middle = len(pool) // 2
    left = _subset_list(pool[:middle], source_costs)
    right = sorted(_subset_list(pool[middle:], source_costs))
    values = [value for value, _ in right]
    candidates = {}
    for left_sum, left_mask in left:
        target = (pool_total + cost_b - cost_a) / 2 - left_sum
        position = bisect_left(values, target)
        for j in range(max(0, position - 2), min(len(right), position + 3)):
            right_sum, right_mask = right[j]
            mask = left_mask | (right_mask << middle)
            count = mask.bit_count()
            if (not fixed_a and count == 0) or (not fixed_b and count == len(pool)):
                continue
            difference = abs(2 * (cost_a + left_sum + right_sum) - (cost_a + cost_b + pool_total))
            count_difference = abs(len(fixed_a) + count - len(fixed_b) - len(pool) + count)
            candidates[mask] = difference, count_difference, mask
    owners = []
    for _, _, mask in sorted(candidates.values())[:candidate_count]:
        new_a = tuple(sorted(fixed_a + tuple(s for bit, s in enumerate(pool) if mask >> bit & 1)))
        new_b = tuple(sorted(fixed_b + tuple(s for bit, s in enumerate(pool) if not mask >> bit & 1)))
        owners.append((new_a, new_b))
    return owners


def _phase_score(plan, goal, capacity):
    if goal is None:
        return plan_score(plan)
    counts = [len(p) for p in plan.rank_packs]
    excess = sum(max(0, sum(p.joint_tokens for p in packs) - goal * capacity) for packs in plan.rank_packs)
    # Progress on one bottleneck rank may be needed before global K can fall.
    return sum(c > goal for c in counts), excess, sum(max(0, c - goal) for c in counts), *plan_score(plan)


def _gpu_pairs(plan, sweep):
    profiles = _profiles(plan)
    count = max(map(len, profiles))
    pairs = list(combinations(range(len(profiles)), 2))
    def distance(pair):
        a, b = pair
        return sum(abs((profiles[a][j] if j < len(profiles[a]) else 0) -
                       (profiles[b][j] if j < len(profiles[b]) else 0)) for j in range(count))
    # Cover every rank before adding the most imbalanced pairs. Rotate the
    # matching so equal profiles do not repeatedly produce the same star.
    circle = list(range(len(profiles)))
    if len(circle) % 2:
        circle.append(None)
    rotating = circle[1:]
    shift = sweep % len(rotating)
    circle = [circle[0], *(rotating[-shift:] + rotating[:-shift] if shift else rotating)]
    selected = [tuple(sorted((a, b))) for a, b in zip(circle[:len(circle) // 2],
                reversed(circle[len(circle) // 2:]), strict=True) if a is not None and b is not None]
    covered = {rank for pair in selected for rank in pair}
    for rank in set(range(len(profiles))) - covered:
        selected.append(min((p for p in pairs if rank in p), key=lambda p: (-distance(p), p)))
    for pair in sorted(pairs, key=lambda pair: (-distance(pair), pair)):
        if pair not in selected:
            selected.append(pair)
        if len(selected) >= len(profiles):
            break
    return selected


def _source_search(plan, sources, costs, source_targets, capacity, budget, goal):
    best, work = plan, plan
    source_costs = tuple(sum(s.target_tokens) for s in sources)
    for sweep in range(budget.source_passes):
        for a, b in _gpu_pairs(work, sweep):
            screened = []
            for owner_a, owner_b in _source_candidates(work, a, b, source_costs,
                    budget.source_pool_size, budget.source_candidates, sweep):
                owners, packs = list(work.rank_sources), list(work.rank_packs)
                owners[a], owners[b] = owner_a, owner_b
                for rank in (a, b):
                    packs[rank] = _bfd([i for s in owners[rank] for i in source_targets[s]], costs, capacity)
                trial = PackingPlan(tuple(owners), tuple(packs))
                screened.append((_phase_score(trial, goal, capacity), trial))
            screened.sort(key=lambda item: item[0])
            finalists = [trial for _, trial in screened[:budget.completion_finalists]]
            finalists.append(work)
            for trial in finalists:
                target = goal if goal is not None else plan_score(trial)[0] - 1
                trial = _repair_count(trial, costs, capacity, target, budget.completion_nodes, (a, b))
                trial = _repair_pack_pairs(trial, costs, capacity, 1, (a, b))
                if _phase_score(trial, goal, capacity) < _phase_score(work, goal, capacity):
                    work = trial
                if plan_score(trial) < plan_score(best):
                    best = trial
        if goal is not None:
            work = _repair_count(work, costs, capacity, goal, budget.completion_nodes)
            if plan_score(work) < plan_score(best):
                best = work
        # A later sweep examines different pools/matchings even after no progress.
    return best


def plan_batch(sources, world_size, max_tokens, strategy="joint", budget=SearchBudget()):
    """Return a deterministic bounded plan without using the training pipeline."""
    sources = tuple(sources)
    if type(world_size) is not int or not 1 <= world_size <= len(sources):
        raise ValueError("world_size must be positive and not exceed the source count")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if strategy not in ("baseline", "local", "joint"):
        raise ValueError("strategy must be baseline, local or joint")
    if len({s.key for s in sources}) != len(sources):
        raise ValueError("source keys must be unique within a batch")
    if any(not s.target_tokens or any(type(t) is not int or not 0 < t <= max_tokens
                                    for t in s.target_tokens) for s in sources):
        raise ValueError("targets must have positive integer costs within capacity")
    costs, source_targets = _target_layout(sources)
    initial = _baseline(sources, world_size, costs, source_targets, max_tokens)
    if strategy == "baseline":
        return initial
    best = _local(initial, costs, max_tokens, budget)
    if strategy == "joint" and world_size > 1:
        lower = math.ceil(sum(costs) / (world_size * max_tokens))
        for goal in range(plan_score(best)[0] - 1, lower - 1, -1):
            trial = _source_search(best, sources, costs, source_targets, max_tokens, budget, goal)
            if plan_score(trial) < plan_score(best):
                best = trial
            if plan_score(best)[0] > goal:
                break
        best = _source_search(best, sources, costs, source_targets, max_tokens, budget, None)
        best = _repair_pack_pairs(best, costs, max_tokens, budget.pack_passes)
    return best if plan_score(best) < plan_score(initial) else initial
