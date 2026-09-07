"""Independent packing checks for the bounded NVARC scheduler."""

from dataclasses import replace
from itertools import combinations, product
import unittest

from diffusion_arc.training.nvarc_schedule import (
    NvarcBatchShape, NvarcSchedulingPolicy, schedule_nvarc_sources,
)


def _policy(overhead=3):
    return NvarcSchedulingPolicy(
        (NvarcBatchShape(2, 4), NvarcBatchShape(2, 10), NvarcBatchShape(4, 7)),
        call_overhead_tokens=overhead,
    )


def _exhaustive_cost(lengths, policy):
    """Enumerate arbitrary source subsets, without sorting or the production DP."""

    def visit(remaining, singleton_left):
        if not remaining:
            return (0, 0) if not singleton_left else (float("inf"), 0)
        candidates = [(s.batch_size, s.max_tokens) for s in policy.shapes]
        if singleton_left:
            candidates.append((1, lengths[remaining[0]]))
        scores = []
        for size, boundary in candidates:
            for rest in combinations(remaining[1:], size - 1):
                group = (remaining[0], *rest)
                if max(lengths[i] for i in group) > boundary:
                    continue
                tail = tuple(i for i in remaining if i not in group)
                cost, calls = visit(tail, singleton_left - (size == 1))
                scores.append((cost + size * boundary + policy.call_overhead_tokens, calls + 1))
        return min(scores, default=(float("inf"), 0))

    return visit(tuple(range(len(lengths))), len(lengths) % 2)


class NvarcScheduleTests(unittest.TestCase):
    def test_matches_exhaustive_arbitrary_groups_for_small_windows(self):
        policy = _policy()
        for count in range(7):
            for lengths in product((3, 6, 9), repeat=count):
                plan = schedule_nvarc_sources(lengths, policy)
                self.assertEqual(
                    (sum(len(b.source_indices) * b.padded_length for b in plan)
                     + len(plan) * policy.call_overhead_tokens, len(plan)),
                    _exhaustive_cost(lengths, policy),
                )
                self.assertEqual(sorted(i for b in plan for i in b.source_indices), list(range(count)))
                self.assertEqual(sum(len(b.source_indices) == 1 for b in plan), count % 2)

    def test_optimizes_whole_window_instead_of_always_taking_largest_batch(self):
        policy = replace(_policy(overhead=0), shapes=(
            NvarcBatchShape(2, 4), NvarcBatchShape(2, 10), NvarcBatchShape(4, 8),
        ))
        plan = schedule_nvarc_sources((6, 3, 6, 3, 3, 3), policy)
        self.assertEqual([(len(b.source_indices), b.padded_length) for b in plan],
                         [(2, 10), (2, 4), (2, 4)])
        self.assertEqual(plan[0].source_indices, (0, 2))

    def test_equal_cost_prefers_fewer_calls_and_equal_lengths_keep_source_order(self):
        policy = _policy(overhead=12)
        plan = schedule_nvarc_sources((3,) * 4, policy)
        self.assertEqual([(b.source_indices, b.padded_length) for b in plan], [((0, 1, 2, 3), 7)])
        self.assertEqual(plan, schedule_nvarc_sources((3,) * 4, replace(policy, shapes=policy.shapes[::-1])))

    def test_singleton_can_cover_longest_source_but_never_appears_in_even_window(self):
        odd = schedule_nvarc_sources((3, 10, 3), _policy())
        self.assertEqual([(b.source_indices, b.padded_length) for b in odd], [((1,), 10), ((0, 2), 4)])
        even = schedule_nvarc_sources((3, 10), _policy())
        self.assertEqual([(b.source_indices, b.padded_length) for b in even], [((1, 0), 10)])
        for count in range(1, 17):
            plan = schedule_nvarc_sources((10,) * count, _policy())
            self.assertEqual(sum(len(b.source_indices) == 1 for b in plan), count % 2)
            self.assertEqual(sum(len(b.source_indices) for b in plan), count)

    def test_singleton_uses_real_length_in_execution_plan_and_cost(self):
        full_length_policy = NvarcSchedulingPolicy((NvarcBatchShape(2, 8400),), 2560)
        for length in (1, 400, 8400):
            with self.subTest(length=length):
                plan = schedule_nvarc_sources((length,), full_length_policy)
                self.assertEqual([(b.source_indices, b.padded_length) for b in plan], [((0,), length)])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            schedule_nvarc_sources((8401,), full_length_policy)
        # Exact singleton cost makes the shortest source the best one to run
        # alone here, even though the scheduler visits the longest first.
        plan = schedule_nvarc_sources((3, 6, 6), _policy())
        self.assertEqual([(b.source_indices, b.padded_length) for b in plan],
                         [((1, 2), 10), ((0,), 3)])

    def test_rejects_invalid_catalog_and_out_of_range_sources(self):
        for size, length in ((True, 10), (1, 10), (3, 10), (16, 10), (2, 0), (2, 8401), (2, 3.5)):
            with self.subTest(size=size, length=length), self.assertRaises(ValueError):
                NvarcBatchShape(size, length)
        for changes in (
            {"shapes": ()}, {"shapes": _policy().shapes * 2},
            {"shapes": (NvarcBatchShape(4, 10),)},
            {"call_overhead_tokens": True}, {"call_overhead_tokens": -1},
            {"shapes": (NvarcBatchShape(2, 4), NvarcBatchShape(4, 10))},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(_policy(), **changes)
        for lengths in ((0,), (11,), (True,), (1.5,)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                schedule_nvarc_sources(lengths, _policy())


if __name__ == "__main__":
    unittest.main()
