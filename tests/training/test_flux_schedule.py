"""CPU checks of padded and packed scheduling within fixed optimizer windows."""

from dataclasses import FrozenInstanceError
from functools import lru_cache
from itertools import permutations, product
import unittest

from diffusion_arc.training.flux_schedule import (
    FluxBatchShape, FluxPackingPolicy, FluxSchedulingPolicy, FluxTargetBatch,
    schedule_flux_targets, schedule_packed_flux_targets,
)


class FluxScheduleTests(unittest.TestCase):
    def test_costs_can_favor_more_padding_and_reject_greedy_largest_groups(self):
        policy = FluxSchedulingPolicy((
            FluxBatchShape(1, 12, 10), FluxBatchShape(2, 8, 6),
            FluxBatchShape(2, 12, 13), FluxBatchShape(4, 12, 30),
        ))
        batches = schedule_flux_targets((4, 11, 7, 5), policy)
        self.assertEqual(batches, (FluxTargetBatch((1, 2), 12), FluxTargetBatch((3, 0), 8)))
        # Pairing the seven-token target at 12 beats isolating the longest one;
        # one call of four is slower than two calls of two.
        self.assertEqual(schedule_flux_targets((), policy), ())
        self.assertEqual(schedule_flux_targets((3,), policy), (FluxTargetBatch((0,), 12),))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            schedule_flux_targets((13,), policy)

    def test_small_cases_match_exhaustive_groupings(self):
        policy = FluxSchedulingPolicy((
            FluxBatchShape(1, 3, 4), FluxBatchShape(1, 8, 8),
            FluxBatchShape(2, 5, 5), FluxBatchShape(2, 8, 12), FluxBatchShape(4, 8, 19),
        ))
        costs = {(s.batch_size, s.max_tokens): s.forward_backward_ms for s in policy.shapes}

        def exhaustive(remaining):
            if not remaining:
                return 0
            best = float("inf")
            for shape in policy.shapes:
                size = shape.batch_size
                if size <= len(remaining) and max(remaining[:size]) <= shape.max_tokens:
                    best = min(best, shape.forward_backward_ms + exhaustive(remaining[size:]))
            return best

        for lengths in product((2, 5, 8), repeat=4):
            with self.subTest(lengths=lengths):
                batches = schedule_flux_targets(lengths, policy)
                actual = sum(costs[len(b.target_indices), b.padded_length] for b in batches)
                expected = min(exhaustive(order) for order in set(permutations(lengths)))
                self.assertEqual(actual, expected)
                self.assertEqual(sorted(i for b in batches for i in b.target_indices), list(range(4)))
                for b in batches:
                    self.assertLessEqual(max(lengths[i] for i in b.target_indices), b.padded_length)

    def test_ties_are_stable_and_independent_of_catalog_order(self):
        shapes = (FluxBatchShape(1, 8, 2), FluxBatchShape(2, 8, 4))
        # At the suffix, fewer calls wins. Stable sorting retains equal lengths.
        for catalog in (shapes, shapes[::-1]):
            result = schedule_flux_targets((7, 3, 7), FluxSchedulingPolicy(catalog))
            self.assertEqual(len(result), 2)
            self.assertEqual(tuple(i for b in result for i in b.target_indices), (0, 2, 1))
            self.assertEqual(result, schedule_flux_targets((7, 3, 7), FluxSchedulingPolicy(shapes)))

    def test_validates_costs_shapes_duplicates_and_tail_coverage(self):
        for cost in (0, -1, True, float("inf"), float("nan"), "5"):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                FluxBatchShape(1, 8, cost)
        for size, length in ((True, 8), (2.0, 8), (3, 8), (1, True), (1, 0), (1, 8401)):
            with self.subTest(size=size, length=length), self.assertRaises(ValueError):
                FluxBatchShape(size, length, 1)
        for shapes in ((), (object(),), (FluxBatchShape(2, 8, 1),),
                       (FluxBatchShape(1, 8, 1), FluxBatchShape(1, 8, 2))):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                FluxSchedulingPolicy(shapes)


class FluxPackingScheduleTests(unittest.TestCase):
    def test_budget_counts_image_tokens_for_every_target(self):
        policy = FluxPackingPolicy()
        self.assertEqual(policy.max_tokens, 12_288)
        # Text alone fits all three; the three image sequences require two calls.
        self.assertEqual(
            schedule_packed_flux_targets((3500, 3500, 3500), policy),
            (FluxTargetBatch((0, 1), None, 8800), FluxTargetBatch((2,), None, 4400)),
        )
        self.assertEqual(
            schedule_packed_flux_targets((8400,), FluxPackingPolicy(9300)),
            (FluxTargetBatch((0,), None, 9300),),
        )

    def test_best_fit_backfills_earlier_packs_and_breaks_ties_by_fifo(self):
        policy = FluxPackingPolicy(10_000)
        # Consecutive descending groups would need three calls: 6000, 6000+4000,
        # 4000. Backfilling yields two full calls and preserves both kinds of tie.
        result = schedule_packed_flux_targets((5100, 5100, 3100, 3100), policy)
        self.assertEqual(
            result,
            (FluxTargetBatch((0, 2), None, 10_000), FluxTargetBatch((1, 3), None, 10_000)),
        )
        self.assertEqual(result, schedule_packed_flux_targets((5100, 5100, 3100, 3100), policy))
        # Both packs have room for 1000, but the second is the tighter fit.
        self.assertEqual(
            schedule_packed_flux_targets((6100, 4100, 3100, 100), policy),
            (FluxTargetBatch((0,), None, 7000), FluxTargetBatch((1, 2, 3), None, 10_000)),
        )

    def test_exact_coverage_budget_and_tail_without_crossing_windows(self):
        policy = FluxPackingPolicy()
        lengths = tuple(1 + index * 173 for index in range(48))
        batches = schedule_packed_flux_targets(lengths, policy)
        self.assertEqual(sorted(i for batch in batches for i in batch.target_indices), list(range(48)))
        for batch in batches:
            self.assertIsNone(batch.padded_length)
            self.assertEqual(batch.joint_tokens, sum(lengths[i] + 900 for i in batch.target_indices))
            self.assertLessEqual(batch.joint_tokens, policy.max_tokens)
        self.assertEqual(schedule_packed_flux_targets((), policy), ())
        tail = (FluxTargetBatch((0,), None, 901),)
        self.assertEqual(schedule_packed_flux_targets((1,), policy), tail)
        self.assertEqual(
            schedule_packed_flux_targets((1, 1), policy),
            (FluxTargetBatch((0, 1), None, 1802),),
        )

    def test_validates_policy_and_prefixes_at_ingestion(self):
        for budget in (True, False, None, "12288", 12_288.0, 0, -1, 9299, 1 << 31):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                FluxPackingPolicy(budget)
        self.assertEqual(FluxPackingPolicy((1 << 31) - 1).max_tokens, (1 << 31) - 1)
        with self.assertRaises(FrozenInstanceError):
            FluxPackingPolicy().max_tokens = 20_000
        for length in (True, False, None, "42", 42.0, 0, -1, 8401):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "prefix lengths"):
                schedule_packed_flux_targets((1, length), FluxPackingPolicy())
        with self.assertRaisesRegex(ValueError, "policy"):
            schedule_packed_flux_targets((1,), object())

    @staticmethod
    def exhaustive_calls(costs, budget):
        """Enumerate all legal subsets, independent of greedy item placement."""
        @lru_cache(None)
        def solve(mask):
            if not mask:
                return 0
            first = mask & -mask
            best = len(costs)
            subset = mask
            while subset:
                if subset & first and sum(cost for i, cost in enumerate(costs) if subset & (1 << i)) <= budget:
                    best = min(best, 1 + solve(mask ^ subset))
                subset = (subset - 1) & mask
            return best
        return solve((1 << len(costs)) - 1)

    def test_small_cases_match_exhaustive_optimum(self):
        policy = FluxPackingPolicy(10_000)
        for costs in product((2000, 3000, 6000, 8000), repeat=4):
            with self.subTest(costs=costs):
                batches = schedule_packed_flux_targets(tuple(cost - 900 for cost in costs), policy)
                self.assertEqual(len(batches), self.exhaustive_calls(costs, policy.max_tokens))

    def test_heuristic_does_not_claim_global_optimality(self):
        costs = (6000, 5000, 3000, 2000, 2000, 2000)
        policy = FluxPackingPolicy(10_000)
        batches = schedule_packed_flux_targets(tuple(cost - 900 for cost in costs), policy)
        self.assertEqual(len(batches), 3)
        # A global search can instead choose 6000+2000+2000 and 5000+3000+2000.
        self.assertEqual(self.exhaustive_calls(costs, policy.max_tokens), 2)


if __name__ == "__main__":
    unittest.main()
