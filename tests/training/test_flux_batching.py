"""CPU checks of token-budget packing within fixed optimizer windows."""

from dataclasses import FrozenInstanceError
from functools import lru_cache
from itertools import product
import unittest

from diffusion_arc.training.step.flux_batching import (
    FluxPackingPolicy, FluxTargetBatch, plan_flux_batches,
)


class FluxBatchPlanningTests(unittest.TestCase):
    def test_budget_counts_image_tokens_for_every_target(self):
        policy = FluxPackingPolicy()
        self.assertEqual(policy.max_tokens, 12_288)
        # Text alone fits all three; the three image sequences require two calls.
        self.assertEqual(
            plan_flux_batches((3500, 3500, 3500), policy),
            (FluxTargetBatch((0, 1), 8800), FluxTargetBatch((2,), 4400)),
        )
        self.assertEqual(
            plan_flux_batches((8400,), FluxPackingPolicy(9300)),
            (FluxTargetBatch((0,), 9300),),
        )

    def test_best_fit_backfills_earlier_packs_and_breaks_ties_by_fifo(self):
        policy = FluxPackingPolicy(10_000)
        # Consecutive descending groups would need three calls: 6000, 6000+4000,
        # 4000. Backfilling yields two full calls and preserves both kinds of tie.
        result = plan_flux_batches((5100, 5100, 3100, 3100), policy)
        self.assertEqual(
            result,
            (FluxTargetBatch((0, 2), 10_000), FluxTargetBatch((1, 3), 10_000)),
        )
        self.assertEqual(result, plan_flux_batches((5100, 5100, 3100, 3100), policy))
        # Both packs have room for 1000, but the second is the tighter fit.
        self.assertEqual(
            plan_flux_batches((6100, 4100, 3100, 100), policy),
            (FluxTargetBatch((0,), 7000), FluxTargetBatch((1, 2, 3), 10_000)),
        )

    def test_exact_coverage_budget_and_tail_without_crossing_windows(self):
        policy = FluxPackingPolicy()
        lengths = tuple(1 + index * 173 for index in range(48))
        batches = plan_flux_batches(lengths, policy)
        self.assertEqual(sorted(i for batch in batches for i in batch.target_indices), list(range(48)))
        for batch in batches:
            self.assertEqual(batch.joint_tokens, sum(lengths[i] + 900 for i in batch.target_indices))
            self.assertLessEqual(batch.joint_tokens, policy.max_tokens)
        self.assertEqual(plan_flux_batches((), policy), ())
        tail = (FluxTargetBatch((0,), 901),)
        self.assertEqual(plan_flux_batches((1,), policy), tail)
        self.assertEqual(
            plan_flux_batches((1, 1), policy),
            (FluxTargetBatch((0, 1), 1802),),
        )

    def test_validates_policy_and_prefixes_at_ingestion(self):
        for budget in (True, False, "12288", 12_288.0, 0, -1, 9299, 1 << 31):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                FluxPackingPolicy(budget)
        self.assertEqual(FluxPackingPolicy((1 << 31) - 1).max_tokens, (1 << 31) - 1)
        with self.assertRaises(FrozenInstanceError):
            FluxPackingPolicy().max_tokens = 20_000
        for length in (True, False, "42", 42.0, 0, -1, 8401):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "prefix lengths"):
                plan_flux_batches((1, length), FluxPackingPolicy())
        with self.assertRaisesRegex(ValueError, "policy"):
            plan_flux_batches((1,), object())

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
                batches = plan_flux_batches(tuple(cost - 900 for cost in costs), policy)
                self.assertEqual(len(batches), self.exhaustive_calls(costs, policy.max_tokens))

    def test_heuristic_does_not_claim_global_optimality(self):
        costs = (6000, 5000, 3000, 2000, 2000, 2000)
        policy = FluxPackingPolicy(10_000)
        batches = plan_flux_batches(tuple(cost - 900 for cost in costs), policy)
        self.assertEqual(len(batches), 3)
        # A global search can instead choose 6000+2000+2000 and 5000+3000+2000.
        self.assertEqual(self.exhaustive_calls(costs, policy.max_tokens), 2)


if __name__ == "__main__":
    unittest.main()
