"""CPU checks of token-budget packing within fixed optimizer windows."""

from dataclasses import FrozenInstanceError
from functools import lru_cache
from itertools import product
import unittest
from unittest.mock import patch

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

    def test_balances_without_adding_calls_and_breaks_equal_length_ties_by_fifo(self):
        policy = FluxPackingPolicy(10_000)
        # Consecutive descending groups would need three calls: 6000, 6000+4000,
        # 4000. Backfilling yields two full calls and preserves both kinds of tie.
        result = plan_flux_batches((5100, 5100, 3100, 3100), policy)
        self.assertEqual(
            result,
            (FluxTargetBatch((0, 2), 10_000), FluxTargetBatch((1, 3), 10_000)),
        )
        self.assertEqual(result, plan_flux_batches((5100, 5100, 3100, 3100), policy))
        # Move the small target out of the full pack, then run the largest first.
        self.assertEqual(
            plan_flux_batches((6100, 4100, 3100, 100), policy),
            (FluxTargetBatch((1, 2), 9000), FluxTargetBatch((0, 3), 8000)),
        )

    def test_exact_coverage_budget_and_tail_without_crossing_windows(self):
        policy = FluxPackingPolicy()
        lengths = tuple(1 + index * 173 for index in range(48))
        batches = plan_flux_batches(lengths, policy)
        self.assertEqual(batches[0].joint_tokens, max(b.joint_tokens for b in batches))
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
        for spread in (True, -1, 1.5, "1000", 1 << 31):
            with self.subTest(spread=spread), self.assertRaisesRegex(ValueError, "max_spread_tokens"):
                FluxPackingPolicy(max_spread_tokens=spread)
        self.assertEqual(FluxPackingPolicy(max_spread_tokens=0).max_spread_tokens, 0)

    def test_source_metadata_is_complete_and_validated(self):
        for metadata in (
            {"source_ids": (0,)}, {"source_bytes": (1,)},
            {"source_ids": (), "source_bytes": (1,)},
            {"source_ids": (True,), "source_bytes": (1,)},
            {"source_ids": (-1,), "source_bytes": (1,)},
            {"source_ids": (1,), "source_bytes": (1,)},
            {"source_ids": (0,), "source_bytes": (1, 2)},
            {"source_ids": (0,), "source_bytes": (0,)},
            {"source_ids": (0,), "source_bytes": (True,)},
        ):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "source_"):
                plan_flux_batches((1000,), FluxPackingPolicy(), **metadata)
        self.assertEqual(plan_flux_batches((), FluxPackingPolicy(), source_ids=(), source_bytes=()), ())

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

    def test_count_search_improves_on_best_fit_decreasing(self):
        costs = (6000, 5000, 3000, 2000, 2000, 2000)
        policy = FluxPackingPolicy(10_000)
        batches = plan_flux_batches(tuple(cost - 900 for cost in costs), policy)
        self.assertEqual(len(batches), 2)
        # A global search can instead choose 6000+2000+2000 and 5000+3000+2000.
        self.assertEqual(self.exhaustive_calls(costs, policy.max_tokens), 2)

    def test_exhausted_search_retains_valid_fallback(self):
        prefixes = (5100, 4100, 2100, 1100, 1100, 1100)
        with patch("diffusion_arc.training.step.flux_batching._SEARCH_NODES", 0):
            batches = plan_flux_batches(prefixes, FluxPackingPolicy(10_000))
        self.assertEqual(len(batches), 3)
        self.assertEqual(sorted(i for b in batches for i in b.target_indices), list(range(6)))
        self.assertLessEqual(batches[0].joint_tokens, 10_000)
        self.assertEqual(batches[0].joint_tokens, max(b.joint_tokens for b in batches))

    def test_byte_weighted_release_orders_equal_largest_packs(self):
        policy = FluxPackingPolicy(10_000)
        prefixes = (5100, 5100, 5100)
        result = plan_flux_batches(prefixes, policy, source_ids=(0, 1, 2), source_bytes=(1, 10, 100))
        self.assertEqual(tuple(b.target_indices for b in result), ((2,), (1,), (0,)))
        # A smaller but heavier source cannot displace the largest first pack.
        result = plan_flux_batches((5500, 5100), policy, source_ids=(0, 1), source_bytes=(1, 100))
        self.assertEqual(tuple(b.target_indices for b in result), ((0,), (1,)))

    def test_impossible_spread_keeps_balanced_count_instead_of_adding_calls(self):
        prefixes = (3500, 3500, 3500)
        policy = FluxPackingPolicy()
        balanced = plan_flux_batches(prefixes, policy)
        result = plan_flux_batches(prefixes, policy, source_ids=(0, 1, 2), source_bytes=(1, 10, 100))
        self.assertEqual(result, balanced)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(b.joint_tokens for b in result), (8800, 4400))

    def test_oom_batch_rank1_finishes_sources_earlier_within_spread(self):
        prefixes = (405, 532, 777, 1598, 2095, 2735, 1617, 1986, 2640,
                    1855, 2508, 3284, 1548, 2166, 2784, 1510, 2254, 3152,
                    700, 1008, 1180, 452, 662, 982)
        ids = tuple(i // 3 for i in range(24))
        sizes = tuple(max(prefixes[i:i+3]) * 7680 * 2 for i in range(0, 24, 3))
        policy = FluxPackingPolicy()
        balanced = plan_flux_batches(prefixes, policy)
        result = plan_flux_batches(prefixes, policy, source_ids=ids, source_bytes=sizes)
        self.assertEqual(result, plan_flux_batches(prefixes, policy, source_ids=ids, source_bytes=sizes))
        # The volume lower bound certifies six calls for this real workload.
        lower_bound = (sum(prefixes) + 24 * 900 + policy.max_tokens - 1) // policy.max_tokens
        self.assertEqual(len(result), lower_bound)
        self.assertEqual(len(result), len(balanced))
        loads = [b.joint_tokens for b in result]
        self.assertEqual(loads[0], max(loads))
        self.assertLessEqual(max(loads), policy.max_tokens)
        self.assertLessEqual(max(loads) - min(loads), policy.max_spread_tokens)
        self.assertEqual(sorted(i for b in result for i in b.target_indices), list(range(24)))
        for batch in result:
            self.assertEqual(batch.joint_tokens, sum(prefixes[i] + 900 for i in batch.target_indices))

        def retention(packs):
            # Integrate live source bytes over pack boundaries, independently of
            # the planner's last-use calculation.
            pending = set(range(24))
            retained = 0
            for batch in packs:
                pending.difference_update(batch.target_indices)
                retained += sum(sizes[source] for source in {ids[i] for i in pending})
            return retained

        self.assertLess(retention(result), retention(balanced))
        strict = FluxPackingPolicy(max_spread_tokens=0)
        self.assertEqual(plan_flux_batches(prefixes, strict, source_ids=ids, source_bytes=sizes),
                         plan_flux_batches(prefixes, strict))


if __name__ == "__main__":
    unittest.main()
