"""Independent contract and objective checks for the standalone CPU planner."""

from dataclasses import replace
from itertools import combinations
import unittest

from diffusion_arc.training.batch_packing import (
    Pack,
    PackingPlan,
    SearchBudget,
    Source,
    _fit_bins,
    _gpu_pairs,
    plan_batch,
    plan_score,
    validate_plan,
)


class DistributedPackingDiagnosticTests(unittest.TestCase):
    def assert_plan_contract(self, sources, plan, world_size, capacity):
        """Rebuild ownership and costs from inputs without planner helpers."""
        target_costs, target_sources = [], []
        for source_id, source in enumerate(sources):
            target_costs.extend(source.target_tokens)
            target_sources.extend([source_id] * len(source.target_tokens))
        self.assertEqual(len(plan.rank_sources), world_size)
        self.assertEqual(len(plan.rank_packs), world_size)
        self.assertEqual(sorted(source_id for rank in plan.rank_sources for source_id in rank),
                         list(range(len(sources))))
        all_targets = []
        profiles = []
        for source_ids, packs in zip(plan.rank_sources, plan.rank_packs, strict=True):
            self.assertTrue(source_ids)
            self.assertTrue(packs)
            loads = []
            for pack in packs:
                self.assertTrue(pack.target_ids)
                for target_id in pack.target_ids:
                    self.assertIn(target_id, range(len(target_costs)))
                    self.assertIn(target_sources[target_id], source_ids)
                load = sum(target_costs[target_id] for target_id in pack.target_ids)
                self.assertEqual(pack.joint_tokens, load)
                self.assertLessEqual(load, capacity)
                all_targets.extend(pack.target_ids)
                loads.append(load)
            self.assertEqual(loads, sorted(loads, reverse=True))
            profiles.append(loads)
        self.assertEqual(sorted(all_targets), list(range(len(target_costs))))
        slots = max(map(len, profiles))
        critical_tokens = sum(max(profile[slot] if slot < len(profile) else 0
                                  for profile in profiles) for slot in range(slots))
        self.assertEqual(plan_score(plan)[:2], (slots, critical_tokens))
        validate_plan(sources, plan, capacity)

    def test_baseline_preserves_whole_source_lpt_and_known_bfd_assignment(self):
        sources = tuple(Source(str(index), costs) for index, costs in enumerate((
            (6000, 5000, 4000), (5000, 4000, 3000),
            (4000, 3000, 2000), (3000, 2000, 1000),
        )))
        plan = plan_batch(sources, 2, 10_000, "baseline")
        self.assertEqual(plan.rank_sources, ((0, 3), (1, 2)))
        self.assertEqual(plan.rank_packs, (
            (Pack((0, 2), 10_000), Pack((1, 9, 10), 10_000), Pack((11,), 1000)),
            (Pack((5, 6, 7), 10_000), Pack((3, 4), 9000), Pack((8,), 2000)),
        ))
        self.assert_plan_contract(sources, plan, 2, 10_000)

    def test_repair_preserves_coverage_and_never_regresses_at_two_four_eight_ranks(self):
        budget = SearchBudget(source_pool_size=8, source_candidates=4, source_passes=1,
                              pack_passes=1, completion_nodes=100, completion_finalists=1)
        for world_size in (2, 4, 8):
            sources = tuple(Source(str(index), (
                950 + index * 137 % 350,
                1650 + index * 97 % 500,
                2550 + index * 61 % 650,
            )) for index in range(world_size * 3))
            baseline = plan_batch(sources, world_size, 14_336, "baseline", budget)
            for strategy in ("baseline", "local", "joint"):
                with self.subTest(world_size=world_size, strategy=strategy):
                    plan = plan_batch(sources, world_size, 14_336, strategy, budget)
                    self.assert_plan_contract(sources, plan, world_size, 14_336)
                    self.assertLessEqual(plan_score(plan), plan_score(baseline))
                    if strategy == "local":
                        self.assertEqual(plan.rank_sources, baseline.rank_sources)

    def test_eight_rank_weak_scale_covers_all_64_sources_and_192_targets(self):
        sources = tuple(Source(str(index), (1000, 1100, 1200)) for index in range(64))
        budget = SearchBudget(source_pool_size=8, source_candidates=4, source_passes=1,
                              pack_passes=1, completion_nodes=100, completion_finalists=1)
        baseline = plan_batch(sources, 8, 14_336, "baseline", budget)
        plan = plan_batch(sources, 8, 14_336, "joint", budget)
        self.assert_plan_contract(sources, plan, 8, 14_336)
        self.assertLessEqual(plan_score(plan), plan_score(baseline))

    def test_all_strategies_are_deterministic_with_equal_source_weights(self):
        sources = (
            Source("a0", (1000, 5500, 6500)), Source("b0", (3500, 4500, 5000)),
            Source("a1", (1000, 5500, 6500)), Source("b1", (3500, 4500, 5000)),
        )
        for strategy in ("baseline", "local", "joint"):
            with self.subTest(strategy=strategy):
                first = plan_batch(sources, 2, 10_000, strategy)
                self.assertEqual(plan_batch(sources, 2, 10_000, strategy), first)
                self.assert_plan_contract(sources, first, 2, 10_000)

    def test_equal_source_weights_keep_witnesses_with_different_pack_profiles(self):
        sources = (
            Source("a0", (1000, 5500, 6500)), Source("b0", (3500, 4500, 5000)),
            Source("a1", (1000, 5500, 6500)), Source("b1", (3500, 4500, 5000)),
        )
        baseline = plan_batch(sources, 2, 10_000, "baseline")
        local = plan_batch(sources, 2, 10_000, "local")
        joint = plan_batch(sources, 2, 10_000, "joint")
        self.assertEqual(baseline.rank_sources, ((0, 2), (1, 3)))
        self.assertEqual(plan_score(baseline)[0], 4)
        # Four A targets exceed half the capacity, so local repair needs four packs.
        self.assertEqual(plan_score(local)[0], 4)
        self.assertEqual(plan_score(joint)[:2], (3, 26_000))
        for owner in joint.rank_sources:
            self.assertEqual(sum(source_id in (0, 2) for source_id in owner), 1)
            self.assertEqual(sum(source_id in (1, 3) for source_id in owner), 1)
        self.assert_plan_contract(sources, joint, 2, 10_000)

    def test_local_pack_dp_improves_aligned_slot_cost_without_moving_sources(self):
        sources = (Source("left", (2000, 3000, 6000)),
                   Source("right", (1000, 2000, 8000)))
        baseline = plan_batch(sources, 2, 10_000, "baseline")
        local = plan_batch(sources, 2, 10_000, "local")
        self.assertEqual(plan_score(baseline)[:2], (2, 12_000))
        self.assertEqual(plan_score(local)[:2], (2, 11_000))
        self.assertEqual(local.rank_sources, baseline.rank_sources)
        self.assertEqual(tuple(tuple(pack.joint_tokens for pack in rank)
                               for rank in local.rank_packs), ((9000, 2000), (9000, 2000)))
        self.assert_plan_contract(sources, local, 2, 10_000)

    def test_local_halving_can_be_worse_than_matching_other_rank_slots(self):
        sources = (Source("left", (2000, 4000, 6000)),
                   Source("right", (1000, 1000, 10_000)))
        aligned = plan_batch(sources, 2, 10_000, "baseline")
        halved = PackingPlan(aligned.rank_sources, (
            (Pack((2,), 6000), Pack((0, 1), 6000)), aligned.rank_packs[1],
        ))
        self.assert_plan_contract(sources, aligned, 2, 10_000)
        self.assert_plan_contract(sources, halved, 2, 10_000)
        self.assertEqual(plan_score(aligned)[:2], (2, 12_000))
        self.assertEqual(plan_score(halved)[:2], (2, 16_000))
        local = plan_batch(sources, 2, 10_000, "local")
        self.assertEqual(plan_score(local)[:2], (2, 12_000))

    def test_completion_repairs_all_three_bottleneck_ranks_before_global_acceptance(self):
        sources = tuple(Source(f"large-{index}", (3000, 5000, 6000)) for index in range(3)) + \
            tuple(Source(f"small-{index}", (2000, 2000, 2000)) for index in range(3))
        baseline = plan_batch(sources, 3, 10_000, "baseline")
        budget = SearchBudget(source_passes=0, pack_passes=0, completion_nodes=1)
        self.assertEqual(tuple(map(len, baseline.rank_packs)), (3, 3, 3))
        self.assertEqual(plan_score(baseline)[:2], (3, 20_000))
        for strategy in ("local", "joint"):
            with self.subTest(strategy=strategy):
                plan = plan_batch(sources, 3, 10_000, strategy, budget)
                self.assertEqual(tuple(map(len, plan.rank_packs)), (2, 2, 2))
                self.assertEqual(plan_score(plan)[:2], (2, 20_000))
                self.assert_plan_contract(sources, plan, 3, 10_000)

    def test_exact_two_bin_repair_fits_overflowing_lpt_with_one_branch_node(self):
        costs = (8000, 7000, 6000, 5000, 4000)
        # LPT puts 8+5+4 and 7+6 together; both feasible bins need 15 instead.
        self.assertEqual((sum(costs[index] for index in (0, 3, 4)),
                          sum(costs[index] for index in (1, 2))), (17_000, 13_000))
        packs = _fit_bins(tuple(range(len(costs))), costs, 15_000, 2, nodes=1)
        self.assertIsNotNone(packs)
        self.assertEqual(tuple(pack.joint_tokens for pack in packs), (15_000, 15_000))
        plan = PackingPlan(((0,),), (packs,))
        self.assert_plan_contract((Source("source", costs),), plan, 1, 15_000)

    def test_equal_rank_profiles_start_with_complete_rotating_matchings(self):
        for world_size in (2, 4, 8):
            sources = tuple(Source(str(index), (1000,)) for index in range(world_size))
            plan = plan_batch(sources, world_size, 10_000, "baseline")
            cycle_pairs = []
            for sweep in range(world_size - 1):
                with self.subTest(world_size=world_size, sweep=sweep):
                    pairs = _gpu_pairs(plan, sweep)
                    self.assertEqual(_gpu_pairs(plan, sweep), pairs)
                    self.assertEqual(len(pairs), min(world_size, world_size * (world_size - 1) // 2))
                    self.assertEqual(len(set(pairs)), len(pairs))
                    matching = pairs[:world_size // 2]
                    self.assertEqual(sorted(rank for pair in matching for rank in pair),
                                     list(range(world_size)))
                    self.assertTrue(all(0 <= left < right < world_size for left, right in pairs))
                    cycle_pairs.extend(matching)
            self.assertEqual(sorted(cycle_pairs), list(combinations(range(world_size), 2)))
            self.assertEqual(_gpu_pairs(plan, world_size - 1), _gpu_pairs(plan, 0))

    def test_odd_rank_bye_is_covered_before_high_distance_pairs_fill_the_budget(self):
        loads = (2000, 1000, 3000, 1000, 3000, 1000, 3000)
        sources = tuple(Source(str(rank), (load,)) for rank, load in enumerate(loads))
        plan = PackingPlan(tuple((rank,) for rank in range(7)),
                           tuple((Pack((rank,), load),) for rank, load in enumerate(loads)))
        self.assert_plan_contract(sources, plan, 7, 5000)
        for sweep in range(7):
            with self.subTest(sweep=sweep):
                pairs = _gpu_pairs(plan, sweep)
                self.assertEqual(_gpu_pairs(plan, sweep), pairs)
                self.assertEqual(len(pairs), 7)
                self.assertEqual(len(set(pairs)), len(pairs))
                self.assertEqual({rank for pair in pairs for rank in pair}, set(range(7)))

    def test_zero_search_budget_keeps_a_valid_incumbent(self):
        sources = tuple(Source(str(index), (1000 + index, 2000 + index, 4000 + index))
                        for index in range(8))
        budget = SearchBudget(source_passes=0, pack_passes=0, completion_nodes=0)
        baseline = plan_batch(sources, 2, 10_000, "baseline", budget)
        for strategy in ("local", "joint"):
            with self.subTest(strategy=strategy):
                plan = plan_batch(sources, 2, 10_000, strategy, budget)
                self.assertEqual(plan, baseline)
                self.assert_plan_contract(sources, plan, 2, 10_000)

    def test_validator_rejects_wrong_coverage_ownership_cost_capacity_and_order(self):
        sources = (Source("left", (2000, 4000, 6000)),
                   Source("right", (1000, 1000, 10_000)))
        plan = plan_batch(sources, 2, 10_000, "baseline")
        malformed = (
            replace(plan, rank_sources=((0,), (0,))),
            replace(plan, rank_sources=((), (0, 1))),
            replace(plan, rank_packs=((), plan.rank_packs[1])),
            replace(plan, rank_packs=((Pack((), 0),), plan.rank_packs[1])),
            replace(plan, rank_packs=((Pack((1, 2), 9999), Pack((0,), 2000)), plan.rank_packs[1])),
            replace(plan, rank_packs=((Pack((1, 2), 10_000), Pack((0, 0), 4000)), plan.rank_packs[1])),
            replace(plan, rank_packs=((Pack((1, 2), 10_000),), plan.rank_packs[1])),
            replace(plan, rank_packs=(plan.rank_packs[1], plan.rank_packs[0])),
            replace(plan, rank_packs=((Pack((0, 1, 2), 12_000),), plan.rank_packs[1])),
            replace(plan, rank_packs=(tuple(reversed(plan.rank_packs[0])), plan.rank_packs[1])),
        )
        for index, invalid in enumerate(malformed):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_plan(sources, invalid, 10_000)

    def test_planner_rejects_invalid_external_costs_rank_counts_and_strategy(self):
        valid = (Source("valid", (1000, 2000, 3000)),)
        for sources, ranks, capacity, strategy in (
            ((), 1, 10_000, "joint"), (valid, 0, 10_000, "joint"),
            (valid, True, 10_000, "joint"), (valid, 2, 10_000, "joint"),
            (valid, 1, 0, "joint"), (valid, 1, True, "joint"),
            (valid, 1, 10_000, "unknown"),
            ((Source("empty", ()),), 1, 10_000, "joint"),
            ((Source("zero", (0,)),), 1, 10_000, "joint"),
            ((Source("bool", (True,)),), 1, 10_000, "joint"),
            ((Source("oversize", (10_001,)),), 1, 10_000, "joint"),
            (valid + valid, 2, 10_000, "joint"),
        ):
            with self.subTest(sources=sources, ranks=ranks, capacity=capacity, strategy=strategy):
                with self.assertRaises(ValueError):
                    plan_batch(sources, ranks, capacity, strategy)


if __name__ == "__main__":
    unittest.main()
