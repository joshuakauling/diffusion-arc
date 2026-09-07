"""CPU checks of measured-cost FLUX packing within fixed optimizer windows."""

from itertools import permutations, product
import unittest

from diffusion_arc.training.flux_schedule import (
    FluxBatchShape, FluxSchedulingPolicy, FluxTargetBatch, schedule_flux_targets,
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


if __name__ == "__main__":
    unittest.main()
