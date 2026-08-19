from __future__ import annotations

import unittest
from random import Random

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcEvaluationTask,
    ArcGeometry,
    ArcPair,
    ArcQuery,
    ArcTrainingTask,
    apply_training_augmentation,
    augment_grid,
    make_augmentation_key,
    make_identity_training_augmentation,
    parse_augmentation_key,
    sample_training_augmentation,
    validate_augmentation_for_task,
)


IDENTITY_COLOR_MAPPING = tuple(range(10))


def make_pair(value: int) -> ArcPair:
    grid = ((value,),)
    return ArcPair(input_grid=grid, output_grid=grid)


def make_spec(
    *,
    geometry: ArcGeometry = ArcGeometry.FLIP_VERTICAL,
    color_mapping: tuple[int, ...] = IDENTITY_COLOR_MAPPING,
    pair_indices: tuple[int, ...] = (27, 3, 4, 18, 12, 8),
) -> ArcAugmentationSpec:
    return ArcAugmentationSpec(
        geometry=geometry,
        color_mapping=color_mapping,
        pair_indices=pair_indices,
    )


class ArcAugmentationKeyTest(unittest.TestCase):
    def test_key_round_trip_preserves_augmentation(self) -> None:
        serialized = make_augmentation_key("nvarc-task-42", make_spec())

        self.assertEqual(
            serialized,
            "nvarc-task-42_FV_0123456789_27,3,4,18,12,8",
        )
        self.assertEqual(
            parse_augmentation_key(serialized),
            ArcAugmentationKey("nvarc-task-42", make_spec()),
        )

        for invalid_task_id in ("", "task_with_separator"):
            with self.assertRaises(ValueError):
                make_augmentation_key(invalid_task_id, make_spec())

    def test_spec_rejects_invalid_mapping_or_pair_selection(self) -> None:
        with self.assertRaises(ValueError):
            make_spec(color_mapping=(0, 1, 2, 3, 4, 5, 6, 7, 8, 8))
        with self.assertRaises(ValueError):
            make_spec(pair_indices=())
        with self.assertRaises(ValueError):
            make_spec(pair_indices=(3, 3))


class ArcAugmentationTaskValidationTest(unittest.TestCase):
    def test_training_indices_must_reference_available_pairs(self) -> None:
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=tuple(make_pair(index) for index in range(30)),
        )
        valid_key = ArcAugmentationKey(task.task_id, make_spec())
        invalid_key = ArcAugmentationKey(
            task.task_id,
            make_spec(pair_indices=(29, 30)),
        )

        validate_augmentation_for_task(valid_key, task)
        with self.assertRaises(ValueError):
            validate_augmentation_for_task(invalid_key, task)

    def test_evaluation_indices_reference_demonstrations_not_queries(self) -> None:
        task = ArcEvaluationTask(
            task_id="evaluation-task",
            demonstrations=(make_pair(0),),
            queries=tuple(
                ArcQuery(query_index=index, input_grid=((index,),))
                for index in range(5)
            ),
        )
        valid_key = ArcAugmentationKey(
            task.task_id,
            make_spec(pair_indices=(0,)),
        )
        query_index_key = ArcAugmentationKey(
            task.task_id,
            make_spec(pair_indices=(1,)),
        )

        validate_augmentation_for_task(valid_key, task)
        with self.assertRaises(ValueError):
            validate_augmentation_for_task(query_index_key, task)

    def test_task_id_must_match(self) -> None:
        task = ArcTrainingTask(task_id="training-task", pairs=(make_pair(0),))
        key = ArcAugmentationKey(
            "another-task",
            make_spec(pair_indices=(0,)),
        )

        with self.assertRaises(ValueError):
            validate_augmentation_for_task(key, task)


class ArcAugmentationSamplingTest(unittest.TestCase):
    def test_identity_keeps_original_order_within_limits(self) -> None:
        large_pair = ArcPair(
            input_grid=(tuple(range(20)),),
            output_grid=(tuple(range(20)),),
        )
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=(make_pair(0), large_pair, make_pair(2), make_pair(3)),
        )

        identity = make_identity_training_augmentation(
            task,
            max_pairs=2,
            max_cells=4,
        )

        self.assertEqual(identity.geometry, ArcGeometry.IDENTITY)
        self.assertEqual(identity.color_mapping, IDENTITY_COLOR_MAPPING)
        self.assertEqual(identity.pair_indices, (0, 2))

    def test_sampling_is_reproducible_and_rotates_geometry(self) -> None:
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=tuple(make_pair(index) for index in range(8)),
        )

        first = sample_training_augmentation(
            task,
            augmentation_index=0,
            max_pairs=4,
            max_cells=8,
            rng=Random(42),
        )
        repeated = sample_training_augmentation(
            task,
            augmentation_index=0,
            max_pairs=4,
            max_cells=8,
            rng=Random(42),
        )
        next_geometry = sample_training_augmentation(
            task,
            augmentation_index=1,
            max_pairs=4,
            max_cells=8,
            rng=Random(42),
        )

        self.assertEqual(first, repeated)
        self.assertEqual(first.geometry, ArcGeometry.IDENTITY)
        self.assertEqual(next_geometry.geometry, ArcGeometry.ROTATE_90)

    def test_sampling_skips_large_pairs_and_respects_both_limits(self) -> None:
        large_pair = ArcPair(
            input_grid=(tuple(range(20)),),
            output_grid=(tuple(range(20)),),
        )
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=(make_pair(0), make_pair(1), large_pair, make_pair(3)),
        )

        sampled = sample_training_augmentation(
            task,
            augmentation_index=0,
            max_pairs=2,
            max_cells=4,
            rng=Random(0),
        )

        self.assertEqual(sampled.pair_indices, (0, 1))

    def test_sampling_fails_when_fewer_than_two_pairs_fit(self) -> None:
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=(make_pair(0), make_pair(1)),
        )

        with self.assertRaises(ValueError):
            sample_training_augmentation(
                task,
                augmentation_index=0,
                max_pairs=2,
                max_cells=2,
                rng=Random(42),
            )


class ArcAugmentationApplicationTest(unittest.TestCase):
    def test_all_geometries_follow_the_axis_based_convention(self) -> None:
        grid = ((1, 2, 3), (4, 5, 6))
        expected = {
            ArcGeometry.IDENTITY: ((1, 2, 3), (4, 5, 6)),
            ArcGeometry.ROTATE_90: ((4, 1), (5, 2), (6, 3)),
            ArcGeometry.ROTATE_180: ((6, 5, 4), (3, 2, 1)),
            ArcGeometry.ROTATE_270: ((3, 6), (2, 5), (1, 4)),
            ArcGeometry.FLIP_HORIZONTAL: ((4, 5, 6), (1, 2, 3)),
            ArcGeometry.FLIP_VERTICAL: ((3, 2, 1), (6, 5, 4)),
            ArcGeometry.FLIP_MAIN_DIAGONAL: ((1, 4), (2, 5), (3, 6)),
            ArcGeometry.FLIP_ANTI_DIAGONAL: ((6, 3), (5, 2), (4, 1)),
        }

        for geometry, expected_grid in expected.items():
            with self.subTest(geometry=geometry):
                augmentation = make_spec(
                    geometry=geometry,
                    pair_indices=(0, 1),
                )
                self.assertEqual(
                    augment_grid(grid, augmentation),
                    expected_grid,
                )

    def test_training_augmentation_orders_and_transforms_pairs(self) -> None:
        task = ArcTrainingTask(
            task_id="training-task",
            pairs=(
                ArcPair(input_grid=((0, 1),), output_grid=((2,),)),
                ArcPair(input_grid=((3, 4),), output_grid=((5,),)),
            ),
        )
        augmentation = make_spec(
            geometry=ArcGeometry.FLIP_VERTICAL,
            color_mapping=tuple(reversed(range(10))),
            pair_indices=(1, 0),
        )

        sample = apply_training_augmentation(task, augmentation)

        self.assertEqual(
            str(sample.key),
            "training-task_FV_9876543210_1,0",
        )
        self.assertEqual(
            sample.pairs,
            (
                ArcPair(input_grid=((5, 6),), output_grid=((4,),)),
                ArcPair(input_grid=((8, 9),), output_grid=((7,),)),
            ),
        )

    def test_rejects_grids_larger_than_thirty_cells_per_axis(self) -> None:
        augmentation = make_spec(
            geometry=ArcGeometry.IDENTITY,
            pair_indices=(0, 1),
        )
        oversized_grids = (
            tuple((0,) for _ in range(31)),
            (tuple(0 for _ in range(31)),),
        )

        for grid in oversized_grids:
            with self.subTest(shape=(len(grid), len(grid[0]))), self.assertRaises(
                ValueError
            ):
                augment_grid(grid, augmentation)


if __name__ == "__main__":
    unittest.main()
