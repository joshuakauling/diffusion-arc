from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

from diffusion_arc.arc_data import (
    ArcEvaluationTargets,
    ArcEvaluationTask,
    ArcPair,
    ArcQuery,
    ArcTarget,
    ArcTrainingTask,
)


class ArcDataTypesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = ArcPair(
            input_grid=((1, 2),),
            output_grid=((3, 4),),
        )

    def test_training_task_contains_only_labelled_pairs(self) -> None:
        task = ArcTrainingTask(task_id="abc123ef", pairs=(self.pair,))

        self.assertEqual(task.task_id, "abc123ef")
        self.assertEqual(task.pairs, (self.pair,))

    def test_evaluation_task_has_no_target_field(self) -> None:
        query = ArcQuery(query_index=0, input_grid=((5,),))
        task = ArcEvaluationTask(
            task_id="abc123ef",
            demonstrations=(self.pair,),
            queries=(query,),
        )

        self.assertEqual(
            {field.name for field in fields(ArcEvaluationTask)},
            {"task_id", "demonstrations", "queries"},
        )
        self.assertFalse(hasattr(task, "targets"))
        self.assertFalse(hasattr(task, "solutions"))

    def test_evaluation_targets_match_queries_by_index(self) -> None:
        query = ArcQuery(query_index=2, input_grid=((5,),))
        target = ArcTarget(query_index=2, output_grid=((9,),))
        targets = ArcEvaluationTargets(task_id="abc123ef", targets=(target,))

        self.assertEqual(query.query_index, targets.targets[0].query_index)

    def test_data_objects_are_frozen(self) -> None:
        task = ArcTrainingTask(task_id="abc123ef", pairs=(self.pair,))

        with self.assertRaises(FrozenInstanceError):
            task.task_id = "deadbeef"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
