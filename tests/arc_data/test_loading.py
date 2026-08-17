from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diffusion_arc.arc_data import (
    ArcEvaluationTargets,
    ArcEvaluationTask,
    ArcPair,
    ArcQuery,
    ArcTarget,
    align_evaluation_targets,
    load_evaluation_targets,
    load_evaluation_task,
    load_training_task,
)


def write_task(root: Path, task_id: str, payload: object) -> Path:
    path = root / f"{task_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ArcTaskLoadingTest(unittest.TestCase):
    def test_training_loads_train_then_labelled_test_pairs(self) -> None:
        payload = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_task(Path(directory), "training-task", payload)
            task = load_training_task(path)

        self.assertEqual(task.task_id, "training-task")
        self.assertEqual(
            task.pairs,
            (
                ArcPair(input_grid=((1,),), output_grid=((2,),)),
                ArcPair(input_grid=((3,),), output_grid=((4,),)),
            ),
        )

    def test_evaluation_separates_queries_and_targets(self) -> None:
        payload = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [
                {"input": [[3]], "output": [[4]]},
                {"input": [[5]], "output": [[6]]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_task(Path(directory), "evaluation-task", payload)
            task = load_evaluation_task(path)
            targets = load_evaluation_targets(path)

        self.assertEqual(
            tuple((query.query_index, query.input_grid) for query in task.queries),
            ((0, ((3,),)), (1, ((5,),))),
        )
        self.assertFalse(hasattr(task.queries[0], "output_grid"))
        self.assertEqual(
            tuple(
                (target.query_index, target.output_grid)
                for target in targets.targets
            ),
            ((0, ((4,),)), (1, ((6,),))),
        )

    def test_invalid_grid_is_rejected(self) -> None:
        payload = {
            "train": [{"input": [[1, 2], [3]], "output": [[4]]}],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_task(Path(directory), "invalid-task", payload)
            with self.assertRaises(ValueError):
                load_training_task(path)

    def test_evaluation_targets_are_validated_and_ordered_by_query(self) -> None:
        task = ArcEvaluationTask(
            task_id="evaluation-task",
            demonstrations=(),
            queries=(
                ArcQuery(query_index=0, input_grid=((1,),)),
                ArcQuery(query_index=1, input_grid=((2,),)),
            ),
        )
        targets = ArcEvaluationTargets(
            task_id="evaluation-task",
            targets=(
                ArcTarget(query_index=1, output_grid=((4,),)),
                ArcTarget(query_index=0, output_grid=((3,),)),
            ),
        )

        aligned = align_evaluation_targets(task, targets)

        self.assertEqual(
            tuple(target.query_index for target in aligned),
            (0, 1),
        )

        with self.assertRaises(ValueError):
            align_evaluation_targets(
                task,
                ArcEvaluationTargets(
                    task_id="evaluation-task",
                    targets=(targets.targets[0],),
                ),
            )
        with self.assertRaises(ValueError):
            align_evaluation_targets(
                task,
                ArcEvaluationTargets(
                    task_id="different-task",
                    targets=targets.targets,
                ),
            )


if __name__ == "__main__":
    unittest.main()
