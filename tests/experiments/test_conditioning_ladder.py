from __future__ import annotations

from pathlib import Path
import runpy
import unittest

import torch

from diffusion_arc.arc_data import ArcGridPrediction


RUNNER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "run_single_task_conditioning_ladder.py"
)


class ConditioningLadderVaeGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = runpy.run_path(str(RUNNER))

    def test_group_summary_records_one_failed_condition(self) -> None:
        summarize = self.namespace["_roundtrip_group_summary"]
        per_condition = {
            "first": {"exact_grid": True},
            "second": {"exact_grid": False},
        }

        summary = summarize(per_condition, ("first", "second"))

        self.assertEqual(summary["exact_grids"], 1)
        self.assertEqual(summary["total_grids"], 2)
        self.assertEqual(summary["exact_grid_fraction"], 0.5)
        self.assertEqual(summary["exact_condition_names"], ["first"])
        self.assertEqual(summary["failed_condition_names"], ["second"])

    def test_roundtrip_records_condition_and_stage_diagnostics(self) -> None:
        roundtrip = self.namespace["_vae_roundtrip"]
        condition_names = tuple(
            condition.name for condition in self.namespace["CONDITION_SPECS"]
        )
        grids = tuple(((index % 10,),) for index in range(len(condition_names)))
        failed_index = condition_names.index("unseen-id")
        predictions = []
        for index, grid in enumerate(grids):
            predicted_grid = ((9,),) if index == failed_index else grid
            predictions.append(
                ArcGridPrediction(
                    class_canvas=torch.zeros((30, 30), dtype=torch.int64),
                    grid=predicted_grid,
                    height=1,
                    width=1,
                    valid=True,
                    invalid_reason=None,
                )
            )
        cursor = 0

        def fake_decode_latents(latents, _vae_decoder):
            nonlocal cursor
            count = latents.shape[0]
            batch = tuple(predictions[cursor : cursor + count])
            cursor += count
            return (
                torch.empty(0),
                batch,
                torch.empty(0),
            )

        function_globals = roundtrip.__globals__
        original_decode = function_globals["_decode_latents"]
        function_globals["_decode_latents"] = fake_decode_latents
        self.addCleanup(
            function_globals.__setitem__,
            "_decode_latents",
            original_decode,
        )

        summary = roundtrip(
            latents=torch.zeros((17, 1)),
            condition_names=condition_names,
            grids=grids,
            vae_decoder=object(),
        )

        self.assertEqual(summary["exact_grids"], 16)
        self.assertEqual(summary["failed_condition_names"], ["unseen-id"])
        self.assertTrue(summary["gate"]["passed"])
        self.assertTrue(summary["gate"]["accepted_with_tolerance"])
        failed = summary["per_condition"]["unseen-id"]
        self.assertEqual(failed["predicted_grid"], ((9,),))
        self.assertEqual(
            failed["mismatched_cells"],
            [{"row": 0, "column": 0, "expected": 8, "predicted": 9}],
        )
        stage_panels = summary["stage_panel_summaries"]
        self.assertEqual(stage_panels["one-condition"]["seen"]["exact_grids"], 1)
        self.assertEqual(
            stage_panels["eight-condition-overfit"]["seen"]["exact_grids"],
            8,
        )
        self.assertEqual(
            stage_panels["eight-condition-overfit"]["unseen"]["exact_grids"],
            7,
        )

    def test_gate_accepts_one_noncanonical_failure(self) -> None:
        validate = self.namespace["_validate_vae_roundtrip_gate"]
        validate(
            {
                "exact_grids": 16,
                "total_grids": 17,
                "failed_condition_names": ["unseen-id"],
                "gate": {"canonical_exact": True},
            }
        )

    def test_gate_rejects_two_failures(self) -> None:
        validate = self.namespace["_validate_vae_roundtrip_gate"]
        with self.assertRaisesRegex(RuntimeError, "15/17"):
            validate(
                {
                    "exact_grids": 15,
                    "total_grids": 17,
                    "failed_condition_names": ["train-id", "unseen-id"],
                    "gate": {"canonical_exact": True},
                }
            )

    def test_gate_rejects_canonical_failure(self) -> None:
        validate = self.namespace["_validate_vae_roundtrip_gate"]
        with self.assertRaisesRegex(RuntimeError, "Canonical"):
            validate(
                {
                    "exact_grids": 16,
                    "total_grids": 17,
                    "failed_condition_names": ["canonical"],
                    "gate": {"canonical_exact": False},
                }
            )


if __name__ == "__main__":
    unittest.main()
