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


class ConditioningEndpointFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        namespace = runpy.run_path(str(RUNNER))
        cls.activate_experiment = namespace["_activate_experiment"]
        cls.function_globals = cls.activate_experiment.__globals__
        cls.experiment = cls.activate_experiment("endpoint-overfit")
        cls.stage = cls.experiment.stages[0]

    @staticmethod
    def _clean_latents() -> torch.Tensor:
        values = torch.linspace(
            -0.75,
            0.75,
            steps=8 * 5 * 128,
            dtype=torch.float32,
        )
        return values.reshape(8, 5, 128)

    @classmethod
    def _synthetic_training_records(cls) -> list[dict[str, object]]:
        records = []
        is_endpoint = cls.function_globals["_is_shared_endpoint_step"]
        for step in range(1, cls.stage.optimizer_steps + 1):
            endpoint = is_endpoint(step)
            sampling_kind = "shared_endpoint" if endpoint else "stratified"
            per_exposure = []
            for index, condition_name in enumerate(cls.stage.condition_names):
                per_exposure.append(
                    {
                        "condition": condition_name,
                        "timestep": 1.0 if endpoint else (index + 0.5) / 8,
                        "timestep_stratum": None if endpoint else index,
                        "sampling_kind": sampling_kind,
                        "total_loss": 2.0 if endpoint else 1.0,
                        "grid_loss": 3.0 if endpoint else 1.5,
                        "padding_loss": 4.0 if endpoint else 0.5,
                    }
                )
            records.append(
                {
                    "step": step,
                    "sampling": {
                        "kind": sampling_kind,
                        "cycle_index": (step - 1) // 8,
                        "cycle_position": (step - 1) % 8,
                        "flow_seed": 2167 + step,
                        "shared_noise_across_condition_rows": endpoint,
                        "timestep_stratum": None if endpoint else "per_exposure",
                    },
                    "per_exposure": per_exposure,
                }
            )
        return records

    def test_endpoint_protocol_selects_one_fresh_base_eight_condition_stage(self) -> None:
        self.assertIs(
            self.function_globals["ACTIVE_EXPERIMENT"],
            self.experiment,
        )
        self.assertEqual(self.experiment.name, "endpoint-overfit")
        self.assertEqual(len(self.experiment.stages), 1)
        self.assertEqual(self.function_globals["STAGES"], self.experiment.stages)

        stage = self.stage
        self.assertEqual(stage.name, "eight-condition-endpoint-overfit")
        self.assertEqual(stage.condition_names, self.function_globals["_TRAIN_CONDITION_NAMES"])
        self.assertEqual(stage.batch_condition_names(), stage.condition_names)
        self.assertEqual(stage.optimizer_steps, 2048)
        self.assertEqual(stage.flow_sampling, "shared-endpoint-every-eight")
        self.assertEqual(stage.evaluation_steps[0], 0)
        self.assertEqual(stage.evaluation_steps[-1], stage.optimizer_steps)
        self.function_globals["_validate_static_protocol"]()

    def test_endpoint_step_boundaries_and_count(self) -> None:
        is_endpoint = self.function_globals["_is_shared_endpoint_step"]
        expected = {
            1: True,
            8: False,
            9: True,
            16: False,
            17: True,
            2041: True,
            2048: False,
        }
        for step, endpoint in expected.items():
            with self.subTest(step=step):
                self.assertIs(is_endpoint(step), endpoint)
        self.assertEqual(
            sum(is_endpoint(step) for step in range(1, 2049)),
            256,
        )

    def test_endpoint_sampling_uses_exact_t1_and_shared_fresh_noise(self) -> None:
        sample = self.function_globals["_sample_training_flow_batch"]
        clean_latents = self._clean_latents()

        first, first_metadata = sample(
            clean_latents,
            stage=self.stage,
            step=1,
            generator=torch.Generator().manual_seed(2168),
        )
        replay, replay_metadata = sample(
            clean_latents,
            stage=self.stage,
            step=1,
            generator=torch.Generator().manual_seed(2168),
        )
        fresh, fresh_metadata = sample(
            clean_latents,
            stage=self.stage,
            step=9,
            generator=torch.Generator().manual_seed(2176),
        )

        self.assertEqual(first_metadata["kind"], "shared_endpoint")
        self.assertTrue(first_metadata["shared_noise_across_condition_rows"])
        self.assertIsNone(first_metadata["timestep_stratum"])
        self.assertEqual(first_metadata["cycle_position"], 0)
        self.assertEqual(first_metadata["flow_seed"], 2168)
        self.assertEqual(fresh_metadata["flow_seed"], 2176)
        self.assertEqual(replay_metadata, first_metadata)
        torch.testing.assert_close(first.timesteps, torch.ones(8))
        torch.testing.assert_close(
            first.noisy_latents,
            first.noisy_latents[:1].expand_as(first.noisy_latents),
        )
        recovered_noise = first.target_velocity + clean_latents
        torch.testing.assert_close(recovered_noise, first.noisy_latents)
        torch.testing.assert_close(
            first.target_velocity,
            first.noisy_latents - clean_latents,
        )
        torch.testing.assert_close(first.timesteps, replay.timesteps)
        torch.testing.assert_close(first.noisy_latents, replay.noisy_latents)
        torch.testing.assert_close(first.target_velocity, replay.target_velocity)
        self.assertFalse(torch.equal(first.noisy_latents[0], fresh.noisy_latents[0]))

    def test_regular_cycle_position_preserves_strata_and_independent_noise(self) -> None:
        sample = self.function_globals["_sample_training_flow_batch"]
        clean_latents = self._clean_latents()

        result, metadata = sample(
            clean_latents,
            stage=self.stage,
            step=8,
            generator=torch.Generator().manual_seed(2175),
        )

        self.assertEqual(metadata["kind"], "stratified")
        self.assertFalse(metadata["shared_noise_across_condition_rows"])
        self.assertEqual(metadata["timestep_stratum"], "per_exposure")
        self.assertEqual(metadata["cycle_position"], 7)
        ordered = result.timesteps.sort().values
        lower = torch.arange(8, dtype=torch.float32) / 8
        upper = torch.arange(1, 9, dtype=torch.float32) / 8
        self.assertTrue(bool(torch.all(ordered >= lower)))
        self.assertTrue(bool(torch.all(ordered < upper)))
        recovered_noise = result.target_velocity + clean_latents
        self.assertFalse(torch.equal(recovered_noise[0], recovered_noise[1]))

    def test_flow_summary_separates_endpoint_from_regular_strata(self) -> None:
        summarize = self.function_globals["_summarize_flow_sampling"]
        records = self._synthetic_training_records()

        summary = summarize(records, self.stage)

        self.assertEqual(
            summary["update_counts"],
            {"shared_endpoint": 256, "stratified": 1792},
        )
        self.assertEqual(
            summary["exposure_counts"],
            {"shared_endpoint": 2048, "stratified": 14336, "total": 16384},
        )
        self.assertEqual(summary["shared_endpoint"]["timestep"], 1.0)
        self.assertEqual(summary["shared_endpoint"]["updates"], 256)
        self.assertEqual(summary["shared_endpoint"]["exposures"], 2048)
        self.assertEqual(summary["shared_endpoint"]["mean_total_loss"], 2.0)
        for stratum in range(8):
            self.assertEqual(
                summary["stratified_timestep_strata"][str(stratum)]["count"],
                1792,
            )
        for condition_name in self.stage.condition_names:
            self.assertEqual(
                summary["per_condition"][condition_name],
                {
                    "total_exposures": 2048,
                    "shared_endpoint_exposures": 256,
                    "stratified_exposures": 1792,
                },
            )

        first_endpoint = records[0]["per_exposure"][0]  # type: ignore[index]
        first_endpoint["timestep_stratum"] = 7  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "regular stratum"):
            summarize(records, self.stage)

    def test_panel_metrics_keep_raw_x8_and_vae_eligible_x7_results(self) -> None:
        panel_metrics = self.function_globals["_panel_metrics"]
        condition_names = self.stage.condition_names

        def diagnostic(*, exact: bool) -> dict[str, object]:
            return {
                "exact_grid": exact,
                "prediction_valid": True,
                "shape_match": True,
                "cell_accuracy": 1.0 if exact else 0.5,
            }

        first_exact = {0, 1, 2}
        second_exact = {0, 1, 2, 3, 4}
        by_seed = [
            {
                "diagnostics": [
                    diagnostic(exact=index in exact_indices)
                    for index in range(8)
                ],
                "oracle_3x3_accuracies": [1.0] * 8,
            }
            for exact_indices in (first_exact, second_exact)
        ]
        vae_exact = frozenset(condition_names[1:])

        seen = panel_metrics(
            by_seed,
            (("seen", slice(0, 8)),),
            condition_names=condition_names,
            vae_exact_condition_names=vae_exact,
        )["seen"]

        self.assertEqual(seen["count"], 8)
        self.assertEqual(seen["exact_grid_pass1"], 3)
        self.assertEqual(seen["exact_grid_pass2"], 5)
        self.assertEqual(seen["exact_grid_pass1_display"], "3/8")
        self.assertEqual(seen["exact_grid_pass2_display"], "5/8")
        eligible = seen["vae_eligible"]
        self.assertEqual(eligible["count"], 7)
        self.assertEqual(eligible["exact_grid_pass1"], 2)
        self.assertEqual(eligible["exact_grid_pass2"], 4)
        self.assertEqual(eligible["exact_grid_pass1_display"], "2/7")
        self.assertEqual(eligible["exact_grid_pass2_display"], "4/7")
        self.assertEqual(eligible["condition_names"], list(condition_names[1:]))
        self.assertEqual(
            eligible["excluded_condition_names"],
            [condition_names[0]],
        )

    def test_display_labels_distinguish_original_seen_and_unseen(self) -> None:
        display_label = self.function_globals["_display_condition_label"]

        self.assertEqual(display_label("canonical", "canonical"), "Original")
        self.assertEqual(display_label("seen", "train-r90"), "Gesehen · r90")
        self.assertEqual(display_label("unseen", "unseen-fh"), "Ungesehen · fh")
        self.assertEqual(display_label("seen", "canonical"), "Original")


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
