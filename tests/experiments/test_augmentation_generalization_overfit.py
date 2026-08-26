from __future__ import annotations

from collections import Counter
from math import factorial
from pathlib import Path
import json
import runpy
import unittest

import torch

from diffusion_arc.arc_data import ArcAugmentationKey, ArcAugmentationSpec
from diffusion_arc.training.config import load_single_gpu_training_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPOSITORY_ROOT
    / "scripts"
    / "experiments"
    / "run_single_task_augmentation_generalization.py"
)
CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "single_gpu_0d3d703e_augmentation_generalization.json"
)
NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-augmentation-generalization-overfitting-modal.ipynb"
)


def _indexed_permutation(index: int) -> tuple[int, ...]:
    values = list(range(10))
    result = []
    remainder = index
    for width in range(10, 0, -1):
        block = factorial(width - 1)
        position, remainder = divmod(remainder, block)
        result.append(values.pop(position))
    return tuple(result)


class AugmentationGeneralizationProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = runpy.run_path(str(RUNNER))
        cls.PreparedView = cls.namespace["PreparedView"]
        cls.TrainingBatchPlan = cls.namespace["TrainingBatchPlan"]
        cls.geometries = cls.namespace["EFFECTIVE_GEOMETRIES"]
        cls.orders = cls.namespace["DEMONSTRATION_ORDERS"]

    @classmethod
    def _view(
        cls,
        geometry_index: int,
        local_index: int,
        *,
        split: int = 0,
        order_index: int | None = None,
    ):
        geometry = cls.geometries[geometry_index]
        unique_index = split * 1280 + geometry_index * 320 + local_index
        augmentation = ArcAugmentationSpec(
            geometry=geometry,
            color_mapping=_indexed_permutation(unique_index),
            pair_indices=(
                *cls.orders[
                    (
                        local_index % len(cls.orders)
                        if order_index is None
                        else order_index
                    )
                ],
                cls.namespace["TARGET_ORIGINAL_PAIR_INDEX"],
            ),
        )
        key = ArcAugmentationKey(
            task_id=cls.namespace["TASK_ID"],
            augmentation=augmentation,
        )
        return cls.PreparedView(
            candidate_index=local_index,
            geometry=geometry,
            key=key,
            sample=None,
            query_grid=((unique_index,),),
            target_grid=((unique_index,),),
            prefix_hash=f"prefix-{unique_index}",
            query_hash=f"query-{unique_index}",
            target_hash=f"target-{unique_index}",
            full_view_hash=f"view-{unique_index}",
        )

    @classmethod
    def _views(cls, per_geometry: int, *, split: int = 0):
        return tuple(
            cls._view(geometry_index, local_index, split=split)
            for geometry_index in range(len(cls.geometries))
            for local_index in range(per_geometry)
        )

    def test_static_protocol_uses_four_equally_weighted_representatives(self) -> None:
        self.namespace["_validate_static_protocol"]()
        self.assertEqual(
            tuple(geometry.value for geometry in self.geometries),
            ("ID", "R90", "R180", "R270"),
        )
        classes = self.namespace["REDUNDANT_GEOMETRY_CLASSES"]
        self.assertEqual({len(members) for members in classes.values()}, {2})
        self.assertEqual(
            {member for members in classes.values() for member in members},
            set(type(self.geometries[0])),
        )

    def test_candidate_augmentation_is_deterministic_and_keeps_pair_four_last(
        self,
    ) -> None:
        build = self.namespace["_candidate_augmentation"]
        for candidate_index in (0, 1, 23, 24, 511):
            for geometry in self.geometries:
                first = build(geometry, candidate_index)
                second = build(geometry, candidate_index)
                self.assertEqual(first, second)
                self.assertEqual(first.geometry, geometry)
                self.assertEqual(first.pair_indices[-1], 4)
                self.assertEqual(set(first.pair_indices[:-1]), {0, 1, 2, 3})
                self.assertEqual(set(first.color_mapping), set(range(10)))

    def test_safe_selection_is_balanced_and_semantically_disjoint(self) -> None:
        candidates = self._views(320)
        select = self.namespace["_select_safe_pool"]
        train, holdout, audit = select(
            candidates,
            {view.serialized_key for view in candidates},
        )

        self.assertEqual(len(train), 512)
        self.assertEqual(len(holdout), 128)
        self.assertEqual(
            Counter(view.geometry for view in train),
            Counter({geometry: 128 for geometry in self.geometries}),
        )
        self.assertEqual(
            Counter(view.geometry for view in holdout),
            Counter({geometry: 32 for geometry in self.geometries}),
        )
        self.assertTrue(
            {view.serialized_key for view in train}.isdisjoint(
                view.serialized_key for view in holdout
            )
        )
        self.assertEqual(
            audit["vae_safe_candidates_by_geometry"],
            {geometry.value: 320 for geometry in self.geometries},
        )

    def test_safe_selection_uses_best_effort_when_one_pair_order_is_sparse(
        self,
    ) -> None:
        candidates = tuple(
            self._view(
                geometry_index,
                local_index,
                order_index=(0 if local_index < 5 else 1 + (local_index - 5) % 23),
            )
            for geometry_index in range(len(self.geometries))
            for local_index in range(320)
        )
        train, holdout, audit = self.namespace["_select_safe_pool"](
            candidates,
            {view.serialized_key for view in candidates},
        )

        self.assertEqual((len(train), len(holdout)), (512, 128))
        self.assertEqual(
            Counter(view.geometry for view in train),
            Counter({geometry: 128 for geometry in self.geometries}),
        )
        self.assertGreaterEqual(min(audit["train_pair_order_counts"].values()), 4)
        self.assertGreaterEqual(min(audit["holdout_pair_order_counts"].values()), 1)

    def test_schedule_gives_every_view_one_endpoint_and_seven_regular_contacts(
        self,
    ) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]
        payload = self.namespace["_schedule_payload"]

        schedule = build(train)
        replay = build(train)

        self.assertEqual(schedule, replay)
        self.assertEqual(len(schedule), 512)
        self.assertEqual(
            [plan.step for plan in schedule if plan.kind == "shared_endpoint"],
            list(range(1, 513, 8)),
        )
        for plan in schedule:
            self.assertEqual(len(set(plan.view_indices)), 8)
            self.assertEqual(
                Counter(train[index].geometry for index in plan.view_indices),
                Counter({geometry: 2 for geometry in self.geometries}),
            )
            if plan.kind == "stratified":
                self.assertEqual(set(plan.timestep_strata or ()), set(range(8)))

        exposure_audit = audit(schedule, train)
        self.assertTrue(exposure_audit["all_exact"])
        self.assertEqual(exposure_audit["view_count"], 512)
        for item in exposure_audit["items"].values():
            self.assertEqual(item["total"], 8)
            self.assertEqual(item["shared_endpoint"], 1)
            self.assertEqual(item["stratified"], 7)
            self.assertEqual(len(set(item["regular_strata"])), 7)
        self.assertEqual(payload(schedule, train), payload(replay, train))

    def test_planned_flow_sampling_uses_exact_shared_t1_and_prescribed_strata(
        self,
    ) -> None:
        sample = self.namespace["_sample_planned_flow_batch"]
        clean = torch.linspace(-1, 1, 8 * 5 * 128).reshape(8, 5, 128)
        endpoint_plan = self.TrainingBatchPlan(
            step=1,
            cycle_index=0,
            cycle_position=0,
            kind="shared_endpoint",
            view_indices=tuple(range(8)),
            timestep_strata=None,
        )
        endpoint, metadata = sample(
            clean,
            batch_plan=endpoint_plan,
            generator=torch.Generator().manual_seed(2168),
        )
        self.assertEqual(metadata["kind"], "shared_endpoint")
        torch.testing.assert_close(endpoint.timesteps, torch.ones(8))
        torch.testing.assert_close(
            endpoint.noisy_latents,
            endpoint.noisy_latents[:1].expand_as(endpoint.noisy_latents),
        )
        torch.testing.assert_close(
            endpoint.target_velocity,
            endpoint.noisy_latents - clean,
        )

        strata = (3, 4, 5, 6, 7, 0, 1, 2)
        regular_plan = self.TrainingBatchPlan(
            step=2,
            cycle_index=0,
            cycle_position=1,
            kind="stratified",
            view_indices=tuple(range(8)),
            timestep_strata=strata,
        )
        regular, metadata = sample(
            clean,
            batch_plan=regular_plan,
            generator=torch.Generator().manual_seed(2169),
        )
        self.assertEqual(metadata["prescribed_timestep_strata"], list(strata))
        for row, stratum in enumerate(strata):
            self.assertGreaterEqual(float(regular.timesteps[row]), stratum / 8)
            self.assertLess(float(regular.timesteps[row]), (stratum + 1) / 8)

    def test_evaluation_panels_use_reduced_intermediate_and_full_final_counts(
        self,
    ) -> None:
        train = self._views(128)
        holdout = self._views(32, split=1)
        select = self.namespace["_evaluation_indices"]
        group = self.namespace["_group_for_global_index"]

        intermediate_panels = []
        for step in (0, 128, 256):
            indices = select(step, train, holdout)
            intermediate_panels.append(indices)
            self.assertEqual(
                Counter(map(group, indices)), {"seen": 32, "holdout": 32, "original": 1}
            )
            self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(intermediate_panels[0], intermediate_panels[1])
        self.assertEqual(intermediate_panels[1], intermediate_panels[2])
        final = select(512, train, holdout)
        self.assertEqual(
            Counter(map(group, final)),
            {"seen": 128, "holdout": 128, "original": 1},
        )
        self.assertEqual(len(final), len(set(final)))
        self.assertTrue(set(intermediate_panels[0]).issubset(final))

    def test_generation_noise_is_keyed_and_chunk_invariant(self) -> None:
        keyed_noise = self.namespace["_keyed_noise"]
        function_globals = keyed_noise.__globals__
        sentinel = object()
        previous = function_globals.get("DEVICE", sentinel)
        function_globals["DEVICE"] = torch.device("cpu")
        try:
            together = keyed_noise(("alpha", "beta"), seed=167, dtype=torch.float32)
            separate = torch.cat(
                (
                    keyed_noise(("alpha",), seed=167, dtype=torch.float32),
                    keyed_noise(("beta",), seed=167, dtype=torch.float32),
                )
            )
            reversed_rows = keyed_noise(
                ("beta", "alpha"), seed=167, dtype=torch.float32
            )
        finally:
            if previous is sentinel:
                function_globals.pop("DEVICE", None)
            else:
                function_globals["DEVICE"] = previous

        torch.testing.assert_close(together, separate)
        torch.testing.assert_close(together[0], reversed_rows[1])
        torch.testing.assert_close(together[1], reversed_rows[0])
        self.assertEqual(tuple(together.shape), (2, 900, 128))

    def test_t1_region_losses_are_vectorized_per_view(self) -> None:
        squared = torch.ones((2, 30, 30, 128), dtype=torch.float32)
        squared[0, :3, :3] = 4.0
        squared[1] = 2.0
        metrics = self.namespace["_per_row_region_losses"](squared.reshape(2, 900, 128))

        self.assertEqual(tuple(metrics.shape), (2, 3))
        torch.testing.assert_close(
            metrics[0],
            torch.tensor(((9 * 4 + 891) / 900, 4.0, 1.0)),
        )
        torch.testing.assert_close(metrics[1], torch.tensor((2.0, 2.0, 2.0)))

    def test_comparison_grid_rendering_uses_compact_canvas(self) -> None:
        render = self.namespace["_uint8_grid_image"]
        image = render(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        self.assertEqual(tuple(image.shape), (3, 48, 48))
        self.assertEqual(image.dtype, torch.uint8)

    def test_flow_summary_verifies_observed_per_view_exposures(self) -> None:
        train = self._views(128)
        schedule = self.namespace["_build_training_schedule"](train)
        audit = self.namespace["_schedule_exposure_audit"](schedule, train)
        records = []
        for plan in schedule:
            exposures = []
            for row, view_index in enumerate(plan.view_indices):
                stratum = (
                    None if plan.timestep_strata is None else plan.timestep_strata[row]
                )
                timestep = 1.0 if stratum is None else (stratum + 0.5) / 8
                view = train[view_index]
                exposures.append(
                    {
                        "key": view.serialized_key,
                        "geometry": view.geometry.value,
                        "sampling_kind": plan.kind,
                        "timestep": timestep,
                        "timestep_stratum": stratum,
                        "total_loss": 1.0,
                        "grid_loss": 2.0,
                        "padding_loss": 0.5,
                    }
                )
            records.append(
                {
                    "step": plan.step,
                    "sampling": {"kind": plan.kind},
                    "per_exposure": exposures,
                }
            )

        summarize = self.namespace["_summarize_flow_sampling"]
        summary = summarize(records, exposure_audit=audit)
        self.assertTrue(summary["exposure_plan_matches_observed_keys"])
        self.assertEqual(
            summary["update_counts"],
            {"shared_endpoint": 64, "stratified": 448},
        )
        self.assertEqual(
            summary["exposure_counts"],
            {"shared_endpoint": 512, "stratified": 3584, "total": 4096},
        )

        records[0]["per_exposure"][0]["key"] = "unplanned"
        with self.assertRaisesRegex(RuntimeError, "unplanned training key"):
            summarize(records, exposure_audit=audit)

    def test_generation_metrics_compute_adaptive_pass2_by_group(self) -> None:
        metric = self.namespace["_generation_panel_metrics"]

        def record(key: str, group: str, geometry: str, *, exact: bool):
            return {
                "key": key,
                "group": group,
                "geometry": geometry,
                "prediction_grid": ((1,),),
                "diagnostic": {
                    "exact_grid": exact,
                    "prediction_valid": True,
                    "shape_match": True,
                    "cell_accuracy": 1.0 if exact else 0.0,
                },
            }

        first = [
            record("seen-a", "seen", self.geometries[0].value, exact=True),
            record("seen-b", "seen", self.geometries[1].value, exact=False),
            record("holdout-a", "holdout", self.geometries[0].value, exact=False),
            record("original", "original", self.geometries[0].value, exact=False),
        ]
        second = [
            record("seen-b", "seen", self.geometries[1].value, exact=True),
            record("holdout-a", "holdout", self.geometries[0].value, exact=False),
            record("original", "original", self.geometries[0].value, exact=False),
        ]
        result = metric(first, second)
        self.assertEqual(result["panels"]["seen"]["exact_grid_pass1"], 1)
        self.assertEqual(result["panels"]["seen"]["exact_grid_pass2"], 2)
        self.assertEqual(result["panels"]["seen"]["seed2_evaluated_count"], 1)
        self.assertEqual(result["panels"]["holdout"]["exact_grid_pass2"], 0)


class AugmentationGeneralizationArtifactTest(unittest.TestCase):
    def test_config_matches_the_approved_training_budget(self) -> None:
        config = load_single_gpu_training_config(CONFIG)
        self.assertEqual(config.loop.max_optimizer_steps, 512)
        self.assertEqual(config.loop.optimizer_batch_size, 8)
        self.assertEqual(config.loop.flux_microbatch_size, 8)
        self.assertEqual(config.loop.min_context_pairs_before_target, 4)
        self.assertEqual(config.optimizer.learning_rate, 1e-5)

    def test_notebook_is_clean_and_all_code_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
