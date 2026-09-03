from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import factorial
from pathlib import Path
import json
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.arc_data import ArcAugmentationKey, ArcAugmentationSpec
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config

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
    / "overfitting"
    / "single_gpu_0d3d703e_augmentation_generalization.json"
)
CONTINUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "overfitting"
    / "single_gpu_0d3d703e_augmentation_generalization_continuation.json"
)
REGION_WEIGHTED_CONTINUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "overfitting"
    / "single_gpu_0d3d703e_region_weighted_continuation.json"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "overfitting"
    / "single_gpu_0d3d703e_region_weighted_endpoint_balanced_continuation.json"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "training"
    / "overfitting"
    / "single_gpu_0d3d703e_region_weighted_endpoint_balanced_long_continuation.json"
)
NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-augmentation-generalization-overfitting-modal.ipynb"
)
CONTINUATION_NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-augmentation-generalization-continuation-modal.ipynb"
)
REGION_WEIGHTED_NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-region-weighted-loss-overfitting-modal.ipynb"
)
REGION_WEIGHTED_CONTINUATION_NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-region-weighted-loss-continuation-modal.ipynb"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-region-weighted-endpoint-balanced-continuation-modal.ipynb"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_NOTEBOOK = (
    REPOSITORY_ROOT
    / "experiments"
    / "training"
    / "single-task-region-weighted-endpoint-balanced-long-continuation-modal.ipynb"
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
        self.assertEqual(
            self.namespace["REGION_LOSS_WEIGHTS"],
            {"grid": 0.6, "boundary": 0.3, "outer_padding": 0.1},
        )
        self.assertAlmostEqual(
            sum(self.namespace["REGION_LOSS_WEIGHTS"].values()),
            1.0,
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

    def test_continuation_schedule_uses_global_steps_and_new_flow_seeds(self) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]

        parent = build(train)
        continuation = build(train, phase_start=1, phase_count=3)
        self.assertEqual(len(continuation), 1536)
        self.assertEqual(continuation[0].step, 513)
        self.assertEqual(continuation[-1].step, 2048)
        self.assertEqual(
            [plan.step for plan in continuation if plan.kind == "shared_endpoint"],
            list(range(513, 2049, 8)),
        )
        parent_flow_seeds = {
            self.namespace["FLOW_SEED"] + plan.step for plan in parent
        }
        continuation_flow_seeds = {
            self.namespace["FLOW_SEED"] + plan.step for plan in continuation
        }
        self.assertTrue(parent_flow_seeds.isdisjoint(continuation_flow_seeds))

        continuation_audit = audit(continuation, train)
        self.assertTrue(continuation_audit["all_exact"])
        self.assertEqual(
            continuation_audit["expected"],
            {"total": 24, "shared_endpoint": 3, "stratified": 21},
        )
        self.assertEqual(
            continuation_audit["expected_update_counts"],
            {"shared_endpoint": 192, "stratified": 1344, "total": 1536},
        )

    def test_region_weighted_continuation_uses_two_complete_phases(self) -> None:
        train = self._views(128)
        continuation = self.namespace["_build_training_schedule"](
            train,
            phase_start=1,
            phase_count=2,
        )
        audit = self.namespace["_schedule_exposure_audit"](continuation, train)

        self.assertEqual(len(continuation), 1024)
        self.assertEqual((continuation[0].step, continuation[-1].step), (513, 1536))
        self.assertEqual(
            audit["expected"],
            {"total": 16, "shared_endpoint": 2, "stratified": 14},
        )
        self.assertEqual(
            audit["expected_update_counts"],
            {"shared_endpoint": 128, "stratified": 896, "total": 1024},
        )

    def test_endpoint_balanced_continuation_splits_updates_evenly(self) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]
        combine = self.namespace["_combine_exposure_audits"]
        endpoint_positions = self.namespace["ENDPOINT_BALANCED_CYCLE_POSITIONS"]
        parent = combine(
            audit(build(train), train),
            audit(build(train, phase_start=1, phase_count=2), train),
        )

        continuation = build(
            train,
            phase_start=3,
            phase_count=1,
            endpoint_cycle_positions=endpoint_positions,
            prior_exposure_audit=parent,
        )
        self.assertEqual(len(continuation), 512)
        self.assertEqual((continuation[0].step, continuation[-1].step), (1537, 2048))
        self.assertEqual(
            [plan.step for plan in continuation if plan.kind == "shared_endpoint"],
            [
                cycle_start + position
                for cycle_start in range(1537, 2049, 8)
                for position in endpoint_positions
            ],
        )
        self.assertEqual(continuation[-1].kind, "stratified")

        exposure_audit = audit(continuation, train)
        self.assertTrue(exposure_audit["all_exact"])
        self.assertEqual(
            exposure_audit["expected"],
            {"total": 8, "shared_endpoint": 4, "stratified": 4},
        )
        self.assertEqual(
            exposure_audit["expected_update_counts"],
            {"shared_endpoint": 256, "stratified": 256, "total": 512},
        )
        for item in exposure_audit["items"].values():
            self.assertEqual(len(item["regular_strata"]), 4)
            self.assertEqual(len(set(item["regular_strata"])), 4)

    def test_endpoint_balanced_schedule_preserves_cumulative_strata_balance(
        self,
    ) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]
        combine = self.namespace["_combine_exposure_audits"]

        parent = combine(
            audit(build(train), train),
            audit(build(train, phase_start=1, phase_count=2), train),
        )
        continuation = audit(
            build(
                train,
                phase_start=3,
                phase_count=1,
                endpoint_cycle_positions=self.namespace[
                    "ENDPOINT_BALANCED_CYCLE_POSITIONS"
                ],
                prior_exposure_audit=parent,
            ),
            train,
        )
        cumulative = combine(parent, continuation)

        self.assertTrue(cumulative["all_exact"])
        self.assertEqual(
            cumulative["expected"],
            {"total": 32, "shared_endpoint": 7, "stratified": 25},
        )

    def test_endpoint_balanced_long_continuation_runs_only_final_evaluation(
        self,
    ) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]
        combine = self.namespace["_combine_exposure_audits"]
        subtract = self.namespace["_subtract_exposure_audits"]
        endpoint_positions = self.namespace["ENDPOINT_BALANCED_CYCLE_POSITIONS"]

        first_phase = audit(build(train), train)
        middle_phases = audit(build(train, phase_start=1, phase_count=2), train)
        through_step_1536 = combine(first_phase, middle_phases)
        endpoint_schedule = build(
            train,
            phase_start=3,
            phase_count=1,
            endpoint_cycle_positions=endpoint_positions,
            prior_exposure_audit=through_step_1536,
        )
        endpoint_phase = audit(endpoint_schedule, train)
        through_step_2048 = combine(through_step_1536, endpoint_phase)
        recovered_step_1536 = subtract(through_step_2048, endpoint_phase)
        replayed_endpoint_schedule = build(
            train,
            phase_start=3,
            phase_count=1,
            endpoint_cycle_positions=endpoint_positions,
            prior_exposure_audit=recovered_step_1536,
        )
        self.assertEqual(replayed_endpoint_schedule, endpoint_schedule)
        self.assertEqual(
            recovered_step_1536["expected"],
            through_step_1536["expected"],
        )
        self.assertTrue(recovered_step_1536["all_exact"])
        continuation = build(
            train,
            phase_start=4,
            phase_count=4,
            endpoint_cycle_positions=endpoint_positions,
            prior_exposure_audit=through_step_2048,
        )
        continuation_audit = audit(continuation, train)
        cumulative = combine(through_step_2048, continuation_audit)

        self.assertEqual(len(continuation), 2048)
        self.assertEqual(
            (continuation[0].step, continuation[-1].step),
            (2049, 4096),
        )
        self.assertEqual(continuation[-1].kind, "stratified")
        self.assertEqual(
            continuation_audit["expected"],
            {"total": 32, "shared_endpoint": 16, "stratified": 16},
        )
        self.assertEqual(
            continuation_audit["expected_update_counts"],
            {"shared_endpoint": 1024, "stratified": 1024, "total": 2048},
        )
        self.assertTrue(continuation_audit["all_exact"])
        self.assertEqual(
            cumulative["expected"],
            {"total": 64, "shared_endpoint": 23, "stratified": 41},
        )
        self.assertTrue(cumulative["all_exact"])
        self.assertEqual(
            self.namespace["REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_EVALUATION_STEPS"],
            (4096,),
        )

    def test_parent_and_continuation_audits_combine_to_32_contacts(self) -> None:
        train = self._views(128)
        build = self.namespace["_build_training_schedule"]
        audit = self.namespace["_schedule_exposure_audit"]
        combine = self.namespace["_combine_exposure_audits"]

        combined = combine(
            audit(build(train), train),
            audit(build(train, phase_start=1, phase_count=3), train),
        )
        self.assertTrue(combined["all_exact"])
        self.assertEqual(
            combined["expected"],
            {"total": 32, "shared_endpoint": 4, "stratified": 28},
        )
        for item in combined["items"].values():
            counts = Counter(item["regular_strata"])
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

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

    def test_continuation_final_panel_matches_the_parent_final_panel(self) -> None:
        train = self._views(128)
        holdout = self._views(32, split=1)
        select = self.namespace["_evaluation_indices"]
        parent = select(512, train, holdout)
        continuation = select(
            2048,
            train,
            holdout,
            evaluation_steps=(1024, 2048),
            final_step=2048,
        )
        self.assertEqual(continuation, parent)

    def test_region_weighted_final_panel_uses_cost_capped_counts(self) -> None:
        train = self._views(128)
        holdout = self._views(32, split=1)
        indices = self.namespace["_evaluation_indices"](
            512,
            train,
            holdout,
            evaluation_steps=(0, 128, 256, 512),
            final_step=512,
            full_final_panel=False,
        )
        groups = Counter(
            self.namespace["_group_for_global_index"](index) for index in indices
        )
        self.assertEqual(groups, {"seen": 32, "holdout": 32, "original": 1})
        self.assertEqual(len(indices), len(set(indices)))

    def test_parent_payload_validation_pins_lineage_and_volume_paths(self) -> None:
        validate = self.namespace["_validate_parent_report_payload"]
        with tempfile.TemporaryDirectory() as temporary:
            volume_root = (Path(temporary) / "vo-internal-id").resolve()
            volume_name = "diffusion-arc-models"
            report_path = (
                volume_root
                / "results"
                / self.namespace["RESULT_FAMILY"]
                / self.namespace["CONTINUATION_PARENT_RUN_ID"]
                / "report.json"
            )
            checkpoint_path = (
                report_path.parent
                / self.namespace["STAGE_NAME"]
                / "flux-transformer-step-0512"
            )
            checkpoint_relative = "/" + str(
                checkpoint_path.relative_to(volume_root)
            )
            parent = {
                "status": "COMPLETE",
                "run_id": self.namespace["CONTINUATION_PARENT_RUN_ID"],
                "source": {
                    "resolved_commit": self.namespace[
                        "CONTINUATION_PARENT_SOURCE_COMMIT"
                    ]
                },
                "durability": {"remote_content_verified": True},
                "protocol": {
                    "name": self.namespace["PROTOCOL"],
                    "optimizer_steps": 512,
                    "final_optimizer_step": 512,
                    "loss": self.namespace["UNWEIGHTED_LOSS_POLICY"],
                },
                "pool": {
                    "pool_sha256": self.namespace[
                        "CONTINUATION_PARENT_POOL_SHA256"
                    ],
                    "counts": {"train": 512, "holdout": 128, "original": 1},
                },
                "stages": {
                    self.namespace["STAGE_NAME"]: {
                        "status": "COMPLETE",
                        "training": {
                            "completed_steps": 512,
                            "target_exposures": 4096,
                        },
                        "saved_checkpoint": {
                            "step": 512,
                            "state_scope": "transformer_only",
                            "remote_file_sizes_verified": True,
                            "path": str(checkpoint_path),
                            "volume_name": volume_name,
                            "volume_relative_path": checkpoint_relative,
                            "manifest": {"config.json": {"bytes": 1, "sha256": "x"}},
                        },
                    }
                },
            }
            resolved, checkpoint = validate(
                parent,
                parent_report_path=report_path,
                volume_root=volume_root,
                volume_name=volume_name,
            )
            self.assertEqual(resolved, checkpoint_path.resolve())
            self.assertEqual(checkpoint["step"], 512)
            parent["source"]["resolved_commit"] = "0" * 40
            with self.assertRaisesRegex(RuntimeError, "source commit"):
                validate(
                    parent,
                    parent_report_path=report_path,
                    volume_root=volume_root,
                    volume_name=volume_name,
                )

    def test_weighted_parent_accepts_only_the_pinned_terminal_upload_failure(
        self,
    ) -> None:
        validate = self.namespace["_validate_parent_report_payload"]
        contract = self.namespace["REGION_WEIGHTED_PARENT_CONTRACT"]
        with tempfile.TemporaryDirectory() as temporary:
            volume_root = (Path(temporary) / "vo-internal-id").resolve()
            volume_name = "diffusion-arc-models"
            report_path = (
                volume_root
                / "results"
                / self.namespace["RESULT_FAMILY"]
                / contract.run_id
                / "report.json"
            )
            checkpoint_path = (
                report_path.parent
                / self.namespace["STAGE_NAME"]
                / "flux-transformer-step-0512"
            )
            checkpoint_relative = "/" + str(
                checkpoint_path.relative_to(volume_root)
            )
            parent = {
                "status": "FAIL",
                "failed_stage": "experiment",
                "completed_at_utc": "2026-08-26T22:08:55+00:00",
                "failures": [
                    {
                        **self.namespace["REGION_WEIGHTED_PARENT_TERMINAL_FAILURE"],
                        "recorded_at_utc": "2026-08-26T22:09:02+00:00",
                    }
                ],
                "run_id": contract.run_id,
                "source": {"resolved_commit": contract.source_commit},
                "durability": {"remote_content_verified": False},
                "protocol": {
                    "name": contract.protocol_name,
                    "optimizer_steps": 512,
                    "final_optimizer_step": contract.final_step,
                    "loss": contract.loss_policy,
                    "region_loss": {
                        "weights": self.namespace["REGION_LOSS_WEIGHTS"]
                    },
                },
                "pool": {
                    "pool_sha256": contract.pool_sha256,
                    "counts": {"train": 512, "holdout": 128, "original": 1},
                },
                "stages": {
                    self.namespace["STAGE_NAME"]: {
                        "status": "COMPLETE",
                        "training": {
                            "completed_steps": 512,
                            "target_exposures": 4096,
                        },
                        "saved_checkpoint": {
                            "step": 512,
                            "state_scope": "transformer_only",
                            "remote_file_sizes_verified": True,
                            "path": str(checkpoint_path),
                            "volume_name": volume_name,
                            "volume_relative_path": checkpoint_relative,
                            "manifest": {
                                "config.json": {"bytes": 1, "sha256": "x"}
                            },
                        },
                    }
                },
            }

            resolved, checkpoint = validate(
                parent,
                parent_report_path=report_path,
                volume_root=volume_root,
                volume_name=volume_name,
                contract=contract,
            )
            self.assertEqual(resolved, checkpoint_path.resolve())
            self.assertEqual(checkpoint["step"], 512)

            parent["failures"][-1]["message"] = "different failure"
            with self.assertRaisesRegex(RuntimeError, "terminal report-upload"):
                validate(
                    parent,
                    parent_report_path=report_path,
                    volume_root=volume_root,
                    volume_name=volume_name,
                    contract=contract,
                )

    def test_endpoint_balanced_parent_accepts_complete_step_1536_run(self) -> None:
        validate = self.namespace["_validate_parent_report_payload"]
        contract = self.namespace[
            "REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_CONTRACT"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            volume_root = (Path(temporary) / "vo-internal-id").resolve()
            volume_name = "diffusion-arc-models"
            report_path = (
                volume_root
                / "results"
                / self.namespace["RESULT_FAMILY"]
                / contract.run_id
                / "report.json"
            )
            checkpoint_path = (
                report_path.parent
                / self.namespace["STAGE_NAME"]
                / "flux-transformer-step-1536"
            )
            parent = {
                "status": "COMPLETE",
                "run_id": contract.run_id,
                "source": {"resolved_commit": contract.source_commit},
                "durability": {"remote_content_verified": True},
                "protocol": {
                    "name": contract.protocol_name,
                    "optimizer_steps": contract.optimizer_steps,
                    "final_optimizer_step": contract.final_step,
                    "loss": contract.loss_policy,
                    "region_loss": {
                        "weights": self.namespace["REGION_LOSS_WEIGHTS"]
                    },
                },
                "pool": {
                    "pool_sha256": contract.pool_sha256,
                    "counts": {"train": 512, "holdout": 128, "original": 1},
                },
                "stages": {
                    self.namespace["STAGE_NAME"]: {
                        "status": "COMPLETE",
                        "training": {
                            "completed_steps": contract.optimizer_steps,
                            "target_exposures": contract.optimizer_steps * 8,
                        },
                        "saved_checkpoint": {
                            "step": contract.final_step,
                            "state_scope": "transformer_only",
                            "remote_file_sizes_verified": True,
                            "path": str(checkpoint_path),
                            "volume_name": volume_name,
                            "volume_relative_path": "/" + str(
                                checkpoint_path.relative_to(volume_root)
                            ),
                            "manifest": {
                                "config.json": {"bytes": 1, "sha256": "x"}
                            },
                        },
                    }
                },
            }

            resolved, checkpoint = validate(
                parent,
                parent_report_path=report_path,
                volume_root=volume_root,
                volume_name=volume_name,
                contract=contract,
            )
            self.assertEqual(resolved, checkpoint_path.resolve())
            self.assertEqual(checkpoint["step"], 1536)

    def test_endpoint_balanced_long_parent_accepts_complete_step_2048_run(
        self,
    ) -> None:
        validate = self.namespace["_validate_parent_report_payload"]
        contract = self.namespace[
            "REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_CONTRACT"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            volume_root = (Path(temporary) / "vo-internal-id").resolve()
            volume_name = "diffusion-arc-models"
            report_path = (
                volume_root
                / "results"
                / self.namespace["RESULT_FAMILY"]
                / contract.run_id
                / "report.json"
            )
            checkpoint_path = (
                report_path.parent
                / self.namespace["STAGE_NAME"]
                / "flux-transformer-step-2048"
            )
            parent = {
                "status": "COMPLETE",
                "run_id": contract.run_id,
                "source": {"resolved_commit": contract.source_commit},
                "durability": {"remote_content_verified": True},
                "protocol": {
                    "name": contract.protocol_name,
                    "optimizer_steps": contract.optimizer_steps,
                    "final_optimizer_step": contract.final_step,
                    "loss": contract.loss_policy,
                    "region_loss": {
                        "weights": self.namespace["REGION_LOSS_WEIGHTS"]
                    },
                },
                "pool": {
                    "pool_sha256": contract.pool_sha256,
                    "counts": {"train": 512, "holdout": 128, "original": 1},
                },
                "stages": {
                    self.namespace["STAGE_NAME"]: {
                        "status": "COMPLETE",
                        "training": {
                            "completed_steps": contract.optimizer_steps,
                            "target_exposures": contract.optimizer_steps * 8,
                        },
                        "saved_checkpoint": {
                            "step": contract.final_step,
                            "state_scope": "transformer_only",
                            "remote_file_sizes_verified": True,
                            "path": str(checkpoint_path),
                            "volume_name": volume_name,
                            "volume_relative_path": "/" + str(
                                checkpoint_path.relative_to(volume_root)
                            ),
                            "manifest": {
                                "config.json": {"bytes": 1, "sha256": "x"}
                            },
                        },
                    }
                },
            }

            resolved, checkpoint = validate(
                parent,
                parent_report_path=report_path,
                volume_root=volume_root,
                volume_name=volume_name,
                contract=contract,
            )
            self.assertEqual(resolved, checkpoint_path.resolve())
            self.assertEqual(checkpoint["step"], 2048)

    def test_final_report_upload_retries_a_transient_block_failure(self) -> None:
        upload = self.namespace["_upload_and_verify_final_report"]
        function_globals = upload.__globals__

        class FakeBatch:
            def __init__(self, volume):
                self.volume = volume

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def put_file(self, local_path: str, remote_path: str) -> None:
                self.volume.payload = Path(local_path).read_bytes()
                self.volume.remote_path = remote_path
                self.volume.upload_count += 1

        class FakeVolume:
            def __init__(self):
                self.payload = b""
                self.remote_path = ""
                self.upload_count = 0
                self.read_count = 0

            def batch_upload(self, *, force: bool):
                self.force = force
                return FakeBatch(self)

            def read_file(self, remote_path: str):
                self.read_count += 1
                self.asserted_remote_path = remote_path
                if self.read_count == 1:
                    raise RuntimeError("block not found")
                return (self.payload,)

        fake_volume = FakeVolume()
        fake_modal = SimpleNamespace(
            Volume=SimpleNamespace(from_name=lambda _: fake_volume)
        )
        sentinel = object()
        previous_report = function_globals.get("REPORT", sentinel)
        previous_path = function_globals.get("REPORT_PATH", sentinel)
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            function_globals["REPORT"] = {"status": "COMPLETE"}
            function_globals["REPORT_PATH"] = report_path
            function_globals["save_report"]()
            try:
                with patch.dict(sys.modules, {"modal": fake_modal}), patch.object(
                    function_globals["time"], "sleep", return_value=None
                ):
                    upload(volume_name="diffusion-arc-models", run_id="retry-test")
                report = function_globals["REPORT"]
                self.assertTrue(report["durability"]["remote_content_verified"])
                self.assertEqual(report["durability"]["upload_attempts"], 2)
                self.assertEqual(fake_volume.upload_count, 2)
                self.assertEqual(fake_volume.read_count, 2)
                self.assertEqual(fake_volume.payload, report_path.read_bytes())
            finally:
                if previous_report is sentinel:
                    function_globals.pop("REPORT", None)
                else:
                    function_globals["REPORT"] = previous_report
                if previous_path is sentinel:
                    function_globals.pop("REPORT_PATH", None)
                else:
                    function_globals["REPORT_PATH"] = previous_path

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

    def test_region_losses_separate_grid_boundary_and_outer_padding(self) -> None:
        views = tuple(
            replace(
                self._view(0, index),
                target_grid=((1, 2, 3), (1, 2, 3), (1, 2, 3)),
            )
            for index in range(2)
        )
        squared = torch.full((2, 30, 30, 128), 4.0, dtype=torch.float32)
        squared[:, :3, :3] = 2.0
        squared[:, 3, :4] = 3.0
        squared[:, :3, 3] = 3.0
        metrics = self.namespace["_per_row_region_losses"](
            squared.reshape(2, 900, 128),
            views,
        )

        self.assertEqual(tuple(metrics.shape), (2, 5))
        expected_total = (9 * 2 + 7 * 3 + 884 * 4) / 900
        expected_padding = (7 * 3 + 884 * 4) / 891
        expected = torch.tensor(
            (expected_total, 2.0, 3.0, 4.0, expected_padding)
        )
        torch.testing.assert_close(metrics[0], expected)
        torch.testing.assert_close(metrics[1], expected)

        masks = self.namespace["_region_masks"](
            views[0].target_grid,
            device=torch.device("cpu"),
        )
        self.assertEqual([int(mask.sum()) for mask in masks], [9, 7, 884, 891])

    def test_region_weighted_objective_uses_per_region_means(self) -> None:
        view = replace(
            self._view(0, 0),
            target_grid=((1, 2, 3), (1, 2, 3), (1, 2, 3)),
        )
        squared = torch.full((1, 30, 30, 128), 4.0, dtype=torch.float32)
        squared[:, :3, :3] = 2.0
        squared[:, 3, :4] = 3.0
        squared[:, :3, 3] = 3.0
        predicted = squared.sqrt().reshape(1, 900, 128).requires_grad_(True)
        target = torch.zeros_like(predicted)

        objective, raw_squared, regions, contributions = self.namespace[
            "_training_objective"
        ](
            predicted,
            target,
            (view,),
            loss_policy=self.namespace["REGION_WEIGHTED_LOSS_POLICY"],
        )
        torch.testing.assert_close(raw_squared, squared.reshape(1, 900, 128))
        torch.testing.assert_close(objective, torch.tensor(2.5))
        torch.testing.assert_close(
            contributions[0],
            torch.tensor((1.2, 0.9, 0.4)),
        )
        torch.testing.assert_close(objective, contributions.sum())
        self.assertEqual(tuple(regions.shape), (1, 5))
        objective.backward()
        self.assertTrue(bool((predicted.grad != 0).all()))

    def test_unweighted_objective_remains_the_global_mse(self) -> None:
        view = replace(
            self._view(0, 0),
            target_grid=((1, 2, 3), (1, 2, 3), (1, 2, 3)),
        )
        predicted = torch.linspace(-1, 1, 900 * 128).reshape(1, 900, 128)
        target = torch.zeros_like(predicted)
        objective, squared, _, _ = self.namespace["_training_objective"](
            predicted,
            target,
            (view,),
            loss_policy=self.namespace["UNWEIGHTED_LOSS_POLICY"],
        )
        torch.testing.assert_close(objective, squared.mean())

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
                    "sampling": {
                        "kind": plan.kind,
                        "flow_seed": self.namespace["FLOW_SEED"] + plan.step,
                        "prescribed_timestep_strata": (
                            None
                            if plan.timestep_strata is None
                            else list(plan.timestep_strata)
                        ),
                    },
                    "per_exposure": exposures,
                }
            )

        summarize = self.namespace["_summarize_flow_sampling"]
        planned_batches = self.namespace["_schedule_payload"](schedule, train)
        summary = summarize(
            records,
            exposure_audit=audit,
            planned_batches=planned_batches,
        )
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
        with self.assertRaisesRegex(RuntimeError, "training keys differ"):
            summarize(
                records,
                exposure_audit=audit,
                planned_batches=planned_batches,
            )

    def test_flow_summary_accepts_the_1536_update_continuation(self) -> None:
        train = self._views(128)
        schedule = self.namespace["_build_training_schedule"](
            train,
            phase_start=1,
            phase_count=3,
        )
        audit = self.namespace["_schedule_exposure_audit"](schedule, train)
        records = []
        for plan in schedule:
            exposures = []
            for row, view_index in enumerate(plan.view_indices):
                stratum = (
                    None if plan.timestep_strata is None else plan.timestep_strata[row]
                )
                view = train[view_index]
                exposures.append(
                    {
                        "key": view.serialized_key,
                        "geometry": view.geometry.value,
                        "sampling_kind": plan.kind,
                        "timestep": 1.0 if stratum is None else (stratum + 0.5) / 8,
                        "timestep_stratum": stratum,
                        "total_loss": 1.0,
                        "grid_loss": 2.0,
                        "padding_loss": 0.5,
                    }
                )
            records.append(
                {
                    "step": plan.step,
                    "sampling": {
                        "kind": plan.kind,
                        "flow_seed": self.namespace["FLOW_SEED"] + plan.step,
                        "prescribed_timestep_strata": (
                            None
                            if plan.timestep_strata is None
                            else list(plan.timestep_strata)
                        ),
                    },
                    "per_exposure": exposures,
                }
            )

        summary = self.namespace["_summarize_flow_sampling"](
            records,
            exposure_audit=audit,
            planned_batches=self.namespace["_schedule_payload"](schedule, train),
        )
        self.assertEqual(
            summary["update_counts"],
            {"shared_endpoint": 192, "stratified": 1344},
        )
        self.assertEqual(summary["exposure_counts"]["total"], 12288)

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

    def test_continuation_config_changes_only_the_approved_step_budget(self) -> None:
        base = load_single_gpu_training_config(CONFIG)
        continuation = load_single_gpu_training_config(CONTINUATION_CONFIG)
        self.assertEqual(continuation.loop.max_optimizer_steps, 1536)
        self.assertEqual(continuation.loop.optimizer_batch_size, 8)
        self.assertEqual(continuation.loop.flux_microbatch_size, 8)
        self.assertEqual(continuation.loop.min_context_pairs_before_target, 4)
        self.assertEqual(continuation.optimizer, base.optimizer)
        self.assertEqual(continuation.models, base.models)
        self.assertEqual(continuation.data, base.data)

    def test_region_weighted_continuation_config_uses_1024_updates(self) -> None:
        base = load_single_gpu_training_config(CONFIG)
        continuation = load_single_gpu_training_config(
            REGION_WEIGHTED_CONTINUATION_CONFIG
        )
        self.assertEqual(continuation.loop.max_optimizer_steps, 1024)
        self.assertEqual(continuation.loop.optimizer_batch_size, 8)
        self.assertEqual(continuation.loop.flux_microbatch_size, 8)
        self.assertEqual(continuation.loop.min_context_pairs_before_target, 4)
        self.assertEqual(continuation.optimizer, base.optimizer)
        self.assertEqual(continuation.models, base.models)
        self.assertEqual(continuation.data, base.data)

    def test_endpoint_balanced_continuation_config_uses_512_updates(self) -> None:
        base = load_single_gpu_training_config(CONFIG)
        continuation = load_single_gpu_training_config(
            REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_CONFIG
        )
        self.assertEqual(continuation.loop.max_optimizer_steps, 512)
        self.assertEqual(continuation.loop.optimizer_batch_size, 8)
        self.assertEqual(continuation.loop.flux_microbatch_size, 8)
        self.assertEqual(continuation.loop.min_context_pairs_before_target, 4)
        self.assertEqual(continuation.optimizer, base.optimizer)
        self.assertEqual(continuation.models, base.models)
        self.assertEqual(continuation.data, base.data)

    def test_endpoint_balanced_long_continuation_config_uses_2048_updates(
        self,
    ) -> None:
        base = load_single_gpu_training_config(CONFIG)
        continuation = load_single_gpu_training_config(
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_CONFIG
        )
        self.assertEqual(continuation.loop.max_optimizer_steps, 2048)
        self.assertEqual(continuation.loop.optimizer_batch_size, 8)
        self.assertEqual(continuation.loop.flux_microbatch_size, 8)
        self.assertEqual(continuation.loop.min_context_pairs_before_target, 4)
        self.assertEqual(continuation.optimizer, base.optimizer)
        self.assertEqual(continuation.models, base.models)
        self.assertEqual(continuation.data, base.data)

    def test_notebook_is_clean_and_all_code_cells_compile(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def test_continuation_notebook_is_clean_and_all_code_cells_compile(self) -> None:
        notebook = json.loads(CONTINUATION_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile("".join(cell["source"]), f"continuation-cell-{index}", "exec")

    def test_region_weighted_notebook_is_clean_and_all_code_cells_compile(
        self,
    ) -> None:
        notebook = json.loads(REGION_WEIGHTED_NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile("".join(cell["source"]), f"region-loss-cell-{index}", "exec")

    def test_region_weighted_continuation_notebook_is_clean_and_compiles(
        self,
    ) -> None:
        notebook = json.loads(
            REGION_WEIGHTED_CONTINUATION_NOTEBOOK.read_text(encoding="utf-8")
        )
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile(
                "".join(cell["source"]),
                f"region-loss-continuation-cell-{index}",
                "exec",
            )

    def test_endpoint_balanced_continuation_notebook_is_clean_and_compiles(
        self,
    ) -> None:
        notebook = json.loads(
            REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_NOTEBOOK.read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile(
                "".join(cell["source"]),
                f"endpoint-balanced-continuation-cell-{index}",
                "exec",
            )

    def test_endpoint_balanced_long_continuation_notebook_is_clean_and_compiles(
        self,
    ) -> None:
        notebook = json.loads(
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_NOTEBOOK.read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(notebook["cells"]), 6)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell.get("outputs", []), [])
            self.assertIsNone(cell.get("execution_count"))
            compile(
                "".join(cell["source"]),
                f"endpoint-balanced-long-continuation-cell-{index}",
                "exec",
            )


if __name__ == "__main__":
    unittest.main()
