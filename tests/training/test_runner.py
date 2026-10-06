"""CPU checks of complete-epoch orchestration with substitute model components."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.training import training_loop, runner
from diffusion_arc.training.run import validation as validation_module
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.step.loss import RegionLossWeights
from diffusion_arc.training.trainer import TargetMetrics, TrainStepResult


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"


class _Samples(list):
    def __init__(self, count):
        keys = tuple(SimpleNamespace(
            task_id=f"task-{index}", augmentation=SimpleNamespace(pair_indices=(0, 1, 2, 3, 4)),
        ) for index in range(count))
        super().__init__(SimpleNamespace(key=key, target_grids=(None,) * 4) for key in keys)
        self.dataset = SimpleNamespace(keys=keys, fingerprint="test-data-identity",
                                       fingerprint_for=lambda expected: "test-data-identity")


def _step_result(sources):
    target_count = len(sources) * 3
    return TrainStepResult(
        loss=1.0, learning_rates=(1e-5,), gradient_norm=0.5, gradient_clipped=False,
        step_seconds=2.0, source_count=len(sources), target_count=target_count,
        nvarc_tokens=len(sources) * 100, nvarc_calls=1, vae_calls=1,
        flux_text_tokens=target_count * 200, flux_image_tokens=target_count * 900,
        flux_joint_tokens=target_count * 1100, flux_forward_calls=6,
        target_metrics=tuple(TargetMetrics(
            loss=1.0, timestep=1.0 if index < (4 if target_count == 48 else 3) else 0.5,
            context_pair_count=2 + index % 3, has_boundary=True,
            mse_canvas=1.0, mse_grid=1.0, mse_boundary=1.0,
            contribution_canvas=0.1, contribution_grid=0.8, contribution_boundary=0.1,
        ) for index in range(target_count)), diagnostics=Mock(),
    )


class RunnerTests(unittest.TestCase):
    def test_weights_only_initialization_keeps_global_progress_but_builds_fresh_optimizer(self):
        with ExitStack() as stack:
            bindings = self._setup(stack)
            recipe = self.config.recipe_dict()
            source = json.loads((ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json").read_text())
            recipe["models"].update({key: source["models"][key] for key in (
                "initialization", "completed_epochs", "transformer_source")})
            recipe["models"]["transformer_checkpoint"] = "/downloaded/transformer"
            recipe["logging"].update(parent_run_id="test-run", resume_parent_run=True)
            self.config.path.write_text(json.dumps(recipe))
            config = load_training_config(self.config.path)
            parent = {"wandb_run": self.logger.run_identity, "workload_totals": {"train/nvarc_tokens": 500}}
            restore = stack.enter_context(patch.object(training_loop, "restore_training_state"))
            progress = runner.run_training(config, parent_run_metadata=parent)
        self.assertEqual((progress.epoch, progress.optimizer_step, progress.optimizer_start_step), (3, 6, 4))
        self.assertEqual((progress.sources_seen, progress.targets_seen), (78, 234))
        self.assertEqual(progress.workload_totals["train/nvarc_tokens"], 3100)
        restore.assert_not_called()
        bindings["build_optimizer"].assert_called_once()
        bindings["build_lr_scheduler"].assert_called_once()
        self.assertEqual(bindings["load_models"].call_args.kwargs["transformer_checkpoint"], Path("/downloaded/transformer"))
        self.assertEqual(bindings["build_dataloader"].call_args.kwargs["start_index"], 0)
        self.assertEqual(bindings["WandbLogger"].call_args.kwargs["resume_run"], self.logger.run_identity)
        self.assertEqual([call.args[0]["progress/optimizer_step"] for call in self.logger.log.call_args_list], [5, 6])
        self.assertEqual(self.evaluate.call_count, 1)
        self.assertTrue(self.logger.run.config.update.call_args.args[0]["epoch_3"]["optimizer_reset"])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        payload = json.loads(RECIPE.read_text())
        payload["data"]["expected_task_count"] = 26
        payload["scheduler"].update(warmup_steps=2, first_update_warmup_fraction=0.5)
        path = Path(temp.name) / "run.json"
        path.write_text(json.dumps(payload))
        self.config = load_training_config(path, expected_type="training_run")
        self.samples = _Samples(26)

    def _setup(self, stack):
        self.models = SimpleNamespace(
            tokenizer=object(), nvarc_encoder=object(), vae_encoder=object(), vae_decoder=object(),
            flux_transformer=object(), flux_compute_dtype=torch.bfloat16, flux_packing_backend=object(),
        )
        self.optimizer, self.scheduler = Mock(), Mock()
        self.sampler, self.loss = object(), object()
        self.validation_samples = tuple(SimpleNamespace(input_ids=(1, 2)) for _ in range(104))
        self.validation_targets = tuple(
            SimpleNamespace(task_id=f"val-{index}", query_index=query, expected_grid=((index % 10,),))
            for index in range(100) for query in range(2 if index in (0, 97, 98, 99) else 1)
        )
        self.sampling_targets = self.validation_targets[:33]
        self.input_grids = {(target.task_id, target.query_index): ((1,),) for target in self.sampling_targets}
        self.validation_packs = tuple(tuple(range(start, start + 13)) for start in range(0, 104, 13))
        self.sampling_packs = (tuple(range(13)), tuple(range(13, 26)), tuple(range(26, 33)))
        self.validation_result = {
            "target_count": 104, "boundary_target_count": 100,
            "uniform_inner_loss": 2.0, "endpoint_loss": 8.0,
            "per_timestep": [dict(timestep=value, weighted_loss=8.0 if value == 1 else 2.0,
                                  canvas_loss=3.0, grid_loss=4.0, boundary_loss=5.0)
                             for value in (.1, .3, .5, .7, .9, 1.)],
            "nvarc_tokens": 0, "flux_text_tokens": 1248, "flux_image_tokens": 561600,
            "flux_joint_tokens": 562848, "flux_forward_calls": 48,
        }
        self.sampling_result = {
            "task_count": 32, "prediction_count": 66, "shape_correct_count": 20,
            "pass_at_1": .25, "pass_at_2": .5, "solved_at_1": 8, "solved_at_2": 16,
            "shape_accuracy": 20 / 66, "cell_accuracy": .8, "invalid_grid_rate": 10 / 66,
            "nvarc_tokens": 0, "flux_text_tokens": 5280, "flux_image_tokens": 2376000,
            "flux_joint_tokens": 2381280, "flux_forward_calls": 240,
        }
        self.step = Mock(side_effect=_step_result)
        self.logger = MagicMock()
        self.logger.__enter__.return_value = self.logger
        self.logger.__exit__.return_value = False
        self.logger.collect_hardware_metrics.return_value = {"system/gpu_peak_allocated_bytes": 123}
        self.logger.build_prediction_table.side_effect = lambda *a, **kw: object()
        self.logger.run_identity = {"id": "test-run", "project": "test-project", "entity": "test-owner"}
        self.events = Mock()
        for name, method in (("begin", self.logger.begin_step), ("step", self.step),
                             ("hardware", self.logger.collect_hardware_metrics),
                             ("log", self.logger.log), ("table", self.logger.build_prediction_table)):
            self.events.attach_mock(method, name)
        patches = {
            "prepare_device": torch.device("cpu"), "seed_process": None,
            "load_models": self.models, "build_dataloader": self.samples,
            "build_optimizer": self.optimizer, "build_lr_scheduler": self.scheduler,
            "build_flow_generator": object(), "build_flow_sampler": self.sampler,
            "build_velocity_loss": self.loss, "Trainer": SimpleNamespace(step=self.step),
            "WandbLogger": self.logger,
            "validate_validation_bundle": {"validation_task_ids": tuple(f"val-{i}" for i in range(100)),
                                           "validation_output_count": 104,
                                           "sampling_task_ids": tuple(f"val-{i}" for i in range(32)),
                                           "sampling_output_count": 33},
            "tokenize_validation_targets": (self.validation_samples, (((0,),),) * 104),
            "prepare_validation_cache": self.validation_targets,
            "load_sampling_input_grids": self.input_grids,
        }
        bindings = {name: stack.enter_context(patch.object(training_loop if hasattr(training_loop, name) else runner, name, return_value=value))
                    for name, value in patches.items()}
        bindings["plan_validation_packs"] = stack.enter_context(patch.object(
            runner, "plan_validation_packs", side_effect=[self.validation_packs, self.sampling_packs],
        ))
        self.evaluate = stack.enter_context(patch.object(
            runner, "evaluate_packed_loss", side_effect=lambda *a, **kw: deepcopy(self.validation_result),
        ))
        self.events.attach_mock(self.evaluate, "validate")
        self.generate = stack.enter_context(patch.object(
            runner, "generate_packed_samples", side_effect=lambda *a, **kw: deepcopy(self.sampling_result),
        ))
        self.events.attach_mock(self.generate, "sample")
        self.save = stack.enter_context(patch.object(runner, "save_checkpoint", return_value={
            "checkpoint/seconds": 1.5, "checkpoint/size_bytes": 1024, "checkpoint/path": "/test/checkpoint",
        }))
        self.events.attach_mock(self.save, "checkpoint")
        return bindings

    def test_complete_epoch_keeps_order_and_partial_batch_without_extra_updates_or_profiling(self):
        with ExitStack() as stack:
            bindings = self._setup(stack)
            profiler = stack.enter_context(patch("torch.profiler.profile"))
            stack.enter_context(patch.object(runner.time, "perf_counter",
                                            side_effect=[10, 13, 14, 20, 30, 35, 36, 45]))
            progress = runner.run_training(self.config)

        self.assertEqual(progress, runner.RunProgress(2, 26, 78, {
            "train/nvarc_tokens": 2600, "train/flux_text_tokens": 15600,
            "train/flux_image_tokens": 70200, "train/flux_joint_tokens": 85800,
            "train/flux_forward_calls": 12,
            "val/nvarc_tokens": 208, "val/flux_text_tokens": 2496,
            "val/flux_image_tokens": 1123200, "val/flux_joint_tokens": 1125696,
            "val/flux_forward_calls": 96,
            "sample/nvarc_tokens": 0, "sample/flux_text_tokens": 10560,
            "sample/flux_image_tokens": 4752000, "sample/flux_joint_tokens": 4762560,
            "sample/flux_forward_calls": 480,
        }))
        self.assertEqual([event[0] for event in self.events.mock_calls],
                         ["validate", "sample", "table", "log", "begin", "step", "hardware", "log",
                          "begin", "step", "hardware", "validate", "sample", "table", "checkpoint", "log"])
        initial, first, final = [call.args[0] for call in self.logger.log.call_args_list]
        self.assertEqual(initial["progress/optimizer_step"], 0)
        self.assertEqual(initial["val/nvarc_tokens"], 208)
        self.assertEqual(initial["val/seconds"], 3)
        self.assertNotIn("train/loss", initial)
        self.assertNotIn("val/loss", first)
        self.assertEqual(final["val/loss"], 2.5)
        self.assertEqual(final["val/nvarc_tokens"], 0)
        self.assertEqual(final["val/seconds"], 5)
        self.assertEqual(initial["sample/seconds"], 6)
        self.assertEqual(final["sample/seconds"], 9)
        self.assertEqual(final["sample/pass_at_2"], .5)
        self.assertNotIn("sample/pass_at_2", first)
        self.assertIsNot(initial["sample/predictions"], final["sample/predictions"])
        self.assertEqual((first["train/targets"], final["train/targets"]), (48, 30))
        self.assertEqual((first["train/targets_t1"], final["train/targets_t1"]), (4, 3))
        self.assertEqual(final["progress/optimizer_step"], 2)
        self.assertEqual(final["progress/epoch_fraction"], 1.0)
        self.assertEqual(final["train/flux_image_tokens_total"], 70200)
        self.assertEqual(final["system/gpu_peak_allocated_bytes"], 123)
        self.assertEqual(final["checkpoint/path"], "/test/checkpoint")
        self.assertEqual(final["checkpoint/size_bytes"], 1024)
        self.assertEqual(final["checkpoint/seconds"], 1.5)
        self.save.assert_called_once()
        self.assertIs(self.save.call_args.args[0], progress)
        self.assertIs(self.save.call_args.kwargs["transformer"], self.models.flux_transformer)
        self.assertIs(self.save.call_args.kwargs["optimizer"], self.optimizer)
        self.assertIs(self.save.call_args.kwargs["scheduler"], self.scheduler)
        self.assertEqual(self.save.call_args.kwargs["wandb_run"], self.logger.run_identity)
        bindings["validate_validation_bundle"].assert_called_once_with(
            self.config.recipe["validation"]["bundle"], split_seed=167, verify_payloads=False,
        )
        bindings["prepare_validation_cache"].assert_called_once()
        self.assertEqual(bindings["prepare_validation_cache"].call_args.kwargs, {
            "nvarc_encoder": self.models.nvarc_encoder, "vae_encoder": self.models.vae_encoder,
            "nvarc_max_tokens": 134400, "vae_batch_size": 48,
        })
        for call in self.evaluate.call_args_list:
            self.assertEqual(call.args, (self.models.flux_transformer, self.validation_targets,
                                        self.validation_packs))
            self.assertEqual(call.kwargs, {"timesteps": (.1, .3, .5, .7, .9, 1.), "seed": 10167,
                                          "compute_dtype": torch.bfloat16,
                                          "backend": self.models.flux_packing_backend,
                                          "weights": RegionLossWeights(.1, .8, .1)})
        bindings["load_sampling_input_grids"].assert_called_once_with(
            self.config.recipe["validation"]["bundle"], tuple(f"val-{i}" for i in range(32)),
        )
        for call in self.generate.call_args_list:
            self.assertEqual(call.args, (self.models.flux_transformer, self.models.vae_decoder,
                                        self.sampling_targets, self.sampling_packs))
            self.assertTrue(all(a is b for a, b in zip(call.args[2], self.validation_targets)))
            self.assertEqual(call.kwargs, {"steps": 40, "seeds": (20167, 20168),
                                          "compute_dtype": torch.bfloat16,
                                          "backend": self.models.flux_packing_backend})
        for call in self.logger.build_prediction_table.call_args_list:
            self.assertEqual(call.kwargs, {"targets": self.sampling_targets, "input_grids": self.input_grids})
        self.logger.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual([call.args[0] for call in self.step.call_args_list],
                         [tuple(self.samples[:16]), tuple(self.samples[16:])])
        bindings["load_models"].assert_called_once_with(self.config, device=torch.device("cpu"),
                                                       transformer_checkpoint=None)
        bindings["build_dataloader"].assert_called_once_with(self.config, tokenizer=self.models.tokenizer,
                                                            start_index=0)
        supplied = bindings["Trainer"].call_args.kwargs
        for name in ("nvarc_encoder", "vae_encoder", "flux_packing_backend"):
            self.assertIs(supplied[name], getattr(self.models, name))
        flux_model = supplied["flux_transformer"]
        self.assertIsInstance(flux_model, PackedFluxForward)
        self.assertIs(flux_model.transformer, self.models.flux_transformer)
        self.assertIs(flux_model.backend, self.models.flux_packing_backend)
        self.assertEqual(flux_model.compute_dtype, self.models.flux_compute_dtype)
        self.assertIs(supplied["flow_sampler"], self.sampler)
        self.assertIs(supplied["loss_function"], self.loss)
        self.assertIs(supplied["optimizer"], self.optimizer)
        self.assertIs(supplied["scheduler"], self.scheduler)
        self.assertEqual(supplied["nvarc_max_tokens"], 134400)
        self.assertEqual(supplied["vae_microbatch_size"], 48)
        self.assertEqual(supplied["flux_max_joint_tokens"], 12288)
        self.assertEqual(supplied["flux_max_spread_tokens"], self.config.flux_packing.max_spread_tokens)
        self.assertEqual(supplied["min_context_pairs_before_target"], 2)
        self.assertEqual(supplied["max_grad_norm"], 1.0)
        self.optimizer.step.assert_not_called()
        self.scheduler.step.assert_not_called()
        profiler.assert_not_called()

    def test_incompatible_prepared_view_fails_before_optimizer_creation_or_training(self):
        for fault in ("count", "duplicate", "pairs"):
            with self.subTest(fault=fault), ExitStack() as stack:
                self.samples = _Samples(25 if fault == "count" else 26)
                if fault == "duplicate":
                    self.samples.dataset.keys[-1].task_id = "task-0"
                if fault == "pairs":
                    self.samples.dataset.keys[-1].augmentation.pair_indices = (0, 1, 2, 3)
                bindings = self._setup(stack)
                with self.assertRaisesRegex(ValueError, "training view"):
                    runner.run_training(self.config)
                bindings["build_optimizer"].assert_not_called()
                self.step.assert_not_called()

    def test_early_exhaustion_or_wrong_reported_counts_cannot_finish_successfully(self):
        for fault in ("empty", "truncated", "reported_count"):
            with self.subTest(fault=fault), ExitStack() as stack:
                self.samples = _Samples(26)
                if fault == "empty":
                    self.samples.clear()
                elif fault == "truncated":
                    self.samples.pop()
                self._setup(stack)
                if fault == "reported_count":
                    self.step.side_effect = lambda sources: replace(_step_result(sources), target_count=48)
                error, message = {
                    "empty": (ValueError, "data exhausted"),
                    "truncated": (RuntimeError, "training counts differ"),
                    "reported_count": (RuntimeError, "target count differs"),
                }[fault]
                with self.assertRaisesRegex(error, message):
                    runner.run_training(self.config)

    def test_step_failure_propagates_without_retry_or_later_updates(self):
        with ExitStack() as stack:
            self._setup(stack)
            self.step.side_effect = FloatingPointError("non-finite loss")
            with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
                runner.run_training(self.config)
        self.step.assert_called_once()
        self.optimizer.step.assert_not_called()
        self.scheduler.step.assert_not_called()
        self.logger.log.assert_called_once()  # Only the initial evaluation was completed.
        self.assertEqual(self.logger.log.call_args.args[0]["progress/optimizer_step"], 0)
        self.assertIs(self.logger.__exit__.call_args.args[0], FloatingPointError)

    def _resume(self, stack):
        bindings = self._setup(stack)
        first = _step_result(self.samples[:16])
        progress = runner.RunProgress(1, 16, 48, {
            f"train/{name}": getattr(first, name)
            for name in self.config.recipe["logging"]["workload_counters"]["names"]
        })
        for phase, result in (("val", self.validation_result), ("sample", self.sampling_result)):
            progress.workload_totals.update({f"{phase}/{name}": result[name]
                for name in self.config.recipe["logging"]["workload_counters"]["names"]})
        progress.workload_totals["val/nvarc_tokens"] = 208
        path = Path("/test/checkpoints/test-run/step-000001")
        config = replace(self.config, recipe=dict(self.config.recipe) | {
            "models": dict(self.config.recipe["models"]) | {"resume_from": path},
        })
        checkpoint = SimpleNamespace(path=path, metadata={
            "schema_version": 1, "recipe": self.config.recipe_dict(),
            "progress": asdict(progress), "wandb_run": self.logger.run_identity,
            "identity": training_loop.checkpoint_identity(
                config, data_fingerprint=self.samples.dataset.fingerprint,
                validation_manifest=bindings["validate_validation_bundle"].return_value,
            ),
        })
        self.samples[:] = self.samples[16:]  # The loader keeps the full dataset and starts at its cursor.
        read = stack.enter_context(patch.object(runner, "load_checkpoint_metadata", return_value=checkpoint))
        restore = stack.enter_context(patch.object(training_loop, "restore_training_state"))
        self.events.attach_mock(bindings["prepare_validation_cache"], "prepare_cache")
        self.events.attach_mock(restore, "restore")
        return config, checkpoint, bindings, read, restore

    def test_resume_restores_after_preparation_and_finishes_only_the_remaining_batch(self):
        with ExitStack() as stack:
            config, checkpoint, bindings, read, restore = self._resume(stack)
            progress = runner.run_training(config)
        read.assert_called_once_with(checkpoint.path, config)
        bindings["load_models"].assert_called_once_with(
            config, device=torch.device("cpu"), transformer_checkpoint=checkpoint.path / "transformer")
        bindings["build_dataloader"].assert_called_once_with(config, tokenizer=self.models.tokenizer, start_index=16)
        bindings["WandbLogger"].assert_called_once_with(
            config, device=torch.device("cpu"), resume_run=self.logger.run_identity)
        restore.assert_called_once()
        self.assertIs(restore.call_args.kwargs["optimizer"], self.optimizer)
        self.assertIs(restore.call_args.kwargs["flow_generator"], bindings["build_flow_generator"].return_value)
        self.assertEqual([event[0] for event in self.events.mock_calls],
                         ["prepare_cache", "restore", "begin", "step", "hardware", "validate",
                          "sample", "table", "checkpoint", "log"])
        self.step.assert_called_once_with(tuple(self.samples))
        self.assertEqual((progress.optimizer_step, progress.sources_seen, progress.targets_seen), (2, 26, 78))
        self.assertEqual(progress.workload_totals["train/flux_forward_calls"], 12)
        self.assertEqual(progress.workload_totals["val/nvarc_tokens"], 416)
        self.assertEqual(progress.workload_totals["sample/flux_forward_calls"], 480)
        metrics = self.logger.log.call_args.args[0]
        self.assertEqual(metrics["progress/optimizer_step"], 2)
        self.assertEqual(metrics["train/targets"], 30)
        self.assertEqual(metrics["train/flux_joint_tokens_total"], 85800)

    def test_resume_rejects_completed_epoch_changed_data_and_failed_restore_before_updates(self):
        for fault in ("complete", "data", "restore", "inventory"):
            with self.subTest(fault=fault), ExitStack() as stack:
                self.samples = _Samples(26)
                config, checkpoint, bindings, read, restore = self._resume(stack)
                if fault == "complete":
                    checkpoint.metadata["progress"].update(optimizer_step=2, sources_seen=26, targets_seen=78)
                elif fault == "data":
                    checkpoint.metadata["identity"]["data_sha256"] = "changed"
                elif fault == "restore":
                    restore.side_effect = ValueError("bad optimizer state")
                else:
                    read.side_effect = ValueError("incomplete checkpoint inventory")
                with self.assertRaises(ValueError):
                    runner.run_training(config)
                self.step.assert_not_called()
                self.logger.log.assert_not_called()
                if fault in ("complete", "inventory"):
                    bindings["load_models"].assert_not_called()
                if fault == "data":
                    bindings["build_optimizer"].assert_not_called()

    def _next_epoch(self, stack):
        bindings = self._setup(stack)
        parent = SimpleNamespace(path=Path("/test/parent/step-000002"), metadata={
            "schema_version": 1, "recipe": self.config.recipe_dict(),
            "progress": asdict(runner.RunProgress(2, 26, 78, {"train/flux_forward_calls": 12})),
            "wandb_run": self.logger.run_identity,
            "identity": training_loop.checkpoint_identity(
                self.config, data_fingerprint="parent-view",
                validation_manifest=bindings["validate_validation_bundle"].return_value),
        })
        payload = self.config.recipe_dict()
        payload["seed"] += 1
        payload["data"]["prepared_split_directory"] = "/test/next/data/splits/train"
        path = self.config.path.parent / "next.json"
        path.write_text(json.dumps(payload))
        config = load_training_config(path, expected_type="training_run")
        self.logger.run.step = 3
        restore = stack.enter_context(patch.object(training_loop, "restore_training_state"))
        return config, parent, bindings, restore

    def test_next_epoch_keeps_global_counters_and_resets_only_the_data_cursor(self):
        with ExitStack() as stack:
            config, parent, bindings, restore = self._next_epoch(stack)
            progress = runner.run_training(config, previous_epoch_checkpoint=parent)
        self.assertEqual((progress.epoch, progress.optimizer_step, progress.sources_seen, progress.targets_seen),
                         (2, 4, 52, 156))
        bindings["build_dataloader"].assert_called_once_with(config, tokenizer=self.models.tokenizer, start_index=0)
        bindings["load_models"].assert_called_once_with(
            config, device=torch.device("cpu"), transformer_checkpoint=parent.path / "transformer")
        bindings["WandbLogger"].assert_called_once_with(
            config, device=torch.device("cpu"), resume_run=parent.metadata["wandb_run"])
        self.assertIs(restore.call_args.args[0], parent)
        self.assertIs(restore.call_args.kwargs["optimizer"], self.optimizer)
        records = [call.args[0] for call in self.logger.log.call_args_list]
        self.assertEqual([row["progress/optimizer_step"] for row in records], [3, 4])
        self.assertEqual([row["train/targets"] for row in records], [48, 30])
        self.assertEqual(records[-1]["progress/epoch_fraction"], 2.)
        self.assertEqual(records[-1]["train/flux_forward_calls_total"], 24)
        self.evaluate.assert_called_once()
        self.generate.assert_called_once()
        self.save.assert_called_once()
        self.assertEqual(self.save.call_args.args[0].optimizer_step, 4)
        self.logger.run.config.update.assert_called_once()

    def test_next_epoch_rejects_changes_to_experiment_or_validation(self):
        for fault in ("loss", "validation", "seed", "incomplete"):
            with self.subTest(fault=fault), ExitStack() as stack:
                config, parent, bindings, restore = self._next_epoch(stack)
                if fault == "loss":
                    parent.metadata["identity"]["recipe"]["optimizer"]["learning_rate"] *= 2
                elif fault == "validation":
                    parent.metadata["identity"]["validation_sha256"] = "changed"
                elif fault == "seed":
                    parent.metadata["identity"]["recipe"]["seed"] = config.seed
                else:
                    parent.metadata["progress"]["optimizer_step"] = 1
                with self.assertRaises(ValueError):
                    runner.run_training(config, previous_epoch_checkpoint=parent)
                restore.assert_not_called()
                self.step.assert_not_called()

    def test_resume_inside_second_epoch_uses_local_data_cursor_and_global_step(self):
        with ExitStack() as stack:
            config, checkpoint, bindings, read, restore = self._resume(stack)
            checkpoint.metadata["progress"].update(epoch=2, optimizer_step=3, sources_seen=42, targets_seen=126)
            progress = runner.run_training(config)
        bindings["build_dataloader"].assert_called_once_with(config, tokenizer=self.models.tokenizer, start_index=16)
        self.assertEqual((progress.optimizer_step, progress.sources_seen, progress.targets_seen), (4, 52, 156))
        self.assertEqual(self.logger.log.call_args.args[0]["progress/optimizer_step"], 4)

    def test_checkpoint_failure_cannot_log_a_successful_save(self):
        with ExitStack() as stack:
            self._setup(stack)
            self.save.side_effect = OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                runner.run_training(self.config)
            self.assertEqual([call.args[0]["progress/optimizer_step"] for call in self.logger.log.call_args_list], [0, 1])
            self.assertIs(self.logger.__exit__.call_args.args[0], OSError)

    def test_logging_preflight_fails_before_model_loading(self):
        with ExitStack() as stack:
            bindings = self._setup(stack)
            bindings["WandbLogger"].side_effect = ValueError("set logging.project")
            with self.assertRaisesRegex(ValueError, "logging.project"):
                runner.run_training(self.config)
        bindings["load_models"].assert_not_called()

    def test_validation_counts_or_payload_failure_stop_before_models_or_training(self):
        for fault in ("task_count", "query_count", "sample_tasks", "sample_queries", "payload"):
            with self.subTest(fault=fault), ExitStack() as stack:
                bindings = self._setup(stack)
                if fault == "payload":
                    bindings["validate_validation_bundle"].side_effect = ValueError("changed payload")
                else:
                    manifest = bindings["validate_validation_bundle"].return_value
                    key = {"task_count": "validation_task_ids", "query_count": "validation_output_count",
                           "sample_tasks": "sampling_task_ids", "sample_queries": "sampling_output_count"}[fault]
                    manifest[key] = () if key.endswith("ids") else 1
                with self.assertRaises(ValueError):
                    runner.run_training(self.config)
                bindings["prepare_device"].assert_not_called()
                bindings["load_models"].assert_not_called()
                self.step.assert_not_called()

    def test_validation_failure_propagates_without_logging_an_incomplete_check(self):
        for failed_call in (1, 2):
            with self.subTest(failed_call=failed_call), ExitStack() as stack:
                self._setup(stack)
                self.evaluate.side_effect = ([deepcopy(self.validation_result)] * (failed_call - 1)
                                             + [FloatingPointError("validation failed")])
                with self.assertRaisesRegex(FloatingPointError, "validation failed"):
                    runner.run_training(self.config)
                self.assertEqual(self.step.call_count, 0 if failed_call == 1 else 2)
                self.assertEqual([call.args[0]["progress/optimizer_step"]
                                  for call in self.logger.log.call_args_list],
                                 [] if failed_call == 1 else [0, 1])
                self.assertIs(self.logger.__exit__.call_args.args[0], FloatingPointError)

    def test_sampling_cache_identity_mismatch_fails_before_any_check_or_update(self):
        with ExitStack() as stack:
            self._setup(stack)
            self.input_grids[("unexpected-task", 0)] = self.input_grids.pop(("val-0", 0))
            with self.assertRaisesRegex(ValueError, "query identities differ"):
                runner.run_training(self.config)
            self.evaluate.assert_not_called()
            self.generate.assert_not_called()
            self.step.assert_not_called()

    def test_sampling_or_table_failure_propagates_without_logging_an_incomplete_check(self):
        for phase in ("generate", "table"):
            for failed_call in (1, 2):
                with self.subTest(phase=phase, failed_call=failed_call), ExitStack() as stack:
                    self._setup(stack)
                    method = self.generate if phase == "generate" else self.logger.build_prediction_table
                    value = deepcopy(self.sampling_result) if phase == "generate" else object()
                    method.side_effect = [value] * (failed_call - 1) + [RuntimeError("sampling failed")]
                    with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                        runner.run_training(self.config)
                    self.assertEqual(self.step.call_count, 0 if failed_call == 1 else 2)
                    self.assertEqual([call.args[0]["progress/optimizer_step"]
                                      for call in self.logger.log.call_args_list],
                                     [] if failed_call == 1 else [0, 1])
                    self.assertIs(self.logger.__exit__.call_args.args[0], RuntimeError)

    def test_sampling_cadence_final_replacement_and_optional_initial_check(self):
        payload = json.loads(self.config.path.read_text())
        payload["data"]["expected_task_count"] = 75
        payload["validation"]["sampling"]["every_optimizer_steps"] = 2
        payload["checkpointing"]["every_optimizer_steps"] = 2
        for at_start in (True, False):
            with self.subTest(at_start=at_start), ExitStack() as stack:
                payload["validation"]["sampling"]["at_start"] = at_start
                payload["validation"]["loss"]["at_start"] = at_start
                self.config.path.write_text(json.dumps(payload))
                config = load_training_config(self.config.path)
                self.samples = _Samples(75)
                self._setup(stack)
                progress = runner.run_training(config)
                records = [call.args[0] for call in self.logger.log.call_args_list]
                sampled = [row for row in records if "sample/predictions" in row]
                self.assertEqual([row["progress/optimizer_step"] for row in sampled],
                                 [0, 2, 5] if at_start else [2, 5])
                self.assertEqual([row["progress/optimizer_step"] for row in records],
                                 list(range(0 if at_start else 1, 6)))
                self.assertEqual(progress.workload_totals["sample/flux_forward_calls"],
                                 720 if at_start else 480)
                self.assertEqual(progress.workload_totals["sample/nvarc_tokens"], 0)

    def test_periodic_validation_final_replacement_and_optional_initial_check(self):
        payload = json.loads(self.config.path.read_text())
        payload["data"]["expected_task_count"] = 75
        payload["validation"]["loss"]["every_optimizer_steps"] = 2
        payload["loss"]["region_weights"] = {"canvas": .2, "grid": .7, "boundary": .1}
        for at_start in (True, False):
            with self.subTest(at_start=at_start), ExitStack() as stack:
                payload["validation"]["loss"]["at_start"] = at_start
                self.config.path.write_text(json.dumps(payload))
                config = load_training_config(self.config.path)
                self.samples = _Samples(75)
                bindings = self._setup(stack)
                progress = runner.run_training(config)
                records = [call.args[0] for call in self.logger.log.call_args_list]
                validation = [row for row in records if "val/loss" in row]
                self.assertEqual([row["progress/optimizer_step"] for row in validation],
                                 [0, 2, 5] if at_start else [2, 5])
                self.assertEqual([row["val/nvarc_tokens"] for row in validation],
                                 [208, 0, 0] if at_start else [208, 0])
                self.assertEqual([row["progress/optimizer_step"] for row in records],
                                 list(range(6)))  # Sampling still runs at step zero.
                bindings["prepare_validation_cache"].assert_called_once()
                self.assertEqual(progress.targets_seen, 225)
                self.assertEqual(progress.workload_totals["val/nvarc_tokens"], 208)
                self.assertEqual(self.evaluate.call_args.kwargs["weights"], RegionLossWeights(.2, .7, .1))

    def test_real_validation_reduction_reaches_logging_without_changing_training_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        model.weight.grad = torch.full_like(model.weight, 2.0)
        targets = tuple(validation_module.ValidationTarget(
            task_id=f"task-{index}", query_index=0,
            text=torch.zeros((length, 7680), dtype=torch.bfloat16),
            clean_latent=torch.zeros((900, 128)), expected_grid=((0,) * side,) * side,
        ) for index, (length, side) in enumerate(((2, 2), (3, 30))))
        packs = validation_module.plan_validation_packs(targets, max_joint_tokens=1000)
        progress = runner.RunProgress(1, 16, 48)
        rng_before = torch.random.get_rng_state().clone()

        def predict(transformer, state, text, positions, times, layout, **kwargs):
            self.assertFalse(transformer.training)
            self.assertFalse(torch.is_grad_enabled())
            return state / times[:, None, None] + transformer.weight[0, 0]

        with patch.object(validation_module, "predict_packed_flux_velocity", side_effect=predict):
            metrics = runner._run_validation(
                self.config, progress, models=SimpleNamespace(
                    flux_transformer=model, flux_compute_dtype=torch.float32,
                    flux_packing_backend=object()),
                device=torch.device("cpu"), targets=targets, packs=packs, preparation_tokens=5,
            )
        self.assertAlmostEqual(metrics["val/loss"], 9.0, places=5)
        self.assertAlmostEqual(metrics["val/mse_boundary_t1"], 9.0, places=5)
        self.assertEqual(metrics["val/boundary_target_count"], 1)
        self.assertEqual(metrics["val/flux_forward_calls"], 12)
        self.assertEqual(metrics["val/flux_joint_tokens"], 10830)
        self.assertEqual(metrics["val/nvarc_tokens"], 5)
        self.assertEqual((progress.optimizer_step, progress.sources_seen, progress.targets_seen), (1, 16, 48))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, 3.0)))
        self.assertTrue(torch.equal(model.weight.grad, torch.full_like(model.weight, 2.0)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_baseline_config_is_rejected_before_device_or_model_work(self):
        config = replace(self.config, config_type="baseline", recipe=None)
        with patch.object(training_loop, "prepare_device") as prepare:
            with self.assertRaisesRegex(ValueError, "training_run config"):
                runner.run_training(config)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
