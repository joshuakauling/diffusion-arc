"""Check metric reductions and explicit W&B/hardware behavior without a network or GPU."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from diffusion_arc.arc_data.image.palette import TARGET_PALETTE_RGB
from diffusion_arc.training.run import logging
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.runner import RunProgress
from diffusion_arc.training.trainer import TargetMetrics, TrainStepResult


RECIPE = Path(__file__).resolve().parents[2] / "configs/training/arc_agi_1_epoch.json"


class _Run:
    def __init__(self):
        self.log, self.define_metric, self.watch, self.finish = Mock(), Mock(), Mock(), Mock()
        self.id, self.project, self.entity = "test-run", "unit-tests", "test-owner"
        self.step, self.resumed, self.summary = 0, False, {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish(exit_code=0 if exc_type is None else 1)
        return False


class LoggingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.payload = json.loads(RECIPE.read_text())
        self.payload["logging"].update(project="unit-tests", entity="test-owner")
        self.payload["deployment"]["output_directory"] = str(self.root / "output")
        self.config = self.load_config()
        self.result = TrainStepResult(
            loss=6.0, learning_rates=(1e-5,), gradient_norm=2.0, gradient_clipped=True,
            step_seconds=2.0, source_count=1, target_count=3,
            nvarc_tokens=100, nvarc_calls=1, vae_calls=1,
            flux_text_tokens=120, flux_image_tokens=2700,
            flux_joint_tokens=2820, flux_forward_calls=2,
            targets=(
                TargetMetrics(3, 1.0, 2, True, 2, 3, 4, 0.2, 2.4, 0.4),
                TargetMetrics(6, 0.4, 3, False, 6, 6, None, 2 / 3, 16 / 3, 0),
                TargetMetrics(9, 0.8, 4, True, 8, 10, 2, 0.8, 8, 0.2),
            ), diagnostics=Mock(),
        )
        self.progress = RunProgress(1, 1, 3, {
            f"train/{name}": getattr(self.result, name)
            for name in self.config.recipe["logging"]["workload_counters"]["names"]
        })

    def load_config(self):
        path = self.root / "run.json"
        path.write_text(json.dumps(self.payload))
        return load_training_config(path, expected_type="training_run")

    def test_target_means_and_contributions_keep_their_distinct_denominators(self):
        metrics = logging.build_training_metrics(self.result, self.progress, self.config)
        self.assertAlmostEqual(metrics["train/mse_canvas"], 16 / 3)
        self.assertAlmostEqual(metrics["train/mse_grid"], 19 / 3)
        self.assertEqual(metrics["train/mse_boundary"], 3)
        self.assertEqual(metrics["train/boundary_target_count"], 2)
        self.assertAlmostEqual(metrics["train/contribution_boundary"], 0.2)
        self.assertAlmostEqual(sum(metrics[f"train/contribution_{region}"]
                                   for region in ("canvas", "grid", "boundary")), metrics["train/loss"])
        self.assertEqual(metrics["train/loss_t1"], 3)
        self.assertEqual(metrics["train/loss_uniform"], 7.5)
        self.assertEqual(metrics["train/targets_uniform"], 2)
        for count, loss in ((2, 3), (3, 6), (4, 9)):
            self.assertEqual(metrics[f"train/loss_demos_{count}"], loss)
            self.assertEqual(metrics[f"train/targets_demos_{count}"], 1)
        self.assertEqual(metrics["train/targets_per_second"], 1.5)
        self.assertEqual(metrics["train/grad_clipped"], 1)
        self.assertEqual(metrics["train/learning_rate"], 1e-5)
        self.assertEqual(metrics["train/flux_joint_tokens_total"], 2820)

    def test_absent_boundary_and_empty_groups_have_no_invented_mean(self):
        result = replace(self.result, targets=(self.result.targets[1],), loss=6, target_count=1)
        metrics = logging.build_training_metrics(result, self.progress, self.config)
        for name in ("mse_boundary", "loss_t1", "loss_demos_2", "loss_demos_4"):
            self.assertNotIn(f"train/{name}", metrics)
        self.assertEqual(metrics["train/boundary_target_count"], 0)
        self.assertEqual(metrics["train/targets_t1"], 0)
        self.assertEqual(metrics["train/contribution_boundary"], 0)

    def test_raw_mse_result_omits_unavailable_regional_measurements(self):
        result = replace(self.result, targets=(TargetMetrics(6, 0.5, 2, True),), target_count=1)
        metrics = logging.build_training_metrics(result, self.progress, self.config)
        self.assertFalse(any(key.startswith(("train/mse_", "train/contribution_")) for key in metrics))

    def test_configured_subset_and_cpu_values_do_not_modify_results_or_progress(self):
        self.payload["logging"]["train_metrics"] = ["progress/optimizer_step", "train/loss"]
        self.payload["logging"]["workload_counters"]["names"] = ["flux_forward_calls"]
        metrics = logging.build_training_metrics(self.result, self.progress, self.load_config())
        self.assertEqual(metrics, {"progress/optimizer_step": 1, "train/loss": 6.0,
                                   "train/flux_forward_calls": 2, "train/flux_forward_calls_total": 2})
        self.assertEqual(self.progress.targets_seen, 3)
        self.assertTrue(all(isinstance(value, (float, int)) for value in metrics.values()))

    def test_validation_time_names_boundary_omission_and_configured_mixture(self):
        self.payload["flow"]["endpoint_targets_per_full_batch"] = 12
        self.payload["validation"]["loss"].update(uniform_proxy_weight=.75, endpoint_weight=.25)
        config = self.load_config()
        result = {
            "target_count": 3, "boundary_target_count": 0, "seconds": 4.5,
            "uniform_inner_loss": 2.0, "endpoint_loss": 10.0,
            "per_timestep": [dict(timestep=time, weighted_loss=10.0 if time == 1 else 2.0,
                                  canvas_loss=index + 1.0, grid_loss=index + 2.0,
                                  boundary_loss=None)
                             for index, time in enumerate((.1, .3, .5, .7, .9, 1.))],
            "nvarc_tokens": 0, "flux_text_tokens": 120,
            "flux_image_tokens": 16200, "flux_joint_tokens": 16320, "flux_forward_calls": 12,
        }
        progress = RunProgress(100, 1600, 4800, {
            f"val/{name}": result[name] + 100 for name in config.recipe["logging"]["workload_counters"]["names"]
        })
        metrics = logging.build_validation_metrics(result, progress, config)
        self.assertEqual(metrics["val/loss"], 4)
        self.assertEqual(metrics["val/loss_uniform_proxy"], 2)
        self.assertEqual(metrics["val/loss_t1"], 10)
        self.assertEqual(metrics["val/mse_canvas_t_0_1"], 1)
        self.assertEqual(metrics["val/mse_grid_t_0_9"], 6)
        self.assertEqual(metrics["val/boundary_target_count"], 0)
        self.assertEqual(metrics["val/seconds"], 4.5)
        self.assertEqual(metrics["val/flux_joint_tokens_total"], 16420)
        self.assertFalse(any(name.startswith("val/mse_boundary") for name in metrics))
        self.assertEqual(metrics["progress/optimizer_step"], 100)
        self.assertEqual(progress.targets_seen, 4800)
        self.assertEqual(result["flux_joint_tokens"], 16320)

        self.payload["logging"]["validation_metrics"] = ["val/seconds"]
        self.payload["logging"]["validation_time_metrics"]["names"] = ["mse_boundary"]
        self.payload["logging"]["workload_counters"]["names"] = []
        self.assertEqual(logging.build_validation_metrics(result, progress, self.load_config()),
                         {"progress/optimizer_step": 100, "val/seconds": 4.5})

    def fake_runtime(self, stack):
        self.sdk = SimpleNamespace(Settings=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)))
        self.sdk.Image = Mock(side_effect=lambda image, **kw: SimpleNamespace(image=image.copy(), **kw))
        self.sdk.Table = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
        self.run = _Run()
        def initialize(**kwargs):
            self.run.settings = kwargs["settings"]
            return self.run
        self.sdk.init = Mock(side_effect=initialize)
        self.nvml = SimpleNamespace(
            nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value="physical-gpu-handle"),
            nvmlDeviceGetUtilizationRates=Mock(return_value=SimpleNamespace(gpu=87)),
            nvmlDeviceGetPowerUsage=Mock(return_value=320000),
            nvmlDeviceGetTemperature=Mock(return_value=63), NVML_TEMPERATURE_GPU=0,
        )
        self.process = Mock()
        self.process.cpu_percent.return_value = 230.0
        self.process.memory_info.return_value = SimpleNamespace(rss=2048)
        self.psutil = SimpleNamespace(Process=Mock(return_value=self.process))
        stack.enter_context(patch.dict("sys.modules", {"wandb": self.sdk, "pynvml": self.nvml,
                                                      "psutil": self.psutil}))
        stack.enter_context(patch.dict("os.environ", {"WANDB_API_KEY": "test-key-do-not-upload"}))
        self.clock = stack.enter_context(patch.object(logging.time, "monotonic", return_value=100.0))
        self.cuda = {}
        for name, value in {
            "get_device_properties": SimpleNamespace(uuid="GPU-selected-by-cuda"),
            "reset_peak_memory_stats": None, "memory_allocated": 100,
            "memory_reserved": 200, "max_memory_allocated": 300,
        }.items():
            self.cuda[name] = stack.enter_context(patch.object(torch.cuda, name, return_value=value))

    def test_sampling_metrics_aliases_counts_and_missing_cell_accuracy(self):
        result = {
            "pass_at_1": .1, "pass_at_2": .2, "solved_at_1": 1, "solved_at_2": 2,
            "task_count": 10, "prediction_count": 20, "shape_correct_count": 0,
            "shape_accuracy": 0., "cell_accuracy": None, "invalid_grid_rate": .5, "seconds": 3.5,
            "nvarc_tokens": 0, "flux_text_tokens": 1600, "flux_image_tokens": 720000,
            "flux_joint_tokens": 721600, "flux_forward_calls": 160,
        }
        progress = RunProgress(300, 4800, 14400, {
            f"sample/{name}": 2 * result[name]
            for name in self.config.recipe["logging"]["workload_counters"]["names"]
        })
        metrics = logging.build_sampling_metrics(result, progress, self.config)
        self.assertEqual(metrics["sample/solved_tasks_at_1"], 1)
        self.assertEqual(metrics["sample/solved_tasks_at_2"], 2)
        self.assertEqual(metrics["sample/prediction_count"], 20)
        self.assertEqual(metrics["sample/shape_correct_count"], 0)
        self.assertNotIn("sample/cell_accuracy", metrics)
        self.assertEqual(metrics["sample/flux_forward_calls_total"], 320)
        self.assertEqual(metrics["progress/optimizer_step"], 300)
        result["cell_accuracy"] = .75
        self.assertEqual(logging.build_sampling_metrics(result, progress, self.config)["sample/cell_accuracy"], .75)
        self.assertEqual(progress.targets_seen, 14400)
        self.payload["logging"]["sampling_metrics"] = ["sample/pass_at_2"]
        self.payload["logging"]["workload_counters"]["names"] = []
        self.assertEqual(logging.build_sampling_metrics(result, progress, self.load_config()),
                         {"sample/pass_at_2": .2, "progress/optimizer_step": 300})

    def test_prediction_tables_use_query_identity_and_preserve_full_invalid_canvas(self):
        valid = torch.full((30, 30), 10)
        valid[0, 0] = 1
        invalid = valid.clone()
        invalid[-1, -1] = 2  # This must remain visible, beyond the expected 1x1 grid.
        samples = [dict(class_canvas=canvas.tolist(), predicted_shape=shape,
                        prediction_valid=ok, exact_grid=ok)
                   for canvas, shape, ok in ((valid, [1, 1], True), (invalid, None, False))]
        result = {"queries": [dict(task_id="task", query_index=index, samples=samples) for index in (1, 0)]}
        targets = [SimpleNamespace(task_id="task", query_index=index, expected_grid=((index + 3,),))
                   for index in (0, 1)]
        inputs = {("task", 0): ((5,),), ("task", 1): ((6,),)}
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with logging.WandbLogger(self.config, device=torch.device("cuda:0")) as logger:
                first = logger.build_prediction_table(result, targets=targets, input_grids=inputs)
                second = logger.build_prediction_table(result, targets=targets, input_grids=inputs)
                self.assertIsNot(first, second)
                self.assertEqual(first.columns, ["task_id", "query_index", "comparison",
                                                 "sample_1_exact", "sample_2_exact"])
                self.assertEqual(first.log_mode, "IMMUTABLE")
                self.assertEqual([row[:2] for row in first.data], [["task", 1], ["task", 0]])
                for query_index, row in zip((1, 0), first.data):
                    self.assertEqual(row[3:], [True, False])
                    self.assertEqual(row[2].caption, f"task / query {query_index}")
                    image = row[2].image
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (1020, 306))
                    # Same cell scale in all panels, regardless of query identity or prediction validity.
                    for panel, color in enumerate((query_index + 5, query_index + 3, 1, 1)):
                        self.assertEqual(image.getpixel((12 + panel * 252 + 4, 58)),
                                         tuple(TARGET_PALETTE_RGB[color].tolist()))
                    self.assertEqual(image.getpixel((12 + 3 * 252 + 236, 290)),
                                     tuple(TARGET_PALETTE_RGB[2].tolist()))
                self.run.log.assert_not_called()
                logger.log({"progress/optimizer_step": 300, "sample/predictions": first})
                self.run.log.assert_called_once_with(
                    {"progress/optimizer_step": 300, "sample/predictions": first}, step=300, commit=True,
                )

    def test_explicit_sdk_settings_config_serialization_and_one_combined_log(self):
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with logging.WandbLogger(self.config, device=torch.device("cuda:1")) as logger:
                settings = self.sdk.init.call_args.kwargs
                self.assertEqual(settings["project"], "unit-tests")
                self.assertEqual(settings["entity"], "test-owner")
                self.assertEqual(settings["mode"], "online")
                self.assertEqual(settings["dir"], str(self.root / "output"))
                serialized = json.dumps(settings["config"])
                self.assertNotIn("test-key-do-not-upload", serialized)
                self.assertIn(str(self.config.models.flux_checkpoint), serialized)
                self.assertTrue(settings["settings"].x_disable_stats)
                self.assertTrue(settings["settings"].x_disable_machine_info)
                self.assertEqual(settings["settings"].console, "off")
                self.assertFalse(settings["save_code"])
                self.assertFalse(settings["sync_tensorboard"])
                logger.begin_step()
                metrics = logging.build_training_metrics(self.result, self.progress, self.config)
                metrics.update(logger.collect_hardware_metrics(phase="train"))
                logger.log(metrics)
                self.run.log.assert_called_once_with(metrics, step=1, commit=True)
                self.run.watch.assert_not_called()
            self.run.finish.assert_called_once_with(exit_code=0)
            self.nvml.nvmlShutdown.assert_called_once()

    def test_resume_keeps_wandb_identity_and_skips_existing_history_without_rewinding(self):
        previous = {"id": "test-run", "project": "unit-tests", "entity": "test-owner"}
        with ExitStack() as stack:
            self.fake_runtime(stack)
            self.run.step, self.run.resumed = 351, True
            self.run.summary = {"system/elapsed_seconds": 1000}
            with logging.WandbLogger(self.config, device=torch.device("cuda:0"), resume_run=previous) as logger:
                supplied = self.sdk.init.call_args.kwargs
                self.assertEqual(supplied["id"], "test-run")
                self.assertEqual(supplied["resume"], "must")
                self.assertEqual(supplied["entity"], "test-owner")
                self.assertIsNone(supplied["config"])
                self.assertNotIn("resume_from", supplied)
                self.assertEqual(logger.run_identity, previous)
                for step in (301, 350, 351, 352):
                    logger.log({"progress/optimizer_step": step, "train/loss": 1.0})
                self.assertEqual([call.kwargs["step"] for call in self.run.log.call_args_list], [351, 352])
                self.clock.return_value = 132
                self.assertEqual(logger.collect_hardware_metrics(phase="train")["system/elapsed_seconds"], 1032)

    def test_resume_requires_same_project_entity_and_an_actual_resumed_run(self):
        previous = {"id": "test-run", "project": "unit-tests", "entity": "test-owner"}
        for name in ("project", "entity"):
            with self.subTest(name=name), patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
                with self.assertRaisesRegex(ValueError, "project/entity"):
                    logging.WandbLogger(self.config, device=torch.device("cuda:0"),
                                        resume_run=previous | {name: "different"})
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with self.assertRaisesRegex(RuntimeError, "existing run"):
                with logging.WandbLogger(self.config, device=torch.device("cuda:0"), resume_run=previous):
                    self.fail("a fresh run cannot substitute for resume")
            self.run.finish.assert_called_once_with(exit_code=1)

    def test_hardware_interval_units_device_identity_and_peak_boundaries(self):
        with ExitStack() as stack:
            self.fake_runtime(stack)
            device = torch.device("cuda:1")
            with logging.WandbLogger(self.config, device=device) as logger:
                self.nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-selected-by-cuda")
                self.process.cpu_percent.assert_called_once_with(interval=None)
                for read in (self.nvml.nvmlDeviceGetUtilizationRates, self.nvml.nvmlDeviceGetPowerUsage,
                             self.nvml.nvmlDeviceGetTemperature):
                    read.reset_mock()  # Discard the one-time sensor preflight.
                logger.begin_step()
                self.cuda["reset_peak_memory_stats"].assert_called_once_with(device)
                for instant in (110, 129.9):
                    self.clock.return_value = instant
                    metrics = logger.collect_hardware_metrics(phase="train")
                    self.assertNotIn("system/gpu_utilization_percent", metrics)
                self.nvml.nvmlDeviceGetUtilizationRates.assert_not_called()
                self.clock.return_value = 130
                metrics = logger.collect_hardware_metrics(phase="train")
                self.assertEqual(metrics, {
                    "system/gpu_allocated_bytes": 100, "system/gpu_reserved_bytes": 200,
                    "system/gpu_peak_allocated_bytes": 300, "system/gpu_utilization_percent": 87,
                    "system/gpu_power_watts": 320, "system/gpu_temperature_celsius": 63,
                    "system/cpu_utilization_percent": 230, "system/ram_rss_bytes": 2048,
                    "system/elapsed_seconds": 30, "system/phase": "train",
                })
                self.cuda["max_memory_allocated"].assert_called_with(device)
                self.clock.return_value = 131
                self.cuda["max_memory_allocated"].return_value = 999
                self.assertEqual(metrics["system/gpu_peak_allocated_bytes"], 300)
                self.assertNotIn("system/ram_rss_bytes", logger.collect_hardware_metrics(phase="train"))
                self.nvml.nvmlDeviceGetPowerUsage.assert_called_once_with("physical-gpu-handle")
                self.run.log.assert_not_called()

    def test_disabled_sensors_are_never_initialized_or_polled(self):
        self.payload["logging"]["hardware"].update(per_step_metrics=[], sampled_metrics=["system/ram_rss_bytes"])
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with logging.WandbLogger(self.load_config(), device=torch.device("cuda:0")) as logger:
                logger.begin_step()
                self.clock.return_value = 131
                metrics = logger.collect_hardware_metrics(phase="train")
                self.assertEqual(set(metrics), {"system/ram_rss_bytes", "system/elapsed_seconds", "system/phase"})
            self.nvml.nvmlInit.assert_not_called()
            self.process.cpu_percent.assert_not_called()
            for mock in self.cuda.values():
                mock.assert_not_called()

    def test_training_error_finishes_run_as_failed_without_suppressing_it(self):
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with self.assertRaisesRegex(FloatingPointError, "step failed"):
                with logging.WandbLogger(self.config, device=torch.device("cuda:0")):
                    raise FloatingPointError("step failed")
            self.run.finish.assert_called_once_with(exit_code=1)
            self.nvml.nvmlShutdown.assert_called_once()

    def test_unsupported_requested_sensor_fails_before_wandb_init_and_releases_nvml(self):
        with ExitStack() as stack:
            self.fake_runtime(stack)
            self.nvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("unsupported power sensor")
            with self.assertRaisesRegex(RuntimeError, "unsupported power sensor"):
                with logging.WandbLogger(self.config, device=torch.device("cuda:0")):
                    self.fail("preflight should fail")
            self.sdk.init.assert_not_called()
            self.nvml.nvmlShutdown.assert_called_once()

    def test_sdk_policy_violation_fails_and_unconfigured_metrics_are_rejected(self):
        with ExitStack() as stack:
            self.fake_runtime(stack)
            self.sdk.Settings.side_effect = lambda **kwargs: SimpleNamespace(**(kwargs | {"x_disable_stats": False}))
            with self.assertRaisesRegex(RuntimeError, "automatic system metrics"):
                with logging.WandbLogger(self.config, device=torch.device("cuda:0")):
                    self.fail("policy check should fail")
            self.run.finish.assert_called_once_with(exit_code=1)
        with ExitStack() as stack:
            self.fake_runtime(stack)
            with logging.WandbLogger(self.config, device=torch.device("cuda:0")) as logger:
                with self.assertRaisesRegex(ValueError, "unconfigured"):
                    logger.log({"progress/optimizer_step": 1, "train/loss_task_1": 3})
                self.run.log.assert_not_called()

    def test_missing_deployment_settings_fail_without_creating_output(self):
        for missing in ("project", "output_directory", "key"):
            with self.subTest(missing=missing), patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}):
                payload = json.loads(json.dumps(self.payload))
                if missing == "project":
                    payload["logging"]["project"] = None
                elif missing == "output_directory":
                    payload["deployment"]["output_directory"] = None
                else:
                    import os
                    os.environ.pop("WANDB_API_KEY")
                config = replace(self.config, recipe=payload)
                with self.assertRaisesRegex(ValueError, "before training"):
                    logging.WandbLogger(config, device=torch.device("cuda:0"))
                self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
