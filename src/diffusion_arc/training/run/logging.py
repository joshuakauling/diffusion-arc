"""Reduce run results, render predictions and send selected measurements to W&B."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import os
from statistics import fmean
import time
from typing import TYPE_CHECKING, Any

import torch

from diffusion_arc.arc_data.image.palette import PAD_CLASS, TARGET_PALETTE_RGB

from ..config import TrainingConfig
from ..trainer import TrainStepResult

if TYPE_CHECKING:
    from ..runner import RunProgress


def build_training_metrics(
    result: TrainStepResult, progress: RunProgress, config: TrainingConfig,
) -> dict[str, float | int]:
    """Build this update's scalars from CPU results and already-advanced progress.

    Regions and subgroups give each target equal weight. Missing means are
    omitted, while empty-group counts remain zero. Boundary contributions include
    boundaryless targets; their raw boundary errors do not. The current optimizer
    has one parameter group, so learning_rate is its rate used for this update.
    """
    logging = config.recipe["logging"]
    metrics = {
        "train/loss": result.loss,
        "train/learning_rate": result.learning_rates[0],
        "train/grad_norm": result.gradient_norm,
        "train/grad_clipped": int(result.gradient_clipped),
        "train/step_seconds": result.step_seconds,
        "train/targets_per_second": result.target_count / result.step_seconds,
        "train/targets": result.target_count,
        "train/boundary_target_count": sum(target.has_boundary for target in result.targets),
        "progress/optimizer_step": progress.optimizer_step,
        "progress/epoch_fraction": progress.targets_seen / config.recipe["loop"]["expected_target_count"],
        "progress/targets_seen": progress.targets_seen,
    }
    for region in ("canvas", "grid", "boundary"):
        for kind in ("mse", "contribution"):
            field = f"{kind}_{region}"
            values = [getattr(target, field) for target in result.targets
                      if getattr(target, field) is not None
                      and (field != "mse_boundary" or target.has_boundary)]
            if values:
                metrics[f"train/{field}"] = fmean(values)

    groups = {
        "t1": [target.loss for target in result.targets if target.timestep == 1.0],
        "uniform": [target.loss for target in result.targets if target.timestep != 1.0],
        **{f"demos_{count}": [target.loss for target in result.targets
                             if target.context_pair_count == count] for count in (2, 3, 4)},
    }
    if "high_noise_min" in config.recipe["flow"]:
        threshold = config.recipe["flow"]["high_noise_min"]
        groups.pop("uniform")
        groups["high_noise"] = [target.loss for target in result.targets if threshold < target.timestep < 1]
        groups["lower_noise"] = [target.loss for target in result.targets if target.timestep <= threshold]
    for name, losses in groups.items():
        metrics[f"train/targets_{name}"] = len(losses)
        if losses:
            metrics[f"train/loss_{name}"] = fmean(losses)
    metrics = {name: metrics[name] for name in logging["train_metrics"] if name in metrics}
    for name in logging["workload_counters"]["names"]:
        key = f"train/{name}"
        metrics[key] = getattr(result, name)
        metrics[key + logging["workload_counters"]["cumulative_suffix"]] = progress.workload_totals[key]
    return metrics


def build_validation_metrics(
    result: Mapping[str, Any], progress: RunProgress, config: TrainingConfig,
) -> dict[str, float | int]:
    """Map one fixed-time evaluation to the configured W&B names.

    The five inner probes approximate the uniform-time loss; the combined loss
    uses the fixed evaluation mixture, even if training times change. Regional means have their
    correct target denominators. Missing boundary means stay absent.
    """
    logging = config.recipe["logging"]
    mixture = config.recipe["validation"]["loss"]
    metrics = {
        "val/loss": (mixture["uniform_proxy_weight"] * result["uniform_inner_loss"]
                     + mixture["endpoint_weight"] * result["endpoint_loss"]),
        "val/loss_uniform_proxy": result["uniform_inner_loss"],
        "val/target_count": result["target_count"],
        "val/boundary_target_count": result["boundary_target_count"],
        "val/seconds": result["seconds"],
    }
    metrics = {name: metrics[name] for name in logging["validation_metrics"]}
    fields = {"loss": "weighted_loss", "mse_canvas": "canvas_loss",
              "mse_grid": "grid_loss", "mse_boundary": "boundary_loss"}
    times = logging["validation_time_metrics"]
    for row, suffix in zip(result["per_timestep"], times["suffixes"], strict=True):
        for name in times["names"]:
            if row[fields[name]] is not None:
                metrics[f"val/{name}_{suffix}"] = row[fields[name]]
    for name in logging["workload_counters"]["names"]:
        key = f"val/{name}"
        metrics[key] = result[name]
        metrics[key + logging["workload_counters"]["cumulative_suffix"]] = progress.workload_totals[key]
    metrics[logging["step_metric"]] = progress.optimizer_step
    return metrics


def build_sampling_metrics(
    result: Mapping[str, Any], progress: RunProgress, config: TrainingConfig,
) -> dict[str, float | int]:
    """Map sampling scores to fixed names; omit cell accuracy when no shapes match."""
    logging = config.recipe["logging"]
    aliases = {"solved_tasks_at_1": "solved_at_1", "solved_tasks_at_2": "solved_at_2"}
    metrics = {}
    for name in logging["sampling_metrics"]:
        field = name.removeprefix("sample/")
        value = result[aliases.get(field, field)]
        if value is not None:
            metrics[name] = value
    for name in logging["workload_counters"]["names"]:
        key = f"sample/{name}"
        metrics[key] = result[name]
        metrics[key + logging["workload_counters"]["cumulative_suffix"]] = progress.workload_totals[key]
    metrics[logging["step_metric"]] = progress.optimizer_step
    return metrics


def _render_sampling_comparison(input_grid, expected_grid, samples):
    """Draw input, reference and both full classified canvases at equal cell scale.

    Predictions come from the completed VAE decode. They are never cropped to
    the reference shape, so invalid padding and stray pixels stay visible.
    """
    from PIL import Image, ImageDraw, ImageFont

    panels = []
    for label, grid in (("input", input_grid), ("expected_output", expected_grid)):
        canvas = torch.full((30, 30), PAD_CLASS, dtype=torch.long)
        canvas[:len(grid), :len(grid[0])] = torch.tensor(grid)
        panels.append((label, f"{len(grid)} x {len(grid[0])}", canvas))
    for index, sample in enumerate(samples, start=1):
        shape = sample["predicted_shape"]
        if sample["prediction_valid"]:
            status = "exact" if sample["exact_grid"] else "incorrect"
            detail = f"{shape[0]} x {shape[1]} | {status}"
        else:
            detail = "invalid grid"
        panels.append((f"sample_{index}", detail, torch.tensor(sample["class_canvas"])))

    panel_size, margin, header = 240, 12, 54
    image = Image.new("RGB", (4 * panel_size + 5 * margin, header + panel_size + margin), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    title_font, detail_font = ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    for index, (title, detail, canvas) in enumerate(panels):
        x = margin + index * (panel_size + margin)
        draw.text((x, 10), title, fill="#111827", font=title_font)
        draw.text((x, 30), detail, fill="#4b5563", font=detail_font)
        pixels = bytes(TARGET_PALETTE_RGB[canvas].flatten().tolist())
        panel = Image.frombytes("RGB", (30, 30), pixels).resize(
            (panel_size, panel_size), resample=Image.Resampling.NEAREST,
        )
        image.paste(panel, (x, header))
        draw.rectangle((x - 1, header - 1, x + panel_size, header + panel_size), outline="#9ca3af")
    return image


def _metric_names(logging: Mapping[str, Any]) -> set[str]:
    """Expand the recipe's finite metric names, including future run services."""
    names = {name for group in ("train_metrics", "validation_metrics", "sampling_metrics",
                                "checkpoint_metrics") for name in logging[group]}
    names.update(f"val/{name}_{suffix}" for name in logging["validation_time_metrics"]["names"]
                 for suffix in logging["validation_time_metrics"]["suffixes"])
    workload = logging["workload_counters"]
    names.update(f"{phase}/{name}{suffix}" for phase in workload["phases"]
                 for name in workload["names"] for suffix in ("", workload["cumulative_suffix"]))
    names.update(logging["hardware"]["per_step_metrics"])
    names.update(logging["hardware"]["sampled_metrics"])
    names.update(("system/elapsed_seconds", "system/phase", logging["predictions"]["key"]))
    return names


class WandbLogger:
    """Own one W&B run and bounded hardware reads for the selected CUDA device.

    Use as a context manager. Call begin_step before training, collect_hardware_metrics
    immediately afterwards, and log once after adding other run-service results.
    No background hardware polling or automatic model/console capture is enabled.
    W&B's context manager finishes the run as successful or failed on exit.
    """

    def __init__(self, config: TrainingConfig, *, device: torch.device, resume_run: dict | None = None) -> None:
        """Check deployment settings before loading models or opening a W&B run."""
        if config.recipe is None:
            raise ValueError("WandbLogger requires a training_run config")
        self.config, self.device = config, device
        self.options = config.recipe["logging"]
        self.resume_run = resume_run
        if resume_run is not None and (
                self.options["project"] != resume_run["project"]
                or self.options["entity"] not in (None, resume_run["entity"])):
            raise ValueError("W&B project/entity must match the checkpoint run")
        self.output_directory = config.recipe["deployment"]["output_directory"]
        if self.options["project"] is None or self.output_directory is None:
            raise ValueError("set logging.project and deployment.output_directory before training")
        if not os.environ.get(self.options["api_key_environment_variable"]):
            raise ValueError("set WANDB_API_KEY in the environment before training")
        self.allowed_metrics = _metric_names(self.options)

    def __enter__(self) -> WandbLogger:
        """Check hardware support, then start the explicitly configured SDK run."""
        import wandb

        with ExitStack() as resources:
            self._sample_readers = self._hardware_readers(resources)
            self.output_directory.mkdir(parents=True, exist_ok=True)
            resume_arguments = {"id": self.resume_run["id"], "resume": "must"} if self.resume_run else {}
            self.run = resources.enter_context(wandb.init(
                entity=self.resume_run["entity"] if self.resume_run else self.options["entity"],
                project=self.options["project"],
                name=self.config.name, mode=self.options["mode"],
                dir=str(self.output_directory),
                config=None if self.resume_run else self.config.recipe_dict(),
                save_code=False, sync_tensorboard=False,
                **resume_arguments,
                settings=wandb.Settings(
                    api_key=os.environ[self.options["api_key_environment_variable"]],
                    x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
                    console="off", disable_code=True, disable_git=True, quiet=True,
                ),
            ))
            if not self.run.settings.x_disable_stats or self.run.settings.console != "off":
                raise RuntimeError("W&B must disable automatic system metrics and console capture")
            self.run.define_metric(self.options["step_metric"])
            self.run.define_metric("*", step_metric=self.options["step_metric"], step_sync=False)
            self.run.define_metric("system/elapsed_seconds")
            self.run.define_metric("system/*", step_metric="system/elapsed_seconds", step_sync=False)
            self._started_at = self._last_sample_at = time.monotonic()
            self._next_log_step = self.run.step if self.resume_run else 0
            self._elapsed_offset = float(self.run.summary.get("system/elapsed_seconds", 0)) if self.resume_run else 0.
            if self.resume_run and (self.run.id != self.resume_run["id"] or not self.run.resumed):
                raise RuntimeError("W&B must resume the checkpoint's existing run")
            self._resources = resources.pop_all()
        return self

    def _hardware_readers(self, resources: ExitStack) -> dict:
        """Bind only requested sensors; unsupported requested measurements fail preflight."""
        requested = self.options["hardware"]["sampled_metrics"]
        readers = {}
        if any(name.startswith("system/gpu_") for name in requested):
            import pynvml

            pynvml.nvmlInit()
            resources.callback(pynvml.nvmlShutdown)
            # UUID respects CUDA_VISIBLE_DEVICES; logical CUDA and NVML indices can differ.
            # PyTorch's CUuuid string lacks the prefix expected by NVML.
            uuid = str(torch.cuda.get_device_properties(self.device).uuid)
            if not uuid.startswith(("GPU-", "MIG-")):
                uuid = "GPU-" + uuid
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            gpu_readers = {
                "system/gpu_utilization_percent": lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "system/gpu_power_watts": lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                "system/gpu_temperature_celsius": lambda: pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU),
            }
            for name in requested:
                if name in gpu_readers:
                    gpu_readers[name]()
                    readers[name] = gpu_readers[name]
        if "system/cpu_utilization_percent" in requested or "system/ram_rss_bytes" in requested:
            import psutil

            process = psutil.Process()
            if "system/cpu_utilization_percent" in requested:
                process.cpu_percent(interval=None)  # Prime the nonblocking, per-core percentage.
                readers["system/cpu_utilization_percent"] = lambda: process.cpu_percent(interval=None)
            if "system/ram_rss_bytes" in requested:
                readers["system/ram_rss_bytes"] = lambda: process.memory_info().rss
        return readers

    def begin_step(self) -> None:
        """Reset the training memory peak immediately before Trainer.step."""
        if "system/gpu_peak_allocated_bytes" in self.options["hardware"]["per_step_metrics"]:
            torch.cuda.reset_peak_memory_stats(self.device)

    def collect_hardware_metrics(self, *, phase: str) -> dict[str, float | int | str]:
        """Read training memory and any due sensors, without transmitting metrics.

        Samples occur at step boundaries, at least sample_interval_seconds apart;
        they are not averages over that interval. CPU utilization is the process
        average since its previous read, with 100% per occupied core, and may span
        phases. GPU utilization is the device's recent NVML sample. No extra CUDA
        synchronization is needed after the already synchronized Trainer.step.
        """
        hardware = self.options["hardware"]
        memory_readers = {
            "system/gpu_allocated_bytes": torch.cuda.memory_allocated,
            "system/gpu_reserved_bytes": torch.cuda.memory_reserved,
            "system/gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated,
        }
        metrics = {name: memory_readers[name](self.device) for name in hardware["per_step_metrics"]}
        now = time.monotonic()
        if self._sample_readers and now - self._last_sample_at >= hardware["sample_interval_seconds"]:
            metrics.update((name, read()) for name, read in self._sample_readers.items())
            self._last_sample_at = now
        if metrics:
            metrics.update({"system/elapsed_seconds": self._elapsed_offset + now - self._started_at,
                            "system/phase": phase})
        return metrics

    def log(self, metrics: Mapping[str, Any]) -> None:
        """Send one update; replayed steps already present in W&B retain their old records."""
        unexpected = metrics.keys() - self.allowed_metrics
        if unexpected:
            raise ValueError(f"unconfigured W&B metrics: {sorted(unexpected)}")
        step = metrics[self.options["step_metric"]]
        if step >= self._next_log_step:
            self.run.log(dict(metrics), step=step, commit=True)
            self._next_log_step = step + 1

    @property
    def run_identity(self) -> dict[str, str]:
        """Persist the resolved W&B identity, including an implicitly selected entity."""
        return {"id": self.run.id, "project": self.run.project, "entity": self.run.entity}

    def build_prediction_table(self, result: Mapping[str, Any], *, targets, input_grids):
        """Create a fresh table for this check; the runner includes it in its one log call."""
        import wandb

        expected = {(target.task_id, target.query_index): target.expected_grid for target in targets}
        rows = []
        for query in result["queries"]:
            key = (query["task_id"], query["query_index"])
            samples = query["samples"]
            comparison = _render_sampling_comparison(input_grids[key], expected[key], samples)
            rows.append([
                *key, wandb.Image(comparison, caption=f"{key[0]} / query {key[1]}", file_type="png"),
                samples[0]["exact_grid"], samples[1]["exact_grid"],
            ])
        return wandb.Table(columns=list(self.options["predictions"]["columns"]),
                           data=rows, log_mode="IMMUTABLE")

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Finish W&B and release NVML, propagating training or logging failures."""
        return self._resources.__exit__(exc_type, exc_value, traceback)


__all__ = ["build_training_metrics", "build_validation_metrics", "build_sampling_metrics", "WandbLogger"]
