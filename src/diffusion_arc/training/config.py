"""Canonical training configuration for baseline benchmarks and finite run recipes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from diffusion_arc._constants import DEFAULT_NVARC_MAX_TOKENS
from .step.flux_batching import FluxPackingPolicy


BASELINE_SCHEMA_VERSION = 10
_SUPPORTED_DTYPES = frozenset({"bfloat16", "float32"})


@dataclass(frozen=True)
class ModelConfig:
    """Local checkpoint paths and explicit model precision choices."""

    nvarc_checkpoint: Path
    flux_checkpoint: Path
    nvarc_dtype: str
    vae_dtype: str
    flux_parameter_dtype: str
    flux_compute_dtype: str


@dataclass(frozen=True)
class DataConfig:
    """One finite stream of individual ARC training sequences."""

    split_directory: Path
    view_name: str
    shuffle: bool
    num_workers: int
    prefetch_factor: int | None = None


@dataclass(frozen=True)
class LoopConfig:
    """Outer-loop target batch size, step limit, and target selection."""

    max_optimizer_steps: int
    optimizer_batch_size: int
    min_context_pairs_before_target: int
    vae_microbatch_size: int | None = None


@dataclass(frozen=True)
class AdamWConfig:
    """Explicit AdamW settings for the trainable FLUX transformer."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float | None = None


@dataclass(frozen=True)
class CosineSchedulerConfig:
    """Linear LR warm-up followed by cosine decay to zero."""

    warmup_steps: int
    total_steps: int


@dataclass(frozen=True)
class LinearWarmupConstantConfig:
    """Linear LR warmup followed by the unchanged peak learning rate."""

    warmup_steps: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Timing windows, independent of the learning-rate schedule."""

    warmup_steps: int
    measured_steps: int
    repetitions: int


@dataclass(frozen=True)
class TrainingConfig:
    """Shared runtime components and an optional immutable finite-epoch recipe.

    Both JSON formats normalize to this type. Entrypoints select their supported
    config_type when loading so benchmark runners cannot ignore recipe semantics.
    Loading never requires local checkpoints, prepared data, or credentials.
    """

    seed: int
    device: str
    models: ModelConfig
    data: DataConfig
    loop: LoopConfig
    optimizer: AdamWConfig
    flux_packing: FluxPackingPolicy
    nvarc_max_tokens: int = DEFAULT_NVARC_MAX_TOKENS
    scheduler: CosineSchedulerConfig | LinearWarmupConstantConfig | None = None
    benchmark: BenchmarkConfig | None = None
    config_type: str = "baseline"
    recipe: Mapping[str, Any] | None = None
    path: Path | None = field(default=None, compare=False)
    name: str | None = field(default=None, compare=False)

    def _require_recipe(self) -> Mapping[str, Any]:
        if self.recipe is None:
            raise ValueError("this property requires a training_run config")
        return self.recipe

    def recipe_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible recipe copy with resolved paths as strings.

        Logging and checkpoints use the same representation. Mutating the copy
        does not change the validated, immutable configuration.
        """
        return _config_values(self._require_recipe())

    @property
    def targets_per_sequence(self) -> int:
        return self._require_recipe()["data"]["pairs_per_sequence"] - self.loop.min_context_pairs_before_target

    @property
    def final_batch_target_count(self) -> int:
        return self._require_recipe()["loop"]["expected_final_batch_target_count"]

    @property
    def endpoint_target_count(self) -> int:
        flow = self._require_recipe()["flow"]
        return ((self.loop.max_optimizer_steps - 1) * flow["endpoint_targets_per_full_batch"]
                + flow["endpoint_targets_in_final_batch"])

    @property
    def validation_steps(self) -> tuple[int, ...]:
        return _event_steps(self._require_recipe()["validation"]["loss"], self.loop.max_optimizer_steps)

    @property
    def sampling_steps(self) -> tuple[int, ...]:
        return _event_steps(self._require_recipe()["validation"]["sampling"], self.loop.max_optimizer_steps)

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        return _event_steps(self._require_recipe()["checkpointing"], self.loop.max_optimizer_steps)


def _config_values(value: Any) -> Any:
    """Convert immutable recipe containers and resolved paths to JSON values."""
    if isinstance(value, Mapping):
        return {key: _config_values(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_config_values(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def load_training_config(path: str | Path, *, expected_type: str | None = None) -> TrainingConfig:
    """Read either strict JSON format and resolve paths relative to its file.

    Specify expected_type at an execution entrypoint to reject unsupported
    experiment semantics before loading any model or creating outputs.
    """
    if expected_type not in (None, "baseline", "training_run"):
        raise ValueError("expected_type must be 'baseline', 'training_run', or None")
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = _object(json.load(handle, object_pairs_hook=_unique_fields), name="config")
    config_type = payload.get("config_type", "baseline")
    if config_type not in ("baseline", "training_run"):
        raise ValueError(f"unsupported config_type: {config_type!r}")
    if expected_type is not None and config_type != expected_type:
        raise ValueError(f"expected {expected_type!r} config, got {config_type!r}")
    if config_type == "training_run":
        return _parse_recipe(payload, config_path=config_path)
    return _parse_baseline(payload, config_path=config_path)


def _parse_baseline(
    payload: Mapping[str, Any], *, config_path: Path,
) -> TrainingConfig:
    """Parse the packed-only baseline schema; reject retired execution fields."""

    base_directory = config_path.parent

    schema_version = _integer(payload.get("schema_version"), name="schema_version")
    if schema_version != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    config_fields = {"schema_version", "seed", "device", "models", "data", "loop",
                     "optimizer", "scheduler", "benchmark", "flux_packing", "nvarc_max_tokens"}
    _fields(payload, config_fields, name="config")

    seed = _integer(payload["seed"], name="seed")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must lie in [0, 2**64)")

    device = _device(payload["device"])

    models_payload = _object(payload["models"], name="models")
    _fields(
        models_payload,
        {
            "nvarc_checkpoint",
            "flux_checkpoint",
            "nvarc_dtype",
            "vae_dtype",
            "flux_parameter_dtype",
            "flux_compute_dtype",
        },
        name="models",
    )
    models = _parse_models(models_payload, base_directory=base_directory)

    data_payload = _object(payload["data"], name="data")
    data_fields = {"split_directory", "view_name", "shuffle", "num_workers", "prefetch_factor"}
    _fields(
        data_payload,
        data_fields,
        name="data",
    )
    data = _parse_data(data_payload, base_directory=base_directory)

    loop_payload = _object(payload["loop"], name="loop")
    loop_fields = {
        "max_optimizer_steps",
        "optimizer_batch_size",
        "min_context_pairs_before_target",
        "vae_microbatch_size",
    }
    _fields(loop_payload, loop_fields, name="loop")
    loop = _parse_loop(loop_payload)

    optimizer_payload = _object(payload["optimizer"], name="optimizer")
    _fields(
        optimizer_payload,
        {"learning_rate", "beta1", "beta2", "epsilon", "weight_decay"},
        name="optimizer",
    )
    optimizer = _parse_optimizer(optimizer_payload)

    scheduler = None
    if payload.get("scheduler") is not None:
        section = _object(payload["scheduler"], name="scheduler")
        _fields(section, {"type", "warmup_steps", "total_steps"}, name="scheduler")
        if section["type"] != "cosine":
            raise ValueError("scheduler.type must be 'cosine'")
        scheduler = CosineSchedulerConfig(
            warmup_steps=_integer(section["warmup_steps"], name="scheduler.warmup_steps"),
            total_steps=_positive_integer(section["total_steps"], name="scheduler.total_steps"),
        )
        if not 0 <= scheduler.warmup_steps < scheduler.total_steps:
            raise ValueError("scheduler requires 0 <= warmup_steps < total_steps")
        if loop.max_optimizer_steps > scheduler.total_steps:
            raise ValueError("scheduler.total_steps must cover loop.max_optimizer_steps")

    benchmark = None
    if payload.get("benchmark") is not None:
        section = _object(payload["benchmark"], name="benchmark")
        _fields(section, {"warmup_steps", "measured_steps", "repetitions"}, name="benchmark")
        benchmark = BenchmarkConfig(
            warmup_steps=_integer(section["warmup_steps"], name="benchmark.warmup_steps"),
            measured_steps=_positive_integer(section["measured_steps"], name="benchmark.measured_steps"),
            repetitions=_positive_integer(section["repetitions"], name="benchmark.repetitions"),
        )
        if benchmark.warmup_steps < 0:
            raise ValueError("benchmark.warmup_steps must not be negative")
        if benchmark.warmup_steps + benchmark.measured_steps != loop.max_optimizer_steps:
            raise ValueError("benchmark warmup_steps + measured_steps must equal loop.max_optimizer_steps")

    section = _object(payload["flux_packing"], name="flux_packing")
    _fields(section, {"max_tokens"}, name="flux_packing")
    flux_packing = FluxPackingPolicy(**section)

    return TrainingConfig(
        path=config_path,
        name=config_path.stem,
        seed=seed,
        device=device,
        models=models,
        data=data,
        loop=loop,
        optimizer=optimizer,
        scheduler=scheduler,
        benchmark=benchmark,
        flux_packing=flux_packing,
        nvarc_max_tokens=_positive_integer(payload["nvarc_max_tokens"], name="nvarc_max_tokens"),
    )


def _device(value: Any) -> str:
    device = _string(value, name="device")
    device_parts = device.split(":")
    if not (
        device_parts[0] == "cuda"
        and len(device_parts) <= 2
        and (len(device_parts) == 1 or device_parts[1].isdigit())
    ):
        raise ValueError("device must be 'cuda' or 'cuda:<index>'")

    return device


def _parse_models(models_payload: Mapping[str, Any], *, base_directory: Path) -> ModelConfig:
    flux_parameter_dtype = _dtype(
        models_payload["flux_parameter_dtype"],
        name="models.flux_parameter_dtype",
    )
    if flux_parameter_dtype != "float32":
        raise ValueError("models.flux_parameter_dtype must be 'float32'")
    flux_compute_dtype = _dtype(
        models_payload["flux_compute_dtype"],
        name="models.flux_compute_dtype",
    )
    if flux_compute_dtype != "bfloat16":
        raise ValueError("models.flux_compute_dtype must be 'bfloat16'")

    models = ModelConfig(
        nvarc_checkpoint=_path(
            models_payload["nvarc_checkpoint"],
            base_directory=base_directory,
            name="models.nvarc_checkpoint",
        ),
        flux_checkpoint=_path(
            models_payload["flux_checkpoint"],
            base_directory=base_directory,
            name="models.flux_checkpoint",
        ),
        nvarc_dtype=_dtype(models_payload["nvarc_dtype"], name="models.nvarc_dtype"),
        vae_dtype=_dtype(models_payload["vae_dtype"], name="models.vae_dtype"),
        flux_parameter_dtype=flux_parameter_dtype,
        flux_compute_dtype=flux_compute_dtype,
    )

    if models.nvarc_dtype != "bfloat16":
        raise ValueError("models.nvarc_dtype must be bfloat16 for the FA3 build")

    return models


def _parse_data(
    data_payload: Mapping[str, Any], *, base_directory: Path,
    split_field: str = "split_directory", shuffle: bool | None = None,
) -> DataConfig:
    if shuffle is None:
        shuffle = data_payload["shuffle"]
    if not isinstance(shuffle, bool):
        raise TypeError("data.shuffle must be a boolean")
    num_workers = _integer(data_payload["num_workers"], name="data.num_workers")
    if num_workers < 0:
        raise ValueError("data.num_workers must not be negative")
    prefetch_factor = data_payload.get("prefetch_factor")
    if num_workers:
        prefetch_factor = _positive_integer(prefetch_factor, name="data.prefetch_factor")
    elif prefetch_factor is not None:
        raise ValueError("data.prefetch_factor must be null when num_workers = 0")
    data = DataConfig(
        split_directory=_path(
            data_payload[split_field],
            base_directory=base_directory,
            name=f"data.{split_field}",
        ),
        view_name=_string(data_payload["view_name"], name="data.view_name"),
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return data


def _parse_loop(loop_payload: Mapping[str, Any]) -> LoopConfig:
    loop_values = {
        field: _positive_integer(loop_payload[field], name=f"loop.{field}")
        for field in ("max_optimizer_steps", "optimizer_batch_size", "min_context_pairs_before_target",
                      "vae_microbatch_size")
    }
    loop = LoopConfig(
        max_optimizer_steps=loop_values["max_optimizer_steps"],
        optimizer_batch_size=loop_values["optimizer_batch_size"],
        min_context_pairs_before_target=loop_values[
            "min_context_pairs_before_target"
        ],
        vae_microbatch_size=loop_values["vae_microbatch_size"],
    )

    return loop


def _parse_optimizer(optimizer_payload: Mapping[str, Any]) -> AdamWConfig:
    learning_rate = _number(
        optimizer_payload["learning_rate"],
        name="optimizer.learning_rate",
    )
    beta1 = _number(optimizer_payload["beta1"], name="optimizer.beta1")
    beta2 = _number(optimizer_payload["beta2"], name="optimizer.beta2")
    epsilon = _number(optimizer_payload["epsilon"], name="optimizer.epsilon")
    weight_decay = _number(
        optimizer_payload["weight_decay"],
        name="optimizer.weight_decay",
    )
    if learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("optimizer beta values must lie in [0, 1)")
    if epsilon <= 0:
        raise ValueError("optimizer.epsilon must be positive")
    if weight_decay < 0:
        raise ValueError("optimizer.weight_decay must not be negative")

    return AdamWConfig(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, weight_decay=weight_decay,
        max_grad_norm=(_positive(optimizer_payload["max_grad_norm"], name="optimizer.max_grad_norm")
                       if "max_grad_norm" in optimizer_payload else None),
    )


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise ValueError(f"{name} fields differ: missing={missing!r}, extra={extra!r}")


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _dtype(value: Any, *, name: str) -> str:
    dtype = _string(value, name=name)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"{name} must be one of {tuple(sorted(_SUPPORTED_DTYPES))!r}")
    return dtype


def _path(value: Any, *, base_directory: Path, name: str) -> Path:
    path = Path(_string(value, name=name))
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _natural(value: Any, *, name: str) -> int:
    value = _integer(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _seed(value: Any, *, name: str) -> int:
    value = _natural(value, name=name)
    if value >= 2**64:
        raise ValueError(f"{name} must lie in [0, 2**64)")
    return value


def _positive(value: Any, *, name: str) -> float:
    value = _number(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _unit(value: Any, *, name: str) -> float:
    value = _number(value, name=name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _nullable(parser):
    return lambda value, *, name: None if value is None else parser(value, name=name)


def _names(allowed: str):
    """Allow explicit subsets of a finite metric vocabulary, never per-task keys."""
    allowed_names = frozenset(allowed.split())

    def parse(value: Any, *, name: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TypeError(f"{name} must be an array of strings")
        if len(set(value)) != len(value) or not set(value) <= allowed_names:
            raise ValueError(f"{name} contains duplicate or unsupported names")
        return value

    return parse


_CADENCE = {
    "every_optimizer_steps": _positive_integer,
    "at_start": bool,
    "at_end": bool,
    "replace_last_periodic_with_final": bool,
}
# Literals define the supported experiment semantics. Tunable numbers have explicit
# validators; changing a string must never silently select an unimplemented path.
_RECIPE_SCHEMA = {
    "config_type": "training_run", "schema_version": 1,
    "name": _string, "seed": _seed,
    "deployment": {
        "provider": "runpod", "gpu_type": _string, "gpu_count": 1,
        "output_directory": _nullable(_string),
    },
    "models": {
        "nvarc_checkpoint": _string, "flux_checkpoint": _string,
        "initialization": "base_checkpoint", "resume_from": _nullable(_string),
        "trainable_modules": ["flux_transformer"],
        "nvarc_dtype": "bfloat16", "vae_dtype": "float32",
        "flux_parameter_dtype": "float32", "flux_compute_dtype": "bfloat16",
    },
    "execution": {
        "device": _string, "distributed_strategy": "single_process",
        "nvarc_attention_backend": "fa3_packed", "nvarc_hidden_layers": [9, 18, 27],
        "nvarc_early_exit_layer": 27, "nvarc_hidden_state_normalization": "none",
        "nvarc_max_tokens": _positive_integer,
        "flux_attention_backend": "fa3_packed", "flux_max_joint_tokens": _positive_integer,
        "activation_checkpointing": False, "profiler": None,
    },
    "data": {
        "pool": "nvarc_train_arc_agi_1_training_only",
        "task_index": _string, "source_archive": _string, "official_training_keys": _string,
        "prepared_split_directory": _string, "view_name": "train",
        "require_both_parents_in_official_training": True,
        "expected_task_count": _positive_integer,
        "task_order": "seeded_shuffle_of_sorted_task_ids", "views_per_task_per_epoch": 1,
        "pairs_per_sequence": 5, "pair_selection": "random_without_replacement",
        "color_permutation": "uniform_all_10_colors", "geometry_sampling": "uniform",
        "geometries": ["ID", "R90", "R180", "R270", "FH", "FV", "FD1", "FD2"],
        "shared_augmentation_across_sequence": True, "vae_safety_filter": False,
        "max_cells_per_sequence": 9000, "max_tokens": 8400, "overlength_policy": "error",
        "num_workers": _natural, "prefetch_factor": _nullable(_positive_integer),
    },
    "canvas": {
        "height": 30, "width": 30, "cell_size": 16, "alignment": "top_left",
        "palette": "greedy_rgb_maximin_v1", "padding": "white_sentinel",
    },
    "loop": {
        "epochs": 1, "max_optimizer_steps": _positive_integer,
        "optimizer_batch_size": _positive_integer, "min_context_pairs_before_target": 2,
        "vae_microbatch_size": _positive_integer,
        "drop_last": False, "expected_target_count": _positive_integer,
        "expected_final_batch_target_count": _positive_integer,
    },
    "loss": {
        "type": "region_weighted_velocity_mse", "reduction_dtype": "float32",
        "region_weights": {"canvas": _unit, "grid": _unit, "boundary": _unit},
        "canvas_region": "all_tokens",
        "boundary_region": "adjacent_bottom_right_including_corner",
        "empty_boundary_policy": "renormalize_remaining_weights",
        "target_reduction": "equal_target_mean",
        "microbatch_reduction": "actual_target_count_weighted",
    },
    "flow": {
        "interpolation": "data_to_noise", "velocity_target": "noise_minus_data",
        "time_distribution": "uniform_with_fixed_endpoint_count",
        "uniform_min": 0.0, "uniform_max": 1.0, "uniform_bounds": "open",
        "endpoint_time": 1.0, "endpoint_targets_per_full_batch": _natural,
        "endpoint_targets_in_final_batch": _natural,
        "endpoint_selection": "random_without_replacement",
        "noise_distribution": "independent_standard_normal_per_target",
        "randomness_assignment": "logical_target_order_before_packing",
    },
    "optimizer": {
        "type": "adamw", "learning_rate": _positive, "beta1": _unit, "beta2": _unit,
        "epsilon": _positive, "weight_decay": _number, "state_dtype": "float32",
        "max_grad_norm": _positive,
    },
    "scheduler": {
        "type": "linear_warmup_constant", "warmup_steps": _positive_integer,
        "first_update_warmup_fraction": _positive, "reset_on_resume": False,
    },
    "validation": {
        "bundle": _string, "split_seed": _seed, "expected_task_count": _positive_integer,
        "expected_query_count": _positive_integer, "cache_device": "cpu",
        "context": "all_original_demonstrations", "augmentation": False,
        "loss": {
            **_CADENCE, "timesteps": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], "noise_seed": _seed,
            "uniform_proxy_weight": _unit, "endpoint_weight": _unit,
        },
        "sampling": {
            **_CADENCE, "expected_task_count": _positive_integer,
            "expected_query_count": _positive_integer, "seeds": [_seed, _seed],
            "steps": _positive_integer, "solver": "euler", "time_schedule": "linear",
            "trajectory_dtype": "float32", "use_target_shape": False,
            "test_time_training": False, "test_time_augmentation": False, "voting": False,
            "task_success": "one_sample_index_solves_all_queries",
            "cell_accuracy_scope": "shape_correct_predictions",
        },
    },
    "logging": {
        "backend": "wandb", "entity": _nullable(_string), "project": _nullable(_string),
        "mode": "online", "api_key_environment_variable": "WANDB_API_KEY",
        "step_metric": "progress/optimizer_step", "every_optimizer_steps": 1,
        "aggregate_across_optimizer_steps": False, "batch_scalars_per_step": True,
        "automatic_system_metrics": False, "console_capture": False,
        "model_watch": False, "histograms": False,
        "train_metrics": _names(
            "train/loss train/mse_canvas train/mse_grid train/mse_boundary "
            "train/contribution_canvas train/contribution_grid train/contribution_boundary "
            "train/loss_t1 train/loss_uniform train/loss_demos_2 train/loss_demos_3 train/loss_demos_4 "
            "train/targets_t1 train/targets_uniform train/targets_demos_2 train/targets_demos_3 "
            "train/targets_demos_4 train/boundary_target_count train/learning_rate train/grad_norm "
            "train/grad_clipped train/step_seconds train/targets_per_second train/targets "
            "progress/optimizer_step progress/epoch_fraction progress/targets_seen"
        ),
        "workload_counters": {
            "phases": ["train", "val", "sample"],
            "names": _names("nvarc_tokens flux_text_tokens flux_image_tokens flux_joint_tokens flux_forward_calls"),
            "cumulative_suffix": "_total",
        },
        "validation_metrics": _names("val/loss val/loss_uniform_proxy val/target_count val/boundary_target_count val/seconds"),
        "validation_time_metrics": {
            "names": _names("loss mse_canvas mse_grid mse_boundary"),
            "suffixes": ["t_0_1", "t_0_3", "t_0_5", "t_0_7", "t_0_9", "t1"],
        },
        "sampling_metrics": _names(
            "sample/pass_at_1 sample/pass_at_2 sample/solved_tasks_at_1 sample/solved_tasks_at_2 "
            "sample/task_count sample/shape_accuracy sample/cell_accuracy sample/invalid_grid_rate "
            "sample/shape_correct_count sample/prediction_count sample/seconds"
        ),
        "checkpoint_metrics": _names("checkpoint/seconds checkpoint/size_bytes checkpoint/path"),
        "hardware": {
            "per_step_metrics": _names("system/gpu_allocated_bytes system/gpu_reserved_bytes system/gpu_peak_allocated_bytes"),
            "sample_interval_seconds": _positive,
            "sampled_metrics": _names(
                "system/gpu_utilization_percent system/gpu_power_watts system/gpu_temperature_celsius "
                "system/cpu_utilization_percent system/ram_rss_bytes"
            ),
        },
        "predictions": {
            "key": "sample/predictions", "cadence": "sampling_checks",
            "columns": ["task_id", "query_index", "comparison", "sample_1_exact", "sample_2_exact"],
            "image_panels": ["input", "expected_output", "sample_1", "sample_2"],
            "show_full_generated_canvas": True,
        },
    },
    "checkpointing": {
        **_CADENCE, "at_start": False, "save_full_state": True,
        "keep_last": _nullable(_positive_integer), "upload_weights_to_wandb": False,
    },
    "final_test": {"enabled": False, "checkpoint_selection": None},
}


def _parse_recipe(payload: Mapping[str, Any], *, config_path: Path) -> TrainingConfig:
    """Normalize a finite run directly through the shared component readers."""
    recipe = _validate(payload, _RECIPE_SCHEMA, name="config")
    models = _parse_models(recipe["models"], base_directory=config_path.parent)
    # Data preparation already froze the seeded task order for this epoch.
    data = _parse_data(recipe["data"], base_directory=config_path.parent,
                       split_field="prepared_split_directory", shuffle=False)
    loop = _parse_loop(recipe["loop"])
    optimizer = _parse_optimizer(recipe["optimizer"])
    device = _device(recipe["execution"]["device"])
    packing = FluxPackingPolicy(max_tokens=recipe["execution"]["flux_max_joint_tokens"])
    _validate_relationships(recipe)
    for section, field in (
        ("models", "nvarc_checkpoint"), ("models", "flux_checkpoint"), ("models", "resume_from"),
        ("deployment", "output_directory"), ("data", "task_index"),
        ("data", "source_archive"), ("data", "official_training_keys"),
        ("data", "prepared_split_directory"), ("validation", "bundle"),
    ):
        value = recipe[section][field]
        if value is not None:
            recipe[section][field] = _path(value, base_directory=config_path.parent, name=f"{section}.{field}")
    return TrainingConfig(
        config_type="training_run", path=config_path, name=recipe["name"],
        seed=recipe["seed"], device=device, recipe=_freeze(recipe), models=models,
        data=data, loop=loop, optimizer=optimizer, flux_packing=packing,
        nvarc_max_tokens=recipe["execution"]["nvarc_max_tokens"],
        scheduler=LinearWarmupConstantConfig(warmup_steps=recipe["scheduler"]["warmup_steps"]),
    )


def _validate_relationships(recipe: dict[str, Any]) -> None:
    data, loop, flow = recipe["data"], recipe["loop"], recipe["flow"]
    targets_per_sequence = data["pairs_per_sequence"] - loop["min_context_pairs_before_target"]
    total = data["expected_task_count"] * targets_per_sequence
    batch = loop["optimizer_batch_size"]
    if batch % targets_per_sequence:
        raise ValueError("loop.optimizer_batch_size must fit whole training sequences")
    steps = (total + batch - 1) // batch
    final_count = (total - 1) % batch + 1
    for name, expected in (
        ("expected_target_count", total), ("max_optimizer_steps", steps),
        ("expected_final_batch_target_count", final_count),
    ):
        if loop[name] != expected:
            raise ValueError(f"loop.{name} must equal {expected} for the complete epoch")
    for name, count in (("endpoint_targets_per_full_batch", batch),
                        ("endpoint_targets_in_final_batch", final_count)):
        if flow[name] > count:
            raise ValueError(f"flow.{name} exceeds its batch target count")
    if final_count == batch and flow["endpoint_targets_in_final_batch"] != flow["endpoint_targets_per_full_batch"]:
        raise ValueError("a full final batch must use the full-batch endpoint count")

    weights = recipe["loss"]["region_weights"]
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("loss.region_weights must sum to one")
    if weights["canvas"] + weights["grid"] <= 0:
        raise ValueError("loss needs positive weight outside the boundary for 30x30 grids")
    schedule = recipe["scheduler"]
    if schedule["warmup_steps"] > steps:
        raise ValueError("scheduler.warmup_steps must not exceed the epoch")
    if not math.isclose(schedule["first_update_warmup_fraction"], 1 / schedule["warmup_steps"], rel_tol=1e-12):
        raise ValueError("scheduler.first_update_warmup_fraction must equal 1 / warmup_steps")

    validation = recipe["validation"]
    sampling, loss = validation["sampling"], validation["loss"]
    if validation["expected_query_count"] < validation["expected_task_count"]:
        raise ValueError("validation must contain at least one query per task")
    if not sampling["expected_task_count"] <= sampling["expected_query_count"] <= validation["expected_query_count"]:
        raise ValueError("sampling query counts must fit the validation pool")
    if sampling["expected_task_count"] > validation["expected_task_count"]:
        raise ValueError("sampling tasks must fit the validation pool")
    if len(set(sampling["seeds"])) != 2:
        raise ValueError("validation.sampling.seeds must be distinct")
    endpoint_fraction = flow["endpoint_targets_per_full_batch"] / batch
    if not (math.isclose(loss["endpoint_weight"], endpoint_fraction, rel_tol=0, abs_tol=1e-12)
            and math.isclose(loss["uniform_proxy_weight"], 1 - endpoint_fraction, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("validation.loss weights must match the full-batch flow mixture")
    for cadence in (loss, sampling, recipe["checkpointing"]):
        if cadence["replace_last_periodic_with_final"] and not cadence["at_end"]:
            raise ValueError("replace_last_periodic_with_final requires at_end")
    checkpoint_steps = _event_steps(recipe["checkpointing"], steps)
    if checkpoint_steps != tuple(step for step in _event_steps(sampling, steps) if step):
        raise ValueError("checkpointing must align with every nonzero sampling check")
    if recipe["logging"]["step_metric"] not in recipe["logging"]["train_metrics"]:
        raise ValueError("logging.train_metrics must include the step metric")


def _event_steps(cadence: Mapping[str, Any], total: int) -> tuple[int, ...]:
    steps = list(range(cadence["every_optimizer_steps"], total + 1, cadence["every_optimizer_steps"]))
    if cadence["at_end"] and (not steps or steps[-1] != total):
        if cadence["replace_last_periodic_with_final"] and steps:
            steps.pop()
        steps.append(total)
    if cadence["at_start"]:
        steps.insert(0, 0)
    return tuple(steps)


def _validate(value: Any, spec: Any, *, name: str) -> Any:
    if isinstance(spec, dict):
        value = _object(value, name=name)
        _fields(value, set(spec), name=name)
        return {key: _validate(value[key], child, name=f"{name}.{key}") for key, child in spec.items()}
    if isinstance(spec, list):
        if not isinstance(value, list) or len(value) != len(spec):
            raise ValueError(f"{name} must be an array of length {len(spec)}")
        return [_validate(item, child, name=f"{name}[{index}]") for index, (item, child) in enumerate(zip(value, spec))]
    if spec is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value
    if callable(spec):
        return spec(value, name=name)
    if isinstance(spec, float):
        value = _number(value, name=name)
    if type(value) is not type(spec) or value != spec:
        raise ValueError(f"{name} must be {spec!r}")
    return value


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config field: {key}")
        result[key] = value
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


__all__ = [
    "BASELINE_SCHEMA_VERSION", "AdamWConfig", "BenchmarkConfig", "CosineSchedulerConfig",
    "LinearWarmupConstantConfig", "DataConfig", "LoopConfig", "ModelConfig",
    "TrainingConfig", "load_training_config",
]
