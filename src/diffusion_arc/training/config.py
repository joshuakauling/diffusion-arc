"""One strict configuration for training, measurement, and profiling workloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from copy import deepcopy
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from diffusion_arc._constants import DEFAULT_NVARC_MAX_TOKENS, NVARC_CONTEXT_WIDTH, NVARC_HIDDEN_SIZE
from .step.flux_batching import FluxPackingPolicy


# Version 2 unifies training and benchmark inputs and makes experiment choices explicit.
# The version describes the JSON contract, not the model or checkpoint file format.
TRAINING_SCHEMA_VERSION = 2
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
    """Validated components and an immutable recipe, independent of the runner.

    Loading never requires local checkpoints, prepared data, or credentials.
    Derived loop counts are kept internally and omitted from recipe_dict().
    """

    seed: int
    device: str
    models: ModelConfig
    data: DataConfig
    loop: LoopConfig
    optimizer: AdamWConfig
    flux_packing: FluxPackingPolicy
    nvarc_hidden_layers: tuple[int, int, int]
    nvarc_max_tokens: int = DEFAULT_NVARC_MAX_TOKENS
    scheduler: CosineSchedulerConfig | LinearWarmupConstantConfig | None = None
    benchmark: BenchmarkConfig | None = None
    config_type: str = "training_run"
    recipe: Mapping[str, Any] | None = None
    path: Path | None = field(default=None, compare=False)
    name: str | None = field(default=None, compare=False)

    def _require_recipe(self) -> Mapping[str, Any]:
        """Return the validated recipe; reject component-only objects used by local fixtures."""
        if self.recipe is None:
            raise ValueError("this property requires a training_run config")
        return self.recipe

    def recipe_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible recipe copy with resolved paths as strings.

        Logging and checkpoints use the same representation. Mutating the copy
        does not change the validated, immutable configuration.
        """
        recipe = _to_json_values(self._require_recipe())
        for key in ("max_optimizer_steps", "expected_target_count", "expected_final_batch_target_count", "epoch_steps"):
            recipe["loop"].pop(key, None)
        return recipe

    @property
    def targets_per_sequence(self) -> int:
        return self._require_recipe()["data"]["pairs_per_sequence"] - self.loop.min_context_pairs_before_target

    @property
    def fsdp_reshard_after_forward(self) -> bool:
        """Block resharding policy; older recipes retain their original behavior."""
        return self._require_recipe()["execution"].get("fsdp_reshard_after_forward", True)

    @property
    def final_batch_target_count(self) -> int:
        """Target count of the prepared epoch tail, independent of a step prefix."""
        recipe = self._require_recipe()
        task_count = recipe["data"]["expected_task_count"]
        if task_count is None:
            return recipe["loop"]["expected_final_batch_target_count"]
        return (task_count * self.targets_per_sequence - 1) % self.loop.optimizer_batch_size + 1

    @property
    def epoch_optimizer_steps(self) -> int | None:
        return self._require_recipe()["loop"]["epoch_steps"]

    @property
    def endpoint_target_count(self) -> int:
        flow = self._require_recipe()["flow"]
        if flow["time_distribution"] == "uniform":
            return 0
        run_final_count = self._require_recipe()["loop"]["expected_final_batch_target_count"]
        final_quota = (flow["endpoint_targets_per_full_batch"]
                       if run_final_count == self.loop.optimizer_batch_size
                       else flow["endpoint_targets_in_final_batch"])
        return (self.loop.max_optimizer_steps - 1) * flow["endpoint_targets_per_full_batch"] + final_quota

    @property
    def validation_steps(self) -> tuple[int, ...]:
        validation = self._require_recipe()["validation"]
        return () if validation is None else _compute_event_steps(validation["loss"], self.loop.max_optimizer_steps)

    @property
    def sampling_steps(self) -> tuple[int, ...]:
        validation = self._require_recipe()["validation"]
        return () if validation is None else _compute_event_steps(validation["sampling"], self.loop.max_optimizer_steps)

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        return _compute_event_steps(self._require_recipe()["checkpointing"], self.loop.max_optimizer_steps)


def _to_json_values(value: Any) -> Any:
    """Convert immutable recipe containers and resolved paths to JSON values."""
    if isinstance(value, Mapping):
        return {key: _to_json_values(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_values(item) for item in value]
    return str(value) if isinstance(value, Path) else value


def load_training_config(path: str | Path, *, expected_type: str | None = None) -> TrainingConfig:
    """Read the strict shared schema; no experiment choices are filled in."""
    if expected_type not in (None, "training_run"):
        raise ValueError("expected_type must be 'training_run' or None; baseline configs are retired")
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_fields)
    return parse_training_config(payload, config_path=config_path)


def parse_training_config(payload: Mapping[str, Any], *, config_path: str | Path) -> TrainingConfig:
    """Validate explicit settings, resolve paths, and build immutable runtime components."""
    payload = _validate_json_object(payload, name="config")
    config_path = Path(config_path).resolve()
    # Adapt the validation schema to the explicitly selected configuration variants.
    schema = deepcopy(_RECIPE_SCHEMA)
    if isinstance(payload.get("execution"), Mapping):
        if "fsdp_reshard_after_forward" in payload["execution"]:
            schema["execution"]["fsdp_reshard_after_forward"] = bool
    for section in ("validation", "logging", "checkpointing", "scheduler"):
        if section in payload and payload[section] is None:
            schema[section] = None
    if isinstance(payload.get("data"), Mapping) and payload["data"].get("source") == "prepared_split":
        schema["data"] = {
            "source": "prepared_split", "prepared_split_directory": _validate_nonempty_string,
            "view_name": "train", "shuffle": bool,
            "expected_task_count": _allow_none(_validate_positive_integer),
            "pairs_per_sequence": _validate_positive_integer,
            "num_workers": _validate_nonnegative_integer, "prefetch_factor": _allow_none(_validate_positive_integer),
        }
    if isinstance(payload.get("loss"), Mapping) and payload["loss"].get("type") == "velocity_mse":
        schema["loss"] = {key: value for key, value in schema["loss"].items()
                          if key in ("reduction_dtype", "target_reduction", "microbatch_reduction")}
        schema["loss"]["type"] = "velocity_mse"
    if isinstance(payload.get("flow"), Mapping) and payload["flow"].get("time_distribution") == "uniform":
        schema["flow"] = {key: value for key, value in schema["flow"].items()
                          if not key.startswith("endpoint_")}
        schema["flow"]["time_distribution"] = "uniform"
    if isinstance(payload.get("scheduler"), Mapping) and payload["scheduler"].get("type") == "cosine":
        schema["scheduler"] = {"type": "cosine", "warmup_steps": _validate_nonnegative_integer,
                               "total_steps": _validate_positive_integer}
    if isinstance(payload.get("benchmark"), Mapping):
        schema["benchmark"] = {"warmup_steps": _validate_nonnegative_integer, "measured_steps": _validate_positive_integer,
                               "repetitions": _validate_positive_integer}
    if isinstance(payload.get("models"), Mapping) and payload["models"].get("initialization") == "transformer_weights":
        schema["models"].update(
            initialization="transformer_weights", transformer_checkpoint=_allow_none(_validate_nonempty_string),
            completed_epochs=_validate_nonnegative_integer,
            transformer_source={"repo_id": _validate_nonempty_string, "revision": _validate_nonempty_string, "subfolder": _validate_nonempty_string},
        )
        if schema["logging"] is not None:
            schema["logging"].update(parent_run_id=_validate_nonempty_string, resume_parent_run=bool)
    if (isinstance(payload.get("flow"), Mapping)
            and payload["flow"].get("time_distribution") == "uniform_with_fixed_endpoint_and_high_noise_counts"):
        schema["flow"].update(
            time_distribution="uniform_with_fixed_endpoint_and_high_noise_counts",
            high_noise_min=_validate_unit_interval, high_noise_max=1.0,
            high_noise_targets_per_full_batch=_validate_nonnegative_integer,
            high_noise_targets_in_final_batch=_validate_nonnegative_integer,
        )

    recipe = _validate_against_schema(payload, schema, name="config")
    models = _parse_model_config(recipe["models"], base_directory=config_path.parent)
    # Data preparation already froze the seeded task order for this epoch.
    data = _parse_data_config(recipe["data"], base_directory=config_path.parent,
                       split_field="prepared_split_directory",
                       shuffle=recipe["data"]["shuffle"] if recipe["data"]["source"] == "prepared_split" else False)
    _derive_loop_counts(recipe)
    loop = _parse_loop_config(recipe["loop"])
    optimizer = _parse_optimizer_config(recipe["optimizer"])
    device = _validate_cuda_device(recipe["execution"]["device"])
    packing = FluxPackingPolicy(
        max_tokens=recipe["execution"]["flux_max_joint_tokens"],
        max_spread_tokens=recipe["execution"]["flux_max_spread_tokens"],
        algorithm=recipe["execution"]["flux_packing_algorithm"],
    )
    _validate_config_relationships(recipe)
    paths = [
        ("models", "nvarc_checkpoint"), ("models", "flux_checkpoint"), ("models", "resume_from"),
        ("deployment", "output_directory"), ("data", "task_index"),
        ("data", "source_archive"), ("data", "official_training_keys"),
        ("data", "prepared_split_directory"), ("validation", "bundle"),
    ]
    if recipe["models"]["initialization"] == "transformer_weights":
        paths.append(("models", "transformer_checkpoint"))
    for section, field in paths:
        if recipe[section] is None or field not in recipe[section]:
            continue
        value = recipe[section][field]
        if value is not None:
            recipe[section][field] = _resolve_config_path(value, base_directory=config_path.parent, name=f"{section}.{field}")
    return TrainingConfig(
        config_type="training_run", path=config_path, name=recipe["name"],
        seed=recipe["seed"], device=device, recipe=_freeze_config_values(recipe), models=models,
        data=data, loop=loop, optimizer=optimizer, flux_packing=packing,
        nvarc_hidden_layers=tuple(recipe["execution"]["nvarc_hidden_layers"]),
        nvarc_max_tokens=recipe["execution"]["nvarc_max_tokens"],
        scheduler=_parse_scheduler_config(recipe["scheduler"], loop.max_optimizer_steps),
        benchmark=None if recipe["benchmark"] is None else BenchmarkConfig(**recipe["benchmark"]),
    )


def _validate_cuda_device(value: Any) -> str:
    """Validate CUDA device syntax without querying hardware availability."""
    device = _validate_nonempty_string(value, name="device")
    device_parts = device.split(":")
    if not (
        device_parts[0] == "cuda"
        and len(device_parts) <= 2
        and (len(device_parts) == 1 or device_parts[1].isdigit())
    ):
        raise ValueError("device must be 'cuda' or 'cuda:<index>'")

    return device


def _parse_model_config(models_payload: Mapping[str, Any], *, base_directory: Path) -> ModelConfig:
    """Validate model precision and resolve checkpoint paths without loading weights."""
    flux_parameter_dtype = _validate_dtype(
        models_payload["flux_parameter_dtype"],
        name="models.flux_parameter_dtype",
    )
    if flux_parameter_dtype != "float32":
        raise ValueError("models.flux_parameter_dtype must be 'float32'")
    flux_compute_dtype = _validate_dtype(
        models_payload["flux_compute_dtype"],
        name="models.flux_compute_dtype",
    )
    if flux_compute_dtype != "bfloat16":
        raise ValueError("models.flux_compute_dtype must be 'bfloat16'")

    models = ModelConfig(
        nvarc_checkpoint=_resolve_config_path(
            models_payload["nvarc_checkpoint"],
            base_directory=base_directory,
            name="models.nvarc_checkpoint",
        ),
        flux_checkpoint=_resolve_config_path(
            models_payload["flux_checkpoint"],
            base_directory=base_directory,
            name="models.flux_checkpoint",
        ),
        nvarc_dtype=_validate_dtype(models_payload["nvarc_dtype"], name="models.nvarc_dtype"),
        vae_dtype=_validate_dtype(models_payload["vae_dtype"], name="models.vae_dtype"),
        flux_parameter_dtype=flux_parameter_dtype,
        flux_compute_dtype=flux_compute_dtype,
    )

    if models.nvarc_dtype != "bfloat16":
        raise ValueError("models.nvarc_dtype must be bfloat16 for the FA3 build")

    return models


def _parse_data_config(
    data_payload: Mapping[str, Any], *, base_directory: Path,
    split_field: str = "split_directory", shuffle: bool | None = None,
) -> DataConfig:
    """Validate loader settings and resolve the split path without opening the dataset."""
    if shuffle is None:
        shuffle = data_payload["shuffle"]
    if not isinstance(shuffle, bool):
        raise TypeError("data.shuffle must be a boolean")
    num_workers = _validate_integer(data_payload["num_workers"], name="data.num_workers")
    if num_workers < 0:
        raise ValueError("data.num_workers must not be negative")
    prefetch_factor = data_payload.get("prefetch_factor")
    if num_workers:
        prefetch_factor = _validate_positive_integer(prefetch_factor, name="data.prefetch_factor")
    elif prefetch_factor is not None:
        raise ValueError("data.prefetch_factor must be null when num_workers = 0")
    data = DataConfig(
        split_directory=_resolve_config_path(
            data_payload[split_field],
            base_directory=base_directory,
            name=f"data.{split_field}",
        ),
        view_name=_validate_nonempty_string(data_payload["view_name"], name="data.view_name"),
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return data


def _parse_loop_config(loop_payload: Mapping[str, Any]) -> LoopConfig:
    """Build loop settings after run and epoch counts have been derived."""
    loop_values = {
        field: _validate_positive_integer(loop_payload[field], name=f"loop.{field}")
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


def _parse_optimizer_config(optimizer_payload: Mapping[str, Any]) -> AdamWConfig:
    """Validate AdamW hyperparameters and return settings without creating an optimizer."""
    learning_rate = _parse_finite_number(
        optimizer_payload["learning_rate"],
        name="optimizer.learning_rate",
    )
    beta1 = _parse_finite_number(optimizer_payload["beta1"], name="optimizer.beta1")
    beta2 = _parse_finite_number(optimizer_payload["beta2"], name="optimizer.beta2")
    epsilon = _parse_finite_number(optimizer_payload["epsilon"], name="optimizer.epsilon")
    weight_decay = _parse_finite_number(
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
        max_grad_norm=(_validate_positive_number(optimizer_payload["max_grad_norm"], name="optimizer.max_grad_norm")
                       if optimizer_payload["max_grad_norm"] is not None else None),
    )


def _validate_json_object(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require a mapping for a JSON object and return it unchanged."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _validate_required_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    """Reject missing and unknown keys; every expected field is required."""
    actual = set(value)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected))
        raise ValueError(f"{name} fields differ: missing={missing!r}, extra={extra!r}")


def _validate_nonempty_string(value: Any, *, name: str) -> str:
    """Require a nonempty string and return it unchanged."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _validate_integer(value: Any, *, name: str) -> int:
    """Require an integer, explicitly rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _validate_positive_integer(value: Any, *, name: str) -> int:
    """Require an integer greater than zero."""
    integer = _validate_integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _parse_finite_number(value: Any, *, name: str) -> float:
    """Accept a finite int or float, reject booleans, and return a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_dtype(value: Any, *, name: str) -> str:
    """Require one of the supported precision names."""
    dtype = _validate_nonempty_string(value, name=name)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"{name} must be one of {tuple(sorted(_SUPPORTED_DTYPES))!r}")
    return dtype


def _resolve_config_path(value: Any, *, base_directory: Path, name: str) -> Path:
    """Resolve a path against the config file's directory without requiring it to exist."""
    path = Path(_validate_nonempty_string(value, name=name))
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _validate_nonnegative_integer(value: Any, *, name: str) -> int:
    """Require an integer greater than or equal to zero."""
    value = _validate_integer(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _validate_nvarc_hidden_layers(value: Any, *, name: str) -> list[int]:
    """Select three ordered one-based block outputs for FLUX's fixed text width."""
    count = NVARC_CONTEXT_WIDTH // NVARC_HIDDEN_SIZE
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must be an array of {count} layer indices")
    layers = [_validate_integer(index, name=f"{name}[{position}]")
              for position, index in enumerate(value)]
    if not all(1 <= index <= 36 for index in layers):
        raise ValueError(f"{name} layer indices must lie in [1, 36]")
    if any(left >= right for left, right in zip(layers, layers[1:])):
        raise ValueError(f"{name} layer indices must be distinct and strictly increasing")
    return layers


def _validate_seed(value: Any, *, name: str) -> int:
    """Require an integer seed in the unsigned 64-bit range."""
    value = _validate_nonnegative_integer(value, name=name)
    if value >= 2**64:
        raise ValueError(f"{name} must lie in [0, 2**64)")
    return value


def _validate_positive_number(value: Any, *, name: str) -> float:
    """Require a finite number greater than zero and return it as a float."""
    value = _parse_finite_number(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_unit_interval(value: Any, *, name: str) -> float:
    """Require a finite number in the inclusive interval [0, 1]."""
    value = _parse_finite_number(value, name=name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _allow_none(parser):
    """Wrap a value parser so explicit null is accepted; missing fields remain errors."""
    return lambda value, *, name: None if value is None else parser(value, name=name)


def _validate_deployment_provider(value: Any, *, name: str) -> str:
    """Require a supported local or cloud execution provider."""
    if value not in ("local", "runpod", "modal"):
        raise ValueError(f"{name} must be local, runpod, or modal")
    return value


def _choice_validator(*choices: str):
    """Build a validator accepting exactly one of the supplied string choices."""

    def parse(value: Any, *, name: str) -> str:
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{name} must be one of {choices!r}")
        return value
    return parse


def _metric_names_validator(allowed: str):
    """Build a validator for unique metric names from a whitespace-separated vocabulary."""
    allowed_names = frozenset(allowed.split())

    def parse(value: Any, *, name: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TypeError(f"{name} must be an array of strings")
        if len(set(value)) != len(value) or not set(value) <= allowed_names:
            raise ValueError(f"{name} contains duplicate or unsupported names")
        return value

    return parse


_CADENCE = {
    "every_optimizer_steps": _validate_positive_integer,
    "at_start": bool,
    "at_end": bool,
    "replace_last_periodic_with_final": bool,
}
# Literals define the supported experiment semantics. Tunable numbers have explicit
# validators; changing a string must never silently select an unimplemented path.
_RECIPE_SCHEMA = {
    "config_type": "training_run", "schema_version": TRAINING_SCHEMA_VERSION,
    "name": _validate_nonempty_string, "seed": _validate_seed,
    "deployment": {
        "provider": _validate_deployment_provider, "gpu_type": _validate_nonempty_string, "gpu_count": _validate_positive_integer,
        "output_directory": _allow_none(_validate_nonempty_string),
    },
    "models": {
        "nvarc_checkpoint": _validate_nonempty_string, "flux_checkpoint": _validate_nonempty_string,
        "initialization": "base_checkpoint", "resume_from": _allow_none(_validate_nonempty_string),
        "trainable_modules": ["flux_transformer"],
        "nvarc_dtype": "bfloat16", "vae_dtype": "float32",
        "flux_parameter_dtype": "float32", "flux_compute_dtype": "bfloat16",
    },
    "execution": {
        "device": _validate_nonempty_string, "distributed_strategy": _choice_validator("single_process", "ddp", "fsdp2"),
        "nvarc_attention_backend": "fa3_packed", "nvarc_hidden_layers": _validate_nvarc_hidden_layers,
        "nvarc_early_exit_layer": _validate_positive_integer, "nvarc_hidden_state_normalization": "none",
        "nvarc_max_tokens": _validate_positive_integer,
        "flux_attention_backend": "fa3_packed", "flux_max_joint_tokens": _validate_positive_integer,
        "flux_max_spread_tokens": _validate_nonnegative_integer,
        "flux_packing_algorithm": _choice_validator("legacy_bfd_v1", "balanced_release_v1"),
        "activation_checkpointing": False, "profiler": None,
    },
    "data": {
        "source": "epoch_pool", "pool": "nvarc_train_arc_agi_1_training_only",
        "task_index": _validate_nonempty_string, "source_archive": _validate_nonempty_string, "official_training_keys": _validate_nonempty_string,
        "prepared_split_directory": _validate_nonempty_string, "view_name": "train",
        "require_both_parents_in_official_training": True,
        "expected_task_count": _validate_positive_integer,
        "task_order": "seeded_shuffle_of_sorted_task_ids", "views_per_task_per_epoch": 1,
        "pairs_per_sequence": 5, "pair_selection": "random_without_replacement",
        "color_permutation": "uniform_all_10_colors", "geometry_sampling": "uniform",
        "geometries": ["ID", "R90", "R180", "R270", "FH", "FV", "FD1", "FD2"],
        "shared_augmentation_across_sequence": True, "vae_safety_filter": False,
        "max_cells_per_sequence": 9000, "max_tokens": 8400, "overlength_policy": "error",
        "num_workers": _validate_nonnegative_integer, "prefetch_factor": _allow_none(_validate_positive_integer),
    },
    "canvas": {
        "height": 30, "width": 30, "cell_size": 16, "alignment": "top_left",
        "palette": "greedy_rgb_maximin_v1", "padding": "white_sentinel",
    },
    "loop": {
        "epochs": _allow_none(_validate_positive_integer), "max_steps": _allow_none(_validate_positive_integer),
        "optimizer_batch_size": _validate_positive_integer, "min_context_pairs_before_target": 2,
        "vae_microbatch_size": _validate_positive_integer,
        "drop_last": False,
    },
    "loss": {
        "type": "region_weighted_velocity_mse", "reduction_dtype": "float32",
        "region_weights": {"canvas": _validate_unit_interval, "grid": _validate_unit_interval, "boundary": _validate_unit_interval},
        "canvas_region": "all_tokens",
        "boundary_region": "adjacent_bottom_right_including_corner",
        "empty_boundary_policy": "renormalize_remaining_weights",
        "target_reduction": "equal_target_mean",
        "microbatch_reduction": "actual_target_count_weighted",
    },
    "flow": {
        "rng_scheme": _choice_validator("legacy_stream_v1", "global_step_target_v1"),
        "interpolation": "data_to_noise", "velocity_target": "noise_minus_data",
        "time_distribution": "uniform_with_fixed_endpoint_count",
        "uniform_min": 0.0, "uniform_max": 1.0, "uniform_bounds": "open",
        "endpoint_time": 1.0, "endpoint_targets_per_full_batch": _validate_nonnegative_integer,
        "endpoint_targets_in_final_batch": _validate_nonnegative_integer,
        "endpoint_selection": "random_without_replacement",
        "noise_distribution": "independent_standard_normal_per_target",
        "randomness_assignment": "logical_target_order_before_packing",
    },
    "optimizer": {
        "type": "adamw", "learning_rate": _validate_positive_number, "beta1": _validate_unit_interval, "beta2": _validate_unit_interval,
        "epsilon": _validate_positive_number, "weight_decay": _parse_finite_number, "state_dtype": "float32",
        "max_grad_norm": _allow_none(_validate_positive_number),
    },
    "scheduler": {
        "type": "linear_warmup_constant", "warmup_steps": _validate_positive_integer,
        "first_update_warmup_fraction": _validate_positive_number, "reset_on_resume": False,
    },
    "validation": {
        "bundle": _validate_nonempty_string, "split_seed": _validate_seed, "expected_task_count": _validate_positive_integer,
        "expected_query_count": _validate_positive_integer, "cache_device": "cpu",
        "context": "all_original_demonstrations", "augmentation": False,
        "loss": {
            **_CADENCE, "timesteps": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], "noise_seed": _validate_seed,
            "uniform_proxy_weight": _validate_unit_interval, "endpoint_weight": _validate_unit_interval,
        },
        "sampling": {
            **_CADENCE, "expected_task_count": _validate_positive_integer,
            "expected_query_count": _validate_positive_integer, "seeds": [_validate_seed, _validate_seed],
            "steps": _validate_positive_integer, "solver": "euler", "time_schedule": "linear",
            "trajectory_dtype": "float32", "use_target_shape": False,
            "test_time_training": False, "test_time_augmentation": False, "voting": False,
            "task_success": "one_sample_index_solves_all_queries",
            "cell_accuracy_scope": "shape_correct_predictions",
        },
    },
    "logging": {
        "backend": "wandb", "entity": _allow_none(_validate_nonempty_string), "project": _allow_none(_validate_nonempty_string),
        "mode": "online", "api_key_environment_variable": "WANDB_API_KEY",
        "step_metric": "progress/optimizer_step", "every_optimizer_steps": 1,
        "aggregate_across_optimizer_steps": False, "batch_scalars_per_step": True,
        "automatic_system_metrics": False, "console_capture": False,
        "model_watch": False, "histograms": False,
        "train_metrics": _metric_names_validator(
            "train/loss train/mse_canvas train/mse_grid train/mse_boundary "
            "train/contribution_canvas train/contribution_grid train/contribution_boundary "
            "train/loss_t1 train/loss_uniform train/loss_demos_2 train/loss_demos_3 train/loss_demos_4 "
            "train/targets_t1 train/targets_uniform train/targets_demos_2 train/targets_demos_3 "
            "train/targets_demos_4 train/boundary_target_count train/learning_rate train/grad_norm "
            "train/loss_high_noise train/loss_lower_noise train/targets_high_noise train/targets_lower_noise "
            "train/grad_clipped train/step_seconds train/targets_per_second train/targets "
            "progress/optimizer_step progress/epoch_fraction progress/targets_seen"
        ),
        "workload_counters": {
            "phases": ["train", "val", "sample"],
            "names": _metric_names_validator("nvarc_tokens flux_text_tokens flux_image_tokens flux_joint_tokens flux_forward_calls"),
            "cumulative_suffix": "_total",
        },
        "validation_metrics": _metric_names_validator("val/loss val/loss_uniform_proxy val/target_count val/boundary_target_count val/seconds"),
        "validation_time_metrics": {
            "names": _metric_names_validator("loss mse_canvas mse_grid mse_boundary"),
            "suffixes": ["t_0_1", "t_0_3", "t_0_5", "t_0_7", "t_0_9", "t1"],
        },
        "sampling_metrics": _metric_names_validator(
            "sample/pass_at_1 sample/pass_at_2 sample/solved_tasks_at_1 sample/solved_tasks_at_2 "
            "sample/task_count sample/shape_accuracy sample/cell_accuracy sample/invalid_grid_rate "
            "sample/shape_correct_count sample/prediction_count sample/seconds"
        ),
        "checkpoint_metrics": _metric_names_validator("checkpoint/seconds checkpoint/size_bytes checkpoint/path"),
        "hardware": {
            "per_step_metrics": _metric_names_validator("system/gpu_allocated_bytes system/gpu_reserved_bytes system/gpu_peak_allocated_bytes"),
            "sample_interval_seconds": _validate_positive_number,
            "sampled_metrics": _metric_names_validator(
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
        "keep_last": _allow_none(_validate_positive_integer), "upload_weights_to_wandb": False,
    },
    "final_test": {"enabled": False, "checkpoint_selection": None},
    "benchmark": None,
}


def _derive_loop_counts(recipe: dict[str, Any]) -> None:
    """Add derived run and epoch counts in place without reading prepared data."""
    loop, data = recipe["loop"], recipe["data"]
    epochs, max_steps = loop["epochs"], loop["max_steps"]
    if (epochs is None) == (max_steps is None):
        raise ValueError("loop requires exactly one of epochs or max_steps; set the other to null")
    if epochs is not None and epochs != 1:
        raise ValueError("loop.epochs must be 1; prepare and continue the next augmentation epoch separately")
    per_sequence = data["pairs_per_sequence"] - loop["min_context_pairs_before_target"]
    if per_sequence <= 0:
        raise ValueError("data.pairs_per_sequence must exceed loop.min_context_pairs_before_target")
    batch = loop["optimizer_batch_size"]
    if batch % per_sequence:
        raise ValueError("loop.optimizer_batch_size must fit whole training sequences")
    task_count = data["expected_task_count"]
    if task_count is None and epochs is not None:
        raise ValueError("epoch runs require data.expected_task_count")
    if (task_count is None and recipe["models"]["initialization"] == "transformer_weights"
            and recipe["models"]["completed_epochs"] > 0):
        raise ValueError("completed epoch offsets require data.expected_task_count")
    if task_count is None and recipe["checkpointing"] is not None:
        raise ValueError("checkpointing requires data.expected_task_count for the resume cursor")
    full_targets = None if task_count is None else task_count * per_sequence
    epoch_steps = None if full_targets is None else (full_targets + batch - 1) // batch
    steps = epoch_steps if epochs is not None else max_steps
    if epoch_steps is not None and steps > epoch_steps:
        raise ValueError("loop.max_steps exceeds the prepared epoch; prepare the next epoch explicitly")
    targets = steps * batch if full_targets is None else min(steps * batch, full_targets)
    loop.update(max_optimizer_steps=steps, expected_target_count=targets,
                expected_final_batch_target_count=(targets - 1) % batch + 1,
                epoch_steps=epoch_steps)


def _parse_scheduler_config(schedule: Mapping[str, Any] | None, steps: int):
    """Validate schedule parameters and return settings, or None when explicitly disabled."""
    if schedule is None:
        return None
    if schedule["type"] == "cosine":
        if not 0 <= schedule["warmup_steps"] < schedule["total_steps"]:
            raise ValueError("scheduler requires 0 <= warmup_steps < total_steps")
        if steps > schedule["total_steps"]:
            raise ValueError("scheduler.total_steps must cover the run step limit")
        return CosineSchedulerConfig(schedule["warmup_steps"], schedule["total_steps"])
    if not math.isclose(schedule["first_update_warmup_fraction"], 1 / schedule["warmup_steps"], rel_tol=1e-12):
        raise ValueError("scheduler.first_update_warmup_fraction must equal 1 / warmup_steps")
    return LinearWarmupConstantConfig(schedule["warmup_steps"])


def _validate_config_relationships(recipe: dict[str, Any]) -> None:
    """Check compatibility across execution, batching, sampling, and service settings."""
    data, loop, flow = recipe["data"], recipe["loop"], recipe["flow"]

    execution = recipe["execution"]
    if execution["nvarc_early_exit_layer"] != execution["nvarc_hidden_layers"][-1]:
        raise ValueError("execution.nvarc_early_exit_layer must equal the highest selected NVARC layer")

    # Match the execution strategy to the GPU count, random sampling, and packing policy.
    strategy = recipe["execution"]["distributed_strategy"]
    if "fsdp_reshard_after_forward" in execution and strategy != "fsdp2":
        raise ValueError("execution.fsdp_reshard_after_forward requires distributed_strategy=fsdp2")
    if strategy == "single_process" and recipe["deployment"]["gpu_count"] != 1:
        raise ValueError("single_process execution requires deployment.gpu_count=1")
    if strategy in ("ddp", "fsdp2") and flow["rng_scheme"] != "global_step_target_v1":
        raise ValueError(f"{strategy} execution requires flow.rng_scheme=global_step_target_v1")
    if strategy == "fsdp2" and recipe["execution"]["flux_packing_algorithm"] != "legacy_bfd_v1":
        raise ValueError("fsdp2 execution requires execution.flux_packing_algorithm=legacy_bfd_v1")

    batch = loop["optimizer_batch_size"]
    steps = loop["max_optimizer_steps"]
    # Flow quotas describe full batches and the prepared epoch tail. A bounded
    # prefix does not change the last full batch into a different sampling policy.
    task_count = data["expected_task_count"]
    final_count = (loop["expected_final_batch_target_count"] if task_count is None else
                   (task_count * (data["pairs_per_sequence"] - loop["min_context_pairs_before_target"]) - 1) % batch + 1)
    for name, count in (("endpoint_targets_per_full_batch", batch),
                        ("endpoint_targets_in_final_batch", final_count)):
        if flow.get(name, 0) > count:
            raise ValueError(f"flow.{name} exceeds its batch target count")
    if final_count == batch and flow.get("endpoint_targets_in_final_batch", 0) != flow.get("endpoint_targets_per_full_batch", 0):
        raise ValueError("a full final batch must use the full-batch endpoint count")
    if "high_noise_min" in flow:
        if not 0 < flow["high_noise_min"] < flow["high_noise_max"]:
            raise ValueError("high-noise interval must lie strictly above zero and below one")
        for suffix, count in (("per_full_batch", batch), ("in_final_batch", final_count)):
            if flow[f"endpoint_targets_{suffix}"] + flow[f"high_noise_targets_{suffix}"] > count:
                raise ValueError("endpoint and high-noise quotas exceed the batch target count")
        if final_count == batch and flow["high_noise_targets_in_final_batch"] != flow["high_noise_targets_per_full_batch"]:
            raise ValueError("a full final batch must use the full-batch high-noise count")

    # Require a pinned transformer source and a valid parent run ID when logging is enabled.
    if recipe["models"]["initialization"] == "transformer_weights":
        import re
        source = recipe["models"]["transformer_source"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", source["repo_id"]):
            raise ValueError("transformer_source.repo_id must be an owner/repository")
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("transformer_source.revision must pin an immutable 40-character commit SHA")
        if (not re.fullmatch(r"[A-Za-z0-9_./-]+", source["subfolder"])
                or any(part in ("", ".", "..") for part in source["subfolder"].split("/"))):
            raise ValueError("transformer_source.subfolder must be a safe relative path")
        if recipe["logging"] is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", recipe["logging"]["parent_run_id"]):
            raise ValueError("logging.parent_run_id must identify a W&B run")

    # Require region weights to sum to one and cover grids without a boundary region.
    if recipe["loss"]["type"] == "region_weighted_velocity_mse":
        weights = recipe["loss"]["region_weights"]
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("loss.region_weights must sum to one")
        if weights["canvas"] + weights["grid"] <= 0:
            raise ValueError("loss needs positive weight outside the boundary for 30x30 grids")

    # Check the learning-rate schedule and benchmark duration against the run length.
    _parse_scheduler_config(recipe["scheduler"], steps)
    benchmark = recipe["benchmark"]
    if benchmark is not None and benchmark["warmup_steps"] + benchmark["measured_steps"] != steps:
        raise ValueError("benchmark warmup_steps + measured_steps must equal the run step limit")

    # The logging step metric must be included in the recorded training metrics.
    if recipe["logging"] is not None and recipe["logging"]["step_metric"] not in recipe["logging"]["train_metrics"]:
        raise ValueError("logging.train_metrics must include the step metric")

    # Cadences define when checkpoints and validation run: periodically, at start, or at end.
    # Replacing the last periodic event requires an enabled event at run end.
    cadences = [recipe["checkpointing"]] if recipe["checkpointing"] is not None else []
    if recipe["validation"] is not None:
        cadences.extend((recipe["validation"]["loss"], recipe["validation"]["sampling"]))
    for cadence in cadences:
        if cadence["replace_last_periodic_with_final"] and not cadence["at_end"]:
            raise ValueError("replace_last_periodic_with_final requires at_end")

    # If validation is enabled, its sampling subset must fit the available tasks and queries.
    validation = recipe["validation"]
    if validation is None:
        return
    sampling, loss = validation["sampling"], validation["loss"]
    if validation["expected_query_count"] < validation["expected_task_count"]:
        raise ValueError("validation must contain at least one query per task")
    if not sampling["expected_task_count"] <= sampling["expected_query_count"] <= validation["expected_query_count"]:
        raise ValueError("sampling query counts must fit the validation pool")
    if sampling["expected_task_count"] > validation["expected_task_count"]:
        raise ValueError("sampling tasks must fit the validation pool")
    if len(set(sampling["seeds"])) != 2:
        raise ValueError("validation.sampling.seeds must be distinct")

    # Match validation loss weights to the flow mixture, except for the high-noise variant below.
    endpoint_fraction = flow.get("endpoint_targets_per_full_batch", 0) / batch
    if "high_noise_min" in flow:
        # Keep the original evaluation mixture when changing the training distribution.
        if not math.isclose(loss["endpoint_weight"] + loss["uniform_proxy_weight"], 1.0, abs_tol=1e-12):
            raise ValueError("validation.loss weights must sum to one")
    elif not (math.isclose(loss["endpoint_weight"], endpoint_fraction, rel_tol=0, abs_tol=1e-12)
            and math.isclose(loss["uniform_proxy_weight"], 1 - endpoint_fraction, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("validation.loss weights must match the full-batch flow mixture")


def _compute_event_steps(cadence: Mapping[str, Any] | None, total: int) -> tuple[int, ...]:
    """Compute event update indices, including optional start/end checks; None disables events."""
    if cadence is None:
        return ()
    steps = list(range(cadence["every_optimizer_steps"], total + 1, cadence["every_optimizer_steps"]))
    if cadence["at_end"] and (not steps or steps[-1] != total):
        if cadence["replace_last_periodic_with_final"] and steps:
            steps.pop()
        steps.append(total)
    if cadence["at_start"]:
        steps.insert(0, 0)
    return tuple(steps)


def _validate_against_schema(value: Any, spec: Any, *, name: str) -> Any:
    """Recursively enforce required fields, validators, and fixed literals without filling defaults."""
    if isinstance(spec, dict):
        value = _validate_json_object(value, name=name)
        _validate_required_fields(value, set(spec), name=name)
        return {key: _validate_against_schema(value[key], child, name=f"{name}.{key}") for key, child in spec.items()}
    if isinstance(spec, list):
        if not isinstance(value, list) or len(value) != len(spec):
            raise ValueError(f"{name} must be an array of length {len(spec)}")
        return [_validate_against_schema(item, child, name=f"{name}[{index}]") for index, (item, child) in enumerate(zip(value, spec))]
    if spec is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value
    if callable(spec):
        return spec(value, name=name)
    if isinstance(spec, float):
        value = _parse_finite_number(value, name=name)
    if type(value) is not type(spec) or value != spec:
        raise ValueError(f"{name} must be {spec!r}")
    return value


def _reject_duplicate_json_fields(pairs):
    """JSON object_pairs_hook that rejects duplicate keys before they can be overwritten."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config field: {key}")
        result[key] = value
    return result


def _freeze_config_values(value: Any) -> Any:
    """Recursively copy dictionaries to read-only mappings and lists to tuples."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_config_values(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_config_values(item) for item in value)
    return value


__all__ = [
    "TRAINING_SCHEMA_VERSION", "AdamWConfig", "BenchmarkConfig", "CosineSchedulerConfig",
    "LinearWarmupConstantConfig", "DataConfig", "LoopConfig", "ModelConfig",
    "TrainingConfig", "load_training_config", "parse_training_config",
]
