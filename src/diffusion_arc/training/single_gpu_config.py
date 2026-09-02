"""Strict JSON configuration for the initial single-GPU training entrypoint."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SINGLE_GPU_TRAINING_SCHEMA_VERSION = 1
_SUPPORTED_DTYPES = frozenset({"bfloat16", "float32"})


@dataclass(frozen=True)
class SingleGpuModelConfig:
    """Local checkpoint paths and explicit model precision choices."""

    nvarc_checkpoint: Path
    flux_checkpoint: Path
    nvarc_dtype: str
    vae_dtype: str
    flux_parameter_dtype: str
    flux_compute_dtype: str


@dataclass(frozen=True)
class SingleGpuDataConfig:
    """One finite stream of individual ARC training sequences."""

    split_directory: Path
    view_name: str
    shuffle: bool
    num_workers: int


@dataclass(frozen=True)
class SingleGpuLoopConfig:
    """Target and microbatch limits owned by the single-GPU loop."""

    max_optimizer_steps: int
    optimizer_batch_size: int
    flux_microbatch_size: int
    min_context_pairs_before_target: int


@dataclass(frozen=True)
class AdamWConfig:
    """Explicit AdamW settings for the trainable FLUX transformer."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float


@dataclass(frozen=True)
class SingleGpuTrainingConfig:
    """Complete validated configuration for one initial training process."""

    seed: int
    device: str
    models: SingleGpuModelConfig
    data: SingleGpuDataConfig
    loop: SingleGpuLoopConfig
    optimizer: AdamWConfig


def load_single_gpu_training_config(
    path: str | Path,
) -> SingleGpuTrainingConfig:
    """Load one strict config and resolve its paths relative to the JSON file."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = _object(json.load(handle), name="config")

    _fields(
        payload,
        {"schema_version", "seed", "device", "models", "data", "loop", "optimizer"},
        name="config",
    )
    schema_version = _integer(payload["schema_version"], name="schema_version")
    if schema_version != SINGLE_GPU_TRAINING_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")

    seed = _integer(payload["seed"], name="seed")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must lie in [0, 2**64)")

    device = _string(payload["device"], name="device")
    device_parts = device.split(":")
    if not (
        device_parts[0] == "cuda"
        and len(device_parts) <= 2
        and (len(device_parts) == 1 or device_parts[1].isdigit())
    ):
        raise ValueError("device must be 'cuda' or 'cuda:<index>'")

    base_directory = config_path.parent
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

    models = SingleGpuModelConfig(
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

    data_payload = _object(payload["data"], name="data")
    _fields(
        data_payload,
        {"split_directory", "view_name", "shuffle", "num_workers"},
        name="data",
    )
    shuffle = data_payload["shuffle"]
    if not isinstance(shuffle, bool):
        raise TypeError("data.shuffle must be a boolean")
    num_workers = _integer(data_payload["num_workers"], name="data.num_workers")
    if num_workers != 0:
        raise ValueError("initial single-GPU training requires data.num_workers = 0")
    data = SingleGpuDataConfig(
        split_directory=_path(
            data_payload["split_directory"],
            base_directory=base_directory,
            name="data.split_directory",
        ),
        view_name=_string(data_payload["view_name"], name="data.view_name"),
        shuffle=shuffle,
        num_workers=num_workers,
    )

    loop_payload = _object(payload["loop"], name="loop")
    loop_fields = {
        "max_optimizer_steps",
        "optimizer_batch_size",
        "flux_microbatch_size",
        "min_context_pairs_before_target",
    }
    _fields(loop_payload, loop_fields, name="loop")
    loop_values = {
        field: _positive_integer(loop_payload[field], name=f"loop.{field}")
        for field in loop_fields
    }
    loop = SingleGpuLoopConfig(**loop_values)

    optimizer_payload = _object(payload["optimizer"], name="optimizer")
    _fields(
        optimizer_payload,
        {"learning_rate", "beta1", "beta2", "epsilon", "weight_decay"},
        name="optimizer",
    )
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

    return SingleGpuTrainingConfig(
        seed=seed,
        device=device,
        models=models,
        data=data,
        loop=loop,
        optimizer=AdamWConfig(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
        ),
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


__all__ = [
    "SINGLE_GPU_TRAINING_SCHEMA_VERSION",
    "AdamWConfig",
    "SingleGpuDataConfig",
    "SingleGpuLoopConfig",
    "SingleGpuModelConfig",
    "SingleGpuTrainingConfig",
    "load_single_gpu_training_config",
]
