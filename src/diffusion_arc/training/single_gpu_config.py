"""Strict JSON configuration for the initial single-GPU training entrypoint."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffusion_arc._constants import NVARC_MAX_TOKENS
from diffusion_arc.sequence_buckets import SequenceLengthBuckets
from .nvarc_schedule import NvarcBatchShape, NvarcSchedulingPolicy
from .flux_schedule import FluxBatchShape, FluxSchedulingPolicy


SINGLE_GPU_TRAINING_SCHEMA_VERSION = 6
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4, 5, SINGLE_GPU_TRAINING_SCHEMA_VERSION})
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
    prefetch_factor: int | None = None


@dataclass(frozen=True)
class SingleGpuLoopConfig:
    """Loop limits; nvarc_source_batch_size is the fixed preparation window."""

    max_optimizer_steps: int
    optimizer_batch_size: int
    flux_microbatch_size: int | None
    min_context_pairs_before_target: int
    nvarc_source_batch_size: int = 1
    vae_microbatch_size: int | None = None


@dataclass(frozen=True)
class AdamWConfig:
    """Explicit AdamW settings for the trainable FLUX transformer."""

    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float


@dataclass(frozen=True)
class CosineSchedulerConfig:
    """Linear LR warm-up followed by cosine decay to zero."""

    warmup_steps: int
    total_steps: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Timing windows, independent of the learning-rate schedule."""

    warmup_steps: int
    measured_steps: int
    repetitions: int


@dataclass(frozen=True)
class SingleGpuPaddingConfig:
    """Independent fixed text-length boundaries; None preserves exact padding."""

    nvarc_bucket_boundaries: tuple[int, ...] | None = None
    flux_bucket_boundaries: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SingleGpuTrainingConfig:
    """Complete validated configuration for one initial training process."""

    seed: int
    device: str
    models: SingleGpuModelConfig
    data: SingleGpuDataConfig
    loop: SingleGpuLoopConfig
    optimizer: AdamWConfig
    scheduler: CosineSchedulerConfig | None = None
    benchmark: BenchmarkConfig | None = None
    padding: SingleGpuPaddingConfig = SingleGpuPaddingConfig()
    nvarc_scheduling: NvarcSchedulingPolicy | None = None
    flux_scheduling: FluxSchedulingPolicy | None = None


def load_single_gpu_training_config(
    path: str | Path,
) -> SingleGpuTrainingConfig:
    """Load one strict config and resolve its paths relative to the JSON file."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = _object(json.load(handle), name="config")

    schema_version = _integer(payload.get("schema_version"), name="schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    config_fields = {"schema_version", "seed", "device", "models", "data", "loop", "optimizer"}
    if schema_version >= 3:
        config_fields.update({"scheduler", "benchmark"})
    if schema_version >= 4:
        config_fields.add("padding")
    if schema_version >= 5:
        config_fields.add("nvarc_scheduling")
    if schema_version >= 6:
        config_fields.add("flux_scheduling")
    _fields(payload, config_fields, name="config")

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
    data_fields = {"split_directory", "view_name", "shuffle", "num_workers"}
    if schema_version >= 5:
        data_fields.add("prefetch_factor")
    _fields(
        data_payload,
        data_fields,
        name="data",
    )
    shuffle = data_payload["shuffle"]
    if not isinstance(shuffle, bool):
        raise TypeError("data.shuffle must be a boolean")
    num_workers = _integer(data_payload["num_workers"], name="data.num_workers")
    if schema_version < 5 and num_workers != 0:
        raise ValueError("initial single-GPU training requires data.num_workers = 0")
    if num_workers < 0:
        raise ValueError("data.num_workers must not be negative")
    prefetch_factor = data_payload.get("prefetch_factor")
    if num_workers:
        prefetch_factor = _positive_integer(prefetch_factor, name="data.prefetch_factor")
    elif prefetch_factor is not None:
        raise ValueError("data.prefetch_factor must be null when num_workers = 0")
    data = SingleGpuDataConfig(
        split_directory=_path(
            data_payload["split_directory"],
            base_directory=base_directory,
            name="data.split_directory",
        ),
        view_name=_string(data_payload["view_name"], name="data.view_name"),
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    loop_payload = _object(payload["loop"], name="loop")
    loop_fields = {
        "max_optimizer_steps",
        "optimizer_batch_size",
        "flux_microbatch_size",
        "min_context_pairs_before_target",
    }
    if schema_version >= 2:
        loop_fields.update(
            {
                "nvarc_source_batch_size",
                "vae_microbatch_size",
            }
        )
    _fields(loop_payload, loop_fields, name="loop")
    loop_values = {
        field: _positive_integer(loop_payload[field], name=f"loop.{field}")
        for field in loop_fields
        if field != "flux_microbatch_size"
    }
    flux_microbatch_size = loop_payload["flux_microbatch_size"]
    if schema_version < 6 or flux_microbatch_size is not None:
        flux_microbatch_size = _positive_integer(flux_microbatch_size, name="loop.flux_microbatch_size")
    loop = SingleGpuLoopConfig(
        max_optimizer_steps=loop_values["max_optimizer_steps"],
        optimizer_batch_size=loop_values["optimizer_batch_size"],
        flux_microbatch_size=flux_microbatch_size,
        min_context_pairs_before_target=loop_values[
            "min_context_pairs_before_target"
        ],
        nvarc_source_batch_size=loop_values.get("nvarc_source_batch_size", 1),
        vae_microbatch_size=loop_values.get("vae_microbatch_size"),
    )

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

    padding = SingleGpuPaddingConfig()
    if schema_version >= 4:
        section = _object(payload["padding"], name="padding")
        fields = {"nvarc_bucket_boundaries", "flux_bucket_boundaries"}
        _fields(section, fields, name="padding")
        padding = SingleGpuPaddingConfig(**{
            field: _bucket_boundaries(section[field], name=f"padding.{field}")
            for field in fields
        })

    nvarc_scheduling = None
    if payload.get("nvarc_scheduling") is not None:
        section = _object(payload["nvarc_scheduling"], name="nvarc_scheduling")
        _fields(section, {"shapes", "call_overhead_tokens"}, name="nvarc_scheduling")
        if not isinstance(section["shapes"], list):
            raise TypeError("nvarc_scheduling.shapes must be an array")
        shapes = []
        for value in section["shapes"]:
            shape = _object(value, name="NVARC shape")
            _fields(shape, {"batch_size", "max_tokens"}, name="NVARC shape")
            shapes.append(NvarcBatchShape(**shape))
        nvarc_scheduling = NvarcSchedulingPolicy(
            shapes=tuple(shapes),
            call_overhead_tokens=section["call_overhead_tokens"],
        )
        if NvarcBatchShape(2, NVARC_MAX_TOKENS) not in nvarc_scheduling.shapes:
            raise ValueError(f"nvarc_scheduling must cover {NVARC_MAX_TOKENS} tokens")
        if padding.nvarc_bucket_boundaries is not None:
            raise ValueError("nvarc_scheduling and padding.nvarc_bucket_boundaries are mutually exclusive")

    flux_scheduling = None
    if payload.get("flux_scheduling") is not None:
        section = _object(payload["flux_scheduling"], name="flux_scheduling")
        _fields(section, {"shapes"}, name="flux_scheduling")
        if not isinstance(section["shapes"], list):
            raise TypeError("flux_scheduling.shapes must be an array")
        shapes = []
        for value in section["shapes"]:
            shape = _object(value, name="FLUX shape")
            _fields(shape, {"batch_size", "max_tokens", "forward_backward_ms"}, name="FLUX shape")
            shapes.append(FluxBatchShape(**shape))
        flux_scheduling = FluxSchedulingPolicy(tuple(shapes))
        if max(shape.max_tokens for shape in flux_scheduling.shapes) != NVARC_MAX_TOKENS:
            raise ValueError(f"flux_scheduling must cover {NVARC_MAX_TOKENS} tokens")
        if padding.flux_bucket_boundaries is not None or flux_microbatch_size is not None:
            raise ValueError("flux_scheduling requires null flux_microbatch_size and flux_bucket_boundaries")
    elif flux_microbatch_size is None:
        raise ValueError("loop.flux_microbatch_size must be positive without flux_scheduling")

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
        scheduler=scheduler,
        benchmark=benchmark,
        padding=padding,
        nvarc_scheduling=nvarc_scheduling,
        flux_scheduling=flux_scheduling,
    )


def _bucket_boundaries(value: Any, *, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array or null")
    boundaries = tuple(_integer(item, name=name) for item in value)
    SequenceLengthBuckets(boundaries)
    if boundaries[-1] != NVARC_MAX_TOKENS:
        raise ValueError(f"{name} must end at {NVARC_MAX_TOKENS}")
    return boundaries


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
    "BenchmarkConfig",
    "CosineSchedulerConfig",
    "SingleGpuDataConfig",
    "SingleGpuLoopConfig",
    "SingleGpuModelConfig",
    "SingleGpuPaddingConfig",
    "SingleGpuTrainingConfig",
    "load_single_gpu_training_config",
]
