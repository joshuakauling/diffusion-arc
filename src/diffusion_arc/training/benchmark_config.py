"""Select a fixed benchmark view and timing window for an unchanged training recipe."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from .config import (
    TrainingConfig, _natural, _path, _positive_integer, _string, _unique_fields,
    _validate, load_training_config,
)


_SCHEMA = {
    "config_type": "ddp_benchmark",
    "schema_version": 1,
    "training_recipe": _string,
    "data": {
        "split_directory": _string,
        "view_name": _string,
        "expected_source_count": _positive_integer,
    },
    "repetitions": _positive_integer,
    "smoke_steps": _positive_integer,
    "warmup_steps": _natural,
    "measured_steps": _positive_integer,
}


@dataclass(frozen=True)
class DdpBenchmarkConfig:
    """A bounded run with global batch size and the original training schedule.

    Only ``training.data`` points at the benchmark view. ``training.recipe`` and
    ``training.path`` retain the complete input recipe for provenance; its epoch
    counts and validation/checkpoint cadences are not benchmark instructions.
    """

    training: TrainingConfig
    expected_source_count: int
    repetitions: int
    smoke_steps: int
    warmup_steps: int
    measured_steps: int
    path: Path

    @property
    def benchmark_steps(self) -> int:
        return self.warmup_steps + self.measured_steps


def load_ddp_benchmark_config(path: str | Path) -> DdpBenchmarkConfig:
    """Validate the benchmark contract without loading checkpoints or data.

    All paths are relative to the file that declares them. The referenced
    training_run is loaded normally, then only its normalized data selection is
    replaced. Timing warmup never changes the learning-rate warmup schedule.
    """

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = _validate(
            json.load(handle, object_pairs_hook=_unique_fields), _SCHEMA, name="config",
        )
    recipe_path = _path(
        payload["training_recipe"], base_directory=config_path.parent, name="training_recipe",
    )
    training = load_training_config(recipe_path, expected_type="training_run")
    if training.recipe["models"]["resume_from"] is not None:
        raise ValueError("DDP benchmarks require base checkpoints; resume_from must be null")
    data = payload["data"]
    training = replace(training, data=replace(
        training.data,
        split_directory=_path(
            data["split_directory"], base_directory=config_path.parent, name="data.split_directory",
        ),
        view_name=data["view_name"],
    ))
    config = DdpBenchmarkConfig(
        training=training, expected_source_count=data["expected_source_count"],
        repetitions=payload["repetitions"],
        smoke_steps=payload["smoke_steps"], warmup_steps=payload["warmup_steps"],
        measured_steps=payload["measured_steps"], path=config_path,
    )
    if config.smoke_steps > config.benchmark_steps:
        raise ValueError("smoke_steps must not exceed warmup_steps + measured_steps")
    if config.benchmark_steps > training.loop.max_optimizer_steps:
        raise ValueError("benchmark steps must not exceed the training recipe's max_optimizer_steps")
    if (config.expected_source_count * training.targets_per_sequence
            != config.benchmark_steps * training.loop.optimizer_batch_size):
        raise ValueError(
            "expected_source_count * targets_per_sequence must equal "
            "benchmark steps * optimizer_batch_size; the complete view must fill full batches"
        )
    return config


__all__ = ["DdpBenchmarkConfig", "load_ddp_benchmark_config"]
