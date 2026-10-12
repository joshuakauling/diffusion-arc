"""Select a benchmark view, timing window and execution settings for a training recipe."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

from .config import (
    TrainingConfig, load_training_config, parse_training_config,
    _choice_validator, _reject_duplicate_json_fields,
    _resolve_config_path, _validate_against_schema, _validate_nonempty_string,
    _validate_json_object, _validate_nonnegative_integer, _validate_positive_integer,
)


_SCHEMA = {
    "config_type": "ddp_benchmark",
    "schema_version": 1,
    "training_recipe": _validate_nonempty_string,
    "execution": {
        "distributed_strategy": _choice_validator("single_process", "ddp", "fsdp2"),
        "gpu_count": _validate_positive_integer,
        "flux_packing_algorithm": _choice_validator("legacy_bfd_v1", "balanced_release_v1"),
        "rng_scheme": _choice_validator("legacy_stream_v1", "global_step_target_v1"),
    },
    "data": {
        "split_directory": _validate_nonempty_string,
        "view_name": _validate_nonempty_string,
        "expected_source_count": _validate_positive_integer,
    },
    "repetitions": _validate_positive_integer,
    "smoke_steps": _validate_positive_integer,
    "warmup_steps": _validate_nonnegative_integer,
    "measured_steps": _validate_positive_integer,
}


@dataclass(frozen=True)
class TrainingBenchmarkConfig:
    """A bounded run with global batch size and the original training schedule.

    The wrapper explicitly selects its execution strategy and benchmark view.
    The recipe retains the original training schedule and objective; its epoch
    counts and validation/checkpoint cadences are not benchmark instructions.
    Optional precomputed packing changes only this bounded benchmark's batch plans.
    """

    training: TrainingConfig
    expected_source_count: int
    repetitions: int
    smoke_steps: int
    warmup_steps: int
    measured_steps: int
    path: Path
    precompute_packing: bool = False
    fsdp_gradient_sync: str = "last_microbatch"

    @property
    def benchmark_steps(self) -> int:
        return self.warmup_steps + self.measured_steps


def load_training_benchmark_config(path: str | Path) -> TrainingBenchmarkConfig:
    """Validate the benchmark contract without loading checkpoints or data.

    All paths are relative to the file that declares them. The referenced
    training_run is loaded normally, then explicit execution choices are validated
    and the data view is selected. Optional pack-capacity and FSDP grouping/resharding
    overrides retain the recipe's settings when omitted. Precomputed packing
    requires prepared token metadata and indexed flow RNG. Timing warmup never
    changes the LR schedule.
    """

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        payload = _validate_json_object(
            json.load(handle, object_pairs_hook=_reject_duplicate_json_fields), name="config",
        )
    schema = deepcopy(_SCHEMA)
    if "precompute_packing" in payload:
        schema["precompute_packing"] = bool
    if "fsdp_gradient_sync" in payload:
        schema["fsdp_gradient_sync"] = _choice_validator("last_microbatch", "per_microbatch")
    optional_execution = {"flux_max_joint_tokens": _validate_positive_integer,
                          "fsdp_reshard_after_forward": bool,
                          "fsdp_blocks_per_group": _validate_positive_integer,
                          "rank_batch_assignment": _choice_validator(
                              "contiguous_v1", "token_balanced_v1", "joint_packing_v1",
                          )}
    if isinstance(payload.get("execution"), dict):
        for name, validator in optional_execution.items():
            if name in payload["execution"]:
                schema["execution"][name] = validator
    payload = _validate_against_schema(payload, schema, name="config")
    recipe_path = _resolve_config_path(
        payload["training_recipe"], base_directory=config_path.parent, name="training_recipe",
    )
    training = load_training_config(recipe_path, expected_type="training_run")
    if training.recipe["models"]["resume_from"] is not None:
        raise ValueError("DDP benchmarks require base checkpoints; resume_from must be null")
    execution = payload["execution"]
    if "fsdp_gradient_sync" in payload and execution["distributed_strategy"] != "fsdp2":
        raise ValueError("fsdp_gradient_sync requires distributed_strategy=fsdp2")
    recipe = training.recipe_dict()
    recipe["execution"].update(
        distributed_strategy=execution["distributed_strategy"],
        flux_packing_algorithm=execution["flux_packing_algorithm"],
    )
    for name in optional_execution:
        if name in execution:
            recipe["execution"][name] = execution[name]
    recipe["deployment"]["gpu_count"] = execution["gpu_count"]
    recipe["flow"]["rng_scheme"] = execution["rng_scheme"]
    training = parse_training_config(recipe, config_path=recipe_path)
    data = payload["data"]
    training = replace(training, data=replace(
        training.data,
        split_directory=_resolve_config_path(
            data["split_directory"], base_directory=config_path.parent, name="data.split_directory",
        ),
        view_name=data["view_name"],
    ))
    config = TrainingBenchmarkConfig(
        training=training, expected_source_count=data["expected_source_count"],
        repetitions=payload["repetitions"],
        smoke_steps=payload["smoke_steps"], warmup_steps=payload["warmup_steps"],
        measured_steps=payload["measured_steps"], path=config_path,
        precompute_packing=payload.get("precompute_packing", False),
        fsdp_gradient_sync=payload.get("fsdp_gradient_sync", "last_microbatch"),
    )
    if config.precompute_packing:
        if training.recipe["execution"]["distributed_strategy"] not in ("ddp", "fsdp2"):
            raise ValueError("precompute_packing requires distributed_strategy=ddp or fsdp2")
        if training.recipe["flow"]["rng_scheme"] != "global_step_target_v1":
            raise ValueError("precompute_packing requires global_step_target_v1 flow RNG")
        if training.rank_batch_assignment != "token_balanced_v1":
            raise ValueError("precompute_packing requires rank_batch_assignment=token_balanced_v1")
        if training.flux_packing.algorithm != "legacy_bfd_v1":
            raise ValueError("precompute_packing requires flux_packing_algorithm=legacy_bfd_v1")
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


__all__ = ["TrainingBenchmarkConfig", "load_training_benchmark_config"]
