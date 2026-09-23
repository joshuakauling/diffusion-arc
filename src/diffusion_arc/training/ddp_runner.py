"""Compatibility entry for bounded DDP experiments on the common runner."""

from dataclasses import replace

from .config import TrainingConfig
from .distributed import distributed_runtime
from .runner import RunProgress, run_training
from .run.experiment import TrainingExperiment, print_record


# Retain the old import name for callers that only inspect global counters.
DistributedProgress = RunProgress


def run_training_ddp(config: TrainingConfig, *, max_steps=3, emit=print_record,
                     source_identity=None, expected_source_count=None, measurement=None,
                     diagnostic_batch=None, trace_directory=None, memory_history=False,
                     memory_snapshot_step=None, profile_warmup_steps=None):
    """Preserve the historical benchmark launch contract, recording its overrides.

    Full Single-GPU/DDP epochs use run_training directly with an explicit recipe.
    This measurement entry retains the original DDP RNG and packing settings for
    older benchmark recipes; validation, logging and saves are explicitly omitted.
    """
    if config.config_type != "training_run" or config.recipe is None:
        raise ValueError("run_training_ddp requires a training_run recipe")
    experiment = TrainingExperiment(max_steps, emit, source_identity, expected_source_count,
        measurement, diagnostic_batch, trace_directory, memory_history, memory_snapshot_step,
        profile_warmup_steps)
    experiment.validate(config)
    experiment.input_recipe = config.recipe_dict()
    with distributed_runtime(config) as runtime:
        recipe = config.recipe_dict()
        recipe["execution"].update(distributed_strategy="ddp", flux_packing_algorithm="balanced_release_v1")
        recipe["deployment"]["gpu_count"] = runtime.world_size
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        effective = replace(config, recipe=recipe,
                            flux_packing=replace(config.flux_packing, algorithm="balanced_release_v1"))
        return run_training(effective, experiment=experiment, runtime=runtime)
