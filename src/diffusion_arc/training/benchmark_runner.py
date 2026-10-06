"""Bounded measurement entry point using the shared training loop."""

from contextlib import ExitStack

from .training_loop import run_training_loop, runtime_context
from .config import TrainingConfig
from .run.experiment import BenchmarkInstrumentation, print_record
from .run.state import RunProgress


class _BenchmarkCallbacks:
    """Adapt measurement hooks without enabling training services."""

    def __init__(self, config, experiment):
        self.config, self.experiment = config, experiment

    def open(self, stack, runtime):
        self.experiment.open(stack, runtime)

    def prepare(self, stack, models, dataset, identity, validation_manifest):
        self.experiment.start(stack, self.config, models, dataset)

    def initial(self):
        pass

    def before_step(self):
        self.experiment.before_step()

    def completed_step(self):
        self.experiment.completed_step()

    def after_step(self, run_step, batch_plan, local_result, result, progress, **context):
        self.experiment.after_step(run_step, batch_plan, local_result, progress)

    def finish(self, progress):
        self.experiment.finish(progress)


def run_benchmark(config, *, experiment: BenchmarkInstrumentation, runtime=None):
    """Measure explicitly configured training behavior without evaluation or saves."""
    experiment.validate(config)
    if experiment.diagnostic_batch_number is not None and config.recipe["flow"]["rng_scheme"] != "global_step_target_v1":
        raise ValueError("repeated diagnostic batches require global_step_target_v1 flow RNG")
    with runtime_context(config, runtime) as runtime, ExitStack() as stack:
        return run_training_loop(config, runtime=runtime, stack=stack,
            callbacks=_BenchmarkCallbacks(config, experiment), progress=RunProgress(), experiment=experiment)


def run_training_benchmark(config: TrainingConfig, *, max_steps=3, emit=print_record,
                     source_identity=None, expected_source_count=None, measurement=None,
                     diagnostic_batch_number=None, trace_directory=None, memory_history=False,
                     profile_warmup_steps=None):
    """Run the configured strategy without implicit recipe or precision overrides."""
    experiment = BenchmarkInstrumentation(max_steps, emit, source_identity, expected_source_count,
        measurement, diagnostic_batch_number, trace_directory, memory_history,
        profile_warmup_steps)
    return run_benchmark(config, experiment=experiment)


__all__ = ["run_benchmark", "run_training_benchmark"]
