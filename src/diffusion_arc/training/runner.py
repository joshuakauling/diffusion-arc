"""Run a shared single-GPU/DDP/FSDP2 epoch or bounded training experiment."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import partial
import time

import torch

from .config import TrainingConfig
from .training_loop import run_training_loop, runtime_context, WORKLOAD_COUNTERS
from .distributed import TrainingRuntime
from .run.state import RunProgress, aggregate_results
from .run.checkpointing import (Checkpoint, load_checkpoint_metadata, save_checkpoint, validate_epoch_continuation)
from .run.logging import WandbLogger, build_sampling_metrics, build_training_metrics, build_validation_metrics
from .step.loss import RegionLossWeights
from .run.validation import (
    evaluate_packed_loss, generate_packed_samples, load_sampling_input_grids,
    plan_validation_packs, prepare_validation_cache,
    tokenize_validation_targets, validate_validation_bundle,
)


def _prepare_training_start(config, previous_epoch_checkpoint, parent_run_metadata):
    """Determine initial progress, checkpoint metadata, and the logging run to resume."""
    recipe = config.recipe
    epoch_steps = config.epoch_optimizer_steps or config.loop.max_optimizer_steps
    resume_from = recipe["models"]["resume_from"]
    if previous_epoch_checkpoint is not None:
        if resume_from is not None:
            raise ValueError("choose either an in-epoch resume or a new epoch")
        validate_epoch_continuation(previous_epoch_checkpoint, config)
    checkpoint = load_checkpoint_metadata(resume_from, config) if resume_from else previous_epoch_checkpoint
    progress = RunProgress(**checkpoint.metadata["progress"]) if checkpoint else RunProgress()
    weights_start = recipe["models"]["initialization"] == "transformer_weights"
    if weights_start and not checkpoint:
        if recipe["models"]["transformer_checkpoint"] is None:
            raise ValueError("prepare the initial transformer checkpoint before training")
        completed = recipe["models"]["completed_epochs"]
        progress = RunProgress(
            optimizer_step=completed * epoch_steps,
            sources_seen=completed * (recipe["data"]["expected_task_count"] or 0),
            targets_seen=completed * (recipe["data"]["expected_task_count"] or 0) * config.targets_per_sequence, epoch=completed + 1,
            optimizer_start_step=completed * epoch_steps,
            workload_totals=dict(parent_run_metadata["workload_totals"]) if parent_run_metadata else {})
        if recipe["logging"] is not None and recipe["logging"].get("resume_parent_run", False) and parent_run_metadata is None:
            raise ValueError("verify the parent W&B run before continuing its history")
    if previous_epoch_checkpoint is not None:
        progress.epoch += 1
    if progress.optimizer_step >= (progress.epoch - 1) * epoch_steps + config.loop.max_optimizer_steps:
        raise ValueError("checkpoint already reached this run's step limit")
    resume_run = checkpoint.metadata["wandb_run"] if checkpoint else (
        parent_run_metadata["wandb_run"] if weights_start and recipe["logging"] is not None and recipe["logging"].get("resume_parent_run", False) else None)
    return progress, checkpoint, resume_run


def _load_validation_manifest(config):
    """Read prepared validation metadata and check the configured task and query counts."""
    validation = config.recipe["validation"]
    validation_manifest = validate_validation_bundle(validation["bundle"], split_seed=validation["split_seed"],
                                          verify_payloads=False)
    if (len(validation_manifest["validation_task_ids"]) != validation["expected_task_count"]
            or validation_manifest["validation_output_count"] != validation["expected_query_count"]):
        raise ValueError("validation bundle task/query counts differ from the run recipe")
    sampling = validation["sampling"]
    if (len(validation_manifest["sampling_task_ids"]) != sampling["expected_task_count"]
            or validation_manifest["sampling_output_count"] != sampling["expected_query_count"]):
        raise ValueError("sampling bundle task/query counts differ from the run recipe")
    return validation_manifest


class _TrainingCallbacks:
    """Keep logging on rank zero and run sharded evaluation on all FSDP ranks."""

    def __init__(self, config, runtime, progress, checkpoint, resume_run, previous_epoch_checkpoint):
        self.config, self.runtime, self.progress = config, runtime, progress
        self.checkpoint, self.resume_run, self.previous_epoch_checkpoint = checkpoint, resume_run, previous_epoch_checkpoint
        self.logger = None

    def open(self, stack, runtime):
        self.wandb_run = None
        if self.config.recipe["logging"] is None:
            return
        @contextmanager
        def logging_context():
            with self.logger:
                yield
        def enter():
            self.logger = WandbLogger(self.config, device=self.runtime.device, resume_run=self.resume_run)
            stack.enter_context(logging_context())
            return self.logger.run_identity
        self.wandb_run = self.runtime.run_on_primary_and_broadcast(enter)

    def prepare(self, stack, models, dataset, identity, validation_manifest):
        if self.runtime.strategy == "fsdp2":
            self._prepare(models, validation_manifest, identity)
        else:
            self.runtime.run_on_primary_and_broadcast(lambda: self._prepare(models, validation_manifest, identity))

    def _prepare(self, models, validation_manifest, identity):
        config, progress = self.config, self.progress
        recipe, validation = config.recipe, config.recipe["validation"]
        if self.logger is not None and recipe["models"]["initialization"] == "transformer_weights" and not self.checkpoint:
            self.logger.run.config.update({f"epoch_{progress.epoch}": {
                "recipe": config.recipe_dict(), "optimizer_reset": True, "start_step": progress.optimizer_step,
                "data_sha256": identity["data_sha256"]}}, allow_val_change=True)
        if self.logger is not None and self.previous_epoch_checkpoint is not None:
            self.logger.run.config.update({f"epoch_{progress.epoch}": {
                "augmentation_seed": config.seed, "parent_checkpoint": str(self.previous_epoch_checkpoint.path),
                "data_sha256": identity["data_sha256"], "start_step": progress.optimizer_step}}, allow_val_change=True)
        if validation is None:
            return
        sampling = validation["sampling"]
        samples, grids = tokenize_validation_targets(validation["bundle"], validation_manifest, models.tokenizer)
        targets = prepare_validation_cache(samples, grids, nvarc_encoder=models.nvarc_encoder,
            vae_encoder=models.vae_encoder, nvarc_max_tokens=config.nvarc_max_tokens,
            vae_batch_size=config.loop.vae_microbatch_size)
        packs = plan_validation_packs(targets, max_joint_tokens=config.flux_packing.max_tokens)
        self.preparation_tokens = sum(len(sample.input_ids) for sample in samples)
        self.validate = partial(_run_validation, config, progress, models=models, device=self.runtime.device,
                                targets=targets, packs=packs)
        ids = set(validation_manifest["sampling_task_ids"])
        sampling_targets = tuple(target for target in targets if target.task_id in ids)
        inputs = load_sampling_input_grids(validation["bundle"], validation_manifest["sampling_task_ids"])
        if (len(sampling_targets) != sampling["expected_query_count"]
                or {(target.task_id, target.query_index) for target in sampling_targets} != inputs.keys()):
            raise ValueError("sampling cache and input query identities differ")
        sample_packs = plan_validation_packs(sampling_targets, max_joint_tokens=config.flux_packing.max_tokens)
        self.sample = partial(_run_sampling, config, progress, models=models, device=self.runtime.device,
                             targets=sampling_targets, packs=sample_packs, input_grids=inputs, logger=self.logger)

    def initial(self):
        if self.runtime.strategy == "fsdp2":
            self._initial()
            self.progress.workload_totals = self.runtime.run_on_primary_and_broadcast(lambda: self.progress.workload_totals)
        else:
            self.progress.workload_totals = self.runtime.run_on_primary_and_broadcast(self._initial)

    def _initial(self):
        metrics = {}
        if not self.checkpoint and self.resume_run is None:
            if 0 in self.config.validation_steps:
                metrics.update(self.validate(preparation_tokens=self.preparation_tokens))
                self.preparation_tokens = 0
            if 0 in self.config.sampling_steps:
                metrics.update(self.sample())
        if metrics and self.logger is not None:
            self.logger.log(metrics)
        return self.progress.workload_totals

    def measurements(self, result, hardware):
        self.metrics = (build_training_metrics(result, self.progress, self.config)
                        if self.config.recipe["logging"] is not None else {})
        self.metrics.update(hardware)
        step = self.progress.optimizer_step - (self.progress.epoch - 1) * (self.config.epoch_optimizer_steps or self.config.loop.max_optimizer_steps)
        if step in self.config.validation_steps:
            self.metrics.update(self.validate(preparation_tokens=self.preparation_tokens))
            self.preparation_tokens = 0
        if step in self.config.sampling_steps:
            self.metrics.update(self.sample())
        return self.progress.workload_totals

    def before_step(self):
        if self.config.recipe["logging"] is not None:
            self.runtime.run_on_primary_and_broadcast(lambda: self.logger.begin_step())
        if self.runtime.device.type == "cuda" and (not self.runtime.is_primary or self.logger is None):
            torch.cuda.reset_peak_memory_stats(self.runtime.device)

    def completed_step(self):
        pass

    def after_step(self, run_step, batch_plan, local_result, result, progress, *, models,
                   optimizer, scheduler, flow_generator, trainer, identity):
        runtime, config = self.runtime, self.config
        logging = config.recipe["logging"]
        hardware = (runtime.run_on_primary_and_broadcast(lambda: self.logger.collect_hardware_metrics(phase="train"))
                    if logging is not None else {})
        local_memory = ({"system/gpu_allocated_bytes": torch.cuda.memory_allocated(runtime.device),
                         "system/gpu_reserved_bytes": torch.cuda.memory_reserved(runtime.device),
                         "system/gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device)}
                        if logging is not None and runtime.device.type == "cuda" else {})
        if logging is not None:
            local_memory = {k: v for k, v in local_memory.items() if k in logging["hardware"]["per_step_metrics"]}
        ranks = runtime.all_gather_objects((local_result.step_seconds, local_result.padding_forward_calls, local_memory))
        if runtime.strategy != "single_process":
            for rank, (seconds, padding, memory) in enumerate(ranks):
                hardware[f"train/rank_{rank}/step_seconds"] = seconds
                if runtime.strategy == "fsdp2":
                    hardware[f"train/rank_{rank}/padding_forward_calls"] = padding
                for name, value in memory.items():
                    hardware[f"system/rank_{rank}/{name.removeprefix('system/')}"] = value
                    hardware[name] = max(hardware.get(name, 0), value)
        if runtime.strategy == "fsdp2":
            self.measurements(result, hardware)
            progress.workload_totals = runtime.run_on_primary_and_broadcast(lambda: progress.workload_totals)
        else:
            progress.workload_totals = runtime.run_on_primary_and_broadcast(lambda: self.measurements(result, hardware))
        step_offset = (progress.epoch - 1) * (config.epoch_optimizer_steps or config.loop.max_optimizer_steps)
        if progress.optimizer_step - step_offset in config.checkpoint_steps:
            saved = save_checkpoint(progress, config=config, transformer=models.flux_transformer,
                optimizer=optimizer, scheduler=scheduler, flow_generator=flow_generator, device=runtime.device,
                identity=identity, wandb_run=self.wandb_run,
                **(dict(runtime=runtime, completed_update=trainer.completed_update)
                   if runtime.strategy != "single_process" else {}))
            if runtime.is_primary:
                self.metrics.update(saved)
        if logging is not None:
            runtime.run_on_primary_and_broadcast(lambda: self.logger.log(self.metrics))

    def finish(self, progress):
        config = self.config
        epoch_steps = config.epoch_optimizer_steps or config.loop.max_optimizer_steps
        epoch_sources = config.recipe["data"]["expected_task_count"] or 0
        run_targets = config.recipe["loop"]["expected_target_count"]
        expected = ((progress.epoch - 1) * epoch_steps + config.loop.max_optimizer_steps,
                    (progress.epoch - 1) * epoch_sources + run_targets // config.targets_per_sequence,
                    (progress.epoch - 1) * epoch_sources * config.targets_per_sequence + run_targets)
        actual = (progress.optimizer_step, progress.sources_seen, progress.targets_seen)
        if actual != expected:
            raise RuntimeError(f"training counts differ from the configured run: expected {expected}, got {actual}")


def run_training(config: TrainingConfig, *, previous_epoch_checkpoint: Checkpoint | None = None,
                 parent_run_metadata: dict | None = None,
                 runtime: TrainingRuntime | None = None) -> RunProgress:
    """Run configured training services around the shared model/update lifecycle."""
    state = _prepare_training_start(config, previous_epoch_checkpoint, parent_run_metadata)
    progress, checkpoint, resume_run = state
    validation_manifest = _load_validation_manifest(config) if config.recipe["validation"] is not None else None
    with runtime_context(config, runtime) as runtime, ExitStack() as stack:
        callbacks = _TrainingCallbacks(config, runtime, progress, checkpoint, resume_run, previous_epoch_checkpoint)
        return run_training_loop(config, runtime=runtime, stack=stack, callbacks=callbacks,
            progress=progress, checkpoint=checkpoint, previous_epoch_checkpoint=previous_epoch_checkpoint, validation_manifest=validation_manifest)


def _run_validation(config: TrainingConfig, progress: RunProgress, *,
                    models, device: torch.device, targets, packs,
                    preparation_tokens: int = 0) -> dict[str, float | int]:
    """Time one loss check, advance validation counters and build its log values.

    Cache preparation is excluded from the recurring duration. Its NVARC tokens
    are attributed once to the first check, even when initial validation is off.
    evaluate_packed_loss restores the model mode and uses independent noise RNGs.
    """
    loss = config.recipe["validation"]["loss"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = evaluate_packed_loss(
        models.flux_transformer, targets, packs,
        timesteps=loss["timesteps"], seed=loss["noise_seed"],
        compute_dtype=models.flux_compute_dtype, backend=models.flux_packing_backend,
        weights=(RegionLossWeights(**config.recipe["loss"]["region_weights"])
                 if config.recipe["loss"]["type"] == "region_weighted_velocity_mse"
                 else RegionLossWeights(canvas=1.0, grid=0.0, boundary=0.0)),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["seconds"] = time.perf_counter() - started
    result["nvarc_tokens"] += preparation_tokens
    for name in WORKLOAD_COUNTERS:
        key = f"val/{name}"
        progress.workload_totals[key] = progress.workload_totals.get(key, 0) + result[name]
    return build_validation_metrics(result, progress, config) if config.recipe["logging"] is not None else {}


def _run_sampling(config: TrainingConfig, progress: RunProgress, *,
                  models, device: torch.device, targets, packs, input_grids,
                  logger: WandbLogger | None) -> dict:
    """Generate fixed-seed predictions, score them and prepare one W&B table.

    Generation receives cached conditioning only; expected outputs are used for
    scoring and display afterward. Duration includes generation, decoding and
    scoring, but excludes image rendering and W&B serialization/upload.
    """
    sampling = config.recipe["validation"]["sampling"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = generate_packed_samples(
        models.flux_transformer, models.vae_decoder, targets, packs,
        steps=sampling["steps"], seeds=sampling["seeds"],
        compute_dtype=models.flux_compute_dtype, backend=models.flux_packing_backend,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["seconds"] = time.perf_counter() - started
    for name in WORKLOAD_COUNTERS:
        key = f"sample/{name}"
        progress.workload_totals[key] = progress.workload_totals.get(key, 0) + result[name]
    metrics = build_sampling_metrics(result, progress, config) if config.recipe["logging"] is not None else {}
    if logger is not None:
        metrics[config.recipe["logging"]["predictions"]["key"]] = logger.build_prediction_table(
            result, targets=targets, input_grids=input_grids,
        )
    return metrics


__all__ = ["RunProgress", "run_training"]
