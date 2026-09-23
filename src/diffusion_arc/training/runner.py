"""Run a shared Single-GPU/DDP epoch or a bounded training experiment."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward
from .config import TrainingConfig
from .run.checkpointing import (
    Checkpoint, checkpoint_identity, checkpoint_identities_match, read_checkpoint,
    restore_training_state, save_checkpoint, validate_epoch_continuation,
)
from .data import build_dataloader, iter_optimizer_batches
from .distributed import DistributedRuntime, distributed_runtime
from .distributed_data import build_distributed_dataloader
from .step.distributed_flow import build_distributed_flow_sampler
from .step.flow_sampling import build_flow_generator, build_flow_sampler
from .run.logging import WandbLogger, build_sampling_metrics, build_training_metrics, build_validation_metrics
from .run.experiment import TrainingExperiment, training_settings
from .step.loss import RegionLossWeights, build_velocity_loss
from .step.lr_schedule import build_lr_scheduler
from .setup import build_optimizer, load_models, prepare_device, seed_process
from .trainer import Trainer, StepDiagnostics
from .step.encoding import PreparationMetrics
from .profiling import training_range
from .run.validation import (
    evaluate_packed_loss, generate_packed_samples, load_sampling_input_grids,
    plan_validation_packs, prepare_validation_cache,
    tokenize_validation_targets, validate_validation_bundle,
)


@dataclass
class RunProgress:
    """Global counts, including the update at which this optimizer was created."""

    optimizer_step: int = 0
    sources_seen: int = 0
    targets_seen: int = 0
    workload_totals: dict[str, int] = field(default_factory=dict)
    epoch: int = 1
    optimizer_start_step: int = 0


def aggregate_results(results):
    """Combine CPU metrics with global target weighting and rank-local timing."""
    if len(results) == 1:
        return results[0]
    target_count = sum(result.target_count for result in results)
    counters = ("source_count", "target_count", "nvarc_tokens", "nvarc_calls", "vae_calls",
                "flux_text_tokens", "flux_image_tokens", "flux_joint_tokens", "flux_forward_calls")
    source_offset = target_offset = 0
    sources, lengths, nvarc_batches, packs, prefixes = [], [], [], [], []
    for result in results:
        diagnostics = result.diagnostics
        preparation = diagnostics.preparation
        sources.extend(preparation.sources)
        lengths.extend(preparation.source_lengths)
        nvarc_batches.extend(tuple(i + source_offset for i in batch) for batch in preparation.nvarc_batches)
        packs.extend(replace(batch, target_indices=tuple(i + target_offset for i in batch.target_indices))
                     for batch in diagnostics.flux_batches)
        prefixes.extend(diagnostics.target_prefix_tokens)
        source_offset += result.source_count
        target_offset += result.target_count
    return replace(results[0], **{name: sum(getattr(r, name) for r in results) for name in counters},
        loss=sum(r.loss * r.target_count for r in results) / target_count,
        step_seconds=max(r.step_seconds for r in results),
        gradient_norm=max(r.gradient_norm for r in results),
        gradient_clipped=any(r.gradient_clipped for r in results),
        targets=tuple(target for result in results for target in result.targets),
        diagnostics=StepDiagnostics(PreparationMetrics(tuple(sources), tuple(lengths), tuple(nvarc_batches),
                                    sum(r.vae_calls for r in results)), tuple(packs), tuple(prefixes)))


def _start_progress(config, previous_epoch, weights_parent):
    recipe = config.recipe
    resume_from = recipe["models"]["resume_from"]
    if previous_epoch is not None:
        if resume_from is not None:
            raise ValueError("choose either an in-epoch resume or a new epoch")
        validate_epoch_continuation(previous_epoch, config)
    checkpoint = read_checkpoint(resume_from, config) if resume_from else previous_epoch
    progress = RunProgress(**checkpoint.metadata["progress"]) if checkpoint else RunProgress()
    weights_start = recipe["models"]["initialization"] == "transformer_weights"
    if weights_start and not checkpoint:
        if recipe["models"]["transformer_checkpoint"] is None:
            raise ValueError("prepare the initial transformer checkpoint before training")
        completed = recipe["models"]["completed_epochs"]
        progress = RunProgress(
            optimizer_step=completed * config.loop.max_optimizer_steps,
            sources_seen=completed * recipe["data"]["expected_task_count"],
            targets_seen=completed * recipe["loop"]["expected_target_count"], epoch=completed + 1,
            optimizer_start_step=completed * config.loop.max_optimizer_steps,
            workload_totals=dict(weights_parent["workload_totals"]) if weights_parent else {})
        if recipe["logging"]["resume_parent_run"] and weights_parent is None:
            raise ValueError("verify the parent W&B run before continuing its history")
    if previous_epoch is not None:
        progress.epoch += 1
    if progress.optimizer_step == progress.epoch * config.loop.max_optimizer_steps:
        raise ValueError("checkpoint already completed this epoch")
    resume_run = checkpoint.metadata["wandb_run"] if checkpoint else (
        weights_parent["wandb_run"] if weights_start and recipe["logging"]["resume_parent_run"] else None)
    return progress, checkpoint, resume_run


def _validation_manifest(config):
    validation = config.recipe["validation"]
    manifest = validate_validation_bundle(validation["bundle"], split_seed=validation["split_seed"],
                                          verify_payloads=False)
    if (len(manifest["validation_task_ids"]) != validation["expected_task_count"]
            or manifest["validation_output_count"] != validation["expected_query_count"]):
        raise ValueError("validation bundle task/query counts differ from the run recipe")
    sampling = validation["sampling"]
    if (len(manifest["sampling_task_ids"]) != sampling["expected_task_count"]
            or manifest["sampling_output_count"] != sampling["expected_query_count"]):
        raise ValueError("sampling bundle task/query counts differ from the run recipe")
    return manifest


class _TrainingEvents:
    """Keep full-run services on rank zero, with collective checkpoint boundaries."""

    def __init__(self, config, runtime, progress, checkpoint, resume_run, previous_epoch):
        self.config, self.runtime, self.progress = config, runtime, progress
        self.checkpoint, self.resume_run, self.previous_epoch = checkpoint, resume_run, previous_epoch

    def open(self, stack):
        @contextmanager
        def logging_context():
            with self.logger:
                yield
        def enter():
            self.logger = WandbLogger(self.config, device=self.runtime.device, resume_run=self.resume_run)
            stack.enter_context(logging_context())
            return self.logger.run_identity
        self.wandb_run = self.runtime.primary(enter)

    def prepare(self, models, manifest, identity):
        config, progress = self.config, self.progress
        recipe, validation = config.recipe, config.recipe["validation"]
        sampling = validation["sampling"]
        if recipe["models"]["initialization"] == "transformer_weights" and not self.checkpoint:
            self.logger.run.config.update({f"epoch_{progress.epoch}": {
                "recipe": config.recipe_dict(), "optimizer_reset": True, "start_step": progress.optimizer_step,
                "data_sha256": identity["data_sha256"]}}, allow_val_change=True)
        if self.previous_epoch is not None:
            self.logger.run.config.update({f"epoch_{progress.epoch}": {
                "augmentation_seed": config.seed, "parent_checkpoint": str(self.previous_epoch.path),
                "data_sha256": identity["data_sha256"], "start_step": progress.optimizer_step}}, allow_val_change=True)
        samples, grids = tokenize_validation_targets(validation["bundle"], manifest, models.tokenizer)
        targets = prepare_validation_cache(samples, grids, nvarc_encoder=models.nvarc_encoder,
            vae_encoder=models.vae_encoder, nvarc_max_tokens=config.nvarc_max_tokens,
            vae_batch_size=config.loop.vae_microbatch_size)
        packs = plan_validation_packs(targets, max_joint_tokens=config.flux_packing.max_tokens)
        self.preparation_tokens = sum(len(sample.input_ids) for sample in samples)
        self.validate = partial(_run_validation, config, progress, models=models, device=self.runtime.device,
                                targets=targets, packs=packs)
        ids = set(manifest["sampling_task_ids"])
        sampling_targets = tuple(target for target in targets if target.task_id in ids)
        inputs = load_sampling_input_grids(validation["bundle"], manifest["sampling_task_ids"])
        if (len(sampling_targets) != sampling["expected_query_count"]
                or {(target.task_id, target.query_index) for target in sampling_targets} != inputs.keys()):
            raise ValueError("sampling cache and input query identities differ")
        sample_packs = plan_validation_packs(sampling_targets, max_joint_tokens=config.flux_packing.max_tokens)
        self.sample = partial(_run_sampling, config, progress, models=models, device=self.runtime.device,
                             targets=sampling_targets, packs=sample_packs, input_grids=inputs, logger=self.logger)

    def initial(self):
        metrics = {}
        if not self.checkpoint and self.resume_run is None:
            if 0 in self.config.validation_steps:
                metrics.update(self.validate(preparation_tokens=self.preparation_tokens))
                self.preparation_tokens = 0
            if 0 in self.config.sampling_steps:
                metrics.update(self.sample())
        if metrics:
            self.logger.log(metrics)
        return self.progress.workload_totals

    def measurements(self, result, hardware):
        self.metrics = build_training_metrics(result, self.progress, self.config)
        self.metrics.update(hardware)
        step = self.progress.optimizer_step - (self.progress.epoch - 1) * self.config.loop.max_optimizer_steps
        if step in self.config.validation_steps:
            self.metrics.update(self.validate(preparation_tokens=self.preparation_tokens))
            self.preparation_tokens = 0
        if step in self.config.sampling_steps:
            self.metrics.update(self.sample())
        return self.progress.workload_totals


def run_training(config: TrainingConfig, *, previous_epoch: Checkpoint | None = None,
                 weights_parent: dict | None = None, experiment: TrainingExperiment | None = None,
                 runtime: DistributedRuntime | None = None) -> RunProgress:
    """Use one model/data/update lifecycle for native Single GPU and DDP.

    Full runs preserve all recipe services and epoch/resume semantics. An explicit
    experiment selects bounded measurement events instead. A supplied runtime is
    caller-owned, which also permits small CPU/Gloo integration tests.
    """
    if config.config_type != "training_run" or config.recipe is None:
        raise ValueError("run_training requires a training_run config")
    if experiment is not None:
        experiment.validate(config)
        if previous_epoch is not None or weights_parent is not None:
            raise ValueError("bounded experiments cannot continue an earlier run")
    state = _start_progress(config, previous_epoch, weights_parent)
    manifest = _validation_manifest(config) if experiment is None else None
    strategy = config.recipe["execution"]["distributed_strategy"]
    context = (nullcontext(runtime) if runtime is not None else
               distributed_runtime(config, timeout_seconds=1800) if strategy == "ddp" else
               nullcontext(DistributedRuntime(0, 0, 1, prepare_device(config), None, "single_process")))
    with context as runtime, ExitStack() as stack:
        if runtime.strategy != strategy or runtime.world_size != config.recipe["deployment"]["gpu_count"]:
            raise ValueError("runtime strategy/world size differs from the configured execution")
        return _run_training(config, runtime, stack, previous_epoch, experiment, state, manifest)


def _run_training(config, runtime, stack, previous_epoch, experiment, state, manifest):
    recipe, device = config.recipe, runtime.device
    full_run = experiment is None
    progress, checkpoint, resume_run = state
    step_offset = (progress.epoch - 1) * config.loop.max_optimizer_steps
    start_step = progress.optimizer_step - step_offset
    source_offset = (progress.epoch - 1) * recipe["data"]["expected_task_count"]
    events = _TrainingEvents(config, runtime, progress, checkpoint, resume_run, previous_epoch) if full_run else None
    if full_run:
        events.open(stack)
    else:
        experiment.open(stack, runtime)
    seed_process(config.seed)
    models = load_models(config, device=device, transformer_checkpoint=(
        checkpoint.path / "transformer" if checkpoint else recipe["models"].get("transformer_checkpoint")))
    indexed_flow = recipe["flow"]["rng_scheme"] == "global_step_target_v1"
    plans = None
    if indexed_flow:
        loader, plans = build_distributed_dataloader(config, tokenizer=models.tokenizer,
            rank=runtime.rank, world_size=runtime.world_size, start_step=start_step,
            expected_source_count=experiment.expected_source_count if experiment else None,
            diagnostic_batch=experiment.diagnostic_batch if experiment else None,
            diagnostic_repeats=experiment.max_steps if experiment else 3)
    else:
        if runtime.strategy != "single_process" or experiment:
            raise ValueError("DDP and bounded experiments require global_step_target_v1 flow RNG")
        loader = build_dataloader(config, tokenizer=models.tokenizer, start_index=progress.sources_seen - source_offset)
        keys = loader.dataset.keys
        if len(keys) != recipe["data"]["expected_task_count"] or len({key.task_id for key in keys}) != len(keys):
            raise ValueError("training view must contain exactly the recipe's distinct tasks")
        if any(len(key.augmentation.pair_indices) != recipe["data"]["pairs_per_sequence"] for key in keys):
            raise ValueError("training view pair counts do not match the run recipe")
    max_steps = experiment.max_steps if experiment else config.loop.max_optimizer_steps - start_step
    if plans is not None:
        if (experiment is None or experiment.diagnostic_batch is None and experiment.expected_source_count is None):
            if len(plans) != config.loop.max_optimizer_steps - start_step:
                raise ValueError("global batch plan does not match the recipe's optimizer-step count")
        if len(plans) < max_steps:
            raise ValueError("the data view has too few global batches for this run")
        plans = plans[:max_steps]
        if experiment and experiment.expected_source_count is not None and any(
                batch.global_target_count != config.loop.optimizer_batch_size for batch in plans):
            raise ValueError("benchmark data must provide full global optimizer batches")
    experiment_settings = ({name: getattr(experiment, name) for name in (
        "max_steps", "expected_source_count", "measurement", "diagnostic_batch", "trace_directory",
        "memory_history", "memory_snapshot_step", "profile_warmup_steps")} if experiment else None)
    identities = runtime.gather((config.recipe_dict(), training_settings(config), loader.dataset.fingerprint,
                                 max_steps, progress, experiment_settings))
    if any(identity != identities[0] for identity in identities):
        raise ValueError("DDP ranks have different training settings, recipes, data fingerprints or step limits")
    identity = checkpoint_identity(config, data_fingerprint=loader.dataset.fingerprint,
                                   validation_manifest=manifest) if full_run else None
    if checkpoint:
        saved = checkpoint.metadata["identity"]
        if previous_epoch is not None:
            if saved["validation_sha256"] != identity["validation_sha256"]:
                raise ValueError("the validation split must remain unchanged between epochs")
        elif not checkpoint_identities_match(saved, identity):
            raise ValueError("checkpoint data content or validation split differs from this run; audit legacy data identities explicitly on CPU")
    flow_generator = build_flow_generator(config, device=device)
    flow_sampler = (build_distributed_flow_sampler(config, step=step_offset + plans[0].step,
                    global_target_count=plans[0].global_target_count, target_indices=plans[0].target_indices)
                    if indexed_flow else build_flow_sampler(config, generator=flow_generator))
    forward = models.flux_transformer
    if runtime.strategy == "ddp":
        forward = DDP(PackedFluxForward(forward, compute_dtype=models.flux_compute_dtype,
                      backend=models.flux_packing_backend),
                      device_ids=[device.index] if device.type == "cuda" else None,
                      process_group=runtime.process_group, forward_sync_buffers=False, gradient_as_bucket_view=True)
    optimizer = build_optimizer(config, flux_transformer=models.flux_transformer)
    scheduler = build_lr_scheduler(config, optimizer=optimizer)
    trainer = Trainer(nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
        flux_transformer=forward, optimizer=optimizer, scheduler=scheduler, flow_sampler=flow_sampler,
        loss_function=build_velocity_loss(config), flux_compute_dtype=models.flux_compute_dtype,
        flux_packing_backend=models.flux_packing_backend, nvarc_max_tokens=config.nvarc_max_tokens,
        vae_microbatch_size=config.loop.vae_microbatch_size, flux_max_joint_tokens=config.flux_packing.max_tokens,
        flux_max_spread_tokens=config.flux_packing.max_spread_tokens, flux_packing_algorithm=config.flux_packing.algorithm,
        min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
        max_grad_norm=config.optimizer.max_grad_norm, trace=experiment.trace if experiment else None)
    checkpoint_options = {"runtime": runtime} if runtime.strategy == "ddp" else {}
    if full_run:
        runtime.primary(lambda: events.prepare(models, manifest, identity))
    else:
        experiment.start(stack, config, models, loader.dataset)
    if checkpoint:
        restore_training_state(checkpoint, transformer=models.flux_transformer, optimizer=optimizer,
            scheduler=scheduler, flow_generator=flow_generator, device=device, **checkpoint_options)
    if full_run:
        progress.workload_totals = runtime.primary(events.initial)
    if plans is not None:
        samples = iter(loader)
        batches = iter(plans)
    else:
        batches = iter_optimizer_batches(loader, optimizer_batch_size=config.loop.optimizer_batch_size,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target, max_optimizer_steps=max_steps)

    # One update loop for both strategies and both full-run/measurement events.
    for repetition, batch in enumerate(batches, 1):
        if full_run:
            runtime.primary(lambda: events.logger.begin_step())
            if device.type == "cuda" and not runtime.is_primary:
                torch.cuda.reset_peak_memory_stats(device)
        else:
            experiment.before_step()
        with training_range(f"optimizer_step/{progress.optimizer_step + 1}"):
            with training_range("data/retrieval"):
                sequences = tuple(next(samples) for _ in batch.source_indices) if plans is not None else batch
            if indexed_flow:
                trainer.flow_sampler = build_distributed_flow_sampler(config, step=step_offset + batch.step,
                    global_target_count=batch.global_target_count, target_indices=batch.target_indices)
            if experiment:
                experiment.retrieved(repetition, batch, sequences)
            try:
                local_result = trainer.step(sequences)
            except BaseException as error:
                if experiment:
                    experiment.failed(error)
                raise
        if experiment:
            experiment.completed_step()
        local_memory = ({"system/gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                         "system/gpu_reserved_bytes": torch.cuda.memory_reserved(device),
                         "system/gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
                        if full_run and device.type == "cuda" else {})
        local_memory = {name: value for name, value in local_memory.items()
                        if name in recipe["logging"]["hardware"]["per_step_metrics"]}
        hardware = runtime.primary(lambda: events.logger.collect_hardware_metrics(phase="train")) if full_run else None
        with training_range("distributed/metrics"):
            ranks = runtime.gather((local_result, local_memory))
            result = aggregate_results([item[0] for item in ranks])
        if full_run and runtime.strategy == "ddp":
            for rank, (rank_result, memory) in enumerate(ranks):
                hardware[f"train/rank_{rank}/step_seconds"] = rank_result.step_seconds
                for name, value in memory.items():
                    hardware[f"system/rank_{rank}/{name.removeprefix('system/')}"] = value
                    hardware[name] = max(hardware.get(name, 0), value)
        if plans is not None and result.target_count != batch.global_target_count:
            raise RuntimeError("executed target count differs from the global batch plan")
        progress.optimizer_step += 1
        progress.sources_seen += result.source_count
        progress.targets_seen += result.target_count
        for name in recipe["logging"]["workload_counters"]["names"]:
            key = f"train/{name}"
            progress.workload_totals[key] = progress.workload_totals.get(key, 0) + getattr(result, name)
        if full_run:
            progress.workload_totals = runtime.primary(lambda: events.measurements(result, hardware))
            if progress.optimizer_step - step_offset in config.checkpoint_steps:
                saved = save_checkpoint(progress, config=config, transformer=models.flux_transformer,
                    optimizer=optimizer, scheduler=scheduler, flow_generator=flow_generator, device=device,
                    identity=identity, wandb_run=events.wandb_run,
                    **(dict(runtime=runtime, completed_update=trainer.completed_update) if runtime.strategy == "ddp" else {}))
                if runtime.is_primary:
                    events.metrics.update(saved)
            runtime.primary(lambda: events.logger.log(events.metrics))
        else:
            experiment.after_step(repetition, batch, local_result, progress)
    if full_run:
        expected = tuple(progress.epoch * count for count in (config.loop.max_optimizer_steps,
            recipe["data"]["expected_task_count"], recipe["loop"]["expected_target_count"]))
        actual = (progress.optimizer_step, progress.sources_seen, progress.targets_seen)
        if actual != expected:
            raise RuntimeError(f"training epoch counts differ from the recipe: expected {expected}, got {actual}")
    else:
        experiment.finish(progress)
    return progress


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
        weights=RegionLossWeights(**config.recipe["loss"]["region_weights"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result["seconds"] = time.perf_counter() - started
    result["nvarc_tokens"] += preparation_tokens
    for name in config.recipe["logging"]["workload_counters"]["names"]:
        key = f"val/{name}"
        progress.workload_totals[key] = progress.workload_totals.get(key, 0) + result[name]
    return build_validation_metrics(result, progress, config)


def _run_sampling(config: TrainingConfig, progress: RunProgress, *,
                  models, device: torch.device, targets, packs, input_grids,
                  logger: WandbLogger) -> dict:
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
    for name in config.recipe["logging"]["workload_counters"]["names"]:
        key = f"sample/{name}"
        progress.workload_totals[key] = progress.workload_totals.get(key, 0) + result[name]
    metrics = build_sampling_metrics(result, progress, config)
    metrics[config.recipe["logging"]["predictions"]["key"]] = logger.build_prediction_table(
        result, targets=targets, input_grids=input_grids,
    )
    return metrics


__all__ = ["RunProgress", "run_training"]
