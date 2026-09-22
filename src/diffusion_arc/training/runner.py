"""Train one restartable single-GPU epoch with validation, sampling and W&B logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import time

import torch

from .config import TrainingConfig
from .run.checkpointing import (
    Checkpoint, checkpoint_identity, read_checkpoint, restore_training_state, save_checkpoint,
    validate_epoch_continuation,
)
from .data import build_dataloader, iter_optimizer_batches
from .step.flow_sampling import build_flow_generator, build_flow_sampler
from .run.logging import WandbLogger, build_sampling_metrics, build_training_metrics, build_validation_metrics
from .step.loss import RegionLossWeights, build_velocity_loss
from .step.lr_schedule import build_lr_scheduler
from .setup import build_optimizer, load_models, prepare_device, seed_process
from .trainer import Trainer
from .run.validation import (
    evaluate_packed_loss, generate_packed_samples, load_sampling_input_grids,
    plan_validation_packs, prepare_validation_cache,
    tokenize_validation_targets, validate_validation_bundle,
)


@dataclass
class RunProgress:
    """Global training counts and the step at which the current optimizer was created."""

    optimizer_step: int = 0
    sources_seen: int = 0
    targets_seen: int = 0
    workload_totals: dict[str, int] = field(default_factory=dict)
    epoch: int = 1
    optimizer_start_step: int = 0


def run_training(config: TrainingConfig, *, previous_epoch: Checkpoint | None = None,
                 weights_parent: dict | None = None) -> RunProgress:
    """Construct training components and consume the recipe's complete epoch.

    The caller loads and validates the configuration before calling this function.
    The runner owns data iteration, progress and event timing; Trainer.step owns
    each optimizer update. No profiler is started here.

    Validation and sampling reuse fixed CPU-cached queries and contribute to the
    same W&B record as training. Full-state checkpoints follow the sampling events.
    A previous_epoch checkpoint starts a fresh data view while retaining global
    counters and optimizer/RNG state. Ordinary resume stays inside the saved epoch.
    A weights-only initialization retains global epoch numbering but creates fresh
    optimizer/scheduler/RNG state; weights_parent supplies verified W&B lineage.
    """
    if config.config_type != "training_run" or config.recipe is None:
        raise ValueError("run_training requires a training_run config")
    recipe = config.recipe
    expected_sources = recipe["data"]["expected_task_count"]
    expected_targets = recipe["loop"]["expected_target_count"]
    validation_steps = config.validation_steps
    sampling_steps = config.sampling_steps
    checkpoint_steps = config.checkpoint_steps
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
            sources_seen=completed * expected_sources, targets_seen=completed * expected_targets,
            epoch=completed + 1, optimizer_start_step=completed * config.loop.max_optimizer_steps,
            workload_totals=dict(weights_parent["workload_totals"]) if weights_parent else {},
        )
        if recipe["logging"]["resume_parent_run"] and weights_parent is None:
            raise ValueError("verify the parent W&B run before continuing its history")
    if previous_epoch is not None:
        progress.epoch += 1
    step_offset = (progress.epoch - 1) * config.loop.max_optimizer_steps
    source_offset = (progress.epoch - 1) * expected_sources
    if progress.optimizer_step == step_offset + config.loop.max_optimizer_steps:
        raise ValueError("checkpoint already completed this epoch")

    # Reject incompatible or changed validation data before allocating models.
    validation = recipe["validation"]
    manifest = validate_validation_bundle(validation["bundle"], split_seed=validation["split_seed"])
    if (len(manifest["validation_task_ids"]) != validation["expected_task_count"]
            or manifest["validation_output_count"] != validation["expected_query_count"]):
        raise ValueError("validation bundle task/query counts differ from the run recipe")
    sampling = validation["sampling"]
    if (len(manifest["sampling_task_ids"]) != sampling["expected_task_count"]
            or manifest["sampling_output_count"] != sampling["expected_query_count"]):
        raise ValueError("sampling bundle task/query counts differ from the run recipe")

    # Reuse the same model preparation and training components as the B3 baseline.
    device = prepare_device(config)
    resume_run = checkpoint.metadata["wandb_run"] if checkpoint else (
        weights_parent["wandb_run"] if weights_start and recipe["logging"]["resume_parent_run"] else None)
    logger = WandbLogger(config, device=device, resume_run=resume_run)
    seed_process(config.seed)
    models = load_models(config, device=device,
                         transformer_checkpoint=checkpoint.path / "transformer" if checkpoint else
                         recipe["models"].get("transformer_checkpoint"))
    training_samples = build_dataloader(config, tokenizer=models.tokenizer,
                                        start_index=progress.sources_seen - source_offset)

    # Check the prepared view against this epoch before any optimizer update.
    keys = training_samples.dataset.keys
    if len(keys) != expected_sources or len({key.task_id for key in keys}) != expected_sources:
        raise ValueError(f"training view must contain exactly {expected_sources} distinct tasks")
    if any(len(key.augmentation.pair_indices) != recipe["data"]["pairs_per_sequence"] for key in keys):
        raise ValueError("training view pair counts do not match the run recipe")
    data_fingerprint = training_samples.dataset.fingerprint
    if checkpoint and previous_epoch is None:
        expected_fingerprint = checkpoint.metadata["identity"]["data_sha256"]
        if data_fingerprint != expected_fingerprint:
            data_fingerprint = training_samples.dataset.fingerprint_for(expected_fingerprint)
    identity = checkpoint_identity(config, data_fingerprint=data_fingerprint,
                                   validation_manifest=manifest)
    if checkpoint:
        saved_identity = checkpoint.metadata["identity"]
        if previous_epoch is not None:
            if saved_identity["validation_sha256"] != identity["validation_sha256"]:
                raise ValueError("the validation split must remain unchanged between epochs")
        elif saved_identity != identity:
            raise ValueError("checkpoint data content or validation split differs from this run")

    optimizer = build_optimizer(config, flux_transformer=models.flux_transformer)
    scheduler = build_lr_scheduler(config, optimizer=optimizer)
    flow_generator = build_flow_generator(config, device=device)
    flow_sampler = build_flow_sampler(config, generator=flow_generator)
    loss_function = build_velocity_loss(config)
    trainer = Trainer(
        nvarc_encoder=models.nvarc_encoder,
        vae_encoder=models.vae_encoder,
        flux_transformer=models.flux_transformer,
        optimizer=optimizer,
        scheduler=scheduler,
        flow_sampler=flow_sampler,
        loss_function=loss_function,
        flux_compute_dtype=models.flux_compute_dtype,
        flux_packing_backend=models.flux_packing_backend,
        nvarc_max_tokens=config.nvarc_max_tokens,
        vae_microbatch_size=config.loop.vae_microbatch_size,
        flux_max_joint_tokens=config.flux_packing.max_tokens,
        min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
        max_grad_norm=config.optimizer.max_grad_norm,
    )
    with logger:
        if weights_start and not checkpoint:
            logger.run.config.update({f"epoch_{progress.epoch}": {
                "recipe": config.recipe_dict(), "optimizer_reset": True,
                "start_step": progress.optimizer_step, "data_sha256": identity["data_sha256"],
            }}, allow_val_change=True)
        if previous_epoch is not None:
            logger.run.config.update({f"epoch_{progress.epoch}": {
                "augmentation_seed": config.seed, "parent_checkpoint": str(previous_epoch.path),
                "data_sha256": identity["data_sha256"], "start_step": progress.optimizer_step,
            }}, allow_val_change=True)
        # Frozen encoders run once; all later checks reuse detached CPU tensors.
        samples, grids = tokenize_validation_targets(validation["bundle"], manifest, models.tokenizer)
        validation_targets = prepare_validation_cache(
            samples, grids, nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            nvarc_max_tokens=config.nvarc_max_tokens, vae_batch_size=config.loop.vae_microbatch_size,
        )
        validation_packs = plan_validation_packs(
            validation_targets, max_joint_tokens=config.flux_packing.max_tokens,
        )
        preparation_tokens = sum(len(sample.input_ids) for sample in samples)
        validate = partial(
            _run_validation, config, progress, models=models, device=device,
            targets=validation_targets, packs=validation_packs,
        )
        sampling_ids = set(manifest["sampling_task_ids"])
        sampling_targets = tuple(target for target in validation_targets if target.task_id in sampling_ids)
        input_grids = load_sampling_input_grids(validation["bundle"], manifest["sampling_task_ids"])
        if (len(sampling_targets) != sampling["expected_query_count"]
                or {(target.task_id, target.query_index) for target in sampling_targets} != input_grids.keys()):
            raise ValueError("sampling cache and input query identities differ")
        sampling_packs = plan_validation_packs(
            sampling_targets, max_joint_tokens=config.flux_packing.max_tokens,
        )
        sample = partial(
            _run_sampling, config, progress, models=models, device=device,
            targets=sampling_targets, packs=sampling_packs, input_grids=input_grids, logger=logger,
        )
        save = partial(
            save_checkpoint, config=config, transformer=models.flux_transformer, optimizer=optimizer,
            scheduler=scheduler, flow_generator=flow_generator, device=device,
            identity=identity, wandb_run=logger.run_identity,
        )
        if checkpoint:
            # Loading models/cache can consume RNG draws. Restore training RNGs last.
            restore_training_state(
                checkpoint, transformer=models.flux_transformer, optimizer=optimizer, scheduler=scheduler,
                flow_generator=flow_generator, device=device,
            )
        initial_metrics = {}
        if not checkpoint and resume_run is None and 0 in validation_steps:
            initial_metrics.update(validate(preparation_tokens=preparation_tokens))
            preparation_tokens = 0
        if not checkpoint and resume_run is None and 0 in sampling_steps:
            initial_metrics.update(sample())
        if initial_metrics:
            logger.log(initial_metrics)

        # Event plans include the final step, replacing the last periodic event.
        for sequences in iter_optimizer_batches(
            training_samples,
            optimizer_batch_size=config.loop.optimizer_batch_size,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
            max_optimizer_steps=step_offset + config.loop.max_optimizer_steps - progress.optimizer_step,
        ):
            logger.begin_step()
            result = trainer.step(sequences)
            # Read memory before validation or sampling can change the training peak.
            hardware_metrics = logger.collect_hardware_metrics(phase="train")
            progress.optimizer_step += 1
            progress.sources_seen += result.source_count
            progress.targets_seen += result.target_count
            for name in recipe["logging"]["workload_counters"]["names"]:
                key = f"train/{name}"
                progress.workload_totals[key] = progress.workload_totals.get(key, 0) + getattr(result, name)
            metrics = build_training_metrics(result, progress, config)
            metrics.update(hardware_metrics)

            epoch_step = progress.optimizer_step - step_offset
            if epoch_step in validation_steps:
                metrics.update(validate(preparation_tokens=preparation_tokens))
                preparation_tokens = 0
            if epoch_step in sampling_steps:
                metrics.update(sample())
            if epoch_step in checkpoint_steps:
                metrics.update(save(progress))
            logger.log(metrics)

        expected = tuple(progress.epoch * count for count in
                         (config.loop.max_optimizer_steps, expected_sources, expected_targets))
        actual = (progress.optimizer_step, progress.sources_seen, progress.targets_seen)
        if actual != expected:
            raise RuntimeError(f"training epoch counts differ from the recipe: expected {expected}, got {actual}")
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
