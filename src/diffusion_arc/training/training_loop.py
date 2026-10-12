"""Shared model, data and optimizer lifecycle, independent of run instrumentation."""

from contextlib import contextmanager
from dataclasses import replace

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward
from .data import build_dataloader, iter_optimizer_batches
from .distributed import TrainingRuntime, distributed_runtime
from .distributed_data import BackgroundBatchPlans, RankBatchPlan, build_distributed_dataloader
from .distribution_strategy.fsdp import shard_flux
from .profiling import training_range
from .run.checkpointing import checkpoint_identity, checkpoint_identities_match, restore_training_state
from .run.experiment import training_settings
from .run.state import aggregate_results
from .setup import build_optimizer, load_models, prepare_device, seed_process
from .step.indexed_flow import build_indexed_flow_sampler
from .step.flow_sampling import build_flow_generator, build_flow_sampler
from .step.loss import build_velocity_loss
from .step.lr_schedule import build_lr_scheduler
from .trainer import Trainer


WORKLOAD_COUNTERS = ("nvarc_tokens", "flux_text_tokens", "flux_image_tokens",
                     "flux_joint_tokens", "flux_forward_calls")


@contextmanager
def runtime_context(config, runtime=None):
    """Create the configured runtime, or validate a caller-owned runtime."""
    strategy = config.recipe["execution"]["distributed_strategy"]
    if runtime is None and strategy in ("ddp", "fsdp2"):
        with distributed_runtime(config, timeout_seconds=1800) as owned:
            with runtime_context(config, owned) as checked:
                yield checked
        return
    if runtime is None:
        runtime = TrainingRuntime(0, 0, 1, prepare_device(config), None, "single_process")
    if runtime.strategy != strategy or runtime.world_size != config.recipe["deployment"]["gpu_count"]:
        raise ValueError("runtime strategy/world size differs from the configured execution")
    yield runtime


def _stream_batches(loader, config, max_steps, start_step):
    """Attach batch identities to the stateful single-process data stream."""
    source_start = 0
    for epoch_batch_index, sequences in enumerate(iter_optimizer_batches(loader,
            optimizer_batch_size=config.loop.optimizer_batch_size,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
            max_optimizer_steps=max_steps), start_step):
        count = sum(max(0, len(s.target_grids) - config.loop.min_context_pairs_before_target + 1)
                    for s in sequences)
        batch_plan = RankBatchPlan(epoch_batch_index, tuple(range(source_start, source_start + len(sequences))),
                                 tuple(range(count)), count)
        source_start += len(sequences)
        yield batch_plan, sequences


def run_training_loop(config, *, runtime, stack, callbacks, progress, checkpoint=None,
                previous_epoch_checkpoint=None, validation_manifest=None, experiment=None):
    """Prepare models, data and optimizer, restore saved state, and run the training loop."""
    # Derive the position within this epoch and the number of remaining updates.
    recipe, device = config.recipe, runtime.device
    epoch_steps = config.epoch_optimizer_steps or config.loop.max_optimizer_steps
    step_offset = (progress.epoch - 1) * epoch_steps
    start_step = progress.optimizer_step - step_offset
    source_offset = (progress.epoch - 1) * (recipe["data"]["expected_task_count"] or 0)
    max_steps = experiment.max_steps if experiment else config.loop.max_optimizer_steps - start_step

    # Open logging or diagnostics, seed the process, and load model components.
    callbacks.open(stack, runtime)
    seed_process(config.seed)
    models = load_models(config, device=device, transformer_checkpoint=(
        checkpoint.path / "transformer" if checkpoint else recipe["models"].get("transformer_checkpoint")),
        **({"transformer_config_only": True} if checkpoint and runtime.strategy == "fsdp2" else {}))

    # Build this rank's batch plans, or resume the single-process data stream.
    indexed_flow = recipe["flow"]["rng_scheme"] == "global_step_target_v1"
    batch_plans = None
    if indexed_flow:
        loader, batch_plans = build_distributed_dataloader(config, tokenizer=models.tokenizer,
            rank=runtime.rank, world_size=runtime.world_size, start_step=start_step,
            expected_source_count=experiment.expected_source_count if experiment else None,
            diagnostic_batch_number=experiment.diagnostic_batch_number if experiment else None,
            diagnostic_repeats=experiment.max_steps if experiment else 3,
            **({"precompute_packing": experiment.precompute_packing, "max_batches": max_steps}
               if experiment and (experiment.precompute_packing or config.rank_batch_assignment == "joint_packing_v1") else {}))
        if isinstance(batch_plans, BackgroundBatchPlans):
            stack.callback(batch_plans.close)
        if len(batch_plans) < max_steps:
            raise ValueError("the data view has too few global batches for this run")
        if recipe["loop"]["epochs"] is not None and experiment is None and len(batch_plans) != max_steps:
            raise ValueError("global batch plan does not match the epoch's optimizer-step count")
        if isinstance(batch_plans, BackgroundBatchPlans):
            batch_plans.truncate(max_steps)
            preflight_plans = batch_plans.original_plans
        else:
            batch_plans = batch_plans[:max_steps]
            preflight_plans = batch_plans
        if experiment and experiment.expected_source_count is not None and any(
                batch_plan.global_target_count != config.loop.optimizer_batch_size for batch_plan in preflight_plans):
            raise ValueError("benchmark data must provide full global optimizer batches")
    else:
        if runtime.strategy != "single_process":
            raise ValueError("distributed execution requires global_step_target_v1 flow RNG")
        loader = build_dataloader(config, tokenizer=models.tokenizer,
                                  start_index=progress.sources_seen - source_offset)
        keys = loader.dataset.keys
        expected_sources = recipe["data"]["expected_task_count"]
        if expected_sources is not None and (len(keys) != expected_sources
                or len({key.task_id for key in keys}) != expected_sources):
            raise ValueError("training view must contain exactly the recipe's distinct tasks")
        if any(len(key.augmentation.pair_indices) != recipe["data"]["pairs_per_sequence"] for key in keys):
            raise ValueError("training view pair counts do not match the run recipe")

    # Require all ranks to agree on settings, data identity, and starting progress.
    experiment_settings = ({name: getattr(experiment, name) for name in (
        "max_steps", "expected_source_count", "measurement", "diagnostic_batch_number", "trace_directory",
        "memory_history", "profile_warmup_steps", "precompute_packing", "fsdp_gradient_sync")} if experiment else None)
    rank_assignment = getattr(loader.dataset, "rank_assignment_metadata", None)
    identities = runtime.all_gather_objects((config.recipe_dict(), training_settings(config), loader.dataset.fingerprint, rank_assignment,
                                 max_steps, progress, experiment_settings))
    if any(identity != identities[0] for identity in identities):
        raise ValueError("ranks have different training settings, recipes, data fingerprints or step limits")

    # Check that the saved checkpoint belongs to the expected data and validation split.
    identity = checkpoint_identity(config, data_fingerprint=loader.dataset.fingerprint,
                                   validation_manifest=validation_manifest)
    if rank_assignment is not None:
        identity["rank_batch_assignment"] = rank_assignment
    if checkpoint:
        saved = checkpoint.metadata["identity"]
        if previous_epoch_checkpoint is not None:
            if saved["validation_sha256"] != identity["validation_sha256"]:
                raise ValueError("the validation split must remain unchanged between epochs")
        elif not checkpoint_identities_match(saved, identity):
            raise ValueError("checkpoint data content or validation split differs from this run")

    # Configure flow sampling and apply DDP or FSDP before creating the optimizer.
    flow_generator = build_flow_generator(config, device=device)
    flow_sampler = (build_indexed_flow_sampler(config, step=step_offset + batch_plans[0].epoch_batch_index,
                    global_target_count=batch_plans[0].global_target_count, target_indices=batch_plans[0].target_indices)
                    if indexed_flow else build_flow_sampler(config, generator=flow_generator))
    flux_model = PackedFluxForward(models.flux_transformer, compute_dtype=models.flux_compute_dtype,
                                   backend=models.flux_packing_backend)
    if runtime.strategy == "ddp":
        flux_model = DDP(flux_model, device_ids=[device.index] if device.type == "cuda" else None,
                      process_group=runtime.process_group, forward_sync_buffers=False, gradient_as_bucket_view=True)
    elif runtime.strategy == "fsdp2":
        flux_model = shard_flux(flux_model, runtime=runtime,
                                reshard_after_forward=config.fsdp_reshard_after_forward,
                                blocks_per_group=config.fsdp_blocks_per_group)

    # Assemble the optimizer, scheduler, and Trainer for one complete optimizer update.
    optimizer = build_optimizer(config, flux_transformer=flux_model)
    scheduler = build_lr_scheduler(config, optimizer=optimizer)
    trainer = Trainer(nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
        flux_transformer=flux_model, optimizer=optimizer, scheduler=scheduler, flow_sampler=flow_sampler,
        loss_function=build_velocity_loss(config), flux_compute_dtype=models.flux_compute_dtype,
        flux_packing_backend=models.flux_packing_backend, nvarc_max_tokens=config.nvarc_max_tokens,
        vae_microbatch_size=config.loop.vae_microbatch_size, flux_max_joint_tokens=config.flux_packing.max_tokens,
        flux_max_spread_tokens=config.flux_packing.max_spread_tokens, flux_packing_algorithm=config.flux_packing.algorithm,
        min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
        max_grad_norm=config.optimizer.max_grad_norm)
    if experiment and experiment.fsdp_gradient_sync == "per_microbatch":
        # Restore the original FSDP reduction policy only in bounded experiments.
        trainer.distribution_strategy.synchronize_each_microbatch = True

    # Prepare evaluation caches before restoring optimizer, scheduler, and RNG state.
    evaluation_models = replace(models, flux_transformer=flux_model) if runtime.strategy == "fsdp2" else models
    callbacks.prepare(stack, evaluation_models, loader.dataset, identity, validation_manifest)
    if checkpoint:
        restore_training_state(checkpoint, transformer=models.flux_transformer, optimizer=optimizer,
            scheduler=scheduler, flow_generator=flow_generator, device=device,
            **({"runtime": runtime} if runtime.strategy != "single_process" else {}))
    callbacks.initial()

    # Execute the remaining updates. run_step starts at 1 for this invocation;
    # progress.optimizer_step counts completed updates across epochs and resumes.
    samples = iter(loader) if batch_plans is not None else None
    batch_iterator = iter(batch_plans) if batch_plans is not None else _stream_batches(loader, config, max_steps, start_step)
    executed = 0
    for run_step in range(1, max_steps + 1):
        callbacks.before_step()
        with training_range(f"optimizer_step/{progress.optimizer_step + 1}"):
            # Load this rank's sources for the next optimizer batch.
            try:
                with training_range("data/retrieval"):
                    if batch_plans is not None:
                        batch_plan = next(batch_iterator)
                        sequences = tuple(next(samples) for _ in batch_plan.source_indices)
                    else:
                        batch_plan, sequences = next(batch_iterator)
            except StopIteration as error:
                raise ValueError("training data exhausted before the configured step limit") from error
            # Derive indexed noise and timesteps from the global batch identity.
            if indexed_flow:
                trainer.flow_sampler = build_indexed_flow_sampler(config, step=step_offset + batch_plan.epoch_batch_index,
                    global_target_count=batch_plan.global_target_count, target_indices=batch_plan.target_indices)
            # Trainer handles microbatches, backward passes, and the optimizer/scheduler update.
            local_result = trainer.step(sequences,
                **({"batch_plan": batch_plan} if batch_plan.flux_plan is not None else {}))
        callbacks.completed_step()
        if isinstance(batch_plans, BackgroundBatchPlans):
            batch_plans.release(run_step - 1)

        # Combine rank results and advance the global training counters.
        with training_range("distributed/metrics"):
            result = aggregate_results(runtime.all_gather_objects(local_result))
        if result.target_count != batch_plan.global_target_count:
            raise RuntimeError("executed target count differs from the global batch plan")
        progress.optimizer_step += 1
        progress.sources_seen += result.source_count
        progress.targets_seen += result.target_count
        for name in WORKLOAD_COUNTERS:
            key = f"train/{name}"
            progress.workload_totals[key] = progress.workload_totals.get(key, 0) + getattr(result, name)

        # Run scheduled logging, evaluation, checkpoints, or benchmark measurements.
        callbacks.after_step(run_step, batch_plan, local_result, result, progress,
            models=models, optimizer=optimizer, scheduler=scheduler, flow_generator=flow_generator,
            trainer=trainer, identity=identity)
        executed += 1

    if executed != max_steps:
        raise RuntimeError("training stopped before the configured number of updates")
    callbacks.finish(progress)
    return progress
