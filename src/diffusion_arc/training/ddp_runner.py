"""Run a bounded DDP training experiment with global batches and JSON output.

The input training_run recipe supplies data/model/optimizer/loss settings.
This entrypoint explicitly replaces execution and flow RNG assignment and does
not execute its validation, sampling, W&B, checkpoint or resume orchestration.
"""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import time
from importlib.metadata import version

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward
from .config import TrainingConfig
from .distributed import distributed_runtime
from .distributed_data import build_distributed_dataloader
from .profiling import torch_capture, torch_profile_settings, training_range
from .run.benchmarking import BenchmarkWindow, summarize_benchmark
from .run.memory_diagnostics import CudaMemoryHistory
from .setup import build_optimizer, load_models, seed_process
from .step.distributed_flow import DistributedFlowSampler, FLOW_SEED_SCHEME
from .step.flow_sampling import EndpointFlowPolicy
from .step.loss import build_velocity_loss
from .step.lr_schedule import build_lr_scheduler
from .trainer import Trainer


@dataclass(frozen=True)
class DistributedProgress:
    optimizer_step: int = 0
    sources_seen: int = 0
    targets_seen: int = 0


def _print_record(record: dict) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


def _gather(value, runtime) -> list:
    """Collect small setup/diagnostic records; model tensors stay on their rank."""
    values = [None] * runtime.world_size
    dist.all_gather_object(values, value, group=runtime.process_group)
    return values


def _training_settings(config: TrainingConfig) -> dict:
    """Record effective component settings, including Python API overrides."""
    components = {name: asdict(getattr(config, name))
                  for name in ("models", "data", "loop", "optimizer", "flux_packing")}
    components.update(seed=config.seed, nvarc_max_tokens=config.nvarc_max_tokens,
                      scheduler=None if config.scheduler is None else dict(
                          type=type(config.scheduler).__name__, **asdict(config.scheduler)))
    return json.loads(json.dumps(components, default=str, allow_nan=False))


def run_training_ddp(
    config: TrainingConfig, *, max_steps: int = 3,
    emit: Callable[[dict], None] = _print_record, source_identity: dict | None = None,
    expected_source_count: int | None = None, measurement: BenchmarkWindow | None = None,
    diagnostic_batch: int | None = None, trace_directory: Path | None = None,
    memory_history: bool = False,
    memory_snapshot_step: int | None = None,
    profile_warmup_steps: int | None = None,
) -> DistributedProgress:
    """Train a prefix of the recipe with torchrun, including world_size=1.

    optimizer_batch_size remains global. All ranks run the same updates and
    participate in metric aggregation; only rank zero emits JSON records.
    max_steps truncates execution, never the full-epoch LR or endpoint policy.
    An explicit source count permits a fixed benchmark view of the same recipe.
    Benchmark timing synchronizes ranks before each step and excludes outer
    barriers, diagnostic aggregation and output. All updates still train.

    diagnostic_batch selects a one-based global batch, repeated max_steps times
    from fresh weights/optimizer state with its original flow seed. It is not a
    checkpoint resume or a throughput benchmark. Every rank writes its own trace.
    memory_history records the full step sequence with bounded CUDA allocation
    history and a snapshot before distributed cleanup, on failure or completion.
    memory_snapshot_step additionally saves immediately after the first local
    backward in that global step, preserving its allocation history.
    profile_warmup_steps selects a separate Torch capture of the following update
    on every rank. Its instrumented timings are never benchmark measurements.
    """
    if config.config_type != "training_run" or config.recipe is None:
        raise ValueError("run_training_ddp requires a training_run recipe")
    if config.recipe["models"]["resume_from"] is not None:
        raise ValueError("the DDP training-only runner does not support resume")
    if type(max_steps) is not int or not 1 <= max_steps <= config.loop.max_optimizer_steps:
        raise ValueError("max_steps must be between one and the recipe's optimizer-step count")
    if measurement is not None and max_steps != measurement.total_steps:
        raise ValueError("max_steps must equal the benchmark warmup plus measured steps")
    if diagnostic_batch is not None and (
        type(diagnostic_batch) is not int or diagnostic_batch <= 0
        or measurement is not None or trace_directory is None
    ):
        raise ValueError("diagnostic_batch requires a positive batch number, trace directory and no benchmark")
    if type(memory_history) is not bool or memory_history and (
        trace_directory is None or measurement is not None or diagnostic_batch is not None
    ):
        raise ValueError("memory_history requires a trace directory and no benchmark or diagnostic batch")
    if memory_snapshot_step is not None and (
        type(memory_snapshot_step) is not int or not 1 <= memory_snapshot_step <= max_steps or not memory_history
    ):
        raise ValueError("memory_snapshot_step requires memory_history and a step within the executed prefix")
    profile_settings = None
    if profile_warmup_steps is not None:
        profile_settings = torch_profile_settings(profile_warmup_steps)
        if (trace_directory is None or measurement is not None or diagnostic_batch is not None
                or memory_history or max_steps != profile_warmup_steps + 1):
            raise ValueError("Torch profiling requires warmup + one step, a trace directory and no other diagnostics")
    with distributed_runtime(config) as runtime, (
        CudaMemoryHistory(trace_directory, runtime.rank, runtime.device, capture_step=memory_snapshot_step)
        if memory_history else nullcontext()
    ) as memory:
        trace = (memory.observe if memory_snapshot_step is not None else memory.trace) if memory else None
        if diagnostic_batch is not None:
            from .run.batch_diagnostics import BatchTrace
            trace = BatchTrace(trace_directory, runtime.rank, runtime.device)
        seed_process(config.seed)
        models = load_models(config, device=runtime.device)
        loader, plans = build_distributed_dataloader(
            config, tokenizer=models.tokenizer, rank=runtime.rank, world_size=runtime.world_size,
            expected_source_count=expected_source_count,
            diagnostic_batch=diagnostic_batch, diagnostic_repeats=max_steps,
        )
        settings = _training_settings(config)
        window = asdict(measurement) if measurement is not None else None
        identities = _gather((config.recipe_dict(), settings, loader.dataset.fingerprint, max_steps,
                              expected_source_count, window, diagnostic_batch, memory_history,
                              memory_snapshot_step, profile_settings), runtime)
        if any(identity != identities[0] for identity in identities):
            raise ValueError("DDP ranks have different training settings, recipes, data fingerprints or step limits")
        if diagnostic_batch is None and expected_source_count is None and len(plans) != config.loop.max_optimizer_steps:
            raise ValueError("global batch plan does not match the recipe's optimizer-step count")
        if len(plans) < max_steps:
            raise ValueError("the data view has too few global batches for this run")
        plans = plans[:max_steps]
        if expected_source_count is not None and any(
            batch.global_target_count != config.loop.optimizer_batch_size for batch in plans
        ):
            raise ValueError("benchmark data must provide full global optimizer batches")
        flow = config.recipe["flow"]
        endpoint_policy = EndpointFlowPolicy(
            config.loop.optimizer_batch_size, flow["endpoint_targets_per_full_batch"],
            config.final_batch_target_count, flow["endpoint_targets_in_final_batch"],
        )

        def sampler_for(batch):
            return DistributedFlowSampler(
                config.seed, batch.step, batch.global_target_count, batch.target_indices,
                endpoint_policy.endpoint_count(batch.global_target_count),
            )

        # DDP registers the same FLUX parameters that the optimizer updates.
        forward = PackedFluxForward(models.flux_transformer, compute_dtype=models.flux_compute_dtype,
                                    backend=models.flux_packing_backend)
        forward = DDP(forward, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                      process_group=runtime.process_group,
                      forward_sync_buffers=False, gradient_as_bucket_view=True)
        optimizer = build_optimizer(config, flux_transformer=models.flux_transformer)
        scheduler = build_lr_scheduler(config, optimizer=optimizer)
        trainer = Trainer(
            nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            flux_transformer=forward, optimizer=optimizer, scheduler=scheduler,
            flow_sampler=sampler_for(plans[0]), loss_function=build_velocity_loss(config),
            flux_compute_dtype=models.flux_compute_dtype, flux_packing_backend=models.flux_packing_backend,
            nvarc_max_tokens=config.nvarc_max_tokens, vae_microbatch_size=config.loop.vae_microbatch_size,
            flux_max_joint_tokens=config.flux_packing.max_tokens,
            flux_max_spread_tokens=config.flux_packing.max_spread_tokens,
            min_context_pairs_before_target=config.loop.min_context_pairs_before_target,
            max_grad_norm=config.optimizer.max_grad_norm,
            trace=trace,
        )
        devices = _gather(dict(rank=runtime.rank, device=str(runtime.device),
                               name=torch.cuda.get_device_name(runtime.device)
                               if runtime.device.type == "cuda" else "CPU test runtime",
                               total_memory_bytes=torch.cuda.get_device_properties(runtime.device).total_memory
                               if runtime.device.type == "cuda" else None,
                               allocator_backend=torch.cuda.memory.get_allocator_backend()
                               if runtime.device.type == "cuda" else None,
                               allocator_settings=torch.cuda.memory._snapshot()["allocator_settings"]
                               if runtime.device.type == "cuda" else None,
                               flux_backend=models.flux_packing_backend.binding
                               if runtime.device.type == "cuda" else type(models.flux_packing_backend).__name__,
                               torch_num_threads=torch.get_num_threads(),
                               fa3_version=version("flash-attn-3") if runtime.device.type == "cuda" else None), runtime)
        if runtime.rank == 0:
            emit(dict(event="start", mode=("ddp_torch_profile" if profile_settings else
                                          "ddp_memory_diagnostic" if memory else
                                          "ddp_batch_diagnostic" if diagnostic_batch is not None else
                                          "ddp_benchmark" if measurement else "ddp_training_only"),
                      memory_history=memory.settings if memory else None,
                      torch_profile=profile_settings,
                      diagnostic_batch=diagnostic_batch,
                      diagnostic_state=("full data sequence from step one; bounded CUDA allocation history; "
                                        "no per-pack CUDA synchronization" if memory else
                                        "fresh base weights and AdamW; repeated batch with original flow seed; "
                                        "ordinary LR warmup starts at update one; no per-pack CUDA synchronization"
                                        if trace else None),
                      measurement=window, expected_source_count=expected_source_count,
                      timing=("Torch Profiler timings include instrumentation; not a throughput benchmark" if profile_settings else
                              "diagnostic wall time includes inline tracing; not a throughput benchmark" if trace else
                              "rank barrier then CUDA-synchronized data retrieval and Trainer.step; maximum rank duration; diagnostics excluded"),
                      world_size=runtime.world_size,
                      backend=dist.get_backend(runtime.process_group), max_steps=max_steps,
                      global_batch_targets=config.loop.optimizer_batch_size,
                      flow_seed_scheme=FLOW_SEED_SCHEME, input_recipe=config.recipe_dict(),
                      effective_training=settings,
                      omitted_recipe_sections=["validation", "logging", "checkpointing"],
                      data_fingerprint=loader.dataset.fingerprint, devices=devices,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      python_version=platform.python_version(),
                      package_versions={name: version(name) for name in ("transformers", "diffusers", "accelerate")},
                      matmul_precision=torch.get_float32_matmul_precision(),
                      allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      source_identity=source_identity))

        progress = DistributedProgress()
        step_records = []
        samples = iter(loader)
        with (torch_capture(trace_directory, runtime.rank, runtime.device,
                            warmup_steps=profile_warmup_steps)
              if profile_settings else nullcontext()) as profiler:
            for repetition, batch in enumerate(plans, 1):
                trainer.flow_sampler = sampler_for(batch)
                # Align measurement starts after previous diagnostics, then include
                # data retrieval and the complete update with its DDP communication.
                with training_range("distributed/step_barrier"):
                    dist.barrier(group=runtime.process_group)
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                started = time.perf_counter()
                with training_range(f"optimizer_step/{repetition}"):
                    with training_range("data/retrieval"):
                        sequences = tuple(next(samples) for _ in batch.source_indices)
                    if trace:
                        trace("batch_start", dict(repetition=repetition, global_batch=batch.step + 1,
                                                  source_indices=batch.source_indices,
                                                  source_keys=tuple(str(s.key) for s in sequences)))
                    try:
                        result = trainer.step(sequences)
                    except BaseException as error:
                        if trace and not memory:
                            trace("failure", dict(error_type=type(error).__name__, message=str(error)))
                        raise
                    if runtime.device.type == "cuda":
                        torch.cuda.synchronize(runtime.device)
                seconds = time.perf_counter() - started
                if trace:
                    trace("batch_end", {})
                local = dict(rank=runtime.rank, loss=result.loss, targets=result.target_count,
                             sources=result.source_count, seconds=seconds,
                             trainer_seconds=result.step_seconds, gradient_norm=result.gradient_norm,
                             learning_rates=result.learning_rates, flux_packs=result.flux_forward_calls,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device)
                             if runtime.device.type == "cuda" else None,
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device)
                             if runtime.device.type == "cuda" else None,
                             source_indices=batch.source_indices, target_indices=batch.target_indices,
                             source_keys=tuple(source.key for source in result.diagnostics.preparation.sources),
                             target_prefix_tokens=result.diagnostics.target_prefix_tokens,
                             endpoint_targets=sum(target.timestep == 1.0 for target in result.targets))
                with training_range("distributed/metrics"):
                    ranks = _gather(local, runtime)
                targets = sum(rank["targets"] for rank in ranks)
                if targets != batch.global_target_count:
                    raise RuntimeError("executed target count differs from the global batch plan")
                progress = DistributedProgress(progress.optimizer_step + 1,
                                               progress.sources_seen + sum(rank["sources"] for rank in ranks),
                                               progress.targets_seen + targets)
                if runtime.rank == 0:
                    record = dict(event="step", optimizer_step=progress.optimizer_step,
                              global_batch=batch.step + 1,
                              phase=(("warmup" if repetition <= profile_warmup_steps else "profile") if profile_settings else
                                     "diagnostic" if trace else "smoke" if measurement is None else
                                     "warmup" if batch.step < measurement.warmup_steps else "measure"),
                              loss=sum(rank["loss"] * rank["targets"] for rank in ranks) / targets,
                              targets=targets, sources=sum(rank["sources"] for rank in ranks),
                              targets_seen=progress.targets_seen, sources_seen=progress.sources_seen,
                              max_rank_seconds=max(rank["seconds"] for rank in ranks), ranks=ranks)
                    step_records.append(record)
                    emit(record)
                if profiler is not None:
                    profiler.step()
        if runtime.rank == 0:
            emit(dict(event="complete", optimizer_steps=progress.optimizer_step,
                      sources_seen=progress.sources_seen, targets_seen=progress.targets_seen,
                      benchmark=summarize_benchmark(step_records)))
        return progress
