"""Benchmark and profiling instrumentation attached to the shared training loop."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import torch
import torch.distributed as dist

from ..profiling import torch_capture, torch_profile_settings
from .benchmarking import BenchmarkWindow, summarize_benchmark
from .memory_diagnostics import CudaMemoryHistory


def print_record(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def training_settings(config):
    components = {name: asdict(getattr(config, name))
                  for name in ("models", "data", "loop", "optimizer", "flux_packing")}
    components.update(seed=config.seed, nvarc_max_tokens=config.nvarc_max_tokens,
                      scheduler=None if config.scheduler is None else dict(
                          type=type(config.scheduler).__name__, **asdict(config.scheduler)))
    return json.loads(json.dumps(components, default=str, allow_nan=False))


@dataclass
class BenchmarkInstrumentation:
    """Explicitly replace full-run services with bounded, recorded measurements."""

    max_steps: int = 3
    emit: Callable = print_record
    source_identity: dict | None = None
    expected_source_count: int | None = None
    measurement: BenchmarkWindow | None = None
    diagnostic_batch_number: int | None = None
    trace_directory: Path | None = None
    memory_history: bool = False
    profile_warmup_steps: int | None = None
    input_recipe: dict | None = None
    precompute_packing: bool = False
    fsdp_gradient_sync: str = "last_microbatch"

    def validate(self, config):
        if config.recipe["models"]["resume_from"] is not None:
            raise ValueError("bounded experiments do not support resume")
        if config.recipe["models"]["initialization"] != "base_checkpoint":
            raise ValueError("bounded experiments require base checkpoint initialization")
        if type(self.precompute_packing) is not bool:
            raise ValueError("precompute_packing must be a boolean")
        if self.fsdp_gradient_sync not in ("last_microbatch", "per_microbatch"):
            raise ValueError("fsdp_gradient_sync must be last_microbatch or per_microbatch")
        if (self.fsdp_gradient_sync == "per_microbatch" and
                config.recipe["execution"]["distributed_strategy"] != "fsdp2"):
            raise ValueError("per-microbatch gradient synchronization requires fsdp2")
        if self.precompute_packing and (
                config.recipe["execution"]["distributed_strategy"] not in ("ddp", "fsdp2") or
                config.recipe["flow"]["rng_scheme"] != "global_step_target_v1" or
                config.rank_batch_assignment != "token_balanced_v1" or
                config.flux_packing.algorithm != "legacy_bfd_v1"):
            raise ValueError("precomputed packing requires distributed indexed flow, token balancing and legacy BFD")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= config.loop.max_optimizer_steps:
            raise ValueError("max_steps must be between one and the recipe's optimizer-step count")
        if self.measurement is not None and self.max_steps != self.measurement.total_steps:
            raise ValueError("max_steps must equal the benchmark warmup plus measured steps")
        if self.diagnostic_batch_number is not None and (
            type(self.diagnostic_batch_number) is not int or self.diagnostic_batch_number <= 0
            or self.measurement is not None or self.trace_directory is None
        ):
            raise ValueError("diagnostic_batch_number requires a positive batch number, trace directory and no benchmark")
        if type(self.memory_history) is not bool or self.memory_history and (
            self.trace_directory is None or self.measurement is not None or self.diagnostic_batch_number is not None
        ):
            raise ValueError("memory_history requires a trace directory and no benchmark or diagnostic batch")
        self.profile_settings = None
        if self.profile_warmup_steps is not None:
            self.profile_settings = torch_profile_settings(self.profile_warmup_steps)
            if (self.trace_directory is None or self.measurement is not None or self.diagnostic_batch_number is not None
                    or self.memory_history or self.max_steps != self.profile_warmup_steps + 1):
                raise ValueError("Torch profiling requires warmup + one step, a trace directory and no other diagnostics")

    def open(self, stack: ExitStack, runtime):
        self.runtime, self.records, self.memory, self.profiler = runtime, [], None, None
        if self.memory_history:
            self.memory = stack.enter_context(CudaMemoryHistory(
                self.trace_directory, runtime.rank, runtime.device))

    def start(self, stack, config, models, dataset):
        runtime = self.runtime
        self.background_packing = config.rank_batch_assignment == "joint_packing_v1"
        devices = runtime.all_gather_objects(dict(
            rank=runtime.rank, device=str(runtime.device),
            name=torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else "CPU test runtime",
            total_memory_bytes=torch.cuda.get_device_properties(runtime.device).total_memory if runtime.device.type == "cuda" else None,
            allocator_backend=torch.cuda.memory.get_allocator_backend() if runtime.device.type == "cuda" else None,
            allocator_settings=torch.cuda.memory._snapshot()["allocator_settings"] if runtime.device.type == "cuda" else None,
            flux_backend=models.flux_packing_backend.binding if runtime.device.type == "cuda" else type(models.flux_packing_backend).__name__,
            torch_num_threads=torch.get_num_threads(),
            fa3_version=version("flash-attn-3") if runtime.device.type == "cuda" else None))
        if runtime.is_primary:
            prefix = runtime.strategy
            self.emit(dict(event="start", mode=(f"{prefix}_torch_profile" if self.profile_settings else
                           f"{prefix}_memory_diagnostic" if self.memory else f"{prefix}_batch_diagnostic" if self.diagnostic_batch_number is not None else
                           f"{prefix}_benchmark" if self.measurement else f"{prefix}_training_only"),
                memory_history=self.memory.settings if self.memory else None, torch_profile=self.profile_settings,
                precompute_packing=self.precompute_packing,
                fsdp_gradient_sync=self.fsdp_gradient_sync if runtime.strategy == "fsdp2" else None,
                diagnostic_batch=self.diagnostic_batch_number,
                diagnostic_state=("full data sequence from step one; bounded CUDA allocation history; no per-pack CUDA synchronization"
                                  if self.memory else "fresh base weights and AdamW; repeated batch with original flow seed; ordinary LR warmup starts at update one; no per-pack CUDA synchronization"
                                  if self.diagnostic_batch_number is not None else None),
                measurement=asdict(self.measurement) if self.measurement else None,
                expected_source_count=self.expected_source_count,
                timing=("Torch Profiler timings include instrumentation; not a throughput benchmark" if self.profile_settings else
                        "CUDA allocation history is enabled; not a throughput benchmark" if self.memory else
                        "rank barrier then CUDA-synchronized data retrieval and Trainer.step; maximum rank duration; background plan waits during retrieval included; first plan and initial DataLoader prefetch excluded; diagnostics excluded" if self.background_packing else
                        "rank barrier then CUDA-synchronized data retrieval and Trainer.step; maximum rank duration; diagnostics excluded"),
                world_size=runtime.world_size, backend=dist.get_backend(runtime.process_group) if runtime.process_group is not None else None,
                max_steps=self.max_steps, global_batch_targets=config.loop.optimizer_batch_size,
                flow_seed_scheme=config.recipe["flow"]["rng_scheme"],
                input_recipe=self.input_recipe if self.input_recipe is not None else config.recipe_dict(),
                effective_recipe=config.recipe_dict(),
                effective_training=training_settings(config), omitted_recipe_sections=["validation", "logging", "checkpointing"],
                data_fingerprint=dataset.fingerprint, devices=devices, torch_version=torch.__version__,
                rank_batch_assignment=getattr(dataset, "rank_assignment_metadata", None),
                cuda_version=torch.version.cuda, python_version=platform.python_version(),
                package_versions={name: version(name) for name in ("transformers", "diffusers", "accelerate")},
                matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), source_identity=self.source_identity))
            if self.precompute_packing:
                self.emit(dict(event="packing_plan",
                    metadata=dataset.rank_assignment_metadata["precomputed_packing"],
                    planning_seconds=dataset.precomputed_packing_seconds,
                    timing="CPU planning before the measured steps; excluded from throughput",
                    ranks=[[asdict(plan) for plan in rank_plans]
                           for rank_plans in dataset.precomputed_packing_plans]))
        if self.profile_settings:
            self.profiler = stack.enter_context(torch_capture(self.trace_directory, runtime.rank, runtime.device,
                                                             warmup_steps=self.profile_warmup_steps))

    def before_step(self):
        self.runtime.barrier()
        if self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)
            torch.cuda.reset_peak_memory_stats(self.runtime.device)
        self.started = time.perf_counter()

    def completed_step(self):
        if self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)
        self.seconds = time.perf_counter() - self.started
        self.peaks = {name: read(self.runtime.device) if self.runtime.device.type == "cuda" else None
                      for name, read in (("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                                         ("peak_reserved_bytes", torch.cuda.max_memory_reserved))}

    def after_step(self, run_step, batch_plan, result, progress):
        runtime = self.runtime
        seconds = self.seconds
        local = dict(rank=runtime.rank, loss=result.loss, targets=result.target_count, sources=result.source_count,
            seconds=seconds, trainer_seconds=result.step_seconds, gradient_norm=result.gradient_norm,
            learning_rates=result.learning_rates, flux_packs=result.flux_forward_calls,
            padding_forward_calls=result.padding_forward_calls,
            **self.peaks,
            source_indices=batch_plan.source_indices, target_indices=batch_plan.target_indices,
            source_keys=tuple(source.key for source in result.diagnostics.preparation.sources),
            target_prefix_tokens=result.diagnostics.target_prefix_tokens,
            endpoint_targets=sum(target.timestep == 1.0 for target in result.target_metrics))
        if self.background_packing:
            # Persist each executed plan without materializing future batches.
            local["flux_plan"] = asdict(batch_plan.flux_plan)
        ranks = runtime.all_gather_objects(local)
        targets = sum(rank["targets"] for rank in ranks)
        if targets != batch_plan.global_target_count:
            raise RuntimeError("executed target count differs from the global batch plan")
        if runtime.is_primary:
            record = dict(event="step", optimizer_step=progress.optimizer_step, global_batch=batch_plan.epoch_batch_index + 1,
                phase=(("warmup" if run_step <= self.profile_warmup_steps else "profile") if self.profile_settings else
                       "diagnostic" if self.memory or self.diagnostic_batch_number is not None else "smoke" if self.measurement is None else
                       "warmup" if batch_plan.epoch_batch_index < self.measurement.warmup_steps else "measure"),
                loss=sum(rank["loss"] * rank["targets"] for rank in ranks) / targets,
                targets=targets, sources=sum(rank["sources"] for rank in ranks), targets_seen=progress.targets_seen,
                sources_seen=progress.sources_seen, max_rank_seconds=max(rank["seconds"] for rank in ranks), ranks=ranks)
            self.records.append(record)
            self.emit(record)
        if self.profiler is not None:
            self.profiler.step()

    def finish(self, progress):
        if self.runtime.is_primary:
            self.emit(dict(event="complete", optimizer_steps=progress.optimizer_step,
                sources_seen=progress.sources_seen, targets_seen=progress.targets_seen,
                benchmark=summarize_benchmark(self.records)))
