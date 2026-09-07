"""Initial consumer-first single-GPU training loop."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import islice

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.contracts import FlowMatchingBatch
from diffusion_arc.model.flow_matching import sample_flow_matching_batch
from diffusion_arc.model.flux_vae import FluxVaeEncoder
from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
from diffusion_arc.sequence_buckets import SequenceLengthBuckets

from .flux_input import FluxInput, PreparationMetrics, prepare_flux_inputs_batch
from .flux_loss import compute_flux_microbatch_loss
from .flux_microbatch import build_flux_microbatch
from .flux_schedule import FluxSchedulingPolicy, FluxTargetBatch, schedule_flux_targets
from .nvarc_schedule import NvarcSchedulingPolicy


@dataclass(frozen=True)
class OptimizerStepMetrics:
    """Measurements captured for one completed optimizer update."""

    loss: Tensor
    target_count: int
    microbatch_count: int
    gradient_norm: float
    gradient_clipped: bool
    learning_rates: tuple[float, ...]
    step_seconds: float


def _require_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _train_optimizer_batch(
    flux_inputs: Sequence[FluxInput],
    *,
    transformer: nn.Module,
    optimizer: Optimizer,
    flux_compute_dtype: torch.dtype,
    flux_microbatch_size: int | None,
    generator: torch.Generator | None,
    on_optimizer_step: Callable[[Tensor, int, int], None] | None,
    max_grad_norm: float | None,
    on_optimizer_step_metrics: Callable[[OptimizerStepMetrics], None] | None,
    flux_length_buckets: SequenceLengthBuckets,
    flux_scheduling: FluxSchedulingPolicy | None,
    on_flux_schedule: Callable[[tuple[FluxTargetBatch, ...]], None] | None,
) -> None:
    """Train one fixed optimizer batch through one or more FLUX microbatches."""

    # These metrics time FLUX and the optimizer only. The benchmark entrypoint
    # separately includes data loading and frozen-model preparation in its timer.
    if on_optimizer_step_metrics is not None:
        _synchronize_transformer_device(transformer)
        step_started = time.perf_counter()
    else:
        step_started = None
    optimizer_target_count = len(flux_inputs)
    reported_loss: Tensor | None = None
    microbatch_count = 0

    prefix_lengths = tuple(target.prefix_end for target in flux_inputs)
    target_flows = None
    if flux_scheduling is None:
        batches = tuple(
            FluxTargetBatch(
                tuple(range(start, min(start + flux_microbatch_size, optimizer_target_count))),
                flux_length_buckets.padded_length(max(prefix_lengths[start : start + flux_microbatch_size])),
            )
            for start in range(0, optimizer_target_count, flux_microbatch_size)
        )
    else:
        batches = schedule_flux_targets(prefix_lengths, flux_scheduling)
        # Assign randomness in FIFO order using the same per-target draws as B1.
        # Execution order and physical batch size then cannot reassign noise/time.
        target_flows = tuple(
            sample_flow_matching_batch(target.clean_latent.unsqueeze(0), generator=generator)
            for target in flux_inputs
        )
    # Benchmark metadata: report the actual plan, indexed in logical FIFO order.
    if on_flux_schedule is not None:
        on_flux_schedule(batches)

    for batch in batches:
        microbatch = build_flux_microbatch(
            tuple(flux_inputs[index] for index in batch.target_indices),
            padded_length=batch.padded_length,
        )
        flow_kwargs = {}
        if target_flows is not None:
            selected = tuple(target_flows[index] for index in batch.target_indices)
            flow_kwargs["flow_batch"] = FlowMatchingBatch(
                noisy_latents=torch.cat(tuple(flow.noisy_latents for flow in selected)),
                timesteps=torch.cat(tuple(flow.timesteps for flow in selected)),
                target_velocity=torch.cat(tuple(flow.target_velocity for flow in selected)),
            )
        loss = compute_flux_microbatch_loss(
            transformer,
            microbatch,
            compute_dtype=flux_compute_dtype,
            generator=generator,
            **flow_kwargs,
        )
        if loss.ndim != 0:
            raise RuntimeError("FLUX microbatch loss must be a scalar")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("FLUX microbatch loss is non-finite")
        # Weight microbatch means by target count to preserve the optimizer-batch
        # mean, including a smaller final microbatch. Use it for loss and gradients.
        weight = microbatch.target_count / optimizer_target_count
        loss_contribution = loss.detach() * weight
        if reported_loss is None:
            reported_loss = loss_contribution
        else:
            reported_loss = reported_loss + loss_contribution

        (loss * weight).backward()
        microbatch_count += 1

    gradient_norm: float | None = None
    gradient_clipped = False
    # Gradient metrics and optional clipping before the optimizer update.
    if max_grad_norm is not None or on_optimizer_step_metrics is not None:
        trainable_parameters = tuple(
            parameter
            for parameter in transformer.parameters()
            if parameter.requires_grad
        )
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=math.inf if max_grad_norm is None else max_grad_norm,
            error_if_nonfinite=True,
            foreach=False,
        )
        gradient_norm = float(gradient_norm_tensor.detach().item())
        gradient_clipped = bool(
            max_grad_norm is not None and gradient_norm > max_grad_norm
        )
    optimizer.step()
    # Optional step metrics: finish timing after the optimizer update.
    if on_optimizer_step_metrics is not None:
        _synchronize_transformer_device(transformer)
        if step_started is None:
            raise AssertionError("metrics timing did not record a start")
        step_seconds = time.perf_counter() - step_started

    if reported_loss is None:
        raise AssertionError("non-empty optimizer batch produced no loss")
    # Step callback: report the update and let the entrypoint advance its schedules.
    if on_optimizer_step is not None:
        on_optimizer_step(
            reported_loss,
            optimizer_target_count,
            microbatch_count,
        )
    # Metrics callback: publish timing, gradient, and learning-rate measurements.
    if on_optimizer_step_metrics is not None:
        if gradient_norm is None:
            raise AssertionError("metrics callback requires a gradient norm")
        on_optimizer_step_metrics(
            OptimizerStepMetrics(
                loss=reported_loss,
                target_count=optimizer_target_count,
                microbatch_count=microbatch_count,
                gradient_norm=gradient_norm,
                gradient_clipped=gradient_clipped,
                learning_rates=tuple(
                    float(group["lr"]) for group in optimizer.param_groups
                ),
                step_seconds=step_seconds,
            )
        )

    # Callbacks can inspect the completed gradients; frozen preparation cannot use them.
    optimizer.zero_grad(set_to_none=True)


def _synchronize_transformer_device(transformer: nn.Module) -> None:
    """Synchronize CUDA timing while remaining a no-op for CPU tests."""

    try:
        parameter = next(transformer.parameters())
    except StopIteration:
        return
    if parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)


def run_single_gpu_training(
    training_samples: Iterable[ArcTokenizedTrainingSample],
    *,
    nvarc_runner: NvarcBatchRunner,
    vae_encoder: FluxVaeEncoder,
    transformer: nn.Module,
    optimizer: Optimizer,
    flux_compute_dtype: torch.dtype,
    max_optimizer_steps: int,
    optimizer_batch_size: int,
    flux_microbatch_size: int | None = 1,
    min_context_pairs_before_target: int = 1,
    nvarc_source_batch_size: int = 1,
    vae_microbatch_size: int | None = None,
    flux_bucket_boundaries: tuple[int, ...] | None = None,
    nvarc_scheduling: NvarcSchedulingPolicy | None = None,
    flux_scheduling: FluxSchedulingPolicy | None = None,
    on_preparation: Callable[[PreparationMetrics], None] | None = None,
    on_flux_schedule: Callable[[tuple[FluxTargetBatch, ...]], None] | None = None,
    generator: torch.Generator | None = None,
    on_optimizer_step: Callable[[Tensor, int, int], None] | None = None,
    max_grad_norm: float | None = None,
    on_optimizer_step_metrics: Callable[[OptimizerStepMetrics], None] | None = None,
) -> None:
    """Train from one sample stream until its end or an exact step limit.

    ``nvarc_source_batch_size`` limits the number of consecutive raw samples in
    each preparation window, before eligibility filtering. Optional scheduling
    subdivides that window into physical NVARC batches; returned targets retain
    logical source order. Window boundaries do not define optimizer batches:
    the target FIFO carries any surplus into the next update.

    A complete optimizer batch contains ``optimizer_batch_size`` targets,
    processed in FLUX microbatches. Stream exhaustion flushes a smaller tail.
    Reaching ``max_optimizer_steps`` stops without consuming that tail.
    ``flux_scheduling`` reorders only the selected optimizer batch and requires
    ``flux_microbatch_size=None`` with no separate FLUX padding boundaries.
    """

    _require_positive_integer(max_optimizer_steps, name="max_optimizer_steps")
    _require_positive_integer(optimizer_batch_size, name="optimizer_batch_size")
    if flux_scheduling is None:
        _require_positive_integer(flux_microbatch_size, name="flux_microbatch_size")
    elif flux_microbatch_size is not None or flux_bucket_boundaries is not None:
        raise ValueError("flux_scheduling requires null flux_microbatch_size and flux_bucket_boundaries")
    _require_positive_integer(
        nvarc_source_batch_size,
        name="nvarc_source_batch_size",
    )
    _require_positive_integer(
        min_context_pairs_before_target,
        name="min_context_pairs_before_target",
    )
    if vae_microbatch_size is not None:
        _require_positive_integer(
            vae_microbatch_size,
            name="vae_microbatch_size",
        )
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if on_optimizer_step is not None and not callable(on_optimizer_step):
        raise TypeError("on_optimizer_step must be callable")
    if max_grad_norm is not None:
        if (
            isinstance(max_grad_norm, bool)
            or not isinstance(max_grad_norm, (int, float))
            or not math.isfinite(max_grad_norm)
            or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be a finite positive number or None")
        max_grad_norm = float(max_grad_norm)
    if on_optimizer_step_metrics is not None and not callable(
        on_optimizer_step_metrics
    ):
        raise TypeError("on_optimizer_step_metrics must be callable")

    flux_length_buckets = SequenceLengthBuckets(flux_bucket_boundaries)
    optimizer.zero_grad(set_to_none=True)
    transformer.train()
    flux_input_buffer: deque[FluxInput] = deque()
    optimizer_steps = 0
    training_iterator = iter(training_samples)
    stream_exhausted = False

    while optimizer_steps < max_optimizer_steps:
        source_batch = tuple(islice(training_iterator, nvarc_source_batch_size))
        if not source_batch:
            stream_exhausted = True
            break
        flux_input_buffer.extend(
            prepare_flux_inputs_batch(
                source_batch,
                nvarc_runner=nvarc_runner,
                vae_encoder=vae_encoder,
                min_context_pairs_before_target=min_context_pairs_before_target,
                vae_microbatch_size=vae_microbatch_size,
                nvarc_scheduling=nvarc_scheduling,
                on_preparation=on_preparation,
            )
        )

        while (
            len(flux_input_buffer) >= optimizer_batch_size
            and optimizer_steps < max_optimizer_steps
        ):
            optimizer_inputs = tuple(
                flux_input_buffer.popleft()
                for _ in range(optimizer_batch_size)
            )
            _train_optimizer_batch(
                optimizer_inputs,
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=flux_compute_dtype,
                flux_microbatch_size=flux_microbatch_size,
                generator=generator,
                on_optimizer_step=on_optimizer_step,
                max_grad_norm=max_grad_norm,
                on_optimizer_step_metrics=on_optimizer_step_metrics,
                flux_length_buckets=flux_length_buckets,
                flux_scheduling=flux_scheduling,
                on_flux_schedule=on_flux_schedule,
            )
            optimizer_steps += 1

    if stream_exhausted and flux_input_buffer and optimizer_steps < max_optimizer_steps:
        _train_optimizer_batch(
            tuple(flux_input_buffer),
            transformer=transformer,
            optimizer=optimizer,
            flux_compute_dtype=flux_compute_dtype,
            flux_microbatch_size=flux_microbatch_size,
            generator=generator,
            on_optimizer_step=on_optimizer_step,
            max_grad_norm=max_grad_norm,
            on_optimizer_step_metrics=on_optimizer_step_metrics,
            flux_length_buckets=flux_length_buckets,
            flux_scheduling=flux_scheduling,
            on_flux_schedule=on_flux_schedule,
        )


__all__ = ["OptimizerStepMetrics", "run_single_gpu_training"]
