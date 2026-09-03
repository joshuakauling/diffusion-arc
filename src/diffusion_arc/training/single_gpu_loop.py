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
from diffusion_arc.model.flux_vae import FluxVaeEncoder
from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner

from .flux_input import FluxInput, prepare_flux_inputs_batch
from .flux_loss import compute_flux_microbatch_loss
from .flux_microbatch import build_flux_microbatch


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
    flux_microbatch_size: int,
    generator: torch.Generator | None,
    on_optimizer_step: Callable[[Tensor, int, int], None] | None,
    max_grad_norm: float | None,
    on_optimizer_step_metrics: Callable[[OptimizerStepMetrics], None] | None,
) -> None:
    """Train one fixed optimizer batch through one or more FLUX microbatches."""

    if on_optimizer_step_metrics is not None:
        _synchronize_transformer_device(transformer)
        step_started = time.perf_counter()
    else:
        step_started = None
    optimizer_target_count = len(flux_inputs)
    optimizer.zero_grad(set_to_none=True)
    reported_loss: Tensor | None = None
    microbatch_count = 0

    for start in range(0, optimizer_target_count, flux_microbatch_size):
        microbatch = build_flux_microbatch(
            flux_inputs[start : start + flux_microbatch_size]
        )
        loss = compute_flux_microbatch_loss(
            transformer,
            microbatch,
            compute_dtype=flux_compute_dtype,
            generator=generator,
        )
        if loss.ndim != 0:
            raise RuntimeError("FLUX microbatch loss must be a scalar")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("FLUX microbatch loss is non-finite")
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
    if on_optimizer_step_metrics is not None:
        _synchronize_transformer_device(transformer)
        if step_started is None:
            raise AssertionError("metrics timing did not record a start")
        step_seconds = time.perf_counter() - step_started

    if reported_loss is None:
        raise AssertionError("non-empty optimizer batch produced no loss")
    if on_optimizer_step is not None:
        on_optimizer_step(
            reported_loss,
            optimizer_target_count,
            microbatch_count,
        )
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
    flux_microbatch_size: int = 1,
    min_context_pairs_before_target: int = 1,
    nvarc_source_batch_size: int = 1,
    vae_microbatch_size: int | None = None,
    generator: torch.Generator | None = None,
    on_optimizer_step: Callable[[Tensor, int, int], None] | None = None,
    max_grad_norm: float | None = None,
    on_optimizer_step_metrics: Callable[[OptimizerStepMetrics], None] | None = None,
) -> None:
    """Train from one sample stream until its end or an exact step limit.

    Consecutive source sequences are prepared in fixed NVARC batches without
    reordering. A FIFO queue holds their targets until a complete optimizer
    batch is available. Stream exhaustion flushes one smaller tail batch.
    Reaching ``max_optimizer_steps`` stops without consuming that tail.
    """

    _require_positive_integer(max_optimizer_steps, name="max_optimizer_steps")
    _require_positive_integer(optimizer_batch_size, name="optimizer_batch_size")
    _require_positive_integer(flux_microbatch_size, name="flux_microbatch_size")
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
        )


__all__ = ["OptimizerStepMetrics", "run_single_gpu_training"]
