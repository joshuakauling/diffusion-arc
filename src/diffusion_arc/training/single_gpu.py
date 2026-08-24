"""Initial consumer-first single-GPU training policy."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.flux_image import FluxVaeEncoder
from diffusion_arc.model.nvarc import NvarcBatchRunner

from .flux_forward import compute_flux_microbatch_loss
from .flux_input import FluxInput, prepare_flux_inputs
from .flux_microbatch import build_flux_microbatch


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
) -> None:
    """Train one fixed optimizer batch through one or more FLUX microbatches."""

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
        weight = microbatch.target_count / optimizer_target_count
        if on_optimizer_step is not None:
            loss_contribution = loss.detach() * weight
            if reported_loss is None:
                reported_loss = loss_contribution
            else:
                reported_loss = reported_loss + loss_contribution

        (loss * weight).backward()
        microbatch_count += 1

    optimizer.step()

    if on_optimizer_step is not None:
        if reported_loss is None:
            raise AssertionError("non-empty optimizer batch produced no loss")
        on_optimizer_step(
            reported_loss,
            optimizer_target_count,
            microbatch_count,
        )


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
    generator: torch.Generator | None = None,
    on_optimizer_step: Callable[[Tensor, int, int], None] | None = None,
) -> None:
    """Train from one sample stream until its end or an exact step limit.

    A FIFO queue holds prepared targets until a complete optimizer batch is
    available. Stream exhaustion flushes one smaller tail batch. Reaching
    ``max_optimizer_steps`` stops immediately without consuming that tail.
    """

    _require_positive_integer(max_optimizer_steps, name="max_optimizer_steps")
    _require_positive_integer(optimizer_batch_size, name="optimizer_batch_size")
    _require_positive_integer(flux_microbatch_size, name="flux_microbatch_size")
    _require_positive_integer(
        min_context_pairs_before_target,
        name="min_context_pairs_before_target",
    )
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    if on_optimizer_step is not None and not callable(on_optimizer_step):
        raise TypeError("on_optimizer_step must be callable")

    transformer.train()
    flux_input_buffer: deque[FluxInput] = deque()
    optimizer_steps = 0

    for sample in training_samples:
        flux_input_buffer.extend(
            prepare_flux_inputs(
                sample,
                nvarc_runner=nvarc_runner,
                vae_encoder=vae_encoder,
                min_context_pairs_before_target=min_context_pairs_before_target,
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
            )
            optimizer_steps += 1

        if optimizer_steps == max_optimizer_steps:
            break
    else:
        if flux_input_buffer and optimizer_steps < max_optimizer_steps:
            _train_optimizer_batch(
                tuple(flux_input_buffer),
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=flux_compute_dtype,
                flux_microbatch_size=flux_microbatch_size,
                generator=generator,
                on_optimizer_step=on_optimizer_step,
            )


__all__ = ["run_single_gpu_training"]
