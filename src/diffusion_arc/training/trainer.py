"""Train FLUX on one optimizer batch, from NVARC/VAE encoding to the update."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from diffusion_arc._constants import DEFAULT_NVARC_MAX_TOKENS
from diffusion_arc.arc_data.canvas import CANVAS_SIZE
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.contracts import TARGET_LATENT_TOKEN_COUNT
from diffusion_arc.model.flux.velocity import flux_transformer_device

from .step.encoding import FluxInput, PreparationMetrics, encode_training_batch
from .step.flow_sampling import FlowMatchingBatch, select_flow_batch
from .step.flux_microbatch import compute_flux_microbatch_loss
from .step.flux_batching import FluxPackingPolicy, FluxTargetBatch, plan_flux_batches
from .step.loss import LossResult, VelocityLoss
from .profiling import training_range


@dataclass(frozen=True)
class TargetMetrics:
    """Loss and sampling details for one target, stored as Python values.

    ``timestep`` is the sampled flow time; ``context_pair_count`` is the number
    of preceding demonstration pairs. ``mse_*`` holds unweighted velocity errors
    for each region, and ``contribution_*`` holds their weighted loss terms.

    Regional values are None when the loss does not provide them. With the
    regional loss, a 30x30 grid has ``has_boundary=False``, a boundary MSE of
    None and a boundary contribution of zero.
    """

    loss: float
    timestep: float
    context_pair_count: int
    has_boundary: bool
    mse_canvas: float | None = None
    mse_grid: float | None = None
    mse_boundary: float | None = None
    contribution_canvas: float | None = None
    contribution_grid: float | None = None
    contribution_boundary: float | None = None


@dataclass(frozen=True)
class StepDiagnostics:
    """Batch layout details used to inspect and compare benchmark runs.

    ``preparation`` records source identities, lengths and NVARC/VAE batching.
    ``flux_batches`` records which targets shared each FLUX call. Its target
    indices and ``target_prefix_tokens`` refer to positions in
    ``TrainStepResult.targets``. These details contain no model tensors.
    """

    preparation: PreparationMetrics
    flux_batches: tuple[FluxTargetBatch, ...]
    target_prefix_tokens: tuple[int, ...]


@dataclass(frozen=True)
class TrainStepResult:
    """Measurements returned after one optimizer update.

    All counters and timings describe this step only. Learning rates are the
    values used for the update; the gradient norm is measured before clipping.
    ``targets`` contains individual measurements in input sequence order, then
    target order within each sequence. Its tuple position is the target index.

    The caller can group these measurements for logging and track run progress.
    Detailed batch layouts are separate under ``diagnostics``. The result holds
    ordinary Python values, with no tensors or gradient graphs.
    """

    loss: float
    learning_rates: tuple[float, ...]
    gradient_norm: float
    gradient_clipped: bool
    step_seconds: float
    source_count: int
    target_count: int
    nvarc_tokens: int
    nvarc_calls: int
    vae_calls: int
    flux_text_tokens: int
    flux_image_tokens: int
    flux_joint_tokens: int
    flux_forward_calls: int
    targets: tuple[TargetMetrics, ...]
    diagnostics: StepDiagnostics


class _StepLosses:
    """Collect already-computed losses across the FLUX microbatches of one step.

    Store measurements in original target order and detach them from autograd.
    The trainer uses them to build its result after the optimizer update.
    """

    def __init__(self, target_count: int, device: torch.device) -> None:
        """Allocate loss storage for all targets in this optimizer batch."""
        self.target_count = target_count
        self.total: Tensor | None = None
        self.per_target = torch.empty(target_count, device=device, dtype=torch.float32)
        self.regions: Tensor | None = None

    def record(self, indices: tuple[int, ...], result: LossResult) -> None:
        """Store one microbatch's measurements at its original target indices.

        Add its mean loss to the reported total, weighted by its share of the
        step's target count. Backpropagation is handled by ``Trainer.step``.
        """
        contribution = result.loss.detach() * (len(indices) / self.target_count)
        self.total = contribution if self.total is None else self.total + contribution
        self.per_target[list(indices)] = result.per_target.detach()
        if result.regions is not None:
            if self.regions is None:
                self.regions = torch.empty((self.target_count, 6), device=result.loss.device, dtype=torch.float32)
            region = result.regions
            self.regions[list(indices)] = torch.cat((
                region.region_means, region.contributions,
            ), dim=1).detach()

    def target_metrics(
        self, targets: Sequence[FluxInput], flows: Sequence[FlowMatchingBatch],
    ) -> tuple[TargetMetrics, ...]:
        """Combine recorded losses, flow times and context counts into CPU records.

        Return one ``TargetMetrics`` per target in original input order.
        Grouping and averaging for logging are left to the caller.
        """
        # Transfer small detached measurements together, then discard all tensor owners.
        times = torch.cat(tuple(flow.timesteps for flow in flows)).detach()
        rows = torch.cat((self.per_target[:, None], times[:, None],
                          *(() if self.regions is None else (self.regions,))), dim=1).cpu().tolist()
        metrics = []
        for target, row in zip(targets, rows, strict=True):
            has_boundary = target.grid_shape != (CANVAS_SIZE, CANVAS_SIZE)
            regions = {}
            if self.regions is not None:
                canvas, grid, boundary, canvas_part, grid_part, boundary_part = row[2:]
                regions = dict(
                    mse_canvas=canvas, mse_grid=grid,
                    mse_boundary=boundary if has_boundary else None,
                    contribution_canvas=canvas_part, contribution_grid=grid_part,
                    contribution_boundary=boundary_part,
                )
            metrics.append(TargetMetrics(
                loss=row[0], timestep=row[1], context_pair_count=target.context_pair_count,
                has_boundary=has_boundary, **regions,
            ))
        return tuple(metrics)


class Trainer:
    """Run one optimizer update for each batch of sequences passed to ``step``.

    Holds the models, optimizer, scheduler, flow sampler, loss function and
    execution limits. NVARC and VAE remain frozen; only FLUX is trained.
    The outer loop selects data, logs returned measurements and handles
    validation, sampling and checkpoints.
    """

    def __init__(
        self,
        *,
        nvarc_encoder: nn.Module,
        vae_encoder: nn.Module,
        flux_transformer: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        flow_sampler: Callable[[Sequence[Tensor]], tuple[FlowMatchingBatch, ...]],
        loss_function: VelocityLoss,
        flux_compute_dtype: torch.dtype,
        flux_packing_backend: object,
        nvarc_max_tokens: int = DEFAULT_NVARC_MAX_TOKENS,
        vae_microbatch_size: int | None = None,
        flux_max_joint_tokens: int = 12_288,
        min_context_pairs_before_target: int = 1,
        max_grad_norm: float | None = None,
    ) -> None:
        """Check the supplied components and prepare them for training.

        NVARC and VAE must already be frozen. Set them to evaluation mode, set
        FLUX to training mode and clear any existing optimizer gradients.
        NVARC batches are limited by source tokens, FLUX batches by text plus
        image tokens, and VAE batches by the number of target images.
        """
        for name, value in (("nvarc_max_tokens", nvarc_max_tokens),
                            ("min_context_pairs_before_target", min_context_pairs_before_target)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vae_microbatch_size is not None and (type(vae_microbatch_size) is not int or vae_microbatch_size <= 0):
            raise ValueError("vae_microbatch_size must be a positive integer or None")
        if max_grad_norm is not None and (isinstance(max_grad_norm, bool)
                or not isinstance(max_grad_norm, (int, float)) or not math.isfinite(max_grad_norm)
                or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive or None")
        if flux_packing_backend is None:
            raise ValueError("training requires a preflighted FA3 backend")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("scheduler must belong to the supplied optimizer")
        if not callable(flow_sampler) or not callable(loss_function):
            raise TypeError("flow_sampler and loss_function must be callable")
        if any(p.requires_grad for model in (nvarc_encoder, vae_encoder) for p in model.parameters()):
            raise ValueError("NVARC and VAE must already be frozen")
        self.packing = FluxPackingPolicy(flux_max_joint_tokens)
        self.nvarc_encoder, self.vae_encoder, self.flux_transformer = nvarc_encoder, vae_encoder, flux_transformer
        self.optimizer, self.scheduler = optimizer, scheduler
        self.flow_sampler, self.loss_function = flow_sampler, loss_function
        self.flux_compute_dtype, self.flux_packing_backend = flux_compute_dtype, flux_packing_backend
        self.nvarc_max_tokens, self.vae_microbatch_size = nvarc_max_tokens, vae_microbatch_size
        self.min_context_pairs_before_target, self.max_grad_norm = min_context_pairs_before_target, max_grad_norm
        self.parameters = tuple(p for p in flux_transformer.parameters() if p.requires_grad)
        self.device = flux_transformer_device(flux_transformer)
        nvarc_encoder.eval()
        vae_encoder.eval()
        flux_transformer.train()
        optimizer.zero_grad(set_to_none=True)

    def step(self, sequences: Sequence[ArcTokenizedTrainingSample]) -> TrainStepResult:
        """Train on the supplied sequences with one optimizer update.

        Encode sequences with frozen NVARC and target images with frozen VAE.
        Sample flow times and noise for eligible targets, then pack them into
        FLUX microbatches. Accumulate gradients with equal weight per target,
        optionally clip them, update parameters once and advance the supplied
        scheduler once.

        Clear gradients after the update, then return the step's total loss,
        workload counters, individual target measurements and batch diagnostics
        as Python values for the outer loop. Errors propagate to the caller.
        """
        started = self._time()
        learning_rates = tuple(float(group["lr"]) for group in self.optimizer.param_groups)
        # Frozen preparation includes token-bounded NVARC and image-bounded VAE calls.
        with training_range("prepare/sources"):
            encoded = encode_training_batch(
                sequences, nvarc_encoder=self.nvarc_encoder, vae_encoder=self.vae_encoder,
                min_context_pairs_before_target=self.min_context_pairs_before_target,
                nvarc_max_tokens=self.nvarc_max_tokens, vae_microbatch_size=self.vae_microbatch_size,
            )
        targets = encoded.inputs
        if not targets:
            raise ValueError("step requires at least one eligible training target")
        prefixes = tuple(target.prefix_end for target in targets)
        # Assign time and noise to each target before grouping targets into microbatches.
        flows = self.flow_sampler(tuple(target.clean_latent for target in targets))
        batches = plan_flux_batches(prefixes, self.packing)
        losses = _StepLosses(len(targets), self.device)

        # Run FLUX microbatches and accumulate gradients for one optimizer update.
        for batch in batches:
            inputs = tuple(targets[index] for index in batch.target_indices)
            flow = select_flow_batch(flows, batch.target_indices)
            result = compute_flux_microbatch_loss(
                inputs, flow, transformer=self.flux_transformer, compute_dtype=self.flux_compute_dtype,
                packing_backend=self.flux_packing_backend, max_joint_tokens=self.packing.max_tokens,
                loss_function=self.loss_function,
            )
            if not bool(torch.isfinite(result.loss.detach()).item()):
                raise FloatingPointError("FLUX microbatch loss is non-finite")
            losses.record(batch.target_indices, result)
            with training_range("train/flux_backward"):
                (result.loss * (len(batch.target_indices) / len(targets))).backward()
            del result, flow, inputs

        # Capture the pre-clip norm and used LR, then advance exactly one update.
        norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, max_norm=math.inf if self.max_grad_norm is None else self.max_grad_norm,
            error_if_nonfinite=True, foreach=False,
        )
        gradient_norm = float(norm)
        with training_range("train/adamw"):
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Release gradient storage before validation, sampling or the next encoding.
        self.optimizer.zero_grad(set_to_none=True)

        target_metrics = losses.target_metrics(targets, flows)
        loss = float(losses.total)
        preparation = encoded.preparation
        target_count = len(targets)
        text_tokens = sum(prefixes)
        image_tokens = target_count * TARGET_LATENT_TOKEN_COUNT
        del encoded, targets, flows, losses, norm
        elapsed = self._time() - started
        return TrainStepResult(
            loss=loss, learning_rates=learning_rates, gradient_norm=gradient_norm,
            gradient_clipped=self.max_grad_norm is not None and gradient_norm > self.max_grad_norm,
            step_seconds=elapsed, source_count=len(preparation.sources), target_count=target_count,
            nvarc_tokens=sum(preparation.source_lengths), nvarc_calls=len(preparation.nvarc_batches),
            vae_calls=preparation.vae_calls, flux_text_tokens=text_tokens, flux_image_tokens=image_tokens,
            flux_joint_tokens=text_tokens + image_tokens, flux_forward_calls=len(batches),
            targets=target_metrics,
            diagnostics=StepDiagnostics(preparation, batches, prefixes),
        )

    def _time(self) -> float:
        """Read the step timer after waiting for any pending CUDA work."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


__all__ = ["Trainer", "TrainStepResult", "TargetMetrics", "StepDiagnostics"]
