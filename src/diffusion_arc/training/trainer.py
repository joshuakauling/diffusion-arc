"""Train FLUX on one optimizer batch, from NVARC/VAE encoding to the update."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from diffusion_arc._constants import DEFAULT_NVARC_MAX_TOKENS
from diffusion_arc.arc_data.image.canvas import CANVAS_SIZE
from diffusion_arc.arc_data.text.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.contracts import TARGET_LATENT_TOKEN_COUNT
from diffusion_arc.model.flux.packing import PackedFluxForward

from .distribution_strategy import build_distribution_strategy
from .step.encoding import FluxInput, PreparationMetrics, encode_training_batch
from .step.flow_sampling import FlowMatchingBatch, select_flow_batch
from .step.flux_microbatch import compute_flux_microbatch_loss, compute_fsdp_dummy_loss
from .step.flux_batching import FluxPackingPolicy, FluxMicrobatchPlan, plan_flux_batches
from .step.loss import LossResult, VelocityLoss
from .profiling import training_range

if TYPE_CHECKING:
    from .distributed_data import RankBatchPlan


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
    ``flux_microbatch_plans`` records which targets shared each FLUX call. Its target
    indices and ``target_prefix_tokens`` refer to positions in
    ``TrainStepResult.target_metrics``. These details contain no model tensors.
    """

    preparation: PreparationMetrics
    flux_microbatch_plans: tuple[FluxMicrobatchPlan, ...]
    target_prefix_tokens: tuple[int, ...]


@dataclass(frozen=True)
class TrainStepResult:
    """Measurements returned after one optimizer update.

    Loss, target measurements, counters and timings describe the local rank's
    work in this step. DDP callers aggregate these in the outer loop; the
    gradient norm describes the synchronized gradients. Learning rates are the
    values used for the update; the gradient norm is measured before clipping.
    ``target_metrics`` contains individual measurements in input sequence order, then
    target order within each sequence. Its tuple position is the target index.

    The caller can group these measurements for logging and track run progress.
    Detailed batch layouts are separate under ``diagnostics``. The result holds
    ordinary Python values, with no tensors or gradient graphs.

    Work counters include FSDP padding calls; ``padding_forward_calls`` separates
    their overhead from real targets, which alone determine the loss and cursor.
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
    target_metrics: tuple[TargetMetrics, ...]
    diagnostics: StepDiagnostics
    padding_forward_calls: int = 0


class _StepLossMetrics:
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
        self, flux_inputs: Sequence[FluxInput], flows: Sequence[FlowMatchingBatch],
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
        for flux_input, row in zip(flux_inputs, rows, strict=True):
            has_boundary = flux_input.grid_shape != (CANVAS_SIZE, CANVAS_SIZE)
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
                loss=row[0], timestep=row[1], context_pair_count=flux_input.context_pair_count,
                has_boundary=has_boundary, **regions,
            ))
        return tuple(metrics)


class Trainer:
    """Run one optimizer update for each batch of sequences passed to ``step``.

    Holds the models, optimizer, scheduler, flow sampler, loss function and
    execution limits. NVARC and VAE remain frozen; only FLUX is trained.
    The outer loop selects data, logs returned measurements and handles
    validation, sampling and checkpoints. ``completed_update`` is true only
    at an initialized or successfully cleared optimizer-update boundary.

    DDP callers pass DDP(PackedFluxForward(...)) as ``flux_transformer`` with
    forward_sync_buffers=False and static_graph=False, using normal gradient
    averaging without custom communication hooks or join(). Every rank calls
    step for the same global update and must supply at least one eligible
    target. Source assignment and globally consistent flow draws/endpoint
    quotas belong to the caller; the sampler receives only local latents.
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
        flux_max_spread_tokens: int = 1_000,
        flux_packing_algorithm: str = "balanced_release_v1",
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
        # Direct single-device callers may supply a raw transformer instead of a prepared model.
        flux_model = flux_transformer
        if not isinstance(flux_model, (PackedFluxForward, DDP, FSDPModule)):
            flux_model = PackedFluxForward(flux_model, compute_dtype=flux_compute_dtype,
                                           backend=flux_packing_backend)
        self.distribution_strategy = build_distribution_strategy(
            flux_model, compute_dtype=flux_compute_dtype, backend=flux_packing_backend,
        )
        self.flux_model = self.distribution_strategy.flux_model
        self.packing = FluxPackingPolicy(flux_max_joint_tokens, flux_max_spread_tokens, flux_packing_algorithm)
        self.nvarc_encoder, self.vae_encoder, self.flux_transformer = nvarc_encoder, vae_encoder, flux_transformer
        self.optimizer, self.scheduler = optimizer, scheduler
        self.flow_sampler, self.loss_function = flow_sampler, loss_function
        self.flux_compute_dtype, self.flux_packing_backend = flux_compute_dtype, flux_packing_backend
        self.nvarc_max_tokens, self.vae_microbatch_size = nvarc_max_tokens, vae_microbatch_size
        self.min_context_pairs_before_target, self.max_grad_norm = min_context_pairs_before_target, max_grad_norm
        self.parameters = self.distribution_strategy.parameters
        self.device = self.distribution_strategy.device
        nvarc_encoder.eval()
        vae_encoder.eval()
        flux_transformer.train()
        self.optimizer.zero_grad(set_to_none=self.distribution_strategy.set_grad_to_none)
        self.completed_update = True

    def step(self, sequences: Sequence[ArcTokenizedTrainingSample], *,
             batch_plan: RankBatchPlan | None = None) -> TrainStepResult:
        """Train on the supplied sequences with one optimizer update.

        Encode sequences with frozen NVARC and target images with frozen VAE.
        Sample flow times and noise for local targets, then pack them into
        FLUX microbatches. Accumulate gradients with equal weight per global target,
        optionally clip them, update parameters once and advance the supplied
        scheduler once.

        DDP synchronizes during the last local backward. FSDP2 synchronizes only
        during the final scheduled backward of each optimizer batch,
        padding shorter rank schedules with zero-weight model calls.
        A precomputed global batch plan supplies target counts and packed calls;
        otherwise counts are reduced before encoding. Loss-finiteness is checked
        collectively afterward in either path.
        Non-finite losses/gradients abort every rank before the update. Other
        failures require the caller's process supervision; this is not a
        distributed recovery loop.

        Clear gradients after the update, then return the local step's loss,
        workload counters, individual target measurements and batch diagnostics
        as Python values for the outer loop. Errors propagate to the caller.
        """
        self.completed_update = False
        started = self._time()
        learning_rates = tuple(float(group["lr"]) for group in self.optimizer.param_groups)
        # Plan from CPU metadata so even an empty rank joins this preflight.
        local_target_count = sum(
            max(0, len(sequence.target_grids) - self.min_context_pairs_before_target + 1)
            for sequence in sequences
        )
        with training_range("train/target_count"):
            if batch_plan is None:
                global_target_count = self.distribution_strategy.all_reduce_target_count(local_target_count)
            else:
                if batch_plan.flux_plan is None:
                    raise ValueError("batch_plan requires a precomputed FLUX plan")
                if local_target_count != len(batch_plan.target_indices) or local_target_count == 0:
                    raise ValueError("local target count differs from the precomputed batch plan")
                global_target_count = batch_plan.global_target_count
        # Frozen preparation includes token-bounded NVARC and image-bounded VAE calls.
        with training_range("prepare/sources"):
            encoded = encode_training_batch(
                sequences, nvarc_encoder=self.nvarc_encoder, vae_encoder=self.vae_encoder,
                min_context_pairs_before_target=self.min_context_pairs_before_target,
                nvarc_max_tokens=self.nvarc_max_tokens, vae_microbatch_size=self.vae_microbatch_size,
            )
        flux_inputs = encoded.inputs
        prefixes = tuple(flux_input.prefix_end for flux_input in flux_inputs)
        # Flow draws retain original target order regardless of the packed call order.
        if batch_plan is None:
            flows = self.flow_sampler(tuple(flux_input.clean_latent for flux_input in flux_inputs))
            nvarc_sequence_states = dict.fromkeys(flux_input.nvarc_sequence_state for flux_input in flux_inputs)
            nvarc_sequence_state_ids = {state: index for index, state in enumerate(nvarc_sequence_states)}
            microbatch_plans = plan_flux_batches(
                prefixes, self.packing,
                source_ids=tuple(nvarc_sequence_state_ids[flux_input.nvarc_sequence_state] for flux_input in flux_inputs),
                source_bytes=tuple(state.allocated_bytes for state in nvarc_sequence_states),
            )
        else:
            if prefixes != batch_plan.flux_plan.target_prefix_tokens:
                raise ValueError("encoded prefix lengths differ from the precomputed batch plan")
            microbatch_plans = batch_plan.flux_plan.microbatch_plans
            flows = self.flow_sampler(tuple(flux_input.clean_latent for flux_input in flux_inputs))
        step_loss_metrics = _StepLossMetrics(len(flux_inputs), self.device)
        # FSDP sorts packs and adds dummy calls; single-device and DDP keep the plans unchanged.
        microbatch_schedule = self.distribution_strategy.plan_microbatch_calls(microbatch_plans,
            **({"collective_call_count": batch_plan.flux_plan.collective_call_count} if batch_plan else {}))

        # Run FLUX microbatches and accumulate gradients for one optimizer update.
        for index, microbatch_plan in enumerate(microbatch_schedule):
            # The strategy selects synchronization, including the final FSDP dummy call.
            with self.distribution_strategy.microbatch_context(is_last=index == len(microbatch_schedule) - 1):
                if microbatch_plan is None:
                    backward_loss = compute_fsdp_dummy_loss(self.flux_model, device=self.device)
                else:
                    inputs = tuple(flux_inputs[i] for i in microbatch_plan.target_indices)
                    flow = select_flow_batch(flows, microbatch_plan.target_indices)
                    result = compute_flux_microbatch_loss(
                        inputs, flow, transformer=self.flux_model,
                        max_joint_tokens=self.packing.max_tokens, loss_function=self.loss_function,
                    )
                    self.distribution_strategy.check_microbatch_loss(result.loss)
                    step_loss_metrics.record(microbatch_plan.target_indices, result)
                    weight = self.distribution_strategy.loss_weight(len(microbatch_plan.target_indices), global_target_count)
                    backward_loss = result.loss * weight
                with training_range("train/flux_backward"):
                    backward_loss.backward()
            del backward_loss
            if microbatch_plan is not None:
                del result, flow, inputs

        # Distributed execution must finish all backwards before rejecting a loss.
        with training_range("train/loss_check"):
            self.distribution_strategy.check_step_loss(step_loss_metrics.total)

        # Capture the pre-clip norm and used LR, then advance exactly one update.
        with training_range("train/clip_grad"):
            norm = self._clip_gradients()
            gradient_norm = float(norm)
        with training_range("train/adamw"):
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Keep DDP bucket views for the next accumulation; release other gradient storage.
        with training_range("train/zero_grad"):
            self.optimizer.zero_grad(set_to_none=self.distribution_strategy.set_grad_to_none)
        self.completed_update = True

        # Collect step metrics and packing diagnostics.
        executed_microbatch_plans = tuple(plan for plan in microbatch_schedule if plan is not None)
        dummy_call_count = len(microbatch_schedule) - len(executed_microbatch_plans)
        target_metrics = step_loss_metrics.target_metrics(flux_inputs, flows)
        loss = float(step_loss_metrics.total)
        preparation = encoded.preparation
        target_count = len(flux_inputs)
        # Work counters include synthetic calls; target/loss metrics never do.
        text_tokens = sum(prefixes) + dummy_call_count
        image_tokens = (target_count + dummy_call_count) * TARGET_LATENT_TOKEN_COUNT
        del encoded, flux_inputs, flows, step_loss_metrics, norm
        elapsed = self._time() - started
        return TrainStepResult(
            loss=loss, learning_rates=learning_rates, gradient_norm=gradient_norm,
            gradient_clipped=self.max_grad_norm is not None and gradient_norm > self.max_grad_norm,
            step_seconds=elapsed, source_count=len(preparation.sources), target_count=target_count,
            nvarc_tokens=sum(preparation.source_lengths), nvarc_calls=len(preparation.nvarc_batches),
            vae_calls=preparation.vae_calls, flux_text_tokens=text_tokens, flux_image_tokens=image_tokens,
            flux_joint_tokens=text_tokens + image_tokens, flux_forward_calls=len(microbatch_schedule),
            target_metrics=target_metrics,
            diagnostics=StepDiagnostics(preparation, executed_microbatch_plans, prefixes),
            padding_forward_calls=dummy_call_count,
        )

    def _clip_gradients(self) -> Tensor:
        """Check the gradient norm and apply the optional clipping limit."""
        return torch.nn.utils.clip_grad_norm_(
            self.parameters, max_norm=math.inf if self.max_grad_norm is None else self.max_grad_norm,
            error_if_nonfinite=True, foreach=False,
        )

    def _time(self) -> float:
        """Read the step timer after waiting for any pending CUDA work."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


__all__ = ["Trainer", "TrainStepResult", "TargetMetrics", "StepDiagnostics"]
