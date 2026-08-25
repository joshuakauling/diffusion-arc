"""Fixed-noise flow probes for comparable overfitting diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from diffusion_arc.arc_data import Grid, render_grid_canvas
from diffusion_arc.model import FluxTextInputs
from diffusion_arc.model.flux_execution import build_flux_image_position_ids
from diffusion_arc.model.flux_image import FluxVaeEncoder

from .flux_forward import compute_flux_microbatch_loss
from .flux_microbatch import FluxMicrobatch


@dataclass(frozen=True)
class FixedFlowProbeResult:
    """One deterministic mean flow loss over fixed target microbatches."""

    loss: float
    target_count: int
    microbatch_count: int


def prepare_flow_probe_microbatch(
    text_inputs: FluxTextInputs,
    expected_grids: Sequence[Grid],
    *,
    vae_encoder: FluxVaeEncoder,
) -> FluxMicrobatch:
    """Encode scoring-only targets beside already target-free text inputs."""

    grids = tuple(expected_grids)
    if not grids:
        raise ValueError("expected_grids must not be empty")
    if text_inputs.encoder_hidden_states.shape[0] != len(grids):
        raise ValueError("text input and expected grid batch sizes must match")

    target_images = torch.stack(tuple(render_grid_canvas(grid) for grid in grids))
    clean_latents = vae_encoder(target_images)
    return FluxMicrobatch(
        text_inputs=text_inputs,
        clean_latents=clean_latents,
        image_position_ids=build_flux_image_position_ids(
            len(grids),
            clean_latents.device,
        ),
    )


def evaluate_fixed_flow_probe(
    transformer: nn.Module,
    microbatches: Sequence[FluxMicrobatch],
    *,
    compute_dtype: torch.dtype,
    seed: int,
) -> FixedFlowProbeResult:
    """Evaluate a repeatable flow loss with identical noise and timesteps."""

    batches = tuple(microbatches)
    if not batches:
        raise ValueError("microbatches must not be empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must lie in [0, 2**64)")

    target_count = sum(batch.target_count for batch in batches)
    generator = torch.Generator(device=batches[0].clean_latents.device).manual_seed(seed)
    previous_training_mode = transformer.training
    transformer.eval()
    weighted_loss = 0.0
    try:
        with torch.no_grad():
            for batch in batches:
                if batch.clean_latents.device != batches[0].clean_latents.device:
                    raise ValueError("fixed flow probe microbatches must share one device")
                loss = compute_flux_microbatch_loss(
                    transformer,
                    batch,
                    compute_dtype=compute_dtype,
                    generator=generator,
                )
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise FloatingPointError("fixed flow probe loss is non-finite")
                weighted_loss += float(loss.detach().item()) * (
                    batch.target_count / target_count
                )
    finally:
        transformer.train(previous_training_mode)

    return FixedFlowProbeResult(
        loss=weighted_loss,
        target_count=target_count,
        microbatch_count=len(batches),
    )


__all__ = [
    "FixedFlowProbeResult",
    "evaluate_fixed_flow_probe",
    "prepare_flow_probe_microbatch",
]
