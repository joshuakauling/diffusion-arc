"""Reusable Euler generation and exact-grid diagnostics for Diffusion-ARC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diffusion_arc.arc_data._grid import validate_arc_grid
from diffusion_arc.arc_data.decoding import (
    ArcGridPrediction,
    classify_arc_cells,
    decode_arc_grids,
)
from diffusion_arc.arc_data.types import Grid
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE, FluxTextInputs
from diffusion_arc.model.flux_execution import (
    build_flux_image_position_ids,
    flux_transformer_device,
    predict_flux_velocity,
)


@dataclass(frozen=True)
class ArcGenerationBatch:
    """Generated latents, RGB images, and strict PAD-delimited predictions."""

    timestep_schedule: Tensor
    final_latents: Tensor
    rgb_images: Tensor
    class_canvases: Tensor
    predictions: tuple[ArcGridPrediction, ...]


@dataclass(frozen=True)
class ExactGridDiagnostic:
    """Exact-grid diagnostic computed after generation without fallback behavior."""

    prediction_valid: bool
    invalid_reason: str | None
    predicted_shape: tuple[int, int] | None
    expected_shape: tuple[int, int]
    shape_match: bool
    cell_accuracy: float | None
    exact_grid: bool


def linear_timestep_schedule(
    num_steps: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return ``num_steps`` uniform Euler intervals from one to zero."""

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if device is not None and not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device or None")
    return torch.linspace(
        1.0,
        0.0,
        steps=num_steps + 1,
        device=device,
        dtype=torch.float32,
    )


def generate_arc_batch(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    text_inputs: FluxTextInputs,
    *,
    timestep_schedule: Tensor | Sequence[float],
    compute_dtype: torch.dtype,
    generator: torch.Generator,
) -> ArcGenerationBatch:
    """Generate one batch from cached NVARC conditioning and decode it once."""

    if not isinstance(vae_decoder, nn.Module):
        raise TypeError("vae_decoder must be a torch.nn.Module")
    if not isinstance(text_inputs, FluxTextInputs):
        raise TypeError("text_inputs must be a FluxTextInputs instance")
    if text_inputs.encoder_hidden_states.ndim != 3:
        raise ValueError("encoder_hidden_states must have shape [B, L, D]")
    batch_size = text_inputs.encoder_hidden_states.shape[0]
    if batch_size == 0:
        raise ValueError("text_inputs must contain at least one sample")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    transformer_device = flux_transformer_device(transformer)
    schedule = _prepare_timestep_schedule(
        timestep_schedule,
        device=transformer_device,
    )
    image_position_ids = build_flux_image_position_ids(
        batch_size,
        transformer_device,
    )

    previous_training_mode = transformer.training
    transformer.eval()
    try:
        with torch.inference_mode():
            latents = torch.randn(
                (batch_size, *TARGET_LATENT_SHAPE),
                device=transformer_device,
                dtype=torch.float32,
                generator=generator,
            )
            for timestep, next_timestep in zip(
                schedule[:-1],
                schedule[1:],
                strict=True,
            ):
                timestep_batch = timestep.expand(batch_size)
                velocity = predict_flux_velocity(
                    transformer,
                    latents,
                    text_inputs,
                    image_position_ids,
                    timestep_batch,
                    compute_dtype=compute_dtype,
                )
                delta = (next_timestep - timestep).to(dtype=latents.dtype)
                latents = latents + delta * velocity.to(dtype=latents.dtype)

            decoded_images = vae_decoder(latents)
    finally:
        transformer.train(previous_training_mode)

    rgb_images = _normalized_images_to_uint8(decoded_images)
    class_canvases = classify_arc_cells(rgb_images)
    predictions = decode_arc_grids(class_canvases)
    return ArcGenerationBatch(
        timestep_schedule=schedule.detach().to(device="cpu"),
        final_latents=(
            latents.detach().to(device="cpu", dtype=torch.float32).clone()
        ),
        rgb_images=rgb_images,
        class_canvases=class_canvases,
        predictions=predictions,
    )


def diagnose_exact_grids(
    predictions: Sequence[ArcGridPrediction],
    expected_grids: Sequence[Grid],
) -> tuple[ExactGridDiagnostic, ...]:
    """Compare completed predictions with targets without affecting generation."""

    prediction_batch = tuple(predictions)
    expected_batch = tuple(expected_grids)
    if len(prediction_batch) != len(expected_batch):
        raise ValueError("predictions and expected_grids must have equal lengths")

    diagnostics: list[ExactGridDiagnostic] = []
    for prediction, expected_grid in zip(
        prediction_batch,
        expected_batch,
        strict=True,
    ):
        if not isinstance(prediction, ArcGridPrediction):
            raise TypeError("predictions must contain ArcGridPrediction values")
        validate_arc_grid(expected_grid)
        expected_shape = (len(expected_grid), len(expected_grid[0]))
        predicted_shape = (
            None
            if prediction.height is None or prediction.width is None
            else (prediction.height, prediction.width)
        )
        shape_match = bool(
            prediction.valid and predicted_shape == expected_shape
        )
        cell_accuracy: float | None = None
        exact_grid = False
        if shape_match:
            if prediction.grid is None:
                raise AssertionError("a valid shape-matched prediction must expose a grid")
            predicted_tensor = torch.tensor(prediction.grid, dtype=torch.int64)
            expected_tensor = torch.tensor(expected_grid, dtype=torch.int64)
            matches = predicted_tensor == expected_tensor
            cell_accuracy = float(matches.to(dtype=torch.float32).mean().item())
            exact_grid = bool(matches.all().item())

        diagnostics.append(
            ExactGridDiagnostic(
                prediction_valid=prediction.valid,
                invalid_reason=prediction.invalid_reason,
                predicted_shape=predicted_shape,
                expected_shape=expected_shape,
                shape_match=shape_match,
                cell_accuracy=cell_accuracy,
                exact_grid=exact_grid,
            )
        )
    return tuple(diagnostics)


def _prepare_timestep_schedule(
    schedule: Tensor | Sequence[float],
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(schedule, Tensor):
        if schedule.requires_grad:
            raise ValueError("timestep_schedule must not require gradients")
        prepared = schedule.detach().to(device=device, dtype=torch.float32)
    else:
        if isinstance(schedule, (str, bytes)):
            raise TypeError("timestep_schedule must be a tensor or numeric sequence")
        try:
            values = tuple(schedule)
        except TypeError as error:
            raise TypeError(
                "timestep_schedule must be a tensor or numeric sequence"
            ) from error
        prepared = torch.tensor(values, device=device, dtype=torch.float32)

    if prepared.ndim != 1 or prepared.numel() < 2:
        raise ValueError("timestep_schedule must be one-dimensional with two or more values")
    if not bool(torch.isfinite(prepared).all().item()):
        raise ValueError("timestep_schedule must contain only finite values")
    if prepared[0].item() != 1.0 or prepared[-1].item() != 0.0:
        raise ValueError("timestep_schedule must start at 1.0 and end at 0.0")
    if not bool((prepared[:-1] > prepared[1:]).all().item()):
        raise ValueError("timestep_schedule must be strictly decreasing")
    return prepared


def _normalized_images_to_uint8(images: Tensor) -> Tensor:
    if not isinstance(images, Tensor):
        raise TypeError("VAE decoder output must be a torch.Tensor")
    if not images.dtype.is_floating_point:
        raise TypeError("VAE decoder output must use a floating-point dtype")
    return (
        images.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(dtype=torch.uint8)
    )


__all__ = [
    "ArcGenerationBatch",
    "ExactGridDiagnostic",
    "diagnose_exact_grids",
    "generate_arc_batch",
    "linear_timestep_schedule",
]
