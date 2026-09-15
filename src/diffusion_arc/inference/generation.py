"""Target-free padded and packed Euler generation for Diffusion-ARC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from diffusion_arc.arc_data.canvas import (
    ArcGridPrediction,
    classify_arc_cells,
    decode_arc_grids,
    decode_fixed_arc_grids,
)
from diffusion_arc.model.contracts import (
    DEFAULT_FLUX_SPATIAL_SHAPE,
    TARGET_LATENT_CHANNELS,
    FluxSpatialShape,
    FluxTextInputs,
)
from diffusion_arc.model.flux.packing import predict_packed_flux_velocity
from diffusion_arc.model.flux.velocity import (
    build_flux_image_position_ids,
    flux_transformer_device,
    predict_flux_velocity,
)

from .conditioning import (
    PackedGenerationInput,
    evaluation_mode,
    sample_conditioning_noise,
    stage_packed_conditioning,
    validate_packed_execution,
    validate_sampling_seed,
)


@dataclass(frozen=True)
class ArcGenerationBatch:
    """Generated latents, RGB images, and strict PAD-delimited predictions."""

    timestep_schedule: Tensor
    final_latents: Tensor
    rgb_images: Tensor
    class_canvases: Tensor
    predictions: tuple[ArcGridPrediction, ...]


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
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
    infer_padding_boundary: bool = True,
) -> ArcGenerationBatch:
    """Generate one batch from cached NVARC conditioning and decode it once."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    batch_size = _validate_generation_inputs(
        vae_decoder=vae_decoder,
        text_inputs=text_inputs,
        spatial_shape=spatial_shape,
        infer_padding_boundary=infer_padding_boundary,
    )
    transformer_device = flux_transformer_device(transformer)
    schedule = _prepare_timestep_schedule(
        timestep_schedule,
        device=transformer_device,
        require_unit_start=True,
    )
    initial_latents = torch.randn(
        (
            batch_size,
            spatial_shape.token_count,
            TARGET_LATENT_CHANNELS,
        ),
        device=transformer_device,
        dtype=torch.float32,
        generator=generator,
    )
    return _integrate_and_decode_arc_batch(
        transformer=transformer,
        vae_decoder=vae_decoder,
        text_inputs=text_inputs,
        initial_latents=initial_latents,
        schedule=schedule,
        compute_dtype=compute_dtype,
        spatial_shape=spatial_shape,
        infer_padding_boundary=infer_padding_boundary,
    )


def continue_arc_batch(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    text_inputs: FluxTextInputs,
    *,
    initial_latents: Tensor,
    timestep_schedule: Tensor | Sequence[float],
    compute_dtype: torch.dtype,
    spatial_shape: FluxSpatialShape = DEFAULT_FLUX_SPATIAL_SHAPE,
    infer_padding_boundary: bool = True,
) -> ArcGenerationBatch:
    """Continue an ARC Euler trajectory from explicit latents at ``schedule[0]``.

    This diagnostic entry point accepts schedules that begin anywhere in
    ``(0, 1]`` and end at zero. Normal generation remains constrained to a
    complete one-to-zero trajectory through :func:`generate_arc_batch`.
    """

    batch_size = _validate_generation_inputs(
        vae_decoder=vae_decoder,
        text_inputs=text_inputs,
        spatial_shape=spatial_shape,
        infer_padding_boundary=infer_padding_boundary,
    )
    if not isinstance(initial_latents, Tensor):
        raise TypeError("initial_latents must be a torch.Tensor")
    if not initial_latents.dtype.is_floating_point:
        raise TypeError("initial_latents must use a floating-point dtype")
    if initial_latents.requires_grad:
        raise ValueError("initial_latents must not require gradients")
    expected_shape = (
        batch_size,
        spatial_shape.token_count,
        TARGET_LATENT_CHANNELS,
    )
    if tuple(initial_latents.shape) != expected_shape:
        raise ValueError(
            f"initial_latents must have shape {expected_shape}, "
            f"received {tuple(initial_latents.shape)}"
        )

    transformer_device = flux_transformer_device(transformer)
    schedule = _prepare_timestep_schedule(
        timestep_schedule,
        device=transformer_device,
        require_unit_start=False,
    )
    prepared_latents = initial_latents.detach().to(
        device=transformer_device,
        dtype=torch.float32,
    )
    return _integrate_and_decode_arc_batch(
        transformer=transformer,
        vae_decoder=vae_decoder,
        text_inputs=text_inputs,
        initial_latents=prepared_latents,
        schedule=schedule,
        compute_dtype=compute_dtype,
        spatial_shape=spatial_shape,
        infer_padding_boundary=infer_padding_boundary,
    )


def generate_packed_arc_samples(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    inputs: Sequence[PackedGenerationInput],
    packs: Sequence[Sequence[int]],
    *,
    steps: int,
    seeds: Sequence[int],
    compute_dtype: torch.dtype,
    backend: object,
) -> tuple[tuple[ArcGridPrediction, ...], ...]:
    """Generate full 30×30 canvases, returning predictions by query then seed.

    Every trajectory uses FP32 Euler integration and query-keyed noise. Inputs
    contain no supervision, output dimensions, or target latents.
    """

    values, groups, device = validate_packed_execution(
        transformer, inputs, packs, compute_dtype, backend,
    )
    if not isinstance(vae_decoder, nn.Module):
        raise TypeError("vae_decoder must be a torch.nn.Module")
    sample_seeds = tuple(seeds)
    for seed in sample_seeds:
        validate_sampling_seed(seed)
    if not sample_seeds or len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("seeds must contain distinct integers")
    schedule = linear_timestep_schedule(steps, device=device)
    predictions: dict[tuple[int, int], ArcGridPrediction] = {}
    with evaluation_mode(transformer):
        for indices in groups:
            batch = tuple(values[index] for index in indices)
            text, layout, positions = stage_packed_conditioning(batch, device)
            for sample_index, seed in enumerate(sample_seeds):
                latents = torch.stack(tuple(
                    sample_conditioning_noise(query, seed, ("generation", sample_index), device)
                    for query in batch
                ))
                for timestep, next_timestep in zip(schedule[:-1], schedule[1:], strict=True):
                    velocity = predict_packed_flux_velocity(
                        transformer, latents, text, positions, timestep.expand(len(batch)),
                        layout, compute_dtype=compute_dtype, backend=backend,
                    )
                    latents = latents + (next_timestep - timestep) * velocity.float()
                if not bool(torch.isfinite(latents).all().item()):
                    raise RuntimeError("generated latents are non-finite")
                images = vae_decoder(latents, spatial_shape=DEFAULT_FLUX_SPATIAL_SHAPE)
                if not bool(torch.isfinite(images).all().item()):
                    raise RuntimeError("decoded generation images are non-finite")
                decoded = decode_arc_grids(classify_arc_cells(_normalized_images_to_uint8(images)))
                if len(decoded) != len(indices):
                    raise RuntimeError("decoded prediction count disagrees with the pack")
                for index, prediction in zip(indices, decoded, strict=True):
                    predictions[index, sample_index] = prediction
    return tuple(tuple(predictions[index, sample_index] for sample_index in range(len(sample_seeds)))
                 for index in range(len(values)))


def _prepare_timestep_schedule(
    schedule: Tensor | Sequence[float],
    *,
    device: torch.device,
    require_unit_start: bool,
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
    first = prepared[0].item()
    if require_unit_start and first != 1.0:
        raise ValueError("timestep_schedule must start at 1.0")
    if not require_unit_start and not 0.0 < first <= 1.0:
        raise ValueError("timestep_schedule must start in (0.0, 1.0]")
    if prepared[-1].item() != 0.0:
        raise ValueError("timestep_schedule must end at 0.0")
    if not bool((prepared[:-1] > prepared[1:]).all().item()):
        raise ValueError("timestep_schedule must be strictly decreasing")
    return prepared


def _validate_generation_inputs(
    *,
    vae_decoder: nn.Module,
    text_inputs: FluxTextInputs,
    spatial_shape: FluxSpatialShape,
    infer_padding_boundary: bool,
) -> int:
    if not isinstance(vae_decoder, nn.Module):
        raise TypeError("vae_decoder must be a torch.nn.Module")
    if not isinstance(text_inputs, FluxTextInputs):
        raise TypeError("text_inputs must be a FluxTextInputs instance")
    if text_inputs.encoder_hidden_states.ndim != 3:
        raise ValueError("encoder_hidden_states must have shape [B, L, D]")
    batch_size = text_inputs.encoder_hidden_states.shape[0]
    if batch_size == 0:
        raise ValueError("text_inputs must contain at least one sample")
    if not isinstance(spatial_shape, FluxSpatialShape):
        raise TypeError("spatial_shape must be a FluxSpatialShape")
    if spatial_shape.height != spatial_shape.width:
        raise ValueError("ARC generation requires a square spatial_shape")
    if not isinstance(infer_padding_boundary, bool):
        raise TypeError("infer_padding_boundary must be a boolean")
    return batch_size


def _integrate_and_decode_arc_batch(
    *,
    transformer: nn.Module,
    vae_decoder: nn.Module,
    text_inputs: FluxTextInputs,
    initial_latents: Tensor,
    schedule: Tensor,
    compute_dtype: torch.dtype,
    spatial_shape: FluxSpatialShape,
    infer_padding_boundary: bool,
) -> ArcGenerationBatch:
    batch_size = initial_latents.shape[0]
    transformer_device = flux_transformer_device(transformer)
    image_position_ids = build_flux_image_position_ids(
        batch_size,
        transformer_device,
        spatial_shape=spatial_shape,
    )
    previous_training_mode = transformer.training
    transformer.eval()
    try:
        with torch.inference_mode():
            latents = initial_latents
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

            decoded_images = vae_decoder(
                latents,
                spatial_shape=spatial_shape,
            )
    finally:
        transformer.train(previous_training_mode)

    rgb_images = _normalized_images_to_uint8(decoded_images)
    class_canvases = classify_arc_cells(
        rgb_images,
        canvas_size=spatial_shape.height,
        include_pad=infer_padding_boundary,
    )
    if infer_padding_boundary:
        predictions = decode_arc_grids(
            class_canvases,
            canvas_size=spatial_shape.height,
        )
    else:
        predictions = decode_fixed_arc_grids(
            class_canvases,
            canvas_size=spatial_shape.height,
        )
    return ArcGenerationBatch(
        timestep_schedule=schedule.detach().to(device="cpu"),
        final_latents=(
            latents.detach().to(device="cpu", dtype=torch.float32).clone()
        ),
        rgb_images=rgb_images,
        class_canvases=class_canvases,
        predictions=predictions,
    )


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
    "continue_arc_batch",
    "generate_arc_batch",
    "generate_packed_arc_samples",
    "linear_timestep_schedule",
]
