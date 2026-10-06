"""Measure recovery from oracle noised targets with the production packed flow.

Clean latents construct diagnostic starting states and score outputs; the model
receives only those states, time, and target-free cached NVARC conditioning.
These oracle trajectories are distinct from the direct-sampling benchmark.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict
import math
from numbers import Real

import torch
from torch import Tensor, nn

from diffusion_arc.arc_data.image.canvas import classify_arc_cells, decode_arc_grids
from diffusion_arc.inference.conditioning import (
    evaluation_mode,
    sample_query_latent_noise,
    stage_packed_conditioning,
    validate_packed_execution,
    validate_sampling_seed,
)
from diffusion_arc.inference.generation import (
    _normalized_images_to_uint8,
    linear_timestep_schedule,
)
from diffusion_arc.model.contracts import DEFAULT_FLUX_SPATIAL_SHAPE
from diffusion_arc.model.flux.packing import predict_packed_flux_velocity
from diffusion_arc.training.run.validation import ValidationTarget, diagnose_exact_grids


def diagnostic_schedules(
    start_times: Sequence[float] = (1.0,), steps_per_unit: int | None = None,
    generation_schedules: Sequence[str] = ("linear",), *,
    generation_steps: Sequence[int] | None = None, device=None,
) -> tuple[tuple[str, float, Tensor], ...]:
    """Build recovery probes or a paired comparison of full-noise step counts.

    generation_steps selects independent linear trajectories from one to zero.
    The half-cosine t_i = cos(pi*i/(2*N)) spends more evaluations near t=1.
    The cosine comparison changes spacing while retaining the linear forward budget.
    """
    if generation_steps is not None:
        counts = tuple(generation_steps)
        if (not 1 <= len(counts) <= 8 or any(type(n) is not int or not 1 <= n <= 1000 for n in counts)
                or len(set(counts)) != len(counts)):
            raise ValueError("generation_steps must contain one to eight distinct step counts in [1, 1000]")
        if steps_per_unit is not None or tuple(start_times) != (1.0,) or tuple(generation_schedules) != ("linear",):
            raise ValueError("generation_steps only supports linear t=1 runs without steps_per_unit")
        return tuple(("linear", 1.0, linear_timestep_schedule(n, device=device)) for n in counts)
    linear = linear_timestep_schedule(steps_per_unit, device=device)
    starts = _validate_start_times(start_times, steps_per_unit)
    names = tuple(generation_schedules)
    if names not in (("linear",), ("linear", "cosine_high_noise")):
        raise ValueError("generation_schedules must be linear, optionally followed by cosine_high_noise")
    plans = [("linear", start_t, linear[steps_per_unit - steps:]) for start_t, steps in starts]
    if "cosine_high_noise" in names:
        if 1.0 not in start_times:
            raise ValueError("schedule comparisons require the t=1 linear baseline")
        fractions = torch.linspace(0, 1, steps_per_unit + 1, device=device, dtype=torch.float64)
        cosine = torch.cos(fractions * (math.pi / 2)).float()
        cosine[0], cosine[-1] = 1.0, 0.0
        plans.append(("cosine_high_noise", 1.0, cosine))
    return tuple(plans)


def iter_partial_denoising(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    targets: Sequence[ValidationTarget],
    packs: Sequence[Sequence[int]],
    *,
    start_times: Sequence[float] = (1.0,),
    steps_per_unit: int | None = None,
    seeds: Sequence[int],
    compute_dtype: torch.dtype,
    backend: object,
    generation_schedules: Sequence[str] = ("linear",),
    generation_steps: Sequence[int] | None = None,
) -> Iterator[dict]:
    """Yield scored controls and Euler continuations, using uniform latent MSE.

    Recovery start times must lie on the full linear integration grid. Step
    sweeps always start from pure noise. Each query and seed shares its noise
    across every trajectory. Records include sampling_steps to identify the
    trajectory, while steps counts executed updates (zero for control records).
    Model modes and inference context are restored before every record is
    yielded, so the caller can persist partial results or stop iteration safely.
    """

    values = tuple(targets)
    if not values or any(not isinstance(target, ValidationTarget) for target in values):
        raise ValueError("targets must contain at least one ValidationTarget")
    conditioning, groups, device = validate_packed_execution(
        transformer, tuple(target.conditioning for target in values), packs,
        compute_dtype, backend,
    )
    if not isinstance(vae_decoder, nn.Module):
        raise TypeError("vae_decoder must be a torch.nn.Module")
    plans = diagnostic_schedules(start_times, steps_per_unit, generation_schedules,
                                 generation_steps=generation_steps, device=device)
    sample_seeds = tuple(seeds)
    for seed in sample_seeds:
        validate_sampling_seed(seed)
    if not sample_seeds or len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("seeds must contain distinct integers")

    for indices in groups:
        batch = tuple(values[index] for index in indices)
        queries = tuple(conditioning[index] for index in indices)
        text, layout, positions = stage_packed_conditioning(queries, device)
        clean = torch.stack(tuple(target.clean_latent for target in batch)).to(device)
        with evaluation_mode(vae_decoder):
            records = _decoded_records(
                vae_decoder, clean, clean, batch, kind="clean_roundtrip",
                start_t=None, seed=None, steps=0, schedule=None, sampling_steps=None,
            )
        yield from records

        for sample_index, seed in enumerate(sample_seeds):
            noise = torch.stack(tuple(
                sample_query_latent_noise(query, seed, ("generation", sample_index), device)
                for query in queries
            ))
            target_velocity = noise - clean
            for schedule_name, start_t, trajectory in plans:
                steps = len(trajectory) - 1
                # Full-noise generation never uses the target to construct its state.
                initial = noise.clone() if start_t == 1.0 else (1.0 - trajectory[0]) * clean + trajectory[0] * noise
                with evaluation_mode(vae_decoder):
                    records = _decoded_records(
                        vae_decoder, initial, clean, batch, kind="noisy_start",
                        start_t=start_t, seed=seed, steps=0, schedule=schedule_name, sampling_steps=steps,
                    )
                yield from records

                with evaluation_mode(transformer), evaluation_mode(vae_decoder):
                    latents = initial
                    initial_v_mse = one_step_z0_mse = None
                    for step_index, (timestep, next_timestep) in enumerate(zip(
                        trajectory[:-1], trajectory[1:], strict=True,
                    )):
                        velocity = predict_packed_flux_velocity(
                            transformer, latents, text, positions, timestep.expand(len(batch)),
                            layout, compute_dtype=compute_dtype, backend=backend,
                        ).float()
                        if step_index == 0:
                            initial_v_mse = _latent_mse(velocity, target_velocity)
                            one_step_z0_mse = _latent_mse(initial - timestep * velocity, clean)
                        latents = latents + (next_timestep - timestep) * velocity
                    records = _decoded_records(
                        vae_decoder, latents, clean, batch, kind="denoised",
                        start_t=start_t, seed=seed, steps=steps, schedule=schedule_name, sampling_steps=steps,
                    )
                    for index, record in enumerate(records):
                        record["initial_v_mse"] = initial_v_mse[index]
                        record["one_step_z0_mse"] = one_step_z0_mse[index]
                yield from records


def _validate_start_times(start_times: Sequence[float], steps_per_unit: int) -> tuple:
    starts = tuple(start_times)
    if not starts or any(
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in starts
    ):
        raise ValueError("start_times must be non-empty and contain finite numbers in (0, 1]")
    aligned = tuple((float(value), round(value * steps_per_unit)) for value in starts)
    if any(steps <= 0 or not math.isclose(value * steps_per_unit, steps, abs_tol=1e-6)
           for value, steps in aligned):
        raise ValueError("start_times must align with the steps_per_unit integration grid")
    if len({steps for _, steps in aligned}) != len(aligned):
        raise ValueError("start_times must contain distinct integration grid points")
    return aligned


def _latent_mse(actual: Tensor, expected: Tensor) -> list[float]:
    losses = (actual - expected).square().flatten(1).mean(1).tolist()
    if not all(math.isfinite(value) for value in losses):
        raise RuntimeError("partial-denoising latent errors are non-finite")
    return losses


def _decoded_records(
    vae_decoder: nn.Module,
    latents: Tensor,
    clean: Tensor,
    targets: Sequence[ValidationTarget],
    *,
    kind: str,
    start_t: float | None,
    seed: int | None,
    steps: int,
    schedule: str | None,
    sampling_steps: int | None,
) -> list[dict]:
    if not bool(torch.isfinite(latents).all().item()):
        raise RuntimeError("partial-denoising latents are non-finite")
    images = vae_decoder(latents, spatial_shape=DEFAULT_FLUX_SPATIAL_SHAPE)
    if not bool(torch.isfinite(images).all().item()):
        raise RuntimeError("partial-denoising decoded images are non-finite")
    predictions = decode_arc_grids(classify_arc_cells(_normalized_images_to_uint8(images)))
    diagnostics = diagnose_exact_grids(predictions, tuple(target.expected_grid for target in targets))
    losses = _latent_mse(latents, clean)
    return [
        {
            "task_id": target.task_id, "query_index": target.query_index,
            "kind": kind, "start_t": start_t, "seed": seed, "steps": steps, "schedule": schedule,
            "sampling_steps": sampling_steps,
            "latent_mse": loss, **asdict(diagnostic),
            "class_canvas": prediction.class_canvas.tolist(),
            "predicted_grid": None if prediction.grid is None else [list(row) for row in prediction.grid],
        }
        for target, prediction, diagnostic, loss in zip(
            targets, predictions, diagnostics, losses, strict=True,
        )
    ]


__all__ = ["diagnostic_schedules", "iter_partial_denoising"]
