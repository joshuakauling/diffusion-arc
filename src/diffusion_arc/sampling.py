"""Official BFL FLUX.2 schedule and backward Euler integration.

The schedule equations track ``src/flux2/sampling.py`` at BFL commit
``50fe5162777813d869182b139e83b10743caef15``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Return the empirical shift used by BFL's released FLUX.2 sampler."""

    image_seq_len = _positive_integer(image_seq_len, name="image_seq_len")
    num_steps = _positive_integer(num_steps, name="num_steps")
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    mu_200 = a2 * image_seq_len + b2
    mu_10 = a1 * image_seq_len + b1
    slope = (mu_200 - mu_10) / 190.0
    intercept = mu_200 - 200.0 * slope
    return float(slope * num_steps + intercept)


def generalized_time_snr_shift(
    timesteps: Tensor,
    mu: float,
    sigma: float = 1.0,
) -> Tensor:
    """Apply BFL's generalized time/SNR shift elementwise."""

    if not isinstance(timesteps, Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if not torch.is_floating_point(timesteps):
        raise TypeError("timesteps must use a floating-point dtype")
    if not torch.isfinite(timesteps).all():
        raise ValueError("timesteps contain non-finite values")
    if bool(((timesteps < 0) | (timesteps > 1)).any()):
        raise ValueError("timesteps must lie in [0, 1]")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1 / timesteps - 1) ** sigma)


def bfl_time_schedule(
    num_steps: int,
    image_seq_len: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return BFL's shifted ``num_steps + 1`` boundaries from 1 to 0."""

    num_steps = _positive_integer(num_steps, name="num_steps")
    image_seq_len = _positive_integer(image_seq_len, name="image_seq_len")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be floating-point")

    timesteps = torch.linspace(
        1,
        0,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )
    mu = compute_empirical_mu(image_seq_len, num_steps)
    shifted = generalized_time_snr_shift(timesteps, mu, 1.0)
    if shifted.shape != (num_steps + 1,):
        raise AssertionError("BFL schedule has an unexpected shape")
    if not torch.isfinite(shifted).all():
        raise FloatingPointError("BFL schedule contains non-finite values")
    return shifted


def _as_schedule(
    timesteps: Tensor | Sequence[float],
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(timesteps, Tensor):
        schedule = timesteps.to(device=device, dtype=torch.float32)
    else:
        schedule = torch.tensor(timesteps, device=device, dtype=torch.float32)
    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("timesteps must be one-dimensional with at least two values")
    if not torch.isfinite(schedule).all():
        raise ValueError("timesteps contain non-finite values")
    if float(schedule[0].item()) != 1.0:
        raise ValueError("Euler integration must start at t=1")
    if float(schedule[-1].item()) != 0.0:
        raise ValueError("Euler integration must end at t=0")
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("timesteps must be strictly decreasing")
    return schedule


def euler_integrate(
    initial: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    timesteps: Tensor | Sequence[float],
) -> Tensor:
    """Integrate a velocity field backward using BFL's Euler convention.

    ``velocity_fn`` receives the current state and a batch-shaped time tensor
    ``[B]``.  This helper intentionally implements conditioned sampling only;
    classifier-free guidance is outside the initial Diffusion-ARC architecture.
    """

    if not isinstance(initial, Tensor):
        raise TypeError("initial must be a torch.Tensor")
    if initial.ndim != 3 or any(size <= 0 for size in initial.shape):
        raise ValueError("initial must have non-empty shape [B, N, C]")
    if not torch.is_floating_point(initial):
        raise TypeError("initial must use a floating-point dtype")
    if not torch.isfinite(initial).all():
        raise ValueError("initial contains non-finite values")
    if not callable(velocity_fn):
        raise TypeError("velocity_fn must be callable")

    schedule = _as_schedule(timesteps, device=initial.device)
    state = initial
    for current, following in zip(schedule[:-1], schedule[1:]):
        time_batch = torch.ones(
            (state.shape[0],),
            device=state.device,
            dtype=state.dtype,
        ) * current.to(dtype=state.dtype)
        velocity = velocity_fn(state, time_batch)
        if not isinstance(velocity, Tensor):
            raise TypeError("velocity_fn must return a torch.Tensor")
        if velocity.shape != state.shape:
            raise ValueError(
                f"velocity shape {tuple(velocity.shape)} must equal state shape "
                f"{tuple(state.shape)}"
            )
        if velocity.device != state.device or velocity.dtype != state.dtype:
            raise ValueError("velocity must match the state device and dtype")
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("velocity_fn returned non-finite values")
        delta = (following - current).to(device=state.device, dtype=state.dtype)
        state = state + delta * velocity
        if not torch.isfinite(state).all():
            raise FloatingPointError("Euler integration produced non-finite values")
    return state
