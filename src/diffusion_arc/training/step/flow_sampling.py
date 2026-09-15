"""Sample straight-path flow states with uniform, stratified, or endpoint-quota times."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import TrainingConfig

import torch
from torch import Tensor

from diffusion_arc.model.contracts import validate_target_latents


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One independently sampled straight-path state per target exposure.

    ``noisy_latents`` contains the interpolated transformer input ``z_t``.
    ``timesteps`` contains one normalized flow time per target exposure.
    ``target_velocity`` is the straight-path supervision ``epsilon - z_0``.
    """

    noisy_latents: Tensor
    timesteps: Tensor
    target_velocity: Tensor



@dataclass(frozen=True)
class EndpointFlowPolicy:
    """Explicit quotas for a complete optimizer batch and its possible epoch tail."""

    full_batch_size: int
    endpoint_targets_per_full_batch: int
    final_batch_size: int
    endpoint_targets_in_final_batch: int

    def __post_init__(self) -> None:
        for name in ("full_batch_size", "final_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.final_batch_size > self.full_batch_size:
            raise ValueError("final_batch_size must not exceed full_batch_size")
        for name, size in (("endpoint_targets_per_full_batch", self.full_batch_size),
                           ("endpoint_targets_in_final_batch", self.final_batch_size)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= size:
                raise ValueError(f"{name} must be an integer in [0, {size}]")
        if (self.final_batch_size == self.full_batch_size
                and self.endpoint_targets_in_final_batch != self.endpoint_targets_per_full_batch):
            raise ValueError("a full final batch must use the full-batch endpoint count")

    def endpoint_count(self, target_count: int) -> int:
        if target_count == self.full_batch_size:
            return self.endpoint_targets_per_full_batch
        if target_count == self.final_batch_size:
            return self.endpoint_targets_in_final_batch
        raise ValueError(f"unexpected optimizer batch size {target_count} for the endpoint policy")


def sample_endpoint_flow_batches(
    clean_latents: Sequence[Tensor],
    *,
    endpoint_count: int,
    generator: torch.Generator | None = None,
) -> tuple[FlowMatchingBatch, ...]:
    """Return one FP32 flow sample per target in the supplied logical order.

    Uniform times use the equally spaced FP32 lattice strictly inside (0, 1).
    Random endpoint positions are sampled without replacement. Per-target noise
    draws are independent of all later changes to pack order and pack capacity.
    """
    clean = tuple(clean_latents)
    if not clean:
        raise ValueError("clean_latents must contain at least one target")
    if type(endpoint_count) is not int or not 0 <= endpoint_count <= len(clean):
        raise ValueError("endpoint_count must be an integer between zero and target count")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    for latent in clean:
        if not isinstance(latent, Tensor) or latent.ndim != 2:
            raise ValueError("each clean latent must have shape [tokens, channels]")
        validate_target_latents(latent.unsqueeze(0), name="clean_latent")
        if latent.dtype != torch.float32 or latent.device != clean[0].device or latent.shape != clean[0].shape:
            raise ValueError("recipe latents must share shape/device and use float32")

    device = clean[0].device
    # Exclude both endpoints exactly, including the rare zero from torch.rand.
    times = torch.randint(1, 2**24, (len(clean),), device=device,
                          generator=generator).to(torch.float32) / 2**24
    endpoint_indices = torch.randperm(len(clean), device=device, generator=generator)[:endpoint_count]
    times[endpoint_indices] = 1.0
    flows = []
    for index, latent in enumerate(clean):
        noise = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=generator)
        time = times[index:index + 1]
        flows.append(FlowMatchingBatch(
            noisy_latents=((1 - time) * latent + time * noise).unsqueeze(0),
            timesteps=time,
            target_velocity=(noise - latent).unsqueeze(0),
        ))
    return tuple(flows)




def sample_flow_matching_batch(
    clean_latents: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    """Sample uniform flow times and Gaussian noise for clean target latents.

    Each target exposure receives an independent scalar ``t`` and an
    independent standard-Gaussian tensor ``epsilon``. The returned state uses
    ``z_t = (1 - t) * z_0 + t * epsilon`` and the supervision target is
    ``epsilon - z_0``.
    """

    validate_target_latents(clean_latents, name="clean_latents")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    target_count = clean_latents.shape[0]
    timesteps = torch.rand(
        (target_count,),
        device=clean_latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    broadcast_timesteps = timesteps.to(dtype=clean_latents.dtype).reshape(
        target_count,
        1,
        1,
    )
    noisy_latents = (
        (1 - broadcast_timesteps) * clean_latents
        + broadcast_timesteps * noise
    )
    target_velocity = noise - clean_latents

    return FlowMatchingBatch(
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        target_velocity=target_velocity,
    )



def sample_stratified_flow_matching_batch(
    clean_latents: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowMatchingBatch:
    """Sample one uniformly jittered flow time from every batch stratum.

    For ``K`` target exposures, the sampler draws one time from each interval
    ``[i / K, (i + 1) / K)`` and randomly assigns those times to the batch
    entries. Gaussian noise remains independent for every target exposure.
    The random assignment prevents a fixed condition slot from being tied to
    one particular flow-time region across optimizer steps.
    """

    validate_target_latents(clean_latents, name="clean_latents")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")

    target_count = clean_latents.shape[0]
    jitter = torch.rand(
        (target_count,),
        device=clean_latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    strata = torch.arange(
        target_count,
        device=clean_latents.device,
        dtype=torch.float32,
    )
    ordered_timesteps = (strata + jitter) / target_count
    assignment = torch.randperm(
        target_count,
        device=clean_latents.device,
        generator=generator,
    )
    timesteps = ordered_timesteps.index_select(0, assignment)
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    broadcast_timesteps = timesteps.to(dtype=clean_latents.dtype).reshape(
        target_count,
        1,
        1,
    )
    noisy_latents = (
        (1 - broadcast_timesteps) * clean_latents
        + broadcast_timesteps * noise
    )
    target_velocity = noise - clean_latents

    return FlowMatchingBatch(
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        target_velocity=target_velocity,
    )



def build_flow_generator(
    config: TrainingConfig,
    *,
    device: torch.device,
) -> torch.Generator:
    """Build a device-local flow generator independent of DataLoader draws."""

    return torch.Generator(device=device).manual_seed(config.seed)



@dataclass(frozen=True)
class UniformFlowSampler:
    """Bind the generator and preserve B3's independent draws in target order."""

    generator: torch.Generator

    def __call__(self, clean_latents: Sequence[Tensor]) -> tuple[FlowMatchingBatch, ...]:
        return tuple(
            sample_flow_matching_batch(latent.unsqueeze(0), generator=self.generator)
            for latent in clean_latents
        )


@dataclass(frozen=True)
class EndpointFlowSampler:
    """Bind endpoint quotas independently of the selected loss or pack layout."""

    generator: torch.Generator
    policy: EndpointFlowPolicy

    def __call__(self, clean_latents: Sequence[Tensor]) -> tuple[FlowMatchingBatch, ...]:
        return sample_endpoint_flow_batches(
            clean_latents, endpoint_count=self.policy.endpoint_count(len(clean_latents)),
            generator=self.generator,
        )


def build_flow_sampler(
    config: TrainingConfig, *, generator: torch.Generator,
) -> UniformFlowSampler | EndpointFlowSampler:
    """Resolve the configured time distribution once, before consuming data."""
    if config.config_type == "baseline":
        return UniformFlowSampler(generator)
    flow = config.recipe["flow"]
    return EndpointFlowSampler(generator, EndpointFlowPolicy(
        full_batch_size=config.loop.optimizer_batch_size,
        endpoint_targets_per_full_batch=flow["endpoint_targets_per_full_batch"],
        final_batch_size=config.final_batch_target_count,
        endpoint_targets_in_final_batch=flow["endpoint_targets_in_final_batch"],
    ))


def select_flow_batch(
    flows: Sequence[FlowMatchingBatch], indices: Sequence[int],
) -> FlowMatchingBatch:
    """Gather fixed logical draws for one physical call without resampling."""
    selected = tuple(flows[index] for index in indices)
    return FlowMatchingBatch(
        noisy_latents=torch.cat(tuple(flow.noisy_latents for flow in selected)),
        timesteps=torch.cat(tuple(flow.timesteps for flow in selected)),
        target_velocity=torch.cat(tuple(flow.target_velocity for flow in selected)),
    )


__all__ = [
    "FlowMatchingBatch", "EndpointFlowPolicy", "sample_endpoint_flow_batches",
    "sample_flow_matching_batch", "sample_stratified_flow_matching_batch", "build_flow_generator",
    "UniformFlowSampler", "EndpointFlowSampler", "build_flow_sampler", "select_flow_batch",
]
