"""Assign flow randomness by global step and target, independently of rank layout."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib

import torch
from torch import Tensor

from diffusion_arc.model.contracts import validate_target_latents
from .flow_sampling import FlowMatchingBatch


FLOW_SEED_SCHEME = "global_step_target_v1"


def _seed(seed: int, step: int, stream: str, target: int = 0) -> int:
    identity = f"{FLOW_SEED_SCHEME}:{seed}:{step}:{stream}:{target}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class DistributedFlowSampler:
    """Bind one global batch's endpoint quota and local target positions.

    Small time/endpoint draws use a shared CPU seed. Only local noise tensors
    are generated, on their latent device, with a seed per global target.
    One and many ranks agree on the same device type and package versions.
    This deliberately differs from the single-GPU runner's stateful RNG stream.
    """

    seed: int
    step: int
    global_target_count: int
    target_indices: tuple[int, ...]
    endpoint_count: int
    observe_draw: Callable[[int, Tensor, Tensor], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(self.global_target_count) is not int or self.global_target_count <= 0:
            raise ValueError("global_target_count must be a positive integer")
        if type(self.endpoint_count) is not int or not 0 <= self.endpoint_count <= self.global_target_count:
            raise ValueError("endpoint_count must fit the global batch")
        if (not self.target_indices or any(type(i) is not int or not 0 <= i < self.global_target_count
                                          for i in self.target_indices)
                or len(set(self.target_indices)) != len(self.target_indices)):
            raise ValueError("target_indices must be distinct positions in the global batch")

    def __call__(self, clean_latents: Sequence[Tensor]) -> tuple[FlowMatchingBatch, ...]:
        if len(clean_latents) != len(self.target_indices):
            raise ValueError("encoded local target count differs from the global batch plan")
        generator = torch.Generator(device="cpu").manual_seed(_seed(self.seed, self.step, "time"))
        times = torch.randint(1, 2**24, (self.global_target_count,), generator=generator).float() / 2**24
        endpoints = torch.randperm(self.global_target_count, generator=generator)[:self.endpoint_count]
        times[endpoints] = 1.0
        device = clean_latents[0].device
        local_times = times[list(self.target_indices)].to(device)
        noise_generator = torch.Generator(device=device)
        flows = []
        for position, (index, latent) in enumerate(zip(self.target_indices, clean_latents, strict=True)):
            validate_target_latents(latent.unsqueeze(0), name="clean_latent")
            if latent.dtype != torch.float32 or latent.device != device:
                raise ValueError("recipe latents must use float32 on the local device")
            noise_generator.manual_seed(_seed(self.seed, self.step, "noise", index))
            noise = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=noise_generator)
            time = local_times[position:position + 1]
            if self.observe_draw is not None:
                self.observe_draw(index, time, noise)
            flows.append(FlowMatchingBatch(
                noisy_latents=((1 - time) * latent + time * noise).unsqueeze(0),
                timesteps=time,
                target_velocity=(noise - latent).unsqueeze(0),
            ))
        return tuple(flows)
