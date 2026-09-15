"""Pack one optimizer window under a joint text/image token budget."""

from __future__ import annotations

from dataclasses import dataclass

from diffusion_arc._constants import NVARC_MAX_TOKENS


# The fixed 30 x 30 target canvas contributes one image token per ARC cell.
_FLUX_IMAGE_TOKENS = 900
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class FluxPackingPolicy:
    """Maximum joint text/image tokens per packed call, including full sources."""

    max_tokens: int = 12_288

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        minimum = NVARC_MAX_TOKENS + _FLUX_IMAGE_TOKENS
        if not minimum <= self.max_tokens <= _INT32_MAX:
            raise ValueError(f"max_tokens must lie in [{minimum}, {_INT32_MAX}]")


@dataclass(frozen=True)
class FluxTargetBatch:
    """Physical call with indices into the original FIFO optimizer batch."""

    target_indices: tuple[int, ...]
    joint_tokens: int


def plan_flux_batches(
    prefix_lengths: tuple[int, ...],
    policy: FluxPackingPolicy,
) -> tuple[FluxTargetBatch, ...]:
    """Use best-fit-decreasing to reduce calls within this optimizer window.

    Each target costs its valid text length plus 900 image tokens. Descending
    lengths retain FIFO order on ties; each target fills the tightest available
    pack, preferring the earliest pack on equal remaining space. The heuristic
    preserves every target without padding or splitting, but does not guarantee
    the globally smallest number of calls.
    """

    if not isinstance(policy, FluxPackingPolicy):
        raise ValueError("policy must be a FluxPackingPolicy")
    for length in prefix_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= NVARC_MAX_TOKENS:
            raise ValueError(f"prefix lengths must be integers in [1, {NVARC_MAX_TOKENS}]")

    packs: list[list[int]] = []
    remaining: list[int] = []
    order = sorted(range(len(prefix_lengths)), key=lambda index: -prefix_lengths[index])
    for index in order:
        cost = prefix_lengths[index] + _FLUX_IMAGE_TOKENS
        best = min(
            (pack_index for pack_index, free in enumerate(remaining) if free >= cost),
            key=lambda pack_index: (remaining[pack_index], pack_index),
            default=None,
        )
        if best is None:
            packs.append([index])
            remaining.append(policy.max_tokens - cost)
        else:
            packs[best].append(index)
            remaining[best] -= cost

    return tuple(
        FluxTargetBatch(tuple(indices), policy.max_tokens - free)
        for indices, free in zip(packs, remaining)
    )


__all__ = [
    "FluxPackingPolicy", "FluxTargetBatch", "plan_flux_batches",
]
