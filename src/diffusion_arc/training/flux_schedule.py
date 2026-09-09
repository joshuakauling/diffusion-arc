"""Schedule one optimizer window by measured padded costs or packed token budget."""

from __future__ import annotations

from dataclasses import dataclass
import math

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


@dataclass(frozen=True, order=True)
class FluxBatchShape:
    """One allowed shape and its median forward/backward time per whole call."""

    batch_size: int
    max_tokens: int
    forward_backward_ms: float

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be an integer")
        if self.batch_size not in (1, 2, 4, 8):
            raise ValueError("FLUX batch_size must be 1, 2, 4, or 8")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 0 < self.max_tokens <= NVARC_MAX_TOKENS
        ):
            raise ValueError(f"max_tokens must lie in [1, {NVARC_MAX_TOKENS}]")
        if (
            isinstance(self.forward_backward_ms, bool)
            or not isinstance(self.forward_backward_ms, (int, float))
            or not math.isfinite(self.forward_backward_ms)
            or self.forward_backward_ms <= 0
        ):
            raise ValueError("forward_backward_ms must be finite and positive")


@dataclass(frozen=True)
class FluxSchedulingPolicy:
    """Explicit shape/cost catalog, including singleton coverage for any tail."""

    shapes: tuple[FluxBatchShape, ...]

    def __post_init__(self) -> None:
        shapes = tuple(self.shapes)
        if not shapes or any(not isinstance(shape, FluxBatchShape) for shape in shapes):
            raise ValueError("shapes must contain FluxBatchShape objects")
        keys = {(shape.batch_size, shape.max_tokens) for shape in shapes}
        if len(keys) != len(shapes):
            raise ValueError("shapes must not contain duplicate batch_size/max_tokens pairs")
        if (1, max(shape.max_tokens for shape in shapes)) not in keys:
            raise ValueError("the catalog must include a singleton at its largest token boundary")
        object.__setattr__(self, "shapes", tuple(sorted(shapes)))


@dataclass(frozen=True)
class FluxTargetBatch:
    """Physical call with indices into the original FIFO optimizer batch."""

    target_indices: tuple[int, ...]
    padded_length: int | None
    joint_tokens: int | None = None


def schedule_packed_flux_targets(
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
        FluxTargetBatch(tuple(indices), None, policy.max_tokens - free)
        for indices, free in zip(packs, remaining)
    )


def schedule_flux_targets(
    prefix_lengths: tuple[int, ...],
    policy: FluxSchedulingPolicy,
) -> tuple[FluxTargetBatch, ...]:
    """Minimize summed measured call times for a descending-length partition.

    Sorting is stable. Each group contains real targets only, possibly promoted
    to a larger boundary when that lowers the total cost. Ties prefer fewer
    calls, then canonical traversal by ascending batch size and token boundary.
    No target crosses an optimizer boundary; singleton shapes cover short tails.
    """

    if prefix_lengths and max(prefix_lengths) > max(shape.max_tokens for shape in policy.shapes):
        raise ValueError("prefix length exceeds the FLUX scheduling catalog")
    order = tuple(sorted(range(len(prefix_lengths)), key=lambda index: -prefix_lengths[index]))
    count = len(order)
    best = [None] * (count + 1)
    best[count] = (0.0, 0, ())
    for start in range(count - 1, -1, -1):
        for shape in policy.shapes:
            end = start + shape.batch_size
            if end > count or prefix_lengths[order[start]] > shape.max_tokens:
                continue
            tail_cost, tail_calls, tail_batches = best[end]
            cost = shape.forward_backward_ms + tail_cost
            calls = 1 + tail_calls
            if best[start] is None or (cost, calls) < best[start][:2]:
                batch = FluxTargetBatch(order[start:end], shape.max_tokens)
                best[start] = (cost, calls, (batch, *tail_batches))
    return best[0][2]


__all__ = [
    "FluxBatchShape", "FluxPackingPolicy", "FluxSchedulingPolicy", "FluxTargetBatch",
    "schedule_flux_targets", "schedule_packed_flux_targets",
]
