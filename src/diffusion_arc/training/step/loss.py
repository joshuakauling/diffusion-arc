"""Equal-target flow losses over the full canvas, true grid, and adjacent boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from torch import Tensor
import torch
from torch.nn import functional as F

from diffusion_arc.model.contracts import validate_target_latents

if TYPE_CHECKING:
    from ..config import TrainingConfig


@dataclass(frozen=True)
class RegionLossWeights:
    """Mass assigned to each region mean, independent of the grid dimensions."""

    canvas: float = 0.1
    grid: float = 0.8
    boundary: float = 0.1

    def __post_init__(self) -> None:
        values = (self.canvas, self.grid, self.boundary)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("region weights must be finite, non-negative numbers")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("region weights must sum to one")
        if self.canvas + self.grid <= 0:
            raise ValueError("canvas and grid weights must have positive combined mass")


@dataclass(frozen=True)
class RegionLossResult:
    """FP32 per-target loss, raw region means, and weighted region contributions.

    Region columns are canvas, grid, boundary. An absent boundary contributes
    zero; its weight is redistributed over the other two regions.
    """

    per_target: Tensor
    region_means: Tensor
    contributions: Tensor
    has_boundary: Tensor


@dataclass(frozen=True)
class RegionLossReduction:
    """Prepared masks reusable for repeated losses over the same target shapes."""

    masks: Tensor
    weights: Tensor
    normalizer: Tensor
    has_boundary: Tensor

    def compute(self, predicted_velocity: Tensor, target_velocity: Tensor) -> RegionLossResult:
        """Reduce channels and spatial regions in FP32 without device synchronization."""

        expected_prefix = (self.masks.shape[0], 900)
        if (predicted_velocity.ndim != 3 or predicted_velocity.shape[:2] != expected_prefix
                or predicted_velocity.shape[2] < 1
                or predicted_velocity.shape != target_velocity.shape):
            raise ValueError("velocities must have matching [targets, 900, channels] shapes")
        if (predicted_velocity.device != self.masks.device
                or target_velocity.device != self.masks.device):
            raise ValueError("velocities and prepared region masks must share a device")
        if not predicted_velocity.is_floating_point() or not target_velocity.is_floating_point():
            raise ValueError("velocities must have floating-point dtypes")
        cell_error = (predicted_velocity.float() - target_velocity.float()).square().mean(dim=-1)
        means = (cell_error[:, None, :] * self.masks).sum(dim=-1)
        weighted = means * self.weights
        return RegionLossResult(
            per_target=weighted.sum(dim=-1) / self.normalizer,
            region_means=means,
            contributions=weighted / self.normalizer[:, None],
            has_boundary=self.has_boundary,
        )


def build_region_loss_reduction(
    grid_shapes: Sequence[tuple[int, int]],
    *,
    device: torch.device | str,
    weights: RegionLossWeights = RegionLossWeights(),
) -> RegionLossReduction:
    """Prepare row-major 30×30 masks using CPU shape metadata only.

    The boundary is the clipped one-cell bottom/right L, including its corner.
    Only a 30×30 grid has no boundary. The canvas always includes all 900 cells,
    so its contribution overlaps both other regions by design.
    """

    if not isinstance(weights, RegionLossWeights):
        raise TypeError("weights must be RegionLossWeights")
    shapes = tuple(grid_shapes)
    if not shapes:
        raise ValueError("grid_shapes must contain at least one target")
    for shape in shapes:
        if (not isinstance(shape, (tuple, list)) or len(shape) != 2
                or any(type(side) is not int or not 1 <= side <= 30 for side in shape)):
            raise ValueError("grid shapes must contain integer heights and widths in [1, 30]")
    masks = torch.zeros((len(shapes), 3, 30, 30), dtype=torch.float32)
    normalizer = torch.ones(len(shapes), dtype=torch.float32)
    has_boundary = torch.ones(len(shapes), dtype=torch.bool)
    masks[:, 0] = 1.0 / 900.0
    for index, (height, width) in enumerate(shapes):
        masks[index, 1, :height, :width] = 1.0 / (height * width)
        extent_h, extent_w = min(height + 1, 30), min(width + 1, 30)
        boundary_count = extent_h * extent_w - height * width
        if boundary_count:
            masks[index, 2, :extent_h, :extent_w] = 1.0 / boundary_count
            masks[index, 2, :height, :width] = 0.0
        else:
            normalizer[index] = weights.canvas + weights.grid
            has_boundary[index] = False
    return RegionLossReduction(
        masks=masks.flatten(2).to(device),
        weights=torch.tensor((weights.canvas, weights.grid, weights.boundary),
                             dtype=torch.float32, device=device),
        normalizer=normalizer.to(device),
        has_boundary=has_boundary.to(device),
    )


def compute_region_velocity_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    grid_shapes: Sequence[tuple[int, int]],
    *,
    weights: RegionLossWeights = RegionLossWeights(),
) -> RegionLossResult:
    """Compute the shape-normalized objective before equal-target batch reduction."""

    reduction = build_region_loss_reduction(grid_shapes, device=predicted_velocity.device, weights=weights)
    return reduction.compute(predicted_velocity, target_velocity)


def flow_matching_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
) -> Tensor:
    """Return the unweighted mean squared velocity error in float32."""

    validate_target_latents(predicted_velocity, name="predicted_velocity")
    validate_target_latents(target_velocity, name="target_velocity")
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "predicted_velocity and target_velocity must have identical shapes"
        )
    if predicted_velocity.device != target_velocity.device:
        raise ValueError(
            "predicted_velocity and target_velocity must use the same device"
        )

    return F.mse_loss(
        predicted_velocity.float(),
        target_velocity.float(),
        reduction="mean",
    )


@dataclass(frozen=True)
class LossResult:
    """Differentiable microbatch mean and per-target values for caller-side reporting."""

    loss: Tensor
    per_target: Tensor
    regions: RegionLossResult | None = None


def velocity_mse(
    prediction: Tensor, target: Tensor, grid_shapes: Sequence[tuple[int, int]],
) -> LossResult:
    """Compute plain velocity MSE; grid shapes are unused by this objective."""
    loss = flow_matching_loss(prediction, target)
    # Keep the established scalar reduction for backward; diagnostics need no graph.
    with torch.no_grad():
        per_target = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return LossResult(loss, per_target)


@dataclass(frozen=True)
class RegionVelocityLoss:
    """Apply fixed region weights using the supplied output-grid dimensions."""

    weights: RegionLossWeights

    def __call__(
        self, prediction: Tensor, target: Tensor, grid_shapes: Sequence[tuple[int, int]],
    ) -> LossResult:
        result = compute_region_velocity_loss(
            prediction, target, grid_shapes, weights=self.weights,
        )
        return LossResult(result.per_target.mean(), result.per_target, result)


VelocityLoss = Callable[[Tensor, Tensor, Sequence[tuple[int, int]]], LossResult]


def build_velocity_loss(config: TrainingConfig) -> VelocityLoss:
    """Resolve the objective before training; sampling is configured separately."""
    if config.config_type == "baseline":
        return velocity_mse
    return RegionVelocityLoss(RegionLossWeights(**config.recipe["loss"]["region_weights"]))


__all__ = [
    "flow_matching_loss", "RegionLossWeights", "RegionLossResult", "RegionLossReduction",
    "build_region_loss_reduction", "compute_region_velocity_loss", "LossResult", "velocity_mse",
    "RegionVelocityLoss", "VelocityLoss", "build_velocity_loss",
]
