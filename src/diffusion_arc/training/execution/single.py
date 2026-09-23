"""Packed FLUX execution without a process group."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.model.flux.velocity import flux_transformer_device


class SingleExecution:
    """Own model execution and update mechanics shared by the training loop."""

    world_size = 1
    process_group = None
    set_grad_to_none = True

    def __init__(self, transformer: nn.Module, *, compute_dtype: torch.dtype, backend: object) -> None:
        if isinstance(transformer, PackedFluxForward):
            if transformer.compute_dtype != compute_dtype or transformer.backend is not backend:
                raise ValueError("packed forward precision/backend must match the trainer")
            self.forward = transformer
        else:
            self.forward = PackedFluxForward(transformer, compute_dtype=compute_dtype, backend=backend)
        self.parameters = tuple(p for p in transformer.parameters() if p.requires_grad)
        self.device = flux_transformer_device(transformer)

    def microbatch_context(self, *, is_last: bool):
        """Enclose the forward and backward for one accumulation microbatch."""
        return nullcontext()

    def global_target_count(self, local_count: int) -> int:
        if not local_count:
            raise ValueError("step requires at least one eligible training target")
        return local_count

    def loss_weight(self, target_count: int, global_target_count: int) -> float:
        """Give each global target equal weight in the optimizer update."""
        return self.world_size * target_count / global_target_count

    def check_microbatch_loss(self, loss: Tensor) -> None:
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError("FLUX microbatch loss is non-finite")

    def check_step_loss(self, loss: Tensor) -> None:
        """Single-process losses have already been checked before backward."""

    def clip_gradients(self, max_grad_norm: float | None) -> Tensor:
        return torch.nn.utils.clip_grad_norm_(
            self.parameters, max_norm=math.inf if max_grad_norm is None else max_grad_norm,
            error_if_nonfinite=True, foreach=False,
        )

    def zero_grad(self, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=self.set_grad_to_none)
