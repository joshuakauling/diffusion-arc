"""DDP execution with one synchronization after local packed accumulation."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward

from .single import SingleExecution


class DdpExecution(SingleExecution):
    """Adapt normal DDP gradient averaging to unequal local target counts.

    DDP must wrap the packed forward with buffer synchronization and static
    graphs disabled. Custom communication hooks and join() are unsupported.
    Ranks may perform different numbers of packs, but each needs a target.
    """

    def __init__(self, transformer: DDP, *, compute_dtype: torch.dtype, backend: object) -> None:
        if not isinstance(transformer.module, PackedFluxForward):
            raise TypeError("DDP must wrap PackedFluxForward for packed training")
        if transformer.forward_sync_buffers or transformer.static_graph:
            raise ValueError("packed DDP requires forward_sync_buffers=False and static_graph=False")
        super().__init__(transformer.module, compute_dtype=compute_dtype, backend=backend)
        self.forward = transformer
        self.process_group = transformer.process_group
        self.world_size = dist.get_world_size(self.process_group)
        # Retaining bucket views avoids reallocating full-model gradient buffers.
        self.set_grad_to_none = not transformer.gradient_as_bucket_view

    def microbatch_context(self, *, is_last: bool):
        return nullcontext() if is_last else self.forward.no_sync()

    def global_target_count(self, local_count: int) -> int:
        """Reject empty ranks collectively before any rank enters FLUX."""
        counts = torch.tensor((local_count, int(local_count == 0)), device=self.device)
        dist.all_reduce(counts, group=self.process_group)
        total, empty_ranks = counts.tolist()
        if empty_ranks:
            raise ValueError("each DDP rank requires at least one eligible training target")
        return total

    def check_microbatch_loss(self, loss: Tensor) -> None:
        """Defer rejection until every rank finishes its local backwards."""

    def check_step_loss(self, loss: Tensor) -> None:
        finite = torch.isfinite(loss).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(finite.item()):
            raise FloatingPointError("FLUX loss is non-finite on a DDP rank")
