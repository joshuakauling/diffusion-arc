"""DDP execution with one synchronization after local packed accumulation."""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_arc.model.flux.packing import PackedFluxForward

from .distributed import DistributedStrategy


class DdpStrategy(DistributedStrategy):
    """Adapt normal DDP gradient averaging to unequal local target counts.

    DDP must wrap the packed forward with buffer synchronization and static
    graphs disabled. Custom communication hooks and join() are unsupported.
    Ranks may perform different numbers of packs, but each needs a target.
    """

    strategy_label = "DDP"

    def __init__(self, transformer: DDP, *, compute_dtype: torch.dtype, backend: object) -> None:
        if not isinstance(transformer.module, PackedFluxForward):
            raise TypeError("DDP must wrap PackedFluxForward for packed training")
        if transformer.forward_sync_buffers or transformer.static_graph:
            raise ValueError("packed DDP requires forward_sync_buffers=False and static_graph=False")
        super().__init__(transformer.module, compute_dtype=compute_dtype, backend=backend)
        self.flux_model = transformer
        self.process_group = transformer.process_group
        self.world_size = dist.get_world_size(self.process_group)
        # Retaining bucket views avoids reallocating full-model gradient buffers.
        self.set_grad_to_none = not transformer.gradient_as_bucket_view

    def microbatch_context(self, *, is_last: bool):
        return nullcontext() if is_last else self.flux_model.no_sync()
