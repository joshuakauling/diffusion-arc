"""Grouped FLUX sharding with batch-end reduction and an explicit benchmark override."""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from diffusion_arc.model.flux.packing import PackedFluxForward

from .distributed import DistributedStrategy


def shard_flux(flux_model: PackedFluxForward, *, runtime, reshard_after_forward: bool = True,
               blocks_per_group: int = 1):
    """Shard consecutive block groups before the packed root and resume allocation.

    Keep FP32 parameters and reductions with the existing BF16 autocast path.
    Casting gathered weights to BF16 would also change normalization precision;
    that is a separate numerical experiment, not part of this first strategy.
    Blocks may retain gathered weights until backward to avoid its all-gather;
    parameters reshard after every backward. By default, gradients synchronize
    in the final scheduled microbatch; bounded benchmarks can reduce every call.
    """
    if not isinstance(flux_model, PackedFluxForward):
        raise TypeError("shard_flux requires a prepared PackedFluxForward model")
    if type(blocks_per_group) is not int or blocks_per_group < 1:
        raise ValueError("blocks_per_group must be a positive integer")
    transformer = flux_model.transformer
    mesh = DeviceMesh.from_group(runtime.process_group, device_type=runtime.device.type)
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    blocks = (*transformer.transformer_blocks, *transformer.single_transformer_blocks)
    # Group existing modules in forward order without changing parameter names
    # or the packed forward. A final partial group retains all remaining blocks.
    for start in range(0, len(blocks), blocks_per_group):
        group = list(blocks[start:start + blocks_per_group])
        fully_shard(group[0] if len(group) == 1 else group, mesh=mesh,
                    reshard_after_forward=reshard_after_forward, mp_policy=policy)
    fully_shard(flux_model, mesh=mesh, reshard_after_forward=False, mp_policy=policy)
    # On resume, meta parameters have no weight data yet. Allocate GPU memory
    # for this rank's shards; the checkpoint weights are loaded later.
    if next(flux_model.parameters()).is_meta:
        flux_model.to_empty(device=runtime.device)
    flux_model.set_requires_gradient_sync(True)
    flux_model.set_reshard_after_backward(True)
    return flux_model


class FsdpStrategy(DistributedStrategy):
    """Pad descending local packs to one collective-compatible call count."""

    strategy_label = "FSDP2"

    def __init__(self, transformer, *, compute_dtype, backend):
        if not isinstance(transformer, PackedFluxForward) or not isinstance(transformer, FSDPModule):
            raise TypeError("FSDP2 must shard the PackedFluxForward root")
        super().__init__(transformer, compute_dtype=compute_dtype, backend=backend)
        mesh = self.parameters[0].device_mesh
        self.process_group = mesh.get_group()
        self.world_size = dist.get_world_size(self.process_group)
        self.synchronize_each_microbatch = False

    @contextmanager
    def microbatch_context(self, *, is_last: bool):
        """Reduce at batch end, or after every call in the benchmark comparison.

        Earlier calls retain full local gradients while weights still follow
        their configured resharding policy. Restore the synchronization setting
        on exit; a failed backward still requires the caller's supervision.
        """
        self.flux_model.set_requires_gradient_sync(self.synchronize_each_microbatch or is_last)
        try:
            yield
        finally:
            self.flux_model.set_requires_gradient_sync(True)

    def plan_microbatch_calls(self, microbatch_plans, *, collective_call_count=None):
        """Use a precomputed count or agree on the longest rank schedule."""
        microbatch_plans = tuple(microbatch_plans)
        if collective_call_count is None:
            microbatch_plans = tuple(sorted(
                microbatch_plans, key=lambda microbatch_plan: -microbatch_plan.joint_tokens,
            ))
            count = torch.tensor(len(microbatch_plans), device=self.device, dtype=torch.int64)
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.process_group)
            collective_call_count = int(count.item())
        elif type(collective_call_count) is not int or collective_call_count < len(microbatch_plans):
            raise ValueError("collective_call_count must cover the local FSDP packs")
        microbatch_schedule = microbatch_plans + (None,) * (collective_call_count - len(microbatch_plans))
        return microbatch_schedule
