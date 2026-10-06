"""Blockwise FLUX sharding with synchronized gradient-shard accumulation."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from diffusion_arc.model.flux.packing import PackedFluxForward

from .distributed import DistributedStrategy


def shard_flux(flux_model: PackedFluxForward, *, runtime):
    """Shard blocks before the called packed root, then materialize resume shards.

    Keep FP32 parameters and reductions with the existing BF16 autocast path.
    Casting gathered weights to BF16 would also change normalization precision;
    that is a separate numerical experiment, not part of this first strategy.
    """
    if not isinstance(flux_model, PackedFluxForward):
        raise TypeError("shard_flux requires a prepared PackedFluxForward model")
    transformer = flux_model.transformer
    mesh = DeviceMesh.from_group(runtime.process_group, device_type=runtime.device.type)
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    for block in (*transformer.transformer_blocks, *transformer.single_transformer_blocks):
        fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
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

    def plan_microbatch_calls(self, microbatch_plans):
        microbatch_plans = tuple(sorted(
            microbatch_plans, key=lambda microbatch_plan: -microbatch_plan.joint_tokens,
        ))
        count = torch.tensor(len(microbatch_plans), device=self.device, dtype=torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.process_group)
        microbatch_schedule = microbatch_plans + (None,) * (int(count.item()) - len(microbatch_plans))
        return microbatch_schedule
