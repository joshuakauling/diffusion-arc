"""Shared target weighting and collective loss checks for distributed training."""

import torch
import torch.distributed as dist
from torch import Tensor

from .single import SingleDeviceStrategy


class DistributedStrategy(SingleDeviceStrategy):
    """Check rank-local inputs before forwards and losses after all backwards."""

    strategy_label = "distributed"

    def all_reduce_target_count(self, local_count: int) -> int:
        counts = torch.tensor((local_count, int(local_count == 0)), device=self.device)
        dist.all_reduce(counts, group=self.process_group)
        total, empty_ranks = counts.tolist()
        if empty_ranks:
            raise ValueError(f"each {self.strategy_label} rank requires at least one eligible training target")
        return total

    def check_microbatch_loss(self, loss: Tensor) -> None:
        # A local exception here could strand peers in backward collectives.
        pass

    def check_step_loss(self, loss: Tensor) -> None:
        finite = torch.isfinite(loss).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(finite.item()):
            raise FloatingPointError(f"FLUX loss is non-finite on a {self.strategy_label} rank")
