"""Select a distribution strategy for single-device, DDP or FSDP2 training."""

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FSDPModule

from .ddp import DdpStrategy
from .single import SingleDeviceStrategy
from .fsdp import FsdpStrategy


def build_distribution_strategy(
    transformer: nn.Module, *, compute_dtype: torch.dtype, backend: object,
) -> SingleDeviceStrategy:
    """Keep wrapper detection at the boundary; the trainer uses one contract."""
    strategy_class = (DdpStrategy if isinstance(transformer, DDP) else
                      FsdpStrategy if isinstance(transformer, FSDPModule) else SingleDeviceStrategy)
    return strategy_class(transformer, compute_dtype=compute_dtype, backend=backend)


__all__ = ["SingleDeviceStrategy", "DdpStrategy", "FsdpStrategy", "build_distribution_strategy"]
