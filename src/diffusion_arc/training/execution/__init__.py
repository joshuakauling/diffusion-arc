"""Select execution mechanics for a prepared single-process or DDP model."""

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .ddp import DdpExecution
from .single import SingleExecution


def build_execution(
    transformer: nn.Module, *, compute_dtype: torch.dtype, backend: object,
) -> SingleExecution:
    """Keep wrapper detection at the boundary; the trainer uses one contract."""
    execution = DdpExecution if isinstance(transformer, DDP) else SingleExecution
    return execution(transformer, compute_dtype=compute_dtype, backend=backend)


__all__ = ["SingleExecution", "DdpExecution", "build_execution"]
