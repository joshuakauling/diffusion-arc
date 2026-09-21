"""Own a torchrun process group and select the process's CUDA device."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import os

import torch
import torch.distributed as dist

from .config import TrainingConfig
from .setup import prepare_device


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    process_group: object


def _torchrun_ranks() -> tuple[int, int, int]:
    try:
        rank, local_rank, world_size, local_world_size = (
            int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
    except (KeyError, ValueError) as error:
        raise ValueError("start the DDP runner with torchrun (RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE)") from error
    if not (0 <= rank < world_size and 0 <= local_rank < local_world_size <= world_size):
        raise ValueError("invalid torchrun rank/world-size settings")
    if local_world_size > torch.cuda.device_count():
        raise ValueError("torchrun requested more local processes than visible CUDA devices")
    return rank, local_rank, world_size


@contextmanager
def distributed_runtime(config: TrainingConfig, *, timeout_seconds: int = 120):
    """Initialize NCCL after device selection and always release our group.

    LOCAL_RANK selects a visible CUDA device, overriding the recipe's single
    device. The caller must let failures reach torchrun's process supervisor.
    No barrier is used during error cleanup.
    """
    if dist.is_initialized():
        raise RuntimeError("the DDP runner must own its process group")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    rank, local_rank, world_size = _torchrun_ranks()
    # Each rank gets the same CPU tensor parallelism, independent of container size.
    torch.set_num_threads(8)
    device = prepare_device(replace(config, device=f"cuda:{local_rank}"))
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=timeout_seconds))
    try:
        yield DistributedRuntime(rank, local_rank, world_size, device, dist.group.WORLD)
    finally:
        dist.destroy_process_group()
