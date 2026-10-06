"""Compatibility imports for the former DDP benchmark runner."""

from .benchmark_runner import run_training_benchmark as run_training_ddp
from .run.state import RunProgress as DistributedProgress


__all__ = ["run_training_ddp", "DistributedProgress"]
