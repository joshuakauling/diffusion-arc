"""Learning-rate schedule indexed by the optimizer update that will use it."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_cosine_schedule_with_warmup

from ..config import CosineSchedulerConfig, LinearWarmupConstantConfig, TrainingConfig


class LinearWarmupConstantLR(LRScheduler):
    """Use base_lr / warmup_steps on update 1, then stay at base_lr after warmup.

    Advance once after each optimizer update. The ordinary scheduler state_dict
    preserves the completed-update count so restoring it does not restart warmup.
    """

    def __init__(self, optimizer: Optimizer, *, warmup_steps: int) -> None:
        if type(warmup_steps) is not int or warmup_steps <= 0:
            raise ValueError("warmup_steps must be a positive integer")
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        fraction = min(self.last_epoch + 1, self.warmup_steps) / self.warmup_steps
        return [base_lr * fraction for base_lr in self.base_lrs]


def build_lr_scheduler(
    config: TrainingConfig, *, optimizer: Optimizer
) -> LRScheduler | None:
    """Construct the configured schedule without changing its update convention."""
    schedule = config.scheduler
    if schedule is None:
        return None
    if isinstance(schedule, LinearWarmupConstantConfig):
        return LinearWarmupConstantLR(optimizer, warmup_steps=schedule.warmup_steps)
    if isinstance(schedule, CosineSchedulerConfig):
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=schedule.warmup_steps,
            num_training_steps=schedule.total_steps,
        )
    raise TypeError(f"unsupported LR schedule: {type(schedule).__name__}")


__all__ = ["LinearWarmupConstantLR", "build_lr_scheduler"]
