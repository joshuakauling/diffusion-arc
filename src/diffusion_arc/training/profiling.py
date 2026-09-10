"""Opt-in NVTX ranges and a bounded CUDA profiler capture for training."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch


NSYS_WARMUP_STEPS = 4
NSYS_ACTIVE_STEPS = 3
_NVTX_ENABLED = ContextVar("training_nvtx_enabled", default=False)


def training_range(name: str, *, torch_range: bool = False):
    """Use NVTX only inside Nsight runs; preserve existing Torch ranges otherwise."""
    if _NVTX_ENABLED.get():
        return torch.cuda.nvtx.range(name)
    return torch.profiler.record_function(name) if torch_range else nullcontext()


class NsightCapture:
    """Capture updates 5–7, with model loading and four real updates excluded."""

    def __init__(self, device):
        self.device = device
        self.completed_steps = 0
        self.active = False
        self.step_range = False
        self.memory_samples = []

    def record_memory(self):
        """Read allocator counters without adding a CUDA synchronization."""
        stats = torch.cuda.memory_stats(self.device)
        self.memory_samples.append({
            "completed_updates": self.completed_steps,
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "allocation_retries": stats.get("num_alloc_retries", 0),
            "oom_count": stats.get("num_ooms", 0),
        })
        # Each following peak covers one callback-to-callback update interval.
        torch.cuda.reset_peak_memory_stats(self.device)

    def step(self):
        """Advance after AdamW and the learning-rate scheduler have completed."""
        self.completed_steps += 1
        self.record_memory()
        if self.completed_steps == NSYS_WARMUP_STEPS + NSYS_ACTIVE_STEPS:
            self.close()
        else:
            if self.step_range:
                torch.cuda.nvtx.range_pop()
                self.step_range = False
            if self.completed_steps == NSYS_WARMUP_STEPS:
                torch.cuda.synchronize(self.device)
                torch.cuda.profiler.start()
                self.active = True
            if self.active:
                torch.cuda.nvtx.range_push(f"optimizer_step/{self.completed_steps + 1}")
                self.step_range = True

    def close(self):
        """Drain GPU work before stopping, including when training raises."""
        if self.active:
            try:
                torch.cuda.synchronize(self.device)
            finally:
                try:
                    if self.step_range:
                        torch.cuda.nvtx.range_pop()
                        self.step_range = False
                finally:
                    self.active = False
                    torch.cuda.profiler.stop()


@contextmanager
def nsight_capture(device):
    """Restore annotation state on failure and reject incomplete input streams."""
    capture = NsightCapture(device)
    token = _NVTX_ENABLED.set(True)
    try:
        capture.record_memory()
        yield capture
        if capture.completed_steps != NSYS_WARMUP_STEPS + NSYS_ACTIVE_STEPS:
            raise RuntimeError("Nsight capture requires seven completed optimizer updates")
    finally:
        try:
            capture.close()
        finally:
            _NVTX_ENABLED.reset(token)
