"""Flush rank-local stage and allocator records without synchronizing CUDA streams."""

import json
from pathlib import Path
import time

import torch


class BatchTrace:
    """Keep the last entered pack/stage visible even if another rank aborts."""

    def __init__(self, directory: Path, rank: int, device: torch.device, *, echo: bool = True):
        self.path = Path(directory) / f"rank-{rank}-trace.jsonl"
        self.rank, self.device = rank, device
        self.context = {}
        self.last_stage = None
        self.echo = echo

    def __call__(self, stage: str, details: dict) -> None:
        if stage == "batch_start":
            self.context = dict(details)
        elif stage == "flux_forward_start":
            self.context.update(details)
        memory = {}
        if self.device.type == "cuda":
            stats = torch.cuda.memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            memory = {key: stats[key] for key in (
                "allocated_bytes.all.current", "reserved_bytes.all.current",
                "active_bytes.all.current", "inactive_split_bytes.all.current",
                "allocated_bytes.all.peak", "num_alloc_retries", "num_ooms",
            )}
            memory.update(device_free_bytes=free, device_total_bytes=total)
        record = {"event": "diagnostic", "rank": self.rank, "stage": stage,
                  "previous_stage": self.last_stage, "time_ns": time.time_ns(),
                  **self.context, **details, "memory": memory}
        self.last_stage = stage
        line = json.dumps(record, allow_nan=False)
        # Separate files avoid rank-zero collectives in the error-reporting path.
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if self.echo:
            print(line, flush=True)
