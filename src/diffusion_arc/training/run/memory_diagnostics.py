"""Bound CUDA allocation history and save rank-local snapshots before DDP cleanup."""

import json
import os
from pathlib import Path
import sys

import torch


class CudaMemoryHistory:
    """Preserve the allocator's OOM event without synchronizing or retrying training.

    A snapshot taken after exception unwinding may already show freed tensors.
    Its OOM history event reconstructs the allocator state at the failed request.
    The ring buffer bounds event history, not the full allocation lifetime.
    """

    MAX_ENTRIES = 100_000

    def __init__(self, directory: Path, rank: int, device: torch.device):
        if device.type != "cuda":
            raise ValueError("CUDA memory history requires a CUDA device")
        self.directory, self.rank, self.device = Path(directory), rank, device
        self.settings = dict(max_entries=self.MAX_ENTRIES, context="all", stacks="all",
                             allocator_backend=torch.cuda.memory.get_allocator_backend(),
                             allocator_environment={name: os.environ.get(name) for name in
                                                    ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")})

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=self.MAX_ENTRIES,
            device=self.device, clear_history=True,
        )
        print(json.dumps(dict(event="memory_history_started", rank=self.rank, **self.settings)), flush=True)
        return self

    def _save(self, error):
        reason = ("oom" if isinstance(error, torch.OutOfMemoryError)
                  else "complete" if error is None else "failure")
        path = self.directory / f"rank-{self.rank}-{reason}.pickle"
        metadata = dict(event="memory_snapshot", rank=self.rank, reason=reason, snapshot=str(path),
                        settings=self.settings, error_type=type(error).__name__ if error else None,
                        message=str(error) if error else None,
                        snapshot_state=("after exception unwinding; inspect the OOM history event"
                                        if reason == "oom" else
                                        "after exception unwinding" if error else "after the final training step"))
        try:
            # Snapshot first: even memory counters or failure logging may fail.
            temporary = path.with_suffix(".pickle.partial")
            torch.cuda.memory._dump_snapshot(str(temporary))
            temporary.replace(path)
            metadata["snapshot_bytes"] = path.stat().st_size
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                metadata["memory_after_snapshot"] = dict(
                    stats=torch.cuda.memory_stats(self.device), device_free_bytes=free, device_total_bytes=total,
                )
            except Exception as stats_error:
                metadata["memory_stats_error"] = repr(stats_error)
        except Exception as snapshot_error:
            metadata["snapshot_error"] = repr(snapshot_error)
            raise
        finally:
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            print(json.dumps(metadata, allow_nan=False), flush=True)

    def __exit__(self, error_type, error, traceback):
        diagnostic_errors = []
        for action in (lambda: self._save(error),
                       lambda: torch.cuda.memory._record_memory_history(enabled=None, device=self.device)):
            try:
                action()
            except Exception as failure:
                diagnostic_errors.append(failure)
                if error is not None:
                    error.add_note(f"CUDA memory diagnostics failed: {failure!r}")
                    print(f"Rank {self.rank} memory diagnostics failed: {failure!r}", file=sys.stderr, flush=True)
        if error is None and diagnostic_errors:
            raise diagnostic_errors[0]
        return False
