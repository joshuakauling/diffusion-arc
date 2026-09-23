"""Bound CUDA allocation history and save rank-local snapshots before DDP cleanup."""

import json
import os
from pathlib import Path
import sys

import torch

from .batch_diagnostics import BatchTrace


class CudaMemoryHistory:
    """Preserve the allocator's OOM event without synchronizing or retrying training.

    A snapshot taken after exception unwinding may already show freed tensors.
    Its OOM history event reconstructs the allocator state at the failed request.
    The ring buffer bounds event history, not the full allocation lifetime.
    """

    MAX_ENTRIES = 100_000

    def __init__(self, directory: Path, rank: int, device: torch.device, *, capture_step: int | None = None):
        if device.type != "cuda":
            raise ValueError("CUDA memory history requires a CUDA device")
        if capture_step is not None and (type(capture_step) is not int or capture_step < 1):
            raise ValueError("capture_step must be a positive global step")
        self.directory, self.rank, self.device = Path(directory), rank, device
        self.capture_step, self.captured = capture_step, False
        self.trace = BatchTrace(self.directory, rank, device, echo=False)
        self.settings = dict(max_entries=self.MAX_ENTRIES, context="all", stacks="all",
                             allocator_backend=torch.cuda.memory.get_allocator_backend(),
                             allocator_environment={name: os.environ.get(name) for name in
                                                    ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")})
        if capture_step is not None:
            self.settings["snapshot_after_first_backward_step"] = capture_step

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=self.MAX_ENTRIES,
            device=self.device, clear_history=True,
        )
        print(json.dumps(dict(event="memory_history_started", rank=self.rank, **self.settings)), flush=True)
        return self

    def observe(self, stage: str, details: dict) -> None:
        """Keep the original stage trace; snapshot only after the selected backward."""
        self.trace(stage, details)
        if (self.capture_step is not None and not self.captured and stage == "flux_backward_end"
                and self.trace.context.get("global_batch") == self.capture_step
                and self.trace.context.get("pack") == 1):
            # Preserve the allocation sequence leading to the old failure point:
            # no snapshot writes or new CUDA synchronization before this backward.
            self._save(None, suffix=f"step-{self.capture_step}-pack-1-backward")
            self.captured = True

    def _save(self, error, *, suffix: str | None = None):
        reason = ("checkpoint" if suffix else "oom" if isinstance(error, torch.OutOfMemoryError)
                  else "complete" if error is None else "failure")
        path = self.directory / f"rank-{self.rank}-{suffix or reason}.pickle"
        metadata = dict(event="memory_snapshot", rank=self.rank, reason=reason, snapshot=str(path),
                        last_stage=self.trace.last_stage, **self.trace.context,
                        settings=self.settings, error_type=type(error).__name__ if error else None,
                        message=str(error) if error else None,
                        snapshot_state=("after the selected backward returns; no added CUDA synchronization"
                                        if suffix else
                                        "after exception unwinding; inspect the OOM history event"
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
        snapshot_error = error
        if error is None and self.capture_step is not None and not self.captured:
            snapshot_error = RuntimeError("requested first-backward memory snapshot was not captured")
            diagnostic_errors.append(snapshot_error)
        for action in (lambda: self._save(snapshot_error),
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
