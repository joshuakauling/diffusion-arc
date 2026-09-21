"""Measurement windows and local artifacts for training-only GPU experiments."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics


@dataclass(frozen=True)
class BenchmarkWindow:
    """Exclude initial training updates from benchmark statistics."""

    warmup_steps: int
    measured_steps: int

    def __post_init__(self):
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        if type(self.measured_steps) is not int or self.measured_steps <= 0:
            raise ValueError("measured_steps must be a positive integer")

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.measured_steps


def summarize_benchmark(rows: list[dict]) -> dict:
    """Pool measured updates using the slowest rank duration for each step."""
    measured = [row for row in rows if row["phase"] == "measure"]
    if not measured:
        return {"measured_steps": 0}
    times = sorted(row["max_rank_seconds"] for row in measured)
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("measured step times must be finite and positive")
    position = 0.95 * (len(times) - 1)
    lower = int(position)
    p95 = times[lower] + (times[min(lower + 1, len(times) - 1)] - times[lower]) * (position - lower)
    seconds = sum(times)
    peaks = {name: max((rank[name] for row in measured for rank in row["ranks"]
                       if rank[name] is not None), default=None)
             for name in ("peak_allocated_bytes", "peak_reserved_bytes")}
    return dict(measured_steps=len(measured), measured_seconds=seconds,
                median_step_seconds=statistics.median(times), p95_step_seconds=p95,
                targets_per_second=sum(row["targets"] for row in measured) / seconds,
                sources_per_second=sum(row["sources"] for row in measured) / seconds,
                **peaks)


def summarize_repetitions(directories: list[Path], *, window: BenchmarkWindow,
                          expected_repetitions: int) -> dict:
    """Pool complete, comparable fresh runs; label an unfinished series as partial."""
    if not 0 < len(directories) <= expected_repetitions:
        raise ValueError("expected one to expected_repetitions completed runs")
    pooled, repetitions = [], []
    reference_manifest = reference_workload = None
    workload_fields = ("rank", "targets", "sources", "source_indices", "target_indices",
                       "source_keys", "target_prefix_tokens", "endpoint_targets",
                       "flux_packs", "learning_rates")
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        if (summary["event"] != "complete" or summary["optimizer_steps"] != window.total_steps
                or len(rows) != window.total_steps):
            raise ValueError(f"incomplete benchmark repetition: {directory.name}")
        for step, row in enumerate(rows, 1):
            phase = "warmup" if step <= window.warmup_steps else "measure"
            if (row["optimizer_step"] != step or row["phase"] != phase
                    or [rank["rank"] for rank in row["ranks"]] != list(range(manifest["world_size"]))):
                raise ValueError(f"invalid step window or ranks: {directory.name}")
        benchmark = summarize_benchmark(rows)
        if summary["benchmark"] != benchmark:
            raise ValueError(f"benchmark summary disagrees with step records: {directory.name}")
        # Output paths differ; code, recipe, dataset and runtime must be identical.
        manifest["source_identity"].pop("command", None)
        workload = [[{key: rank[key] for key in workload_fields} for rank in row["ranks"]]
                    for row in rows]
        if reference_manifest is None:
            reference_manifest, reference_workload = manifest, workload
        elif manifest != reference_manifest or workload != reference_workload:
            raise ValueError(f"benchmark repetitions differ in provenance or workload: {directory.name}")
        pooled.extend(rows)
        repetitions.append(dict(directory=directory.name, benchmark=benchmark))
    measured = [row for row in pooled if row["phase"] == "measure"]
    peaks = [dict(rank=rank, **{
        name: max((row["ranks"][rank][name] for row in measured
                   if row["ranks"][rank][name] is not None), default=None)
        for name in ("peak_allocated_bytes", "peak_reserved_bytes")
    }) for rank in range(reference_manifest["world_size"])]
    return dict(status="complete" if len(directories) == expected_repetitions else "partial",
                completed_repetitions=len(directories), expected_repetitions=expected_repetitions,
                warmup_steps_per_repetition=window.warmup_steps,
                measured_steps_per_repetition=window.measured_steps,
                benchmark=summarize_benchmark(pooled), peak_memory_by_rank=peaks,
                repetitions=repetitions)


class RunArtifacts:
    """Write rank-zero JSON records to a new directory; preserve partial runs."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.stream = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / "steps.jsonl").open("x", encoding="utf-8")
        return self

    def _write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def __call__(self, record: dict):
        if record["event"] == "start":
            self._write_json("manifest.json", record)
        elif record["event"] == "step":
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.stream.flush()
        elif record["event"] == "complete":
            self._write_json("summary.json", record)
        print(json.dumps(record, allow_nan=False), flush=True)

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is not None:
                self._write_json("failure.json", dict(type=error_type.__name__, message=str(error)))
        finally:
            self.stream.close()
