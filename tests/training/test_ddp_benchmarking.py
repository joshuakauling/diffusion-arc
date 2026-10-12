"""Warmup exclusion, slowest-rank statistics, and persistent partial artifacts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from diffusion_arc.training.run.benchmarking import (
    BenchmarkWindow, RunArtifacts, summarize_benchmark, summarize_repetitions,
)


def _row(phase, seconds, peak):
    return dict(event="step", phase=phase, max_rank_seconds=seconds, targets=48, sources=16,
                ranks=[dict(peak_allocated_bytes=peak, peak_reserved_bytes=peak * 2),
                       dict(peak_allocated_bytes=peak + 1, peak_reserved_bytes=peak * 3)])


def write_repetition(directory, times, *, warmup_steps=1):
    directory.mkdir(parents=True)
    rows = []
    for step, seconds in enumerate(times, 1):
        row = dict(_row("warmup" if step <= warmup_steps else "measure", seconds, 10),
                   optimizer_step=step)
        for rank, record in enumerate(row["ranks"]):
            record.update(rank=rank, targets=24, sources=8, source_indices=[rank],
                          target_indices=[rank], source_keys=[f"task-{rank}"],
                          target_prefix_tokens=[100], endpoint_targets=2,
                          flux_packs=3, learning_rates=[0.001])
        rows.append(row)
    (directory / "manifest.json").write_text(json.dumps(dict(
        world_size=2, seed=167, source_identity=dict(source_sha256="same-code", command=[str(directory)]),
    )))
    (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "summary.json").write_text(json.dumps(dict(
        event="complete", optimizer_steps=len(rows), benchmark=summarize_benchmark(rows),
    )))


class DdpBenchmarkingTests(unittest.TestCase):
    def test_excludes_warmup_from_latency_throughput_and_peak_memory(self):
        rows = [_row("warmup", 10000, 99999), _row("measure", 4, 10),
                _row("measure", 6, 30), _row("measure", 8, 20)]
        result = summarize_benchmark(rows)
        self.assertEqual(result["measured_steps"], 3)
        self.assertEqual(result["measured_seconds"], 18)
        self.assertEqual(result["median_step_seconds"], 6)
        self.assertAlmostEqual(result["p95_step_seconds"], 7.8)
        self.assertEqual(result["targets_per_second"], 8)
        self.assertEqual(result["sources_per_second"], 48 / 18)
        self.assertEqual(result["peak_allocated_bytes"], 31)
        self.assertEqual(result["peak_reserved_bytes"], 90)
        self.assertEqual(summarize_benchmark([_row("smoke", 1, 20)]), {"measured_steps": 0})

    def test_cpu_memory_is_unavailable_and_invalid_timings_are_rejected(self):
        row = _row("measure", 2, 5)
        row["ranks"] = [dict(peak_allocated_bytes=None, peak_reserved_bytes=None)]
        summary = summarize_benchmark([row])
        self.assertIsNone(summary["peak_allocated_bytes"])
        self.assertIsNone(summary["peak_reserved_bytes"])
        for seconds in (0, -1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "finite and positive"):
                summarize_benchmark([dict(row, max_rank_seconds=seconds)])

    def test_validates_window(self):
        self.assertEqual(BenchmarkWindow(10, 50).total_steps, 60)
        for values in ((-1, 50), (True, 50), (10, 0), (10, True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BenchmarkWindow(*values)

    def test_pools_repetitions_and_labels_partial_series(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [Path(directory) / str(index) for index in range(3)]
            for run, seconds in zip(runs, (2, 4, 8)):
                write_repetition(run, [9999, seconds])
            options = dict(window=BenchmarkWindow(1, 1), expected_repetitions=3)
            self.assertEqual(summarize_repetitions(runs[:1], **options)["status"], "partial")
            result = summarize_repetitions(runs, **options)
            self.assertEqual((result["status"], result["completed_repetitions"]), ("complete", 3))
            self.assertEqual(result["benchmark"]["measured_steps"], 3)
            self.assertEqual(result["benchmark"]["median_step_seconds"], 4)
            self.assertAlmostEqual(result["benchmark"]["p95_step_seconds"], 7.6)
            self.assertAlmostEqual(result["benchmark"]["targets_per_second"], 144 / 14)
            self.assertEqual(result["peak_memory_by_rank"][1]["peak_allocated_bytes"], 11)

    def test_rejects_incomplete_windows_and_changed_provenance_or_workload(self):
        for change in ("incomplete", "phase", "workload", "provenance", "summary"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                runs = [Path(directory) / str(index) for index in range(2)]
                for run in runs:
                    write_repetition(run, [100, 2])
                steps = runs[1] / "steps.jsonl"
                rows = [json.loads(line) for line in steps.read_text().splitlines()]
                if change == "incomplete":
                    rows.pop()
                elif change == "phase":
                    rows[0]["phase"] = "measure"
                elif change == "workload":
                    rows[1]["ranks"][1]["source_indices"] = [999]
                else:
                    path = runs[1] / ("manifest.json" if change == "provenance" else "summary.json")
                    record = json.loads(path.read_text())
                    if change == "provenance":
                        record["seed"] = 42
                    else:
                        record["benchmark"]["targets_per_second"] = 1000
                    path.write_text(json.dumps(record))
                steps.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    summarize_repetitions(runs, window=BenchmarkWindow(1, 1), expected_repetitions=3)

    def test_persists_success_and_failure_without_overwriting_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with RunArtifacts(root / "success") as emit:
                emit(dict(event="start", seed=167))
                emit(dict(event="packing_plan", metadata=dict(plan_sha256="prepared-plan"),
                          planning_seconds=0.5, ranks=[[dict(epoch_batch_index=0)]]))
                emit(_row("measure", 2, 5))
                emit(dict(event="complete", benchmark=dict(measured_steps=1)))
            self.assertEqual(json.loads((root / "success/manifest.json").read_text())["seed"], 167)
            self.assertEqual(len((root / "success/steps.jsonl").read_text().splitlines()), 1)
            packing = json.loads((root / "success/packing_plans.json").read_text())
            self.assertEqual(packing["metadata"]["plan_sha256"], "prepared-plan")
            self.assertEqual(packing["planning_seconds"], 0.5)
            self.assertEqual(json.loads((root / "success/summary.json").read_text())["benchmark"]["measured_steps"], 1)
            with self.assertRaises(FileExistsError):
                with RunArtifacts(root / "success"):
                    self.fail("existing run was overwritten")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with RunArtifacts(root / "failed") as emit:
                    emit(dict(event="start"))
                    emit(_row("warmup", 2, 5))
                    raise RuntimeError("training failed")
            self.assertTrue((root / "failed/steps.jsonl").read_text())
            self.assertFalse((root / "failed/summary.json").exists())
            self.assertEqual(json.loads((root / "failed/failure.json").read_text())["type"], "RuntimeError")

    def test_repetitions_reject_changed_pack_membership_with_equal_call_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [Path(directory) / str(index) for index in range(2)]
            for run in runs:
                write_repetition(run, [100, 2])
                path = run / "steps.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    for rank in row["ranks"]:
                        rank["flux_plan"] = dict(microbatch_plans=[dict(target_indices=[0], joint_tokens=1000)],
                            target_prefix_tokens=[100], collective_call_count=3)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            options = dict(window=BenchmarkWindow(1, 1), expected_repetitions=2)
            self.assertEqual(summarize_repetitions(runs, **options)["status"], "complete")
            path = runs[1] / "steps.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["ranks"][0]["flux_plan"]["microbatch_plans"][0]["target_indices"] = [1]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "workload"):
                summarize_repetitions(runs, **options)


if __name__ == "__main__":
    unittest.main()
