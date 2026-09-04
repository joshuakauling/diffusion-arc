"""CPU-only checks of benchmark boundaries, failure handling, and orchestration."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from diffusion_arc.training.single_gpu_config import (
    BenchmarkConfig, load_single_gpu_training_config,
)
from scripts import benchmark_single_gpu as bench
from tests.training.test_dataset import build_layout


def _config():
    config = load_single_gpu_training_config(
        Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b0.json"
    )
    return replace(
        config,
        loop=replace(config.loop, max_optimizer_steps=3, optimizer_batch_size=3),
        benchmark=BenchmarkConfig(warmup_steps=1, measured_steps=2, repetitions=3),
    )


def _samples(count):
    return [SimpleNamespace(key=f"source-{i}", prefix_token_ends=(10, 20, 30, 40)) for i in range(count)]


class BenchmarkTests(unittest.TestCase):
    def test_file_check_uses_existing_layout_without_loading_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            split, _ = build_layout(Path(directory))
            config = replace(_config(), data=replace(_config().data, split_directory=split))
            with patch.object(bench, "load_single_gpu_training_config", return_value=config):
                resolved, files = bench.check_inputs(Path("unused.json"))
                self.assertIs(resolved, config)
                self.assertEqual(len(files), 4)
                (split / "augmented_keys/train.json").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "input is not prepared"):
                    bench.check_inputs(Path("unused.json"))

    def _mock_cuda(self, stack):
        for name in ("synchronize", "reset_peak_memory_stats"):
            stack.enter_context(patch.object(torch.cuda, name))
        stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=100))
        stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=200))

    def test_full_pipeline_timing_excludes_warmup_and_bookkeeping(self):
        optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
        scheduler = Mock()
        recorder = bench._StepRecorder(_config(), "cuda:0", optimizer, scheduler)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            stack.enter_context(patch.object(bench.time, "perf_counter", side_effect=[0, 10, 11, 13, 14, 18, 19]))
            recorder.start()
            for _sample in recorder.samples(_samples(3)):
                recorder.step(torch.tensor(0.5), 3, 3)
        self.assertEqual([row["seconds"] for row in recorder.rows], [10, 2, 4])
        self.assertEqual(scheduler.step.call_count, 3)
        self.assertEqual(recorder.rows[0]["target_prefix_tokens"], [20, 30, 40])
        summary = bench.summarize(recorder.rows)
        self.assertEqual(summary["measured_steps"], 2)
        self.assertEqual(summary["median_step_seconds"], 3)
        self.assertAlmostEqual(summary["p95_step_seconds"], 3.9)
        self.assertEqual(summary["targets_per_second"], 1)
        self.assertAlmostEqual(summary["completed_sources_per_second"], 2 / 6)
        self.assertEqual(summary["peak_allocated_bytes"], 100)

    def test_metadata_tracks_fifo_targets_across_optimizer_boundaries(self):
        config = _config()
        config = replace(config, loop=replace(config.loop, optimizer_batch_size=4, nvarc_source_batch_size=2))
        recorder = bench._StepRecorder(config, "cuda:0", SimpleNamespace(param_groups=[{"lr": 1e-5}]), None)
        samples = _samples(2)
        samples[1].prefix_token_ends = (15, 25, 35, 50)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            list(recorder.samples(samples))
            recorder.step(torch.tensor(0.5), 4, 4)
            self.assertEqual(recorder.rows[0]["nvarc_padded_tokens"], 100)
            self.assertEqual(recorder.rows[0]["nvarc_unpadded_tokens"], 90)
            self.assertEqual(recorder.rows[0]["target_prefix_tokens"], [20, 30, 40, 25])
            self.assertEqual(list(recorder.pending_targets), [(35, False), (50, True)])
            with self.assertRaisesRegex(RuntimeError, "incomplete optimizer batch"):
                recorder.step(torch.tensor(0.5), 2, 2)
            self.assertEqual(recorder.rows[-1]["targets"], 2)

    def test_learning_rate_record_is_the_rate_used_not_the_next_rate(self):
        config = _config()
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-5)
        scheduler = bench.build_flux_scheduler(config, optimizer=optimizer)
        recorder = bench._StepRecorder(config, "cuda:0", optimizer, scheduler)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            for _ in recorder.samples(_samples(2)):
                optimizer.step()
                recorder.step(torch.tensor(0.5), 3, 3)
        self.assertEqual(recorder.rows[0]["learning_rate"], 0)
        self.assertAlmostEqual(recorder.rows[1]["learning_rate"], 1e-5 / 213)
        self.assertEqual(scheduler.last_epoch, 2)

    def test_missing_split_fails_before_gpu_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(_config(), data=replace(_config().data, split_directory=Path(directory) / "missing"))
            with (
                patch.object(bench, "load_single_gpu_training_config", return_value=config),
                patch.object(bench, "prepare_single_gpu_device") as prepare,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "not prepared"):
                    bench.benchmark_once(Path("config.json"), Path(directory) / "result")
            prepare.assert_not_called()
            self.assertFalse((Path(directory) / "result").exists())

    def test_orchestration_uses_fresh_sequential_processes_and_saves_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = Mock()
            def child(command, *, check):
                self.assertTrue(check)
                self.assertIn("--single-repetition", command)
                destination = Path(command[command.index("--output-dir") + 1])
                destination.mkdir()
                (destination / "steps.jsonl").write_text("")
            with (
                patch.object(bench, "check_inputs", return_value=(_config(), [])),
                patch.object(bench.subprocess, "run", side_effect=child) as run,
            ):
                bench.run_repetitions(root / "config.json", root / "success", on_repetition_end=commit)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(commit.call_count, 3)
            self.assertEqual(json.loads((root / "success/manifest.json").read_text())["status"], "complete")
            with (
                patch.object(bench, "check_inputs", return_value=(_config(), [])),
                patch.object(bench.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "child")) as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    bench.run_repetitions(root / "config.json", root / "failed", on_repetition_end=commit)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(json.loads((root / "failed/summary.json").read_text())["status"], "failed")

    def test_shared_loop_wiring_and_incomplete_run_is_not_a_result(self):
        for completed_steps in (3, 2):
            with self.subTest(completed_steps=completed_steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.json"
                config_path.write_text("{}")
                models = SimpleNamespace(
                    tokenizer=object(), nvarc_runner=object(), vae_encoder=object(),
                    transformer=object(), flux_compute_dtype=torch.bfloat16,
                )
                optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
                def run(samples, **kwargs):
                    self.assertEqual(kwargs["max_optimizer_steps"], 3)
                    self.assertEqual(kwargs["vae_microbatch_size"], 1)
                    self.assertNotIn("on_optimizer_step_metrics", kwargs)
                    for index, _ in enumerate(samples):
                        if index == completed_steps:
                            break
                        kwargs["on_optimizer_step"](torch.tensor(0.5), 3, 3)
                with ExitStack() as stack:
                    self._mock_cuda(stack)
                    for name, result in {
                        "check_inputs": (_config(), []), "prepare_single_gpu_device": "cuda:0",
                        "seed_single_gpu_process": None, "load_single_gpu_models": models,
                        "build_single_sample_loader": _samples(3), "build_flux_optimizer": optimizer,
                        "build_flux_scheduler": Mock(), "build_flow_generator": object(),
                    }.items():
                        stack.enter_context(patch.object(bench, name, return_value=result))
                    stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="fake GPU"))
                    stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1024)))
                    stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
                    stack.enter_context(patch.object(bench, "run_single_gpu_training", side_effect=run))
                    profiler = stack.enter_context(patch.object(torch.profiler, "profile", side_effect=AssertionError("no profiler")))
                    if completed_steps == 3:
                        bench.benchmark_once(config_path, root / "result")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "requires 3 full steps"):
                            bench.benchmark_once(config_path, root / "result")
                    profiler.assert_not_called()
                manifest = json.loads((root / "result/manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete" if completed_steps == 3 else "failed")
                self.assertEqual(len((root / "result/steps.jsonl").read_text().splitlines()), completed_steps)


if __name__ == "__main__":
    unittest.main()
