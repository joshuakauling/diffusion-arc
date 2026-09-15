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

from scripts import benchmark_single_gpu as bench
import torch

from diffusion_arc.training.config import BenchmarkConfig, load_training_config
from diffusion_arc.training.step.encoding import PreparationMetrics, PreparedSource
from diffusion_arc.training.step.flux_batching import (
    FluxTargetBatch,
)
from diffusion_arc.training.trainer import StepDiagnostics, TargetMetrics, TrainStepResult

from tests.training.test_data import build_layout


def _config():
    config = load_training_config(
        Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b3.json"
    , expected_type="baseline")
    return replace(
        config,
        loop=replace(config.loop, max_optimizer_steps=3, optimizer_batch_size=3,
                     vae_microbatch_size=1),
        benchmark=BenchmarkConfig(warmup_steps=1, measured_steps=2, repetitions=3),
    )


def _samples(count):
    return [SimpleNamespace(key=f"source-{i}", prefix_token_ends=(10, 20, 30, 40),
                            target_grids=(None,) * 4) for i in range(count)]


def _result(samples, *, nvarc_batches=None, flux_batches=None, learning_rates=(1e-5,)):
    """CPU result fixture; target metadata belongs entirely to this update."""
    sources = tuple(PreparedSource(str(sample.key), sample.prefix_token_ends[1:]) for sample in samples)
    lengths = tuple(sample.prefix_token_ends[-1] for sample in samples)
    prefixes = tuple(length for source in sources for length in source.prefix_ends)
    if nvarc_batches is None:
        nvarc_batches = (tuple(range(len(samples))),)
    if flux_batches is None:
        flux_batches = tuple(FluxTargetBatch((i,), prefix + 900) for i, prefix in enumerate(prefixes))
    preparation = PreparationMetrics(sources, lengths, nvarc_batches, vae_calls=1)
    return TrainStepResult(
        loss=0.5, learning_rates=learning_rates, gradient_norm=0.25, gradient_clipped=False,
        step_seconds=1.5, source_count=len(samples), target_count=len(prefixes),
        diagnostics=StepDiagnostics(preparation, flux_batches, prefixes),
        targets=tuple(TargetMetrics(loss=0.5, timestep=0.5, context_pair_count=i % 3 + 2,
                                    has_boundary=True) for i in range(len(prefixes))),
        nvarc_tokens=sum(lengths), nvarc_calls=len(nvarc_batches), vae_calls=1,
        flux_text_tokens=sum(prefixes), flux_image_tokens=900 * len(prefixes),
        flux_joint_tokens=sum(prefixes) + 900 * len(prefixes), flux_forward_calls=len(flux_batches),
    )


class BenchmarkTests(unittest.TestCase):
    def test_observed_nvarc_batches_use_step_local_source_indices(self):
        config = replace(_config(), loop=replace(_config().loop, optimizer_batch_size=6))
        recorder = bench._StepRecorder(config, "cuda:0")
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            recorder.step(_result(_samples(2), nvarc_batches=((0,), (1,))))
            recorder.step(_result(_samples(2)))
        self.assertEqual([b["source_indices"] for b in recorder.rows[0]["nvarc_batches"]], [[0], [1]])
        self.assertEqual([b["source_indices"] for b in recorder.rows[1]["nvarc_batches"]], [[0, 1]])
        self.assertEqual([row["nvarc_calls"] for row in recorder.rows], [2, 1])
        self.assertEqual([row["nvarc_processed_tokens"] for row in recorder.rows], [80, 80])

    def test_records_packed_nvarc_and_flux_token_workload(self):
        config = replace(_config(), loop=replace(_config().loop, optimizer_batch_size=6))
        recorder = bench._StepRecorder(config, "cuda:0")
        samples = _samples(2)
        samples[0].prefix_token_ends = (10, 256, 257, 513)
        batches = (FluxTargetBatch((0, 1, 2), 3726), FluxTargetBatch((3, 4, 5), 2790))
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            recorder.step(_result(samples, flux_batches=batches))
        row = recorder.rows[0]
        self.assertEqual(row["nvarc_processed_tokens"], 513 + 40)
        self.assertEqual(row["target_prefix_tokens"], [256, 257, 513, 20, 30, 40])
        self.assertEqual([batch["joint_tokens"] for batch in row["flux_batches"]], [3726, 2790])
        self.assertEqual(row["completed_sources"], 2)
        self.assertEqual(row["flux_joint_tokens"], 6516)
        self.assertEqual(row["gradient_norm"], 0.25)
        self.assertEqual(row["step_seconds"], 1.5)
        self.assertNotIn("flux_padded_prefix_tokens", row)

    def test_file_check_uses_existing_layout_without_loading_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            split, _ = build_layout(Path(directory))
            config = replace(_config(), data=replace(_config().data, split_directory=split))
            with patch.object(bench, "load_training_config", return_value=config):
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
        recorder = bench._StepRecorder(_config(), "cuda:0")
        with ExitStack() as stack:
            self._mock_cuda(stack)
            stack.enter_context(patch.object(bench.time, "perf_counter", side_effect=[0, 10, 11, 13, 14, 18, 19]))
            recorder.start()
            for sample in _samples(3):
                recorder.step(_result([sample]))
        self.assertEqual([row["seconds"] for row in recorder.rows], [10, 2, 4])
        self.assertEqual(recorder.rows[0]["target_prefix_tokens"], [20, 30, 40])
        summary = bench.summarize(recorder.rows)
        self.assertEqual(summary["measured_steps"], 2)
        self.assertEqual(summary["median_step_seconds"], 3)
        self.assertAlmostEqual(summary["p95_step_seconds"], 3.9)
        self.assertEqual(summary["targets_per_second"], 1)
        self.assertAlmostEqual(summary["completed_sources_per_second"], 2 / 6)
        self.assertEqual(summary["peak_allocated_bytes"], 100)

    def test_incomplete_result_is_retained_for_failure_diagnostics(self):
        config = replace(_config(), loop=replace(_config().loop, optimizer_batch_size=6))
        recorder = bench._StepRecorder(config, "cuda:0")
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            with self.assertRaisesRegex(RuntimeError, "incomplete optimizer batch"):
                recorder.step(_result(_samples(1)))
        self.assertEqual(recorder.rows[-1]["targets"], 3)

    def test_learning_rate_record_comes_from_the_returned_used_rate(self):
        recorder = bench._StepRecorder(_config(), "cuda:0")
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            recorder.step(_result(_samples(1), learning_rates=(0.0,)))
            recorder.step(_result(_samples(1), learning_rates=(1e-5 / 213,)))
        self.assertEqual(recorder.rows[0]["learning_rate"], 0)
        self.assertAlmostEqual(recorder.rows[1]["learning_rate"], 1e-5 / 213)

    def test_missing_split_fails_before_gpu_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(_config(), data=replace(_config().data, split_directory=Path(directory) / "missing"))
            with (
                patch.object(bench, "load_training_config", return_value=config),
                patch.object(bench, "prepare_device") as prepare,
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

    def test_trainer_wiring_and_incomplete_run_is_not_a_result(self):
        for completed_steps in (3, 2):
            with self.subTest(completed_steps=completed_steps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.json"
                config_path.write_text("{}")
                models = SimpleNamespace(
                    tokenizer=object(), nvarc_encoder=object(), vae_encoder=object(),
                    flux_transformer=object(), flux_compute_dtype=torch.bfloat16,
                    flux_packing_backend=SimpleNamespace(binding="flash_attn_3.flash_attn_interface.flash_attn_varlen_func"),
                )
                optimizer = SimpleNamespace(param_groups=[{"lr": 1e-5}])
                scheduler = Mock()
                config = _config()
                flow_sampler, loss_function = object(), object()
                executed = []
                def make_trainer(**kwargs):
                    self.assertIs(kwargs["nvarc_encoder"], models.nvarc_encoder)
                    self.assertIs(kwargs["vae_encoder"], models.vae_encoder)
                    self.assertIs(kwargs["flux_transformer"], models.flux_transformer)
                    self.assertIs(kwargs["optimizer"], optimizer)
                    self.assertIs(kwargs["scheduler"], scheduler)
                    self.assertIs(kwargs["flow_sampler"], flow_sampler)
                    self.assertIs(kwargs["loss_function"], loss_function)
                    self.assertEqual(kwargs["nvarc_max_tokens"], config.nvarc_max_tokens)
                    self.assertFalse(any(name.startswith("on_") for name in kwargs))
                    def step(sequences):
                        executed.append(sequences)
                        return _result(sequences, flux_batches=(FluxTargetBatch((2, 0, 1), 2790),))
                    return SimpleNamespace(step=step)
                with ExitStack() as stack:
                    self._mock_cuda(stack)
                    for name, result in {
                        "check_inputs": (config, []), "prepare_device": "cuda:0",
                        "seed_process": None, "load_models": models,
                        "build_dataloader": _samples(completed_steps), "build_optimizer": optimizer,
                        "build_lr_scheduler": scheduler, "build_flow_generator": object(),
                        "build_flow_sampler": flow_sampler, "build_velocity_loss": loss_function,
                    }.items():
                        stack.enter_context(patch.object(bench, name, return_value=result))
                    stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value="fake GPU"))
                    stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1024)))
                    stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
                    stack.enter_context(patch.object(bench, "Trainer", side_effect=make_trainer))
                    stack.enter_context(patch.object(bench, "version", return_value="test-pinned"))
                    profiler = stack.enter_context(patch.object(torch.profiler, "profile", side_effect=AssertionError("no profiler")))
                    if completed_steps == 3:
                        bench.benchmark_once(config_path, root / "result")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "requires 3 full steps"):
                            bench.benchmark_once(config_path, root / "result")
                    profiler.assert_not_called()
                    scheduler.step.assert_not_called()
                manifest = json.loads((root / "result/manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete" if completed_steps == 3 else "failed")
                rows = [json.loads(line) for line in (root / "result/steps.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), completed_steps)
                self.assertEqual(len(executed), completed_steps)
                self.assertEqual(manifest["execution"]["flux_packed_token_budget"], 12288)
                self.assertEqual(manifest["execution"]["modulation_reduction"], "fp32")
                self.assertEqual(rows[0]["flux_batches"], [dict(batch_size=3,
                    joint_tokens=2790, target_indices=[2, 0, 1])])


if __name__ == "__main__":
    unittest.main()
