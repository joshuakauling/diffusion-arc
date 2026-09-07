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
    BenchmarkConfig, SingleGpuPaddingConfig, load_single_gpu_training_config,
)
from diffusion_arc.training.flux_input import PreparationMetrics, PreparedSource
from diffusion_arc.training.nvarc_schedule import NvarcBatchShape, NvarcSchedulingPolicy, NvarcSourceBatch
from diffusion_arc.training.flux_schedule import FluxBatchShape, FluxSchedulingPolicy, FluxTargetBatch
from diffusion_arc.sequence_buckets import SequenceLengthBuckets
from diffusion_arc.model import NvarcBatchRunner
from scripts import benchmark_single_gpu as bench
from tests.training.test_dataset import build_layout
from tests.model.test_nvarc_encoder import FakeNvarc
from tests.training.test_single_gpu_loop import (
    _sample, NumberedVaeEncoder, RecordingLoss, ScalarTransformer,
)


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


def _preparation(samples, padded_length=40):
    return PreparationMetrics(
        tuple(PreparedSource(str(sample.key), sample.prefix_token_ends[1:]) for sample in samples),
        (NvarcSourceBatch(tuple(range(len(samples))), padded_length),),
    )


def _record_fixed_step(recorder, loss, target_count, microbatch_count):
    """Supply a fixed-plan fixture when testing the recorder without training."""

    size = recorder.config.loop.flux_microbatch_size
    lengths = [length for length, _ in list(recorder.pending_targets)[:target_count]]
    buckets = SequenceLengthBuckets(recorder.config.padding.flux_bucket_boundaries)
    recorder.flux_schedule(tuple(
        FluxTargetBatch(tuple(range(start, min(start + size, target_count))),
                        buckets.padded_length(max(lengths[start:start + size])))
        for start in range(0, target_count, size)
    ))
    recorder.step(loss, target_count, microbatch_count)


class BenchmarkTests(unittest.TestCase):
    def test_b2_records_executed_shapes_and_fifo_tails_without_consuming_extra_window(self):
        config = _config()
        config = replace(config, loop=replace(config.loop, nvarc_source_batch_size=4,
                                             vae_microbatch_size=48, flux_microbatch_size=None),
                         flux_scheduling=FluxSchedulingPolicy((FluxBatchShape(1, 8, 2), FluxBatchShape(2, 8, 3))),
                         nvarc_scheduling=NvarcSchedulingPolicy(
                             (NvarcBatchShape(2, 4), NvarcBatchShape(2, 8)), 0,
                         ))
        source_samples = [_sample(count) for count in (2, 3, 1, 2, 2)]
        source_samples = [replace(s, key=replace(s.key, task_id=f"{i:08x}"))
                          for i, s in enumerate(source_samples)]
        consumed, executed, executed_flux = [], [], []

        def loss(model, batch, **kwargs):
            executed_flux.append((tuple(batch.clean_latents[:, 0, 0].tolist()),
                                  batch.text_inputs.encoder_hidden_states.shape[1]))
            return RecordingLoss()(model, batch, **kwargs)

        class Nvarc(FakeNvarc):
            def forward(self, **kwargs):
                executed.append(tuple(kwargs["input_ids"].shape))
                return super().forward(**kwargs)

        def samples():
            for sample in source_samples:
                consumed.append(sample)
                yield sample

        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=0.01)
        recorder = bench._StepRecorder(config, "cuda:0", optimizer, None)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            stack.enter_context(patch("diffusion_arc.training.flux_input._render_valid_grid_canvas",
                                      return_value=torch.zeros(1, 1, 1)))
            stack.enter_context(patch("diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
                                      side_effect=loss))
            recorder.start()
            bench.run_single_gpu_training(
                samples(), nvarc_runner=NvarcBatchRunner(Nvarc()), vae_encoder=NumberedVaeEncoder(),
                transformer=transformer, optimizer=optimizer, flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=1, optimizer_batch_size=3, min_context_pairs_before_target=2,
                nvarc_source_batch_size=4, vae_microbatch_size=48,
                nvarc_scheduling=config.nvarc_scheduling, on_preparation=recorder.preparation,
                flux_microbatch_size=None, flux_scheduling=config.flux_scheduling,
                on_flux_schedule=recorder.flux_schedule,
                on_optimizer_step=recorder.step,
            )
        row, = recorder.rows
        self.assertEqual(consumed, source_samples[:4])
        self.assertEqual(row["nvarc_calls"], len(executed))
        self.assertEqual([(b["batch_size"], b["padded_length"]) for b in row["nvarc_batches"]], executed)
        self.assertEqual(row["nvarc_padded_tokens"], sum(size * length for size, length in executed))
        self.assertEqual(row["nvarc_unpadded_tokens"], 14)
        self.assertEqual(executed, [(1, 6), (2, 4)])
        self.assertEqual(row["nvarc_padded_tokens"], 14)
        self.assertEqual([s["key"] for s in row["prepared_sources"]],
                         [str(source_samples[i].key) for i in (0, 1, 3)])
        self.assertEqual(row["target_prefix_tokens"], [4, 4, 6])
        self.assertEqual(list(recorder.pending_targets), [(4, True)])
        self.assertEqual(row["flux_microbatches"], 2)
        self.assertEqual(row["flux_padded_prefix_tokens"], [8, 8, 8])
        self.assertEqual([i for b in row["flux_batches"] for i in b["target_indices"]], [2, 0, 1])
        self.assertEqual([(tuple(i + 1 for i in b["target_indices"]), b["padded_length"])
                          for b in row["flux_batches"]], executed_flux)
        self.assertIsNone(recorder.flux_batches)

    def test_observed_batches_index_sources_across_multiple_preparation_windows(self):
        config = replace(_config(), loop=replace(_config().loop, optimizer_batch_size=6))
        recorder = bench._StepRecorder(config, "cuda:0", SimpleNamespace(param_groups=[{"lr": 1e-5}]), None)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            for sample in _samples(2):
                recorder.preparation(_preparation([sample]))
            _record_fixed_step(recorder, torch.tensor(0.5), 6, 6)
            recorder.preparation(_preparation(_samples(2), 64))
            _record_fixed_step(recorder, torch.tensor(0.5), 6, 6)
        self.assertEqual([b["source_indices"] for b in recorder.rows[0]["nvarc_batches"]], [[0], [1]])
        self.assertEqual([b["source_indices"] for b in recorder.rows[1]["nvarc_batches"]], [[0, 1]])
        self.assertEqual([row["nvarc_calls"] for row in recorder.rows], [2, 1])
        self.assertEqual([row["nvarc_padded_tokens"] for row in recorder.rows], [80, 128])

    def test_b1_records_real_and_bucketed_lengths_separately(self):
        config = load_single_gpu_training_config(
            Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b1.json"
        )
        recorder = bench._StepRecorder(config, "cuda:0", SimpleNamespace(param_groups=[{"lr": 1e-5}]), None)
        samples = _samples(16)
        samples[0].prefix_token_ends = (10, 256, 257, 513)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            recorder.preparation(_preparation(samples, 4096))
            _record_fixed_step(recorder, torch.tensor(0.5), 48, 48)
        row = recorder.rows[0]
        self.assertEqual(row["nvarc_unpadded_tokens"], 513 + 15 * 40)
        self.assertEqual(row["nvarc_padded_tokens"], 16 * 4096)
        self.assertEqual(row["target_prefix_tokens"], [256, 257, 513] + [20, 30, 40] * 15)
        self.assertEqual(row["flux_padded_prefix_tokens"], [256, 512, 768] + [256] * 45)
        self.assertEqual([s["key"] for s in row["prepared_sources"]], [f"source-{i}" for i in range(16)])
        self.assertEqual(row["completed_sources"], 16)

    def test_bucket_metadata_uses_each_physical_microbatch_maximum(self):
        config = _config()
        config = replace(config,
                         loop=replace(config.loop, optimizer_batch_size=4,
                                      nvarc_source_batch_size=2, flux_microbatch_size=2),
                         padding=SingleGpuPaddingConfig((64, 8400), (32, 64, 8400)))
        recorder = bench._StepRecorder(config, "cuda:0", SimpleNamespace(param_groups=[{"lr": 1e-5}]), None)
        samples = _samples(2)
        samples[1].prefix_token_ends = (15, 25, 35, 50)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            recorder.preparation(_preparation(samples, 64))
            _record_fixed_step(recorder, torch.tensor(0.5), 4, 2)
        self.assertEqual(recorder.rows[0]["nvarc_padded_tokens"], 128)
        self.assertEqual(recorder.rows[0]["flux_padded_prefix_tokens"], [32, 32, 64, 64])
        self.assertEqual(list(recorder.pending_targets), [(35, False), (50, True)])

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
            for sample in _samples(3):
                recorder.preparation(_preparation([sample]))
                _record_fixed_step(recorder, torch.tensor(0.5), 3, 3)
        self.assertEqual([row["seconds"] for row in recorder.rows], [10, 2, 4])
        self.assertEqual(scheduler.step.call_count, 3)
        self.assertEqual(recorder.rows[0]["target_prefix_tokens"], [20, 30, 40])
        self.assertEqual(recorder.rows[0]["flux_padded_prefix_tokens"], [20, 30, 40])
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
            recorder.preparation(_preparation(samples, 50))
            _record_fixed_step(recorder, torch.tensor(0.5), 4, 4)
            self.assertEqual(recorder.rows[0]["nvarc_padded_tokens"], 100)
            self.assertEqual(recorder.rows[0]["nvarc_unpadded_tokens"], 90)
            self.assertEqual(recorder.rows[0]["target_prefix_tokens"], [20, 30, 40, 25])
            self.assertEqual(list(recorder.pending_targets), [(35, False), (50, True)])
            with self.assertRaisesRegex(RuntimeError, "incomplete optimizer batch"):
                _record_fixed_step(recorder, torch.tensor(0.5), 2, 2)
            self.assertEqual(recorder.rows[-1]["targets"], 2)

    def test_learning_rate_record_is_the_rate_used_not_the_next_rate(self):
        config = _config()
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-5)
        scheduler = bench.build_flux_scheduler(config, optimizer=optimizer)
        recorder = bench._StepRecorder(config, "cuda:0", optimizer, scheduler)
        with ExitStack() as stack:
            self._mock_cuda(stack)
            recorder.start()
            for sample in _samples(2):
                recorder.preparation(_preparation([sample]))
                optimizer.step()
                _record_fixed_step(recorder, torch.tensor(0.5), 3, 3)
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
                config = replace(
                    _config(), padding=SingleGpuPaddingConfig(flux_bucket_boundaries=(256, 512, 8400)),
                    nvarc_scheduling=NvarcSchedulingPolicy((NvarcBatchShape(2, 8400),), 2560),
                )
                def run(samples, **kwargs):
                    self.assertEqual(kwargs["max_optimizer_steps"], 3)
                    self.assertEqual(kwargs["vae_microbatch_size"], 1)
                    self.assertEqual(kwargs["flux_bucket_boundaries"], (256, 512, 8400))
                    self.assertNotIn("on_optimizer_step_metrics", kwargs)
                    self.assertIs(kwargs["nvarc_scheduling"], config.nvarc_scheduling)
                    self.assertIs(kwargs["flux_scheduling"], config.flux_scheduling)
                    for index, sample in enumerate(samples):
                        if index == completed_steps:
                            break
                        kwargs["on_preparation"](_preparation([sample]))
                        kwargs["on_flux_schedule"](tuple(FluxTargetBatch((i,), 256) for i in range(3)))
                        kwargs["on_optimizer_step"](torch.tensor(0.5), 3, 3)
                with ExitStack() as stack:
                    self._mock_cuda(stack)
                    for name, result in {
                        "check_inputs": (config, []), "prepare_single_gpu_device": "cuda:0",
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
