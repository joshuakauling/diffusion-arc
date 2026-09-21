"""Exercise real global planning, flow sampling, DDP steps and metric aggregation."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import gzip
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.training.config import load_training_config, LinearWarmupConstantConfig
from diffusion_arc.training.ddp_runner import run_training_ddp
from diffusion_arc.training.distributed import DistributedRuntime
from diffusion_arc.training.distributed_data import plan_distributed_batches
from diffusion_arc.training.run.batch_diagnostics import BatchTrace
from diffusion_arc.training.run.benchmarking import BenchmarkWindow
from diffusion_arc.training.setup import build_optimizer
from diffusion_arc.training.step.distributed_flow import DistributedFlowSampler
from diffusion_arc.training.step.loss import build_velocity_loss
from diffusion_arc.training.step.lr_schedule import build_lr_scheduler
from diffusion_arc.training.trainer import Trainer
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_trainer import (
    FrozenText, FrozenVae, _model, _render, _gradients, GRAD_TOLERANCE, PARAM_TOLERANCE,
)
from tests.training.test_trainer import _sample


ROOT = Path(__file__).resolve().parents[2]


def _config():
    base = load_training_config(ROOT / "configs/training/arc_agi_1_epoch.json")
    recipe = base.recipe_dict()
    recipe["data"]["expected_task_count"] = 11
    recipe["flow"].update(endpoint_targets_per_full_batch=2, endpoint_targets_in_final_batch=1)
    recipe["loop"].update(optimizer_batch_size=12, max_optimizer_steps=3,
                          expected_target_count=33, expected_final_batch_target_count=9)
    return replace(base, recipe=recipe, data=replace(base.data, num_workers=0),
                   loop=replace(base.loop, optimizer_batch_size=12, max_optimizer_steps=3),
                   optimizer=replace(base.optimizer, learning_rate=1e-4, max_grad_norm=0.1),
                   scheduler=LinearWarmupConstantConfig(warmup_steps=2),
                   flux_packing=replace(base.flux_packing, max_tokens=9300))


class Sources(Dataset):
    def __init__(self, *args, source_count=11, **kwargs):
        self.fingerprint = "fixed-test-sources"
        self.keys = tuple(SimpleNamespace(task_id=f"task-{i}",
                          augmentation=SimpleNamespace(pair_indices=tuple(range(5)))) for i in range(source_count))
        self.visited = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.visited.append(index)
        source = _sample(4)
        return replace(source, input_ids=tuple(index * 11 + i for i in source.input_ids),
                       target_grids=tuple(((index % 10,),) for _ in range(4)))


def _snapshot(trainer, model, gradients):
    return dict(gradients=gradients, parameters=deepcopy(model.state_dict()),
                optimizer=deepcopy(trainer.optimizer.state_dict()),
                scheduler=deepcopy(trainer.scheduler.state_dict()))


def _plain_reference(initial, source_count=11, diagnostic_batch=None):
    """Use the ordinary Trainer without a process group, on each whole global batch."""
    config, sources, backend = _config(), Sources(source_count=source_count), SegmentedReference()
    model = _model(initial)
    optimizer = build_optimizer(config, flux_transformer=model)
    plans = plan_distributed_batches((3,) * source_count, optimizer_batch_size=12, rank=0, world_size=1)
    if diagnostic_batch is not None:
        plans = (plans[diagnostic_batch - 1],) * 3
    trainer = Trainer(
        nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(),
        flux_transformer=PackedFluxForward(model, compute_dtype=torch.float32, backend=backend),
        optimizer=optimizer, scheduler=build_lr_scheduler(config, optimizer=optimizer),
        flow_sampler=lambda _: (), loss_function=build_velocity_loss(config),
        flux_compute_dtype=torch.float32, flux_packing_backend=backend,
        min_context_pairs_before_target=2, flux_max_joint_tokens=9300, max_grad_norm=0.1,
    )
    snapshots, losses = [], []
    clip = torch.nn.utils.clip_grad_norm_
    for plan in plans:
        trainer.flow_sampler = DistributedFlowSampler(config.seed, plan.step, plan.global_target_count,
                                                      plan.target_indices, 1 if plan.global_target_count == 9 else 2)
        gradients = {}
        def capture(parameters, **kwargs):
            gradients.update(_gradients(model))
            return clip(parameters, **kwargs)
        with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                patch("torch.nn.utils.clip_grad_norm_", capture):
            result = trainer.step(tuple(sources[i] for i in plan.source_indices))
        snapshots.append(_snapshot(trainer, model, gradients))
        losses.append(result.loss)
    return snapshots, losses


def _execute(rank, world_size, initial, expected, losses, *, mismatch=False, max_steps=3, benchmark=False,
             diagnostic_batch=None, trace_directory=None, memory_history=False, profile=False,
             memory_snapshot_step=None):
    source_count = 12 if benchmark or memory_history or profile else 11
    backend, model, dataset = SegmentedReference(), _model(initial), Sources(source_count=source_count)
    config = _config()
    if mismatch and rank == 1:
        config = replace(config, optimizer=replace(config.optimizer, learning_rate=0.05))
    records, snapshots = [], []
    options = dict(expected_source_count=source_count, measurement=BenchmarkWindow(1, 2)) if benchmark else {}
    if diagnostic_batch is not None:
        options.update(diagnostic_batch=diagnostic_batch, trace_directory=trace_directory)
    if memory_history:
        options.update(memory_history=True, trace_directory=trace_directory,
                       expected_source_count=source_count)
        if memory_snapshot_step is not None:
            options["memory_snapshot_step"] = memory_snapshot_step
    if profile:
        options.update(profile_warmup_steps=2, trace_directory=trace_directory,
                       expected_source_count=source_count)
    durations = (100 + rank, 3 + rank, 5 + rank)
    clock = iter(value for start, duration in zip((0, 200, 400), durations) for value in (start, start + duration))
    models = SimpleNamespace(tokenizer=object(), flux_transformer=model, nvarc_encoder=FrozenText(),
                             vae_encoder=FrozenVae(), flux_compute_dtype=torch.float32,
                             flux_packing_backend=backend)

    lifecycle, captures = [], []

    @contextmanager
    def runtime(_):
        lifecycle.append("runtime_enter")
        try:
            yield DistributedRuntime(rank, rank, world_size, torch.device("cpu"), dist.group.WORLD)
        finally:
            lifecycle.append("runtime_exit")

    class CpuMemoryHistory:
        def __init__(self, directory, trace_rank, device, *, capture_step=None):
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.trace = BatchTrace(directory, trace_rank, device, echo=False)
            self.settings = {"max_entries": 100_000}
            self.capture_step = capture_step

        def observe(self, stage, details):
            self.trace(stage, details)
            if (stage == "flux_backward_end" and self.trace.context["pack"] == 1
                    and self.trace.context["global_batch"] == self.capture_step):
                captures.append(self.trace.context.copy())

        def __enter__(self):
            lifecycle.append("history_enter")
            return self

        def __exit__(self, error_type, error, traceback):
            assert error is None
            lifecycle.append("history_exit")

    def load_models(*args, **kwargs):
        lifecycle.append("load_models")
        return models

    class RecordingTrainer(Trainer):
        def step(self, sequences):
            clip, gradients = torch.nn.utils.clip_grad_norm_, {}
            def capture(parameters, **kwargs):
                gradients.update(_gradients(model))
                return clip(parameters, **kwargs)
            with patch("torch.nn.utils.clip_grad_norm_", capture):
                result = super().step(sequences)
            # Reset values without detaching gradients from the communication buckets.
            for parameter in model.parameters():
                assert parameter.grad is not None and parameter.grad._is_view()
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=0, rtol=0)
            snapshots.append(_snapshot(self, model, gradients))
            return result

    with patch("diffusion_arc.training.ddp_runner.distributed_runtime", runtime), \
            patch("diffusion_arc.training.ddp_runner.load_models", load_models), \
            patch("diffusion_arc.training.ddp_runner.CudaMemoryHistory", CpuMemoryHistory), \
            patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset), \
            patch("diffusion_arc.training.ddp_runner.Trainer", RecordingTrainer), \
            patch("diffusion_arc.training.run.batch_diagnostics.print", create=True), \
            patch("diffusion_arc.training.ddp_runner.time", SimpleNamespace(perf_counter=lambda: next(clock))), \
            patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
        if mismatch:
            try:
                run_training_ddp(config, max_steps=max_steps, emit=records.append)
            except ValueError as error:
                assert "different training settings" in str(error)
            else:
                raise AssertionError("mismatched effective settings were accepted")
            assert not dataset.visited and not snapshots and not records
            return
        progress = run_training_ddp(config, max_steps=max_steps, emit=records.append, **options)
    if memory_history:
        assert lifecycle == ["runtime_enter", "history_enter", "load_models", "history_exit", "runtime_exit"]
        if memory_snapshot_step is not None:
            assert len(captures) == 1 and captures[0]["global_batch"] == memory_snapshot_step
    expected_targets = source_count * 3 if max_steps == 3 and diagnostic_batch is None else 12 * max_steps
    assert (progress.optimizer_step, progress.sources_seen, progress.targets_seen) == (max_steps, expected_targets // 3, expected_targets)
    plans = plan_distributed_batches((3,) * source_count, optimizer_batch_size=12, rank=rank, world_size=world_size)[:max_steps]
    if diagnostic_batch is not None:
        plans = (plans[diagnostic_batch - 1],) * max_steps
    assert dataset.visited == [index for plan in plans for index in plan.source_indices]
    for actual, reference in zip(snapshots, expected[:max_steps], strict=True):
        torch.testing.assert_close(actual["gradients"], reference["gradients"], **GRAD_TOLERANCE)
        torch.testing.assert_close(actual["parameters"], reference["parameters"], **PARAM_TOLERANCE)
        torch.testing.assert_close(actual["optimizer"], reference["optimizer"], **GRAD_TOLERANCE)
        torch.testing.assert_close(actual["scheduler"], reference["scheduler"], atol=0, rtol=0)
    for parameter in model.parameters():
        peer = parameter.detach().clone()
        dist.broadcast(peer, src=0)
        torch.testing.assert_close(parameter, peer, atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for m in (models.nvarc_encoder, models.vae_encoder)
               for p in m.parameters())
    if rank:
        assert records == []
    else:
        assert [record["event"] for record in records] == ["start"] + ["step"] * max_steps + ["complete"]
        assert records[0]["mode"] == ("ddp_torch_profile" if profile else
                                      "ddp_memory_diagnostic" if memory_history else
                                      "ddp_batch_diagnostic" if diagnostic_batch else
                                      "ddp_benchmark" if benchmark else "ddp_training_only")
        if memory_history:
            assert records[0]["memory_history"] == {"max_entries": 100_000}
            assert records[0]["measurement"] is None
            assert "not a throughput benchmark" in records[0]["timing"]
            assert [row["phase"] for row in records[1:-1]] == ["diagnostic"] * max_steps
            assert [row["global_batch"] for row in records[1:-1]] == list(range(1, max_steps + 1))
        assert records[0]["world_size"] == world_size
        assert records[0]["backend"] == "gloo"
        for step, record in enumerate(records[1:-1]):
            torch.testing.assert_close(record["loss"], losses[step], atol=3e-6, rtol=3e-5)
            assert sum(r["endpoint_targets"] for r in record["ranks"]) == (
                1 if step == 2 and not (benchmark or memory_history or profile) and diagnostic_batch is None else 2)
            assert sum(r["targets"] for r in record["ranks"]) == record["targets"]
            assert record["max_rank_seconds"] == max(r["seconds"] for r in record["ranks"])
        if benchmark:
            assert [row["phase"] for row in records[1:-1]] == ["warmup", "measure", "measure"]
            summary = records[-1]["benchmark"]
            assert summary["measured_steps"] == 2
            assert summary["measured_seconds"] == 8 + 2 * (world_size - 1)
            assert summary["targets_per_second"] == 24 / summary["measured_seconds"]
            assert summary["median_step_seconds"] == 4 + world_size - 1
        else:
            assert records[-1]["benchmark"] == {"measured_steps": 0}
        json.dumps(records, allow_nan=False)
    if profile:
        with gzip.open(trace_directory / f"rank-{rank}.trace.json.gz", "rt") as handle:
            events = json.load(handle)["traceEvents"]
        names = {event["name"] for event in events}
        assert {name for name in names if name.startswith("optimizer_step/")} == {"optimizer_step/3"}
        assert {"prepare/nvarc", "prepare/vae", "train/flux_forward", "train/flux_backward",
                "train/adamw", "distributed/metrics", "data/retrieval"} <= names
        if rank == 0:
            assert [row["phase"] for row in records[1:-1]] == ["warmup", "warmup", "profile"]
    if diagnostic_batch is not None or memory_history:
        rows = [json.loads(line) for line in (trace_directory / f"rank-{rank}-trace.jsonl").read_text().splitlines()]
        assert [r["repetition"] for r in rows if r["stage"] == "batch_end"] == list(range(1, max_steps + 1))
        assert all(r["global_batch"] == (r["repetition"] if memory_history else diagnostic_batch) for r in rows)
        for repetition in range(1, len(snapshots) + 1):
            trace = [r for r in rows if r["repetition"] == repetition]
            forward = [r for r in trace if r["stage"] == "flux_forward_start"]
            backward = [r for r in trace if r["stage"] == "flux_backward_end"]
            assert len(forward) == len(backward) > 0
            assert [r["sync"] for r in forward] == [False] * (len(forward) - 1) + [True]
            assert all(r["rank"] == rank for r in trace)


def _worker(rank, directory, initial, expected, losses, benchmark_expected, benchmark_losses,
            diagnostic_expected, diagnostic_losses):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "store-two").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _execute(rank, 2, initial, expected, losses)
        _execute(rank, 2, initial, expected, losses, mismatch=True)
        _execute(rank, 2, initial, benchmark_expected, benchmark_losses, benchmark=True)
        _execute(rank, 2, initial, diagnostic_expected, diagnostic_losses, diagnostic_batch=2,
                 trace_directory=Path(directory))
        _execute(rank, 2, initial, benchmark_expected, benchmark_losses, memory_history=True,
                 trace_directory=Path(directory) / "memory")
        _execute(rank, 2, initial, benchmark_expected, benchmark_losses, profile=True,
                 trace_directory=Path(directory) / "profile")
        _execute(rank, 2, initial, benchmark_expected, benchmark_losses, memory_history=True,
                 memory_snapshot_step=2, max_steps=2, trace_directory=Path(directory) / "selected-memory")
    finally:
        dist.destroy_process_group()


class DdpRunnerTests(unittest.TestCase):
    def test_plain_then_one_and_two_processes_agree_including_uneven_final_batch(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(167)
            initial = deepcopy(_model().state_dict())
            expected, losses = _plain_reference(initial)
            benchmark_expected, benchmark_losses = _plain_reference(initial, source_count=12)
            diagnostic_expected, diagnostic_losses = _plain_reference(initial, diagnostic_batch=2)
            with tempfile.TemporaryDirectory(prefix="ddp-runner-") as directory:
                dist.init_process_group("gloo", init_method=(Path(directory) / "store-one").as_uri(),
                                        rank=0, world_size=1, timeout=timedelta(seconds=30))
                try:
                    _execute(0, 1, initial, expected, losses)
                    _execute(0, 1, initial, expected, losses, max_steps=1)
                    _execute(0, 1, initial, benchmark_expected, benchmark_losses, benchmark=True)
                    with tempfile.TemporaryDirectory() as trace_directory:
                        _execute(0, 1, initial, diagnostic_expected, diagnostic_losses,
                                 diagnostic_batch=2, trace_directory=Path(trace_directory))
                    with tempfile.TemporaryDirectory() as trace_directory:
                        _execute(0, 1, initial, benchmark_expected, benchmark_losses,
                                 memory_history=True, trace_directory=Path(trace_directory))
                    with tempfile.TemporaryDirectory() as trace_directory:
                        _execute(0, 1, initial, benchmark_expected, benchmark_losses,
                                 profile=True, trace_directory=Path(trace_directory))
                    with tempfile.TemporaryDirectory() as trace_directory:
                        _execute(0, 1, initial, benchmark_expected, benchmark_losses, memory_history=True,
                                 memory_snapshot_step=2, max_steps=2, trace_directory=Path(trace_directory))
                finally:
                    dist.destroy_process_group()
                processes = mp.spawn(_worker, args=(directory, initial, expected, losses,
                                                    benchmark_expected, benchmark_losses,
                                                    diagnostic_expected, diagnostic_losses), nprocs=2, join=False)
                try:
                    deadline = time.monotonic() + 90
                    while not processes.join(timeout=1):
                        if time.monotonic() > deadline:
                            raise TimeoutError("DDP runner test exceeded 90 seconds")
                finally:
                    for process in processes.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in processes.processes:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)

    def test_memory_history_sees_loading_oom_before_distributed_cleanup(self):
        events, seen_errors = [], []
        error = torch.OutOfMemoryError("simulated model allocation failure")

        @contextmanager
        def runtime(_):
            events.append("runtime_enter")
            try:
                yield DistributedRuntime(0, 0, 1, torch.device("cpu"), None)
            finally:
                events.append("runtime_exit")

        @contextmanager
        def history(directory, rank, device, *, capture_step=None):
            events.append("history_enter")
            try:
                yield SimpleNamespace(trace=BatchTrace(directory, rank, device, echo=False))
            except torch.OutOfMemoryError as failure:
                events.append("snapshot")
                seen_errors.append(failure)
                raise
            finally:
                events.append("history_exit")

        def fail_loading(*args, **kwargs):
            events.append("load_models")
            raise error

        with tempfile.TemporaryDirectory() as directory, \
                patch("diffusion_arc.training.ddp_runner.distributed_runtime", runtime), \
                patch("diffusion_arc.training.ddp_runner.CudaMemoryHistory", history), \
                patch("diffusion_arc.training.ddp_runner.load_models", fail_loading), \
                self.assertRaises(torch.OutOfMemoryError) as raised:
            run_training_ddp(_config(), max_steps=1, memory_history=True, trace_directory=Path(directory))
        self.assertIs(raised.exception, error)
        self.assertEqual(seen_errors, [error])
        self.assertEqual(events, ["runtime_enter", "history_enter", "load_models", "snapshot",
                                  "history_exit", "runtime_exit"])

    def test_rejects_resume_and_bad_step_limits_before_distributed_setup(self):
        config = _config()
        recipe = deepcopy(config.recipe)
        recipe["models"]["resume_from"] = "checkpoint"
        cases = [(replace(config, recipe=recipe), 1), (replace(config, config_type="baseline"), 1)]
        cases += [(config, steps) for steps in (0, True, 4)]
        with patch("diffusion_arc.training.ddp_runner.distributed_runtime") as runtime:
            for config, steps in cases:
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    run_training_ddp(config, max_steps=steps)
            runtime.assert_not_called()
            with self.assertRaisesRegex(ValueError, "warmup plus measured"):
                run_training_ddp(_config(), max_steps=1, measurement=BenchmarkWindow(1, 2))


if __name__ == "__main__":
    unittest.main()
