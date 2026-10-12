"""Verify lazy spawned packing agrees with prepared plans and training updates."""

from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from diffusion_arc.training.benchmark_runner import run_training_benchmark
from diffusion_arc.training.config import parse_training_config
from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.distributed_data import (
    BackgroundBatchPlans,
    build_distributed_dataloader,
    plan_distributed_batches,
    precompute_distributed_packing,
)
from diffusion_arc.training.source_lengths import write_source_token_lengths
from diffusion_arc.training.run.benchmarking import BenchmarkWindow, RunArtifacts
from diffusion_arc.training.run.experiment import BenchmarkInstrumentation
from diffusion_arc.training.runner import run_training
from diffusion_arc.training.setup import TrainingModels
from diffusion_arc.training.step.indexed_flow import IndexedFlowSampler
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources, _config
from tests.training.test_ddp_trainer import (
    FrozenText, FrozenVae, _model, _render, GRAD_TOLERANCE, LOSS_TOLERANCE, PARAM_TOLERANCE,
)
from tests.training.test_distributed_data import RecordingDataset
from tests.training.test_precomputed_packing import _record_draws, _snapshot, _SnapshotTrainer, _trainer


def _prefixes(source_count):
    lengths = ((300, 800, 2100), (1700, 2400, 5000),
               (450, 1100, 2900), (1000, 2000, 4300),
               (1200, 2000, 3800), (200, 500, 1700),
               (300, 1100, 3500), (1800, 3000, 6000))
    return tuple(lengths[index % len(lengths)] for index in range(source_count))


def _plans(prefixes, *, world_size, sources_per_batch):
    return tuple(plan_distributed_batches(
        tuple(len(source) for source in prefixes),
        optimizer_batch_size=3 * sources_per_batch,
        rank=rank, world_size=world_size,
        source_joint_tokens=tuple(sum(prefix + 900 for prefix in source) for source in prefixes),
    ) for rank in range(world_size))


def _joint_config(*, source_count, num_workers=0, prefetch_factor=None):
    base = _config(source_count=source_count, world_size=1)
    recipe = base.recipe_dict()
    recipe["execution"].update(rank_batch_assignment="joint_packing_v1",
                                flux_packing_algorithm="legacy_bfd_v1")
    recipe["data"].update(num_workers=num_workers, prefetch_factor=prefetch_factor)
    return parse_training_config(recipe, config_path=base.path)


def _prepare_dataset(dataset, directory, prefixes):
    dataset.split_directory = Path(directory)
    dataset.view_name, dataset.fingerprint, dataset.max_tokens = "train", "fixed-test-sources", 8400
    write_source_token_lengths(dataset, prefixes, min_context_pairs_before_target=2, provenance={})
    return dataset


class _ObservedExecutor(ProcessPoolExecutor):
    """Observe the real executor without replacing its spawned CPU work."""

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = []
        self.spawned_processes = ()
        self.maximum_pending = 0
        type(self).instances.append(self)

    def submit(self, *args, **kwargs):
        future = super().submit(*args, **kwargs)
        self.submitted.append(future)
        self.spawned_processes = tuple(self._processes.values())
        self.maximum_pending = max(self.maximum_pending, sum(not item.done() for item in self.submitted))
        return future


class BackgroundPackingTests(unittest.TestCase):
    def test_spawned_plans_match_precompute_for_one_two_four_and_eight_ranks(self):
        prefixes, keys = _prefixes(16), tuple(f"source-{index}" for index in range(16))
        for world_size in (1, 2, 4, 8):
            with self.subTest(world_size=world_size):
                original = _plans(prefixes, world_size=world_size, sources_per_batch=8)
                expected, _ = precompute_distributed_packing(
                    original, source_keys=keys, source_prefix_tokens=prefixes, max_tokens=9300,
                )
                actual = []
                for rank in range(world_size):
                    background = BackgroundBatchPlans(original, rank=rank, source_keys=keys,
                        source_prefix_tokens=prefixes, max_tokens=9300)
                    try:
                        self.assertEqual(background.original_plans, original[rank])
                        self.assertEqual(len(background), 2)
                        selected = tuple(background[index] for index in range(len(background)))
                        self.assertEqual(selected, expected[rank])
                        actual.append(selected)
                    finally:
                        background.close()
                for batch in range(2):
                    sources = range(batch * 8, (batch + 1) * 8)
                    self.assertEqual(sorted(source for plans in actual
                                            for source in plans[batch].source_indices), list(sources))
                    self.assertEqual(sorted(target for plans in actual
                                            for target in plans[batch].target_indices), list(range(24)))
                    calls = max(len(plans[batch].flux_plan.microbatch_plans) for plans in actual)
                    for plans in actual:
                        plan = plans[batch]
                        self.assertTrue(plan.source_indices)
                        target_ids = tuple(target for source in plan.source_indices
                                           for target in range(3 * (source % 8), 3 * (source % 8 + 1)))
                        self.assertEqual(plan.target_indices, target_ids)
                        self.assertEqual(plan.flux_plan.collective_call_count, calls)
                        local_prefixes = tuple(prefix for source in plan.source_indices for prefix in prefixes[source])
                        self.assertEqual(plan.flux_plan.target_prefix_tokens, local_prefixes)
                        self.assertEqual(sorted(target for pack in plan.flux_plan.microbatch_plans
                                                for target in pack.target_indices), list(range(len(target_ids))))
                        for pack in plan.flux_plan.microbatch_plans:
                            self.assertEqual(pack.joint_tokens,
                                             sum(local_prefixes[target] + 900 for target in pack.target_indices))
                            self.assertLessEqual(pack.joint_tokens, 9300)

    def test_start_is_lazy_and_truncation_bounds_spawned_work(self):
        prefixes, keys = _prefixes(32), tuple(f"source-{index}" for index in range(32))
        original = _plans(prefixes, world_size=2, sources_per_batch=4)
        _ObservedExecutor.instances = []
        with patch("diffusion_arc.training.distributed_data.ProcessPoolExecutor", _ObservedExecutor):
            background = BackgroundBatchPlans(original, rank=0, source_keys=keys,
                source_prefix_tokens=prefixes, max_tokens=9300, prefetch_batches=4)
            processes = ()
            try:
                self.assertEqual(_ObservedExecutor.instances, [])
                background.truncate(3)
                self.assertEqual(len(background), 3)
                self.assertEqual(background.original_plans, original[0][:3])
                self.assertEqual(_ObservedExecutor.instances, [])
                first = background[0]
                self.assertIs(background[0], first)
                self.assertEqual(len(_ObservedExecutor.instances), 1)
                executor = _ObservedExecutor.instances[0]
                self.assertEqual(executor._mp_context.get_start_method(), "spawn")
                self.assertEqual(executor._max_workers, 1)
                self.assertLessEqual(executor.maximum_pending, 4)
                self.assertEqual(len(executor.submitted), 3)
                with self.assertRaisesRegex(RuntimeError, "truncate.*before"):
                    background.truncate(1)
                processes = tuple(executor._processes.values())
            finally:
                background.close()
            self.assertTrue(processes)
            self.assertTrue(all(not process.is_alive() for process in processes))
            background.close()

    def test_replayed_batches_use_one_spawned_job_and_release_after_the_final_occurrence(self):
        prefixes, keys = _prefixes(8), tuple(f"source-{index}" for index in range(8))
        original = _plans(prefixes, world_size=2, sources_per_batch=4)
        repeated = tuple((plans[1],) * 6 for plans in original)
        _ObservedExecutor.instances = []
        with patch("diffusion_arc.training.distributed_data.ProcessPoolExecutor", _ObservedExecutor):
            background = BackgroundBatchPlans(repeated, rank=0, source_keys=keys,
                source_prefix_tokens=prefixes, max_tokens=9300)
            try:
                first = background[0]
                for index in range(len(background)):
                    self.assertIs(background[index], first)
                    background.release(index)
                    self.assertEqual(first.epoch_batch_index in background._ready, index < len(background) - 1)
                self.assertEqual(len(_ObservedExecutor.instances), 1)
                self.assertEqual(len(_ObservedExecutor.instances[0].submitted), 1)
            finally:
                background.close()

    def test_worker_failure_closes_executor_and_clears_outstanding_work(self):
        prefixes = ((9000, 10000, 11000),) + _prefixes(15)
        keys = tuple(f"source-{index}" for index in range(16))
        original = _plans(prefixes, world_size=2, sources_per_batch=4)
        _ObservedExecutor.instances = []
        with patch("diffusion_arc.training.distributed_data.ProcessPoolExecutor", _ObservedExecutor):
            background = BackgroundBatchPlans(original, rank=0, source_keys=keys,
                source_prefix_tokens=prefixes, max_tokens=9300)
            try:
                with self.assertRaisesRegex(ValueError, "targets.*within capacity"):
                    background[0]
                self.assertTrue(background._closed)
                self.assertEqual(background._pending, {})
                self.assertEqual(background._ready, {})
                self.assertEqual(len(_ObservedExecutor.instances), 1)
                executor = _ObservedExecutor.instances[0]
                self.assertTrue(executor.spawned_processes)
                self.assertTrue(all(not process.is_alive() for process in executor.spawned_processes))
                self.assertTrue(all(future.done() for future in executor.submitted))
            finally:
                background.close()

    def test_worker_prefetch_preserves_plan_sample_alignment_and_resume(self):
        prefixes = _prefixes(40)
        config = _joint_config(source_count=40, num_workers=2, prefetch_factor=8)
        expected, _ = precompute_distributed_packing(
            _plans(prefixes, world_size=2, sources_per_batch=4),
            source_keys=RecordingDataset(source_count=40).keys,
            source_prefix_tokens=prefixes, max_tokens=9300,
        )
        _ObservedExecutor.instances = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("diffusion_arc.training.distributed_data.ProcessPoolExecutor", _ObservedExecutor):
            dataset = _prepare_dataset(RecordingDataset(source_count=40), directory, prefixes)
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                loader, background = build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2)
            self.assertEqual(dataset.visited, [])
            self.assertIsInstance(background, BackgroundBatchPlans)
            self.assertEqual(_ObservedExecutor.instances, [])
            metadata = deepcopy(dataset.rank_assignment_metadata)
            self.assertIn("background_packing", metadata)
            pickle.dumps(loader.dataset)
            self.assertFalse(any(isinstance(value, (Future, ProcessPoolExecutor, BackgroundBatchPlans))
                                 for value in vars(loader.dataset).values()))
            iterator = None
            received = []
            try:
                iterator = iter(loader)
                self.assertGreater(background._next_requested, 4)
                for index in range(len(background)):
                    plan = background[index]
                    samples = tuple(islice(iterator, len(plan.source_indices)))
                    self.assertEqual(samples, tuple(f"sample-{source}" for source in plan.source_indices))
                    self.assertEqual(plan, expected[0][index])
                    received.append(plan)
                    self.assertLessEqual(len(background._pending), 4)
                    self.assertLessEqual(len(background._ready), 2 * config.data.prefetch_factor + 1)
                    background.release(index)
                    self.assertNotIn(plan.epoch_batch_index, background._ready)
                self.assertEqual(tuple(iterator), ())
                self.assertEqual(dataset.rank_assignment_metadata, metadata)
                self.assertEqual(len(_ObservedExecutor.instances), 1)
                self.assertLessEqual(_ObservedExecutor.instances[0].maximum_pending, 4)
            finally:
                if iterator is not None:
                    iterator._shutdown_workers()
                background.close()
            resumed_dataset = _prepare_dataset(RecordingDataset(source_count=40), directory, prefixes)
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=resumed_dataset):
                resumed_loader, resumed = build_distributed_dataloader(
                    config, tokenizer=object(), rank=0, world_size=2, start_step=3,
                )
            iterator = None
            try:
                iterator = iter(resumed_loader)
                suffix = []
                for index in range(len(resumed)):
                    plan = resumed[index]
                    self.assertEqual(tuple(islice(iterator, len(plan.source_indices))),
                                     tuple(f"sample-{source}" for source in plan.source_indices))
                    suffix.append(plan)
                    resumed.release(index)
                self.assertEqual(tuple(suffix), tuple(received[3:]))
                self.assertEqual(resumed_dataset.rank_assignment_metadata, metadata)
            finally:
                if iterator is not None:
                    iterator._shutdown_workers()
                resumed.close()

    def test_background_plans_preserve_three_real_adamw_updates(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        initial = deepcopy(_model().state_dict())
        config = _joint_config(source_count=12)
        dataset = Sources(source_count=12)
        prefixes = tuple(dataset[index].prefix_token_ends[1:] for index in range(len(dataset)))
        dataset.visited.clear()
        with tempfile.TemporaryDirectory() as directory:
            _prepare_dataset(dataset, directory, prefixes)
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                loader, background = build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=1)
            snapshots = []
            try:
                for planned in (False, True):
                    model, trainer = _trainer(initial)
                    iterator = iter(loader) if planned else None
                    updates = []
                    for index in range(3):
                        plan = background[index] if planned else background.original_plans[index]
                        sequences = (tuple(islice(iterator, len(plan.source_indices))) if planned else
                                     tuple(dataset[source] for source in plan.source_indices))
                        draws = []
                        trainer.flow_sampler = IndexedFlowSampler(config.seed, plan.epoch_batch_index,
                            plan.global_target_count, plan.target_indices, 2, observe_draw=_record_draws(draws))
                        with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                                (patch("diffusion_arc.training.trainer.plan_flux_batches",
                                       side_effect=AssertionError("runtime packing used")) if planned else nullcontext()):
                            result = trainer.step(sequences, batch_plan=plan) if planned else trainer.step(sequences)
                        updates.append(_snapshot(model, trainer, result, draws, []))
                        if planned:
                            background.release(index)
                    snapshots.append(updates)
            finally:
                background.close()
        for actual, expected in zip(snapshots[1], snapshots[0], strict=True):
            for field, tolerance in (
                ("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
                ("optimizer", GRAD_TOLERANCE), ("loss", LOSS_TOLERANCE),
                ("gradient_norm", GRAD_TOLERANCE), ("target_losses", LOSS_TOLERANCE),
            ):
                torch.testing.assert_close(actual[field], expected[field], **tolerance)
            torch.testing.assert_close(actual["draws"], expected["draws"], atol=0, rtol=0)

    def test_benchmark_and_training_execute_dynamic_plans_and_record_used_packing(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        initial = deepcopy(_model().state_dict())
        benchmark_config = _joint_config(source_count=12)
        recipe = benchmark_config.recipe_dict()
        recipe["execution"]["distributed_strategy"] = "single_process"
        training_config = parse_training_config(recipe, config_path=benchmark_config.path)
        release_plan, close_plans = BackgroundBatchPlans.release, BackgroundBatchPlans.close
        begin_step = BenchmarkInstrumentation.before_step
        snapshots, received, events, records, closed = {}, {}, {}, [], []
        _ObservedExecutor.instances = []

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            with RunArtifacts(output) as artifacts:
                for mode, config in (("benchmark", benchmark_config), ("training", training_config)):
                    dataset = _prepare_dataset(Sources(source_count=12), directory, ((4, 6, 8),) * 12)
                    models = TrainingModels(
                        tokenizer=object(), flux_transformer=_model(initial),
                        nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(), vae_decoder=None,
                        flux_compute_dtype=torch.float32, flux_packing_backend=SegmentedReference(),
                    )
                    snapshots[mode], received[mode], events[mode] = [], [], []
                    runtime = TrainingRuntime(0, 0, 1, torch.device("cpu"), None,
                                              config.recipe["execution"]["distributed_strategy"])

                    class ObserveTrainer(_SnapshotTrainer):
                        def step(trainer, sequences, *, batch_plan=None):
                            self.assertIsNotNone(batch_plan)
                            self.assertIsNotNone(batch_plan.flux_plan)
                            self.assertEqual(tuple(sequence.input_ids[0] // 11 for sequence in sequences),
                                             batch_plan.source_indices)
                            events[mode].append("trainer_step")
                            received[mode].append(batch_plan)
                            result = super().step(sequences, batch_plan=batch_plan)
                            snapshot = _snapshot(models.flux_transformer, trainer, result, (), [])
                            snapshot["scheduler"] = deepcopy(trainer.scheduler.state_dict())
                            snapshots[mode].append(snapshot)
                            return result

                    def release(plans, index):
                        self.assertEqual(len(snapshots[mode]), index + 1)
                        release_plan(plans, index)
                        self.assertNotIn(plans.original_plans[index].epoch_batch_index, plans._ready)
                        events[mode].append("release")

                    def close(plans):
                        close_plans(plans)
                        events[mode].append("close")
                        self.assertTrue(plans._closed)
                        self.assertEqual(plans._pending, {})
                        self.assertEqual(plans._ready, {})
                        closed.append(plans)

                    def before_step(experiment):
                        events[mode].append("before_step")
                        return begin_step(experiment)

                    def emit(record):
                        records.append(record)
                        artifacts(record)

                    with ExitStack() as patches:
                        for target, options in (
                            ("diffusion_arc.training.training_loop.load_models", dict(return_value=models)),
                            ("diffusion_arc.training.training_loop.distributed_runtime", dict(return_value=nullcontext(runtime))),
                            ("diffusion_arc.training.training_loop.DDP", dict(side_effect=lambda model, **_: model)),
                            ("diffusion_arc.training.training_loop.Trainer", dict(new=ObserveTrainer)),
                            ("diffusion_arc.training.distributed_data.ArcTrainingDataset", dict(return_value=dataset)),
                            ("diffusion_arc.training.distributed_data.ProcessPoolExecutor", dict(new=_ObservedExecutor)),
                            ("diffusion_arc.training.distributed_data.precompute_distributed_packing",
                             dict(side_effect=AssertionError("eager precomputed packing used"))),
                            ("diffusion_arc.training.trainer.plan_flux_batches",
                             dict(side_effect=AssertionError("runtime packing used"))),
                            ("diffusion_arc.training.step.encoding._render_valid_grid_canvas", dict(new=_render)),
                            ("diffusion_arc.training.run.benchmarking.print", dict(create=True)),
                        ):
                            patches.enter_context(patch(target, **options))
                        patches.enter_context(patch.object(BackgroundBatchPlans, "release", release))
                        patches.enter_context(patch.object(BackgroundBatchPlans, "close", close))
                        patches.enter_context(patch.object(BenchmarkInstrumentation, "before_step", before_step))
                        if mode == "benchmark":
                            benchmark_progress = run_training_benchmark(
                                config, max_steps=3, emit=emit, expected_source_count=12,
                                measurement=BenchmarkWindow(1, 2), precompute_packing=False,
                            )
                        else:
                            training_progress = run_training(config, runtime=runtime)
                    self.assertEqual(dataset.visited,
                                     [source for plan in received[mode] for source in plan.source_indices])
            manifest = json.loads((output / "manifest.json").read_text())
            steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse((output / "packing_plans.json").exists())

        self.assertEqual((benchmark_progress.optimizer_step, benchmark_progress.sources_seen,
                          benchmark_progress.targets_seen), (3, 12, 36))
        self.assertEqual(training_progress, benchmark_progress)
        self.assertEqual(events["benchmark"], ["before_step", "trainer_step", "release"] * 3 + ["close"])
        self.assertEqual(events["training"], ["trainer_step", "release"] * 3 + ["close"])
        self.assertEqual(len(closed), 2)
        self.assertEqual(len(_ObservedExecutor.instances), 2)
        self.assertTrue(all(executor.spawned_processes for executor in _ObservedExecutor.instances))
        self.assertTrue(all(not process.is_alive() for executor in _ObservedExecutor.instances
                            for process in executor.spawned_processes))
        self.assertEqual([record["event"] for record in records], ["start", "step", "step", "step", "complete"])
        self.assertFalse(manifest["precompute_packing"])
        self.assertEqual(manifest["rank_batch_assignment"]["algorithm"], "joint_packing_v1")
        self.assertEqual(manifest["rank_batch_assignment"]["background_packing"], closed[0].metadata)
        self.assertIn("background plan waits during retrieval included", manifest["timing"])
        self.assertIn("first plan and initial DataLoader prefetch excluded", manifest["timing"])
        self.assertEqual(summary["benchmark"]["measured_steps"], 2)
        self.assertEqual(received["training"], received["benchmark"])
        for plan, step in zip(received["benchmark"], steps, strict=True):
            saved = step["ranks"][0]
            self.assertEqual(saved["source_indices"], list(plan.source_indices))
            self.assertEqual(saved["target_indices"], list(plan.target_indices))
            self.assertEqual(saved["flux_plan"], json.loads(json.dumps(asdict(plan.flux_plan))))
        for actual, expected in zip(snapshots["training"], snapshots["benchmark"], strict=True):
            for field, tolerance in (
                ("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
                ("optimizer", GRAD_TOLERANCE), ("loss", LOSS_TOLERANCE),
                ("gradient_norm", GRAD_TOLERANCE), ("target_losses", LOSS_TOLERANCE),
            ):
                torch.testing.assert_close(actual[field], expected[field], **tolerance)
            self.assertEqual(actual["scheduler"], expected["scheduler"])


if __name__ == "__main__":
    unittest.main()
