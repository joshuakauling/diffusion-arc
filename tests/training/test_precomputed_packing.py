"""Verify CPU batch plans preserve draws, losses and distributed optimizer updates."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.benchmark_runner import run_training_benchmark
from diffusion_arc.training.config import parse_training_config
from diffusion_arc.training.distributed_data import (
    BackgroundBatchPlans, RankBatchPlan, plan_distributed_batches, precompute_distributed_packing,
)
from diffusion_arc.training.distribution_strategy.fsdp import FsdpStrategy, shard_flux
from diffusion_arc.training.distribution_strategy.single import SingleDeviceStrategy
from diffusion_arc.training.step.flux_batching import FluxMicrobatchPlan, PrecomputedFluxPlan
from diffusion_arc.training.step.indexed_flow import IndexedFlowSampler
from diffusion_arc.training.step.loss import velocity_mse
from diffusion_arc.training.run.benchmarking import BenchmarkWindow, RunArtifacts
from diffusion_arc.training.run.experiment import BenchmarkInstrumentation
from diffusion_arc.training.setup import TrainingModels
from diffusion_arc.training.trainer import Trainer
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import Sources, _config
from tests.training.test_ddp_trainer import (
    FrozenText, FrozenVae, _model, _render,
    GRAD_TOLERANCE, LOSS_TOLERANCE, PARAM_TOLERANCE,
)
from tests.training.test_numerical_runner import UnevenSources, _uneven_packs


def _full_tensor(value):
    return (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().clone()


class _SnapshotTrainer(Trainer):
    def _clip_gradients(self):
        self.pre_clip_gradients = {
            name: _full_tensor(parameter.grad)
            for name, parameter in self.flux_transformer.named_parameters()
        }
        return super()._clip_gradients()


def _trainer(initial, *, runtime=None):
    model, backend = _model(initial), SegmentedReference()
    packed = PackedFluxForward(model, compute_dtype=torch.float32, backend=backend)
    if runtime is not None:
        packed = shard_flux(packed, runtime=runtime)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trainer = _SnapshotTrainer(
        nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(),
        flux_transformer=packed, optimizer=optimizer, scheduler=None,
        flow_sampler=lambda _: (), loss_function=velocity_mse,
        flux_compute_dtype=torch.float32, flux_packing_backend=backend,
        min_context_pairs_before_target=2, flux_max_joint_tokens=9300,
        max_grad_norm=0.1,
    )
    return model, trainer


def _snapshot(model, trainer, result, draws, collectives):
    return dict(
        gradients=trainer.pre_clip_gradients,
        parameters={name: _full_tensor(parameter) for name, parameter in model.named_parameters()},
        optimizer={name: {key: _full_tensor(value) for key, value in trainer.optimizer.state[parameter].items()}
                   for name, parameter in model.named_parameters()},
        loss=result.loss, gradient_norm=result.gradient_norm,
        target_losses=tuple(target.loss for target in result.target_metrics),
        draws=draws, collectives=collectives,
        flux_forward_calls=result.flux_forward_calls,
        padding_forward_calls=result.padding_forward_calls,
        packs=tuple(pack.target_indices for pack in result.diagnostics.flux_microbatch_plans),
    )


def _record_draws(draws):
    def observe(target, timestep, noise):
        draws.append(dict(target=target, timestep=timestep.detach().clone(), noise=noise.detach().clone()))
    return observe


def _precomputed(plan, prefixes, packs, *, calls):
    return replace(plan, flux_plan=PrecomputedFluxPlan(
        microbatch_plans=packs, target_prefix_tokens=prefixes,
        collective_call_count=calls,
    ))


def _single_updates(initial, *, precomputed):
    model, trainer = _trainer(initial)
    dataset = Sources(source_count=4)
    sequences = tuple(dataset[index] for index in range(4))
    prefixes = tuple(prefix for sequence in sequences for prefix in sequence.prefix_token_ends[1:])
    plan = RankBatchPlan(0, (0, 1, 2, 3), tuple(range(12)), 12)
    # Rearrange targets across sources and use a different number of calls.
    groups = ((11, 0, 5), (1, 10, 4), (9, 2, 7), (8, 3, 6))
    packs = tuple(FluxMicrobatchPlan(group, sum(prefixes[i] + 900 for i in group)) for group in groups)
    plan = _precomputed(plan, prefixes, packs, calls=6)
    snapshots = []
    for step in range(3):
        draws = []
        trainer.flow_sampler = IndexedFlowSampler(1729, step, 12, plan.target_indices, 2,
                                                 observe_draw=_record_draws(draws))
        with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                patch("diffusion_arc.training.trainer.plan_flux_batches",
                      side_effect=AssertionError("runtime packing used") if precomputed else _uneven_packs), \
                (patch.object(trainer.distribution_strategy, "all_reduce_target_count",
                              side_effect=AssertionError("target count collective used"))
                 if precomputed else nullcontext()), \
                patch.object(dist, "all_reduce", side_effect=AssertionError("single-device collective used")):
            result = trainer.step(sequences, batch_plan=plan) if precomputed else trainer.step(sequences)
        snapshots.append(_snapshot(model, trainer, result, draws, []))
    return snapshots


def _fsdp_updates(rank, initial, *, precomputed, synchronize_each_microbatch=False):
    runtime = TrainingRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD, "fsdp2")
    model, trainer = _trainer(initial, runtime=runtime)
    trainer.distribution_strategy.synchronize_each_microbatch = synchronize_each_microbatch
    dataset = UnevenSources(source_count=4)
    plan = plan_distributed_batches((3,) * 4, optimizer_batch_size=12, rank=rank, world_size=2)[0]
    sequences = tuple(dataset[index] for index in plan.source_indices)
    prefixes = tuple(prefix for sequence in sequences for prefix in sequence.prefix_token_ends[1:])
    packs = tuple(sorted(_uneven_packs(prefixes), key=lambda pack: -pack.joint_tokens))
    plan = _precomputed(plan, prefixes, packs, calls=6)
    snapshots = []
    for step in range(3):
        draws, collectives = [], []
        trainer.flow_sampler = IndexedFlowSampler(1729, step, 12, plan.target_indices, 2,
                                                 observe_draw=_record_draws(draws))
        all_reduce = dist.all_reduce

        def observe_collective(value, *args, **kwargs):
            collectives.append(dict(dtype=str(value.dtype), size=value.numel()))
            if precomputed and value.dtype == torch.int64:
                raise AssertionError("runtime count collective used")
            return all_reduce(value, *args, **kwargs)

        with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                patch("diffusion_arc.training.trainer.plan_flux_batches",
                      side_effect=AssertionError("runtime packing used") if precomputed else _uneven_packs), \
                (patch.object(trainer.distribution_strategy, "all_reduce_target_count",
                              side_effect=AssertionError("target count collective used"))
                 if precomputed else nullcontext()), \
                patch.object(dist, "all_reduce", observe_collective):
            result = trainer.step(sequences, batch_plan=plan) if precomputed else trainer.step(sequences)
        snapshot = _snapshot(model, trainer, result, draws, collectives)
        snapshot["global_target_indices"] = plan.target_indices
        snapshots.append(snapshot)
    return snapshots


def _background_fsdp_updates(rank, initial):
    runtime = TrainingRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD, "fsdp2")
    model, trainer = _trainer(initial, runtime=runtime)
    dataset = UnevenSources(source_count=4)
    originals = tuple(tuple(replace(plan_distributed_batches((3,) * 4,
        optimizer_batch_size=12, rank=owner, world_size=2)[0], epoch_batch_index=step)
        for step in range(3)) for owner in range(2))
    prefixes = tuple(dataset[index].prefix_token_ends[1:] for index in range(4))
    plans = BackgroundBatchPlans(originals, rank=rank,
        source_keys=tuple(f"source-{index}" for index in range(4)),
        source_prefix_tokens=prefixes, max_tokens=9300)
    snapshots = []
    try:
        for step, plan in enumerate(plans):
            sequences = tuple(dataset[index] for index in plan.source_indices)
            draws, collectives = [], []
            trainer.flow_sampler = IndexedFlowSampler(1729, step, 12, plan.target_indices, 2,
                                                     observe_draw=_record_draws(draws))
            all_reduce = dist.all_reduce

            def observe_collective(value, *args, **kwargs):
                collectives.append(dict(dtype=str(value.dtype), size=value.numel()))
                if value.dtype == torch.int64:
                    raise AssertionError("runtime count collective used")
                return all_reduce(value, *args, **kwargs)

            with patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                    patch("diffusion_arc.training.trainer.plan_flux_batches",
                          side_effect=AssertionError("runtime packing used")), \
                    patch.object(dist, "all_reduce", observe_collective):
                result = trainer.step(sequences, batch_plan=plan)
            snapshot = _snapshot(model, trainer, result, draws, collectives)
            snapshot["global_target_indices"] = plan.target_indices
            snapshots.append(snapshot)
            plans.release(step)
    finally:
        plans.close()
    return snapshots


def _worker(rank, directory, initial):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "gloo-store").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for mode in ("runtime", "precomputed", "precomputed_per_microbatch"):
            snapshots = _fsdp_updates(rank, initial, precomputed=mode != "runtime",
                                     synchronize_each_microbatch=mode == "precomputed_per_microbatch")
            torch.save(snapshots, Path(directory) / f"rank-{rank}-{mode}.pt")
        torch.save(_background_fsdp_updates(rank, initial), Path(directory) / f"rank-{rank}-background.pt")
    finally:
        dist.destroy_process_group()


class PrecomputedPackingTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        self.initial = deepcopy(_model().state_dict())

    def assert_same_update(self, actual, expected):
        for field, tolerance in (
            ("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
            ("optimizer", GRAD_TOLERANCE), ("loss", LOSS_TOLERANCE),
            ("gradient_norm", GRAD_TOLERANCE), ("target_losses", LOSS_TOLERANCE),
        ):
            torch.testing.assert_close(actual[field], expected[field], **tolerance)
        torch.testing.assert_close(actual["draws"], expected["draws"], atol=0, rtol=0)

    def test_reordered_precomputed_packs_preserve_three_real_adamw_updates(self):
        reference = _single_updates(self.initial, precomputed=False)
        planned = _single_updates(self.initial, precomputed=True)
        for actual, expected in zip(planned, reference, strict=True):
            self.assert_same_update(actual, expected)
            self.assertEqual(actual["flux_forward_calls"], 4)
            self.assertEqual(expected["flux_forward_calls"], 6)
            self.assertEqual(actual["padding_forward_calls"], 0)
            self.assertEqual(actual["packs"], ((11, 0, 5), (1, 10, 4), (9, 2, 7), (8, 3, 6)))

    def test_precomputed_two_rank_fsdp_preserves_updates_and_removes_count_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(_worker, args=(directory, self.initial), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 120
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError("precomputed FSDP CPU test exceeded 120 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            results = {(rank, mode): torch.load(Path(directory) / f"rank-{rank}-{mode}.pt", weights_only=True)
                       for rank in range(2) for mode in ("runtime", "precomputed", "precomputed_per_microbatch", "background")}
        for rank in range(2):
            for update in range(3):
                reference = results[(rank, "runtime")][update]
                planned = results[(rank, "precomputed")][update]
                self.assert_same_update(planned, reference)
                per_microbatch = results[(rank, "precomputed_per_microbatch")][update]
                self.assert_same_update(per_microbatch, planned)
                self.assertEqual(per_microbatch["flux_forward_calls"], planned["flux_forward_calls"])
                self.assertEqual(per_microbatch["padding_forward_calls"], planned["padding_forward_calls"])
                self.assertFalse(any(row["dtype"] == "torch.int64" for row in per_microbatch["collectives"]))
                self.assertEqual(planned["flux_forward_calls"], 6)
                self.assertEqual(planned["padding_forward_calls"], 3 if rank == 0 else 0)
                self.assertEqual(planned["packs"], reference["packs"])
                self.assertEqual([row for row in reference["collectives"] if row["dtype"] == "torch.int64"],
                                 [dict(dtype="torch.int64", size=2), dict(dtype="torch.int64", size=1)])
                self.assertFalse(any(row["dtype"] == "torch.int64" for row in planned["collectives"]))
                self.assertTrue(any(row["dtype"] == "torch.int32" for row in planned["collectives"]))
                for field in ("gradients", "parameters", "optimizer"):
                    torch.testing.assert_close(planned[field], results[(0, "precomputed")][update][field],
                                               atol=0, rtol=0)
                background = results[(rank, "background")][update]
                for field, tolerance in (("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
                                         ("optimizer", GRAD_TOLERANCE), ("gradient_norm", GRAD_TOLERANCE)):
                    torch.testing.assert_close(background[field], reference[field], **tolerance)
                self.assertFalse(any(row["dtype"] == "torch.int64" for row in background["collectives"]))
        # Ownership and local losses change; compare the complete global workload.
        for update in range(3):
            for mode in ("runtime", "background"):
                global_draws = sorted((draw for rank in range(2)
                    for draw in results[(rank, mode)][update]["draws"]), key=lambda draw: draw["target"])
                global_losses = sorted((target, loss) for rank in range(2)
                    for target, loss in zip(results[(rank, mode)][update]["global_target_indices"],
                                            results[(rank, mode)][update]["target_losses"], strict=True))
                if mode == "runtime":
                    expected_draws, expected_losses = global_draws, global_losses
                else:
                    torch.testing.assert_close(global_draws, expected_draws, atol=0, rtol=0)
                    torch.testing.assert_close(global_losses, expected_losses, **LOSS_TOLERANCE)

    def test_source_and_target_ids_survive_global_precomputation(self):
        prefixes = ((300, 800, 2100), (1700, 2400, 5000),
                    (450, 1100, 2900), (1000, 2000, 4300),
                    (1200, 2000, 3800), (200, 500, 1700),
                    (300, 1100, 3500), (1800, 3000, 6000))
        counts = tuple(len(source) for source in prefixes)
        keys = tuple(f"source-{index}" for index in range(len(prefixes)))
        original = tuple(plan_distributed_batches(
            counts, optimizer_batch_size=12, rank=rank, world_size=2,
            source_joint_tokens=tuple(sum(prefix + 900 for prefix in source) for source in prefixes),
        ) for rank in range(2))
        plans, metadata = precompute_distributed_packing(
            original, source_keys=keys, source_prefix_tokens=prefixes, max_tokens=9300,
        )
        repeated, repeated_metadata = precompute_distributed_packing(
            original, source_keys=keys, source_prefix_tokens=prefixes, max_tokens=9300,
        )
        self.assertEqual(plans, repeated)
        self.assertEqual(metadata, repeated_metadata)
        for batch in range(2):
            source_to_targets = {source: tuple(range(3 * (source % 4), 3 * (source % 4 + 1)))
                                 for source in range(batch * 4, batch * 4 + 4)}
            self.assertEqual(sorted(source for rank in plans for source in rank[batch].source_indices),
                             list(range(batch * 4, batch * 4 + 4)))
            self.assertEqual(sorted(target for rank in plans for target in rank[batch].target_indices),
                             list(range(12)))
            shared_count = max(len(rank[batch].flux_plan.microbatch_plans) for rank in plans)
            for rank in plans:
                plan = rank[batch]
                expected_ids = tuple(target for source in plan.source_indices for target in source_to_targets[source])
                expected_prefixes = tuple(prefix for source in plan.source_indices for prefix in prefixes[source])
                self.assertEqual(plan.epoch_batch_index, batch)
                self.assertEqual(plan.target_indices, expected_ids)
                self.assertEqual(plan.flux_plan.target_prefix_tokens, expected_prefixes)
                self.assertEqual(plan.global_target_count, 12)
                self.assertEqual(plan.flux_plan.collective_call_count, shared_count)
                calls = plan.flux_plan.microbatch_plans
                self.assertEqual(sorted(index for call in calls for index in call.target_indices),
                                 list(range(len(expected_ids))))
                for call in calls:
                    self.assertEqual(call.joint_tokens, sum(expected_prefixes[i] + 900 for i in call.target_indices))
                    self.assertLessEqual(call.joint_tokens, 9300)

    def test_benchmark_precomputes_before_timing_and_persists_plan_used_by_trainer(self):
        base = _config(source_count=8, world_size=1)
        recipe = base.recipe_dict()
        recipe["execution"].update(rank_batch_assignment="token_balanced_v1",
                                    flux_packing_algorithm="legacy_bfd_v1")
        config = parse_training_config(recipe, config_path=base.path)
        dataset, backend = Sources(source_count=8), SegmentedReference()
        models = TrainingModels(
            tokenizer=object(), flux_transformer=_model(self.initial),
            nvarc_encoder=FrozenText(), vae_encoder=FrozenVae(), vae_decoder=None,
            flux_compute_dtype=torch.float32, flux_packing_backend=backend,
        )
        # Exercise orchestration on one CPU without creating a DDP process group.
        runtime = TrainingRuntime(0, 0, 1, torch.device("cpu"), None, "ddp")
        lifecycle, received_plans, records = [], [], []
        before_step = BenchmarkInstrumentation.before_step

        def precompute(*args, **kwargs):
            lifecycle.append("precompute")
            return precompute_distributed_packing(*args, **kwargs)

        def begin_step(experiment):
            lifecycle.append("before_step")
            return before_step(experiment)

        class ObserveTrainer(Trainer):
            def step(self, sequences, *, batch_plan=None):
                lifecycle.append("trainer_step")
                assert batch_plan is not None and batch_plan.flux_plan is not None
                assert tuple(sequence.input_ids[0] // 11 for sequence in sequences) == batch_plan.source_indices
                received_plans.append(batch_plan)
                return super().step(sequences, batch_plan=batch_plan)

        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "token-lengths.json"
            metadata_path.write_text("prepared test metadata", encoding="utf-8")
            output = Path(directory) / "benchmark"
            with RunArtifacts(output) as artifacts:
                def emit(record):
                    records.append(record)
                    artifacts(record)

                with patch("diffusion_arc.training.training_loop.load_models", return_value=models), \
                        patch("diffusion_arc.training.training_loop.distributed_runtime",
                              return_value=nullcontext(runtime)), \
                        patch("diffusion_arc.training.training_loop.DDP", side_effect=lambda model, **_: model), \
                        patch("diffusion_arc.training.training_loop.Trainer", ObserveTrainer), \
                        patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset), \
                        patch("diffusion_arc.training.distributed_data.load_source_token_lengths",
                              return_value=((4, 6, 8),) * 8), \
                        patch("diffusion_arc.training.distributed_data.source_lengths_path", return_value=metadata_path), \
                        patch("diffusion_arc.training.distributed_data.precompute_distributed_packing", precompute), \
                        patch.object(BenchmarkInstrumentation, "before_step", begin_step), \
                        patch("diffusion_arc.training.trainer.plan_flux_batches",
                              side_effect=AssertionError("runtime packing used")), \
                        patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render), \
                        patch("diffusion_arc.training.run.benchmarking.print", create=True):
                    progress = run_training_benchmark(
                        config, max_steps=2, emit=emit, expected_source_count=8,
                        measurement=BenchmarkWindow(1, 1), precompute_packing=True,
                    )
            manifest = json.loads((output / "manifest.json").read_text())
            artifact = json.loads((output / "packing_plans.json").read_text())
            steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(lifecycle, ["precompute", "before_step", "trainer_step", "before_step", "trainer_step"])
        self.assertEqual([row["event"] for row in records], ["start", "packing_plan", "step", "step", "complete"])
        self.assertEqual(dataset.visited, [source for plan in received_plans for source in plan.source_indices])
        self.assertEqual((progress.optimizer_step, progress.sources_seen, progress.targets_seen), (2, 8, 24))
        self.assertTrue(manifest["precompute_packing"])
        self.assertEqual(artifact["metadata"], manifest["rank_batch_assignment"]["precomputed_packing"])
        self.assertGreaterEqual(artifact["planning_seconds"], 0)
        self.assertEqual(len(artifact["ranks"]), 1)
        self.assertEqual(summary["benchmark"]["measured_steps"], 1)
        for plan, saved, step in zip(received_plans, artifact["ranks"][0], steps, strict=True):
            self.assertEqual(saved["source_indices"], list(plan.source_indices))
            self.assertEqual(saved["target_indices"], list(plan.target_indices))
            self.assertEqual(saved["flux_plan"]["target_prefix_tokens"], list(plan.flux_plan.target_prefix_tokens))
            self.assertEqual(saved["source_indices"], step["ranks"][0]["source_indices"])
            self.assertEqual(saved["target_indices"], step["ranks"][0]["target_indices"])

    def test_wrong_count_or_prefix_fails_without_optimizer_or_flow_work(self):
        sequences = (Sources(source_count=1)[0],)
        actual_prefixes = sequences[0].prefix_token_ends[1:]
        for failure in ("count", "prefix"):
            with self.subTest(failure=failure):
                model, trainer = _trainer(self.initial)
                prefixes = actual_prefixes[:-1] if failure == "count" else (5, *actual_prefixes[1:])
                ids = tuple(range(len(prefixes)))
                plan = _precomputed(RankBatchPlan(0, (0,), ids, 3), prefixes,
                    (FluxMicrobatchPlan(ids, sum(prefix + 900 for prefix in prefixes)),), calls=1)
                with patch.object(trainer.optimizer, "step", side_effect=AssertionError("optimizer used")), \
                        patch.object(trainer, "flow_sampler", side_effect=AssertionError("flow draws used")), \
                        patch.object(trainer.flux_model, "forward", side_effect=AssertionError("FLUX used")), \
                        patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
                    with self.assertRaisesRegex(ValueError, "count|prefix|plan"):
                        trainer.step(sequences, batch_plan=plan)
                self.assertFalse(trainer.completed_update)
                torch.testing.assert_close(model.state_dict(), self.initial, atol=0, rtol=0)

    def test_cpu_padding_preserves_supplied_order_and_single_device_uses_only_real_calls(self):
        calls = (FluxMicrobatchPlan((0,), 901), FluxMicrobatchPlan((1,), 910))
        fsdp = object.__new__(FsdpStrategy)
        fsdp.device, fsdp.process_group = torch.device("cpu"), None
        single = object.__new__(SingleDeviceStrategy)
        with patch.object(dist, "all_reduce", side_effect=AssertionError("pack count collective used")):
            self.assertEqual(fsdp.plan_microbatch_calls(calls, collective_call_count=4),
                             calls + (None, None))
            self.assertEqual(single.plan_microbatch_calls(calls, collective_call_count=4), calls)
            for count in (0, 1, True):
                with self.subTest(count=count), self.assertRaises((ValueError, TypeError)):
                    fsdp.plan_microbatch_calls(calls, collective_call_count=count)


if __name__ == "__main__":
    unittest.main()
