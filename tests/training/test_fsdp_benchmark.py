"""Compare default FSDP accumulation to a reduction reference in the benchmark."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusion_arc.training.benchmark_runner import run_training_benchmark
from diffusion_arc.training.config import parse_training_config
from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.distribution_strategy.fsdp import FsdpStrategy, shard_flux
from diffusion_arc.training.run.benchmarking import BenchmarkWindow
from diffusion_arc.training.setup import TrainingModels
from diffusion_arc.training.step.flux_microbatch import compute_fsdp_dummy_loss
from diffusion_arc.training.trainer import Trainer
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_runner import _config
from tests.training.test_ddp_trainer import (
    FrozenText, FrozenVae, _model, _render, GRAD_TOLERANCE, PARAM_TOLERANCE,
)
from tests.training.test_numerical_runner import UnevenSources, _uneven_packs


def _full_tensor(value):
    """Materialize one parameter or moment on every rank for numerical comparison."""
    return (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().clone()


def _benchmark_updates(rank, initial, *, synchronize_each_microbatch, blocks_per_group=1,
                       fsdp_gradient_sync="last_microbatch"):
    model, backend = _model(initial), SegmentedReference()
    dataset = UnevenSources(source_count=12)
    base = _config(source_count=12, world_size=2)
    recipe = base.recipe_dict()
    recipe["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
    recipe["execution"]["fsdp_blocks_per_group"] = blocks_per_group
    config = parse_training_config(recipe, config_path=base.path)
    assert config.fsdp_reshard_after_forward is True
    models = TrainingModels(tokenizer=object(), flux_transformer=model, nvarc_encoder=FrozenText(),
                            vae_encoder=FrozenVae(), vae_decoder=None, flux_compute_dtype=torch.float32,
                            flux_packing_backend=backend)
    snapshots, records = [], []

    @contextmanager
    def runtime(phase_config, **kwargs):
        yield TrainingRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD, "fsdp2")

    class RecordingTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            assert isinstance(self.distribution_strategy, FsdpStrategy)
            assert self.packing.algorithm == "legacy_bfd_v1"

        def _clip_gradients(self):
            self.pre_clip_gradients = {name: _full_tensor(parameter.grad)
                                       for name, parameter in model.named_parameters()}
            return super()._clip_gradients()

        def step(self, sequences):
            contexts = []
            microbatch_context = self.distribution_strategy.microbatch_context

            @contextmanager
            def observe_context(*, is_last):
                contexts.append(dict(is_last=is_last, dummy=False))
                with microbatch_context(is_last=is_last):
                    yield

            def dummy_loss(*args, **kwargs):
                contexts[-1]["dummy"] = True
                return compute_fsdp_dummy_loss(*args, **kwargs)

            with patch.object(self.distribution_strategy, "microbatch_context", observe_context), \
                    patch.object(self.flux_model, "set_requires_gradient_sync",
                                 wraps=self.flux_model.set_requires_gradient_sync) as sync, \
                    patch("diffusion_arc.training.trainer.compute_fsdp_dummy_loss", dummy_loss), \
                    patch.object(dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single) as reduce:
                result = super().step(sequences)
            moments = {name: {key: _full_tensor(value) for key, value in self.optimizer.state[parameter].items()}
                       for name, parameter in model.named_parameters()}
            snapshots.append(dict(gradients=self.pre_clip_gradients,
                parameters={name: _full_tensor(parameter) for name, parameter in model.named_parameters()},
                optimizer=moments, scheduler=deepcopy(self.scheduler.state_dict()),
                contexts=contexts, sync_calls=[call.args[0] for call in sync.call_args_list],
                reduce_scatter_calls=reduce.call_count, loss=result.loss, gradient_norm=result.gradient_norm,
                padding_forward_calls=result.padding_forward_calls,
                flux_forward_calls=result.flux_forward_calls,
                real_packs=len(result.diagnostics.flux_microbatch_plans)))
            assert all(parameter.grad is None for parameter in model.parameters())
            return result

    # Recreate the previous reduction order only as a test oracle.
    reference_context = (patch.object(FsdpStrategy, "microbatch_context", lambda self, *, is_last: nullcontext())
                         if synchronize_each_microbatch else nullcontext())
    with reference_context, \
            patch("diffusion_arc.training.training_loop.distributed_runtime", runtime), \
            patch("diffusion_arc.training.training_loop.load_models", return_value=models), \
            patch("diffusion_arc.training.training_loop.shard_flux", wraps=shard_flux) as shard, \
            patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset), \
            patch("diffusion_arc.training.training_loop.Trainer", RecordingTrainer), \
            patch("diffusion_arc.training.trainer.plan_flux_batches", side_effect=_uneven_packs), \
            patch("diffusion_arc.training.step.encoding._render_valid_grid_canvas", _render):
        progress = run_training_benchmark(config, max_steps=3, expected_source_count=12,
                                         measurement=BenchmarkWindow(1, 2), emit=records.append,
                                         fsdp_gradient_sync=fsdp_gradient_sync)
    assert shard.call_count == 1 and shard.call_args.kwargs["reshard_after_forward"] is True
    assert shard.call_args.kwargs["blocks_per_group"] == blocks_per_group
    assert (progress.optimizer_step, progress.sources_seen, progress.targets_seen) == (3, 12, 36)
    assert len(snapshots) == 3
    assert all(not parameter.requires_grad and parameter.grad is None
               for encoder in (models.nvarc_encoder, models.vae_encoder) for parameter in encoder.parameters())
    if rank == 0:
        assert [row["event"] for row in records] == ["start", "step", "step", "step", "complete"]
        assert records[0]["mode"] == "fsdp2_benchmark"
        assert records[0]["fsdp_gradient_sync"] == fsdp_gradient_sync
        assert [row["phase"] for row in records[1:-1]] == ["warmup", "measure", "measure"]
        assert records[-1]["benchmark"]["measured_steps"] == 2
    else:
        assert records == []
    return snapshots


def _worker(rank, directory, initial):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "gloo-store").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for mode in ("reference", "default", "grouped", "per_microbatch"):
            snapshots = _benchmark_updates(rank, initial, synchronize_each_microbatch=mode == "reference",
                                           blocks_per_group=5 if mode == "grouped" else 1,
                                           fsdp_gradient_sync="per_microbatch" if mode == "per_microbatch" else "last_microbatch")
            torch.save(snapshots, Path(directory) / f"rank-{rank}-{mode}.pt")
    finally:
        dist.destroy_process_group()


class FsdpBenchmarkTests(unittest.TestCase):
    def test_default_batch_end_sync_matches_reference_across_uneven_rank_schedules(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(167)
        initial = deepcopy(_model().state_dict())
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(_worker, args=(directory, initial), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 120
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError("FSDP benchmark CPU test exceeded 120 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            results = {(rank, mode): torch.load(Path(directory) / f"rank-{rank}-{mode}.pt", weights_only=True)
                       for rank in range(2) for mode in ("reference", "default", "grouped", "per_microbatch")}
        # One double block, one single block, and the packed root each reduce once.
        reduction_groups = 3
        for rank in range(2):
            for update in range(3):
                baseline, delayed = (results[(rank, mode)][update] for mode in ("reference", "default"))
                grouped = results[(rank, "grouped")][update]
                per_microbatch = results[(rank, "per_microbatch")][update]
                for field, tolerance in (("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
                                         ("optimizer", GRAD_TOLERANCE), ("loss", GRAD_TOLERANCE),
                                         ("gradient_norm", GRAD_TOLERANCE)):
                    torch.testing.assert_close(per_microbatch[field], baseline[field], **tolerance)
                torch.testing.assert_close(per_microbatch["scheduler"], baseline["scheduler"], atol=0, rtol=0)
                self.assertEqual(per_microbatch["reduce_scatter_calls"], 6 * reduction_groups)
                self.assertEqual(per_microbatch["sync_calls"], [True, True] * 6)
                self.assertEqual(per_microbatch["contexts"], baseline["contexts"])
                torch.testing.assert_close(delayed["gradients"], baseline["gradients"], **GRAD_TOLERANCE)
                torch.testing.assert_close(delayed["parameters"], baseline["parameters"], **PARAM_TOLERANCE)
                torch.testing.assert_close(delayed["optimizer"], baseline["optimizer"], **GRAD_TOLERANCE)
                torch.testing.assert_close(delayed["scheduler"], baseline["scheduler"], atol=0, rtol=0)
                torch.testing.assert_close(delayed["loss"], baseline["loss"], **GRAD_TOLERANCE)
                for field, tolerance in (("gradients", GRAD_TOLERANCE), ("parameters", PARAM_TOLERANCE),
                                         ("optimizer", GRAD_TOLERANCE), ("loss", GRAD_TOLERANCE),
                                         ("gradient_norm", GRAD_TOLERANCE)):
                    torch.testing.assert_close(grouped[field], delayed[field], **tolerance)
                self.assertEqual(grouped["contexts"], delayed["contexts"])
                self.assertEqual(grouped["sync_calls"], delayed["sync_calls"])
                self.assertEqual(grouped["reduce_scatter_calls"], 2)  # Both blocks share a group plus root.
                self.assertEqual(baseline["reduce_scatter_calls"], 6 * reduction_groups)
                self.assertEqual(delayed["reduce_scatter_calls"], reduction_groups)
                self.assertEqual(baseline["sync_calls"], [])
                self.assertEqual(delayed["sync_calls"], [False, True] * 5 + [True, True])
                self.assertEqual(delayed["contexts"], baseline["contexts"])
                for snapshot in (baseline, delayed, grouped, per_microbatch):
                    self.assertEqual(snapshot["flux_forward_calls"], 6)
                    self.assertEqual(snapshot["real_packs"], 3 if rank == 0 else 6)
                    self.assertEqual(snapshot["padding_forward_calls"], 3 if rank == 0 else 0)
                    self.assertEqual([call["is_last"] for call in snapshot["contexts"]], [False] * 5 + [True])
                    self.assertEqual([call["dummy"] for call in snapshot["contexts"]],
                                     [False] * 3 + [rank == 0] * 3)
                    self.assertGreater(snapshot["gradient_norm"], 0.1)
                    self.assertEqual(snapshot["gradients"].keys(), initial.keys())
                    self.assertTrue(all(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                                        for state in snapshot["optimizer"].values()))
            for mode in ("reference", "default", "grouped", "per_microbatch"):
                for update in range(3):
                    for field in ("gradients", "parameters", "optimizer"):
                        torch.testing.assert_close(results[(rank, mode)][update][field],
                                                   results[(0, mode)][update][field], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
