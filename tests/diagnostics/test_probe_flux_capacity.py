"""Check exact pack budgets, real AdamW accumulation and bounded worker failures."""

from copy import deepcopy
from contextlib import contextmanager
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.diagnostics import probe_flux_capacity as probe
from diffusion_arc.model.flux.packing import PackedFluxForward
from diffusion_arc.training.distributed import TrainingRuntime
from diffusion_arc.training.distribution_strategy.fsdp import FsdpStrategy
from diffusion_arc.training.step.loss import velocity_mse
from tests.model.test_flux_packing import SegmentedReference
from tests.training.test_ddp_trainer import _model


def _execute_updates(initial, runtime, *, sharded=False, reshard_after_forward=True):
    model = _model(initial)
    backend = SegmentedReference()
    flux_model = PackedFluxForward(model, compute_dtype=torch.float32, backend=backend)
    if sharded:
        flux_model = probe.shard_probe_flux(flux_model, runtime=runtime,
                                           reshard_after_forward=reshard_after_forward)
    optimizer = torch.optim.AdamW(flux_model.parameters(), lr=1e-4, foreach=False)
    # Identical rank-local inputs make the plain execution an independent oracle.
    generator = torch.Generator().manual_seed(71)
    sources = (torch.randn((6, 12), generator=generator),)
    observed = []
    gradients = []

    def progress(operation, update):
        if operation == "build_pack":
            observed.append((update, any(p.grad is not None for p in flux_model.parameters())))
        elif operation == "clip_optimizer":
            gradients.append({name: (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad)
                              .detach().clone() for name, p in model.named_parameters() if p.grad is not None})

    with patch.object(dist, "reduce_scatter_single", wraps=dist.reduce_scatter_single) as reduce:
        rows = probe.run_probe_updates(flux_model, optimizer, runtime=runtime, sources=sources,
            text_lengths=(2, 4), updates=2, loss_function=velocity_mse, max_grad_norm=1.,
            generator=generator, on_progress=progress)
    parameters = {name: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().clone()
                  for name, p in model.named_parameters()}
    return rows, observed, parameters, gradients, reduce.call_count


def _fsdp_worker(rank, directory, initial):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "gloo-store").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        runtime = TrainingRuntime(rank, rank, 2, torch.device("cpu"), dist.group.WORLD, "fsdp2")
        for reshard_after_forward in (True, False):
            rows, observed, parameters, gradients, reduction_calls = _execute_updates(initial,
                runtime, sharded=True, reshard_after_forward=reshard_after_forward)
            torch.save(dict(rows=rows, observed=observed, parameters=parameters, gradients=gradients,
                            gradient_reduce_calls=reduction_calls),
                       Path(directory) / f"rank-{rank}-{reshard_after_forward}.pt")
    finally:
        dist.destroy_process_group()


class FluxCapacityTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_budget_is_exact_including_images_at_segment_boundaries(self):
        expected = {9300: (8400,), 9301: (7500, 1), 12288: (8400, 2088),
                    16000: (8400, 5800), 18600: (8400, 8400), 18601: (8400, 7500, 1),
                    49152: (8400, 8400, 8400, 8400, 8400, 1752)}
        for budget, lengths in expected.items():
            with self.subTest(budget=budget):
                self.assertEqual(probe.probe_text_lengths(budget), lengths)
        # Include both sides of every new-segment boundary in the searched range.
        for budget in range(9300, 49153):
            lengths = probe.probe_text_lengths(budget)
            self.assertTrue(all(1 <= length <= 8400 for length in lengths))
            self.assertEqual(sum(lengths) + 900 * len(lengths), budget)
            self.assertEqual(len(lengths), (budget + 9299) // 9300)

    def test_preflight_needs_no_dataset_and_rejects_non_fsdp_recipe(self):
        config = probe.load_probe_config(probe.REPOSITORY_ROOT / "configs/training/benchmarks/fsdp2_training.json",
                                        require_checkpoints=False)
        self.assertEqual(config.recipe["execution"]["distributed_strategy"], "fsdp2")
        with self.assertRaisesRegex(ValueError, "fresh two-GPU FSDP2"):
            probe.load_probe_config(probe.REPOSITORY_ROOT / "configs/training/benchmarks/ddp_training.json",
                                    require_checkpoints=False)
        with self.assertRaisesRegex(ValueError, "capacity probe does not support benchmark fsdp_gradient_sync"):
            probe.load_probe_config(probe.REPOSITORY_ROOT /
                "configs/training/benchmarks/fsdp2_16k_per_microbatch_sync_precomputed_packing.json",
                require_checkpoints=False)
        for options in ((12288, 0, 49152, 2), (9000, 1024, 49152, 2),
                        (12288, 1024, 10000, 2), (12288, 1024, 49152, 1),
                        (True, 1024, 49152, 2), (12288, 1024, 223201, 2)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                probe.validate_options(*options)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            probe.validate_options(12288, 1024, 49152, 2, "false")
        self.assertTrue(probe.build_parser().parse_args([]).reshard_after_forward)
        self.assertFalse(probe.build_parser().parse_args(["--no-reshard-after-forward"]).reshard_after_forward)

    def test_gradient_sync_is_restored_after_a_failed_microbatch(self):
        for synchronize_each_microbatch in (False, True):
            with self.subTest(synchronize_each_microbatch=synchronize_each_microbatch):
                model = Mock()
                strategy = object.__new__(FsdpStrategy)
                strategy.flux_model = model
                strategy.synchronize_each_microbatch = synchronize_each_microbatch
                with self.assertRaisesRegex(RuntimeError, "backward failed"):
                    with strategy.microbatch_context(is_last=False):
                        raise RuntimeError("backward failed")
                self.assertEqual(model.set_requires_gradient_sync.call_args_list,
                                 [call(synchronize_each_microbatch), call(True)])

    def test_search_stops_on_first_oom_and_preserves_tested_bracket(self):
        seen = []

        def evaluate(tokens):
            seen.append(tokens)
            return dict(status="success" if tokens < 15360 else "oom", joint_tokens=tokens)

        result = probe.search_capacity(evaluate, start_tokens=12288, step_tokens=1024, max_tokens=49152)
        self.assertEqual(seen, [12288, 13312, 14336, 15360])
        self.assertEqual(result["largest_tested_success"], 14336)
        self.assertEqual(result["smallest_tested_oom"], 15360)
        self.assertTrue(result["boundary_known"])
        self.assertFalse(result["full_pipeline_capacity_validated"])

    def test_initial_oom_walks_down_and_maximum_is_explicitly_not_a_bound(self):
        seen = []

        def evaluate(tokens):
            seen.append(tokens)
            return dict(status="success" if tokens <= 10000 else "oom")

        result = probe.search_capacity(evaluate, start_tokens=12288, step_tokens=1024, max_tokens=49152)
        self.assertEqual(seen, [12288, 11264, 10240, 9300])
        self.assertEqual((result["largest_tested_success"], result["smallest_tested_oom"]), (9300, 10240))
        result = probe.search_capacity(lambda _: dict(status="success"), start_tokens=12288,
                                       step_tokens=1024, max_tokens=13000)
        self.assertEqual(result["largest_tested_success"], 13000)
        self.assertTrue(result["search_limit_reached"])
        self.assertFalse(result["boundary_known"])

    def test_non_oom_errors_and_timeouts_stop_without_inventing_a_bound(self):
        for status in ("error", "time_limit"):
            outcomes = iter((dict(status="success"), dict(status=status)))
            result = probe.search_capacity(lambda _: next(outcomes), start_tokens=12288,
                                           step_tokens=1024, max_tokens=49152)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["largest_tested_success"], 12288)
            self.assertIsNone(result["smallest_tested_oom"])

    def test_two_updates_initialize_moments_and_accumulate_gradients_between_packs(self):
        torch.manual_seed(31)
        rows, observed, parameters, _, _ = _execute_updates(deepcopy(_model().state_dict()),
            TrainingRuntime(0, 0, 1, torch.device("cpu"), None))
        self.assertEqual(observed, [(1, False), (1, True), (2, False), (2, True)])
        self.assertEqual([row["update"] for row in rows], [1, 2])
        expected_bytes = 2 * sum(probe._tensor_bytes(t) for t in parameters.values())
        self.assertEqual([row["adamw_moment_bytes"] for row in rows], [expected_bytes] * 2)
        self.assertTrue(all(row["gradient_norm"] > 0 and row["loss"] > 0 for row in rows))

    def test_actual_fsdp_updates_match_single_process_and_moments_remain_sharded(self):
        torch.manual_seed(31)
        initial = deepcopy(_model().state_dict())
        expected, observed, parameters, gradients, _ = _execute_updates(initial,
            TrainingRuntime(0, 0, 1, torch.device("cpu"), None))
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(_fsdp_worker, args=(directory, initial), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        raise TimeoutError("FSDP capacity CPU test exceeded 90 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for reshard_after_forward in (True, False):
                rank_rows = []
                for rank in range(2):
                    actual = torch.load(Path(directory) / f"rank-{rank}-{reshard_after_forward}.pt", weights_only=True)
                    self.assertEqual(actual["observed"], [(i, False) for i, _ in observed])
                    torch.testing.assert_close(torch.tensor([r["loss"] for r in actual["rows"]]),
                                               torch.tensor([r["loss"] for r in expected]), rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(actual["parameters"], parameters, rtol=2e-4, atol=2e-6)
                    torch.testing.assert_close(actual["gradients"], gradients, rtol=1e-5, atol=1e-6)
                    # Two updates, each reducing one double block, one single block and the root.
                    self.assertEqual(actual["gradient_reduce_calls"], 2 * 3)
                    rank_rows.append(actual["rows"])
                for update in range(2):
                    self.assertEqual(sum(rows[update]["adamw_moment_bytes"] for rows in rank_rows),
                                     expected[update]["adamw_moment_bytes"])

    def test_worker_reports_require_both_passes_and_an_explicit_probe_oom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(rank, status, phase="probe", **kwargs):
                probe._write_json(root / f"rank-{rank}.json", dict(rank=rank, status=status, phase=phase,
                    joint_tokens=12288, updates=[dict(update=i, peak_allocated_bytes=100 + rank,
                                                    peak_reserved_bytes=200 + rank) for i in (1, 2)], **kwargs))

            def collect(code, timed_out=False):
                return probe.collect_candidate(root, tokens=12288, updates=2, returncode=code, timed_out=timed_out)

            write(0, "success")
            self.assertEqual(collect(0)["status"], "error")
            write(1, "success")
            self.assertEqual(collect(0)["peak_allocated_bytes"], 101)
            self.assertEqual(collect(1)["status"], "error")
            write(0, "error", phase="setup", error_type="OutOfMemoryError")
            self.assertEqual(collect(1)["status"], "error")
            self.assertEqual(collect(1, True)["status"], "error")
            write(0, "oom")
            write(1, "error", error_type="DistBackendError")
            self.assertEqual(collect(1)["status"], "oom")
            self.assertEqual(collect(1, True)["status"], "oom")
            write(1, "error", error_type="ValueError")
            self.assertEqual(collect(1)["status"], "error")
            self.assertEqual(collect(1, True)["status"], "error")
            write(0, "running")
            write(1, "running")
            self.assertEqual(collect(1, True)["status"], "time_limit")

    def test_worker_writes_oom_before_cleanup_and_setup_oom_is_an_error(self):
        config = probe.load_probe_config(probe.REPOSITORY_ROOT / "configs/training/benchmarks/fsdp2_training.json",
                                         require_checkpoints=False)
        models = SimpleNamespace(flux_transformer=_model(), flux_compute_dtype=torch.float32,
                                 flux_packing_backend=SegmentedReference())
        for setup_failure in (False, True):
            with self.subTest(setup_failure=setup_failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = SimpleNamespace(config=config.path, output_dir=root, candidate_tokens=12288, updates=2,
                                       reshard_after_forward=False)
                expected_status = "error" if setup_failure else "oom"

                @contextmanager
                def runtime(_config):
                    try:
                        yield TrainingRuntime(0, 0, 2, torch.device("cuda:0"), object(), "fsdp2")
                    finally:
                        # torchrun may kill peers during cleanup; evidence must exist first.
                        record = json.loads((root / "rank-0.json").read_text())
                        self.assertEqual(record["status"], expected_status)
                        self.assertEqual(record["phase"], "setup" if setup_failure else "probe")
                        self.assertFalse(record["fsdp_reshard_after_forward"])
                        self.assertFalse(record["fsdp_root_reshard_after_forward"])
                        self.assertTrue(record["fsdp_reshard_after_backward"])
                        self.assertEqual(record["fsdp_gradient_sync"], "last_microbatch")

                with patch.dict(probe.os.environ, {"RANK": "0"}), \
                     patch.object(probe, "load_probe_config", return_value=config), \
                     patch.object(probe, "distributed_runtime", runtime), \
                     patch.object(probe, "seed_process"), \
                     patch.object(probe, "load_models", side_effect=torch.cuda.OutOfMemoryError("setup OOM")
                                  if setup_failure else lambda *_a, **_kw: models), \
                     patch.object(probe, "shard_probe_flux", side_effect=lambda model, **_kw: model) as shard, \
                     patch.object(probe.torch, "Generator"), \
                     patch.object(probe, "allocate_source_reserve", return_value=((), torch.zeros(1), {})), \
                     patch.object(probe.torch.cuda, "get_device_properties",
                                  return_value=SimpleNamespace(name="test GPU", total_memory=1000)), \
                     patch.object(probe, "version", return_value="test"), \
                     patch.object(probe, "_memory", return_value={}), \
                     patch.object(probe, "run_probe_updates", side_effect=torch.cuda.OutOfMemoryError("probe OOM")):
                    self.assertEqual(probe.run_candidate(args), 1 if setup_failure else probe.OOM_EXIT_CODE)
                    if not setup_failure:
                        self.assertFalse(shard.call_args.kwargs["reshard_after_forward"])

    def test_search_persists_each_candidate_and_starts_fresh_torchrun_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(config=probe.REPOSITORY_ROOT / "configs/training/benchmarks/fsdp2_training.json",
                                   output_dir=Path(directory) / "output", start_tokens=12288,
                                   step_tokens=1024, max_tokens=13312, updates=2, reshard_after_forward=False)
            calls, commits = [], []

            def worker(command, *, log, timeout):
                calls.append(command)
                root = Path(command[command.index("--output-dir") + 1])
                tokens = int(command[command.index("--candidate-tokens") + 1])
                for rank in range(2):
                    probe._write_json(root / f"rank-{rank}.json", dict(rank=rank, status="success", phase="probe",
                        joint_tokens=tokens, updates=[dict(update=i, peak_allocated_bytes=100,
                                                          peak_reserved_bytes=200) for i in (1, 2)]))
                return 0, False

            original_loader = probe.load_probe_config
            with patch.object(probe, "load_probe_config", side_effect=lambda path: original_loader(path, require_checkpoints=False)), \
                 patch.object(probe, "_run_worker", side_effect=worker):
                result = probe.run_search(args, on_candidate_end=lambda: commits.append(True), timeout_seconds=60)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(commits), 2)
            self.assertNotEqual(calls[0][calls[0].index("--output-dir") + 1],
                                calls[1][calls[1].index("--output-dir") + 1])
            self.assertTrue(all("--nproc_per_node=2" in command and "--max_restarts=0" in command for command in calls))
            self.assertTrue(all("--no-reshard-after-forward" in command for command in calls))
            manifest = json.loads((args.output_dir / "manifest.json").read_text())
            self.assertFalse(manifest["fsdp_reshard_after_forward"])
            self.assertFalse(manifest["fsdp_root_reshard_after_forward"])
            self.assertTrue(manifest["fsdp_reshard_after_backward"])
            self.assertEqual(manifest["fsdp_gradient_sync"], "last_microbatch")
            self.assertFalse(manifest["frozen_encoder_execution_measured"])
            self.assertFalse(manifest["global_training_batch_reproduced"])
            self.assertEqual(json.loads((args.output_dir / "summary.json").read_text())["largest_tested_success"], 13312)


if __name__ == "__main__":
    unittest.main()
