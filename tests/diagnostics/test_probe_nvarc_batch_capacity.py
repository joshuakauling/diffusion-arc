"""CPU-only checks of NVARC capacity probing and worker isolation."""

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from diffusion_arc.model.contracts import NVARC_CONTEXT_WIDTH, NvarcBatchOutput
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.setup import TrainingModels
from scripts.diagnostics import probe_nvarc_batch_capacity as probe
from scripts.diagnostics._nvarc_tokenizer import make_nvarc_tokenizer


def _config():
    return load_training_config(
        probe.REPOSITORY_ROOT / "configs/training/benchmarks/single_gpu_b3.json"
    , expected_type="baseline")


class CapacitySearchTests(unittest.TestCase):
    def test_doubles_then_bisects_success_oom_boundary(self):
        seen = []

        def evaluate(size):
            seen.append(size)
            return {"batch_size": size, "status": "success" if size <= 6 else "oom"}

        result = probe.search_capacity(evaluate, maximum=32)
        self.assertEqual(seen, [1, 2, 4, 8, 6, 7])
        self.assertEqual(result["largest_tested_success"], 6)
        self.assertEqual(result["smallest_tested_oom"], 7)
        self.assertTrue(result["boundary_known"])
        self.assertFalse(result["full_pipeline_capacity_validated"])

    def test_non_power_of_two_limit_is_not_reported_as_capacity(self):
        result = probe.search_capacity(
            lambda size: {"batch_size": size, "status": "success"}, maximum=6,
        )
        self.assertEqual([row["batch_size"] for row in result["candidates"]], [1, 2, 4, 6])
        self.assertEqual(result["largest_tested_success"], 6)
        self.assertIsNone(result["smallest_tested_oom"])
        self.assertFalse(result["boundary_known"])
        self.assertTrue(result["search_limit_reached"])

    def test_first_candidate_oom_and_other_errors_are_distinct(self):
        oom = probe.search_capacity(lambda size: {"status": "oom"}, maximum=8)
        self.assertIsNone(oom["largest_tested_success"])
        self.assertEqual(oom["smallest_tested_oom"], 1)
        error = probe.search_capacity(
            lambda size: {"status": "success" if size == 1 else "error"}, maximum=8,
        )
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["largest_tested_success"], 1)
        self.assertIsNone(error["smallest_tested_oom"])
        self.assertFalse(error["boundary_known"])
        self.assertEqual(len(error["candidates"]), 2)

    def test_rejects_invalid_limits_before_evaluation(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.search_capacity(lambda size: self.fail("must not evaluate"), maximum=value)


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = make_nvarc_tokenizer(model_max_length=10000)

    def test_exact_8400_token_sources_exclude_the_final_output(self):
        sample = probe.make_sample(self.tokenizer, seed=167)
        self.assertEqual(sample.prefix_token_ends[1:], (4668, 6534, 8400))
        self.assertEqual(len(sample.input_ids), 9330)
        ids, offsets = probe.make_probe_inputs(
            self.tokenizer, seed=167, batch_size=3, sequence_length=8400,
        )
        self.assertEqual(offsets.tolist(), [0, 8400, 16800, 25200])
        self.assertEqual(ids.dtype, torch.int64)
        self.assertFalse(bool((ids == 13).any()))
        for source in ids.reshape(3, 8400):
            self.assertEqual(source.tolist(), list(sample.input_ids[:8400]))
        self.assertEqual(sample, probe.make_sample(self.tokenizer, seed=167))
        self.assertNotEqual(sample.input_ids, probe.make_sample(self.tokenizer, seed=168).input_ids)
        with self.assertRaisesRegex(ValueError, "expected 8192"):
            probe.make_probe_inputs(self.tokenizer, seed=167, batch_size=1, sequence_length=8192)

    def test_reserve_preserves_conservative_storage_without_cross_step_retention(self):
        with (
            patch.object(probe, "NVARC_CONTEXT_WIDTH", 4),
            patch.object(probe, "TARGET_LATENT_TOKEN_COUNT", 2),
            patch.object(probe, "TARGET_LATENT_CHANNELS", 3),
        ):
            tensors, metadata = probe.allocate_buffer_reserve(
                _config(), device=torch.device("cpu"), sequence_length=2,
            )
        self.assertEqual(metadata["source_slots"], 16)
        self.assertEqual(metadata["target_slots"], 48)
        self.assertEqual(metadata["additional_latent_target_slots"], 48)
        self.assertEqual(len(tensors), 17)
        self.assertEqual(tensors[0].dtype, torch.bfloat16)
        self.assertEqual(tensors[-1].dtype, torch.float32)
        self.assertEqual(metadata["hidden_bytes"], 16 * 2 * 4 * 2)
        self.assertEqual(metadata["latent_bytes"], 96 * 2 * 3 * 4)
        self.assertEqual(metadata["total_bytes"], sum(probe._tensor_bytes(t) for t in tensors))

    def test_initialization_allocates_representative_gradients_after_training_cleanup(self):
        class Nvarc(nn.Module):
            def forward(self, ids, offsets):
                return NvarcBatchOutput((torch.ones(len(ids), 4),))

        class Vae(nn.Module):
            def forward(self, images):
                return torch.ones(len(images), 900, 128)

        class Flux(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def forward(self, *, hidden_states, **kwargs):
                return SimpleNamespace(sample=hidden_states * self.weight)

        models = TrainingModels(vae_decoder=object(), flux_packing_backend=object(),
            tokenizer=self.tokenizer, nvarc_encoder=Nvarc(), vae_encoder=Vae(),
            flux_transformer=Flux(), flux_compute_dtype=torch.bfloat16,
        )
        config = _config()
        optimizer = probe.build_optimizer(config, flux_transformer=models.flux_transformer)
        cleared_gradients = []
        zero_grad = optimizer.zero_grad

        def clear_gradients(**kwargs):
            zero_grad(**kwargs)
            cleared_gradients.append(models.flux_transformer.weight.grad)

        def packed_forward(transformer, noisy, *args, **kwargs):
            return noisy * transformer.weight

        with patch.object(optimizer, "zero_grad", side_effect=clear_gradients), \
             patch("diffusion_arc.model.flux.packing.predict_packed_flux_velocity", side_effect=packed_forward):
            metadata = probe.initialize_training_memory(
                config, models, optimizer, device=torch.device("cpu"),
            )
        self.assertEqual(cleared_gradients, [None, None])
        parameter = models.flux_transformer.weight
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad))
        self.assertEqual(parameter.grad.item(), 0)
        self.assertEqual(parameter.item(), 0.5)  # B3's first scheduled LR is zero.
        self.assertEqual(optimizer.state[parameter]["step"].item(), 1)
        self.assertEqual(metadata["gradient_bytes"], 4)
        self.assertEqual(metadata["adamw_moment_bytes"], 8)
        self.assertEqual(metadata["initialization_targets"], 1)
        self.assertEqual(metadata["learning_rate"], 0)

    def test_initialization_rejects_retained_gradients_and_missing_optimizer_state(self):
        models = SimpleNamespace(tokenizer=self.tokenizer, flux_transformer=nn.Linear(1, 1),
                                 flux_packing_backend=object())
        models.nvarc_encoder = models.vae_encoder = object()
        models.flux_compute_dtype = torch.bfloat16
        optimizer = probe.build_optimizer(_config(), flux_transformer=models.flux_transformer)
        with patch.object(probe, "Trainer"):
            with self.assertRaisesRegex(RuntimeError, "AdamW exp_avg"):
                probe.initialize_training_memory(_config(), models, optimizer, device=torch.device("cpu"))
            for parameter in models.flux_transformer.parameters():
                parameter.grad = torch.ones_like(parameter)
            with self.assertRaisesRegex(RuntimeError, "release every FLUX gradient"):
                probe.initialize_training_memory(_config(), models, optimizer, device=torch.device("cpu"))


class CandidateTests(unittest.TestCase):
    def _runtime(self, stack, directory, runner):
        args = SimpleNamespace(
            config=Path("unused.json"), output_dir=directory / "candidate",
            sequence_length=2, repetitions=3, candidate_batch_size=1,
        )
        models = SimpleNamespace(tokenizer=object(), flux_transformer=object(), nvarc_encoder=runner)
        reserve = [torch.zeros(1)]
        for name, result in {
            "_load_config": _config(), "_environment_metadata": {},
            "prepare_device": torch.device("cpu"),
            "seed_process": None, "load_models": models,
            "build_optimizer": object(), "initialize_training_memory": {"optimizer_steps": 1},
            "allocate_buffer_reserve": (reserve, {"total_bytes": 4}),
            "make_probe_inputs": (torch.tensor([1, 2]), torch.tensor([0, 2])),
        }.items():
            stack.enter_context(patch.object(probe, name, return_value=result))
        for name in ("synchronize", "reset_peak_memory_stats"):
            stack.enter_context(patch.object(torch.cuda, name))
        stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(
            name="fake", total_memory=1000, major=9, minor=0,
        )))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(patch.object(probe, "_memory", return_value={
            "allocated_bytes": 80, "reserved_bytes": 150,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
        }))
        return args

    def test_repeats_full_runner_and_releases_output_before_next_call(self):
        previous = []
        calls = []

        def runner(ids, offsets):
            self.assertTrue(all(reference() is None for reference in previous))
            calls.append((ids.tolist(), offsets.tolist()))
            state = torch.zeros(2, NVARC_CONTEXT_WIDTH, dtype=torch.bfloat16)
            previous.append(weakref.ref(state))
            return NvarcBatchOutput((state,))

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = self._runtime(stack, Path(directory), runner)
            result = probe.run_candidate(args)
            saved = json.loads((args.output_dir / "result.json").read_text())
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(calls), 3)
        self.assertEqual(saved["retained_reserve_bytes"], 4)
        self.assertEqual(saved["peak_allocated_bytes"], 100)
        self.assertEqual(saved["allocated_headroom_bytes"], 900)
        self.assertEqual(len(saved["repetitions"]), 3)
        self.assertTrue(all(reference() is None for reference in previous))
        self.assertEqual(saved["repetitions"][0]["output_bytes"], 2 * NVARC_CONTEXT_WIDTH * 2)

    def test_only_typed_probe_oom_is_a_capacity_failure_and_partial_rows_survive(self):
        for error, expected in (
            (torch.cuda.OutOfMemoryError("test OOM"), "oom"),
            (RuntimeError("CUDA out of memory"), "error"),
        ):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                calls = []

                def runner(*_args):
                    calls.append(1)
                    if len(calls) == 2:
                        raise error
                    return NvarcBatchOutput((torch.zeros(2, NVARC_CONTEXT_WIDTH, dtype=torch.bfloat16),))

                args = self._runtime(stack, Path(directory), runner)
                result = probe.run_candidate(args)
                saved = json.loads((args.output_dir / "result.json").read_text())
                self.assertEqual(result["status"], expected)
                self.assertEqual(len(saved["repetitions"]), 1)
                self.assertEqual(saved["phase"], "probe")
                self.assertEqual(len(calls), 2)
                self.assertEqual(saved["failure_memory"]["peak_allocated_bytes"], 100)

    def test_setup_oom_and_invalid_outputs_abort_instead_of_reducing_batch(self):
        for fail_setup in (True, False):
            with self.subTest(fail_setup=fail_setup), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                args = self._runtime(stack, Path(directory), lambda *_: NvarcBatchOutput(()))
                if fail_setup:
                    stack.enter_context(patch.object(
                        probe, "initialize_training_memory", side_effect=torch.cuda.OutOfMemoryError("setup"),
                    ))
                result = probe.run_candidate(args)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["phase"], "setup" if fail_setup else "probe")
                self.assertEqual(result["repetitions"], [])
                self.assertEqual(result["failure_memory"]["peak_reserved_bytes"], 200)


class OrchestrationTests(unittest.TestCase):
    def _args(self, directory):
        config = directory / "config.json"
        config.write_text("{}")
        return SimpleNamespace(
            config=config, output_dir=directory / "results",
            sequence_length=8400, max_batch_size=4, repetitions=3,
        )

    def test_search_uses_separate_sequential_processes_and_saves_every_candidate(self):
        commands = []

        def worker(command, **kwargs):
            commands.append(command)
            self.assertFalse(kwargs["check"])
            self.assertEqual(command[1], "-B")
            destination = Path(command[command.index("--output-dir") + 1])
            size = int(command[command.index("--candidate-batch-size") + 1])
            destination.mkdir()
            status = "success" if size <= 2 else "oom"
            (destination / "result.json").write_text(json.dumps({
                "status": status, "batch_size": size,
                "repetitions": [{"seconds": 1}] * 3 if status == "success" else [],
            }))
            kwargs["stdout"].write("worker log\n")
            return SimpleNamespace(returncode=0 if status == "success" else probe.OOM_EXIT_CODE)

        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            persisted_counts = []

            def persist():
                rows = json.loads((args.output_dir / "candidates.json").read_text())
                persisted_counts.append(len(rows))
                self.assertTrue(Path(rows[-1]["result_path"]).is_file())

            with (
                patch.object(probe, "_load_config", return_value=_config()),
                patch.object(probe, "_source_metadata", return_value={}),
                patch.object(probe.subprocess, "run", side_effect=worker),
            ):
                summary = probe.run_search(args, on_candidate_end=persist)
            self.assertEqual(summary["largest_tested_success"], 2)
            self.assertEqual(summary["smallest_tested_oom"], 3)
            self.assertEqual(len(commands), 4)
            self.assertEqual(persisted_counts, [1, 2, 3, 4])
            self.assertEqual(len(list(args.output_dir.glob("batch-*/result.json"))), 4)
            self.assertEqual(len(list(args.output_dir.glob("batch-*.log"))), 4)
            self.assertEqual(len(json.loads((args.output_dir / "candidates.json").read_text())), 4)
            self.assertEqual(json.loads((args.output_dir / "manifest.json").read_text())["status"], "complete")

    def test_missing_or_incomplete_worker_result_is_not_an_oom(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as directory:
                args = self._args(Path(directory))

                def worker(command, **_kwargs):
                    if incomplete:
                        destination = Path(command[command.index("--output-dir") + 1])
                        destination.mkdir()
                        (destination / "result.json").write_text(json.dumps({
                            "status": "success", "batch_size": 1, "repetitions": [{}],
                        }))
                    return SimpleNamespace(returncode=0 if incomplete else -9)

                with (
                    patch.object(probe, "_load_config", return_value=_config()),
                    patch.object(probe, "_source_metadata", return_value={}),
                    patch.object(probe.subprocess, "run", side_effect=worker) as run,
                ):
                    persisted = []
                    summary = probe.run_search(args, on_candidate_end=lambda: persisted.append(
                        json.loads((args.output_dir / "candidates.json").read_text())[-1]["status"]
                    ))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(persisted, ["error"])
                self.assertEqual(summary["status"], "error")
                self.assertIsNone(summary["smallest_tested_oom"])
                self.assertEqual(json.loads((args.output_dir / "summary.json").read_text())["status"], "error")

    def test_remote_provenance_is_verified_without_querying_git_in_the_container(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            args.max_batch_size = 1
            metadata = {
                "git_commit": "local-commit", "git_status": " M local-file",
                "source_sha256": {"source.py": "hash"},
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "modal_resources": {"gpu": "H200"},
            }

            def worker(command, **_kwargs):
                destination = Path(command[command.index("--output-dir") + 1])
                destination.mkdir()
                (destination / "result.json").write_text(json.dumps({
                    "status": "success", "batch_size": 1, "repetitions": [{}, {}, {}],
                }))
                return SimpleNamespace(returncode=0)

            with (
                patch.object(probe, "_load_config", return_value=_config()),
                patch.object(probe, "_source_hashes", return_value=metadata["source_sha256"]),
                patch.object(probe, "_source_metadata", side_effect=AssertionError("no remote Git")),
                patch.object(probe.subprocess, "run", side_effect=worker),
            ):
                result = probe.run_search(args, metadata=metadata)
            self.assertEqual(result["status"], "complete")
            manifest = json.loads((args.output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["git_commit"], "local-commit")
            self.assertEqual(manifest["git_status"], " M local-file")
            self.assertEqual(manifest["launch_metadata"], metadata)

    def test_changed_upload_aborts_before_workers(self):
        for field in ("source_sha256", "config_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self._args(Path(directory))
                metadata = {
                    "git_commit": "local", "git_status": "", "source_sha256": {},
                    "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                }
                metadata[field] = "mismatch"
                with (
                    patch.object(probe, "_load_config", return_value=_config()),
                    patch.object(probe, "_source_hashes", return_value={}),
                    patch.object(probe.subprocess, "run") as run,
                ):
                    with self.assertRaisesRegex(ValueError, "differs from the launch metadata"):
                        probe.run_search(args, metadata=metadata)
                run.assert_not_called()
                self.assertFalse(args.output_dir.exists())

    def test_missing_checkpoints_and_existing_output_fail_before_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            config = _config()
            config = replace(config, models=replace(
                config.models, nvarc_checkpoint=root / "missing", flux_checkpoint=root / "missing2",
            ))
            with (
                patch.object(probe, "load_training_config", return_value=config),
                patch.object(probe.subprocess, "run") as run,
            ):
                with self.assertRaises(FileNotFoundError):
                    probe.run_search(args)
            run.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            args.output_dir.mkdir()
            with (
                patch.object(probe, "_load_config", return_value=_config()),
                patch.object(probe.subprocess, "run") as run,
            ):
                with self.assertRaises(FileExistsError):
                    probe.run_search(args)
            run.assert_not_called()

    def test_cli_defaults_and_invalid_arguments(self):
        parser = probe.build_parser()
        args = parser.parse_args(["--config", "b3.json"])
        self.assertEqual((args.sequence_length, args.max_batch_size, args.repetitions), (8400, 64, 3))
        for arguments in (
            ["--sequence-length", "8192"], ["--max-batch-size", "0"], ["--repetitions", "0"],
        ):
            with self.subTest(arguments=arguments), patch("sys.stderr"), self.assertRaises(SystemExit):
                parser.parse_args(["--config", "b3.json", *arguments])


if __name__ == "__main__":
    unittest.main()
