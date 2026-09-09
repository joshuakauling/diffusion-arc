"""CPU-only validation of FLUX shape probing, accumulation, and bounded workers."""

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from diffusion_arc.model.contracts import NvarcBatchOutput
from scripts.diagnostics._nvarc_tokenizer import make_nvarc_tokenizer
from scripts.diagnostics import probe_flux_batch_shapes as probe


CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/flux_batch_shapes.json"


def configuration():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def success(length, size, policy):
    return {
        "status": "success", "text_length": length, "batch_size": size,
        "calls": ([{"phase": "warmup"}] * policy.warmup_calls
                  + [{"phase": "measured"}] * policy.measured_calls),
    }


class ConfigTests(unittest.TestCase):
    def test_checked_in_matrix_and_existing_training_policy(self):
        policy, training = configuration()
        self.assertEqual(policy.text_bucket_boundaries, (
            487, 778, 1090, 1348, 1578, 1818, 2126, 2422,
            2749, 2990, 3325, 3714, 4210, 4837, 6100, 8400,
        ))
        self.assertEqual(policy.microbatch_sizes, (1, 2, 4, 8))
        self.assertEqual((policy.warmup_calls, policy.measured_calls, policy.max_duration_seconds), (2, 5, 1200))
        self.assertEqual(training.loop.optimizer_batch_size, 48)
        self.assertFalse(hasattr(training, "nvarc_scheduling"))

    def test_bad_user_values_fail_before_checkpoint_access(self):
        for field, value in (
            ("schema_version", True), ("warmup_calls", 0), ("measured_calls", True),
            ("max_duration_seconds", -1), ("text_bucket_boundaries", [487, 487]),
            ("text_bucket_boundaries", [1, 8400]), ("text_bucket_boundaries", [8401]),
            ("microbatch_sizes", [2, 4]), ("microbatch_sizes", [1, 3]),
            ("microbatch_sizes", [1, True]), ("training_config", None), ("unexpected", 1),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                raw = json.loads(CONFIG.read_text())
                raw[field] = value
                path = Path(directory) / "probe.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path)

    def test_incompatible_training_and_missing_checkpoints_are_rejected(self):
        _policy, training = configuration()
        with patch.object(probe, "load_single_gpu_training_config", return_value=replace(
            training, models=replace(training.models, flux_parameter_dtype="bfloat16"),
        )), self.assertRaisesRegex(ValueError, "precision"):
            probe.load_probe_config(CONFIG, require_checkpoints=False)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            probe, "load_single_gpu_training_config", return_value=replace(
                training, models=replace(training.models, nvarc_checkpoint=Path(directory) / "missing"),
            ),
        ), self.assertRaises(FileNotFoundError):
            probe.load_probe_config(CONFIG)


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.training = configuration()
        self.device = torch.device("cpu")
        self.reserve = [torch.arange(48, dtype=torch.bfloat16).reshape(12, 4), torch.ones(1, 900, 128)]
        self.model = nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(0.5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0, foreach=False)
        self.model.weight.grad = torch.ones_like(self.model.weight)
        self.optimizer.step()
        self.grad_pointer = self.model.weight.grad.data_ptr()
        self.models = SimpleNamespace(transformer=self.model, flux_compute_dtype=torch.bfloat16)

    def _cuda(self, stack):
        events = []

        def event(**_kwargs):
            position = len(events)
            result = SimpleNamespace(position=position, record=Mock())
            result.elapsed_time = lambda other: (other.position - position) * 10
            events.append(result)
            return result

        for name in ("synchronize", "reset_peak_memory_stats"):
            stack.enter_context(patch.object(torch.cuda, name))
        stack.enter_context(patch.object(torch.cuda, "Event", side_effect=event))
        stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=1000)))
        stack.enter_context(patch.object(probe, "_memory", return_value={
            "allocated_bytes": 80, "reserved_bytes": 100,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 150,
        }))

    def test_real_padding_masks_and_independent_microbatch_storage(self):
        batch = probe.build_probe_microbatch(self.reserve, text_length=12, batch_size=4)
        self.assertEqual(batch.text_inputs.encoder_hidden_states.shape, (4, 12, 4))
        self.assertEqual(batch.text_inputs.text_attention_mask.sum(dim=1).tolist(), [11, 10, 9, 8])
        self.assertEqual(batch.clean_latents.shape, (4, 900, 128))
        self.assertNotEqual(batch.text_inputs.encoder_hidden_states.data_ptr(), self.reserve[0].data_ptr())
        for row, prefix in enumerate((11, 10, 9, 8)):
            torch.testing.assert_close(batch.text_inputs.encoder_hidden_states[row, :prefix], self.reserve[0][:prefix])
            self.assertTrue(bool((batch.text_inputs.encoder_hidden_states[row, prefix:] == 0).all()))
        singleton = probe.build_probe_microbatch(self.reserve, text_length=12, batch_size=1)
        self.assertFalse(singleton.text_inputs.text_attention_mask[0, -1])

    def test_two_warmups_five_measurements_keep_gradients_and_weight_by_48(self):
        seen = []

        def loss(model, batch, **kwargs):
            self.assertEqual(model.weight.grad.data_ptr(), self.grad_pointer)
            self.assertEqual(model.weight.grad.item(), 0)
            self.assertEqual(self.optimizer.state[model.weight]["exp_avg"].dtype, torch.float32)
            self.assertEqual(batch.target_count, 2)
            self.assertEqual(kwargs["compute_dtype"], torch.bfloat16)
            seen.append(1)
            return (model.weight.square() * 3).sum()

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self._cuda(stack)
            stack.enter_context(patch.object(probe, "compute_flux_microbatch_loss", side_effect=loss))
            record = probe.run_shape(
                self.policy, self.training, self.models, self.optimizer, self.reserve,
                device=self.device, text_length=12, batch_size=2, result_path=Path(directory) / "shape.json",
            )
            saved = json.loads((Path(directory) / "shape.json").read_text())
        self.assertEqual(len(seen), 7)
        self.assertEqual(record["status"], "success")
        self.assertEqual([row["phase"] for row in record["calls"]], ["warmup"] * 2 + ["measured"] * 5)
        self.assertEqual(record["first_call"], record["calls"][0])
        self.assertAlmostEqual(record["seconds_per_target"], 0.01)
        self.assertEqual(record["allocated_headroom_bytes"], 900)
        self.assertEqual(saved["peak_reserved_bytes"], 150)
        self.assertEqual(self.model.weight.item(), 0.5)
        self.assertAlmostEqual(self.model.weight.grad.item(), 3 * 2 / 48)
        self.assertEqual(self.model.weight.grad.data_ptr(), self.grad_pointer)
        self.assertEqual(self.optimizer.state[self.model.weight]["step"].item(), 2)

    def test_typed_oom_is_distinct_from_errors_and_retains_completed_calls(self):
        for error, status in ((torch.cuda.OutOfMemoryError("test"), "oom"), (RuntimeError("out of memory"), "error")):
            calls = []

            def loss(*_args, **_kwargs):
                calls.append(1)
                if len(calls) == 3:
                    raise error
                return self.model.weight.square().sum()

            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                self._cuda(stack)
                stack.enter_context(patch.object(probe, "compute_flux_microbatch_loss", side_effect=loss))
                result = probe.run_shape(
                    self.policy, self.training, self.models, self.optimizer, self.reserve,
                    device=self.device, text_length=12, batch_size=2, result_path=Path(directory) / "shape.json",
                )
                self.assertEqual(result["status"], status)
                self.assertEqual(len(result["calls"]), 2)
                self.assertEqual(result["phase"], "forward_backward")


class RuntimeTests(unittest.TestCase):
    def test_production_preparation_initializes_lr_zero_moments_gradients_and_templates(self):
        class Nvarc(nn.Module):
            def forward(self, ids, offsets):
                return NvarcBatchOutput(tuple(
                    torch.ones(int(end - start), 4, dtype=torch.bfloat16)
                    for start, end in zip(offsets[:-1], offsets[1:])
                ))

        class Vae(nn.Module):
            def forward(self, images):
                return torch.full((len(images), 900, 128), 0.25)

        class Flux(nn.Module):
            gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def forward(self, *, hidden_states, **_kwargs):
                return SimpleNamespace(sample=hidden_states * self.weight)

        _policy, training = configuration()
        models = SimpleNamespace(
            tokenizer=make_nvarc_tokenizer(model_max_length=10000),
            nvarc_runner=Nvarc(), vae_encoder=Vae(), transformer=Flux(), flux_compute_dtype=torch.bfloat16,
        )
        reserve = [torch.zeros(8400, 4, dtype=torch.bfloat16), torch.zeros(1, 900, 128)]
        with ExitStack() as stack:
            for name, value in (
                ("prepare_single_gpu_device", torch.device("cpu")), ("load_single_gpu_models", models),
                ("allocate_buffer_reserve", (reserve, {"total_bytes": 1})), ("_memory", {}),
            ):
                stack.enter_context(patch.object(probe, name, return_value=value))
            for name in ("synchronize", "empty_cache"):
                stack.enter_context(patch.object(torch.cuda, name))
            _device, _models, optimizer, returned_reserve, metadata = probe.prepare_runtime(training)
        self.assertIs(returned_reserve, reserve)
        self.assertTrue(bool((reserve[0] == 1).all()))
        self.assertTrue(bool((reserve[-1][0] == 0.25).all()))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        self.assertEqual(models.transformer.weight.item(), 0.5)
        self.assertIsNotNone(models.transformer.weight.grad)
        self.assertTrue(bool(torch.isfinite(models.transformer.weight.grad)))
        self.assertEqual(metadata["initialization"]["gradient_bytes"], 4)
        self.assertEqual(metadata["initialization"]["adamw_moment_bytes"], 8)
        self.assertEqual(metadata["initialization"]["learning_rate"], 0)


class WorkerTests(unittest.TestCase):
    def test_model_reuse_and_immediate_exit_after_oom(self):
        policy, training = configuration()
        policy = replace(policy, text_bucket_boundaries=(12, 24, 36), microbatch_sizes=(1, 2, 4))
        seen = []

        def shape(*_args, text_length, batch_size, **_kwargs):
            seen.append((text_length, batch_size))
            if (text_length, batch_size) == (24, 2):
                return {"text_length": text_length, "batch_size": batch_size, "status": "oom"}
            return success(text_length, batch_size, policy)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(probe, "load_probe_config", return_value=(policy, training)),
            patch.object(probe, "_environment_metadata", return_value={}),
            patch.object(probe, "prepare_runtime", return_value=("cpu", None, None, [], {})) as setup,
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="fake", total_memory=1000)),
            patch.object(probe, "run_shape", side_effect=shape),
        ):
            result = probe.run_worker(SimpleNamespace(
                config=CONFIG, output_dir=Path(directory) / "worker", worker_start_length_index=0,
            ))
        setup.assert_called_once()
        self.assertEqual(seen, [(12, 1), (12, 2), (12, 4), (24, 1), (24, 2)])
        self.assertEqual(result["next_length_index"], 2)
        self.assertEqual(result["cases"][-1]["status"], "skipped")
        self.assertTrue(probe._valid_worker_result(result, policy=policy, start=0, returncode=2))

    def test_setup_oom_does_not_become_a_shape_bound(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(probe, "load_probe_config", return_value=configuration()),
            patch.object(probe, "_environment_metadata", return_value={}),
            patch.object(probe, "prepare_runtime", side_effect=torch.cuda.OutOfMemoryError("setup")),
        ):
            result = probe.run_worker(SimpleNamespace(
                config=CONFIG, output_dir=Path(directory) / "worker", worker_start_length_index=0,
            ))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["phase"], "setup")
        self.assertEqual(result["cases"], [])


class SupervisorTests(unittest.TestCase):
    def test_restarts_only_after_oom_and_persists_each_shape(self):
        policy, training = configuration()
        policy = replace(policy, text_bucket_boundaries=(12, 24), microbatch_sizes=(1, 2, 4))
        starts = []
        progress = []

        def worker(command, *, on_progress, timeout_seconds, **_kwargs):
            start = int(command[command.index("--worker-start-length-index") + 1])
            starts.append(start)
            self.assertGreater(timeout_seconds, 0)
            self.assertLessEqual(timeout_seconds, 1200)
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir()
            if start == 0:
                cases = [success(12, 1, policy), {"text_length": 12, "batch_size": 2, "status": "oom"},
                         {"text_length": 12, "batch_size": 4, "status": "skipped"}]
                status, code = "oom", 2
            else:
                cases = [success(24, size, policy) for size in policy.microbatch_sizes]
                status, code = "complete", 0
            for count in range(1, len(cases) + 1):
                probe._write_json(destination / "result.json", {
                    "status": status, "cases": cases[:count], "next_length_index": start + 1,
                })
                on_progress()
            return code, False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(probe, "load_probe_config", return_value=(policy, training)),
            patch.object(probe, "source_metadata", return_value={}),
            patch.object(probe, "launch_worker", side_effect=worker),
        ):
            args = SimpleNamespace(config=CONFIG, output_dir=Path(directory) / "results")
            result = probe.run_probe(args, on_progress=lambda: progress.append(
                len(json.loads((args.output_dir / "summary.json").read_text())["cases"]),
            ))
        self.assertEqual(starts, [0, 1])
        self.assertEqual(progress, [1, 2, 3, 4, 5, 6, 6])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["largest_tested_success_by_length"], {"12": 1, "24": 4})
        self.assertFalse(result["training_policy_validated"])

    def test_timeout_and_missing_result_stop_without_retries(self):
        policy, training = configuration()
        for timeout, expected in ((True, "timeout"), (False, "error")):
            with (
                self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory,
                patch.object(probe, "load_probe_config", return_value=(policy, training)),
                patch.object(probe, "source_metadata", return_value={}),
                patch.object(probe, "launch_worker", return_value=(-9, timeout)) as worker,
            ):
                args = SimpleNamespace(config=CONFIG, output_dir=Path(directory) / "results")
                result = probe.run_probe(args)
                manifest = json.loads((args.output_dir / "manifest.json").read_text())
            worker.assert_called_once()
            self.assertEqual(result["status"], expected)
            self.assertEqual(manifest["status"], expected)
            self.assertEqual(result["cases"], [])

    def test_incomplete_success_and_skips_without_oom_are_rejected(self):
        policy, _training = configuration()
        policy = replace(policy, text_bucket_boundaries=(12,), microbatch_sizes=(1,))
        record = {"status": "complete", "next_length_index": 1, "cases": [success(12, 1, policy)]}
        self.assertTrue(probe._valid_worker_result(record, policy=policy, start=0, returncode=0))
        record["cases"][0]["calls"].pop()
        self.assertFalse(probe._valid_worker_result(record, policy=policy, start=0, returncode=0))
        record["cases"][0]["status"] = "skipped"
        self.assertFalse(probe._valid_worker_result(record, policy=policy, start=0, returncode=0))

    def test_changed_upload_fails_before_launch(self):
        policy, training = configuration()
        for field in ("source_sha256", "config_sha256"):
            metadata = {
                "source_sha256": probe.source_hashes(), "config_sha256": probe.config_hashes(CONFIG, policy),
            }
            metadata[field] = {}
            with (
                self.subTest(field=field), tempfile.TemporaryDirectory() as directory,
                patch.object(probe, "load_probe_config", return_value=(policy, training)),
                patch.object(probe, "source_metadata", side_effect=AssertionError("no remote Git")),
                patch.object(probe, "launch_worker") as worker,
            ):
                destination = Path(directory) / "results"
                with self.assertRaisesRegex(ValueError, "differs from launch"):
                    probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), metadata=metadata)
                self.assertFalse(destination.exists())
            worker.assert_not_called()

    def test_remote_sweep_verifies_real_hashes_and_preserves_local_identity_without_git(self):
        policy, training = configuration()
        policy = replace(policy, text_bucket_boundaries=(487,), microbatch_sizes=(1,))
        metadata = {
            "git_commit": "local-commit", "git_status": " M local-change",
            "source_sha256": probe.source_hashes(), "config_sha256": probe.config_hashes(CONFIG, policy),
        }

        def worker(command, **_kwargs):
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir()
            probe._write_json(destination / "result.json", {
                "status": "complete", "next_length_index": 1, "cases": [success(487, 1, policy)],
            })
            return 0, False

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(probe, "load_probe_config", return_value=(policy, training)),
            patch.object(probe, "source_metadata", side_effect=AssertionError("no remote Git")),
            patch.object(probe.subprocess, "run", side_effect=FileNotFoundError("git")) as git,
            patch.object(probe, "launch_worker", side_effect=worker),
        ):
            destination = Path(directory) / "results"
            result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), metadata=metadata)
            manifest = json.loads((destination / "manifest.json").read_text())
        git.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(manifest["git_commit"], "local-commit")
        self.assertEqual(manifest["git_status"], " M local-change")
        self.assertEqual(manifest["source_sha256"], metadata["source_sha256"])
        self.assertEqual(manifest["config_sha256"], metadata["config_sha256"])

    def test_actual_cpu_worker_is_killed_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            code, timed_out = probe.launch_worker(
                [sys.executable, "-B", "-u", "-c", "import time; print('FLUX_SHAPE_COMPLETE test'); time.sleep(10)"],
                log_path=Path(directory) / "worker.log", timeout_seconds=0.3, on_progress=lambda: None,
            )
        self.assertTrue(timed_out)
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
