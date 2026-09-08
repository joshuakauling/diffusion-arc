"""CPU checks of masked alignment comparisons, measurement isolation, and failures."""

from contextlib import ExitStack
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
from diffusion_arc.training.nvarc_schedule import NvarcBatchShape, NvarcSchedulingPolicy
from scripts.diagnostics import probe_nvarc_batch_shapes as probe
from scripts.diagnostics._nvarc_tokenizer import make_nvarc_tokenizer


CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/nvarc_batch_shapes.json"


def configuration():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


class TinyNvarc(nn.Module):
    """Small masked SDPA model with the production hidden-state return contract."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(32, 8)
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *, input_ids, attention_mask, **_kwargs):
        hidden = self.embeddings(input_ids)[:, None]
        length = input_ids.shape[-1]
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        mask = attention_mask[:, None, None, :] & causal[None, None]
        output = F.scaled_dot_product_attention(hidden, hidden, hidden, attn_mask=mask)[:, 0]
        return SimpleNamespace(hidden_states=(output,) * 28)


class ConfigAndPlanTests(unittest.TestCase):
    def test_actual_catalog_produces_72_shapes_with_fixed_masked_inputs(self):
        policy, training = configuration()
        plan = probe.build_plan(policy, training)
        self.assertEqual(plan, probe.build_plan(policy, training))
        self.assertEqual(len(plan), 20)
        self.assertEqual(sum(len(g["padded_lengths"]) for g in plan), 72)
        keys = {(g["batch_size"], length) for g in plan for length in g["padded_lengths"]}
        self.assertEqual(len(keys), 72)
        lookup = {(g["batch_size"], g["baseline_tokens"]): g for g in plan}
        self.assertEqual(lookup[2, 3022]["padded_lengths"], [3022, 3024, 3040, 3072])
        self.assertEqual(lookup[4, 1083]["padded_lengths"], [1083, 1088, 1152])
        self.assertEqual(lookup[2, 8400]["padded_lengths"], [8400])
        self.assertEqual(lookup[2, 8400]["excluded_above_limit"], [8416, 8448])
        for group in plan:
            self.assertTrue(all(0 < length < group["baseline_tokens"] for length in group["real_lengths"]))
            for order in group["round_orders"]:
                self.assertEqual(sorted(order), group["padded_lengths"])

    def test_invalid_configs_fail_before_checkpoint_access(self):
        for field, value in (
            ("schema_version", True), ("warmup_calls", 0), ("measured_calls", True),
            ("rounds", -1), ("max_duration_seconds", 1.5),
            ("alignment_multiples", []), ("alignment_multiples", [8, 8]),
            ("alignment_multiples", [8, 16]), ("alignment_multiples", [True]),
            ("profile_groups", None), ("profile_groups", [[2, True]]),
            ("profile_groups", [[2, 3022], [2, 3022]]),
            ("training_config", None), ("unexpected", 1),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                raw = json.loads(CONFIG.read_text())
                raw[field] = value
                path = Path(directory) / "probe.json"
                path.write_text(json.dumps(raw))
                with patch.object(probe, "load_single_gpu_training_config") as loader:
                    with self.assertRaises(ValueError):
                        probe.load_probe_config(path)
                    loader.assert_not_called()

    def test_incompatible_training_and_missing_checkpoint_fail(self):
        _, training = configuration()
        for changed in (
            replace(training, nvarc_scheduling=None),
            replace(training, models=replace(training.models, nvarc_dtype="float32")),
            replace(training, nvarc_scheduling=NvarcSchedulingPolicy((NvarcBatchShape(2, 2),), 0)),
        ):
            with patch.object(probe, "load_single_gpu_training_config", return_value=changed):
                with self.assertRaises(ValueError):
                    probe.load_probe_config(CONFIG, require_checkpoints=False)
        with tempfile.TemporaryDirectory() as directory:
            changed = replace(training, models=replace(
                training.models, nvarc_checkpoint=Path(directory) / "missing",
            ))
            with patch.object(probe, "load_single_gpu_training_config", return_value=changed):
                with self.assertRaises(FileNotFoundError):
                    probe.load_probe_config(CONFIG)

    def test_dry_run_never_loads_models_or_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            probe, "prepare_runtime", side_effect=AssertionError("must not load"),
        ), patch("sys.stdout", new_callable=io.StringIO) as output:
            destination = Path(directory) / "absent"
            self.assertEqual(probe.main(["--config", str(CONFIG), "--dry-run",
                                         "--output-dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
            self.assertEqual(json.loads(output.getvalue())["unique_physical_shapes"], 72)


class OutputComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.templates = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])] * 2
        self.group = {"real_lengths": [7, 6]}
        self.ids, self.offsets = probe.build_group_inputs(self.templates, self.group)

    def test_unmodified_runner_outputs_across_aligned_lengths(self):
        runner = NvarcBatchRunner(TinyNvarc())
        original = F.scaled_dot_product_attention
        forward = runner.model.forward

        def unchanged_forward(**kwargs):
            self.assertIs(F.scaled_dot_product_attention, original)
            self.assertFalse(runner.model._forward_pre_hooks)
            return forward(**kwargs)

        reference = None
        for length in (8, 16, 32):
            with patch.object(runner.model, "forward", new=unchanged_forward):
                states = probe.collect_comparison_outputs(
                    runner, self.ids, self.offsets, padded_length=length,
                )
            self.assertEqual([list(s.shape) for s in states], [[7, 24], [6, 24]])
            if reference is None:
                reference = states
            for actual, expected in zip(states, reference):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            self.assertIs(F.scaled_dot_product_attention, original)
            self.assertFalse(runner.model._forward_pre_hooks)

    def test_real_qwen_outputs_agree_across_aligned_lengths(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=4, max_position_embeddings=32)
        config._attn_implementation = "sdpa"
        for dtype in (torch.float32, torch.bfloat16):
            runner = NvarcBatchRunner(Qwen3ForCausalLM(config).to(dtype=dtype))
            reference = None
            for length in (8, 16):
                with self.subTest(dtype=dtype, length=length):
                    states = probe.collect_comparison_outputs(
                        runner, self.ids, self.offsets, padded_length=length,
                    )
                    if reference is None:
                        reference = states
                    for actual, expected in zip(states, reference):
                        tolerance = 3e-6 if dtype == torch.float32 else 0.01
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    self.assertFalse(runner.model.training)
                    self.assertTrue(all(not p.requires_grad for p in runner.parameters()))


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.policy = replace(configuration()[0], warmup_calls=1, measured_calls=2, rounds=2)
        self.group = {"batch_size": 2, "baseline_tokens": 8, "real_lengths": [7, 6],
                      "padded_lengths": [8, 16], "round_orders": [[16, 8], [8, 16]],
                      "excluded_above_limit": [], "profile": True}
        self.runner = NvarcBatchRunner(TinyNvarc())
        self.templates = [torch.arange(1, 11)] * 2

    def test_interleaved_measurements_use_unmodified_runner_and_separate_profiles(self):
        seen = []
        original = F.scaled_dot_product_attention

        def measure(runner, ids, offsets, *, padded_length, **_kwargs):
            self.assertIs(F.scaled_dot_product_attention, original)
            self.assertFalse(runner.model._forward_pre_hooks)
            self.assertEqual(offsets.tolist(), [0, 7, 13])
            seen.append(padded_length)
            return {"wall_ms": 4 if padded_length == 8 else 3, "gpu_span_ms": 2,
                    "peak_allocated_bytes": 100, "peak_reserved_bytes": 150}

        def profile(*_args, **_kwargs):
            self.assertEqual(len(seen), 12)
            return {"trace": "example.json", "sdpa_operators": {"sdpa": 1}}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            torch.cuda, "Event", return_value=object(),
        ), patch.object(probe, "measure_call", side_effect=measure), patch.object(
            probe, "profile_call", side_effect=profile,
        ) as profiler:
            probe.run_group(self.policy, self.runner, self.templates, self.group,
                            device="cpu", directory=Path(directory), persist=lambda: None)
        self.assertEqual(seen, [16] * 3 + [8] * 6 + [16] * 3)
        self.assertEqual(profiler.call_count, 2)
        self.assertEqual(self.group["status"], "complete")
        aligned = self.group["cases"][1]
        self.assertEqual(aligned["median_paired_wall_ratio"], 0.75)
        self.assertEqual(aligned["wall_ms"]["round_medians"], [3, 3])
        self.assertEqual(len(aligned["calls"]), 6)
        self.assertFalse(aligned["numerical_difference"]["agreement_threshold_applied"])
        expected = {k: v for k, v in self.group.items() if k in (
            "batch_size", "baseline_tokens", "real_lengths", "padded_lengths",
            "round_orders", "excluded_above_limit", "profile",
        )}
        result = {"status": "complete", "groups": [self.group]}
        self.assertTrue(probe.validate_completion(result, self.policy, [expected]))
        aligned["calls"].pop()
        self.assertFalse(probe.validate_completion(result, self.policy, [expected]))

    def test_oom_preserves_completed_calls_and_never_claims_a_completed_group(self):
        record = {"wall_ms": 1, "gpu_span_ms": 1, "peak_allocated_bytes": 100,
                  "peak_reserved_bytes": 150}
        for error, status in ((torch.cuda.OutOfMemoryError("probe"), "oom"),
                              (RuntimeError("out of memory"), "error")):
            group = {k: v for k, v in self.group.items() if k != "cases"}
            with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.cuda, "Event", return_value=object(),
            ), patch.object(probe, "measure_call", side_effect=[record, error]), self.assertRaises(type(error)):
                probe.run_group(self.policy, self.runner, self.templates, group,
                                device="cpu", directory=Path(directory), persist=lambda: None)
            self.assertEqual(group["status"], status)
            self.assertEqual(group["cases"][1]["status"], status)
            self.assertEqual(len(group["cases"][1]["calls"]), 1)
            self.assertFalse(any(c["status"] == "success" for c in group["cases"]))

    def test_whole_runner_timing_contains_cpu_inputs_and_releases_outputs(self):
        ids, offsets = probe.build_group_inputs(self.templates, self.group)
        events = [Mock(), Mock()]
        events[0].elapsed_time.return_value = 2.5
        with patch.object(torch.cuda, "synchronize"), patch.object(
            torch.cuda, "reset_peak_memory_stats",
        ), patch.object(probe, "_memory", return_value={"peak_allocated_bytes": 100}):
            row = probe.measure_call(self.runner, ids, offsets, padded_length=8,
                                     device="cpu", events=events)
        events[0].record.assert_called_once()
        events[1].record.assert_called_once()
        events[1].synchronize.assert_called_once()
        self.assertEqual(row["gpu_span_ms"], 2.5)
        self.assertGreater(row["wall_ms"], 0)
        self.assertEqual(ids.device.type, "cpu")


class RuntimeTests(unittest.TestCase):
    def test_real_helpers_retain_lr_zero_optimizer_and_gradient_storage(self):
        class Nvarc(nn.Module):
            def forward(self, ids, offsets):
                from diffusion_arc.model.contracts import NvarcBatchOutput
                return NvarcBatchOutput((torch.ones(len(ids), 4),))

        class Vae(nn.Module):
            def forward(self, images):
                return torch.ones(len(images), 900, 128)

        class Flux(nn.Module):
            gradient_checkpointing = False

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.5))

            def forward(self, *, hidden_states, **_kwargs):
                return SimpleNamespace(sample=hidden_states * self.weight)

        _, training = configuration()
        models = SimpleNamespace(
            tokenizer=make_nvarc_tokenizer(model_max_length=10000),
            nvarc_runner=Nvarc(), vae_encoder=Vae(), transformer=Flux(), flux_compute_dtype=torch.bfloat16,
        )
        reserve = [torch.ones(1)]
        with ExitStack() as stack:
            for name, value in (("prepare_single_gpu_device", torch.device("cpu")),
                                ("load_single_gpu_models", models), ("_memory", {}),
                                ("allocate_buffer_reserve", (reserve, {"total_bytes": 4}))):
                stack.enter_context(patch.object(probe, name, return_value=value))
            for name in ("synchronize", "empty_cache"):
                stack.enter_context(patch.object(torch.cuda, name))
            _, _, optimizer, returned, templates, metadata = probe.prepare_runtime(training)
        self.assertIs(returned, reserve)
        self.assertEqual(models.transformer.weight.item(), 0.5)
        self.assertIsNotNone(models.transformer.weight.grad)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        self.assertEqual(metadata["initialization"]["adamw_moment_bytes"], 8)
        self.assertEqual(len(templates), 8)
        self.assertTrue(all(len(t) == 8400 for t in templates))
        self.assertFalse(torch.equal(templates[0], templates[1]))


class SupervisorTests(unittest.TestCase):
    def test_one_runtime_is_reused_for_all_groups(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(probe, "load_probe_config", return_value=configuration()))
            stack.enter_context(patch.object(probe, "_environment_metadata", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                name="fake", total_memory=1000, major=9, minor=0,
            )))
            stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
            runner = object()
            setup = stack.enter_context(patch.object(probe, "prepare_runtime", return_value=(
                "cpu", SimpleNamespace(nvarc_runner=runner), object(), [object()], [], {},
            )))
            group = stack.enter_context(patch.object(probe, "run_group"))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            result = probe.run_worker(SimpleNamespace(config=CONFIG, output_dir=Path(directory)))
        setup.assert_called_once()
        self.assertEqual(group.call_count, 20)
        self.assertTrue(all(call.args[1] is runner for call in group.call_args_list))
        self.assertEqual(result["status"], "complete")

    def test_setup_oom_is_an_error_without_shape_capacity_claims(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            probe, "load_probe_config", return_value=configuration(),
        ), patch.object(probe, "_environment_metadata", return_value={}), patch.object(
            probe, "prepare_runtime", side_effect=torch.cuda.OutOfMemoryError("setup"),
        ):
            result = probe.run_worker(SimpleNamespace(config=CONFIG, output_dir=Path(directory)))
            saved = json.loads((Path(directory) / "result.json").read_text())
        self.assertEqual(result["status"], "error")
        self.assertEqual(saved["phase"], "setup")
        self.assertTrue(all("cases" not in group for group in saved["groups"]))

    def test_timeout_and_incomplete_worker_preserve_partial_records(self):
        for timeout in (True, False):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "results"

                def child(_command, *, on_progress, **_kwargs):
                    probe._write_json(destination / "result.json", {
                        "status": "running", "groups": [{"cases": [{"status": "running", "calls": [1]}]}],
                    })
                    on_progress()
                    return (-9 if timeout else 0), timeout

                persisted = []

                def commit():
                    summary = destination / "summary.json"
                    persisted.append(json.loads(summary.read_text()) if summary.is_file() else None)

                with patch.object(probe, "load_probe_config", return_value=configuration()), patch.object(
                    probe, "source_metadata", return_value={},
                ), patch.object(probe, "launch_worker", side_effect=child):
                    result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination),
                                             on_progress=commit)
                saved = json.loads((destination / "summary.json").read_text())
                self.assertEqual(result["status"], "timeout" if timeout else "error")
                self.assertEqual(saved["groups"][0]["cases"][0]["calls"], [1])
                self.assertFalse(saved["training_policy_validated"])
                report = (destination / "report.md").read_text()
                self.assertIn(f"**{result['status']}**", report)
                self.assertIn("| running | — | — | — | — |", report)
                self.assertEqual(len(persisted), 3)
                self.assertIsNone(persisted[0])
                self.assertEqual(persisted[1]["status"], "running")
                self.assertEqual(persisted[2]["status"], result["status"])

    def test_remote_metadata_is_verified_without_remote_git(self):
        policy, _training = configuration()
        hashes = probe.source_hashes()
        self.assertIn("scripts/cloud/probe_nvarc_batch_shapes_modal.py", hashes)
        metadata = {"git_commit": "local-head", "git_status": " M local-work",
                    "source_sha256": hashes, "config_sha256": probe.config_hashes(CONFIG, policy),
                    "command": ["modal", "run", "launcher.py"], "modal_version": "test"}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            probe, "load_probe_config", return_value=configuration(),
        ), patch.object(probe, "source_metadata", side_effect=AssertionError("no remote Git")), patch.object(
            probe, "launch_worker", return_value=(1, False),
        ) as launch:
            destination = Path(directory) / "matching"
            probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), metadata=metadata)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["git_commit"], "local-head")
            self.assertEqual(manifest["launch_metadata"], metadata)
            launch.reset_mock()
            for field in ("source_sha256", "config_sha256"):
                invalid = {**metadata, field: {}}
                destination = Path(directory) / field
                with self.assertRaisesRegex(ValueError, "differs from launch metadata"):
                    probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), metadata=invalid)
                self.assertFalse(destination.exists())
            launch.assert_not_called()

    def test_progress_commit_error_keeps_last_worker_record_in_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results"

            def child(_command, *, on_progress, **_kwargs):
                probe._write_json(destination / "result.json", {
                    "status": "running", "groups": [{"cases": [{"status": "running", "calls": [1]}]}],
                })
                on_progress()

            commit = Mock(side_effect=[None, RuntimeError("commit failed"), None])
            with patch.object(probe, "load_probe_config", return_value=configuration()), patch.object(
                probe, "source_metadata", return_value={},
            ), patch.object(probe, "launch_worker", side_effect=child), self.assertRaisesRegex(RuntimeError, "commit failed"):
                probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), on_progress=commit)
            saved = json.loads((destination / "summary.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["groups"][0]["cases"][0]["calls"], [1])


class WorkerTransportTests(unittest.TestCase):
    def test_streamed_progress_is_visible_before_worker_exit_and_logs_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            progress = Mock(side_effect=lambda: self.assertIn(probe.PROGRESS_MARKER, log.read_text()))
            command = [sys.executable, "-B", "-u", "-c",
                       f"print('diagnostic'); print('{probe.PROGRESS_MARKER} 2x3022')"]
            code, timed_out = probe.launch_worker(command, log_path=log,
                                                   timeout_seconds=10, on_progress=progress)
            self.assertEqual(code, 0)
            self.assertFalse(timed_out)
            progress.assert_called_once()
            self.assertIn("diagnostic", log.read_text())

    def test_silent_worker_is_killed_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            code, timed_out = probe.launch_worker(
                [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                log_path=Path(directory) / "worker.log", timeout_seconds=0.1,
                on_progress=Mock(side_effect=AssertionError("no progress expected")),
            )
        self.assertTrue(timed_out)
        self.assertNotEqual(code, 0)

    def test_progress_failure_kills_the_child_and_propagates(self):
        real_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as directory, patch.object(subprocess, "Popen", side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                probe.launch_worker(
                    [sys.executable, "-B", "-u", "-c",
                     f"import time; print('{probe.PROGRESS_MARKER} test'); time.sleep(30)"],
                    log_path=Path(directory) / "worker.log", timeout_seconds=10,
                    on_progress=Mock(side_effect=RuntimeError("commit failed")),
                )
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
