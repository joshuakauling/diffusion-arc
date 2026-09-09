"""CPU-only checks of controlled FLUX alignment comparisons and their launcher protocol."""

from contextlib import ExitStack, redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from scripts.diagnostics import _flux_alignment as alignment
from scripts.diagnostics import probe_flux_batch_shapes as probe
from tests.diagnostics import test_probe_flux_batch_shapes as legacy_tests
from tests.training.test_flux_loss import _tiny_transformer


CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/flux_alignment.json"


def configuration():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def tiny_group():
    return {
        "batch_size": 2, "baseline_text_length": 10,
        "real_prefix_lengths": [9, 8], "flow_seed": 17,
        "padded_lengths": [10, 12, 16],
        "labels": {"10": ["original"], "12": ["joint_8"], "16": ["text_8"]},
        "round_orders": [[10, 12, 16], [12, 16, 10], [16, 10, 12]],
        "profile": False,
    }


def fake_call(value=10):
    return {**dict.fromkeys(alignment.TIMINGS, value), "loss": 1.0,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 120}


def completed_group(policy, plan):
    return {
        **copy.deepcopy(plan), "status": "complete", "profiles": [],
        "calls": [
            {"text_length": length, "round": index, "call": call,
             "phase": "warmup" if call < policy.warmup_calls else "measured",
             **fake_call(999 if call < policy.warmup_calls else 10 + index)}
            for index, order in enumerate(plan["round_orders"]) for length in order
            for call in range(policy.warmup_calls + policy.measured_calls)
        ],
    }


class ConfigTests(unittest.TestCase):
    def test_140_cases_fixed_inputs_deduplication_and_joint_offset(self):
        policy, training = configuration()
        groups = alignment.build_plan(policy, training)
        self.assertEqual(len(groups), 24)
        self.assertEqual(sum(len(g["padded_lengths"]) for g in groups), 140)
        self.assertEqual({(g["batch_size"], g["baseline_text_length"]) for g in groups},
                         {(s.batch_size, s.max_tokens) for s in training.flux_scheduling.shapes})
        example = next(g for g in groups if (g["batch_size"], g["baseline_text_length"]) == (1, 2422))
        self.assertEqual(example["padded_lengths"], [2422, 2424, 2428, 2432])
        self.assertIn("text_32", example["labels"]["2432"])
        self.assertIn("joint_32", example["labels"]["2428"])
        self.assertEqual((2428 + alignment.IMAGE_TOKENS) % 32, 0)
        for group in groups:
            self.assertEqual(len(group["real_prefix_lengths"]), group["batch_size"])
            self.assertLess(max(group["real_prefix_lengths"]), min(group["padded_lengths"]))
            self.assertLessEqual(max(group["padded_lengths"]), 8400)
            for order in group["round_orders"]:
                self.assertEqual(sorted(order), group["padded_lengths"])
            if len(group["padded_lengths"]) > 1:
                self.assertEqual(len({tuple(order) for order in group["round_orders"]}), 3)
        largest = next(g for g in groups if g["baseline_text_length"] == 8400)
        self.assertEqual(largest["padded_lengths"], [8400])
        self.assertTrue(all(c["reason"] == "text_limit" for c in largest["excluded"]))
        self.assertEqual(groups, alignment.build_plan(policy, training))

    def test_invalid_configs_fail_before_loading_models(self):
        for field, value in (
            ("rounds", True), ("warmup_calls", 0), ("measured_calls", -1),
            ("max_duration_seconds", 0), ("alignment_multiples", [8, 8]),
            ("alignment_multiples", [16]), ("alignment_multiples", []),
            ("profile_groups", [[1, 2422], [1, 2422]]), ("profile_groups", [[True, 2422]]),
            ("profile_groups", [[2]]), ("profile_groups", [[8, 8400]]),
            ("training_config", None), ("extra", 1),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                raw = json.loads(CONFIG.read_text())
                raw["training_config"] = str(configuration()[0].training_config)
                raw[field] = value
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)
        policy, training = configuration()
        with patch.object(probe, "load_single_gpu_training_config", return_value=replace(training, flux_scheduling=None)):
            with self.assertRaisesRegex(ValueError, "explicit FLUX"):
                probe.load_probe_config(CONFIG, require_checkpoints=False)

    def test_dry_run_needs_no_checkpoints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            destination = Path(directory) / "output"
            with patch.object(probe, "prepare_runtime") as runtime:
                self.assertEqual(probe.main(["--config", str(CONFIG), "--dry-run", "--output-dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
        runtime.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["candidate_cases"], 140)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.training = configuration()
        self.reserve = [torch.randn(10, 16, dtype=torch.bfloat16), torch.randn(1, 900, 128)]
        self.group = {**tiny_group(), "status": "running", "calls": [], "profiles": []}
        self.device = torch.device("cpu")

    def test_real_masks_fixed_flow_and_tiny_production_flux_gradients(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.manual_seed(17)
        model = _tiny_transformer()
        flow = alignment.make_flow_batch(self.reserve, self.group, device=self.device)
        repeated = alignment.make_flow_batch(self.reserve, self.group, device=self.device)
        for name in ("noisy_latents", "target_velocity", "timesteps"):
            torch.testing.assert_close(getattr(flow, name), getattr(repeated, name), rtol=0, atol=0)
        losses, gradients = [], []
        for length in self.group["padded_lengths"]:
            model.zero_grad(set_to_none=True)
            batch = probe.build_probe_microbatch(self.reserve, text_length=length, batch_size=2, prefix_lengths=[9, 8])
            self.assertEqual(batch.text_inputs.text_attention_mask.sum(1).tolist(), [9, 8])
            for row, prefix in enumerate((9, 8)):
                torch.testing.assert_close(batch.text_inputs.encoder_hidden_states[row, :prefix], self.reserve[0][:prefix])
                self.assertFalse(batch.text_inputs.text_attention_mask[row, prefix:].any())
            loss = probe.compute_flux_microbatch_loss(model, batch, compute_dtype=torch.bfloat16, flow_batch=flow)
            (loss * (2 / 48)).backward()
            losses.append(loss.item())
            gradients.append(torch.cat([p.grad.flatten().clone() for p in model.parameters() if p.grad is not None]))
        for loss, gradient in zip(losses[1:], gradients[1:]):
            self.assertAlmostEqual(loss, losses[0], delta=.002)
            self.assertLess(float((gradient - gradients[0]).norm() / gradients[0].norm()), .02)

    def test_measured_calls_reuse_flow_and_gradient_storage_without_optimizer_updates(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0, foreach=False)
        model.weight.grad = torch.zeros_like(model.weight)
        pointer = model.weight.grad.data_ptr()
        models = SimpleNamespace(transformer=model, flux_compute_dtype=torch.bfloat16)
        flows = []

        def loss(_model, batch, *, compute_dtype, flow_batch):
            self.assertEqual(batch.text_inputs.text_attention_mask.sum(1).tolist(), [9, 8])
            self.assertEqual(model.weight.grad.data_ptr(), pointer)
            self.assertEqual(model.weight.grad.item(), 0)
            flows.append(flow_batch)
            return model.weight.square().sum() * 3

        with ExitStack() as stack:
            legacy_tests.ShapeTests._cuda(self, stack)
            stack.enter_context(patch.object(probe, "compute_flux_microbatch_loss", side_effect=loss))
            step = stack.enter_context(patch.object(optimizer, "step"))
            alignment.run_group(
                self.policy, self.training, models, optimizer, self.reserve, self.group,
                device=self.device, state={}, persist=Mock(),
            )
        self.assertEqual(len(flows), 63)
        self.assertTrue(all(flow is flows[0] for flow in flows))
        self.assertEqual(model.weight.item(), .5)
        self.assertAlmostEqual(model.weight.grad.item(), 3 * 2 / 48)
        step.assert_not_called()
        self.assertEqual(self.group["status"], "complete")
        self.assertEqual(len([c for c in self.group["calls"] if c["phase"] == "measured"]), 45)

    def test_worker_preserves_partial_oom_without_retrying_neighboring_shapes(self):
        state = {"count": 0}

        def call(*args, **kwargs):
            state["count"] += 1
            kwargs["state"]["phase"] = "backward"
            if state["count"] == 4:
                raise torch.cuda.OutOfMemoryError("test allocation")
            return fake_call()

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(probe, "prepare_runtime", return_value=(self.device, None, None, self.reserve, {})))
            stack.enter_context(patch.object(probe, "_environment_metadata", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="cpu-test", total_memory=1000)))
            stack.enter_context(patch.object(alignment, "build_plan", return_value=[tiny_group()]))
            stack.enter_context(patch.object(alignment, "measure_call", side_effect=call))
            output_dir = Path(directory) / "worker"
            result = alignment.run_worker(SimpleNamespace(output_dir=output_dir), self.policy, self.training)
            saved = json.loads((output_dir / "result.json").read_text())
        self.assertEqual(result["status"], "oom")
        self.assertEqual(result, saved)
        self.assertEqual(len(result["groups"][0]["calls"]), 3)
        self.assertEqual(result["active_case"]["text_length"], 10)
        self.assertEqual(result["phase"], "backward")
        self.assertIn("test allocation", result["error"])
        self.assertEqual(state["count"], 4)

    def test_profiler_is_separate_from_cost_calls_and_does_not_retain_tensor_shapes(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        models = SimpleNamespace(transformer=model, flux_compute_dtype=torch.bfloat16)
        profiler = MagicMock()
        profiler.__enter__.return_value = profiler
        profiler.key_averages.return_value = [SimpleNamespace(key="aten::attention"), SimpleNamespace(key="aten::add")]
        profiler.export_chrome_trace.side_effect = lambda path: Path(path).write_text("test trace")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            legacy_tests.ShapeTests._cuda(self, stack)
            factory = stack.enter_context(patch.object(torch.profiler, "profile", return_value=profiler))
            warmup = stack.enter_context(patch.object(alignment, "measure_call", return_value=fake_call()))
            stack.enter_context(patch.object(probe, "compute_flux_microbatch_loss", side_effect=lambda *a, **k: model.weight.square().sum()))
            alignment.profile_group(
                self.training, models, optimizer, self.reserve, self.group,
                device=self.device, state={}, output_dir=Path(directory), persist=Mock(),
            )
            self.assertTrue(all((Path(directory) / entry["trace"]).is_file() for entry in self.group["profiles"]))
        self.assertEqual(warmup.call_count, 3)
        self.assertEqual(self.group["calls"], [])
        self.assertEqual([entry["text_length"] for entry in self.group["profiles"]], [10, 12, 16])
        self.assertTrue(all(entry["attention_operators"] == ["aten::attention"] for entry in self.group["profiles"]))
        self.assertFalse(factory.call_args.kwargs["record_shapes"])
        self.assertFalse(factory.call_args.kwargs["profile_memory"])

    def test_worker_profiles_only_after_all_comparisons_and_stops_on_setup_failure(self):
        calls = []
        plans = [{**tiny_group(), "profile": True}, {**tiny_group(), "batch_size": 1}]

        def group(*args, **kwargs):
            calls.append("measure")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(probe, "prepare_runtime", return_value=(self.device, None, None, self.reserve, {})))
            stack.enter_context(patch.object(probe, "_environment_metadata", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="cpu-test", total_memory=1000)))
            stack.enter_context(patch.object(alignment, "build_plan", return_value=plans))
            stack.enter_context(patch.object(alignment, "run_group", side_effect=group))
            stack.enter_context(patch.object(alignment, "profile_group", side_effect=lambda *a, **k: calls.append("profile")))
            result = alignment.run_worker(SimpleNamespace(output_dir=Path(directory) / "success"), self.policy, self.training)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(calls, ["measure", "measure", "profile"])
            with patch.object(probe, "prepare_runtime", side_effect=torch.cuda.OutOfMemoryError("setup")):
                result = alignment.run_worker(SimpleNamespace(output_dir=Path(directory) / "failure"), self.policy, self.training)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["phase"], "setup")
        self.assertEqual(result["groups"], [])


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.training = configuration()
        self.plan = tiny_group()
        self.group = completed_group(self.policy, self.plan)

    def test_statistics_exclude_warmups_and_preserve_rounds(self):
        rows = alignment.summarize_groups([self.group], self.policy)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["timings"]["forward_backward_ms"]["median"], 11)
            self.assertEqual(row["round_forward_backward_ms"], {"0": 10, "1": 11, "2": 12})
            self.assertEqual(row["speedup_vs_measured_baseline"], 1)
            self.assertEqual(row["measured_calls"], 15)
        self.group["calls"] = self.group["calls"][:7]
        row = alignment.summarize_groups([self.group], self.policy)[0]
        self.assertEqual(row["status"], "partial")
        self.assertNotIn("speedup_vs_measured_baseline", row)

    def test_missing_duplicate_nonfinite_and_missing_trace_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {"status": "complete", "groups": [self.group]}
            self.assertTrue(alignment.validate_complete(record, [self.plan], self.policy, output_dir=Path(directory)))
            for modify in (
                lambda g: g["calls"].pop(),
                lambda g: g["calls"].append(g["calls"][0]),
                lambda g: g["calls"][0].update(forward_ms=float("nan")),
                lambda g: g.update(real_prefix_lengths=[8, 7]),
            ):
                changed = copy.deepcopy(self.group)
                modify(changed)
                self.assertFalse(alignment.validate_complete({"status": "complete", "groups": [changed]}, [self.plan], self.policy, output_dir=Path(directory)))
            self.plan["profile"] = self.group["profile"] = True
            self.group["profiles"] = [{"text_length": length, "trace": "missing.json.gz"} for length in self.plan["padded_lengths"]]
            self.assertFalse(alignment.validate_complete(record, [self.plan], self.policy, output_dir=Path(directory)))

    def test_supervisor_timeout_and_original_errors_preserve_partial_table(self):
        for status in ("complete", "timeout", "error", "incomplete"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(config=CONFIG, output_dir=Path(directory) / "output")

                def launch(command, *, on_progress, **kwargs):
                    self.assertIn("--alignment-worker", command)
                    destination = args.output_dir / "worker"
                    destination.mkdir()
                    group = copy.deepcopy(self.group)
                    if status != "complete":
                        group["calls"] = group["calls"][:7]
                    record = {"status": "complete" if status != "error" else "oom", "groups": [group],
                              "active_case": {"text_length": 12}, "phase": "backward"}
                    if status == "error":
                        record["error"] = "OutOfMemoryError: original allocation"
                    probe._write_json(destination / "result.json", record)
                    on_progress()
                    return (1 if status == "error" else 0), status == "timeout"

                with (
                    patch.object(alignment, "build_plan", return_value=[self.plan]),
                    patch.object(probe, "source_metadata", return_value={}),
                    patch.object(probe, "launch_worker", side_effect=launch),
                ):
                    result = alignment.run_probe(args, self.policy, self.training, on_progress=Mock())
                self.assertEqual(result["status"], "error" if status == "incomplete" else status)
                self.assertEqual(len(result["cases"]), 3 if status == "complete" else 1)
                self.assertEqual(json.loads((args.output_dir / "summary.json").read_text()), result)
                self.assertIn("F+B ms", (args.output_dir / "report.md").read_text())
                if status == "error":
                    self.assertIn("original allocation", result["error"])

    def test_source_hash_mismatch_fails_before_worker_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(config=CONFIG, output_dir=Path(directory) / "output")
            with patch.object(probe, "launch_worker") as worker, self.assertRaisesRegex(ValueError, "differs"):
                alignment.run_probe(args, self.policy, self.training, metadata={"source_sha256": {}})
            worker.assert_not_called()
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
