"""Offline validation, timing, evidence and failure handling for the FLUX probe."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from diffusion_arc.model.contracts import FlowMatchingBatch
from scripts.diagnostics import probe_flux_packing as probe
from tests.diagnostics.test_flux_packing import ReferenceBackend

CONFIG = probe.REPOSITORY_ROOT / "configs/diagnostics/flux_packing.json"


def configuration():
    return probe.load_probe_config(CONFIG, require_checkpoints=False)


def row(value=10):
    return {**dict.fromkeys(probe.TIMINGS, value), "loss": 1.0,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 120}


def complete_result(policy, training, directory):
    groups = []
    (directory / "traces").mkdir(exist_ok=True)
    for plan in probe.build_plan(policy, training):
        cases = []
        for variant in ("sdpa_b2", *policy.backends):
            name = f"{plan['name']}_{variant}.json.gz"
            (directory / "traces" / name).write_bytes(b"test trace placeholder")
            cases.append(dict(variant=variant, status="complete", comparison={"passed": True},
                profile={"trace": name}, calls=[
                    dict(round=r, call=i, phase="warmup" if i < policy.warmup_calls else "measured",
                         **row(999 if i < policy.warmup_calls else 10 + r))
                    for r in range(policy.rounds) for i in range(policy.warmup_calls + policy.measured_calls)]))
        groups.append({**plan, "status": "complete", "cases": cases})
    return dict(status="complete", phase="complete", groups=groups)


class ConfigTests(unittest.TestCase):
    def test_fixed_b2_groups_and_no_alignment_or_regrouping(self):
        policy, training = configuration()
        plan = probe.build_plan(policy, training)
        self.assertEqual(len(plan), 6)
        self.assertEqual(sum(g["batch_size"] for g in plan), 14)
        self.assertEqual(plan[0]["text_lengths"], [468, 406, 357, 288])
        self.assertEqual(plan[-1]["text_lengths"], [4745, 4227])
        self.assertEqual(plan[4]["real_joint_tokens"], 9300)
        self.assertEqual(plan[4]["baseline_joint_tokens"], 9300)
        for group in plan:
            for order in group["round_orders"]:
                self.assertEqual(set(order), {"sdpa_b2", "fa2", "fa3", "fa4"})

    def test_dry_run_writes_nothing_and_needs_no_checkpoint_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stream:
            destination = Path(directory) / "output"
            with patch.object(probe.shapes, "prepare_runtime") as runtime:
                self.assertEqual(probe.main(["--config", str(CONFIG), "--dry-run", "--output-dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
        runtime.assert_not_called()
        output = json.loads(stream.getvalue())
        self.assertEqual(output["cases"], 24)
        self.assertEqual(output["measured_calls"], 360)

    def test_forward_dry_run_declares_twelve_forwards_without_training_runtime(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stream:
            destination = Path(directory) / "output"
            with patch.object(probe.shapes, "prepare_runtime") as runtime:
                self.assertEqual(probe.main(["--config", str(CONFIG), "--forward-only", "--dry-run",
                                             "--output-dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
        runtime.assert_not_called()
        result = json.loads(stream.getvalue())
        self.assertEqual(result["mode"], "forward_only")
        self.assertEqual(result["groups"][0]["text_lengths"], [468, 406])
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["variants"], ["fa3_a", "fa3_b", "fa3_ab", "fa3_ab_repeat",
            "fa3_ab_changed_b", "fa3_ab_changed_a", "fa3_ba", "fa3_ba_repeat",
            "fa3_ba_changed_a", "fa3_ba_changed_b", "sdpa_a", "sdpa_b"])
        self.assertEqual(result["max_duration_seconds"], 480)
        self.assertTrue(result["no_grad"])
        self.assertEqual(result["full_forward_calls"], 12)
        for field in ("backward_calls", "optimizer_initializations", "optimizer_updates", "measured_calls"):
            self.assertEqual(result[field], 0)
        self.assertEqual(result["final_velocity_relative_l2"],
                         configuration()[0].numerical_tolerances["velocity_relative_l2"])
        self.assertIn("original singleton SDPA", result["independent_control"])

    def test_forward_and_correctness_flags_are_mutually_exclusive(self):
        with patch.object(probe, "load_probe_config") as load, redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            probe.main(["--config", str(CONFIG), "--forward-only", "--correctness", "--dry-run"])
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_rejects_scope_changes_and_schedule_drift_at_ingestion(self):
        mutations = [lambda r: r.update(alignment_multiples=[8]),
                     lambda r: r.update(backends=["fa2", "fa2"]),
                     lambda r: r.update(warmup_calls=False),
                     lambda r: r["groups"][0].update(padded_length=512),
                     lambda r: r["groups"][0].update(target_indices=[0, 1, 2, 3]),
                     lambda r: r["groups"][0].update(batch_index=1000),
                     lambda r: r["numerical_tolerances"].update(gradient_relative_l2=float("nan"))]
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                raw = json.loads(CONFIG.read_text())
                raw["training_config"] = str(configuration()[0].training_config)
                mutation(raw)
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    probe.load_probe_config(path, require_checkpoints=False)


class NumericalAndTimingTests(unittest.TestCase):
    def test_streamed_error_metrics_and_nonfinite_detection(self):
        reference = torch.tensor([1., 2., 3.])
        actual = reference + 1
        stats = probe.tensor_difference([(actual, reference)])
        self.assertEqual(stats["max_abs"], 1)
        self.assertEqual(stats["rms"], 1)
        self.assertAlmostEqual(stats["relative_l2"], (3 / 14) ** .5)
        with self.assertRaises(FloatingPointError):
            probe.tensor_difference([(torch.tensor([float("nan")]), torch.ones(1))])

    def test_noncausal_backward_preflight_and_isolation(self):
        backend = ReferenceBackend()
        backend.name = "test_oracle"
        with patch.object(torch.cuda, "synchronize"):
            result = probe.backend_preflight(backend, "cpu", configuration()[0].numerical_tolerances)
        self.assertTrue(result["passed"])

    def test_measurement_keeps_gradient_storage_weights_loss_and_does_not_update(self):
        model = torch.nn.Linear(1, 128, bias=False)
        model.weight.data.fill_(.5)
        model.weight.grad = torch.zeros_like(model.weight)
        original_pointer = model.weight.grad.data_ptr()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        clean = torch.zeros(2, 3, 128)
        flow = FlowMatchingBatch(clean, torch.tensor([.2, .8]), clean)
        data = dict(clean=clean, flow=flow, prefixes=(torch.ones(2, 12), torch.ones(5, 12)))
        group = dict(padded_length=6)
        seen = []
        def forward(m, variant, assembled, input_data, dtype):
            self.assertIs(input_data["flow"], flow)
            seen.append(assembled)
            return m(torch.ones(2, 3, 1))
        event = lambda **kw: SimpleNamespace(record=lambda: None, elapsed_time=lambda other: 1.0)
        with ExitStack() as stack:
            for name in ("synchronize", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, "Event", side_effect=event))
            stack.enter_context(patch.object(probe.shapes, "_memory", return_value={}))
            stack.enter_context(patch.object(probe, "forward_velocity", side_effect=forward))
            step = stack.enter_context(patch.object(optimizer, "step"))
            for variant in ("sdpa_b2", "fa2"):
                result, output = probe.measure_call(model, optimizer, variant, group, data,
                                                    compute_dtype=torch.bfloat16, weight=2 / 48)
                self.assertEqual(result["loss"], .25)
                self.assertFalse(output.requires_grad)
                self.assertEqual(model.weight.grad.data_ptr(), original_pointer)
                torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, (2 / 48) / 128))
            step.assert_not_called()
        self.assertEqual(seen[0].text_attention_mask.sum(1).tolist(), [2, 5])
        self.assertEqual(tuple(seen[1].shape), (7, 12))
        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, .5)))

    def test_backend_evidence_requires_backward_and_rejects_fallbacks(self):
        counts = {"flux_packing::fa3": 25, "flash_attn_3::_flash_attn_forward": 25,
                  "flash_attn_3::_flash_attn_backward": 25}
        probe.validate_backend_events(counts, "fa3", 25)
        for wrong in ({**counts, "flash_attn_3::_flash_attn_backward": 0},
                      {**counts, "aten::scaled_dot_product_attention": 25}):
            with self.assertRaises(RuntimeError):
                probe.validate_backend_events(wrong, "fa3", 25)


class ResultTests(unittest.TestCase):
    def test_forward_supervisor_dispatch_budget_manifest_and_partial_results(self):
        from scripts.diagnostics import _flux_packing_forward as forward

        policy, training = configuration()
        for timed_out in (False, True):
            with self.subTest(timed_out=timed_out), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "output"
                partial = dict(status="running" if timed_out else "complete", mode="forward_only",
                               phase="fa3_ab/forward" if timed_out else "complete", cases=[],
                               numerical_passed=False)
                def launch(command, **kwargs):
                    self.assertIn("--forward-only", command)
                    self.assertNotIn("--correctness", command)
                    self.assertEqual(kwargs["timeout_seconds"], 480)
                    probe.shapes._write_json(destination / "result.json", partial)
                    return (-9, True) if timed_out else (0, False)
                with patch.object(probe, "load_probe_config", return_value=(policy, training)), \
                     patch.object(probe, "source_metadata", return_value={"source_sha256": {}}), \
                     patch.object(probe, "source_hashes", return_value={}), \
                     patch.object(probe, "dataset_hashes", return_value={}), \
                     patch.object(probe.shapes, "launch_worker", side_effect=launch) as launched, \
                     patch.object(forward, "validate_complete") as validate, \
                     patch.object(forward, "write_report") as report, \
                     patch.object(probe, "summarize") as summarize:
                    result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination,
                                                              forward_only=True))
                launched.assert_called_once()
                summarize.assert_not_called()
                report.assert_called_once_with(destination / "report.md", result)
                if timed_out:
                    validate.assert_not_called()
                    self.assertEqual(result["status"], "timeout")
                    self.assertEqual(result["phase"], "fa3_ab/forward")
                else:
                    validate.assert_called_once_with(result)
                    self.assertEqual(result["status"], "complete")
                    self.assertFalse(result["numerical_passed"])
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["mode"], "forward_only")
                self.assertEqual(manifest["effective_worker_seconds"], 480)
                self.assertEqual(manifest["full_forward_calls"], 12)
                self.assertTrue(manifest["no_grad"])
                self.assertEqual(manifest["optimizer_initializations"], 0)
                self.assertEqual(manifest["backward_calls"], 0)
                self.assertNotIn("optimizer_learning_rate_override", manifest)
                self.assertEqual(json.loads((destination / "summary.json").read_text())["rows"], [])

    def test_summary_excludes_warmup_and_partial_rows_have_no_speedup(self):
        policy, training = configuration()
        with tempfile.TemporaryDirectory() as directory:
            result = complete_result(policy, training, Path(directory))
            rows = probe.summarize(result, policy)
            self.assertEqual(len(rows), 24)
            self.assertEqual(rows[0]["timings"]["total_wall_ms"]["median"], 11)
            self.assertEqual(rows[0]["measured_calls"], 15)
            self.assertEqual(rows[0]["speedup_vs_baseline"], 1)
            result["groups"][0]["cases"][0]["status"] = "running"
            rows = probe.summarize(result, policy)
            self.assertTrue(all("speedup_vs_baseline" not in r for r in rows if r["group"] == rows[0]["group"]))

    def test_completion_requires_exact_samples_numerics_and_trace(self):
        policy, training = configuration()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = complete_result(policy, training, directory)
            plan = probe.build_plan(policy, training)
            probe.validate_complete(result, plan, policy, directory)
            for mutate in (lambda c: c["calls"].pop(),
                           lambda c: c["comparison"].update(passed=False),
                           lambda c: c["calls"][0].update(forward_ms=float("nan")),
                           lambda c: c["profile"].update(trace="missing")):
                broken = copy.deepcopy(result)
                mutate(broken["groups"][0]["cases"][0])
                with self.assertRaises(ValueError):
                    probe.validate_complete(broken, plan, policy, directory)

    def test_timeout_preserves_partial_samples_and_original_phase(self):
        policy, training = configuration()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            def launch(*args, **kwargs):
                partial = dict(status="running", phase="maximum_b1/fa3/measured_3", groups=[])
                probe.shapes._write_json(destination / "result.json", partial)
                return -9, True
            with patch.object(probe, "load_probe_config", return_value=(policy, training)), \
                 patch.object(probe, "dataset_hashes", return_value={}), \
                 patch.object(probe.shapes, "launch_worker", side_effect=launch):
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination))
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["phase"], "maximum_b1/fa3/measured_3")
            self.assertTrue((destination / "report.md").is_file())
            self.assertTrue((destination / "result.json").is_file())

    def test_source_mismatch_fails_before_creating_artifacts(self):
        policy, training = configuration()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"
            with patch.object(probe, "load_probe_config", return_value=(policy, training)), \
                 patch.object(probe, "dataset_hashes", return_value={}), self.assertRaises(ValueError):
                probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination), metadata={"source_sha256": {}})
            self.assertFalse(destination.exists())


class WorkerTests(unittest.TestCase):
    def test_forward_mode_dispatches_without_entering_training_worker(self):
        from scripts.diagnostics import _flux_packing_forward as forward

        args = SimpleNamespace(forward_only=True)
        with patch.object(forward, "run_worker", return_value={"status": "complete"}) as worker, \
             patch.object(probe.shapes, "prepare_runtime") as runtime:
            self.assertEqual(probe.run_worker(args), {"status": "complete"})
        worker.assert_called_once_with(args)
        runtime.assert_not_called()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            probe.run_worker(SimpleNamespace(forward_only=True, correctness=True))

    def run_mock_worker(self, directory, *, fail_call=None):
        policy, training = configuration()
        policy = replace(policy, groups=policy.groups[:1])
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.ones_like(model.weight)
        model.config = SimpleNamespace(attention_head_dim=128)
        model.gradient_checkpointing = False
        model.transformer_blocks, model.single_transformer_blocks = [], []
        models = SimpleNamespace(transformer=model, flux_compute_dtype=torch.bfloat16)
        events = []
        def measure(*args, **kwargs):
            events.append("measure")
            self.assertEqual(kwargs["weight"], 4 / 48)
            if fail_call is not None and events.count("measure") == fail_call:
                raise torch.OutOfMemoryError("deliberate test OOM")
            return row(), torch.ones(4, 3, 128)
        def profile(call, model, variant, backend, path):
            events.append("profile")
            path.write_bytes(b"test trace")
            return dict(trace=path.name)
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            replacements = {
                "load_probe_config": (policy, training),
                "backend_preflight": {"passed": True},
                "prepare_inputs": {"short_b4": None},
                "group_inputs": {"layout": None, "input_sha256": "fixed"},
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(probe, name, return_value=value))
            stack.enter_context(patch.object(probe, "FlashBackend", side_effect=lambda name: SimpleNamespace(name=name, binding=name)))
            stack.enter_context(patch.object(probe, "measure_call", side_effect=measure))
            stack.enter_context(patch.object(probe, "profile_case", side_effect=profile))
            stack.enter_context(patch.object(probe, "version", return_value="test"))
            stack.enter_context(patch.object(probe.shapes, "prepare_single_gpu_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(probe.shapes, "prepare_runtime", return_value=(torch.device("cpu"), models, Mock(), [], {})))
            stack.enter_context(patch.object(probe.shapes, "_environment_metadata", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)))
            stack.enter_context(patch.object(probe.traceback, "print_exc"))
            result = probe.run_worker(SimpleNamespace(config=CONFIG, output_dir=directory))
        return result, events, policy, training

    def test_profiles_follow_all_timed_rounds_and_every_case_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result, events, policy, training = self.run_mock_worker(directory)
            self.assertEqual(result["status"], "complete", result.get("error"))
            self.assertEqual(events, ["measure"] * 89 + ["profile"] * 4)
            probe.validate_complete(result, probe.build_plan(policy, training), policy, directory)
            self.assertTrue(all(c["comparison"]["passed"] for c in result["groups"][0]["cases"]))

    def test_oom_stops_without_retry_and_preserves_original_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result, events, _, _ = self.run_mock_worker(directory, fail_call=3)
            self.assertEqual(result["status"], "oom")
            self.assertIn("deliberate test OOM", result["error"])
            self.assertEqual(events, ["measure"] * 3)
            self.assertIn("warmup", result["phase"])
            persisted = json.loads((directory / "result.json").read_text())
            self.assertEqual(persisted["phase"], result["phase"])
