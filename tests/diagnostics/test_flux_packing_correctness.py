"""CPU checks for the one-batch FLUX correctness diagnostic."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from scripts.diagnostics import _flux_packing_correctness as checks
from scripts.diagnostics import probe_flux_packing as probe
from scripts.diagnostics._flux_packing import packing_layout, packed_processors, packed_velocity
from tests.diagnostics.test_flux_packing import tiny_model, inputs
from tests.diagnostics.test_probe_flux_packing import CONFIG, configuration, row


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(72)

    def test_segmented_oracle_matches_block_mask_outputs_and_all_input_gradients(self):
        layout = packing_layout((2, 5), 3, "cpu")
        qkv = tuple(torch.randn(13, 2, 8, requires_grad=True) for _ in range(3))
        owner = layout.joint_samples
        mask = owner[:, None] == owner[None, :]
        expected = F.scaled_dot_product_attention(
            *(t.transpose(0, 1).unsqueeze(0) for t in qkv), attn_mask=mask,
        ).squeeze(0).transpose(0, 1)
        actual = checks.SegmentedSDPA(SDPBackend.MATH)(*qkv, layout)
        upstream = torch.randn_like(expected)
        expected_grad = torch.autograd.grad(expected, qkv, upstream)
        actual_grad = torch.autograd.grad(actual, qkv, upstream)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b, atol=5e-7, rtol=1e-5)

    def test_capture_covers_real_double_single_blocks_and_loss_gradients(self):
        model, data = tiny_model(), inputs()
        layout = packing_layout((2, 5), 3, "cpu")
        capture = checks.CaptureAttention(checks.SegmentedSDPA(SDPBackend.MATH), 1)
        with packed_processors(model, layout, capture):
            output = packed_velocity(model,
                torch.cat((data["encoder_hidden_states"][0, :2], data["encoder_hidden_states"][1])),
                data["hidden_states"], data["img_ids"], data["timestep"], layout)
        output.square().mean().backward()
        self.assertEqual(capture.calls, 2)
        self.assertEqual(set(capture.records), {"transformer_blocks.0", "single_transformer_blocks.0"})
        for record in capture.records.values():
            self.assertEqual(tuple(record["output_gradient"].shape), (13, 2, 8))
            self.assertTrue(torch.isfinite(record["output_gradient"]).all())
            self.assertGreater(record["output_gradient"].abs().sum(), 0)
            self.assertTrue(all(not t.requires_grad for t in record["qkv"]))

    def test_kernel_comparison_uses_same_upstream_even_when_outputs_differ(self):
        layout = packing_layout((2, 5), 3, "cpu")
        oracle = checks.SegmentedSDPA(SDPBackend.MATH)
        class OffsetBackend:
            def __call__(self, q, k, v, layout):
                return oracle(q, k, v, layout) + 1
        record = {"qkv": tuple(torch.randn(13, 2, 8) for _ in range(3)),
                  "output_gradient": torch.randn(13, 2, 8)}
        result = checks.kernel_comparison(record, oracle, OffsetBackend(), layout,
                                           torch.device("cpu"), configuration()[0].numerical_tolerances)
        self.assertFalse(result["passed"])
        self.assertGreater(result["output"]["relative_l2"], .02)
        self.assertEqual(result["combined_gradient"]["relative_l2"], 0)

    def test_gradient_metrics_weight_contributions_and_handle_zero_reference(self):
        rows = [checks.metrics(checks.difference_totals(torch.tensor([3., 4.]), torch.tensor([1., 2.]))),
                checks.metrics(checks.difference_totals(torch.tensor([100.]), torch.tensor([100.])))]
        combined = checks.aggregate(rows)
        self.assertAlmostEqual(combined["relative_l2"], (8 / 10005) ** .5)
        self.assertAlmostEqual(combined["cosine"], 10011 / (10005 * 10025) ** .5)
        undefined = checks.metrics(checks.difference_totals(torch.ones(2), torch.zeros(2)))
        self.assertIsNone(undefined["relative_l2"])
        self.assertFalse(checks.within(undefined["relative_l2"], .05))
        json.dumps(undefined, allow_nan=False)
        with self.assertRaises(FloatingPointError):
            checks.difference_totals(torch.tensor([float("nan")]), torch.ones(1))

    def test_backend_selection_rejects_mixed_or_missing_dispatch(self):
        self.assertEqual(checks.observed_sdpa_backend({"aten::_scaled_dot_product_cudnn_attention": 25}, 25),
                         SDPBackend.CUDNN_ATTENTION)
        with self.assertRaises(ValueError):
            checks.observed_sdpa_backend({"aten::_scaled_dot_product_cudnn_attention": 24,
                                          "aten::_scaled_dot_product_flash_attention": 1}, 25)
        with self.assertRaises(ValueError):
            checks.observed_sdpa_backend({}, 25)

    def test_dry_run_limits_scope_and_does_not_allocate_or_write(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stream:
            destination = Path(directory) / "output"
            with patch.object(probe.shapes, "prepare_runtime") as runtime:
                self.assertEqual(probe.main(["--config", str(CONFIG), "--correctness", "--dry-run",
                                             "--output-dir", str(destination)]), 0)
            self.assertFalse(destination.exists())
            runtime.assert_not_called()
        result = json.loads(stream.getvalue())
        self.assertEqual([g["name"] for g in result["groups"]], ["short_b4"])
        self.assertEqual(result["variants"], list(checks.VARIANTS))
        self.assertEqual(result["max_duration_seconds"], 480)
        self.assertEqual(result["measured_calls"], 0)


class WorkerTests(unittest.TestCase):
    def mock_worker(self, directory, *, fail_second=False):
        policy, training = configuration()
        model = torch.nn.Linear(1, 1, bias=False)
        model.config = SimpleNamespace(attention_head_dim=128)
        model.transformer_blocks, model.single_transformer_blocks = [None], [None]
        models = SimpleNamespace(transformer=model, flux_compute_dtype=torch.bfloat16)
        observed = []
        def call(m, optimizer, variant, backend, group, data, dtype, weight):
            observed.append(variant)
            self.assertEqual(weight, 4 / 48)
            self.assertEqual(group["name"], "short_b4")
            if variant == "sdpa_segmented":
                if fail_second:
                    raise torch.OutOfMemoryError("test OOM")
                backend.calls = 2
                backend.records = {name: {"output_gradient": torch.ones(1)} for name in backend.selected.values()}
            model.weight.grad = torch.full_like(model.weight, {"sdpa_b2": 1., "sdpa_segmented": 1.2, "fa3": 1.3}[variant])
            operators = ({"flash_attn_3::_flash_attn_forward": 2, "flash_attn_3::_flash_attn_backward": 2}
                         if variant == "fa3" else {"aten::_scaled_dot_product_attention_math": 8 if backend else 2})
            return row(), torch.ones(4, 3, 128), operators
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            replacements = {"load_probe_config": (policy, training), "backend_preflight": {"passed": True},
                "prepare_inputs": {"short_b4": None}, "group_inputs": {"layout": None, "input_sha256": "fixed"}}
            for name, value in replacements.items():
                stack.enter_context(patch.object(probe, name, return_value=value))
            stack.enter_context(patch.object(checks, "model_call", side_effect=call))
            stack.enter_context(patch.object(checks, "kernel_comparison", return_value={"passed": True}))
            stack.enter_context(patch.object(checks, "FlashBackend", return_value=SimpleNamespace(binding="fa3")))
            stack.enter_context(patch.object(probe, "version", return_value="test"))
            stack.enter_context(patch.object(probe.shapes, "prepare_single_gpu_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(probe.shapes, "prepare_runtime", return_value=(torch.device("cpu"), models, Mock(), [], {})))
            stack.enter_context(patch.object(probe.shapes, "_environment_metadata", return_value={}))
            stack.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)))
            stack.enter_context(patch.object(checks.traceback, "print_exc"))
            result = checks.run_worker(SimpleNamespace(config=CONFIG, output_dir=directory))
        return result, observed

    def test_failed_numerics_do_not_skip_any_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            result, observed = self.mock_worker(Path(directory))
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertEqual(observed, list(checks.VARIANTS))
        self.assertFalse(result["numerical_passed"])
        self.assertEqual(len(result["kernel_comparisons"]), 2)
        checks.validate_complete(result)
        comparison = result["cases"][1]["comparisons"]["sdpa_b2"]
        self.assertAlmostEqual(comparison["gradient"]["relative_l2"], .2, places=5)
        self.assertEqual(comparison["parameters"]["weight"]["share_of_squared_gradient_error"], 1)
        result["cases"].pop()
        with self.assertRaises(ValueError):
            checks.validate_complete(result)

    def test_oom_stops_without_retry_and_preserves_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            result, observed = self.mock_worker(Path(directory), fail_second=True)
        self.assertEqual(result["status"], "oom")
        self.assertEqual(observed, ["sdpa_b2", "sdpa_segmented"])
        self.assertEqual(result["cases"][0]["status"], "complete")
        self.assertEqual(result["phase"], "sdpa_segmented/forward_backward")

    def test_supervisor_propagates_mode_budget_and_preserves_partial_report(self):
        policy, training = configuration()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "output"
            def launch(command, **kwargs):
                self.assertIn("--correctness", command)
                self.assertEqual(kwargs["timeout_seconds"], 480)
                probe.shapes._write_json(destination / "result.json", dict(status="running", mode="correctness",
                    phase="sdpa_segmented/compare_to_sdpa_b2", cases=[], kernel_comparisons={}))
                return -9, True
            with patch.object(probe, "load_probe_config", return_value=(policy, training)), \
                 patch.object(probe, "dataset_hashes", return_value={}), \
                 patch.object(probe.shapes, "launch_worker", side_effect=launch):
                result = probe.run_probe(SimpleNamespace(config=CONFIG, output_dir=destination, correctness=True))
            self.assertEqual(result["status"], "timeout")
            self.assertIn("sdpa_segmented", result["phase"])
            self.assertIn("correctness diagnostic", (destination / "report.md").read_text())
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["mode"], "correctness")
            self.assertEqual(len(manifest["plan"]), 1)
            self.assertEqual(manifest["effective_worker_seconds"], 480)
