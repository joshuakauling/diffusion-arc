"""Forward-only packing comparisons exercise real blocks and independent slices."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc.model.contracts import FlowMatchingBatch
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_packing_forward as forward
from scripts.diagnostics._flux_packing import packing_layout
from tests.diagnostics.test_flux_packing import ReferenceBackend
from tests.diagnostics.test_probe_flux_packing import configuration


def fixture(double=2, single=2, head_dim=128):
    model = Flux2Transformer2DModel(
        in_channels=128, out_channels=128, num_layers=double, num_single_layers=single,
        attention_head_dim=head_dim, num_attention_heads=1, joint_attention_dim=12,
        axes_dims_rope=(head_dim // 4,) * 4, guidance_embeds=False,
    ).eval()
    images = torch.randn(2, 3, 128)
    positions = torch.zeros(2, 3, 4, dtype=torch.int64)
    positions[:, :, 2] = torch.arange(3)
    # Different positions make a swapped sample identity visible to the original singletons.
    positions[1, :, 1] = 2
    data = dict(prefixes=(torch.randn(2, 12), torch.randn(5, 12)), clean=images.clone(),
                flow=FlowMatchingBatch(images, torch.tensor([.17, .83]), torch.randn_like(images)),
                image_positions=positions, layout=packing_layout((2, 5), 3, "cpu"), input_sha256="fixture")
    return model, data


class ForwardTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(92)
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(sdpa_kernel(SDPBackend.MATH))
        self.context.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="In CPU autocast.*")

    def test_real_multiblock_singletons_packing_original_controls_and_isolation(self):
        model, data = fixture()
        parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        with patch("torch.autograd.backward", side_effect=AssertionError("unexpected backward")), \
             patch("torch.autograd.grad", side_effect=AssertionError("unexpected autograd")):
            result = forward.run_comparisons(model, data, ReferenceBackend(), torch.float32, .00002)
        self.assertEqual([c["variant"] for c in result["cases"]], list(forward.VARIANTS))
        for field in ("packing_forward_passed", "order_forward_passed", "repeatability_passed", "isolation_passed", "independent_controls_passed"):
            self.assertTrue(result[field], field)
        for case in result["cases"]:
            self.assertFalse(case["grad_enabled"])
            self.assertEqual(list(case["outputs"]), ["transformer_blocks.0.input", "transformer_blocks.0.attention_input",
                "transformer_blocks.0.attention_output", "transformer_blocks.0", "transformer_blocks.1",
                "single_transformer_blocks.0", "single_transformer_blocks.1", "norm_out_input", "norm_out", "velocity"])
            for reference, samples in case["comparisons"].items():
                for label, stages in samples.items():
                    if "_changed_" in case["variant"] and label == case["variant"][-1].upper():
                        continue
                    for stage, streams in stages.items():
                        for stream, metrics in streams.items():
                            self.assertLess(metrics["max_abs"], 2e-6, (case["variant"], reference, label, stage, stream))
        for name, p in model.named_parameters():
            self.assertIsNone(p.grad)
            self.assertEqual(p.dtype, torch.float32)
            torch.testing.assert_close(p, parameters[name], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            forward.write_report(path, dict(result, status="complete"))
            self.assertIn("fa3_ab_changed_b", path.read_text())
        json.dumps(result, allow_nan=False)

    def test_canonicalization_uses_distinct_text_image_and_joint_offsets(self):
        text = torch.tensor([10, 11, 20, 21, 22]).reshape(1, 5, 1)
        images = torch.tensor([100, 101, 200, 201]).reshape(1, 4, 1)
        joint = torch.tensor([10, 11, 100, 101, 20, 21, 22, 200, 201]).reshape(1, 9, 1)
        expected = [( [10, 11], [100, 101]), ([20, 21, 22], [200, 201])]
        for kind, output in (("double", (text, images)), ("single", joint)):
            parts = forward.split_block_output(kind, output, (2, 3), 2)
            self.assertEqual([(p["text"].flatten().tolist(), p["image"].flatten().tolist()) for p in parts], expected)
        with self.assertRaises(ValueError):
            forward.split_block_output("double", (images, text), (2, 3), 2)

    def test_sample_slices_preserve_storage_and_perturbation_does_not_mutate_references(self):
        _, data = fixture()
        before = forward.input_identity(data, ("A", "B"))
        a, b = forward.slice_inputs(data, 0, 1), forward.slice_inputs(data, 1, 2)
        self.assertIs(b["prefixes"][0], data["prefixes"][1])
        self.assertEqual(b["flow"].noisy_latents.untyped_storage().data_ptr(), data["flow"].noisy_latents.untyped_storage().data_ptr())
        perturbed = forward.changed_b(data)
        audit = forward.validate_input_reuse(data, (a, b), perturbed)
        self.assertTrue(audit["a_unchanged"])
        self.assertEqual(forward.input_identity(data, ("A", "B")), before)
        broken = forward.slice_inputs(data, 0, 1)
        broken["flow"] = FlowMatchingBatch(broken["flow"].noisy_latents + 1,
                                           broken["flow"].timesteps, broken["flow"].target_velocity)
        with self.assertRaisesRegex(ValueError, "exact prepared inputs"):
            forward.validate_input_reuse(data, (broken, b), perturbed)

    def test_reordering_preserves_identity_and_reuses_the_same_perturbations(self):
        model, data = fixture(head_dim=8)
        expected = forward.input_identity(data, ("A", "B"))
        reversed_data = forward.reordered_inputs(data, (1, 0))
        self.assertEqual(forward.input_identity(reversed_data, ("B", "A")), expected)
        self.assertEqual(reversed_data["layout"].cumulative.tolist(), [0, 8, 13])
        with self.assertRaises(ValueError):
            forward.reordered_inputs(data, (1, 1))
        result = forward.run_comparisons(model, data, ReferenceBackend(), torch.float32, .02)
        self.assertTrue(result["input_identity"]["reordering_preserves_all_inputs"])
        cases = {row["variant"]: row for row in result["cases"]}
        self.assertEqual(cases["fa3_ba"]["labels"], ["B", "A"])
        for order in ("ab", "ba"):
            for changed, unchanged in (("a", "B"), ("b", "A")):
                key = f"fa3_{order}_changed_{changed}"
                self.assertEqual(list(cases[key]["comparisons"]), [f"fa3_{order}"])
                check = result["isolation_checks"][key]
                self.assertEqual(check["unchanged_label"], unchanged)
                self.assertTrue(check["unchanged_outputs_bitwise_equal"])
                self.assertTrue(check["changed_velocity_differs"])
        for name in ("sdpa_a", "sdpa_b"):
            self.assertIn("fa3_ba", cases[name]["comparisons"])

    def test_symmetric_isolation_detects_cross_example_leakage(self):
        class LeakingBackend(ReferenceBackend):
            def __call__(self, q, k, v, layout):
                return super().__call__(q, k, v, layout) + v.mean(dim=0)
        model, data = fixture(head_dim=8)
        result = forward.run_comparisons(model, data, LeakingBackend(), torch.float32, .02)
        self.assertTrue(result["repeatability_passed"])
        self.assertFalse(result["isolation_passed"])
        self.assertTrue(all(not row["unchanged_outputs_bitwise_equal"] for row in result["isolation_checks"].values()))

    def test_failure_restores_hooks_processors_and_does_not_keep_autograd(self):
        model, data = fixture()
        blocks = [*model.transformer_blocks, *model.single_transformer_blocks]
        processors = [b.attn.processor for b in blocks]
        hook_counts = [len(b._forward_hooks) for b in blocks]
        class FailureBackend(ReferenceBackend):
            def __init__(self):
                self.calls = 0
            def __call__(self, q, k, v, layout):
                self.calls += 1
                self_test.assertFalse(torch.is_grad_enabled())
                if self.calls == 2:
                    raise RuntimeError("intentional failure")
                return super().__call__(q, k, v, layout)
        self_test = self
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            forward.model_call(model, data, ("A", "B"), FailureBackend(), torch.float32, retain=True)
        self.assertEqual([b.attn.processor for b in blocks], processors)
        self.assertEqual([len(b._forward_hooks) for b in blocks], hook_counts)

    def test_missing_block_capture_and_nonfinite_outputs_are_rejected(self):
        model, data = fixture()
        capture = forward.BlockCapture(model, (2, 5), 3, ("A", "B"))
        with self.assertRaisesRegex(ValueError, "missing or reordered"):
            capture.finish(torch.zeros_like(data["clean"]))
        with self.assertRaises(FloatingPointError):
            forward.BlockCapture(model, (2, 5), 3, ("A", "B")).record(
                "bad", [{"image": torch.tensor([float("nan")])}, {"image": torch.zeros(1)}])

    def test_preflight_is_forward_only_and_events_require_actual_fa3(self):
        with patch("torch.autograd.grad", side_effect=AssertionError("unexpected autograd")):
            result = forward.preflight(ReferenceBackend(), torch.device("cpu"), .02)
        self.assertTrue(result["passed"])
        counts = {"flash_attn_3::_flash_attn_forward": 25, "flux_packing::fa3": 25}
        forward.validate_forward_events(counts, packed=True, layers=25)
        for invalid in ({}, {**counts, "aten::scaled_dot_product_attention": 25},
                        {**counts, "flash_attn_3::_flash_attn_backward": 1}):
            with self.assertRaises(ValueError):
                forward.validate_forward_events(invalid, packed=True, layers=25)
        forward.validate_forward_events({"aten::_scaled_dot_product_cudnn_attention": 25}, packed=False, layers=25)

    def test_full_completion_checks_coverage_and_preserves_numerical_failure(self):
        model, data = fixture(double=5, single=20, head_dim=8)
        result = forward.run_comparisons(model, data, ReferenceBackend(), torch.float32, .02)
        result.update(backward_calls=0, optimizer_initialized=False, numerical_passed=False)
        for case in result["cases"]:
            case["operators"] = ({"flash_attn_3::_flash_attn_forward": 25, "flux_packing::fa3": 25}
                                 if case["variant"].startswith("fa3") else {"aten::_scaled_dot_product_cudnn_attention": 25})
        forward.validate_complete(result)
        for mutate in (lambda r: r["cases"].pop(),
                       lambda r: r["cases"][2]["comparisons"]["fa3_singletons"]["A"].pop("single_transformer_blocks.19"),
                       lambda r: r.update(optimizer_initialized=True)):
            broken = copy.deepcopy(result)
            mutate(broken)
            with self.assertRaises(ValueError):
                forward.validate_complete(broken)

    def test_plan_selects_exact_two_targets_from_original_group(self):
        policy, training = configuration()
        effective = forward.forward_policy(policy)
        plan = forward.build_plan(effective, training)[0]
        self.assertEqual(plan["target_indices"], [15, 12])
        self.assertEqual(plan["text_lengths"], [468, 406])
        self.assertEqual(plan["cumulative"], [0, 1368, 2674])
        self.assertEqual(plan["reversed_cumulative"], [0, 1306, 2674])
        self.assertEqual(plan["maximum_joint_length"], 1368)
        self.assertEqual(plan["source_group"]["target_indices"], [15, 12, 8, 7])
        self.assertEqual(effective.max_duration_seconds, 480)
        self.assertEqual(policy.max_duration_seconds, 2400)
