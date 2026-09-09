"""Validate the isolated modulation reduction and its complete FLUX backward."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend

from scripts.diagnostics import _flux_backward_layouts as layouts
from scripts.diagnostics._flux_modulation_backward import modulation_gather, observe_modulation_reduction
from scripts.diagnostics._flux_packing import packing_layout
from scripts.diagnostics._flux_packing_preflight import validate_layout
from tests.diagnostics.test_flux_backward_layouts import STAGES, fixture, independent_weighted_backward


class ModulationBackwardTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(192)

    def test_bf16_cancellation_matches_independent_fp64_oracle_in_both_orders(self):
        lengths_by_label = {"A": 257, "B": 131}
        for labels in (("A", "B"), ("B", "A")):
            with self.subTest(labels=labels):
                lengths = tuple(lengths_by_label[label] for label in labels)
                source = torch.randn(4, 2).to(torch.bfloat16).t().requires_grad_()
                self.assertFalse(source.is_contiguous())
                owners = torch.tensor([index for index, length in enumerate(lengths) for _ in range(length)])
                gradient = torch.empty(4, len(owners), dtype=torch.bfloat16).t()
                # Representable BF16 contributions with cancellation and an odd
                # tail make per-token rounding differ from a single final cast.
                for token in range(len(owners)):
                    gradient[token] = torch.tensor((256, 1, -256, .5))[[token % 3, (token + 1) % 3,
                                                                      (token + 2) % 3, 3]]
                self.assertFalse(gradient.is_contiguous())
                before = gradient.clone()
                expected = torch.zeros(source.shape, dtype=torch.float64)
                for token, owner in enumerate(owners.tolist()):
                    expected[owner] += gradient[token].double()
                with observe_modulation_reduction() as evidence:
                    actual = modulation_gather(source, owners, lengths, reduction="fp32")
                    native = source.index_select(0, owners)
                    self.assertTrue(torch.equal(actual, native))
                    self.assertEqual(actual.dtype, native.dtype)
                    self.assertEqual(actual.stride(), native.stride())
                    self.assertEqual(actual.shape, native.shape)
                # Backward can run after the observation context has exited.
                actual.backward(gradient)
                torch.testing.assert_close(source.grad, expected.to(source.dtype), atol=0, rtol=0)
                torch.testing.assert_close(gradient, before, atol=0, rtol=0)
                self.assertEqual(evidence, dict(forward_calls=1, backward_calls=1,
                    source_dtypes=["torch.bfloat16"], reduction_dtype="torch.float32"))

    def test_fp32_gradient_matches_native_for_singleton_and_unequal_segments(self):
        for lengths in ((1,), (1, 5, 2)):
            with self.subTest(lengths=lengths):
                owners = torch.tensor([index for index, length in enumerate(lengths) for _ in range(length)])
                source = torch.randn(len(lengths), 3, requires_grad=True)
                upstream = torch.randn(3, sum(lengths)).t()
                native = torch.autograd.grad(source.index_select(0, owners), source, upstream)[0]
                actual = torch.autograd.grad(modulation_gather(source, owners, lengths, reduction="fp32"),
                                             source, upstream)[0]
                torch.testing.assert_close(actual, native, atol=5e-7, rtol=1e-6)

    def test_preflight_rejects_ownership_that_would_invalidate_segment_reduction(self):
        layout = packing_layout((2, 5), 3, "cpu")
        validate_layout(layout)
        for name in ("text_samples", "image_samples", "joint_samples"):
            with self.subTest(name=name):
                damaged = getattr(layout, name).clone()
                damaged[0] = 1
                with self.assertRaisesRegex(ValueError, "source token/segment contract"):
                    validate_layout(replace(layout, **{name: damaged}))

    def test_first_order_scope_and_configuration_fail_closed(self):
        source = torch.randn(2, 3, requires_grad=True)
        owners = torch.tensor([0, 0, 1])
        with self.assertRaisesRegex(ValueError, "native or fp32"):
            modulation_gather(source, owners, (2, 1), reduction="typo")
        with self.assertRaisesRegex(ValueError, "BF16 or FP32"):
            modulation_gather(source.double(), owners, (2, 1), reduction="fp32")
        output = modulation_gather(source, owners, (2, 1), reduction="fp32")
        first = torch.autograd.grad(output.square().sum(), source, create_graph=True)[0]
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first.sum(), source)
        model, items = fixture()
        for variant, reduction in (("fa3_single", "fp32"), ("fa3_packed_ab", "typo")):
            with self.subTest(variant=variant, reduction=reduction):
                with self.assertRaisesRegex(ValueError, "modulation reduction"):
                    layouts.run_case(model, items, variant, padded_length=7, modulation_reduction=reduction)

    def test_complete_fp32_math_backward_matches_independent_singleton_loss(self):
        model, items = fixture()
        expected, outputs, losses = independent_weighted_backward(model, items)
        for variant in ("cudnn_packed_ab", "cudnn_packed_ba"):
            with self.subTest(variant=variant):
                row, state, _ = layouts.run_case(model, items, variant, padded_length=7,
                    compute_dtype=torch.float32, sdpa_backend=SDPBackend.MATH,
                    capture_stages=STAGES, profile=False, modulation_reduction="fp32")
                self.assertEqual(row["modulation_reduction"], "fp32")
                self.assertEqual(row["modulation_reduction_evidence"], dict(forward_calls=3, backward_calls=3,
                    source_dtypes=["torch.float32"] * 3, reduction_dtype="torch.float32"))
                self.assertEqual(row["gradient_tensors"], len(expected))
                self.assertAlmostEqual(row["backward_loss"], sum(losses.values()) / 48, delta=2e-8)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad, expected[name], atol=3e-8, rtol=3e-4,
                                               msg=lambda msg: f"{variant}/{name}: {msg}")
                for label in "AB":
                    torch.testing.assert_close(state["outputs"][label], outputs[label], atol=2e-6, rtol=2e-5)

    def test_case_rejects_a_mode_flag_without_the_actual_intervention(self):
        model, items = fixture()
        original = layouts.packed_velocity

        def missing_intervention(*args, **kwargs):
            kwargs["modulation_reduction"] = "native"
            return original(*args, **kwargs)

        with patch.object(layouts, "packed_velocity", side_effect=missing_intervention):
            with self.assertRaisesRegex(ValueError, "modulation reduction forward/backward coverage"):
                layouts.run_case(model, items, "cudnn_packed_ab", padded_length=7,
                    compute_dtype=torch.float32, sdpa_backend=SDPBackend.MATH,
                    capture_stages=STAGES, profile=False, modulation_reduction="fp32")

    def test_fa3_wiring_keeps_bf16_forward_exact_with_three_backward_interventions(self):
        model, items = fixture()
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        # This CPU stand-in checks FA3 routing only; it does not validate FA3's
        # numerical kernels, which require the separately budgeted GPU run.
        with patch.object(layouts, "FlashBackend", side_effect=lambda name: layouts.metrics.SegmentedSDPA(SDPBackend.MATH)):
            for variant in ("fa3_packed_ab", "fa3_packed_ba"):
                states = {}
                for reduction in ("native", "fp32"):
                    with self.subTest(variant=variant, reduction=reduction):
                        row, states[reduction], _ = layouts.run_case(model, items, variant, padded_length=7,
                            compute_dtype=torch.bfloat16, sdpa_backend=SDPBackend.MATH,
                            capture_stages=STAGES, profile=False, modulation_reduction=reduction)
                        expected = 3 if reduction == "fp32" else 0
                        evidence = row["modulation_reduction_evidence"]
                        self.assertEqual((evidence["forward_calls"], evidence["backward_calls"]), (expected, expected))
                        self.assertEqual(evidence["source_dtypes"], ["torch.bfloat16"] * expected)
                        for name, parameter in model.named_parameters():
                            self.assertEqual(parameter._version, versions[name])
                            self.assertEqual(parameter.grad.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for label in "AB":
                    self.assertTrue(torch.equal(states["native"]["outputs"][label], states["fp32"]["outputs"][label]))
                    for stage in STAGES:
                        self.assertTrue(torch.equal(states["native"]["boundaries"][stage]["output"][label],
                                                    states["fp32"]["boundaries"][stage]["output"][label]))


if __name__ == "__main__":
    unittest.main()
