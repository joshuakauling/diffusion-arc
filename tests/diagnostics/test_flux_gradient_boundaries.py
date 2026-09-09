"""Analytic edge errors and passive real-autograd boundary observation."""

from contextlib import ExitStack
import json
import math
import unittest
from unittest.mock import patch
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics._flux_gradient_boundaries import GradientBoundaryCapture, token_error_metrics
from scripts.diagnostics import probe_flux_packing as probe
from scripts.diagnostics._flux_packing import packed_processors
from scripts.diagnostics._flux_packing_forward import split_block_output
from tests.diagnostics.test_flux_packing import ReferenceBackend
from tests.diagnostics.test_flux_packing_forward import fixture


class TokenMetricTests(unittest.TestCase):
    def test_known_edge_spike_is_localized_without_token_relative_division(self):
        reference = torch.ones(20, 2)
        reference[9] = 0
        actual = reference.clone()
        actual[9] = torch.tensor([3., 4.])
        result = token_error_metrics(actual, reference, text_length=10, edge_width=2)
        self.assertEqual(result["top_tokens"][0]["position"], 9)
        self.assertEqual(result["top_tokens"][0]["modality"], "text")
        self.assertIn("text_before_image", result["top_tokens"][0]["regions"])
        self.assertAlmostEqual(result["per_token"]["error_rms"][9], math.sqrt(12.5))
        self.assertEqual(result["per_token"]["max_abs"][9], 4.)
        self.assertEqual(result["per_token"]["reference_rms"][9], 0.)
        self.assertAlmostEqual(result["per_token"]["error_rms_over_global_reference_rms"][9],
                               math.sqrt(12.5) / math.sqrt(.95))
        self.assertEqual(result["regions"]["boundary_union"]["token_count"], 8)
        self.assertEqual(result["regions"]["boundary_union"]["squared_error_fraction"], 1.)
        self.assertEqual(result["regions"]["interior"]["max_abs"], 0.)
        self.assertAlmostEqual(result["regions"]["boundary_union"]["error_energy_enrichment"], 2.5)
        json.dumps(result, allow_nan=False)

    def test_uniform_error_has_no_boundary_enrichment_and_short_regions_do_not_double_count(self):
        reference = torch.ones(20, 2)
        result = token_error_metrics(reference + 2, reference, text_length=10, edge_width=2)
        self.assertEqual(result["regions"]["boundary_union"]["error_energy_enrichment"], 1.)
        self.assertEqual(result["regions"]["interior"]["error_energy_enrichment"], 1.)
        for text_length in (0, 3, 5):
            result = token_error_metrics(torch.ones(5, 2), torch.ones(5, 2), text_length=text_length)
            self.assertEqual(result["regions"]["boundary_union"]["token_count"], 5)
            self.assertEqual(result["regions"]["interior"]["token_count"], 0)
            self.assertIsNone(result["regions"]["interior"]["error_rms"])
            self.assertEqual(result["regions"]["boundary_union"]["squared_error_fraction"], 0.)
            if text_length in (0, 5):
                self.assertEqual(result["regions"]["text_before_image"]["token_count"], 0)
                self.assertEqual(result["regions"]["image_after_text"]["token_count"], 0)
            json.dumps(result, allow_nan=False)

    def test_zero_reference_remains_explicit_and_bad_contracts_fail(self):
        result = token_error_metrics(torch.ones(3, 2), torch.zeros(3, 2), text_length=3)
        self.assertTrue(result["reference_is_zero"])
        self.assertFalse(result["error_is_zero"])
        self.assertIsNone(result["per_token"]["error_rms_over_global_reference_rms"])
        self.assertIsNone(result["regions"]["all"]["error_rms_over_global_reference_rms"])
        json.dumps(result, allow_nan=False)
        for value in (torch.ones(2), torch.ones(0, 2), torch.ones(3, 1),
                      torch.full((3, 2), float("nan")), torch.ones(3, 2, dtype=torch.int64)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    token_error_metrics(value, torch.ones(3, 2), text_length=1)
        for kwargs in ({"text_length": -1}, {"text_length": 4}, {"text_length": True},
                       {"text_length": 1, "edge_width": 0}):
            with self.assertRaises(ValueError):
                token_error_metrics(torch.ones(3, 2), torch.ones(3, 2), **kwargs)


class GradientCaptureTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(281)
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(sdpa_kernel(SDPBackend.MATH))
        self.context.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="In CPU autocast.*")

    def test_real_native_and_packed_backward_are_unchanged_and_padding_is_removed(self):
        model, data = fixture(head_dim=8)
        stages = {"double.0": "transformer_blocks.0", "single.0": "single_transformer_blocks.0",
                  "single.1": "single_transformer_blocks.1",
                  "text.mlp.linear_out": "transformer_blocks.0.ff_context.linear_out"}
        lengths = (2, 5)

        for packed in (False, True):
            def canonicalize(stage, output):
                if stage == "text.mlp.linear_out":
                    parts = output[0].split(lengths) if packed else [output[i, :n] for i, n in enumerate(lengths)]
                    return dict(zip(("A", "B"), parts, strict=True))
                if not packed:
                    text, images = output if isinstance(output, tuple) else (output[:, :7], output[:, 7:])
                    return {label: torch.cat((text[i, :length], images[i]))
                            for i, (label, length) in enumerate(zip(("A", "B"), lengths, strict=True))}
                kind = "double" if stage.startswith("double") else "single"
                parts = split_block_output(kind, output, lengths, 3)
                return {label: torch.cat((part["text"], part["image"]))
                        for label, part in zip(("A", "B"), parts, strict=True)}

            def call():
                assembled = torch.cat(data["prefixes"]) if packed else probe.pad_flux_text_prefixes(data["prefixes"], padded_length=7)
                with ExitStack() as stack:
                    if packed:
                        stack.enter_context(packed_processors(model, data["layout"], ReferenceBackend()))
                    return probe.forward_velocity(model, "fa3" if packed else "sdpa_b2", assembled, data, torch.float32)

            model.zero_grad(set_to_none=True)
            expected = call()
            expected.square().mean().backward()
            expected_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            capture = GradientBoundaryCapture(model, canonicalize, stages=stages)
            with capture:
                actual = call()
                actual.square().mean().backward()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for name, p in model.named_parameters():
                if name in expected_gradients:
                    torch.testing.assert_close(p.grad, expected_gradients[name], atol=0, rtol=0)
            self.assertTrue(capture.completed)
            self.assertEqual(set(capture.values), set(stages))
            for stage, observed in capture.values.items():
                for label, length in zip(("A", "B"), lengths, strict=True):
                    wanted = length if stage.startswith("text") else length + 3
                    self.assertEqual(observed["output"][label].shape, (wanted, 8))
                    self.assertEqual(observed["gradient"][label].shape, (wanted, 8))
                    self.assertFalse(observed["gradient"][label].requires_grad)
                    self.assertEqual(observed["gradient"][label].device.type, "cpu")
            self.assertEqual(torch.count_nonzero(capture.values["single.1"]["gradient"]["A"][:2]).item(), 0)
            self.assertEqual(capture.unused_output_paths["single.1"], [])

    def test_analytic_output_gradient_and_unused_branch_are_explicit(self):
        class Branch(torch.nn.Module):
            def forward(self, value):
                return value * 2, value * 3
        model = torch.nn.ModuleDict({"branch": Branch()})
        tensor = torch.randn(4, 2, requires_grad=True)
        capture = GradientBoundaryCapture(model, lambda stage, tree: {"A": torch.cat(tree)}, stages={"test": "branch"})
        with capture:
            first, second = model["branch"](tensor)
            (second * torch.arange(4).reshape(4, 1)).sum().backward()
        self.assertEqual(capture.unused_output_paths, {"test": [[0]]})
        expected = torch.cat((torch.zeros(4, 2), torch.arange(4).reshape(4, 1).expand(4, 2)))
        torch.testing.assert_close(capture.values["test"]["gradient"]["A"], expected)
        tensor.data.zero_()
        self.assertFalse(torch.equal(capture.values["test"]["output"]["A"], torch.zeros(8, 2)))
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            with capture:
                pass

    def test_failures_remove_only_owned_hooks_and_missing_backward_is_not_zero(self):
        model = torch.nn.ModuleDict({"one": torch.nn.Linear(2, 2), "two": torch.nn.Linear(2, 2)})
        original = model["one"].register_forward_hook(lambda module, args, output: None)
        self.addCleanup(original.remove)
        canon = lambda stage, value: {"A": value}
        for failure in ("missing_backward", "duplicate", "registration", "canonicalization"):
            capture = GradientBoundaryCapture(model, canon, stages={"one": "one"})
            with self.subTest(failure=failure), self.assertRaises((RuntimeError, ValueError)):
                with ExitStack() as stack:
                    if failure == "registration":
                        capture.stages["two"] = "two"
                        stack.enter_context(patch.object(model["two"], "register_forward_hook", side_effect=RuntimeError("registration")))
                    if failure == "canonicalization":
                        capture.canonicalize = lambda stage, output: {"A": output.flatten()}
                    with capture:
                        value = model["one"](torch.randn(3, 2))
                        if failure == "duplicate":
                            model["one"](torch.randn(3, 2))
                        if failure == "canonicalization":
                            value.sum().backward()
            self.assertEqual(list(model["one"]._forward_hooks), [original.id])
            self.assertEqual(len(model["two"]._forward_hooks), 0)
            self.assertFalse(capture.completed)


if __name__ == "__main__":
    unittest.main()
