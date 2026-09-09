"""Real FLUX modules validate passive boundary capture and hook lifetime."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch
import warnings

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics import _flux_packing_forward as forward
from scripts.diagnostics import _flux_packing_forward_boundaries as boundaries
from scripts.diagnostics import probe_flux_packing as probe
from scripts.diagnostics._flux_packing import packed_processors
from tests.diagnostics.test_flux_packing import ReferenceBackend
from tests.diagnostics.test_flux_packing_forward import fixture


class RecordingCapture:
    def __init__(self, data, *, fail_at=None):
        self.lengths = tuple(prefix.shape[0] for prefix in data["prefixes"])
        self.image_tokens = data["layout"].image_tokens
        self.records = {}
        self.fail_at = fail_at

    def record(self, name, parts):
        if name == self.fail_at:
            raise RuntimeError("intentional observer failure")
        if name in self.records:
            raise AssertionError("duplicate boundary observation")
        self.records[name] = [{key: value.detach().clone() for key, value in part.items()} for part in parts]


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(137)
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(sdpa_kernel(SDPBackend.MATH))
        self.context.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message="In CPU autocast.*")

    def packed_forward(self, model, data):
        with torch.no_grad(), packed_processors(model, data["layout"], ReferenceBackend()):
            return probe.forward_velocity(model, "fa3", torch.cat(data["prefixes"]), data, torch.float32)

    def test_real_packed_boundaries_match_original_singletons_without_changing_output(self):
        model, data = fixture()
        expected = self.packed_forward(model, data)
        capture = RecordingCapture(data)
        with boundaries.boundary_hooks(model, capture, forward.split_block_output):
            actual = self.packed_forward(model, data)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(list(capture.records), [*boundaries.FIRST_BLOCK_STAGES, *boundaries.OUTPUT_STAGES])

        for index in range(2):
            singleton = forward.slice_inputs(data, index, index + 1)
            reference = RecordingCapture(singleton)
            assembled = probe.pad_flux_text_prefixes(singleton["prefixes"], padded_length=reference.lengths[0])
            with torch.no_grad(), boundaries.boundary_hooks(model, reference, forward.split_block_output):
                probe.forward_velocity(model, "sdpa_b2", assembled, singleton, torch.float32)
            for name, observations in capture.records.items():
                for stream, values in observations[index].items():
                    with self.subTest(example=index, stage=name, stream=stream):
                        torch.testing.assert_close(values, reference.records[name][0][stream], atol=2e-6, rtol=2e-5)

        # Separate text/image lengths expose an accidentally reversed attention tuple.
        attention = capture.records["transformer_blocks.0.attention_output"]
        self.assertEqual([part["text"].shape[0] for part in attention], [2, 5])
        self.assertEqual([part["image"].shape[0] for part in attention], [3, 3])
        norm_input = capture.records["norm_out_input"]
        self.assertEqual(set(norm_input[0]), {"image", "conditioning"})
        self.assertFalse(torch.equal(norm_input[0]["conditioning"], norm_input[1]["conditioning"]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_output_norm_keyword_arguments_preserve_per_example_conditioning(self):
        model, data = fixture()
        capture = RecordingCapture(data)
        images = torch.randn(2, 3, model.inner_dim)
        conditioning = torch.randn(2, model.inner_dim)
        with torch.no_grad(), boundaries.boundary_hooks(model, capture, forward.split_block_output):
            output = model.norm_out(x=images, conditioning_embedding=conditioning)
        self.assertEqual(list(capture.records), ["norm_out_input", "norm_out"])
        for index in range(2):
            torch.testing.assert_close(capture.records["norm_out_input"][index]["image"], images[index], atol=0, rtol=0)
            torch.testing.assert_close(capture.records["norm_out_input"][index]["conditioning"], conditioning[index], atol=0, rtol=0)
            torch.testing.assert_close(capture.records["norm_out"][index]["image"], output[index], atol=0, rtol=0)

    def test_observer_and_registration_failures_remove_only_owned_hooks(self):
        for failure in ("observation", "registration"):
            with self.subTest(failure=failure):
                model, data = fixture()
                block = model.transformer_blocks[0]
                modules = (block, block.attn, model.norm_out)
                original = block.register_forward_pre_hook(lambda module, args: None)
                self.addCleanup(original.remove)
                before = [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in modules]
                capture = RecordingCapture(data, fail_at="transformer_blocks.0.attention_output")
                with ExitStack() as stack:
                    if failure == "registration":
                        stack.enter_context(patch.object(model.norm_out, "register_forward_pre_hook",
                                                         side_effect=RuntimeError("intentional registration failure")))
                    with self.assertRaisesRegex(RuntimeError, "intentional"):
                        with boundaries.boundary_hooks(model, capture, forward.split_block_output):
                            self.packed_forward(model, data)
                self.assertEqual([(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in modules], before)
                self.assertIn(original.id, block._forward_pre_hooks)

    def test_stage_order_requires_a_double_block_and_surrounds_existing_outputs(self):
        block_names = ["transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0"]
        self.assertEqual(boundaries.boundary_stage_order(block_names), [
            "transformer_blocks.0.input", "transformer_blocks.0.attention_input",
            "transformer_blocks.0.attention_output", *block_names, "norm_out_input", "norm_out", "velocity"])
        for names in ([], ["single_transformer_blocks.0"]):
            with self.assertRaisesRegex(ValueError, "transformer_blocks.0"):
                boundaries.boundary_stage_order(names)
        model, data = fixture(double=0)
        with self.assertRaisesRegex(ValueError, "at least one double"):
            with boundaries.boundary_hooks(model, RecordingCapture(data), forward.split_block_output):
                self.fail("entered a capture without a double block")
