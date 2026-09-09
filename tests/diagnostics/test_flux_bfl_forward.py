"""Independent tiny BFL forwards validate observations and the padded reference."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_bfl_forward as comparison
from scripts.diagnostics import _flux_bfl_weights as weights
from scripts.diagnostics._flux_packing import packing_layout


TOLERANCE = {"output_relative_l2": 2e-5, "loss_relative": 2e-6}


def tiny_model():
    return Flux2Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=2, num_single_layers=2,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False,
    ).eval()


def prepared_inputs():
    image_positions = torch.zeros(2, 3, 4, dtype=torch.int64)
    image_positions[:, :, 2] = torch.arange(3)
    source = dict(
        prefixes=(torch.randn(2, 12, dtype=torch.bfloat16),
                  torch.randn(5, 12, dtype=torch.bfloat16)),
        flow=SimpleNamespace(noisy_latents=torch.randn(2, 3, 8, dtype=torch.bfloat16),
                             timesteps=torch.tensor([.17, .83]),
                             target_velocity=torch.randn(2, 3, 8)),
        image_positions=image_positions, layout=packing_layout((2, 5), 3, "cpu"))
    return comparison.prepare_comparison_inputs(source)


def hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in model.modules()]


class BFLForwardTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(37)

    def models(self):
        diffusers = tiny_model()
        bfl, _ = weights.build_bfl_model(diffusers)
        return diffusers, bfl.eval()

    def test_actual_bfl_and_diffusers_match_every_fp32_boundary_and_padded_batch(self):
        diffusers, bfl = self.models()
        data = prepared_inputs()
        references, reference_losses = {}, {}
        stages = ["time_embedding", "image_projection", "text_projection",
                  "double.0", "double.1", "single.0", "single.1", "final_norm", "velocity"]
        for index, label in enumerate(("A", "B")):
            inputs = comparison.call_inputs(data, (index,))
            reference, snapshots = comparison.run_case(bfl, "bfl", inputs, "fp32", TOLERANCE)
            actual, observed = comparison.run_case(
                diffusers, "diffusers", inputs, "fp32", TOLERANCE,
                references={"bfl": snapshots}, reference_losses={"bfl": reference["losses"]})
            self.assertEqual(list(observed[label]), stages)
            self.assertTrue(actual["comparison_summary"]["bfl"][label]["passed"], actual)
            for stage in stages:
                self.assertEqual(set(observed[label][stage]), set(snapshots[label][stage]))
                for stream, value in observed[label][stage].items():
                    with self.subTest(label=label, stage=stage, stream=stream):
                        torch.testing.assert_close(value, snapshots[label][stage][stream],
                                                   atol=2e-6, rtol=2e-5)
            references.update(snapshots)
            reference_losses.update(reference["losses"])
        batch, observed = comparison.run_case(
            diffusers, "diffusers", comparison.call_inputs(data, (0, 1), padded_length=7),
            "fp32", TOLERANCE, references={"bfl": references}, reference_losses={"bfl": reference_losses})
        for label, length in (("A", 2), ("B", 5)):
            self.assertTrue(batch["comparison_summary"]["bfl"][label]["passed"], batch)
            for stage in ("double.0", "double.1", "single.0", "single.1"):
                self.assertEqual(observed[label][stage]["text"].shape[0], length)
                self.assertEqual(observed[label][stage]["image"].shape[0], 3)

    def test_observation_is_passive_and_profiles_real_math_attention(self):
        diffusers, bfl = self.models()
        inputs = comparison.call_inputs(prepared_inputs(), (1,))
        for name, model, api_calls in (("diffusers", diffusers, 4), ("bfl", bfl, 8)):
            with self.subTest(implementation=name):
                before_hooks = hook_counts(model)
                before_state = {key: value.clone() for key, value in model.state_dict().items()}
                with comparison.controlled_precision("fp32", torch.device("cpu")):
                    expected = comparison.forward_call(model, name, inputs)
                row, snapshots = comparison.run_case(model, name, inputs, "fp32", TOLERANCE, profile=True)
                torch.testing.assert_close(snapshots["B"]["velocity"]["image"], expected[0], atol=0, rtol=0)
                self.assertEqual(row["operators"]["aten::scaled_dot_product_attention"], api_calls)
                # BFL's empty reference queries return before dispatching Math SDPA.
                self.assertEqual(row["operators"]["aten::_scaled_dot_product_attention_math"], 4)
                self.assertFalse(any("backward" in operator for operator in row["operators"]))
                self.assertEqual(hook_counts(model), before_hooks)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before_state[key]), key)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for block in (*bfl.double_blocks, *bfl.single_blocks):
            self.assertNotIn("forward_kv_extract", block.__dict__)

    def test_autocast_redispatch_is_not_counted_as_an_extra_attention_execution(self):
        query, key = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
        value = torch.randn_like(query).bfloat16()
        with comparison.controlled_precision("bf16", torch.device("cpu")), torch.profiler.profile() as profile:
            torch.nn.functional.scaled_dot_product_attention(query, key, value)
        operators = {event.key: event.count for event in profile.key_averages()
                     if "scaled_dot_product" in event.key}
        self.assertEqual(operators["aten::scaled_dot_product_attention"], 2)
        self.assertEqual(operators["aten::_scaled_dot_product_attention_math"], 1)
        comparison.validate_attention_operators(operators, "diffusers", 1)
        # Exact counts recorded on H200 in the first real BF16 Diffusers case.
        recorded = {"aten::scaled_dot_product_attention": 50,
                    "aten::_scaled_dot_product_attention_math": 25}
        comparison.validate_attention_operators(recorded, "diffusers", 25)
        for invalid in ({**recorded, "aten::_scaled_dot_product_attention_math": 24},
                        {**recorded, "aten::_scaled_dot_product_attention_math": 26},
                        {**recorded, "flash_attn_3::_flash_attn_forward": 1},
                        {**recorded, "aten::_scaled_dot_product_cudnn_attention": 1},
                        {**recorded, "aten::_scaled_dot_product_attention_math_backward": 1}):
            with self.subTest(operators=invalid), self.assertRaises(ValueError):
                comparison.validate_attention_operators(invalid, "diffusers", 25)

    def test_bfl_direct_method_and_hooks_restore_after_injected_failure(self):
        _, bfl = self.models()
        inputs = comparison.call_inputs(prepared_inputs(), (0,))
        block = bfl.double_blocks[0]
        before_hooks = hook_counts(bfl)
        existing = block.forward_kv_extract
        def fail(*args, **kwargs):
            raise RuntimeError("injected BFL block failure")
        block.forward_kv_extract = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "injected BFL block failure"):
                comparison.run_case(bfl, "bfl", inputs, "fp32", TOLERANCE)
            self.assertIs(block.forward_kv_extract, fail)
            self.assertEqual(hook_counts(bfl), before_hooks)
            for other in (*bfl.double_blocks[1:], *bfl.single_blocks):
                self.assertNotIn("forward_kv_extract", other.__dict__)
        finally:
            del block.forward_kv_extract
        self.assertEqual(block.forward_kv_extract, existing)
        comparison.run_case(bfl, "bfl", inputs, "fp32", TOLERANCE)

    def test_masked_padding_and_other_examples_cannot_change_valid_streams(self):
        model = tiny_model()
        inputs = comparison.call_inputs(prepared_inputs(), (0, 1), padded_length=7)
        _, reference = comparison.run_case(model, "diffusers", inputs, "fp32", TOLERANCE)
        inputs["text"][~inputs["text_mask"]] = 10000
        _, padded = comparison.run_case(model, "diffusers", inputs, "fp32", TOLERANCE)
        for label in ("A", "B"):
            for stage, streams in reference[label].items():
                for stream, value in streams.items():
                    torch.testing.assert_close(padded[label][stage][stream], value, atol=0, rtol=0)
        inputs["text"][0, :2] += 10
        inputs["images"][0] *= -3
        inputs["timesteps"][0] = .95
        _, changed = comparison.run_case(model, "diffusers", inputs, "fp32", TOLERANCE)
        for stage, streams in reference["B"].items():
            for stream, value in streams.items():
                torch.testing.assert_close(changed["B"][stage][stream], value, atol=0, rtol=0)
        self.assertFalse(torch.equal(changed["A"]["velocity"]["image"], reference["A"]["velocity"]["image"]))

    def test_precision_cases_reuse_identical_fp32_inputs_and_parameters(self):
        diffusers, bfl = self.models()
        data = prepared_inputs()
        identity = comparison.input_identity(data)
        inputs = comparison.call_inputs(data, (0,))
        observed = []
        original = comparison.forward_call
        def observe(model, implementation, inputs):
            observed.append({key: comparison.fingerprint(value) for key, value in inputs.items()
                             if isinstance(value, torch.Tensor)})
            return original(model, implementation, inputs)
        for name, model in (("diffusers", diffusers), ("bfl", bfl)):
            with self.subTest(implementation=name), patch.object(comparison, "forward_call", side_effect=observe):
                fp32, _ = comparison.run_case(model, name, inputs, "fp32", TOLERANCE)
                bf16, _ = comparison.run_case(model, name, inputs, "bf16", TOLERANCE)
            self.assertEqual(fp32["outputs"]["image_projection"]["A"]["image"]["dtype"], "torch.float32")
            self.assertEqual(bf16["outputs"]["image_projection"]["A"]["image"]["dtype"], "torch.bfloat16")
            self.assertTrue(all(parameter.dtype == torch.float32 and parameter.grad is None
                                for parameter in model.parameters()))
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(row == observed[0] for row in observed))
        self.assertEqual(comparison.input_identity(data), identity)

    def test_precision_backend_and_grad_contexts_restore_after_failure(self):
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.is_grad_enabled(),
                    torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
                    torch.is_autocast_enabled("cpu"))
        with self.assertRaisesRegex(RuntimeError, "precision context failure"):
            with comparison.controlled_precision("bf16", torch.device("cpu")):
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
                self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
                raise RuntimeError("precision context failure")
        self.assertEqual(previous, (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                         torch.backends.cudnn.allow_tf32, torch.is_grad_enabled(),
                         torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
                         torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
                         torch.is_autocast_enabled("cpu")))

    def test_correct_final_output_cannot_hide_wrong_intermediate_block(self):
        _, bfl = self.models()
        inputs = comparison.call_inputs(prepared_inputs(), (0,))
        reference, snapshots = comparison.run_case(bfl, "bfl", inputs, "fp32", TOLERANCE)
        block = bfl.double_blocks[0]
        original = block.forward_kv_extract
        def perturb(*args, **kwargs):
            image, text, cache = original(*args, **kwargs)
            return image + .5, text, cache
        block.forward_kv_extract = perturb
        final = bfl.final_layer.linear.register_forward_hook(
            lambda module, args, output: snapshots["A"]["velocity"]["image"].unsqueeze(0).clone())
        try:
            actual, _ = comparison.run_case(bfl, "bfl", inputs, "fp32", TOLERANCE,
                references={"reference": snapshots}, reference_losses={"reference": reference["losses"]})
        finally:
            final.remove()
            del block.forward_kv_extract
        summary = actual["comparison_summary"]["reference"]["A"]
        self.assertEqual(summary["velocity_relative_l2"], 0)
        self.assertEqual(summary["loss_relative"], 0)
        self.assertEqual(summary["first_outside_tolerance"], "double.0/image")
        self.assertFalse(summary["passed"])

    def test_original_positions_and_logical_examples_survive_singleton_selection(self):
        data = prepared_inputs()
        data["text_positions"][1][:, 3] += 11
        data["image_positions"][1, :, 0] = 3
        before = comparison.input_identity(data)
        inputs = comparison.call_inputs(data, (1,))
        self.assertEqual(inputs["labels"], ("B",))
        self.assertEqual(inputs["lengths"], (5,))
        self.assertTrue(inputs["text_mask"].all())
        for output, source in (("text", "prefixes"), ("text_positions", "text_positions"),
                               ("images", "images"), ("image_positions", "image_positions"),
                               ("target", "target")):
            torch.testing.assert_close(inputs[output][0], data[source][1], atol=0, rtol=0)
        self.assertEqual(comparison.input_identity(data), before)

    def test_bfl_reference_rejects_padded_or_batched_calls_before_forward(self):
        data = prepared_inputs()
        model = Mock()
        for inputs in (comparison.call_inputs(data, (0,), padded_length=7),
                       comparison.call_inputs(data, (0, 1))):
            with self.assertRaisesRegex(ValueError, "unpadded singleton"):
                comparison.forward_call(model, "bfl", inputs)
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
