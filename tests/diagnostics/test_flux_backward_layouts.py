"""Compare tiny real FLUX backward layouts against an independent weighted loss."""

import json
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_backward_layouts as layouts
from scripts.diagnostics import _flux_backend_replay as replay


STAGES = {"double.0": "transformer_blocks.0", "single.0": "single_transformer_blocks.0",
          "text.mlp.linear_out": "transformer_blocks.0.ff_context.linear_out"}


def fixture():
    model = Flux2Transformer2DModel(in_channels=128, out_channels=128, num_layers=1,
        num_single_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    items = []
    for label, length, timestep in (("A", 2, .17), ("B", 5, .83)):
        image_positions = torch.zeros(1, 3, 4, dtype=torch.int64)
        # Native Diffusers batches share the first example's RoPE grid.
        image_positions[:, :, 2] = torch.arange(3)
        text_positions = torch.zeros(1, length, 4, dtype=torch.int64)
        text_positions[:, :, 3] = torch.arange(length)
        items.append(dict(images=torch.randn(1, 3, 128), text=torch.randn(1, length, 12),
            target=torch.randn(1, 3, 128), timesteps=torch.tensor([timestep]),
            image_positions=image_positions, text_positions=text_positions, labels=(label,)))
    return model, items


def independent_weighted_backward(model, items, global_batch_size=48):
    """Use native singleton calls and the explicit MSE definition, without layout helpers."""
    parameters = tuple(model.parameters())
    combined = [torch.zeros_like(parameter) for parameter in parameters]
    outputs, losses = {}, {}
    with sdpa_kernel(SDPBackend.MATH):
        for item in items:
            output = model(hidden_states=item["images"], encoder_hidden_states=item["text"],
                timestep=item["timesteps"], img_ids=item["image_positions"],
                txt_ids=item["text_positions"], guidance=None, return_dict=True).sample
            loss = F.mse_loss(output.float(), item["target"].float(), reduction="sum") / item["target"].numel()
            gradients = torch.autograd.grad(loss / global_batch_size, parameters)
            for accumulated, gradient in zip(combined, gradients):
                accumulated.add_(gradient)
            label = item["labels"][0]
            outputs[label] = output[0].detach().clone()
            losses[label] = float(loss.detach())
    return dict(zip(dict(model.named_parameters()), combined)), outputs, losses


class BackwardLayoutTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(192)

    def run_case(self, model, items, variant, **options):
        return layouts.run_case(model, items, variant, padded_length=7, compute_dtype=torch.float32,
            sdpa_backend=SDPBackend.MATH, profile=False, capture_stages=STAGES, **options)

    def test_all_cudnn_layouts_match_independent_microbatch_parameter_gradients(self):
        model, items = fixture()
        expected, outputs, losses = independent_weighted_backward(model, items)
        before = {name: (parameter.data_ptr(), parameter._version, parameter.detach().clone())
                  for name, parameter in model.named_parameters()}
        baseline_state = None
        for variant in ("cudnn_single", "cudnn_padded_single", "cudnn_padded_batch",
                        "cudnn_packed_ab", "cudnn_packed_ba"):
            with self.subTest(variant=variant):
                # Stale gradients must be cleared once per logical comparison, then accumulated.
                for parameter in model.parameters():
                    parameter.grad = torch.full_like(parameter, 500)
                row, state, _ = self.run_case(model, items, variant)
                self.assertEqual(row["backward_calls"], 2 if variant.endswith("single") else 1)
                self.assertEqual(row["sample_weights"], {"A": 1 / 48, "B": 1 / 48})
                self.assertAlmostEqual(row["logical_mean_loss"], sum(losses.values()) / 2, places=6)
                self.assertAlmostEqual(row["backward_loss"], sum(losses.values()) / 48, delta=2e-8)
                self.assertEqual(row["gradient_tensors"], len(expected))
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad, expected[name], atol=3e-8, rtol=3e-4,
                                               msg=lambda msg: f"{variant}/{name}: {msg}")
                    pointer, version, saved = before[name]
                    self.assertEqual(parameter.data_ptr(), pointer)
                    self.assertEqual(parameter._version, version)
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    torch.testing.assert_close(parameter, saved, atol=0, rtol=0)
                for label in "AB":
                    torch.testing.assert_close(state["outputs"][label], outputs[label], atol=2e-6, rtol=2e-5)
                if baseline_state is None:
                    baseline_state = state
                else:
                    comparisons = layouts.compare_states(state, baseline_state, {"A": 2, "B": 5}, edge_width=1)
                    json.dumps(comparisons, allow_nan=False)
                    for stage in STAGES:
                        for field in ("output", "gradient"):
                            for label in "AB":
                                torch.testing.assert_close(state["boundaries"][stage][field][label],
                                    baseline_state["boundaries"][stage][field][label], atol=2e-6, rtol=3e-4)
                for record in row["padding_checks"]:
                    self.assertEqual(record["nonzero_gradient_elements"], 0)

    def test_global_batch_weight_changes_all_gradients_without_changing_outputs(self):
        model, items = fixture()
        row48, state48, _ = self.run_case(model, items, "cudnn_packed_ba", global_batch_size=48)
        gradients48 = layouts.gradient_snapshot(model)
        row24, state24, _ = self.run_case(model, items, "cudnn_packed_ba", global_batch_size=24)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients48[name] * 2, atol=0, rtol=0)
        self.assertEqual(row24["backward_loss"], row48["backward_loss"] * 2)
        for label in "AB":
            torch.testing.assert_close(state24["outputs"][label], state48["outputs"][label], atol=0, rtol=0)

    def test_backward_disables_autocast_but_retains_tf32_controls(self):
        model, items = fixture()
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        phases = {"forward": [], "backward": []}

        def record(phase):
            phases[phase].append((torch.is_autocast_enabled("cpu"), torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32))

        forward = model.context_embedder.register_forward_pre_hook(lambda *_: record("forward"))
        backward = model.context_embedder.weight.register_hook(lambda _: record("backward"))
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            layouts.run_case(model, items, "cudnn_padded_batch", padded_length=7,
                compute_dtype=torch.bfloat16, sdpa_backend=SDPBackend.MATH,
                profile=False, capture_stages=STAGES)
            self.assertEqual(phases["forward"], [(True, False, False)])
            self.assertEqual(phases["backward"], [(False, False, False)])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            forward.remove()
            backward.remove()
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
            torch.backends.cudnn.allow_tf32 = previous[2]

    def test_padded_singletons_mask_actual_padding_and_preserve_image_keys(self):
        model, items = fixture()
        with patch.object(model, "forward", wraps=model.forward) as forward:
            row, _, _ = self.run_case(model, items, "cudnn_padded_single")
        self.assertEqual([record["positions"] for record in row["padding_checks"]], [5, 2])
        self.assertEqual(len(forward.call_args_list), 2)
        for call, length in zip(forward.call_args_list, (2, 5)):
            mask = call.kwargs["joint_attention_kwargs"]["attention_mask"]
            expected = torch.tensor([True] * length + [False] * (7 - length) + [True] * 3)
            torch.testing.assert_close(mask[0, 0, 0], expected, atol=0, rtol=0)

        inputs = layouts.assemble(items, (0,), padded_length=7)
        with sdpa_kernel(SDPBackend.MATH):
            original = layouts.native_forward(model, inputs)
            changed = {**inputs, "text": inputs["text"].detach().clone()}
            changed["text"][~inputs["text_mask"]] = 10000
            perturbed = layouts.native_forward(model, changed)
        torch.testing.assert_close(perturbed, original, atol=0, rtol=0)

    def test_ba_assembly_and_observation_preserve_sample_and_timestep_identity(self):
        model, items = fixture()
        marked = [{**item, "image_positions": item["image_positions"].clone()} for item in items]
        marked[1]["image_positions"][:, :, 1] = 99
        inputs = layouts.assemble(marked, (1, 0))
        self.assertEqual(inputs["labels"], ("B", "A"))
        self.assertEqual(inputs["lengths"], (5, 2))
        for row, original in enumerate((marked[1], marked[0])):
            for key in ("timesteps", "images", "target", "image_positions"):
                torch.testing.assert_close(inputs[key][row:row + 1], original[key], atol=0, rtol=0)
            torch.testing.assert_close(inputs["text"][row, :original["text"].shape[1]], original["text"][0], atol=0, rtol=0)

        _, reference, _ = self.run_case(model, items, "cudnn_single")
        _, reversed_state, _ = self.run_case(model, items, "cudnn_packed_ba")
        for stage in STAGES:
            for label, length in (("A", 2), ("B", 5)):
                expected_length = length if stage == "text.mlp.linear_out" else length + 3
                self.assertEqual(reversed_state["boundaries"][stage]["gradient"][label].shape[0], expected_length)
                torch.testing.assert_close(reversed_state["boundaries"][stage]["gradient"][label],
                                           reference["boundaries"][stage]["gradient"][label], atol=2e-7, rtol=3e-4)
        changed = [{**item} for item in items]
        changed[1]["timesteps"] = torch.tensor([.31])
        _, changed_state, _ = self.run_case(model, changed, "cudnn_packed_ba")
        torch.testing.assert_close(changed_state["outputs"]["A"], reversed_state["outputs"]["A"], atol=0, rtol=0)
        self.assertFalse(torch.equal(changed_state["outputs"]["B"], reversed_state["outputs"]["B"]))

    def test_gradient_comparison_distinguishes_norm_and_direction_without_aliasing(self):
        model, items = fixture()
        self.run_case(model, items, "cudnn_single")
        snapshot = layouts.gradient_snapshot(model)
        for name, parameter in model.named_parameters():
            self.assertNotEqual(parameter.grad.data_ptr(), snapshot[name].data_ptr())
            parameter.grad.mul_(2)
        result = layouts.compare_gradients(model, snapshot)
        self.assertAlmostEqual(result["total"]["relative_l2"], 1.0)
        self.assertAlmostEqual(result["total"]["norm_ratio"], 2.0)
        self.assertAlmostEqual(result["total"]["cosine"], 1.0)
        self.assertLess(result["total"]["angle_degrees"], 1e-5)
        self.assertAlmostEqual(sum(row["share_of_squared_error"] for row in result["components"].values()), 1.0)
        json.dumps(result, allow_nan=False)

    def test_case_plan_has_22_backwards_and_at_most_two_retained_references(self):
        cases = layouts.case_plan()
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(case["backward_calls"] for case in cases), 22)
        cache, peak = set(), 0
        for index, case in enumerate(cases):
            self.assertTrue(set(case["references"]).issubset(cache), case)
            future = layouts.retained_after(cases, index)
            cache.intersection_update(future)
            if case["name"] in future:
                cache.add(case["name"])
            peak = max(peak, len(cache))
            self.assertLessEqual(len(cache), 2, case)
        self.assertEqual(cache, set())
        self.assertEqual(peak, 2)
        for backend in ("cudnn", "fa3"):
            row = next(case for case in cases if case["name"] == f"{backend}_packed_ba/0")
            self.assertEqual(row["labels"], ["B", "A"])
            self.assertIn(f"{backend}_packed_ab/0", row["references"])

    def test_fa3_singleton_and_packed_wiring_with_explicit_cpu_kernel_mock(self):
        model, items = fixture()
        observed = []

        class BF16MathStandIn:
            def __init__(self, name):
                self_test.assertEqual(name, "fa3")

            def __call__(self, q, k, v, layout):
                self_test.assertEqual((q.dtype, k.dtype, v.dtype), (torch.bfloat16,) * 3)
                lengths = tuple((layout.cumulative[1:] - layout.cumulative[:-1]).tolist())
                observed.append(lengths)
                with sdpa_kernel(SDPBackend.MATH):
                    return torch.cat([F.scaled_dot_product_attention(
                        a.transpose(0, 1), b.transpose(0, 1), c.transpose(0, 1), is_causal=False,
                        scale=q.shape[-1] ** -0.5).transpose(0, 1)
                        for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))])

        self_test = self
        states, gradients = {}, {}
        with patch.object(layouts, "FlashBackend", BF16MathStandIn), \
                patch.object(replay, "FlashBackend", BF16MathStandIn):
            for variant in ("fa3_single", "fa3_packed_ab", "fa3_packed_ba"):
                _, states[variant], _ = layouts.run_case(model, items, variant, padded_length=7,
                    compute_dtype=torch.bfloat16, sdpa_backend=SDPBackend.MATH,
                    profile=False, capture_stages=STAGES)
                gradients[variant] = layouts.gradient_snapshot(model)
        self.assertEqual(observed, [(5,), (5,), (8,), (8,), (5, 8), (5, 8), (8, 5), (8, 5)])
        for variant in ("fa3_packed_ab", "fa3_packed_ba"):
            for label in "AB":
                torch.testing.assert_close(states[variant]["outputs"][label], states["fa3_single"]["outputs"][label],
                                           atol=.016, rtol=.03)
            error = sum((gradient.double() - gradients["fa3_single"][name].double()).square().sum()
                        for name, gradient in gradients[variant].items())
            norm = sum(gradient.double().square().sum() for gradient in gradients["fa3_single"].values())
            self.assertLess(float((error / norm).sqrt()), .025)
        for label in "AB":
            torch.testing.assert_close(states["fa3_packed_ba"]["outputs"][label],
                                       states["fa3_packed_ab"]["outputs"][label], atol=0, rtol=0)
            for stage in STAGES:
                torch.testing.assert_close(states["fa3_packed_ba"]["boundaries"][stage]["gradient"][label],
                    states["fa3_packed_ab"]["boundaries"][stage]["gradient"][label], atol=1e-7, rtol=.02)


if __name__ == "__main__":
    unittest.main()
