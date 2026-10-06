"""Production packing contracts, CPU numerical oracles and fail-closed FA3 setup."""

from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from diffusion_arc.model.flux import packing
from diffusion_arc.model.flux.modulation import modulation_gather
from diffusion_arc.model.flux.transformer import Flux2Transformer2DModel


class SegmentedReference:
    """Explicit test injection; never selected as a production fallback."""

    def __call__(self, q, k, v, layout):
        lengths = tuple(n + layout.image_tokens for n in layout.text_lengths)
        return torch.cat([
            F.scaled_dot_product_attention(
                a.transpose(0, 1), b.transpose(0, 1), c.transpose(0, 1), is_causal=False,
            ).transpose(0, 1)
            for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))
        ])


def tiny_model(**overrides):
    config = dict(
        in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False,
    )
    return Flux2Transformer2DModel(**(config | overrides)).train()


def examples():
    items = []
    for index, length in enumerate((2, 5, 3)):
        images = torch.randn(1, 3, 8)
        text = torch.randn(length, 12)
        positions = torch.zeros(1, 3, 4, dtype=torch.int64)
        positions[:, :, 2] = torch.arange(3)
        items.append(dict(images=images, text=text, positions=positions,
                          timestep=torch.tensor([.17 + .3 * index]), target=torch.randn_like(images)))
    return items


def run_packed(model, items, *, dtype=torch.float32, backend=None):
    return packing.predict_packed_flux_velocity(
        model, torch.cat([item["images"] for item in items]),
        torch.cat([item["text"] for item in items]),
        torch.cat([item["positions"] for item in items]),
        torch.cat([item["timestep"] for item in items]),
        packing.packing_layout(tuple(len(item["text"]) for item in items), 3, "cpu"),
        compute_dtype=dtype, backend=backend or SegmentedReference(),
    )


class PackedFluxTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(29)

    def test_singletons_and_packs_match_outputs_global_loss_all_gradients_and_update(self):
        baseline, items = tiny_model(), examples()
        initial = deepcopy(baseline.state_dict())
        outputs, loss = [], 0.0
        for item in items:
            item["images"].requires_grad_()
            item["text"].requires_grad_()
            text_ids = torch.zeros(1, len(item["text"]), 4, dtype=torch.int64)
            text_ids[:, :, 3] = torch.arange(len(item["text"]))
            output = baseline(
                hidden_states=item["images"], encoder_hidden_states=item["text"].unsqueeze(0),
                timestep=item["timestep"], img_ids=item["positions"], txt_ids=text_ids,
                guidance=None,
            ).sample
            local_loss = F.mse_loss(output, item["target"]) / len(items)
            local_loss.backward()
            outputs.append(output.detach())
            loss += float(local_loss.detach())
        gradients = {name: parameter.grad.clone() for name, parameter in baseline.named_parameters()}
        input_gradients = [{key: item[key].grad.clone() for key in ("images", "text")} for item in items]
        torch.optim.SGD(baseline.parameters(), lr=.01).step()
        for groups in (((0, 1, 2),), ((2, 1, 0),), ((0, 1), (2,))):
            with self.subTest(groups=groups):
                model = tiny_model()
                model.load_state_dict(initial)
                current_loss = 0.0
                for item in items:
                    item["images"].grad = item["text"].grad = None
                for group in groups:
                    members = [items[index] for index in group]
                    actual = run_packed(model, members)
                    expected = torch.cat([outputs[index] for index in group])
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    local_loss = F.mse_loss(actual, torch.cat([item["target"] for item in members]))
                    local_loss = local_loss * len(group) / len(items)
                    local_loss.backward()
                    current_loss += float(local_loss.detach())
                self.assertAlmostEqual(current_loss, loss, places=6)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=3e-4, msg=name)
                for index, item in enumerate(items):
                    for key in ("images", "text"):
                        torch.testing.assert_close(item[key].grad, input_gradients[index][key], atol=2e-6, rtol=3e-4)
                torch.optim.SGD(model.parameters(), lr=.01).step()
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, baseline.state_dict()[name], atol=2e-7, rtol=2e-5, msg=name)

    def test_packed_examples_are_isolated_with_distinct_timesteps(self):
        model, items = tiny_model(), examples()
        with torch.no_grad():
            expected = run_packed(model, items)
            items[0]["text"].mul_(10)
            items[0]["images"].add_(5)
            items[0]["timestep"].fill_(.99)
            actual = run_packed(model, items)
        torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[0], expected[0]))

    def test_modulation_forward_is_native_and_bf16_backward_matches_fp64_oracle(self):
        for lengths in ((257, 131), (131, 257)):
            source = torch.randn(4, 2).to(torch.bfloat16).t().requires_grad_()
            owners = torch.tensor([owner for owner, length in enumerate(lengths) for _ in range(length)])
            gradient = torch.empty(4, len(owners), dtype=torch.bfloat16).t()
            for index in range(len(owners)):
                gradient[index] = torch.tensor((256, 1, -256, .5))[
                    [index % 3, (index + 1) % 3, (index + 2) % 3, 3]
                ]
            expected = torch.zeros(source.shape, dtype=torch.float64)
            rounded = torch.zeros_like(source)
            for index, owner in enumerate(owners.tolist()):
                expected[owner] += gradient[index].double()
                rounded[owner] += gradient[index]
            actual = modulation_gather(source, owners, lengths)
            native = source.index_select(0, owners)
            torch.testing.assert_close(actual, native, atol=0, rtol=0)
            self.assertEqual(actual.stride(), native.stride())
            actual.backward(gradient)
            torch.testing.assert_close(source.grad, expected.to(source.dtype), atol=0, rtol=0)
            self.assertFalse(torch.equal(rounded, source.grad))

    def test_cuda_style_fp32_qk_norms_cast_to_bf16_and_use_three_fp32_expansions(self):
        class FloatNorm(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(self, value):
                with torch.autocast("cpu", enabled=False):
                    return self.original(value.float())

        observations = []
        class RecordingBackend(SegmentedReference):
            def __call__(self, q, k, v, layout):
                observations.append(tuple(value.dtype for value in (q, k, v)))
                return super().__call__(q, k, v, layout)

        model = tiny_model()
        for block in (*model.transformer_blocks, *model.single_transformer_blocks):
            for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                if hasattr(block.attn, name):
                    setattr(block.attn, name, FloatNorm(getattr(block.attn, name)))
        with patch.object(packing, "modulation_gather", wraps=modulation_gather) as gather:
            output = run_packed(model, examples(), dtype=torch.bfloat16, backend=RecordingBackend())
            output.float().square().mean().backward()
        self.assertEqual(gather.call_count, 3)
        self.assertEqual(observations, [(torch.bfloat16,) * 3] * 2)
        for parameter in model.parameters():
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_processor_restoration_preserves_parameter_names_and_identity_after_failure(self):
        model = tiny_model()
        attention = [block.attn for block in (*model.transformer_blocks, *model.single_transformer_blocks)]
        processors = [item.processor for item in attention]
        parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
        def fail(*args):
            raise RuntimeError("injected kernel failure")
        with self.assertRaisesRegex(RuntimeError, "injected kernel failure"):
            run_packed(model, examples(), backend=fail)
        self.assertEqual([item.processor for item in attention], processors)
        self.assertEqual({name: id(parameter) for name, parameter in model.named_parameters()}, parameters)

    def test_layout_rejects_bad_lengths_and_constructs_independent_positions(self):
        for lengths, image_tokens in (((), 3), ((0,), 3), ((8401,), 3), ((True,), 3), ((2,), 901)):
            with self.subTest(lengths=lengths, image_tokens=image_tokens), self.assertRaises(ValueError):
                packing.packing_layout(lengths, image_tokens, "cpu")
        layout = packing.packing_layout((2, 5), 3, "cpu")
        self.assertEqual(layout.joint_token_offsets.tolist(), [0, 5, 13])
        self.assertEqual(layout.joint_token_offsets.dtype, torch.int32)
        self.assertEqual(layout.text_positions[:, 3].tolist(), [0, 1, 0, 1, 2, 3, 4])
        joint = torch.arange(13).index_select(0, layout.joint_order)
        torch.testing.assert_close(joint[layout.text_indices], torch.arange(7))
        torch.testing.assert_close(joint[layout.image_indices], torch.arange(7, 13))

    def test_backend_binding_passes_noncausal_varlen_contract_without_fallback(self):
        layout = packing.packing_layout((2, 5), 3, "cpu")
        def kernel(q, k, v, **kwargs):
            self.assertFalse(kwargs["causal"])
            self.assertIs(kwargs["cu_seqlens_q"], layout.joint_token_offsets)
            self.assertIs(kwargs["cu_seqlens_k"], layout.joint_token_offsets)
            self.assertEqual(kwargs["max_seqlen_q"], 8)
            self.assertEqual(kwargs["max_seqlen_k"], 8)
            self.assertEqual(kwargs["softmax_scale"], 128 ** -.5)
            return q + k + v
        module = SimpleNamespace(__name__=packing.Flash3Backend.module_name, flash_attn_varlen_func=kernel)
        with patch.object(torch.cuda, "is_available", return_value=True),\
             patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),\
             patch.object(torch.version, "cuda", "13.0"),\
             patch.object(packing, "import_module", return_value=module) as imported,\
             patch.object(packing.Flash3Backend, "_preflight") as preflight:
            backend = packing.Flash3Backend(torch.device("cuda"))
        imported.assert_called_once_with("flash_attn_3.flash_attn_interface")
        preflight.assert_called_once_with()
        q = torch.randn(13, 2, 128, requires_grad=True)
        backend(q, q, q, layout).sum().backward()
        torch.testing.assert_close(q.grad, torch.full_like(q, 3))

    def test_backend_preflight_rejects_missing_module_wrong_hardware_and_failed_kernel(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA Hopper"):
            packing.Flash3Backend("cpu")
        with patch.object(torch.cuda, "is_available", return_value=True),\
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),\
             self.assertRaisesRegex(RuntimeError, "Hopper"):
            packing.Flash3Backend("cuda")

    def test_preflight_executes_backward_rejects_forward_only_and_preserves_rng(self):
        backend = object.__new__(packing.Flash3Backend)
        backend.device = torch.device("cpu")
        rng = torch.get_rng_state().clone()
        kernels = (
            (lambda q, k, v, **kwargs: q + k + v, False),
            (lambda q, k, v, **kwargs: (q + k + v).detach(), True),
            (lambda q, k, v, **kwargs: q + k + v * float("nan"), True),
        )
        for kernel, fails in kernels:
            backend.kernel = kernel
            # Execute the real preflight logic with explicitly injected CPU
            # tensors; this verifies contracts, not CUDA kernel availability.
            with patch.object(torch, "autocast", return_value=nullcontext()):
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "forward/backward preflight failed"):
                        backend._preflight()
                else:
                    backend._preflight()
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)

    def test_missing_fa3_build_and_failed_backward_do_not_fall_back(self):
        with patch.object(torch.cuda, "is_available", return_value=True),\
             patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),\
             patch.object(torch.version, "cuda", "13.0"),\
             patch.object(packing, "import_module", side_effect=ImportError("missing")),\
             self.assertRaisesRegex(RuntimeError, "build is unavailable"):
            packing.Flash3Backend("cuda")
        with patch.object(torch.cuda, "is_available", return_value=True),\
             patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),\
             patch.object(torch.version, "cuda", "13.0"),\
             patch.object(packing, "import_module", return_value=SimpleNamespace(
                 __name__="test_module", flash_attn_varlen_func=lambda *args: None)),\
             patch.object(packing.Flash3Backend, "_preflight", side_effect=RuntimeError("backward unavailable")),\
             self.assertRaisesRegex(RuntimeError, "backward unavailable"):
            packing.Flash3Backend("cuda")

    def test_model_preflight_rejects_unsupported_precision_checkpointing_and_architecture(self):
        model = tiny_model()
        for dtype, checkpointing, expected in ((torch.float32, False, "BF16"),
                                               (torch.bfloat16, True, "checkpointing"),
                                               (torch.bfloat16, False, "head dimension 128")):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                model.gradient_checkpointing = checkpointing
                packing.validate_packed_transformer(model, dtype)
        production_shape = tiny_model(
            in_channels=128, out_channels=128, attention_head_dim=128,
            joint_attention_dim=7680, axes_dims_rope=(32, 32, 32, 32),
        )
        packing.validate_packed_transformer(production_shape, torch.bfloat16)
        production_shape.enable_gradient_checkpointing()
        with self.assertRaisesRegex(ValueError, "checkpointing"):
            with packing.use_packed_attention_processors(production_shape, packing.packing_layout((2,), 3, "cpu"), SegmentedReference()):
                self.fail("checkpointing must fail before installing processors")
        with self.assertRaisesRegex(ValueError, "guidance embeddings"):
            packing.validate_packed_transformer(tiny_model(guidance_embeds=True), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
