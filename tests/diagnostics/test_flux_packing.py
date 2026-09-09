"""Small production FLUX models verify diagnostic packing without CUDA."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from scripts.diagnostics import _flux_packing as packing


class ReferenceBackend:
    """Explicit CPU test oracle; never available as a diagnostic backend."""
    def __call__(self, q, k, v, layout):
        lengths = [n + layout.image_tokens for n in layout.text_lengths]
        return torch.cat([
            F.scaled_dot_product_attention(a.transpose(0, 1), b.transpose(0, 1), c.transpose(0, 1))
            .transpose(0, 1)
            for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))
        ])


def tiny_model(head_dim=8):
    return Flux2Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
        attention_head_dim=head_dim, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(head_dim // 4,) * 4, guidance_embeds=False,
    ).eval()


def inputs(device="cpu"):
    image = torch.randn(2, 3, 8, device=device)
    text = torch.randn(2, 5, 12, device=device)
    image_ids = torch.zeros(2, 3, 4, dtype=torch.int64, device=device)
    image_ids[:, :, 2] = torch.arange(3, device=device)
    text_ids = torch.zeros(2, 5, 4, dtype=torch.int64, device=device)
    text_ids[:, :, 3] = torch.arange(5, device=device)
    mask = torch.tensor([[True, True, False, False, False, True, True, True],
                         [True] * 8], device=device)[:, None, None, :]
    return dict(hidden_states=image, encoder_hidden_states=text,
                timestep=torch.tensor([.17, .83], device=device), img_ids=image_ids,
                txt_ids=text_ids, guidance=None, joint_attention_kwargs={"attention_mask": mask})


def packed_call(model, data, backend=None):
    layout = packing.packing_layout((2, 5), 3, data["hidden_states"].device)
    with packing.packed_processors(model, layout, backend or ReferenceBackend()):
        return packing.packed_velocity(
            model, torch.cat((data["encoder_hidden_states"][0, :2], data["encoder_hidden_states"][1])),
            data["hidden_states"], data["img_ids"], data["timestep"], layout,
        )


class PackedFluxTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(23)

    def test_mixed_lengths_timesteps_outputs_all_parameter_and_input_gradients(self):
        model, data = tiny_model(), inputs()
        data["hidden_states"].requires_grad_()
        data["encoder_hidden_states"].requires_grad_()
        baseline = model(**data).sample
        baseline.square().mean().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        input_gradients = {k: data[k].grad.clone() for k in ("hidden_states", "encoder_hidden_states")}
        model.zero_grad(set_to_none=True)
        for key in input_gradients:
            data[key].grad = None
        actual = packed_call(model, data)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)
        self.assertEqual(set(gradients), {n for n, p in model.named_parameters() if p.grad is not None})
        for name, parameter in model.named_parameters():
            if name in gradients:
                with self.subTest(parameter=name):
                    torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=2e-4)
        for key, gradient in input_gradients.items():
            torch.testing.assert_close(data[key].grad, gradient, atol=2e-6, rtol=2e-4)

    def test_source_isolation_and_padding_values_do_not_leak(self):
        model, data = tiny_model(), inputs()
        with torch.no_grad():
            expected = packed_call(model, data)
            data["encoder_hidden_states"][0, 2:] = 10000
            torch.testing.assert_close(model(**data).sample, expected, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(packed_call(model, data), expected, atol=0, rtol=0)
            data["encoder_hidden_states"][0, :2] += 10
            data["hidden_states"][0] *= 3
            data["timestep"][0] = .95
            actual = packed_call(model, data)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
            self.assertFalse(torch.equal(actual[0], expected[0]))

    def test_cuda_style_fp32_qk_norms_enter_flash_as_bf16_with_gradients(self):
        class Float32Norm(torch.nn.Module):
            """Reproduce CUDA RMSNorm autocast policy on a CPU-only test host."""
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(self, x):
                with torch.autocast("cpu", enabled=False):
                    return self.original(x.float())

        observed = []
        class StrictBackend(ReferenceBackend):
            def __call__(self, q, k, v, layout):
                observed.append((q.dtype, k.dtype, v.dtype))
                if (q.dtype, k.dtype, v.dtype) != (torch.bfloat16,) * 3:
                    raise AssertionError("FlashAttention requires uniform BF16 kernel inputs")
                return super().__call__(q, k, v, layout)

        model, data = tiny_model(), inputs()
        for block in (*model.transformer_blocks, *model.single_transformer_blocks):
            for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                if hasattr(block.attn, name):
                    setattr(block.attn, name, Float32Norm(getattr(block.attn, name)))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = model(**data).sample
        expected.float().square().mean().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = packed_call(model, data, StrictBackend())
        actual.float().square().mean().backward()
        self.assertEqual(observed, [(torch.bfloat16,) * 3] * 2)
        torch.testing.assert_close(actual, expected, atol=.008, rtol=.02)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            if name in gradients:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        error = sum((p.grad - gradients[n]).double().square().sum()
                    for n, p in model.named_parameters() if n in gradients)
        scale = sum(g.double().square().sum() for g in gradients.values())
        self.assertLess(float((error / scale).sqrt()), .02)

    def test_existing_block_forwards_and_fused_projection_process_only_real_tokens(self):
        model, data, shapes = tiny_model(), inputs(), {}
        def record(name):
            def hook(module, args, kwargs, output):
                shapes[name] = tuple((args[0] if args else kwargs["hidden_states"]).shape)
            return hook
        modules = {"double": model.transformer_blocks[0], "single": model.single_transformer_blocks[0],
                   "context": model.context_embedder,
                   "fused": model.single_transformer_blocks[0].attn.to_qkv_mlp_proj}
        handles = [m.register_forward_hook(record(n), with_kwargs=True) for n, m in modules.items()]
        try:
            packed_call(model, data)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes["context"][:2], (1, 7))
        self.assertEqual(shapes["double"][:2], (1, 6))
        self.assertEqual(shapes["single"][:2], (1, 13))
        self.assertEqual(shapes["fused"][:2], (1, 13))

    def test_processors_restore_after_failure_and_parameters_are_shared(self):
        model = tiny_model()
        attention = [b.attn for b in (*model.transformer_blocks, *model.single_transformer_blocks)]
        original = [a.processor for a in attention]
        parameters = {n: id(p) for n, p in model.named_parameters()}
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with packing.packed_processors(model, packing.packing_layout((2, 5), 3, "cpu"), ReferenceBackend()):
                raise RuntimeError("test failure")
        self.assertEqual([a.processor for a in attention], original)
        self.assertEqual({n: id(p) for n, p in model.named_parameters()}, parameters)

    def test_layout_boundaries_order_inverse_and_positions(self):
        layout = packing.packing_layout((2, 5), 3, "cpu")
        self.assertEqual(layout.cumulative.tolist(), [0, 5, 13])
        self.assertEqual(layout.joint_order.tolist(), [0, 1, 7, 8, 9, 2, 3, 4, 5, 6, 10, 11, 12])
        self.assertEqual(layout.text_positions[:, 3].tolist(), [0, 1, 0, 1, 2, 3, 4])
        self.assertEqual(layout.joint_samples.tolist(), [0] * 5 + [1] * 8)
        joint = torch.arange(13)[layout.joint_order]
        torch.testing.assert_close(joint[layout.text_indices], torch.arange(7))
        torch.testing.assert_close(joint[layout.image_indices], torch.arange(7, 13))

    def test_official_backend_binding_noncausal_varlen_and_autograd(self):
        layout = packing.packing_layout((2, 5), 3, "cpu")
        for name in ("fa2", "fa3", "fa4"):
            def kernel(q, k, v, **kwargs):
                self.assertFalse(kwargs["causal"])
                self.assertIs(kwargs["cu_seqlens_q"], layout.cumulative)
                self.assertEqual(kwargs["max_seqlen_q"], 8)
                out = q + k + v
                return (out, None) if name == "fa4" else out
            with patch.object(packing, "import_module", return_value=SimpleNamespace(
                    __name__=packing.FlashBackend.MODULES[name], flash_attn_varlen_func=kernel)):
                backend = packing.FlashBackend(name)
            q = torch.randn(13, 2, 8, requires_grad=True)
            backend(q, q, q, layout).sum().backward()
            torch.testing.assert_close(q.grad, torch.full_like(q, 3))
