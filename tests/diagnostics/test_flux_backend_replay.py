"""Verify the isolated attention comparison without installed CUDA extensions."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics import _flux_backend_replay as replay
from scripts.diagnostics import _flux_packing as packing


class AttentionReplayTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        generator = torch.Generator().manual_seed(59)
        self.qkv = tuple(torch.randn(1, 5, 2, 8, dtype=torch.bfloat16, generator=generator)
                         for _ in range(3))

    def test_math_forward_and_vjp_match_explicit_attention(self):
        leaves = tuple(t.float().requires_grad_() for t in self.qkv)
        upstream = torch.linspace(-1, 1, leaves[0].numel()).reshape_as(leaves[0])
        expected = ((leaves[0].transpose(1, 2) @ leaves[1].transpose(1, 2).transpose(-1, -2))
                    * 8 ** -0.5).softmax(-1) @ leaves[2].transpose(1, 2)
        expected = expected.transpose(1, 2)
        expected_grads = torch.autograd.grad(expected, leaves, upstream)
        actual = replay.attention_forward("fp32_math", *leaves)
        actual_grads = torch.autograd.grad(actual, leaves, upstream)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-6)

    def test_real_math_replay_profiles_both_precisions_and_preserves_inputs(self):
        before = tuple(t.clone() for t in self.qkv)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            result = replay.compare_backends(*self.qkv, 43, profile=True,
                                             backends=("fp32_math", "bf16_math"))
        self.assertTrue(result["input_hashes_unchanged"])
        self.assertFalse(result["model_gradients_measured"])
        self.assertEqual(result["segment_lengths"], [5])
        for label in ("output", "dQ", "dK", "dV"):
            self.assertEqual(result["backends"]["fp32_math"][label]["relative_l2"], 0)
            self.assertLess(result["backends"]["bf16_math"][label]["relative_l2"], .02)
            self.assertEqual(result["backends"]["bf16_math"]["comparisons_vs_bf16_math"][label]["max_abs"], 0)
        for row in result["backends"].values():
            self.assertTrue(row["backend_verified"])
            self.assertTrue(row["inputs_unchanged"])
            self.assertFalse(row["cuda_kernel_capture_available"])
            self.assertEqual(row["evidence_level"], "backend_operators_only")
        for original, saved in zip(self.qkv, before):
            torch.testing.assert_close(original, saved, atol=0, rtol=0)
            self.assertIsNone(original.grad)

    def test_same_upstream_gradient_is_used_even_when_outputs_differ(self):
        def offset_attention(name, q, k, v):
            return q + k + v + (10 if name == "bf16_math" else 0)
        with patch.object(replay, "attention_forward", side_effect=offset_attention):
            result = replay.compare_backends(*self.qkv, 43, backends=("fp32_math", "bf16_math"))
        row = result["backends"]["bf16_math"]
        self.assertGreater(row["output"]["rms"], 9)
        for label in ("dQ", "dK", "dV"):
            self.assertEqual(row[label]["max_abs"], 0)
        expected = torch.randn(self.qkv[0].shape, dtype=torch.bfloat16,
                               generator=torch.Generator().manual_seed(43))
        self.assertEqual(result["upstream_gradient"], replay.fingerprint(expected))

    def test_official_flash_varlen_bindings_receive_single_segment_and_backward(self):
        for name in ("fa2", "fa3", "fa4"):
            with self.subTest(backend=name):
                def kernel(q, k, v, **kwargs):
                    self.assertEqual(q.shape, (5, 2, 8))
                    self.assertTrue(q.is_contiguous())
                    self.assertFalse(kwargs["causal"])
                    self.assertEqual(kwargs["softmax_scale"], 8 ** -0.5)
                    for key in ("cu_seqlens_q", "cu_seqlens_k"):
                        self.assertEqual(kwargs[key].dtype, torch.int32)
                        self.assertEqual(kwargs[key].tolist(), [0, 5])
                    self.assertEqual(kwargs["max_seqlen_q"], 5)
                    self.assertEqual(kwargs["max_seqlen_k"], 5)
                    output = q + 2 * k + 3 * v
                    return (output, None) if name == "fa4" else output
                fake = SimpleNamespace(__name__=packing.FlashBackend.MODULES[name],
                                       flash_attn_varlen_func=kernel)
                leaves = tuple(t.clone().requires_grad_() for t in self.qkv)
                with patch.object(packing, "import_module", return_value=fake):
                    actual = replay.attention_forward(name, *leaves)
                grads = torch.autograd.grad(actual, leaves, torch.ones_like(actual))
                self.assertEqual(actual.shape, (1, 5, 2, 8))
                for grad, scale in zip(grads, (1, 2, 3)):
                    torch.testing.assert_close(grad, torch.full_like(grad, scale), atol=0, rtol=0)

    def test_cudnn_backend_is_forced_with_exact_noncausal_scale(self):
        with patch.object(replay, "sdpa_kernel", return_value=nullcontext()) as scope:
            with patch.object(replay.F, "scaled_dot_product_attention", side_effect=lambda q, k, v, **kw: q) as call:
                actual = replay.attention_forward("cudnn", *self.qkv)
        scope.assert_called_once_with(replay.SDPBackend.CUDNN_ATTENTION)
        self.assertEqual(call.call_args.kwargs, dict(dropout_p=0.0, is_causal=False, scale=8 ** -0.5))
        torch.testing.assert_close(actual, self.qkv[0], atol=0, rtol=0)

    def test_fa4_autograd_events_are_observable_without_cuda_kernel_capture(self):
        class FlashAttnVarlenFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v):
                return q + k + v

            @staticmethod
            def backward(ctx, incoming):
                return incoming, incoming, incoming

        fake = SimpleNamespace(__name__=packing.FlashBackend.MODULES["fa4"],
                               flash_attn_varlen_func=lambda q, k, v, **kwargs:
                                   (FlashAttnVarlenFunc.apply(q, k, v), None))
        with patch.object(packing, "import_module", return_value=fake):
            values, evidence = replay._evaluate("fa4", self.qkv, torch.ones_like(self.qkv[0]), True)
        self.assertEqual(evidence["operators"]["FlashAttnVarlenFunc"], 1)
        self.assertEqual(evidence["operators"]["FlashAttnVarlenFuncBackward"], 1)
        self.assertEqual(evidence["evidence_level"], "backend_operators_only")
        self.assertEqual(values[0].dtype, torch.bfloat16)
        for gradient in values[1:]:
            torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=0, rtol=0)

    def test_profile_validation_rejects_fallback_and_marks_missing_cuda_evidence(self):
        for name in ("fa2", "fa3", "fa4", "cudnn"):
            operators = {f"flux_backend_replay::{name}": 1, "SomeBackward": 1}
            if name in replay.FLASH_OPERATORS:
                operators.update(dict.fromkeys(replay.FLASH_OPERATORS[name], 1))
            if name == "cudnn":
                operators.update({"aten::_scaled_dot_product_cudnn_attention": 1,
                                  "aten::_scaled_dot_product_cudnn_attention_backward": 1})
            if name == "fa4":
                operators.update({"FlashAttnVarlenFunc": 1, "FlashAttnVarlenFuncBackward": 1})
            kernels = ["flash_attn_fwd_kernel", "flash_attn_bwd_kernel"]
            evidence = replay.validate_profile(operators, kernels, name, cuda=True)
            self.assertTrue(evidence["cuda_kernel_capture_available"])
            missing = replay.validate_profile(operators, [], name, cuda=True)
            self.assertFalse(missing["cuda_kernel_capture_available"])
            self.assertEqual(missing["evidence_level"], "backend_operators_only")
            with self.assertRaisesRegex(RuntimeError, "unexpected.*math"):
                replay.validate_profile({**operators, "aten::_scaled_dot_product_attention_math": 1},
                                        kernels, name, cuda=True)
        with self.assertRaisesRegex(RuntimeError, "missing CUDA attention bwd"):
            replay.validate_profile({"flux_backend_replay::fa4": 1, "FlashAttnVarlenFunc": 1,
                                     "FlashAttnVarlenFuncBackward": 1},
                                    ["flash_attn_fwd_kernel"], "fa4", cuda=True)
        with self.assertRaisesRegex(RuntimeError, "missing explicit FlashAttnVarlenFuncBackward"):
            replay.validate_profile({"flux_backend_replay::fa4": 1, "FlashAttnVarlenFunc": 1,
                                     "SomeBackward": 1}, [], "fa4", cuda=True)

    def test_mutating_backend_is_rejected_and_caller_tensors_stay_unchanged(self):
        before = tuple(t.clone() for t in self.qkv)
        def mutating_attention(name, q, k, v):
            with torch.no_grad():
                q.add_(1)
            return q + k + v
        with patch.object(replay, "attention_forward", side_effect=mutating_attention):
            with self.assertRaisesRegex(RuntimeError, "fp32_math.*mutated a Q/K/V input"):
                replay.compare_backends(*self.qkv, 43, backends=("fp32_math", "bf16_math"))
        for original, saved in zip(self.qkv, before):
            torch.testing.assert_close(original, saved, atol=0, rtol=0)

    def test_ingestion_rejects_wrong_dtype_nonfinite_and_unsupported_backend(self):
        with self.assertRaisesRegex(ValueError, "contiguous BF16"):
            replay.compare_backends(*(t.float() for t in self.qkv), 43)
        with self.assertRaisesRegex(ValueError, "require CUDA"):
            replay.compare_backends(*self.qkv, 43)
        with self.assertRaisesRegex(ValueError, "start with fp32_math"):
            replay.compare_backends(*self.qkv, 43, backends=("bf16_math",))
        with self.assertRaisesRegex(ValueError, "63-bit"):
            replay.compare_backends(*self.qkv, -1, backends=("fp32_math",))
        self.qkv[0][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite replay inputs"):
            replay.compare_backends(*self.qkv, 43, backends=("fp32_math",))

    def test_metrics_cover_zero_reference_absolute_error_and_nonfinite(self):
        row = replay.tensor_metrics(torch.tensor([1., 2.]), torch.zeros(2))
        self.assertIsNone(row["relative_l2"])
        self.assertEqual(row["max_abs"], 2)
        self.assertAlmostEqual(row["actual_l2"], 5 ** .5)
        self.assertAlmostEqual(row["rms"], 2.5 ** .5)
        with self.assertRaisesRegex(FloatingPointError, "non-finite attention"):
            replay.tensor_metrics(torch.tensor([float("nan")]), torch.ones(1))

    def test_precision_state_restores_when_backend_raises(self):
        before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32,
                  torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(), torch.is_grad_enabled())
        with patch.object(replay, "attention_forward", side_effect=ValueError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "fp32_math.*test failure"):
                replay.compare_backends(*self.qkv, 43, backends=("fp32_math",))
        after = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32,
                 torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(), torch.is_grad_enabled())
        self.assertEqual(before, after)
