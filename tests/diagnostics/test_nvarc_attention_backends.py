"""Exercise diagnostic backend contracts without installing CUDA extensions."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.diagnostics import _nvarc_attention_backends as backends
from scripts.diagnostics.probe_nvarc_packing import packed_arguments, packing_layout


class BackendTests(unittest.TestCase):
    def test_official_flash_bindings_receive_thd_gqa_and_keyword_boundaries(self):
        q = torch.randn(1, 4, 5, 8)
        k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
        cumulative = torch.tensor([0, 3, 5], dtype=torch.int32)
        for backend, module in backends.FLASH_MODULES.items():
            with self.subTest(backend=backend):
                output = torch.randn(5, 4, 8)
                kernel = Mock(return_value=(output, None) if backend == "fa4" else output)
                with patch.object(backends, "import_module", return_value=SimpleNamespace(
                    __name__=module, flash_attn_varlen_func=kernel,
                )) as imported:
                    adapter = backends.PackedAttention(backend)
                result, weights = adapter.forward(
                    SimpleNamespace(is_causal=True), q, k, v, None, scaling=0.125,
                    cu_seq_lens_q=cumulative, cu_seq_lens_k=cumulative,
                    max_length_q=3, max_length_k=3,
                )
                imported.assert_called_once_with(module)
                self.assertIsNone(weights)
                torch.testing.assert_close(result[0], output)
                args, kwargs = kernel.call_args
                self.assertEqual([tuple(t.shape) for t in args], [(5, 4, 8), (5, 2, 8), (5, 2, 8)])
                self.assertTrue(all(t.is_contiguous() for t in args))
                self.assertIs(kwargs["cu_seqlens_q"], cumulative)
                self.assertEqual(kwargs["softmax_scale"], 0.125)
                self.assertTrue(kwargs["causal"])

    def test_registered_interface_preserves_causality_and_rope_with_real_qwen(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, pad_token_id=13)
        config._attn_implementation = "sdpa"
        torch.manual_seed(19)
        model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        sources = (torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        arguments, _ = packed_arguments(sources, "cpu", layout=packing_layout((3, 2), "source", 8),
                                        pad_token_id=13)
        with torch.no_grad():
            reference = model(**arguments, use_cache=False).logits

        def cpu_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                       max_seqlen_k, softmax_scale, causal):
            self.assertEqual(max_seqlen_q, 8)
            self.assertTrue(causal)
            segments = []
            for start, end in zip(cu_seqlens_q.tolist(), cu_seqlens_q.tolist()[1:]):
                qs = q[start:end].transpose(0, 1)
                ks, vs = (t[start:end].transpose(0, 1).repeat_interleave(2, dim=0) for t in (k, v))
                segments.append(torch.nn.functional.scaled_dot_product_attention(
                    qs, ks, vs, is_causal=True, scale=softmax_scale,
                ).transpose(0, 1))
            return torch.cat(segments)

        kernel = Mock(side_effect=cpu_varlen)
        with patch.object(backends, "import_module", return_value=SimpleNamespace(
            __name__="test", flash_attn_varlen_func=kernel,
        )):
            adapter = backends.PackedAttention("fa2")
        adapter.install()
        model.set_attn_implementation(adapter.name)
        with torch.no_grad():
            actual = model(**arguments, use_cache=False).logits
        self.assertEqual(kernel.call_count, 2)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    def test_cudnn_descriptors_use_ragged_offsets_and_causal_gqa(self):
        graph = Mock()
        graph.tensor.side_effect = lambda **kwargs: Mock(name=kwargs["name"])
        output = Mock()
        graph.sdpa.return_value = (output, None)
        api = SimpleNamespace(
            backend_version=Mock(return_value=92000), create_handle=Mock(return_value="handle"),
            destroy_handle=Mock(), pygraph=Mock(return_value=graph),
            data_type=SimpleNamespace(BFLOAT16="bf16", FLOAT="fp32", INT64="int64", INT32="int32"),
            heur_mode=SimpleNamespace(A="A"),
        )
        with patch.object(backends, "import_module", return_value=api), patch.object(
            torch.backends.cudnn, "version", return_value=92000,
        ):
            kernel = backends.CudnnRagged()
        actual_graph, tensors = kernel._build(2, 128, 8, 2, 128, 0.1)
        self.assertIs(actual_graph, graph)
        descriptions = {c.kwargs["name"]: c.kwargs for c in graph.tensor.call_args_list}
        self.assertEqual(descriptions["q"]["dim"], (2, 8, 128, 128))
        self.assertEqual(descriptions["q"]["stride"], (128 * 8 * 128, 128, 8 * 128, 1))
        self.assertEqual(descriptions["k"]["stride"], (128 * 2 * 128, 128, 2 * 128, 1))
        self.assertIs(descriptions["q"]["ragged_offset"], tensors[4])
        self.assertIs(descriptions["k"]["ragged_offset"], tensors[5])
        self.assertIs(descriptions["v"]["ragged_offset"], tensors[5])
        self.assertTrue(graph.sdpa.call_args.kwargs["use_causal_mask"])
        self.assertTrue(graph.sdpa.call_args.kwargs["use_padding_mask"])
        self.assertFalse(graph.sdpa.call_args.kwargs["generate_stats"])
        output.set_ragged_offset.assert_called_once_with(tensors[4])
        graph.check_support.assert_called_once()
        kernel.close()
        api.destroy_handle.assert_called_once_with("handle")

    def test_cudnn_plan_reuses_descriptors_and_rebinds_buffers(self):
        kernel = backends.CudnnRagged.__new__(backends.CudnnRagged)
        kernel.api, kernel.handle, kernel.plans = Mock(), "handle", {}
        graph = Mock()
        graph.get_workspace_size.return_value = 32
        descriptors = tuple(object() for _ in range(7))
        kernel._build = Mock(return_value=(graph, descriptors))
        q, k, v = torch.randn(5, 8, 128), torch.randn(5, 2, 128), torch.randn(5, 2, 128)
        _, offsets = packed_arguments((torch.arange(3), torch.arange(2)), "cpu")
        metadata = {"batch": 2, "q_offsets": torch.tensor(offsets) * 8 * 128,
                    "kv_offsets": torch.tensor(offsets) * 2 * 128, "lengths": torch.tensor([3, 2])}
        with patch.object(torch.cuda, "current_stream", return_value=SimpleNamespace(cuda_stream=17)):
            first = kernel(q, k, v, metadata=metadata, maximum=3, scale=0.1)
            second_q = q.clone()
            second = kernel(second_q, k, v, metadata=metadata, maximum=3, scale=0.1)
        kernel._build.assert_called_once_with(2, 3, 8, 2, 128, 0.1)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        bindings, workspace = graph.execute.call_args.args
        self.assertIs(bindings[descriptors[0]], second_q)
        self.assertIs(bindings[descriptors[3]], second)
        self.assertIs(bindings[descriptors[4]], metadata["q_offsets"])
        self.assertEqual(workspace.dtype, torch.uint8)
        self.assertEqual(graph.execute.call_args.kwargs, {"handle": "handle"})


if __name__ == "__main__":
    unittest.main()
