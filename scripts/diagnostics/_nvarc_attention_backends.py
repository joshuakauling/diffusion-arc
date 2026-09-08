"""Diagnostic-only adapters for official FA2/FA3/FA4 and cuDNN ragged attention.

Register explicit callables with Transformers instead of switching its shared
lazy FlashAttention import cache. No model layers or production runners change.
"""

from contextlib import nullcontext
from importlib import import_module

import torch

BACKENDS = ("fa2", "fa3", "fa4", "cudnn")
FLASH_MODULES = {
    "fa2": "flash_attn.flash_attn_interface",
    "fa3": "flash_attn_3.flash_attn_interface",
    "fa4": "flash_attn.cute.interface",
}


class CudnnRagged:
    """BF16 THD self-attention using cuDNN Frontend 1.13's graph API.

    Graphs depend on logical maximum lengths, never on sequence contents or
    pointer values. Device-side offsets and lengths are rebound each forward.
    The cache stores plans only; outputs and workspaces do not survive calls.
    """

    def __init__(self):
        self.api = import_module("cudnn")
        if self.api.backend_version() != torch.backends.cudnn.version():
            raise RuntimeError("cuDNN Frontend and PyTorch loaded different cuDNN versions")
        self.handle = self.api.create_handle()
        self.plans = {}

    def close(self):
        self.plans.clear()
        if self.handle is not None:
            self.api.destroy_handle(self.handle)
            self.handle = None

    def _build(self, batch, maximum, q_heads, kv_heads, dim, scale):
        api = self.api
        graph = api.pygraph(io_data_type=api.data_type.BFLOAT16,
                            intermediate_data_type=api.data_type.FLOAT,
                            compute_data_type=api.data_type.FLOAT, handle=self.handle)
        q_offsets = graph.tensor(dim=(batch + 1, 1, 1, 1), stride=(1, 1, 1, 1),
                                 data_type=api.data_type.INT64, name="q_offsets")
        kv_offsets = graph.tensor(dim=(batch + 1, 1, 1, 1), stride=(1, 1, 1, 1),
                                  data_type=api.data_type.INT64, name="kv_offsets")
        lengths = graph.tensor(dim=(batch, 1, 1, 1), stride=(1, 1, 1, 1),
                               data_type=api.data_type.INT32, name="lengths")

        def tensor(name, heads, offsets):
            # Logical BHSD descriptors over contiguous physical THD storage.
            return graph.tensor(dim=(batch, heads, maximum, dim),
                                stride=(maximum * heads * dim, dim, heads * dim, 1),
                                data_type=api.data_type.BFLOAT16,
                                ragged_offset=offsets, name=name)

        q = tensor("q", q_heads, q_offsets)
        k, v = (tensor(name, kv_heads, kv_offsets) for name in ("k", "v"))
        out, _ = graph.sdpa(q=q, k=k, v=v, generate_stats=False, attn_scale=scale,
                            use_padding_mask=True, seq_len_q=lengths, seq_len_kv=lengths,
                            use_causal_mask=True, name="nvarc_ragged_attention")
        out.set_output(True).set_data_type(api.data_type.BFLOAT16)
        out.set_dim((batch, q_heads, maximum, dim))
        out.set_stride((maximum * q_heads * dim, dim, q_heads * dim, 1))
        out.set_ragged_offset(q_offsets)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([api.heur_mode.A])
        graph.check_support()
        graph.build_plans()
        return graph, (q, k, v, out, q_offsets, kv_offsets, lengths)

    def __call__(self, q, k, v, *, metadata, maximum, scale):
        key = (metadata["batch"], maximum, q.shape[1], k.shape[1], q.shape[2], scale)
        self.api.set_stream(self.handle, torch.cuda.current_stream(q.device).cuda_stream)
        if key not in self.plans:
            self.plans[key] = self._build(*key)
        graph, tensors = self.plans[key]
        out = torch.empty_like(q)
        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=q.device)
        values = (q, k, v, out, metadata["q_offsets"], metadata["kv_offsets"], metadata["lengths"])
        graph.execute(dict(zip(tensors, values)), workspace, handle=self.handle)
        return out


class PackedAttention:
    """Bind one official backend to a named Hugging Face attention interface."""

    def __init__(self, backend):
        if backend not in BACKENDS:
            raise ValueError(f"unknown attention backend: {backend}")
        self.backend = backend
        self.name = f"nvarc_probe_{backend}"
        self.observe = False
        if backend == "cudnn":
            self.kernel = CudnnRagged()
            self.binding = "cudnn.pygraph.sdpa/execute (ragged THD)"
        else:
            module = import_module(FLASH_MODULES[backend])
            self.kernel = module.flash_attn_varlen_func
            self.binding = f"{module.__name__}.flash_attn_varlen_func"

    def install(self):
        from transformers import AttentionInterface
        # An unregistered mask interface intentionally returns None in HF 4.56.1;
        # the cumulative lengths below supply all sequence isolation information.
        AttentionInterface.register(self.name, self.forward)

    def close(self):
        if self.backend == "cudnn":
            self.kernel.close()

    def forward(self, module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, **kwargs):
        if (query.shape[0] != 1 or attention_mask is not None or dropout != 0.0
                or not module.is_causal or kwargs.get("cu_seq_lens_q") is None):
            raise ValueError("diagnostic attention requires packed causal self-attention without dropout")
        q, k, v = (t.transpose(1, 2).squeeze(0).contiguous() for t in (query, key, value))
        scope = torch.profiler.record_function(f"nvarc_probe::{self.backend}") if self.observe else nullcontext()
        with scope:
            if self.backend == "cudnn":
                out = self.kernel(q, k, v, metadata=kwargs["nvarc_cudnn_metadata"],
                                  maximum=kwargs["max_length_q"], scale=scaling)
            else:
                out = self.kernel(q, k, v, cu_seqlens_q=kwargs["cu_seq_lens_q"],
                                  cu_seqlens_k=kwargs["cu_seq_lens_k"],
                                  max_seqlen_q=kwargs["max_length_q"],
                                  max_seqlen_k=kwargs["max_length_k"],
                                  softmax_scale=scaling, causal=True)
                # The pinned FA4 API always returns (out, lse), even without stats.
                if self.backend == "fa4":
                    out, _ = out
        return out.unsqueeze(0), None
