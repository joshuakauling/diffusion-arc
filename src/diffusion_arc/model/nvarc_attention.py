"""Explicit FlashAttention-3 Varlen binding for frozen NVARC conditioning."""

from importlib import import_module

NVARC_ATTENTION_IMPLEMENTATION = "nvarc_fa3"


def register_nvarc_attention() -> None:
    """Make FA3 available to NVARC under the name "nvarc_fa3".

    Call this during model setup, before loading NVARC. Transformers stores
    our forward function under this name and calls it from NVARC's attention
    layers. The function passes Q, K and V and the packed source boundaries
    to FA3. These boundaries keep sources separate without a large attention
    mask. FLUX keeps its own attention implementation.
    """

    from transformers import AttentionInterface

    kernel = import_module("flash_attn_3.flash_attn_interface").flash_attn_varlen_func

    def forward(module, query, key, value, attention_mask, scaling=None, **kwargs):
        q, k, v = (tensor.transpose(1, 2).squeeze(0).contiguous()
                   for tensor in (query, key, value))
        output = kernel(
            q, k, v,
            cu_seqlens_q=kwargs["cu_seq_lens_q"],
            cu_seqlens_k=kwargs["cu_seq_lens_k"],
            max_seqlen_q=kwargs["max_length_q"],
            max_seqlen_k=kwargs["max_length_k"],
            softmax_scale=scaling, causal=True,
        )
        return output.unsqueeze(0), None

    AttentionInterface.register(NVARC_ATTENTION_IMPLEMENTATION, forward)
