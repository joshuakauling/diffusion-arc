"""Diagnostic-only packed FLUX execution using the production block modules.

No model weights, block forwards, or global dispatch functions are replaced.
Attention processors are installed explicitly and restored after each case.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import import_module
from itertools import accumulate

import torch

from diffusion_arc._third_party.diffusers_flux2.transformer_flux2 import (
    _get_qkv_projections,
    apply_rotary_emb,
)
from scripts.diagnostics._flux_modulation_backward import modulation_gather


@dataclass(frozen=True)
class PackingLayout:
    text_lengths: tuple[int, ...]
    image_tokens: int
    cumulative: torch.Tensor
    joint_order: torch.Tensor
    text_indices: torch.Tensor
    image_indices: torch.Tensor
    text_samples: torch.Tensor
    image_samples: torch.Tensor
    joint_samples: torch.Tensor
    text_positions: torch.Tensor

    @property
    def maximum(self):
        return max(self.text_lengths) + self.image_tokens


def packing_layout(text_lengths, image_tokens, device):
    """Construct immutable segment metadata once, outside model execution."""
    lengths = tuple(text_lengths)
    if (not lengths or any(type(n) is not int or n < 1 for n in lengths)
            or type(image_tokens) is not int or image_tokens < 1):
        raise ValueError("packing requires positive text and image lengths")
    text_total = sum(lengths)
    text_offsets = (0, *accumulate(lengths))
    order, text_samples, image_samples, joint_samples = [], [], [], []
    for sample, length in enumerate(lengths):
        order.extend(range(text_offsets[sample], text_offsets[sample + 1]))
        image_start = text_total + sample * image_tokens
        order.extend(range(image_start, image_start + image_tokens))
        text_samples.extend([sample] * length)
        image_samples.extend([sample] * image_tokens)
        joint_samples.extend([sample] * (length + image_tokens))
    order = torch.tensor(order, dtype=torch.int64, device=device)
    inverse = torch.argsort(order)
    positions = torch.zeros((text_total, 4), dtype=torch.int64, device=device)
    positions[:, 3] = torch.cat([torch.arange(n, device=device) for n in lengths])
    return PackingLayout(
        lengths, image_tokens,
        torch.tensor([0, *accumulate(n + image_tokens for n in lengths)],
                     dtype=torch.int32, device=device),
        order, inverse[:text_total], inverse[text_total:],
        *(torch.tensor(ids, dtype=torch.int64, device=device)
          for ids in (text_samples, image_samples, joint_samples)),
        positions,
    )


class FlashBackend:
    MODULES = {
        "fa2": "flash_attn.flash_attn_interface",
        "fa3": "flash_attn_3.flash_attn_interface",
        "fa4": "flash_attn.cute.interface",
    }

    def __init__(self, name):
        self.name = name
        module = import_module(self.MODULES[name])
        self.kernel = module.flash_attn_varlen_func
        self.binding = f"{module.__name__}.flash_attn_varlen_func"
        self.observe = False

    def __call__(self, q, k, v, layout):
        scope = torch.profiler.record_function(f"flux_packing::{self.name}") if self.observe else nullcontext()
        with scope:
            result = self.kernel(
                q, k, v, cu_seqlens_q=layout.cumulative, cu_seqlens_k=layout.cumulative,
                max_seqlen_q=layout.maximum, max_seqlen_k=layout.maximum,
                softmax_scale=q.shape[-1] ** -0.5, causal=False,
            )
        # The pinned FA4 public interface returns (output, logsumexp).
        return result[0] if self.name == "fa4" else result


def _attention(backend, layout, q, k, v, rotary):
    q = apply_rotary_emb(q, rotary, sequence_dim=1)
    k = apply_rotary_emb(k, rotary, sequence_dim=1)
    # CUDA autocast runs Q/K RMSNorm in FP32. Direct FlashAttention calls do
    # not perform SDPA's autocast conversion: use the V projection's compute
    # dtype (BF16 in this diagnostic) for all three kernel inputs. Keep the
    # original Q dtype for the output conversion, as in the production processor.
    output = backend(*(t.squeeze(0).to(dtype=v.dtype).contiguous() for t in (q, k, v)), layout)
    return output.unsqueeze(0).flatten(2, 3).to(q.dtype)


class PackedDoubleProcessor:
    def __init__(self, layout, backend):
        self.layout, self.backend = layout, backend

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None):
        q, k, v, tq, tk, tv = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q, k, v, tq, tk, tv = (t.unflatten(-1, (attn.heads, -1)) for t in (q, k, v, tq, tk, tv))
        q, k = attn.norm_q(q), attn.norm_k(k)
        tq, tk = attn.norm_added_q(tq), attn.norm_added_k(tk)
        q, k, v = (torch.cat((text, image), dim=1).index_select(1, self.layout.joint_order)
                   for text, image in ((tq, q), (tk, k), (tv, v)))
        output = _attention(self.backend, self.layout, q, k, v, image_rotary_emb)
        text = attn.to_add_out(output.index_select(1, self.layout.text_indices))
        image = output.index_select(1, self.layout.image_indices)
        image = attn.to_out[1](attn.to_out[0](image))
        return image, text


class PackedSingleProcessor:
    def __init__(self, layout, backend):
        self.layout, self.backend = layout, backend

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = projected.split((3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor), dim=-1)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in qkv.chunk(3, dim=-1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        output = _attention(self.backend, self.layout, q, k, v, image_rotary_emb)
        return attn.to_out(torch.cat((output, attn.mlp_act_fn(mlp)), dim=-1))


@contextmanager
def packed_processors(model, layout, backend):
    """Use the existing processor extension point; restore even after failure."""
    if model.gradient_checkpointing:
        raise ValueError("this diagnostic requires activation checkpointing disabled")
    saved = []
    try:
        for blocks, processor in ((model.transformer_blocks, PackedDoubleProcessor),
                                  (model.single_transformer_blocks, PackedSingleProcessor)):
            for block in blocks:
                saved.append((block.attn, block.attn.processor))
                block.attn.set_processor(processor(layout, backend))
        yield
    finally:
        for attention, original in saved:
            attention.set_processor(original)


def packed_velocity(model, text, images, image_positions, timesteps, layout, *, modulation_reduction="native"):
    """Reuse FLUX's blocks and heads, with example-specific token modulations.

    The caller supplies concatenated *valid* text prefixes and installs packed
    processors. All metadata is validated at preparation/preflight boundaries.
    Tensor assembly, RoPE, modulation expansion and permutations are timed.
    The optional FP32 reduction changes only modulation-gather backward.
    """
    temb = model.time_guidance_embed(timesteps.to(images.dtype) * 1000, None)
    image_mod = modulation_gather(model.double_stream_modulation_img(temb), layout.image_samples,
                                 (layout.image_tokens,) * len(layout.text_lengths),
                                 reduction=modulation_reduction).unsqueeze(0)
    text_mod = modulation_gather(model.double_stream_modulation_txt(temb), layout.text_samples,
                                layout.text_lengths, reduction=modulation_reduction).unsqueeze(0)
    single_mod = modulation_gather(model.single_stream_modulation(temb), layout.joint_samples,
                                  tuple(n + layout.image_tokens for n in layout.text_lengths),
                                  reduction=modulation_reduction).unsqueeze(0)
    image = model.x_embedder(images.flatten(0, 1).unsqueeze(0))
    text = model.context_embedder(text.unsqueeze(0))
    positions = torch.cat((layout.text_positions, image_positions.flatten(0, 1)), dim=0)
    rotary = model.pos_embed(positions.index_select(0, layout.joint_order))
    for block in model.transformer_blocks:
        text, image = block(
            hidden_states=image, encoder_hidden_states=text, temb_mod_img=image_mod,
            temb_mod_txt=text_mod, image_rotary_emb=rotary,
        )
    joint = torch.cat((text, image), dim=1).index_select(1, layout.joint_order)
    for block in model.single_transformer_blocks:
        joint = block(hidden_states=joint, encoder_hidden_states=None,
                      temb_mod=single_mod, image_rotary_emb=rotary)
    image = joint.index_select(1, layout.image_indices).reshape(
        len(layout.text_lengths), layout.image_tokens, -1,
    )
    return model.proj_out(model.norm_out(image, temb))
