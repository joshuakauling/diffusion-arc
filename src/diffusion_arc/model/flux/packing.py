"""Independent packed FLUX sequences using FA3 and the existing block modules.

No model weights, block forwards, or global dispatch functions are replaced.
Attention processors are installed explicitly and restored after each case.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from itertools import accumulate

import torch

from diffusion_arc._third_party.diffusers_flux2.transformer_flux2 import (
    Flux2Transformer2DModel,
    _get_qkv_projections,
    apply_rotary_emb,
)
from diffusion_arc._constants import NVARC_CONTEXT_WIDTH, NVARC_MAX_TOKENS

from ..contracts import TARGET_LATENT_CHANNELS, TARGET_LATENT_TOKEN_COUNT
from .modulation import modulation_gather


@dataclass(frozen=True)
class PackingLayout:
    """Constructor-validated per-example token order and modulation ownership.

    Create layouts with ``packing_layout`` and treat their tensors as immutable.
    Every segment contains its example's valid text followed by its image tokens.
    """
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
    if not lengths or any(type(n) is not int or not 1 <= n <= NVARC_MAX_TOKENS for n in lengths):
        raise ValueError(f"text lengths must be Python integers between 1 and {NVARC_MAX_TOKENS}")
    if type(image_tokens) is not int or not 1 <= image_tokens <= TARGET_LATENT_TOKEN_COUNT:
        raise ValueError(f"image_tokens must be a Python integer between 1 and {TARGET_LATENT_TOKEN_COUNT}")
    text_total = sum(lengths)
    if text_total + len(lengths) * image_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("cumulative sequence length exceeds CUDA int32 capacity")
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


class Flash3Backend:
    """Explicit Hopper BF16 FA3 varlen binding; construction verifies backward.

    The deployment pins the FA3 build. Import or capability failures are fatal;
    this adapter never silently chooses another attention implementation.
    """

    name = "fa3"
    module_name = "flash_attn_3.flash_attn_interface"

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("packed FA3 requires an available CUDA Hopper device")
        if torch.cuda.get_device_capability(self.device)[0] != 9:
            raise RuntimeError("packed FA3 requires Hopper compute capability 9.x")
        if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (12, 3):
            raise RuntimeError("packed FA3 requires CUDA 12.3 or newer")
        try:
            module = import_module(self.module_name)
            self.kernel = module.flash_attn_varlen_func
        except (ImportError, AttributeError, OSError) as error:
            raise RuntimeError("the pinned FA3 varlen forward/backward build is unavailable") from error
        self.binding = f"{module.__name__}.flash_attn_varlen_func"
        self._preflight()

    def _preflight(self):
        """Exercise two unequal BF16/head-128 segments without consuming RNG."""
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast("cuda", enabled=False):
            layout = packing_layout((2, 4), 1, self.device)
            values = torch.arange(8 * 2 * 128, device=self.device, dtype=torch.float32).reshape(8, 2, 128)
            q, k, v = [(values * scale).sin().to(torch.bfloat16).requires_grad_()
                       for scale in (.01, .02, .03)]
            try:
                output = self(q, k, v, layout)
                if output.shape != q.shape or output.dtype != torch.bfloat16:
                    raise RuntimeError("FA3 preflight returned an incompatible output")
                gradients = torch.autograd.grad(output.float().square().mean(), (q, k, v))
                if not all(bool(torch.isfinite(tensor).all().item()) for tensor in (output, *gradients)):
                    raise RuntimeError("FA3 preflight produced non-finite output or gradients")
            except Exception as error:
                raise RuntimeError("FA3 BF16 varlen head-128 forward/backward preflight failed") from error

    def __call__(self, q, k, v, layout):
        return self.kernel(
            q, k, v, cu_seqlens_q=layout.cumulative, cu_seqlens_k=layout.cumulative,
            max_seqlen_q=layout.maximum, max_seqlen_k=layout.maximum,
            softmax_scale=q.shape[-1] ** -0.5, causal=False,
        )


def validate_packed_transformer(model, compute_dtype):
    """Validate stable architecture and precision contracts before training."""
    if not isinstance(model, Flux2Transformer2DModel):
        raise TypeError("packed FLUX requires the repository-local Flux2Transformer2DModel")
    if compute_dtype != torch.bfloat16:
        raise ValueError("packed FA3 training requires BF16 compute")
    if model.gradient_checkpointing:
        raise ValueError("packed FLUX does not support gradient checkpointing")
    if model.config.guidance_embeds or model.time_guidance_embed.guidance_embedder is not None:
        raise ValueError("packed FLUX requires a model without guidance embeddings")
    if model.config.attention_head_dim != 128:
        raise ValueError("packed FA3 requires attention head dimension 128")
    if (model.config.in_channels != TARGET_LATENT_CHANNELS
            or model.out_channels != TARGET_LATENT_CHANNELS
            or model.config.patch_size != 1
            or model.config.joint_attention_dim != NVARC_CONTEXT_WIDTH):
        raise ValueError("packed FLUX requires 128-channel images and 7680-channel NVARC conditioning")
    if not model.transformer_blocks or not model.single_transformer_blocks:
        raise ValueError("packed FLUX requires both double-stream and single-stream blocks")
    if any(block.attn.inner_dim // block.attn.heads != 128
           for block in (*model.transformer_blocks, *model.single_transformer_blocks)):
        raise ValueError("all packed attention blocks require head dimension 128")
    if any(getattr(block.attn, "fused_projections", False) for block in model.transformer_blocks):
        raise ValueError("packed training requires unfused double-stream parameter projections")
    if any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("packed FLUX requires every transformer parameter trainable in FP32")


def _attention(backend, layout, q, k, v, rotary):
    q = apply_rotary_emb(q, rotary, sequence_dim=1)
    k = apply_rotary_emb(k, rotary, sequence_dim=1)
    # CUDA autocast runs Q/K RMSNorm in FP32. Direct FlashAttention calls do
    # not perform SDPA's autocast conversion: use the V projection's compute
    # dtype (BF16 in training) for all three kernel inputs. Keep the
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
        raise ValueError("packed FLUX does not support gradient checkpointing")
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


def _forward_packed_flux(model, text, images, image_positions, timesteps, layout):
    """Reuse FLUX's blocks and heads, with example-specific token modulations.

    The caller supplies concatenated *valid* text prefixes and installs packed
    processors. All metadata is validated at preparation/preflight boundaries.
    FP32 reduction changes only the three modulation-gather backward sums.
    """
    temb = model.time_guidance_embed(timesteps.to(images.dtype) * 1000, None)
    image_mod = modulation_gather(model.double_stream_modulation_img(temb), layout.image_samples,
                                 (layout.image_tokens,) * len(layout.text_lengths)).unsqueeze(0)
    text_mod = modulation_gather(model.double_stream_modulation_txt(temb), layout.text_samples,
                                layout.text_lengths).unsqueeze(0)
    single_mod = modulation_gather(model.single_stream_modulation(temb), layout.joint_samples,
                                  tuple(n + layout.image_tokens for n in layout.text_lengths)).unsqueeze(0)
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


def predict_packed_flux_velocity(
    transformer, latents, text, image_positions, timesteps, layout, *, compute_dtype, backend,
):
    """Run a preflighted model with valid text and independent packed examples.

    All inputs and layout metadata must already share the transformer device.
    Model parameters and their names remain unchanged, including checkpoint keys.
    """
    batch = len(layout.text_lengths)
    if tuple(latents.shape) != (batch, layout.image_tokens, transformer.config.in_channels):
        raise ValueError("latents disagree with the packed example/image layout")
    if tuple(text.shape) != (sum(layout.text_lengths), transformer.config.joint_attention_dim):
        raise ValueError("text must contain exactly the concatenated valid prefixes")
    if image_positions.shape != (batch, layout.image_tokens, 4) or image_positions.dtype != torch.int64:
        raise ValueError("image_positions must contain int64 four-axis coordinates for each image token")
    if timesteps.shape != (batch,) or not timesteps.is_floating_point():
        raise ValueError("timesteps must contain one floating-point value per example")
    device = next(transformer.parameters()).device
    if any(tensor.device != device for tensor in (latents, text, image_positions, timesteps, layout.cumulative)):
        raise ValueError("packed inputs, layout and transformer must share one device")
    if compute_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("packed velocity supports BF16 compute or explicit FP32 reference tests")
    with packed_processors(transformer, layout, backend), torch.autocast(
        device_type=device.type, dtype=compute_dtype, enabled=compute_dtype != torch.float32,
    ):
        output = _forward_packed_flux(transformer, text, latents, image_positions, timesteps, layout)
    if output.shape != latents.shape:
        raise RuntimeError("packed FLUX output shape disagrees with input latents")
    return output
