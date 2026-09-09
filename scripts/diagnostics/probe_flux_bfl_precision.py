#!/usr/bin/env python3
"""Isolate BFL/Diffusers precision differences with small, synthetic CPU controls.

Runs original first Double blocks and independent Q/K normalization/RoPE
replays. No checkpoints, remote services, optimizer or production edits.
The CUDA RMSNorm replay models its FP32 autocast rule, not CUDA execution.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc._third_party.diffusers_flux2 import transformer_flux2 as df
from scripts.diagnostics._flux_bfl_forward import controlled_precision
from scripts.diagnostics._flux_bfl_weights import build_bfl_model, validate_source
from scripts.diagnostics._flux_packing_forward import difference, fingerprint

validate_source()
from scripts.diagnostics._third_party.bfl_flux2 import model as bfl


VARIANTS = ("native", "final_cast", "bfl_norm", "fused_qkv", "bfl_rope",
            "bfl_norm+fused_qkv", "bfl_norm+bfl_rope", "bfl_norm+fused_qkv+bfl_rope")
STAGES = (
    *(f"norm1.{s}" for s in ("image", "text")),
    *(f"modulated.{s}" for s in ("image", "text")),
    *(f"projection.{s}.{q}" for s in ("image", "text") for q in "qkv"),
    *(f"normalized.{s}.{q}" for s in ("image", "text") for q in "qk"),
    "rope.q", "rope.k", "attention",
    *(f"attention_projection.{s}" for s in ("image", "text")),
    *(f"attention_residual.{s}" for s in ("image", "text")),
    *(f"mlp.{s}" for s in ("image", "text")),
    "block.image", "block.text",
)


def fixture(seed):
    """Share FP32 weights and once-drawn block inputs; retain head dimension 128."""
    torch.manual_seed(seed)
    model = df.Flux2Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, num_single_layers=0,
        attention_head_dim=128, num_attention_heads=2, joint_attention_dim=12,
        axes_dims_rope=(32, 32, 32, 32), guidance_embeds=False,
    ).eval().requires_grad_(False)
    # Unit scales hide the rounding boundary before learned scale multiplication.
    for module in model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.weight.uniform_(0.5, 1.5)
    reference, mapping = build_bfl_model(model)
    inputs = {"image": torch.randn(1, 17, 256), "text": torch.randn(1, 13, 256),
              "image_mod": torch.randn(1, 1, 1536) * 0.1,
              "text_mod": torch.randn(1, 1, 1536) * 0.1}
    for stream, length in (("image", 17), ("text", 13)):
        inputs[f"{stream}_ids"] = torch.stack(
            [torch.arange(length) * (axis + 1) for axis in range(4)], dim=-1)[None]
    return model, reference, inputs, mapping


def rope_coefficients(reference, inputs):
    pe_image = reference.pe_embedder(inputs["image_ids"])
    pe_text = reference.pe_embedder(inputs["text_ids"])
    joint = torch.cat((pe_text, pe_image), dim=2)
    cos = joint[0, 0, :, :, 0, 0].repeat_interleave(2, dim=-1)
    sin = joint[0, 0, :, :, 1, 0].repeat_interleave(2, dim=-1)
    return pe_image, pe_text, joint, (cos, sin)


def compare(actual, reference):
    if set(actual) != set(reference):
        raise ValueError("incomplete precision observation coverage")
    return {name: {**difference(actual[name], value),
                   "actual_dtype": str(actual[name].dtype),
                   "reference_dtype": str(value.dtype)} for name, value in reference.items()}


def block_run(model, reference, inputs, precision, variant, implementation="diffusers"):
    """Observe native block operations; interventions affect only the diagnostic call."""
    if variant not in VARIANTS or implementation not in ("bfl", "diffusers"):
        raise ValueError("unknown implementation or precision intervention")
    if implementation == "bfl" and variant != "native":
        raise ValueError("the BFL reference must remain unmodified")
    snapshots = {}

    def record(name, value):
        if name in snapshots:
            raise ValueError(f"duplicate precision observation: {name}")
        snapshots[name] = value.detach().clone()

    def observe(module, name, pre=False):
        if pre:
            handle = module.register_forward_pre_hook(lambda m, args: record(name, args[0]))
        else:
            handle = module.register_forward_hook(lambda m, args, value: record(name, value))
        stack.callback(handle.remove)

    def projected(stream, values):
        for name, value in zip("qkv", values):
            record(f"projection.{stream}.{name}", value.unflatten(-1, (2, 128)))

    def normalized(stream, values):
        for name, value in zip("qk", values):
            record(f"normalized.{stream}.{name}", value.transpose(1, 2))

    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    image, text, mod_image, mod_text = (
        inputs[name].to(dtype) for name in ("image", "text", "image_mod", "text_mod"))
    pe_image, pe_text, _, bfl_freqs = rope_coefficients(reference, inputs)
    native_freqs = model.pos_embed(torch.cat((inputs["text_ids"], inputs["image_ids"]), dim=1)[0])
    freqs = bfl_freqs if "bfl_rope" in variant else native_freqs
    ref_block, block = reference.double_blocks[0], model.transformer_blocks[0]
    with controlled_precision(precision, torch.device("cpu")), ExitStack() as stack:
        if implementation == "bfl":
            for stream, short in (("image", "img"), ("text", "txt")):
                attention = getattr(ref_block, f"{short}_attn")
                observe(getattr(ref_block, f"{short}_norm1"), f"norm1.{stream}")
                observe(attention.qkv, f"modulated.{stream}", pre=True)
                handle = attention.qkv.register_forward_hook(
                    lambda m, args, value, stream=stream: projected(stream, value.chunk(3, -1)))
                stack.callback(handle.remove)
                handle = attention.norm.register_forward_hook(
                    lambda m, args, value, stream=stream: normalized(stream, value))
                stack.callback(handle.remove)
                observe(attention.proj, f"attention_projection.{stream}")
                observe(getattr(ref_block, f"{short}_norm2"), f"attention_residual.{stream}", pre=True)
                observe(getattr(ref_block, f"{short}_mlp"), f"mlp.{stream}")
            original_rope, original_attention = bfl.apply_rope, bfl.causal_attn_fn

            def rope(q, k, pe):
                values = original_rope(q, k, pe)
                for name, value in zip("qk", values):
                    record(f"rope.{name}", value.transpose(1, 2))
                return values

            def attention(*args, **kwargs):
                value = original_attention(*args, **kwargs)
                record("attention", value)
                return value

            stack.enter_context(patch.object(bfl, "apply_rope", rope))
            stack.enter_context(patch.object(bfl, "causal_attn_fn", attention))
            image_out, text_out, _ = ref_block.forward_kv_extract(
                image, text, pe_image, pe_text, df.Flux2Modulation.split(mod_image, 2),
                df.Flux2Modulation.split(mod_text, 2), num_ref_tokens=0)
        else:
            for stream, suffix, short in (("image", "", "img"), ("text", "_context", "txt")):
                observe(getattr(block, f"norm1{suffix}"), f"norm1.{stream}")
                observe(getattr(block, f"norm2{suffix}"), f"attention_residual.{stream}", pre=True)
                observe(getattr(block, f"ff{suffix}"), f"mlp.{stream}")
                output = block.attn.to_out[0] if stream == "image" else block.attn.to_add_out
                observe(output, f"attention_projection.{stream}")
                for q, full in (("q", "query"), ("k", "key")):
                    norm = getattr(block.attn, f"norm_{q}" if stream == "image" else f"norm_added_{q}")
                    if "bfl_norm" in variant:
                        bfl_norm = getattr(getattr(ref_block, f"{short}_attn").norm, f"{full}_norm")
                        stack.enter_context(patch.object(norm, "forward",
                            lambda x, bfl_norm=bfl_norm: bfl_norm(x).to(x.dtype)))
                    elif variant == "final_cast":
                        original = norm.forward
                        stack.enter_context(patch.object(norm, "forward",
                            lambda x, original=original: original(x).to(x.dtype)))
                    observe(norm, f"normalized.{stream}.{q}")
            original_projection = df._get_qkv_projections
            original_rope, original_attention = df.apply_rotary_emb, df.dispatch_attention_fn
            rope_calls = 0

            def projection(attn, hidden_states, encoder_hidden_states):
                record("modulated.image", hidden_states)
                record("modulated.text", encoder_hidden_states)
                if "fused_qkv" in variant:
                    values = (*ref_block.img_attn.qkv(hidden_states).chunk(3, -1),
                              *ref_block.txt_attn.qkv(encoder_hidden_states).chunk(3, -1))
                else:
                    values = original_projection(attn, hidden_states, encoder_hidden_states)
                projected("image", values[:3])
                projected("text", values[3:])
                return values

            def rope(*args, **kwargs):
                nonlocal rope_calls
                value = original_rope(*args, **kwargs)
                record(f"rope.{'qk'[rope_calls]}", value)
                rope_calls += 1
                return value

            def attention(*args, **kwargs):
                value = original_attention(*args, **kwargs)
                record("attention", value.flatten(2, 3))
                return value

            stack.enter_context(patch.object(df, "_get_qkv_projections", projection))
            stack.enter_context(patch.object(df, "apply_rotary_emb", rope))
            stack.enter_context(patch.object(df, "dispatch_attention_fn", attention))
            text_out, image_out = block(image, text, mod_image, mod_text, image_rotary_emb=freqs)
        record("block.image", image_out)
        record("block.text", text_out)
    if set(snapshots) != set(STAGES) or any(not torch.isfinite(v).all() for v in snapshots.values()):
        raise ValueError("incomplete or non-finite first-block observations")
    return {name: snapshots[name] for name in STAGES}


def cuda_norm_replay(reference, inputs, seed):
    """Replay CUDA's RMSNorm FP32 promotion on CPU using identical BF16 Q/K/V.

    Explicit casts model the dtype rule. They do not simulate CUDA kernels,
    performance, gradients, or the full transformer's autocast policy.
    """
    generator = torch.Generator().manual_seed(seed + 1000)
    q, k, v = (torch.randn(1, 2, 30, 128, generator=generator).bfloat16() for _ in range(3))
    norm = reference.double_blocks[0].img_attn.norm
    native_q, native_k = (torch.nn.RMSNorm(128, eps=1e-6) for _ in range(2))
    native_q.weight = torch.nn.Parameter(norm.query_norm.scale.detach(), requires_grad=False)
    native_k.weight = torch.nn.Parameter(norm.key_norm.scale.detach(), requires_grad=False)
    _, _, pe, _ = rope_coefficients(reference, inputs)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        reference_stages = None
        results = {}
        for variant in ("bfl", "cuda_native_norm", "cuda_norm_final_cast", "bfl_norm"):
            if variant in ("bfl", "bfl_norm"):
                nq, nk = norm(q, k, v)
            else:
                nq, nk = native_q(q.float()), native_k(k.float())
                if variant == "cuda_norm_final_cast":
                    nq, nk = nq.to(v), nk.to(v)
            rq, rk = bfl.apply_rope(nq, nk, pe)
            # Both routes enter Math SDPA as BF16 under CUDA autocast.
            aq, ak = rq.to(v), rk.to(v)
            attention = F.scaled_dot_product_attention(aq, ak, v, is_causal=False)
            stages = dict(normalized_q=nq, normalized_k=nk, rope_q=rq, rope_k=rk,
                          kernel_q=aq, kernel_k=ak, attention=attention)
            if reference_stages is None:
                reference_stages = stages
            results[variant] = compare(stages, reference_stages)
        return {"scope": "CPU replay of CUDA RMSNorm dtype rule; shared BFL RoPE coefficients",
                "input_fingerprints": {n: fingerprint(t) for n, t in zip("qkv", (q, k, v))},
                "comparisons": results}


def run(seeds):
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative 31-bit integers")
    source = validate_source()
    cases = []
    with torch.no_grad():
        for seed in seeds:
            model, reference, inputs, mapping = fixture(seed)
            state_before = {n: fingerprint(t) for n, t in model.state_dict().items()}
            identities = {n: fingerprint(t) for n, t in inputs.items()}
            for precision in ("fp32", "bf16"):
                snapshots = block_run(model, reference, inputs, precision, "native", "bfl")
                for variant in VARIANTS:
                    values = block_run(model, reference, inputs, precision, variant)
                    metrics = compare(values, snapshots)
                    cases.append(dict(seed=seed, precision=precision, variant=variant, comparisons=metrics,
                        first_nonidentical=next((name for name, row in metrics.items()
                                                if not row["bitwise_equal"]), None)))
            cases.append(dict(seed=seed, cuda_norm_replay=cuda_norm_replay(reference, inputs, seed)))
            if (state_before != {n: fingerprint(t) for n, t in model.state_dict().items()}
                    or identities != {n: fingerprint(t) for n, t in inputs.items()}):
                raise ValueError("precision controls modified source weights or inputs")
            cases.append(dict(seed=seed, input_fingerprints=identities, weight_fingerprints=state_before,
                              mapping=mapping, model_config=dict(model.config)))
    paths = [Path(__file__), Path(df.__file__), Path(bfl.__file__),
             Path(inspect.getfile(df.apply_rotary_emb)), Path(inspect.getfile(torch.nn.RMSNorm)),
             Path(inspect.getfile(inspect.unwrap(build_bfl_model))),
             Path(inspect.getfile(inspect.unwrap(controlled_precision))),
             Path(inspect.getfile(difference)),
             Path(torch.__file__).parent / "include/ATen/autocast_mode.h"]
    return dict(schema_version=1, status="complete", scope="synthetic CPU first-block controls",
                seeds=seeds, device="cpu", platform=platform.platform(), torch=torch.__version__,
                python=platform.python_version(), num_threads=torch.get_num_threads(),
                packages={name: importlib.metadata.version(name) for name in ("diffusers", "einops")},
                torch_git_revision=torch.version.git_version, source=source,
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                sdpa_backend="math", parameter_dtype="torch.float32", gradients=False,
                cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 73, 109])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(args.seeds)
    result["command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Completed synthetic CPU precision controls: {args.output}")


if __name__ == "__main__":
    main()
