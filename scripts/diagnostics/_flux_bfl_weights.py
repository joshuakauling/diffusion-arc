"""Lossless Diffusers-to-BFL weights for an immutable, forward-only reference.

Identity mappings share storage; only fused Double QKV and final modulation
reordering allocate weights. Neither model may be updated while they coexist.
"""

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import torch


SOURCE_DIRECTORY = Path(__file__).parent / "_third_party" / "bfl_flux2"
REPOSITORY = "https://github.com/black-forest-labs/flux2"
REVISION = "50fe5162777813d869182b139e83b10743caef15"
SOURCE_FILES = {
    "model.py": ("src/flux2/model.py", "e062691d789ec80a04a5fbc421b09d3b7e264305be1dfcb6da65d68c259e18c1"),
    "LICENSE.md": ("LICENSE.md", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
}


def validate_source(directory=None):
    """Verify provenance and source bytes against independent, pinned hashes."""
    directory = SOURCE_DIRECTORY if directory is None else Path(directory)
    try:
        raw = (directory / "source.json").read_bytes()
        source = json.loads(raw)
        if source["repository"] != REPOSITORY or source["revision"] != REVISION:
            raise ValueError("BFL source repository/revision does not match the pinned reference")
        if set(source["files"]) != set(SOURCE_FILES):
            raise ValueError("BFL source file coverage does not match the pinned reference")
        for name, (upstream_path, expected_hash) in SOURCE_FILES.items():
            metadata = source["files"][name]
            if metadata != {"upstream_path": upstream_path, "sha256": expected_hash}:
                raise ValueError(f"BFL source metadata mismatch: {name}")
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected_hash:
                raise ValueError(f"BFL source hash mismatch: {name}")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"BFL source provenance is missing or malformed: {exc}") from exc
    return {**source, "source_json_sha256": hashlib.sha256(raw).hexdigest(), "validated": True}


def _parameters(model, bfl):
    config = model.config
    restrictions = {
        "patch_size": (config.patch_size, 1),
        "out_channels": (config.out_channels or config.in_channels, config.in_channels),
        "timestep_guidance_channels": (config.timestep_guidance_channels, 256),
        "eps": (config.eps, 1e-6),
    }
    for field, (actual, expected) in restrictions.items():
        if actual != expected:
            raise ValueError(f"BFL reference requires {field}={expected}, got {actual}")
    axes = list(config.axes_dims_rope)
    if not axes or any(axis <= 0 or axis % 2 for axis in axes) or sum(axes) != config.attention_head_dim:
        raise ValueError("BFL reference requires positive even RoPE axes covering the head dimension")
    if model.pos_embed.theta != config.rope_theta or list(model.pos_embed.axes_dim) != axes:
        raise ValueError("Diffusers RoPE module disagrees with its configuration")
    for name, module in model.named_modules():
        if "eps" in vars(module) and module.eps != 1e-6:
            raise ValueError(f"BFL reference requires normalization eps=1e-6: {name}")
    return bfl.Flux2Params(
        in_channels=config.in_channels,
        context_in_dim=config.joint_attention_dim,
        hidden_size=config.attention_head_dim * config.num_attention_heads,
        num_heads=config.num_attention_heads,
        depth=config.num_layers,
        depth_single_blocks=config.num_single_layers,
        axes_dim=axes,
        theta=config.rope_theta,
        mlp_ratio=config.mlp_ratio,
        use_guidance_embed=config.guidance_embeds,
    )


def _mapping(params):
    """Declare every source key once, with explicit fused/reordered exceptions."""
    mapping = {}

    def identity(target, source, suffix="weight"):
        mapping[f"{target}.{suffix}"] = ([f"{source}.weight"], "identity")

    for target, source in (
        ("img_in", "x_embedder"), ("txt_in", "context_embedder"),
        ("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1"),
        ("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2"),
        ("double_stream_modulation_img.lin", "double_stream_modulation_img.linear"),
        ("double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear"),
        ("single_stream_modulation.lin", "single_stream_modulation.linear"),
        ("final_layer.linear", "proj_out"),
    ):
        identity(target, source)
    if params.use_guidance_embed:
        for target, source in (("in_layer", "linear_1"), ("out_layer", "linear_2")):
            identity(f"guidance_in.{target}", f"time_guidance_embed.guidance_embedder.{source}")
    mapping["final_layer.adaLN_modulation.1.weight"] = (["norm_out.linear.weight"], "swap_scale_shift")
    for index in range(params.depth):
        target, source = f"double_blocks.{index}", f"transformer_blocks.{index}"
        for modality, prefix, projections, norm_prefix, output in (
            ("img", "ff", ("to_q", "to_k", "to_v"), "norm_", "to_out.0"),
            ("txt", "ff_context", ("add_q_proj", "add_k_proj", "add_v_proj"), "norm_added_", "to_add_out"),
        ):
            mapping[f"{target}.{modality}_attn.qkv.weight"] = (
                [f"{source}.attn.{projection}.weight" for projection in projections], "concat_qkv")
            for qk, short in (("query", "q"), ("key", "k")):
                identity(f"{target}.{modality}_attn.norm.{qk}_norm", f"{source}.attn.{norm_prefix}{short}", "scale")
            identity(f"{target}.{modality}_attn.proj", f"{source}.attn.{output}")
            identity(f"{target}.{modality}_mlp.0", f"{source}.{prefix}.linear_in")
            identity(f"{target}.{modality}_mlp.2", f"{source}.{prefix}.linear_out")
    for index in range(params.depth_single_blocks):
        target, source = f"single_blocks.{index}", f"single_transformer_blocks.{index}.attn"
        identity(f"{target}.linear1", f"{source}.to_qkv_mlp_proj")
        identity(f"{target}.linear2", f"{source}.to_out")
        identity(f"{target}.norm.query_norm", f"{source}.norm_q", "scale")
        identity(f"{target}.norm.key_norm", f"{source}.norm_k", "scale")
    return mapping


def _coverage(actual, expected, description):
    missing, unexpected = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(f"{description} coverage mismatch: missing={missing}, unexpected={unexpected}")


@torch.no_grad()
def build_bfl_model(diffusers_model):
    """Return an evaluated BFL model and JSON audit, without casting or source writes.

    Meta initialization avoids a redundant random-weight allocation. Direct
    mappings share source storage but have independent, frozen Parameter objects.
    The caller must keep both models read-only and retain the source precision.
    """
    source = validate_source()
    bfl = importlib.import_module("scripts.diagnostics._third_party.bfl_flux2.model")
    if Path(bfl.__file__).resolve() != (SOURCE_DIRECTORY / "model.py").resolve():
        raise ValueError("Imported BFL model is not the validated vendored source")
    params = _parameters(diffusers_model, bfl)
    with torch.device("meta"):
        model = bfl.Flux2(params)
    target_state = model.state_dict()
    source_state = diffusers_model.state_dict()
    mapping = _mapping(params)
    _coverage(target_state, mapping, "BFL target")
    consumed = [key for keys, _ in mapping.values() for key in keys]
    if len(consumed) != len(set(consumed)):
        raise ValueError("Diffusers source weights must be consumed exactly once")
    _coverage(source_state, consumed, "Diffusers source")
    formats = {(value.device, value.dtype) for value in source_state.values()}
    if len(formats) != 1 or any(value.is_meta or not value.is_floating_point()
                               or value.layout != torch.strided for value in source_state.values()):
        raise ValueError("BFL mapping requires materialized floating weights with one dtype and device")

    # Validate all shapes before allocating the few non-identity target tensors.
    for target, (keys, operation) in mapping.items():
        expected = tuple(target_state[target].shape)
        if operation == "concat_qkv":
            expected = (expected[0] // 3, *expected[1:])
        for key in keys:
            if tuple(source_state[key].shape) != expected:
                raise ValueError(f"Weight shape mismatch: {key} -> {target}, expected {expected}, got {tuple(source_state[key].shape)}")

    assigned, entries = {}, {}
    shared_bytes = copied_bytes = 0
    for target, (keys, operation) in mapping.items():
        values = [source_state[key] for key in keys]
        if operation == "concat_qkv":
            value = torch.cat(values, dim=0)
        elif operation == "swap_scale_shift":
            scale, shift = values[0].chunk(2, dim=0)
            value = torch.cat((shift, scale), dim=0)
        else:
            value = values[0]
        assigned[target] = value
        entries[target] = dict(source_keys=keys, operation=operation, shape=list(value.shape),
                               dtype=str(value.dtype), numel=value.numel())
        size = value.numel() * value.element_size()
        if operation == "identity":
            shared_bytes += size
        else:
            copied_bytes += size
    model.load_state_dict(assigned, strict=True, assign=True)
    model.eval().requires_grad_(False)
    source_numel = sum(value.numel() for value in source_state.values())
    target_numel = sum(value.numel() for value in model.state_dict().values())
    if source_numel != target_numel or any(value.is_meta for value in model.state_dict().values()):
        raise ValueError("BFL weight mapping did not completely materialize equal parameter counts")
    return model, dict(
        source=source, bfl_parameters=asdict(params), complete=True, lossless=True,
        source_parameter_count=len(source_state), target_parameter_count=len(target_state),
        source_numel=source_numel, target_numel=target_numel,
        shared_storage_bytes=shared_bytes, copied_storage_bytes=copied_bytes,
        dtype=str(next(iter(source_state.values())).dtype),
        device=str(next(iter(source_state.values())).device),
        construction="meta initialization and strict assign; no dtype conversion",
        storage_contract="Identity mappings share storage; neither model may be updated while both are in use",
        entries=entries,
    )
