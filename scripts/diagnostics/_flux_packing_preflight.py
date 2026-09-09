"""Validate diagnostic packing metadata before any transformer/kernel execution.

These explicit CPU scans are deliberately outside the forward hot path. The
8,400-token text limit is separate from the maximum 900 image tokens.
"""

import torch

from scripts.diagnostics._flux_packing import PackingLayout


def validate_layout(layout: PackingLayout, *, max_text_tokens: int = 8400) -> None:
    """Reject malformed or mutated metadata against the labelled segment contract."""
    if not isinstance(layout, PackingLayout):
        raise ValueError("packing preflight requires a PackingLayout")
    if type(max_text_tokens) is not int or max_text_tokens < 1:
        raise ValueError("max_text_tokens must be a positive Python integer")
    lengths = layout.text_lengths
    if (type(lengths) is not tuple or not lengths
            or any(type(length) is not int or not 1 <= length <= max_text_tokens for length in lengths)):
        raise ValueError(f"text_lengths must contain Python integers between 1 and {max_text_tokens}")
    if type(layout.image_tokens) is not int or not 1 <= layout.image_tokens <= 900:
        raise ValueError("image_tokens must be a Python integer between 1 and 900")
    text_count = sum(lengths)
    image_count = len(lengths) * layout.image_tokens
    total = text_count + image_count
    if total > torch.iinfo(torch.int32).max:
        raise ValueError("cumulative sequence length exceeds CUDA int32 capacity")

    shapes = {
        "cumulative": (len(lengths) + 1,), "joint_order": (total,),
        "text_indices": (text_count,), "image_indices": (image_count,),
        "text_samples": (text_count,), "image_samples": (image_count,),
        "joint_samples": (total,), "text_positions": (text_count, 4),
    }
    arrays = {}
    device = None
    for name, shape in shapes.items():
        value = getattr(layout, name)
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_meta:
            raise ValueError(f"{name} must be a materialized dense tensor")
        expected_dtype = torch.int32 if name == "cumulative" else torch.int64
        if value.dtype != expected_dtype:
            raise ValueError(f"{name} requires {expected_dtype}")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} requires shape {shape}")
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError(f"{name} must use the common metadata device {device}")
        arrays[name] = value.detach().cpu()

    # Derive each token's source offset and packed destination from its segment,
    # without calling the layout constructor or trusting any stored inverse.
    expected = {name: [] for name in shapes}
    expected["cumulative"] = [0]
    text_offset = packed_offset = 0
    for sample, length in enumerate(lengths):
        for position in range(length):
            expected["joint_order"].append(text_offset + position)
            expected["text_indices"].append(packed_offset + position)
            expected["text_samples"].append(sample)
            expected["joint_samples"].append(sample)
            expected["text_positions"].append([0, 0, 0, position])
        for position in range(layout.image_tokens):
            expected["joint_order"].append(text_count + sample * layout.image_tokens + position)
            expected["image_indices"].append(packed_offset + length + position)
            expected["image_samples"].append(sample)
            expected["joint_samples"].append(sample)
        text_offset += length
        packed_offset += length + layout.image_tokens
        expected["cumulative"].append(packed_offset)
    for name, values in expected.items():
        if not torch.equal(arrays[name], torch.tensor(values, dtype=arrays[name].dtype)):
            raise ValueError(f"{name} disagrees with the source token/segment contract")
