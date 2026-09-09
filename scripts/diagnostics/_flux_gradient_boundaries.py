"""Locate token errors and passively capture real FLUX output gradients.

Canonical sequences contain valid text followed by image tokens; callers remove
padding and restore sample order before comparing them. Error normalization uses
one reference RMS for the whole sequence, never a near-zero individual token.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import math

import torch


DEFAULT_STAGES = {
    "double.0": "transformer_blocks.0",
    "double.4": "transformer_blocks.4",
    "single.0": "single_transformer_blocks.0",
    "single.19": "single_transformer_blocks.19",
    "text.mlp.linear_out": "transformer_blocks.0.ff_context.linear_out",
}


def token_error_metrics(actual, reference, *, text_length, edge_width=32):
    """Return JSON-safe per-token errors and overlapping boundary summaries.

    Inputs have matching [sequence, ...features] shapes. Ratios are fractions,
    not percentages. A zero global reference RMS makes normalized errors None;
    absolute errors remain available. Empty regions report no RMS or maximum.
    """
    if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise ValueError("token metrics require tensor inputs")
    if actual.shape != reference.shape or actual.ndim < 2 or any(size == 0 for size in actual.shape):
        raise ValueError("token metrics require matching nonempty [sequence, ...features] tensors")
    if not actual.is_floating_point() or not reference.is_floating_point():
        raise ValueError("token metrics require floating point tensors")
    count = actual.shape[0]
    if type(text_length) is not int or not 0 <= text_length <= count:
        raise ValueError("text_length must be an integer within the sequence")
    if type(edge_width) is not int or edge_width < 1:
        raise ValueError("edge_width must be a positive integer")
    actual = actual.detach().to(device="cpu", dtype=torch.float64).reshape(count, -1)
    reference = reference.detach().to(device="cpu", dtype=torch.float64).reshape(count, -1)
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("token metrics require finite inputs")
    error = actual - reference
    token_error_squared = error.square().mean(dim=1)
    error_rms = token_error_squared.sqrt()
    max_abs = error.abs().amax(dim=1)
    token_reference_squared = reference.square().mean(dim=1)
    reference_rms = token_reference_squared.sqrt()
    global_reference_rms = math.sqrt(token_reference_squared.mean().item())
    total_error_energy = token_error_squared.sum().item()
    normalizer = global_reference_rms if global_reference_rms > 0 else None
    positions = torch.arange(count)
    start = positions < edge_width
    end = positions >= count - edge_width
    transition = 0 < text_length < count
    text_edge = ((positions >= max(0, text_length - edge_width)) & (positions < text_length)
                 if transition else torch.zeros(count, dtype=torch.bool))
    image_edge = ((positions >= text_length) & (positions < min(count, text_length + edge_width))
                  if transition else torch.zeros(count, dtype=torch.bool))
    boundary = start | end | text_edge | image_edge

    def region(mask):
        size = mask.sum().item()
        energy = token_error_squared[mask].sum().item()
        rms = math.sqrt(energy / size) if size else None
        share = energy / total_error_energy if total_error_energy > 0 else 0.0
        return {
            "token_count": size,
            "token_fraction": size / count,
            "error_rms": rms,
            "max_abs": max_abs[mask].max().item() if size else None,
            "reference_rms": math.sqrt(token_reference_squared[mask].mean().item()) if size else None,
            "error_rms_over_global_reference_rms": rms / normalizer if size and normalizer else None,
            "squared_error_fraction": share,
            "error_energy_enrichment": share / (size / count) if size else None,
        }

    masks = {
        "all": torch.ones(count, dtype=torch.bool),
        "sequence_start": start,
        "sequence_end": end,
        "text_before_image": text_edge,
        "image_after_text": image_edge,
        "boundary_union": boundary,
        "interior": ~boundary,
        "text": positions < text_length,
        "image": positions >= text_length,
    }
    # Stable ordering keeps equally large errors reproducible and inspectable.
    top = torch.argsort(error_rms, descending=True, stable=True)[:min(16, count)].tolist()
    return {
        "sequence_length": count,
        "text_length": text_length,
        "edge_width": edge_width,
        "reference_rms": global_reference_rms,
        "reference_is_zero": normalizer is None,
        "error_is_zero": total_error_energy == 0,
        "normalization": "whole_sequence_reference_rms",
        "regions_overlap": True,
        "per_token": {
            "error_rms": error_rms.tolist(),
            "max_abs": max_abs.tolist(),
            "reference_rms": reference_rms.tolist(),
            "error_rms_over_global_reference_rms": (error_rms / normalizer).tolist() if normalizer else None,
        },
        "regions": {name: region(mask) for name, mask in masks.items()},
        "top_tokens": [{
            "position": position,
            "modality": "text" if position < text_length else "image",
            "modality_position": position if position < text_length else position - text_length,
            "error_rms": error_rms[position].item(),
            "max_abs": max_abs[position].item(),
            "reference_rms": reference_rms[position].item(),
            "regions": [name for name in masks if name not in ("all", "text", "image") and masks[name][position]],
        } for position in top],
    }


def _map_tensors(value, transform, path=()):
    if isinstance(value, torch.Tensor):
        return transform(value, path)
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, transform, (*path, index)) for index, item in enumerate(value))
    raise ValueError("boundary outputs must be tensors or tuples/lists of tensors")


class GradientBoundaryCapture:
    """Capture one forward/backward without replacing outputs or gradients.

    ``canonicalize(stage, raw_tree)`` receives detached CPU tensors in the native
    output structure (Double blocks return text, image). It returns label ->
    [valid text, image] tensor, or text-only tensors for the MLP stage. Callers
    provide the layout interpretation; this observer never guesses sample order.
    After context exit, ``values[stage]['output'/'gradient'][label]`` contains
    independent CPU copies. Unconsumed output leaves have zero gradients and are
    explicitly listed in ``unused_output_paths``. Use a fresh capture per call.
    """

    def __init__(self, model, canonicalize, *, stages=None):
        self.model = model
        self.canonicalize = canonicalize
        self.stages = dict(DEFAULT_STAGES if stages is None else stages)
        if not self.stages or any(not isinstance(key, str) or not isinstance(value, str)
                                  for key, value in self.stages.items()):
            raise ValueError("capture stages must map names to module paths")
        self.values = {}
        self.unused_output_paths = {}
        self.completed = False
        self._used = False

    def _canonical(self, stage, tree):
        values = self.canonicalize(stage, tree)
        if not isinstance(values, Mapping) or not values:
            raise ValueError("canonicalize must return a nonempty label-to-tensor mapping")
        result = {}
        for label, value in values.items():
            if not isinstance(label, str) or not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError("canonicalize must return string labels and [sequence, channels] tensors")
            result[label] = value.detach().to(device="cpu", copy=True)
        return result

    @contextmanager
    def installed(self):
        if self._used:
            raise ValueError("gradient boundary capture cannot be reused")
        self._used = True
        handles, outputs, gradients = [], {}, {}

        def observer(stage):
            def record(module, args, output):
                if stage in outputs:
                    raise ValueError(f"duplicate boundary forward: {stage}; checkpoint recomputation is unsupported")
                gradients[stage] = {}

                def retain(tensor, path):
                    if not tensor.requires_grad:
                        raise ValueError(f"boundary output does not require gradients: {stage}")

                    def gradient_hook(gradient):
                        if path in gradients[stage]:
                            raise ValueError(f"duplicate boundary backward: {stage}")
                        gradients[stage][path] = gradient.detach().to(device="cpu", copy=True)
                        # Returning None preserves the actual backward tensor.
                        return None

                    handles.append(tensor.register_hook(gradient_hook))
                    return tensor.detach().to(device="cpu", copy=True)

                outputs[stage] = _map_tensors(output, retain)
                return None
            return record

        try:
            for stage, module_path in self.stages.items():
                module = self.model.get_submodule(module_path)
                handles.append(module.register_forward_hook(observer(stage)))
            yield self
            if set(outputs) != set(self.stages):
                raise ValueError("incomplete boundary forward capture")
            for stage, output in outputs.items():
                if not gradients[stage]:
                    raise ValueError(f"missing boundary backward: {stage}")
                unused = []

                def complete_gradient(tensor, path):
                    if path not in gradients[stage]:
                        unused.append(list(path))
                        return torch.zeros_like(tensor)
                    return gradients[stage][path]

                gradient = _map_tensors(output, complete_gradient)
                captured = {"output": self._canonical(stage, output),
                            "gradient": self._canonical(stage, gradient)}
                if captured["output"].keys() != captured["gradient"].keys() or any(
                        value.shape != captured["gradient"][label].shape
                        for label, value in captured["output"].items()):
                    raise ValueError("canonical output and gradient labels/shapes differ")
                self.values[stage] = captured
                self.unused_output_paths[stage] = unused
            self.completed = True
        finally:
            for handle in reversed(handles):
                handle.remove()

    def __enter__(self):
        self._context = self.installed()
        return self._context.__enter__()

    def __exit__(self, *args):
        return self._context.__exit__(*args)
