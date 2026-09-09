"""Change only the first-order gradient reduction of packed modulation gathers."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.autograd.function import once_differentiable


_observation = ContextVar("flux_modulation_backward_observation", default=None)


@contextmanager
def observe_modulation_reduction():
    """Count actual intervention calls without modifying tensors or gradients."""
    evidence = dict(forward_calls=0, backward_calls=0, source_dtypes=[], reduction_dtype=None)
    token = _observation.set(evidence)
    try:
        yield evidence
    finally:
        _observation.reset(token)


class _FP32ModulationGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, owners, lengths):
        ctx.lengths = lengths
        ctx.source_dtype = source.dtype
        # Save the observer itself: autograd may execute outside its context.
        ctx.observation = _observation.get()
        output = source.index_select(0, owners)
        if ctx.observation is not None:
            ctx.observation["forward_calls"] += 1
            ctx.observation["source_dtypes"].append(str(source.dtype))
            ctx.observation["reduction_dtype"] = str(torch.float32)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Ownership is contiguous and validated before the model runs. Sum each
        # example in FP32, then round once at the original modulation boundary.
        reduced = torch.stack([segment.sum(dim=0, dtype=torch.float32)
                               for segment in gradient.split(ctx.lengths, dim=0)])
        if ctx.observation is not None:
            ctx.observation["backward_calls"] += 1
        return reduced.to(ctx.source_dtype), None, None


def modulation_gather(source, owners, lengths, *, reduction="native"):
    """Gather token modulations using preflight-validated contiguous ownership.

    ``source`` has shape [examples, width]; ``owners`` consists of the example
    indices in source order, repeated by ``lengths``. Call ``validate_layout``
    at diagnostic preflight to establish this contract, outside the hot path.
    FP32 reduction supports first-order backward for BF16/FP32 modulations.
    Both modes execute the same index_select in forward, preserving its values,
    dtype, shape and strides.
    """
    if reduction == "native":
        return source.index_select(0, owners)
    if reduction != "fp32":
        raise ValueError("modulation reduction must be native or fp32")
    if source.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("FP32 modulation reduction requires BF16 or FP32 source values")
    return _FP32ModulationGather.apply(source, owners, lengths)
