"""Expand per-example FLUX modulations with FP32 first-order gradient sums."""

import torch
from torch.autograd.function import once_differentiable


class _FP32ModulationGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, owners, lengths):
        ctx.lengths = lengths
        ctx.source_dtype = source.dtype
        return source.index_select(0, owners)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # Each contiguous segment belongs to one example. Round only after
        # reducing all of its token contributions at the modulation boundary.
        reduced = torch.stack([
            segment.sum(dim=0, dtype=torch.float32)
            for segment in gradient.split(ctx.lengths, dim=0)
        ])
        return reduced.to(ctx.source_dtype), None, None


def modulation_gather(source, owners, lengths):
    """Gather token modulations from a validated, contiguous packing layout.

    Forward is the native ``index_select``. Backward sums each example in
    FP32 and converts once to the source dtype. This first-order operation
    changes only the three modulation expansions, not general GEMM reduction.
    """
    if source.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("FP32 modulation reduction requires BF16 or FP32 source values")
    return _FP32ModulationGather.apply(source, owners, lengths)
