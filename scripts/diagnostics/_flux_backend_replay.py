"""Replay identical post-RoPE Q/K/V through explicit attention backends.

These are local attention vector-Jacobian products with a shared synthetic
upstream gradient. They do not measure model, loss, or parameter gradients.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics._flux_packing import FlashBackend


BACKENDS = ("fp32_math", "bf16_math", "cudnn", "fa2", "fa3", "fa4")
SDPA_OPERATORS = {
    "aten::_scaled_dot_product_attention_math": "math",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_flash_attention": "torch_flash",
    "aten::_scaled_dot_product_efficient_attention": "efficient",
}
FLASH_OPERATORS = {
    "fa2": ("flash_attn::_flash_attn_varlen_forward", "flash_attn::_flash_attn_varlen_backward"),
    "fa3": ("flash_attn_3::_flash_attn_forward", "flash_attn_3::_flash_attn_backward"),
}


def fingerprint(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw).hexdigest())


def tensor_metrics(actual, reference):
    """Accumulate errors and norms in FP64 with bounded temporary storage."""
    if actual.shape != reference.shape or not actual.numel():
        raise ValueError("numerical comparison requires equal, nonempty shapes")
    totals = torch.zeros(4, dtype=torch.float64, device=actual.device)
    maximum = torch.zeros((), dtype=torch.float64, device=actual.device)
    for a, b in zip(actual.detach().reshape(-1).split(1 << 20),
                    reference.detach().reshape(-1).split(1 << 20)):
        a = a.double()
        b = b.to(device=actual.device, dtype=torch.float64)
        delta = a - b
        totals += torch.stack((delta.square().sum(), b.square().sum(),
                               a.square().sum(), (a * b).sum()))
        maximum = torch.maximum(maximum, delta.abs().max())
    error, expected, observed, dot = totals.cpu().tolist()
    maximum = maximum.item()
    if not all(math.isfinite(value) for value in (error, expected, observed, dot, maximum)):
        raise FloatingPointError("non-finite attention output or gradient")
    count = actual.numel()
    return dict(rms=math.sqrt(error / count), max_abs=maximum,
                relative_l2=math.sqrt(error / expected) if expected else (0.0 if error == 0 else None),
                reference_l2=math.sqrt(expected), actual_l2=math.sqrt(observed),
                reference_rms=math.sqrt(expected / count), actual_rms=math.sqrt(observed / count),
                cosine=max(-1.0, min(1.0, dot / math.sqrt(expected * observed)))
                       if expected and observed else None,
                count=count, actual_dtype=str(actual.dtype), reference_dtype=str(reference.dtype))


@contextmanager
def replay_precision(device):
    """Disable autocast and TF32 so the selected input dtype is authoritative."""
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous[3])


def attention_forward(name, q, k, v):
    """Noncausal self-attention on validated contiguous [1, S, H, D] inputs."""
    if name not in BACKENDS:
        raise ValueError(f"unknown attention backend: {name}")
    expected = torch.float32 if name == "fp32_math" else torch.bfloat16
    if any(t.dtype != expected for t in (q, k, v)):
        raise ValueError(f"{name} requires {expected} inputs")
    if name in ("fp32_math", "bf16_math", "cudnn"):
        backend = SDPBackend.CUDNN_ATTENTION if name == "cudnn" else SDPBackend.MATH
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False, scale=q.shape[-1] ** -0.5,
            ).transpose(1, 2)
    layout = SimpleNamespace(
        cumulative=torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device),
        maximum=q.shape[1],
    )
    return FlashBackend(name)(q.squeeze(0), k.squeeze(0), v.squeeze(0), layout).unsqueeze(0)


def validate_profile(operators, kernels, name, *, cuda):
    """Verify backend operators; expose unavailable CUDA tracing separately."""
    if operators.get(f"flux_backend_replay::{name}", 0) != 1:
        raise RuntimeError(f"{name}: missing explicit replay scope")
    selected = "cudnn" if name == "cudnn" else "math" if name.endswith("math") else None
    for operator, backend in SDPA_OPERATORS.items():
        if operators.get(operator, 0) != int(backend == selected):
            raise RuntimeError(f"{name}: unexpected {operator} count")
    for backend, names in FLASH_OPERATORS.items():
        for operator in names:
            if operators.get(operator, 0) != int(backend == name):
                raise RuntimeError(f"{name}: unexpected {operator} count")
    if selected is None and any("scaled_dot_product" in key for key in operators):
        raise RuntimeError(f"{name}: unintended SDPA fallback")
    if name == "cudnn" and operators.get("aten::_scaled_dot_product_cudnn_attention_backward", 0) != 1:
        raise RuntimeError("cudnn: missing cuDNN backward operator")
    if selected == "math" and operators.get("aten::_softmax_backward_data", 0) != 1:
        raise RuntimeError(f"{name}: missing Math softmax backward operator")
    if name == "fa4":
        # The pinned CuTe interface uses this autograd function directly. Its
        # forward/backward events remain visible when CUPTI tracing is absent.
        for operator in ("FlashAttnVarlenFunc", "FlashAttnVarlenFuncBackward"):
            if operators.get(operator, 0) != 1:
                raise RuntimeError(f"fa4: missing explicit {operator} execution")
    if not any("backward" in key.lower() for key in operators):
        raise RuntimeError(f"{name}: missing backward evidence")
    if cuda and kernels and name == "fa4":
        # CuTe launches compiled kernels directly, without FA2/FA3 custom ops.
        attention = [key.lower() for key in kernels
                     if any(token in key.lower() for token in ("flash", "fmha", "attn"))]
        for directions in (("fwd", "forward"), ("bwd", "backward", "dgrad")):
            if not any(any(direction in key for direction in directions) for key in attention):
                raise RuntimeError(f"fa4: missing CUDA attention {directions[0]} evidence")
    available = bool(cuda and kernels)
    return dict(cuda_kernel_capture_available=available, cuda_kernel_capture_requested=bool(cuda),
                evidence_level="backend_operators_and_cuda_kernels" if available else "backend_operators_only",
                backend_verified=True)


def _evaluate(name, originals, upstream, profile):
    dtype = torch.float32 if name == "fp32_math" else torch.bfloat16
    leaves = tuple(t.detach().to(dtype=dtype).clone().requires_grad_() for t in originals)
    incoming = upstream.to(dtype=dtype).clone()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if leaves[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities) if profile else nullcontext()
    with profiler as recorded:
        with torch.profiler.record_function(f"flux_backend_replay::{name}"):
            output = attention_forward(name, *leaves)
            if output.shape != originals[0].shape or output.dtype != dtype:
                raise ValueError(f"{name}: attention returned an unexpected shape or dtype")
            gradients = torch.autograd.grad(output, leaves, grad_outputs=incoming)
    for original, leaf in zip(originals, leaves):
        if not torch.equal(leaf.detach().view(torch.uint8), original.to(dtype=dtype).view(torch.uint8)):
            raise RuntimeError(f"{name}: attention mutated a Q/K/V input")
    if not torch.equal(incoming.view(torch.uint8), upstream.to(dtype=dtype).view(torch.uint8)):
        raise RuntimeError(f"{name}: attention backward mutated the upstream gradient")
    evidence = dict(profiled=bool(profile), binding=(
        f"{FlashBackend.MODULES[name]}.flash_attn_varlen_func" if name.startswith("fa")
        else f"torch.nn.functional.scaled_dot_product_attention/{'cuDNN' if name == 'cudnn' else 'Math'}"),
        input_dtype=str(dtype), upstream_dtype=str(dtype), inputs_unchanged=True,
        upstream_unchanged=True, backend_verified=False, evidence_level="not_profiled",
        cuda_kernel_capture_available=None, cuda_kernel_capture_requested=False)
    if profile:
        operators = {event.key: event.count for event in recorded.key_averages()}
        kernels = sorted({event.name for event in recorded.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        evidence.update(validate_profile(operators, kernels, name, cuda=leaves[0].is_cuda))
        evidence.update(operators=operators, cuda_kernel_names=kernels)
    return (output.detach(), *(gradient.detach() for gradient in gradients)), evidence


def compare_backends(q, k, v, seed, profile=False, *, backends=BACKENDS):
    """Compare output and dQ/dK/dV against FP32 Math using identical BF16 values.

    Inputs are single unpadded examples, with normalization and RoPE already
    applied. The FP32 reference promotes those BF16 values without recomputing
    either operation. Numerical disagreements are returned, never thresholded.
    """
    originals = (q, k, v)
    if (q.ndim != 4 or q.shape[0] != 1 or not all(q.shape)
            or any(t.shape != q.shape or t.device != q.device or t.dtype != torch.bfloat16
                   or not t.is_contiguous() for t in originals)):
        raise ValueError("replay requires same-device contiguous BF16 Q/K/V with shape [1, S, H, D]")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("replay seed must be a nonnegative 63-bit integer")
    backends = tuple(backends)
    if (not backends or backends[0] != "fp32_math" or len(set(backends)) != len(backends)
            or any(name not in BACKENDS for name in backends)
            or len(backends) > 1 and backends[1] != "bf16_math"):
        raise ValueError("unique known backends must start with fp32_math, followed by bf16_math")
    if q.device.type != "cuda" and any(name not in ("fp32_math", "bf16_math") for name in backends):
        raise ValueError("cuDNN and FlashAttention replay require CUDA")
    if any(not torch.isfinite(t).all() for t in originals):
        raise FloatingPointError("non-finite replay inputs")
    generator = torch.Generator(device=q.device).manual_seed(seed)
    upstream = torch.randn(q.shape, dtype=torch.bfloat16, device=q.device, generator=generator)
    identities = {label: fingerprint(tensor) for label, tensor in zip(("q", "k", "v"), originals)}
    promoted_identities = {label: fingerprint(tensor.float())
                           for label, tensor in zip(("q", "k", "v"), originals)}
    result = dict(inputs=identities, upstream_gradient=fingerprint(upstream), seed=seed,
                  fp32_inputs=promoted_identities, fp32_upstream_gradient=fingerprint(upstream.float()),
                  upstream_source="one seeded BF16 Gaussian tensor, promoted unchanged for FP32",
                  reference="fp32_math", causal=False, dropout=0.0, softmax_scale=q.shape[-1] ** -0.5,
                  segment_lengths=[q.shape[1]], tf32=False, autocast=False,
                  math_fp16_bf16_reduction=False, model_gradients_measured=False, backends={})
    reference = bf16_reference = None
    with replay_precision(q.device):
        for name in backends:
            try:
                values, evidence = _evaluate(name, originals, upstream, profile)
                if reference is None:
                    reference = values
                measured = {label: tensor_metrics(actual, expected)
                            for label, actual, expected in zip(("output", "dQ", "dK", "dV"), values, reference)}
                result["backends"][name] = dict(**evidence, **measured)
                result["backends"][name]["input_fingerprints"] = (
                    promoted_identities if name == "fp32_math" else identities)
                result["backends"][name]["upstream_fingerprint"] = result[
                    "fp32_upstream_gradient" if name == "fp32_math" else "upstream_gradient"]
                if name == "bf16_math":
                    bf16_reference = values
                    result["backends"]["fp32_math"]["comparisons_vs_bf16_math"] = {
                        label: tensor_metrics(actual, expected)
                        for label, actual, expected in zip(("output", "dQ", "dK", "dV"), reference, values)
                    }
                if bf16_reference is not None:
                    result["backends"][name]["comparisons_vs_bf16_math"] = {
                        label: tensor_metrics(actual, expected)
                        for label, actual, expected in zip(("output", "dQ", "dK", "dV"), values, bf16_reference)
                    }
            except Exception as error:
                raise RuntimeError(f"{name} isolated attention replay failed: {error}") from error
    if identities != {label: fingerprint(tensor) for label, tensor in zip(("q", "k", "v"), originals)}:
        raise RuntimeError("replay modified the caller's Q/K/V tensors")
    result["input_hashes_unchanged"] = True
    return result
