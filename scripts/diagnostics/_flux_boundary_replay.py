"""Replay real loss attention gradients across singleton, padding and packing.

Only post-RoPE Q/K/V and incoming output gradients are reused. This separates
kernel-local boundary effects from the upstream differences in full-model runs.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics import _flux_packing_replay as replay
from scripts.diagnostics._flux_backend_replay import fingerprint, replay_precision, tensor_metrics, validate_profile
from scripts.diagnostics._flux_gradient_boundaries import token_error_metrics


STAGES = ("double.0", "double.4", "single.0", "single.19")
BACKENDS = ("cudnn", "fa3")


def _pad_inputs(captures, upstreams, text_lengths, padded_length):
    """Insert padding between text and image; retain actual valid key positions."""
    lengths = tuple(sample[0].shape[1] for sample in captures)
    image_lengths = tuple(length - text for length, text in zip(lengths, text_lengths, strict=True))
    if (len(captures) != 2 or len(upstreams) != 2 or len(text_lengths) != 2
            or type(padded_length) is not int or any(type(n) is not int or n < 0 for n in text_lengths)
            or padded_length < max(text_lengths) or len(set(image_lengths)) != 1 or image_lengths[0] <= 0):
        raise ValueError("padded replay requires two text lengths and equal nonempty image sequences")

    def pad(value, text):
        return torch.cat((value[0, :text], value.new_zeros((padded_length - text, *value.shape[2:])),
                          value[0, text:]), dim=0)

    values = tuple(torch.stack([pad(sample[axis], text)
                                for sample, text in zip(captures, text_lengths, strict=True)]) for axis in range(3))
    incoming = torch.stack([pad(value, text) for value, text in zip(upstreams, text_lengths, strict=True)])
    positions = torch.arange(padded_length + image_lengths[0], device=values[0].device)
    valid = torch.stack([(positions < text) | (positions >= padded_length) for text in text_lengths])
    return values, incoming, valid


def _evaluate_padded(captures, upstreams, text_lengths, padded_length, *, profile=False, backend="cudnn"):
    """One native batch SDPA call; Math is an explicit CPU test control only."""
    if backend not in ("cudnn", "bf16_math"):
        raise ValueError("padded replay supports cuDNN or explicit BF16 Math control")
    originals, incoming_original, valid = _pad_inputs(captures, upstreams, text_lengths, padded_length)
    leaves = tuple(value.detach().clone().requires_grad_() for value in originals)
    incoming = incoming_original.detach().clone()
    selected = SDPBackend.CUDNN_ATTENTION if backend == "cudnn" else SDPBackend.MATH
    activities = [torch.profiler.ProfilerActivity.CPU]
    if leaves[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities) if profile else nullcontext()
    with profiler as recorded:
        with torch.profiler.record_function(f"flux_backend_replay::{backend}"), sdpa_kernel(selected):
            output = F.scaled_dot_product_attention(
                *(value.transpose(1, 2) for value in leaves), attn_mask=valid[:, None, None, :],
                dropout_p=0.0, is_causal=False, scale=leaves[0].shape[-1] ** -0.5,
            ).transpose(1, 2)
            gradients = torch.autograd.grad(output, leaves, grad_outputs=incoming)
    if any(not torch.equal(value.detach(), reference) for value, reference in zip(leaves, originals, strict=True)):
        raise RuntimeError("padded attention mutated a Q/K/V input")
    if not torch.equal(incoming, incoming_original):
        raise RuntimeError("padded backward mutated the upstream gradient")
    tensors = (output.detach(), *(gradient.detach() for gradient in gradients))
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("non-finite padded attention output or gradient")
    padding_gradients = {
        label: {field: {"max_abs": gradient[index, ~valid[index]].abs().max().item()
                       if (~valid[index]).any() else 0.0,
                       "nonzero_count": gradient[index, ~valid[index]].count_nonzero().item()}
                for field, gradient in zip(replay.FIELDS[1:], gradients, strict=True)}
        for index, label in enumerate(replay.SAMPLES)
    }
    if any(row["nonzero_count"] for sample in padding_gradients.values() for row in sample.values()):
        raise RuntimeError("zero-upstream masked padding received an input gradient")
    values = {label: tuple(value[index, valid[index]] for value in tensors)
              for index, label in enumerate(replay.SAMPLES)}
    evidence = dict(profiled=profile, backend_verified=False, evidence_level="not_profiled",
                    binding=f"torch.nn.functional.scaled_dot_product_attention/{backend}",
                    layout="native_padded_batch", attention_calls=1, batch_size=2,
                    causal=False, dropout=0.0, inputs_unchanged=True, upstream_unchanged=True,
                    padding_gradients=padding_gradients,
                    valid_key_mask=fingerprint(valid),
                    padded_text_length=padded_length,
                    padding_token_counts=[padded_length - length for length in text_lengths])
    if profile:
        operators = {event.key: event.count for event in recorded.key_averages()}
        kernels = sorted({event.name for event in recorded.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        evidence.update(validate_profile(operators, kernels, backend, cuda=leaves[0].is_cuda))
        evidence.update(operators=operators, cuda_kernel_names=kernels)
    return values, evidence


def _cpu(values):
    return {label: tuple(value.detach().to(device="cpu", copy=True) for value in sample)
            for label, sample in values.items()}


def _compare(actual, reference, text_lengths, edge_width):
    return {label: {field: {
        "summary": tensor_metrics(value, expected),
        "tokens": token_error_metrics(value, expected, text_length=text_lengths[label], edge_width=edge_width),
    } for field, value, expected in zip(replay.FIELDS, sample, reference[label], strict=True)}
        for label, sample in actual.items()}


def _validate(records, text_lengths, padded_length, edge_width):
    if not isinstance(records, dict) or set(records) != set(replay.SAMPLES):
        raise ValueError("boundary replay requires exactly A and B")
    if (not isinstance(text_lengths, dict) or set(text_lengths) != set(replay.SAMPLES)
            or any(type(n) is not int or n < 0 for n in text_lengths.values())
            or type(padded_length) is not int or padded_length < max(text_lengths.values())
            or type(edge_width) is not int or edge_width < 1):
        raise ValueError("invalid boundary replay text lengths, padding or edge width")
    identity = {}
    common_tail, image_length = None, None
    for label in replay.SAMPLES:
        if not isinstance(records[label], dict) or set(records[label]) != set(STAGES):
            raise ValueError("boundary replay requires the four selected attention blocks")
        identity[label] = {}
        for stage in STAGES:
            record = records[label][stage]
            if not isinstance(record, dict) or set(record) != {"qkv", "upstream"} or len(record["qkv"]) != 3:
                raise ValueError("boundary records must contain Q/K/V and the real upstream gradient")
            sample = (*record["qkv"], record["upstream"])
            first = sample[0]
            if (not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[0] != 1
                    or not all(first.shape) or first.shape[1] <= text_lengths[label]
                    or any(not isinstance(value, torch.Tensor) or value.shape != first.shape
                           or value.device.type != "cpu" or value.dtype != torch.bfloat16 for value in sample)):
                raise ValueError("boundary replay requires CPU BF16 [1,S,H,D] operands with matching shapes")
            if any(not torch.isfinite(value).all() for value in sample):
                raise FloatingPointError("non-finite boundary replay source")
            tail, images = first.shape[2:], first.shape[1] - text_lengths[label]
            if common_tail is not None and (common_tail != tail or image_length != images):
                raise ValueError("boundary replay requires consistent heads, head dimension and image lengths")
            common_tail, image_length = tail, images
            identity[label][stage] = {field: fingerprint(value)
                                      for field, value in zip(("q", "k", "v", "upstream"), sample, strict=True)}
    return identity


def replay_boundaries(records, *, text_lengths, device, padded_length=487, edge_width=32, on_result=None):
    """Run exactly 72 forward/VJPs, reusing two true-loss singleton captures.

    Every backend/layout is evaluated twice. Only the first selected block's
    first padded/segmented cuDNN and packed FA3 calls are profiled. Partial JSON
    results are published after each backend; failures never trigger a retry.
    """
    identity = _validate(records, text_lengths, padded_length, edge_width)
    device = torch.device(device)
    result = dict(status="running", source="true_loss_cudnn_singleton_backward", inputs=identity,
                  text_lengths=dict(text_lengths), padded_text_length=padded_length, edge_width=edge_width,
                  plan=dict(stages=list(STAGES), backends=list(BACKENDS), repetitions=2,
                            forward_vjp_evaluations=72, attention_forward_operators=88,
                            profiled_evaluations=3, model_parameter_gradients=False),
                  evaluations_attempted=0, evaluations_completed=0, stages={})

    def publish():
        if on_result is not None:
            on_result(result)

    def evaluate(backend, captures, upstreams, order, *, padded=False, profile=False):
        result["evaluations_attempted"] += 1
        if padded:
            values, evidence = _evaluate_padded(captures, upstreams, tuple(text_lengths[label] for label in replay.SAMPLES),
                                                 padded_length, profile=profile)
        else:
            values, evidence = replay._evaluate(backend, captures, upstreams, order, profile=profile)
        result["evaluations_completed"] += 1
        return _cpu(values), evidence

    publish()
    try:
        with replay_precision(device):
            for stage in STAGES:
                captures = tuple(tuple(value.to(device=device).contiguous() for value in records[label][stage]["qkv"])
                                 for label in replay.SAMPLES)
                upstreams = tuple(records[label][stage]["upstream"].to(device=device).contiguous() for label in replay.SAMPLES)
                stage_result = dict(backends={})
                result["stages"][stage] = stage_result
                for backend in BACKENDS:
                    row = dict(executions=[], repeat_comparisons={})
                    stage_result["backends"][backend] = row
                    first_execution = None
                    for repetition in range(2):
                        observed = stage == STAGES[0] and repetition == 0
                        values, dispatch, singletons = {}, {}, {}
                        for index, label in enumerate(replay.SAMPLES):
                            captured, dispatch[f"singleton_{label}"] = evaluate(backend, captures, upstreams, (index,))
                            singletons.update(captured)
                        values["singletons"] = singletons
                        if backend == "cudnn":
                            values["padded"], dispatch["padded"] = evaluate(backend, captures, upstreams, (0, 1),
                                                                            padded=True, profile=observed)
                        values["ab"], dispatch["ab"] = evaluate(backend, captures, upstreams, (0, 1), profile=observed)
                        values["ba"], dispatch["ba"] = evaluate(backend, captures, upstreams, (1, 0))
                        comparisons = {f"{layout}_vs_singletons": _compare(values[layout], singletons, text_lengths, edge_width)
                                       for layout in values if layout != "singletons"}
                        comparisons["ba_vs_ab"] = _compare(values["ba"], values["ab"], text_lengths, edge_width)
                        row["executions"].append(dict(repetition=repetition, dispatch=dispatch, comparisons=comparisons))
                        if first_execution is None:
                            first_execution = values
                        else:
                            row["repeat_comparisons"] = {
                                layout: _compare(values[layout], first_execution[layout], text_lengths, edge_width)
                                for layout in values}
                    del values, first_execution
                    publish()
                del captures, upstreams
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise
    publish()
    return result
