"""Compare one logical FLUX loss across singleton, padded and packed layouts."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from itertools import accumulate
from unittest.mock import patch
import math
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc._third_party.diffusers_flux2 import transformer_flux2 as df
from diffusion_arc.model.flow_matching import flow_matching_loss
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes, build_flux_joint_attention_mask
from scripts.diagnostics import _flux_packing_correctness as metrics
from scripts.diagnostics._flux_backend_replay import attention_forward, fingerprint, tensor_metrics
from scripts.diagnostics._flux_packing import FlashBackend, packing_layout, packed_processors, packed_velocity
from scripts.diagnostics._flux_packing_preflight import validate_layout
from scripts.diagnostics._flux_gradient_boundaries import GradientBoundaryCapture, token_error_metrics, DEFAULT_STAGES
from scripts.diagnostics._flux_modulation_backward import observe_modulation_reduction


VARIANTS = ("cudnn_single", "cudnn_padded_single", "cudnn_padded_batch",
            "cudnn_packed_ab", "cudnn_packed_ba", "fa3_single", "fa3_packed_ab", "fa3_packed_ba")
BASE_REFERENCES = {
    "cudnn_single": (), "cudnn_padded_single": ("cudnn_single",),
    "cudnn_padded_batch": ("cudnn_single", "cudnn_padded_single"),
    "cudnn_packed_ab": ("cudnn_single", "cudnn_padded_batch"),
    "cudnn_packed_ba": ("cudnn_single", "cudnn_packed_ab"),
    "fa3_single": (), "fa3_packed_ab": ("fa3_single",),
    "fa3_packed_ba": ("fa3_single", "fa3_packed_ab"),
}
BLOCKS = ("double.0", "double.4", "single.0", "single.19")


def case_plan():
    return [dict(name=f"{name}/{repeat}", variant=name, repeat=repeat,
                 references=[f"{ref}/0" for ref in BASE_REFERENCES[name]] if repeat == 0 else [f"{name}/0"],
                 backward_calls=2 if name.endswith("single") else 1,
                 labels=["B", "A"] if name.endswith("ba") else ["A", "B"])
            for name in VARIANTS for repeat in range(2)]


def retained_after(cases, index):
    """Only retain gradient snapshots that a future comparison will consume."""
    return {name for row in cases[index + 1:] for name in row["references"]}


@contextmanager
def training_precision(device, sdpa_backend=SDPBackend.CUDNN_ATTENTION):
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.enable_grad(), sdpa_kernel(sdpa_backend):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def assemble(singletons, order, *, padded_length=None):
    """Assemble native model inputs without consulting packing metadata."""
    selected = [singletons[index] for index in order]
    prefixes = [item["text"][0] for item in selected]
    text = pad_flux_text_prefixes(prefixes, padded_length=padded_length or max(map(len, prefixes)))
    # A separate leaf also exposes any gradient incorrectly flowing into padding.
    values = text.encoder_hidden_states.detach().requires_grad_(True)
    return dict(text=values, text_mask=text.text_attention_mask, text_positions=text.text_position_ids,
                images=torch.cat([item["images"] for item in selected]),
                image_positions=torch.cat([item["image_positions"] for item in selected]),
                timesteps=torch.cat([item["timesteps"] for item in selected]),
                target=torch.cat([item["target"] for item in selected]),
                labels=tuple(item["labels"][0] for item in selected), lengths=tuple(map(len, prefixes)))


def native_forward(model, inputs):
    # Padded singleton controls require a mask just as the padded batch does.
    mask = build_flux_joint_attention_mask(inputs["text_mask"], inputs["images"].shape[1])
    masked = len(inputs["labels"]) > 1 or not bool(inputs["text_mask"].all())
    return model(hidden_states=inputs["images"], encoder_hidden_states=inputs["text"],
                 timestep=inputs["timesteps"], img_ids=inputs["image_positions"],
                 txt_ids=inputs["text_positions"], guidance=None,
                 joint_attention_kwargs={"attention_mask": mask} if masked else {}, return_dict=True).sample


def canonicalizer(inputs, packed):
    """Recover sample identities by lengths, independently of PackingLayout indices."""
    lengths, labels = inputs["lengths"], inputs["labels"]
    images, padding = inputs["images"].shape[1], inputs["text"].shape[1]

    def split(stage, output):
        if stage == "text.mlp.linear_out":
            if packed:
                if output.shape[:2] != (1, sum(lengths)):
                    raise ValueError("packed text MLP shape mismatch")
                chunks = output[0].split(lengths)
            else:
                if output.shape[:2] != (len(labels), padding):
                    raise ValueError("native text MLP shape mismatch")
                chunks = [output[i, :length] for i, length in enumerate(lengths)]
            return dict(zip(labels, chunks))
        if stage.startswith("double."):
            text, image = output
            if packed:
                if text.shape[:2] != (1, sum(lengths)) or image.shape[:2] != (1, len(labels) * images):
                    raise ValueError("packed Double shape mismatch")
                chunks = [torch.cat((t, im)) for t, im in zip(text[0].split(lengths), image[0].split(images))]
            else:
                if text.shape[:2] != (len(labels), padding) or image.shape[:2] != (len(labels), images):
                    raise ValueError("native Double shape mismatch")
                chunks = [torch.cat((text[i, :length], image[i])) for i, length in enumerate(lengths)]
        else:
            if packed:
                if output.shape[:2] != (1, sum(lengths) + len(labels) * images):
                    raise ValueError("packed Single shape mismatch")
                chunks = output[0].split(tuple(length + images for length in lengths))
            else:
                if output.shape[:2] != (len(labels), padding + images):
                    raise ValueError("native Single shape mismatch")
                chunks = [torch.cat((output[i, :length], output[i, padding:])) for i, length in enumerate(lengths)]
        return dict(zip(labels, chunks))
    return split


@contextmanager
def singleton_attention(model, backend, *, retain=False):
    """Observe original attention calls; replace only the FA3 kernel boundary."""
    original = df.dispatch_attention_fn
    stages = [*(f"double.{i}" for i in range(len(model.transformer_blocks))),
              *(f"single.{i}" for i in range(len(model.single_transformer_blocks)))]
    state = dict(calls=[], records={})

    def dispatch(q, k, v, *args, **kwargs):
        stage = stages[len(state["calls"])]
        state["calls"].append(stage)
        if backend == "fa3":
            if args or kwargs.get("attn_mask") is not None:
                raise ValueError("FA3 original-forward reference must be unpadded")
            output = attention_forward("fa3", *(value.to(v.dtype).contiguous() for value in (q, k, v)))
        else:
            output = original(q, k, v, *args, **kwargs)
        if retain and stage in BLOCKS:
            record = {"qkv": tuple(value.detach().to(v.dtype).cpu().contiguous() for value in (q, k, v))}
            state["records"][stage] = record
            def gradient_hook(value):
                record["upstream"] = value.detach().cpu().clone()
            output.register_hook(gradient_hook)
        return output

    with patch.object(df, "dispatch_attention_fn", dispatch):
        yield state
    if state["calls"] != stages:
        raise ValueError("incomplete original attention dispatch")
    if retain and (set(state["records"]) != set(BLOCKS)
                   or any("upstream" not in value for value in state["records"].values())):
        raise ValueError("missing real-loss attention capture")


def validate_dispatch(operators, variant, calls, layers):
    expected = layers * (2 if "packed" in variant and variant.startswith("cudnn") else calls)
    if variant.startswith("cudnn"):
        if metrics.observed_sdpa_backend(operators, expected) != SDPBackend.CUDNN_ATTENTION:
            raise ValueError("requested cuDNN did not run")
        if operators.get("aten::_scaled_dot_product_cudnn_attention_backward", 0) != expected:
            raise ValueError("cuDNN backward coverage mismatch")
        if any("flash_attn" in key.lower() for key in operators):
            raise ValueError("unexpected FlashAttention")
    else:
        if any("scaled_dot_product" in key for key in operators):
            raise ValueError("unexpected SDPA in FA3 case")
        for key in ("flash_attn_3::_flash_attn_forward", "flash_attn_3::_flash_attn_backward"):
            if operators.get(key, 0) != expected:
                raise ValueError("FA3 forward/backward coverage mismatch")


def run_case(model, singletons, variant, *, padded_length, global_batch_size=48,
             compute_dtype=torch.bfloat16, sdpa_backend=SDPBackend.CUDNN_ATTENTION,
             capture_stages=DEFAULT_STAGES, profile=True, retain_attention=False,
             modulation_reduction="native"):
    if variant not in VARIANTS or len(singletons) != 2 or global_batch_size < 2:
        raise ValueError("invalid backward comparison contract")
    packed = "packed" in variant
    if modulation_reduction not in ("native", "fp32") or (modulation_reduction == "fp32" and not packed):
        raise ValueError("FP32 modulation reduction requires a packed variant; otherwise use native")
    order = (1, 0) if variant.endswith("ba") else (0, 1)
    groups = [(index,) for index in order] if variant.endswith("single") else [order]
    device = singletons[0]["images"].device
    model.zero_grad(set_to_none=True)
    outputs, losses, captures, attention_records, padding_checks, backward_losses = {}, {}, {}, {}, [], []
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with profiler as recorded, observe_modulation_reduction() as reduction_evidence:
        for indices in groups:
            pad = padded_length if "padded" in variant else None
            inputs = assemble(singletons, indices, padded_length=pad)
            identities = {key: fingerprint(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            if packed:
                layout = packing_layout(inputs["lengths"], inputs["images"].shape[1], device)
                validate_layout(layout)
                kernel = FlashBackend("fa3") if variant.startswith("fa3") else metrics.SegmentedSDPA(sdpa_backend)
                route = packed_processors(model, layout, kernel)
            else:
                route = singleton_attention(model, "fa3" if variant.startswith("fa3") else "cudnn",
                                            retain=retain_attention)
            observation = GradientBoundaryCapture(model, canonicalizer(inputs, packed), stages=capture_stages)
            with training_precision(device, sdpa_backend), observation, route as routing:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=compute_dtype == torch.bfloat16):
                    if packed:
                        text = torch.cat([singletons[index]["text"][0] for index in indices])
                        output = packed_velocity(model, text, inputs["images"], inputs["image_positions"],
                                                 inputs["timesteps"], layout,
                                                 modulation_reduction=modulation_reduction)
                    else:
                        output = native_forward(model, inputs)
                    loss = flow_matching_loss(output, inputs["target"])
                weighted_loss = loss * (len(indices) / global_batch_size)
                backward_losses.append(float(weighted_loss.detach()))
                weighted_loss.backward()
            for index, label in enumerate(inputs["labels"]):
                outputs[label] = output[index].detach().cpu().clone()
                losses[label] = float(flow_matching_loss(output[index:index + 1].detach(), inputs["target"][index:index + 1]))
            for stage, fields in observation.values.items():
                for field, values in fields.items():
                    captures.setdefault(stage, {}).setdefault(field, {}).update(values)
            if retain_attention:
                attention_records[inputs["labels"][0]] = routing["records"]
            if not packed:
                grad = inputs["text"].grad
                if grad is None:
                    raise ValueError("missing text input gradients")
                masked = grad[~inputs["text_mask"]]
                nonzero = int(torch.count_nonzero(masked))
                padding_checks.append(dict(labels=list(inputs["labels"]), positions=int((~inputs["text_mask"]).sum()),
                                           nonzero_gradient_elements=nonzero))
                if nonzero:
                    raise ValueError("text padding received nonzero gradients")
            if identities != {key: fingerprint(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}:
                raise ValueError("model modified shared inputs")
            del output, loss, weighted_loss, observation, inputs
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if any(parameter.grad is None for parameter in trainable.values()):
        raise ValueError("incomplete trainable parameter gradient coverage")
    if any(parameter.dtype != torch.float32 or parameter.grad.dtype != torch.float32 for parameter in trainable.values()):
        raise ValueError("parameters and persistent gradients must stay FP32")
    gradient_norm = float(torch.stack([p.grad.norm(dtype=torch.float64).square() for p in trainable.values()]).sum().sqrt())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("non-finite parameter gradient")
    operators = metrics.attention_operators(recorded) if profile else {}
    if profile:
        validate_dispatch(operators, variant, len(groups), len(model.transformer_blocks) + len(model.single_transformer_blocks))
    expected_reductions = 3 if modulation_reduction == "fp32" else 0
    if (reduction_evidence["forward_calls"] != expected_reductions
            or reduction_evidence["backward_calls"] != expected_reductions):
        raise ValueError("modulation reduction forward/backward coverage mismatch")
    row = dict(variant=variant, backward_calls=len(groups), losses=losses,
               modulation_reduction=modulation_reduction, modulation_reduction_evidence=reduction_evidence,
               logical_mean_loss=sum(losses.values()) / 2, backward_loss=sum(backward_losses),
               sample_weights={"A": 1 / global_batch_size, "B": 1 / global_batch_size},
               elapsed_seconds=elapsed, operators=operators, dispatch_verified=profile,
               parameter_count=sum(p.numel() for p in trainable.values()), gradient_tensors=len(trainable), gradient_norm=gradient_norm,
               parameter_versions={name: p._version for name, p in trainable.items()},
               padding_checks=padding_checks, peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               comparisons={})
    return row, dict(outputs=outputs, boundaries=captures), attention_records


def gradient_snapshot(model):
    return {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def compare_gradients(model, reference):
    current = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}
    if current.keys() != reference.keys():
        raise ValueError("gradient parameter identities differ")
    rows = {name: metrics.metrics(metrics.difference_totals(value, reference[name])) for name, value in current.items()}
    groups = defaultdict(list)
    for name, row in rows.items():
        parts = name.split(".")
        key = ".".join(parts[:2]) if parts[0] in ("transformer_blocks", "single_transformer_blocks") else parts[0]
        groups[key].append(row)
    total = metrics.aggregate(rows.values())
    components = {key: metrics.aggregate(values) for key, values in groups.items()}
    for row in (total, *components.values(), *rows.values()):
        row["norm_ratio"] = math.sqrt(row["actual_squared"] / row["reference_squared"]) if row["reference_squared"] else None
        row["angle_degrees"] = math.degrees(math.acos(row["cosine"])) if row["cosine"] is not None else None
        row["share_of_squared_error"] = row["error_squared"] / total["error_squared"] if total["error_squared"] else 0.0
    return dict(total=total, components=components, parameters=rows)


def compare_states(actual, reference, lengths, *, edge_width=32):
    result = dict(velocity={label: tensor_metrics(actual["outputs"][label], reference["outputs"][label]) for label in "AB"},
                  boundaries={})
    if actual["boundaries"].keys() != reference["boundaries"].keys():
        raise ValueError("boundary stage coverage mismatch")
    for stage, fields in actual["boundaries"].items():
        result["boundaries"][stage] = {field: {label: token_error_metrics(value, reference["boundaries"][stage][field][label],
                    text_length=lengths[label], edge_width=edge_width) for label, value in values.items()}
                for field, values in fields.items()}
    return result
