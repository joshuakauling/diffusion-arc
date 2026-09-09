"""Localize FLUX packing differences on one real batch, without optimizer updates.

The segmented SDPA oracle preserves the packed model layout but executes one
attention call per example. It is a correctness reference, not a timed backend.
"""

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
import gc
import math
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics import probe_flux_packing as probe
from scripts.diagnostics._flux_packing import FlashBackend, packed_processors

VARIANTS = ("sdpa_b2", "sdpa_segmented", "fa3")
WORKER_SECONDS = 480
SDPA_OPERATORS = {
    "aten::_scaled_dot_product_cudnn_attention": SDPBackend.CUDNN_ATTENTION,
    "aten::_scaled_dot_product_flash_attention": SDPBackend.FLASH_ATTENTION,
    "aten::_scaled_dot_product_efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
    "aten::_scaled_dot_product_attention_math": SDPBackend.MATH,
}


def correctness_policy(policy):
    groups = tuple(g for g in policy.groups if g["name"] == "short_b4")
    if len(groups) != 1:
        raise ValueError("correctness mode requires the recorded short_b4 group")
    windows = tuple(w for w in policy.windows if w["name"] == groups[0]["window"])
    return replace(policy, groups=groups, windows=windows, backends=("fa3",),
                   max_duration_seconds=min(policy.max_duration_seconds, WORKER_SECONDS))


def build_plan(policy, training):
    return [{k: v for k, v in group.items() if k != "round_orders"}
            for group in probe.build_plan(policy, training)]


def attention_operators(profile):
    return {e.key: e.count for e in profile.key_averages()
            if any(s in e.key.lower() for s in ("scaled_dot_product", "flash_attn", "flux_packing::"))}


def observed_sdpa_backend(operators, layers):
    selected = [backend for name, backend in SDPA_OPERATORS.items() if operators.get(name, 0)]
    if len(selected) != 1 or sum(operators.get(n, 0) for n in SDPA_OPERATORS) != layers:
        raise ValueError(f"cannot establish a single reference SDPA backend: {operators}")
    return selected[0]


class SegmentedSDPA:
    """Explicitly split joint sequences, using the observed baseline backend."""

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, q, k, v, layout):
        lengths = tuple(n + layout.image_tokens for n in layout.text_lengths)
        with sdpa_kernel(self.backend):
            return torch.cat([
                F.scaled_dot_product_attention(
                    a.transpose(0, 1).unsqueeze(0), b.transpose(0, 1).unsqueeze(0),
                    c.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=False,
                    scale=q.shape[-1] ** -0.5,
                ).squeeze(0).transpose(0, 1)
                for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))
            ])


class CaptureAttention:
    """Retain real post-RoPE Q/K/V and the loss's incoming attention gradient."""

    def __init__(self, backend, double_layers):
        self.backend = backend
        self.selected = {0: "transformer_blocks.0", double_layers: "single_transformer_blocks.0"}
        self.calls = 0
        self.records = {}

    def __call__(self, q, k, v, layout):
        name = self.selected.get(self.calls)
        self.calls += 1
        output = self.backend(q, k, v, layout)
        if name is not None:
            record = {"qkv": tuple(t.detach().cpu().clone() for t in (q, k, v))}
            self.records[name] = record
            def save_gradient(gradient):
                record["output_gradient"] = gradient.detach().cpu().clone()
            output.register_hook(save_gradient)
        return output


def difference_totals(actual, reference):
    """Bounded scratch space; aggregate in FP64, including gradient direction."""
    if actual.shape != reference.shape:
        raise ValueError("comparison shape mismatch")
    totals = torch.zeros(4, dtype=torch.float64, device=actual.device)
    maximum = torch.zeros((), dtype=torch.float32, device=actual.device)
    for a, b in zip(actual.detach().reshape(-1).split(1 << 20), reference.reshape(-1).split(1 << 20)):
        a, b = a.double(), b.to(device=actual.device, dtype=torch.float64)
        delta = a - b
        totals += torch.stack((delta.square().sum(), b.square().sum(), a.square().sum(), (a * b).sum()))
        maximum = torch.maximum(maximum, delta.abs().max().float())
    values = totals.cpu().tolist()
    maximum = maximum.item()
    if not all(math.isfinite(v) for v in (*values, maximum)):
        raise FloatingPointError("non-finite numerical comparison")
    return dict(zip(("error_squared", "reference_squared", "actual_squared", "dot"), values),
                max_abs=maximum, count=actual.numel())


def metrics(totals):
    error, reference, actual = (totals[k] for k in ("error_squared", "reference_squared", "actual_squared"))
    return {**totals, "rms": math.sqrt(error / totals["count"]),
            "relative_l2": math.sqrt(error / reference) if reference else (0.0 if error == 0 else None),
            "cosine": max(-1.0, min(1.0, totals["dot"] / math.sqrt(reference * actual)))
                      if reference and actual else None}


def aggregate(rows):
    rows = list(rows)
    return metrics({**{k: sum(r[k] for r in rows) for k in (
        "error_squared", "reference_squared", "actual_squared", "dot", "count")},
        "max_abs": max(r["max_abs"] for r in rows)})


def within(value, threshold):
    return value is not None and value <= threshold


def compare_model(model, output, loss, reference, tolerances):
    ref_output, ref_gradients, ref_loss = reference
    actual = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    if actual.keys() != ref_gradients.keys():
        raise ValueError("gradient coverage differs from reference")
    parameters = {n: metrics(difference_totals(g, ref_gradients[n])) for n, g in actual.items()}
    components = defaultdict(list)
    for name, row in parameters.items():
        parts = name.split(".")
        component = ".".join(parts[:2]) if parts[0] in ("transformer_blocks", "single_transformer_blocks") else parts[0]
        components[component].append(row)
    gradient = aggregate(parameters.values())
    by_component = {name: aggregate(rows) for name, rows in components.items()}
    for row in (*parameters.values(), *by_component.values()):
        row["share_of_squared_gradient_error"] = (row["error_squared"] / gradient["error_squared"]
                                                  if gradient["error_squared"] else 0.0)
    velocity = metrics(difference_totals(output, ref_output))
    loss_relative = abs(loss - ref_loss) / max(abs(ref_loss), 1e-12)
    passed = (within(velocity["relative_l2"], tolerances["velocity_relative_l2"])
              and within(gradient["relative_l2"], tolerances["gradient_relative_l2"])
              and loss_relative <= tolerances["loss_relative"])
    return dict(velocity=velocity, gradient=gradient, loss_relative=loss_relative,
                parameters=parameters, components=by_component, thresholds=tolerances, passed=passed,
                training_equivalence_established=False)


def snapshot(model, output, loss):
    return (output.detach().cpu().clone(),
            {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}, loss)


def model_call(model, optimizer, variant, backend, group, data, compute_dtype, weight):
    context = packed_processors(model, data["layout"], backend) if backend else nullcontext()
    with context, torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        row, output = probe.measure_call(model, optimizer, variant, group, data,
                                         compute_dtype=compute_dtype, weight=weight)
    return row, output, attention_operators(profile)


def kernel_comparison(record, reference_backend, flash, layout, device, tolerances):
    """Compare VJPs using identical BF16 inputs AND identical upstream gradients."""
    qkv = tuple(t.to(device).detach().requires_grad_() for t in record["qkv"])
    upstream = record["output_gradient"].to(device)
    # Match model autocast, even though inputs at this boundary are already BF16.
    with torch.autocast(device_type=device.type, dtype=qkv[0].dtype, enabled=qkv[0].dtype != torch.float32):
        expected = reference_backend(*qkv, layout)
    expected_gradients = torch.autograd.grad(expected, qkv, grad_outputs=upstream)
    flash.observe = True
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities) as profile:
            with torch.autocast(device_type=device.type, dtype=qkv[0].dtype, enabled=qkv[0].dtype != torch.float32):
                actual = flash(*qkv, layout)
            gradients = torch.autograd.grad(actual, qkv, grad_outputs=upstream)
    finally:
        flash.observe = False
    output = metrics(difference_totals(actual, expected.detach()))
    grad = {name: metrics(difference_totals(a, b))
            for name, a, b in zip(("q", "k", "v"), gradients, expected_gradients)}
    combined = aggregate(grad.values())
    operators = attention_operators(profile)
    if device.type == "cuda":
        probe.validate_backend_events(operators, "fa3", 1)
    passed = (within(output["relative_l2"], tolerances["velocity_relative_l2"])
              and all(within(g["relative_l2"], tolerances["gradient_relative_l2"]) for g in grad.values()))
    return dict(output=output, gradients=grad, combined_gradient=combined, passed=passed,
                thresholds=tolerances, qkv_shapes=[list(t.shape) for t in qkv],
                qkv_dtypes=[str(t.dtype) for t in qkv], upstream_dtype=str(upstream.dtype),
                upstream_rms=float(upstream.float().square().mean().sqrt()),
                upstream_source="sdpa_segmented production weighted flow loss", operators=operators,
                cuda_kernel_names=sorted({e.name for e in profile.events()
                                         if e.device_type == torch.autograd.DeviceType.CUDA}))


def run_worker(args):
    result = dict(status="running", phase="configuration", mode="correctness", cases=[], kernel_comparisons={})
    def persist(phase=None):
        if phase:
            result["phase"] = phase
        probe.shapes._write_json(args.output_dir / "result.json", result)
        print(probe.PROGRESS_MARKER, result["phase"], flush=True)
    try:
        policy, training = probe.load_probe_config(args.config)
        policy = correctness_policy(policy)
        group = build_plan(policy, training)[0]
        result.update(group=group, numerical_tolerances=policy.numerical_tolerances)
        persist("fa3_small_preflight")
        device = probe.shapes.prepare_single_gpu_device(training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("correctness image requires Hopper/SM90")
        flash = FlashBackend("fa3")
        result["backend_preflight"] = probe.backend_preflight(flash, device, policy.numerical_tolerances)
        persist("load_models_and_training_memory")
        device, models, optimizer, reserve, setup = probe.shapes.prepare_runtime(training)
        model = models.transformer
        if model.config.attention_head_dim != 128:
            raise ValueError("correctness image requires head dimension 128")
        result.update(setup=setup, environment=probe.shapes._environment_metadata(),
                      flash_binding=flash.binding, flash_version=probe.version("flash-attn-3"))
        persist("prepare_real_inputs")
        stored = probe.prepare_inputs(policy, training, models)
        data = probe.group_inputs(stored[group["name"]], group, training.seed, device)
        result.update(input_sha256=data["input_sha256"], flow_seed=training.seed)
        layers = len(model.transformer_blocks) + len(model.single_transformer_blocks)
        references = {}
        oracle = capture = None
        for variant in VARIANTS:
            persist(f"{variant}/forward_backward")
            backend = None if variant == "sdpa_b2" else capture if variant == "sdpa_segmented" else flash
            row, output, operators = model_call(model, optimizer, variant, backend, group, data,
                models.flux_compute_dtype, group["batch_size"] / training.loop.optimizer_batch_size)
            case = dict(variant=variant, status="running", diagnostic_call=row,
                        operators=operators, comparisons={}, timings_are_benchmark_results=False)
            result["cases"].append(case)
            if variant == "sdpa_b2":
                selected = observed_sdpa_backend(operators, layers)
                result["reference_sdpa_backend"] = selected.name
                oracle = SegmentedSDPA(selected)
                capture = CaptureAttention(oracle, len(model.transformer_blocks))
            elif variant == "sdpa_segmented":
                if observed_sdpa_backend(operators, layers * group["batch_size"]) != oracle.backend:
                    raise ValueError("segmented SDPA changed the reference backend")
                if capture.calls != layers or set(capture.records) != set(capture.selected.values()):
                    raise ValueError("incomplete double/single attention captures")
                if any("output_gradient" not in r for r in capture.records.values()):
                    raise ValueError("missing captured upstream gradients")
            else:
                # Full-model calls bypass SDPA and use the explicit FA3 binding.
                for op in ("flash_attn_3::_flash_attn_forward", "flash_attn_3::_flash_attn_backward"):
                    if operators.get(op, 0) != layers:
                        raise ValueError(f"full model did not execute {layers} {op} calls")
                if any("scaled_dot_product" in n for n in operators):
                    raise ValueError("unexpected SDPA in the FA3 model pass")
            for name, reference in references.items():
                persist(f"{variant}/compare_to_{name}")
                case["comparisons"][name] = compare_model(model, output, row["loss"], reference,
                                                           policy.numerical_tolerances)
                persist()
            if variant != "fa3":
                references[variant] = snapshot(model, output, row["loss"])
            case["status"] = "complete"
            del output
            persist()
        # Release the two 4B-parameter CPU references before the kernel audit.
        references.clear()
        del reference
        gc.collect()
        for name, record in capture.records.items():
            persist(f"{name}/isolated_attention_forward_backward")
            result["kernel_comparisons"][name] = kernel_comparison(
                record, oracle, flash, data["layout"], device, policy.numerical_tolerances)
            persist()
        result.update(status="complete", phase="complete",
                      numerical_passed=all(c["passed"] for case in result["cases"]
                          for c in case["comparisons"].values())
                          and all(c["passed"] for c in result["kernel_comparisons"].values()))
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        persist()
    return result


def validate_complete(result):
    cases = result.get("cases", [])
    if ([c["variant"] for c in cases] != list(VARIANTS)
            or any(c["status"] != "complete" for c in cases)
            or [set(c["comparisons"]) for c in cases] != [set(), {"sdpa_b2"}, {"sdpa_b2", "sdpa_segmented"}]
            or set(result.get("kernel_comparisons", {})) != {"transformer_blocks.0", "single_transformer_blocks.0"}):
        raise ValueError("incomplete correctness comparisons")


def write_report(path, result):
    def percent(value):
        return f"{100 * value:.4f}%" if value is not None else "undefined (zero reference)"
    lines = ["# FLUX packing correctness diagnostic", "", f"Execution: **{result['status']}**.",
             f"Numerical screening passed: **{result.get('numerical_passed', 'incomplete')}**.", "",
             "One fixed real short_b4 batch. No optimizer updates or speedup claims.",
             "A completed diagnostic can contain failed numerical comparisons.", "",
             f"Reference and segmented SDPA backend: `{result.get('reference_sdpa_backend', 'unknown')}`.", "",
             "| Actual vs reference | Velocity relative L2 | Gradient relative L2 | Gradient cosine | Loss relative | Passed |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for case in result.get("cases", []):
        for reference, comparison in case["comparisons"].items():
            c = comparison
            lines.append(f"| {case['variant']} vs {reference} | {percent(c['velocity']['relative_l2'])} | "
                         f"{percent(c['gradient']['relative_l2'])} | {c['gradient']['cosine']} | "
                         f"{percent(c['loss_relative'])} | {c['passed']} |")
    for case in result.get("cases", []):
        for reference, c in case["comparisons"].items():
            lines += ["", f"## Largest gradient-error contributions: {case['variant']} vs {reference}", "",
                      "| Component | Relative L2 | Share of squared gradient error |", "| --- | ---: | ---: |"]
            for name, row in sorted(c["components"].items(), key=lambda item: item[1]["error_squared"], reverse=True)[:12]:
                lines.append(f"| {name} | {percent(row['relative_l2'])} | {percent(row['share_of_squared_gradient_error'])} |")
    lines += ["", "## Isolated FA3 vs segmented SDPA", "",
              "Identical captured BF16 Q/K/V and identical upstream gradients from the real flow loss.", "",
              "| Block | Output relative L2 | dQ relative L2 | dK relative L2 | dV relative L2 | Passed |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, c in result.get("kernel_comparisons", {}).items():
        values = [percent(c["gradients"][n]["relative_l2"]) for n in ("q", "k", "v")]
        lines.append(f"| {name} | {percent(c['output']['relative_l2'])} | " + " | ".join(values) + f" | {c['passed']} |")
    if result.get("error"):
        lines += ["", f"Failure during `{result['phase']}`: {result['error']}"]
    lines += ["", "Kernel audits sample the first block of each type. CPU tests do not establish CUDA agreement.",
              "Full parameter/component metrics, observed operators and isolated CUDA kernel names are in summary.json."]
    path.write_text("\n".join(lines) + "\n")
