"""Replay frozen FLUX operands across shapes, independently of attention.

Native BF16 operation replays use FP32 model weights. Modulation reductions
differentiate detached module outputs only; their FP32 variants are labelled
diagnostics and do not change model arithmetic or populate parameter gradients.
"""

from contextlib import contextmanager
import math

import torch

from scripts.diagnostics._flux_backend_replay import fingerprint, tensor_metrics


DEFAULT_SEEDS = (173, 419, 701)
LAYOUTS = ("a", "b", "ab", "ba", "ab_repeat")
PRIMITIVES = ("attention_gate", "attention_residual", "norm", "modulation",
              "mlp.linear_in", "mlp.activation", "mlp.linear_out", "mlp",
              "mlp_gate", "output_residual")
CHAIN_STAGES = tuple(name for name in PRIMITIVES if name != "mlp")
MODULATIONS = {"image": "double_stream_modulation_img", "text": "double_stream_modulation_txt",
               "joint": "single_stream_modulation"}
OPERATIONS = ("projection", *(f"modulation.{name}" for name in MODULATIONS),
              *(f"{stream}.{name}" for stream in ("image", "text") for name in (*PRIMITIVES, "chain")))


def _metrics(actual, reference):
    row = tensor_metrics(actual, reference)
    row["bitwise_equal"] = (actual.dtype == reference.dtype and
                            torch.equal(actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)))
    return row


@contextmanager
def _precision():
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def replay_plan(seeds=DEFAULT_SEEDS):
    """Declare bounded operation counts independently of model token lengths."""
    seeds = tuple(seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("operation replays require three distinct nonnegative upstream seeds")
    return dict(layouts=list(LAYOUTS), upstream_seeds=list(seeds),
                forward_operation_groups=26, forward_replay_cases=130,
                forward_comparison_metrics=336, capture_replay_metrics=36,
                reduction_cases=3 * len(seeds) * 2 * 4,
                reduction_comparison_metrics=3 * len(seeds) * 2 * 16,
                reduction_autograd_calls=3 * len(seeds) * 10,
                reduction_layouts=["broadcast", "index_ab", "index_ba", "index_ab_repeat"],
                reduction_precisions=["native_bf16", "fp32_reduction_diagnostic"],
                full_model_forwards=0, model_parameter_gradients=False, optimizer_updates=0)


class OperationCapture:
    """Passively retain one original singleton's operation operands on device."""

    def __init__(self, model):
        if not model.transformer_blocks:
            raise ValueError("operation capture requires a Double block")
        self.model, self.values, self.completed = model, {}, False

    def _record(self, name, value):
        if name in self.values:
            raise ValueError(f"duplicate operation capture: {name}")
        if not isinstance(value, torch.Tensor) or value.shape[0] != 1:
            raise ValueError(f"operation capture requires singleton tensor operands: {name}")
        self.values[name] = value.detach().clone()

    @contextmanager
    def installed(self):
        if self.values:
            raise ValueError("operation capture cannot be reused")
        handles = []

        def pre(module, name):
            def observe(module, args, kwargs):
                self._record(name, args[0] if args else next(iter(kwargs.values())))
            handles.append(module.register_forward_pre_hook(observe, with_kwargs=True))

        def post(module, name):
            handles.append(module.register_forward_hook(lambda module, args, output: self._record(name, output)))

        try:
            block = self.model.transformer_blocks[0]
            def block_input(module, args, kwargs):
                for index, (argument, name) in enumerate((
                    ("hidden_states", "image.input"), ("encoder_hidden_states", "text.input"),
                    ("temb_mod_img", "image.modulation"), ("temb_mod_txt", "text.modulation"))):
                    self._record(f"first.{name}", kwargs[argument] if argument in kwargs else args[index])
            def attention_output(module, args, output):
                image, text = output
                self._record("first.image.attention", image)
                self._record("first.text.attention", text)
            def block_output(module, args, output):
                text, image = output
                self._record("first.image.output", image)
                self._record("first.text.output", text)
            handles.append(block.register_forward_pre_hook(block_input, with_kwargs=True))
            handles.append(block.attn.register_forward_hook(attention_output))
            handles.append(block.register_forward_hook(block_output))
            for stream, norm, mlp in (("image", block.norm2, block.ff),
                                      ("text", block.norm2_context, block.ff_context)):
                prefix = f"first.{stream}"
                pre(norm, f"{prefix}.residual")
                post(norm, f"{prefix}.norm")
                pre(mlp, f"{prefix}.mlp_input")
                post(mlp, f"{prefix}.mlp_output")
                pre(mlp.act_fn, f"{prefix}.linear_in_output")
                pre(mlp.linear_out, f"{prefix}.activation_output")
            for name, attribute in MODULATIONS.items():
                module = getattr(self.model, attribute)
                pre(module, f"modulation.{name}.input")
                post(module, f"modulation.{name}.output")
            pre(self.model.proj_out, "projection.input")
            post(self.model.proj_out, "projection.output")
            yield self
            if len(self.values) != 28:
                raise ValueError("incomplete operation capture")
            self.completed = True
        finally:
            for handle in reversed(handles):
                handle.remove()

    def __enter__(self):
        self._context = self.installed()
        return self._context.__enter__()

    def __exit__(self, *args):
        return self._context.__exit__(*args)


def _split_modulation(value):
    if value.ndim == 2:
        value = value.unsqueeze(1)
    _, _, attention_gate, shift, scale, gate = value.chunk(6, dim=-1)
    return attention_gate, shift, scale, gate


def _frozen_stream(capture, stream):
    values = {key.removeprefix(f"first.{stream}."): value for key, value in capture.values.items()
              if key.startswith(f"first.{stream}.")}
    attention_gate, shift, scale, gate = _split_modulation(values["modulation"])
    # Freeze products in the original singleton shape for isolated add replays.
    return dict(values, attention_gate_value=attention_gate, shift=shift, scale=scale, gate=gate,
                gated_attention=attention_gate * values["attention"], gated_mlp=gate * values["mlp_output"])


def _combine(values, order, length_key="input"):
    lengths = [values[i][length_key].shape[1] for i in order]
    if len(order) == 1:
        return values[order[0]], lengths
    combined = {}
    for key in values[0]:
        if key in ("modulation", "attention_gate_value", "shift", "scale", "gate"):
            continue
        tensors = [values[i][key] for i in order]
        combined[key] = torch.cat(tensors, dim=1)
    # Preserve the production strided chunk views of the expanded six-way
    # modulation, rather than concatenating each gate into a different layout.
    source = torch.cat([values[i]["modulation"] for i in order], dim=0)
    owners = torch.arange(len(order), device=source.device).repeat_interleave(
        torch.tensor(lengths, device=source.device))
    combined["modulation"] = source.index_select(0, owners).unsqueeze(0)
    attention_gate, shift, scale, gate = _split_modulation(combined["modulation"])
    combined.update(attention_gate_value=attention_gate, shift=shift, scale=scale, gate=gate)
    return combined, lengths


def _primitive(block, stream, name, x):
    norm, mlp = (block.norm2, block.ff) if stream == "image" else (block.norm2_context, block.ff_context)
    if name == "attention_gate":
        return x["attention_gate_value"] * x["attention"]
    if name == "attention_residual":
        return x["input"] + x["gated_attention"]
    if name == "norm":
        return norm(x["residual"])
    if name == "modulation":
        return x["norm"] * (1 + x["scale"]) + x["shift"]
    if name == "mlp.linear_in":
        return mlp.linear_in(x["mlp_input"])
    if name == "mlp.activation":
        return mlp.act_fn(x["linear_in_output"])
    if name == "mlp.linear_out":
        return mlp.linear_out(x["activation_output"])
    if name == "mlp":
        return mlp(x["mlp_input"])
    if name == "mlp_gate":
        return x["gate"] * x["mlp_output"]
    if name == "output_residual":
        return x["residual"] + x["gated_mlp"]
    raise ValueError(f"unknown replay primitive: {name}")


def _chain(block, stream, x):
    result = {}
    x = dict(x)
    updates = {"attention_gate": "gated_attention", "attention_residual": "residual", "norm": "norm",
               "modulation": "mlp_input", "mlp.linear_in": "linear_in_output",
               "mlp.activation": "activation_output", "mlp.linear_out": "mlp_output", "mlp_gate": "gated_mlp"}
    for name in CHAIN_STAGES:
        result[name] = _primitive(block, stream, name, x)
        if name in updates:
            x[updates[name]] = result[name]
    return result


def _parts(output, order, lengths, *, batch=False):
    pieces = output.split(1, dim=0) if batch else output.split(lengths, dim=1)
    return {"AB"[index]: value for index, value in zip(order, pieces)}


def _forward_replays(model, captures):
    rows, faithful = [], {}
    streams = {name: tuple(_frozen_stream(capture, name) for capture in captures) for name in ("image", "text")}
    block = model.transformer_blocks[0]
    for operation in OPERATIONS:
        singleton, repeated = {}, {}
        for layout in LAYOUTS:
            order = (0,) if layout == "a" else (1,) if layout == "b" else (1, 0) if layout == "ba" else (0, 1)
            if operation == "projection":
                operands = [captures[i].values["projection.input"] for i in order]
                lengths = [value.shape[1] for value in operands]
                # Full-transformer packing restores [B, image_tokens, D] here.
                value = torch.cat(operands, dim=0)
                outputs = {"output": model.proj_out(value)}
                input_shapes = {"input": list(value.shape)}
                input_strides = {"input": list(value.stride())}
            elif operation.startswith("modulation."):
                name = operation.split(".")[1]
                value = torch.cat([captures[i].values[f"modulation.{name}.input"] for i in order], dim=0)
                lengths = [1] * len(order)
                outputs = {"output": getattr(model, MODULATIONS[name])(value)}
                input_shapes = {"input": list(value.shape)}
                input_strides = {"input": list(value.stride())}
            else:
                stream, name = operation.split(".", 1)
                values, lengths = _combine(streams[stream], order)
                outputs = _chain(block, stream, values) if name == "chain" else {
                    "output": _primitive(block, stream, name, values)}
                input_shapes = {key: list(value.shape) for key, value in values.items()}
                input_strides = {key: list(value.stride()) for key, value in values.items()}
            parts = {stage: _parts(value, order, lengths,
                                  batch=operation == "projection" or operation.startswith("modulation."))
                     for stage, value in outputs.items()}
            row = dict(operation=operation, layout=layout, labels=["AB"[i] for i in order],
                       input_shapes=input_shapes, input_strides=input_strides,
                       output_shapes={stage: {label: list(value.shape) for label, value in labels.items()}
                                      for stage, labels in parts.items()}, comparisons={})
            if len(order) == 1:
                label = "AB"[order[0]]
                singleton.setdefault(operation, {})[label] = {stage: labels[label] for stage, labels in parts.items()}
                if operation == "projection":
                    expected = {"output": captures[order[0]].values["projection.output"]}
                elif operation.startswith("modulation."):
                    expected = {"output": captures[order[0]].values[f"{operation}.output"]}
                elif operation.endswith(".chain"):
                    stream = operation.split(".")[0]
                    values = streams[stream][order[0]]
                    expected = {stage: values[key] for stage, key in (
                        ("attention_residual", "residual"), ("norm", "norm"), ("modulation", "mlp_input"),
                        ("mlp.linear_in", "linear_in_output"), ("mlp.activation", "activation_output"),
                        ("mlp.linear_out", "mlp_output"), ("output_residual", "output"))}
                else:
                    expected = {}
                faithful.setdefault(operation, {})[label] = {
                    stage: _metrics(parts[stage][label], value) for stage, value in expected.items()}
            else:
                row["comparisons"]["singletons"] = {label: {
                    stage: _metrics(labels[label], singleton[operation][label][stage])
                    for stage, labels in parts.items()} for label in ("AB"[i] for i in order)}
                if layout == "ab_repeat":
                    row["comparisons"]["ab"] = {label: {
                        stage: _metrics(labels[label], repeated[operation][stage][label])
                        for stage, labels in parts.items()} for label in ("A", "B")}
                if layout == "ab":
                    repeated[operation] = parts
            row["first_differences"] = {reference: {label: next((stage for stage, metrics in stages.items()
                if not metrics["bitwise_equal"]), None) for label, stages in labels.items()}
                for reference, labels in row["comparisons"].items()}
            rows.append(row)
    return rows, faithful


def reduction_gradient(values, upstream, order, *, indexed):
    """Differentiate broadcast or repeated ownership with identical token gradients."""
    if sorted(order) != list(range(len(values))) or len(values) != len(upstream):
        raise ValueError("reduction order must include each sample once")
    if any(value.shape != (1, gradient.shape[-1]) or gradient.ndim != 3 or gradient.shape[0] != 1
           or value.device != gradient.device or value.dtype != gradient.dtype
           for value, gradient in zip(values, upstream)):
        raise ValueError("modulation values and upstream shapes/dtypes/devices differ")
    with torch.enable_grad():
        if indexed:
            source = torch.cat(values, dim=0).detach().requires_grad_(True)
            owners = torch.cat([torch.full((upstream[i].shape[1],), i, dtype=torch.long, device=source.device)
                                for i in order])
            expanded = source.index_select(0, owners).unsqueeze(0)
            gradient = torch.autograd.grad(expanded, source, torch.cat([upstream[i] for i in order], dim=1))[0]
            return tuple(gradient.split(1, dim=0))
        result = []
        for value, gradient in zip(values, upstream):
            source = value.detach().requires_grad_(True)
            result.append(torch.autograd.grad(source.unsqueeze(1).expand_as(gradient), source, gradient)[0])
        return tuple(result)


def _reductions(captures, seeds):
    rows = []
    for name in MODULATIONS:
        values = tuple(capture.values[f"modulation.{name}.output"] for capture in captures)
        image_lengths = [capture.values["first.image.input"].shape[1] for capture in captures]
        text_lengths = [capture.values["first.text.input"].shape[1] for capture in captures]
        lengths = image_lengths if name == "image" else text_lengths if name == "text" else [
            i + t for i, t in zip(image_lengths, text_lengths)]
        for seed in seeds:
            generator = torch.Generator(device=values[0].device).manual_seed(seed)
            upstream = tuple(torch.randn((1, length, value.shape[-1]), device=value.device,
                                         dtype=torch.float32, generator=generator).to(value.dtype)
                             for value, length in zip(values, lengths))
            upstream_identity = {label: fingerprint(value) for label, value in zip("AB", upstream)}
            fp32_reference = reduction_gradient(tuple(v.float() for v in values),
                                                tuple(v.float() for v in upstream), (0, 1), indexed=False)
            for precision in ("native_bf16", "fp32_reduction_diagnostic"):
                operands = values if precision == "native_bf16" else tuple(v.float() for v in values)
                gradients = upstream if precision == "native_bf16" else tuple(v.float() for v in upstream)
                baseline, repeated = None, None
                for layout in ("broadcast", "index_ab", "index_ba", "index_ab_repeat"):
                    result = fp32_reference if precision == "fp32_reduction_diagnostic" and layout == "broadcast" else (
                        reduction_gradient(operands, gradients, (1, 0) if layout == "index_ba" else (0, 1),
                                           indexed=layout != "broadcast"))
                    comparisons = {"fp32_broadcast": {label: _metrics(value.float(), reference)
                                   for label, value, reference in zip("AB", result, fp32_reference)}}
                    if baseline is None:
                        baseline = result
                    else:
                        comparisons["broadcast"] = {label: _metrics(value, reference)
                            for label, value, reference in zip("AB", result, baseline)}
                    if layout == "index_ab":
                        repeated = result
                    elif layout == "index_ab_repeat":
                        comparisons["index_ab"] = {label: _metrics(value, reference)
                            for label, value, reference in zip("AB", result, repeated)}
                    rows.append(dict(module=name, seed=seed, precision=precision, layout=layout,
                                     value_dtype=str(operands[0].dtype), lengths=lengths,
                                     upstream_identity=upstream_identity, comparisons=comparisons,
                                     gradients={label: fingerprint(value) for label, value in zip("AB", result)}))
    return rows


def operation_replays(model, captures, *, seeds=DEFAULT_SEEDS):
    """Return JSON-safe frozen-operation and modulation-reduction evidence."""
    plan = replay_plan(seeds)
    if len(captures) != 2 or any(not capture.completed or capture.model is not model for capture in captures):
        raise ValueError("two complete singleton captures from this model are required")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("operation replays require FP32 persistent parameters")
    if any(capture.values[f"modulation.{name}.output"].dtype != torch.bfloat16
           for capture in captures for name in MODULATIONS):
        raise ValueError("operation captures must use production BF16 autocast")
    if captures[0].values["projection.input"].shape[1:] != captures[1].values["projection.input"].shape[1:]:
        raise ValueError("final projection requires equal image token shapes")
    identities = {label: {name: fingerprint(value) for name, value in capture.values.items()}
                  for label, capture in zip("AB", captures)}
    device = captures[0].values["projection.input"].device
    with _precision():
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            forwards, faithful = _forward_replays(model, captures)
        with torch.autocast(device_type=device.type, enabled=False):
            reductions = _reductions(captures, seeds)
    after = {label: {name: fingerprint(value) for name, value in capture.values.items()}
             for label, capture in zip("AB", captures)}
    if after != identities:
        raise ValueError("operation replay mutated frozen operands")
    if len(forwards) != plan["forward_replay_cases"] or len(reductions) != plan["reduction_cases"]:
        raise ValueError("incomplete operation replay cases")
    counts = dict(
        forward_comparison_metrics=sum(len(stages) for row in forwards for labels in row["comparisons"].values()
                                       for stages in labels.values()),
        capture_replay_metrics=sum(len(stages) for labels in faithful.values() for stages in labels.values()),
        reduction_comparison_metrics=sum(len(labels) for row in reductions for labels in row["comparisons"].values()))
    if any(count != plan[name] for name, count in counts.items()):
        raise ValueError("incomplete operation replay metrics")
    return dict(status="complete", plan=plan, input_identity=identities,
                input_hashes_unchanged=True, tf32=False, float32_matmul_precision="highest",
                compute_dtype="torch.bfloat16", parameter_dtype="torch.float32",
                capture_replay_bitwise_equal=all(metrics["bitwise_equal"] for labels in faithful.values()
                    for stages in labels.values() for metrics in stages.values()),
                capture_replay=faithful, forward_cases=forwards, reduction_cases=reductions,
                scope="Frozen local operations and detached modulation-output gradients; not full-model backward")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_metric(metric, shape, *, exact=False):
    numeric = ("rms", "max_abs", "reference_l2", "actual_l2", "reference_rms", "actual_rms")
    fields = {*numeric, "relative_l2", "cosine", "count", "actual_dtype", "reference_dtype", "bitwise_equal"}
    _require(set(metric) == fields, "missing or unexpected operation metric fields")
    _require(type(metric["count"]) is int and metric["count"] == math.prod(shape), "operation metric count mismatch")
    for name in numeric:
        value = metric[name]
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 "non-finite or negative operation metric")
    expected = metric["reference_l2"]
    relative = metric["relative_l2"]
    if expected:
        _require(type(relative) in (int, float) and math.isfinite(relative)
                 and math.isclose(relative, metric["rms"] * math.sqrt(metric["count"]) / expected,
                                  rel_tol=1e-10, abs_tol=1e-14), "invalid relative operation metric")
    else:
        _require(relative == (0.0 if metric["rms"] == 0 else None), "invalid zero-reference metric")
    for prefix in ("reference", "actual"):
        _require(math.isclose(metric[f"{prefix}_l2"], metric[f"{prefix}_rms"] * math.sqrt(metric["count"]),
                             rel_tol=1e-10, abs_tol=1e-14), "inconsistent operation norms")
        _require(metric[f"{prefix}_dtype"] in ("torch.float32", "torch.bfloat16"), "invalid operation dtype")
    cosine = metric["cosine"]
    if expected and metric["actual_l2"]:
        _require(type(cosine) in (int, float) and math.isfinite(cosine) and -1 <= cosine <= 1,
                 "invalid operation cosine")
    else:
        _require(cosine is None, "zero-norm operation cosine must be undefined")
    _require(type(metric["bitwise_equal"]) is bool, "invalid exactness evidence")
    if metric["bitwise_equal"]:
        _require(metric["max_abs"] == metric["rms"] == 0
                 and metric["actual_dtype"] == metric["reference_dtype"], "inconsistent exactness evidence")
    _require(not exact or metric["bitwise_equal"], "same-shape operation replay differs from original or repeat")


def _validate_fingerprint(identity):
    _require(set(identity) == {"shape", "dtype", "sha256"}, "incomplete operation fingerprint")
    _require(isinstance(identity["shape"], list) and identity["shape"]
             and all(type(size) is int and size > 0 for size in identity["shape"]), "invalid operand shape")
    _require(identity["dtype"] in ("torch.float32", "torch.bfloat16"), "invalid operand dtype")
    digest = identity["sha256"]
    _require(isinstance(digest, str) and len(digest) == 64
             and all(char in "0123456789abcdef" for char in digest), "invalid operation fingerprint hash")


def _operation_shape(identities, label, operation, stage):
    if operation == "projection" or operation.startswith("modulation."):
        key = f"{operation}.output"
    else:
        stream, primitive = operation.split(".", 1)
        if primitive == "chain":
            primitive = stage
        key = f"first.{stream}." + {
            "attention_gate": "attention", "attention_residual": "residual", "norm": "norm",
            "modulation": "mlp_input", "mlp.linear_in": "linear_in_output", "mlp.activation": "activation_output",
            "mlp.linear_out": "mlp_output", "mlp": "mlp_output", "mlp_gate": "mlp_output", "output_residual": "output",
        }[primitive]
    return identities[label][key]["shape"]


def _validate_operation_result(result, seeds):
    plan = replay_plan(seeds)
    _require(result["status"] == "complete" and result["plan"] == plan, "incomplete operation result or plan")
    _require(result["input_hashes_unchanged"] is True and result["capture_replay_bitwise_equal"] is True,
             "operation operands changed or passive capture replay was not exact")
    _require(result["compute_dtype"] == "torch.bfloat16" and result["parameter_dtype"] == "torch.float32"
             and result["tf32"] is False and result["float32_matmul_precision"] == "highest", "invalid operation precision")
    identities = result["input_identity"]
    _require(set(identities) == {"A", "B"}, "operation result omitted a sample identity")
    names = {"projection.input", "projection.output",
             *(f"modulation.{name}.{kind}" for name in MODULATIONS for kind in ("input", "output")),
             *(f"first.{stream}.{kind}" for stream in ("image", "text") for kind in (
                 "input", "modulation", "attention", "output", "residual", "norm", "mlp_input", "mlp_output",
                 "linear_in_output", "activation_output"))}
    for values in identities.values():
        _require(set(values) == names, "incomplete frozen operand identities")
        for identity in values.values():
            _validate_fingerprint(identity)
            _require(identity["shape"][0] == 1, "captured operand is not a singleton")
    faithful_count = forward_count = reduction_count = 0
    faithful = result["capture_replay"]
    _require(set(faithful) == set(OPERATIONS), "missing capture replay groups")
    for operation, labels in faithful.items():
        stages = ([name for name in CHAIN_STAGES if name not in ("attention_gate", "mlp_gate")]
                  if operation.endswith(".chain") else ["output"]
                  if operation == "projection" or operation.startswith("modulation.") else [])
        _require(set(labels) == {"A", "B"}, "capture replay omitted a sample")
        for label, metrics in labels.items():
            _require(set(metrics) == set(stages), "capture replay omitted a boundary")
            for stage, metric in metrics.items():
                _validate_metric(metric, _operation_shape(identities, label, operation, stage), exact=True)
                faithful_count += 1
    forwards = result["forward_cases"]
    _require([(row["operation"], row["layout"]) for row in forwards]
             == [(operation, layout) for operation in OPERATIONS for layout in LAYOUTS], "missing or reordered forward operation cases")
    for row in forwards:
        operation, layout = row["operation"], row["layout"]
        labels = ["A"] if layout == "a" else ["B"] if layout == "b" else ["B", "A"] if layout == "ba" else ["A", "B"]
        stages = list(CHAIN_STAGES) if operation.endswith(".chain") else ["output"]
        references = [] if len(labels) == 1 else ["singletons", "ab"] if layout == "ab_repeat" else ["singletons"]
        _require(row["labels"] == labels and set(row["output_shapes"]) == set(stages),
                 "invalid operation sample or stage coverage")
        shapes, strides = row["input_shapes"], row["input_strides"]
        _require(set(shapes) == set(strides) and "input" in shapes, "incomplete input layout evidence")
        for name, shape in shapes.items():
            _require(isinstance(shape, list) and shape and all(type(size) is int and size > 0 for size in shape),
                     "invalid operation input shape")
            _require(len(strides[name]) == len(shape) and all(type(stride) is int and stride >= 0 for stride in strides[name]),
                     "invalid operation input strides")
        for stage, shapes in row["output_shapes"].items():
            _require(set(shapes) == set(labels) and all(shape == _operation_shape(identities, label, operation, stage)
                                                   for label, shape in shapes.items()), "invalid replay output shape")
        _require(set(row["comparisons"]) == set(references), "missing operation comparison")
        first = {}
        for reference, samples in row["comparisons"].items():
            _require(set(samples) == set(labels), "operation comparison omitted a sample")
            first[reference] = {}
            for label, metrics in samples.items():
                _require(set(metrics) == set(stages), "operation comparison omitted a boundary")
                for stage, metric in metrics.items():
                    _validate_metric(metric, _operation_shape(identities, label, operation, stage), exact=reference == "ab")
                    forward_count += 1
                first[reference][label] = next((stage for stage in stages if not metrics[stage]["bitwise_equal"]), None)
        _require(row["first_differences"] == first, "incorrect first operation boundary summary")
    reductions = result["reduction_cases"]
    precisions = plan["reduction_precisions"]
    _require([(row["module"], row["seed"], row["precision"], row["layout"]) for row in reductions]
             == [(name, seed, precision, layout) for name in MODULATIONS for seed in seeds
                 for precision in precisions for layout in plan["reduction_layouts"]], "missing or reordered reduction cases")
    incoming = {}
    for row in reductions:
        name, seed, layout = row["module"], row["seed"], row["layout"]
        dtype = "torch.bfloat16" if row["precision"] == "native_bf16" else "torch.float32"
        _require(row["value_dtype"] == dtype, "invalid reduction precision")
        _require(set(row["upstream_identity"]) == set(row["gradients"]) == {"A", "B"}, "reduction omitted a sample")
        lengths = []
        for label in "AB":
            shape = identities[label][f"modulation.{name}.output"]["shape"]
            length = sum(identities[label][f"first.{stream}.input"]["shape"][1]
                         for stream in (("image", "text") if name == "joint" else (name,)))
            lengths.append(length)
            for identity in (row["upstream_identity"][label], row["gradients"][label]):
                _validate_fingerprint(identity)
            _require(row["upstream_identity"][label]["shape"] == [1, length, shape[-1]]
                     and row["upstream_identity"][label]["dtype"] == "torch.bfloat16", "invalid identical upstream contract")
            _require(row["gradients"][label]["shape"] == shape and row["gradients"][label]["dtype"] == dtype,
                     "invalid modulation gradient shape or dtype")
        _require(row["lengths"] == lengths, "wrong reduction ownership lengths")
        key = name, seed
        _require(row["upstream_identity"] == incoming.setdefault(key, row["upstream_identity"]), "reduction upstream changed")
        references = ["fp32_broadcast"] + ([] if layout == "broadcast" else ["broadcast"]) + (
            ["index_ab"] if layout == "index_ab_repeat" else [])
        _require(set(row["comparisons"]) == set(references), "missing reduction comparison")
        for metrics in row["comparisons"].values():
            _require(set(metrics) == {"A", "B"}, "reduction comparison omitted a sample")
            for label, metric in metrics.items():
                _validate_metric(metric, identities[label][f"modulation.{name}.output"]["shape"])
                reduction_count += 1
    _require((forward_count, faithful_count, reduction_count) == (
        plan["forward_comparison_metrics"], plan["capture_replay_metrics"], plan["reduction_comparison_metrics"]),
        "incomplete operation comparison metrics")
    return True


def validate_operation_result(result, *, seeds=DEFAULT_SEEDS):
    """Validate complete persisted coverage and exact capture/forward repeats.

    Repeated-index backward may vary; its differences remain measured evidence.
    Missing, malformed, non-finite, or unfaithful operation results fail validation.
    """
    try:
        return _validate_operation_result(result, tuple(seeds))
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"malformed or incomplete operation result: {error}") from error
