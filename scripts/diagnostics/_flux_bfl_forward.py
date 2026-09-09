"""Observe unmodified BFL and Diffusers forwards on identical logical inputs."""

from contextlib import contextmanager
from functools import wraps

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc.model.flux_conditioning import build_flux_joint_attention_mask, pad_flux_text_prefixes
from scripts.diagnostics._flux_packing_forward import difference, fingerprint


def stage_order(double_blocks, single_blocks):
    return ["time_embedding", "image_projection", "text_projection",
            *(f"double.{i}" for i in range(double_blocks)),
            *(f"single.{i}" for i in range(single_blocks)), "final_norm", "velocity"]


def prepare_comparison_inputs(data):
    """Promote the once-prepared tensors to FP32 for both precision conditions."""
    prefixes = tuple(p.detach().float() for p in data["prefixes"])
    if len(prefixes) != 2 or any(p.ndim != 2 or not p.shape[0] for p in prefixes):
        raise ValueError("BFL comparison requires two nonempty text prefixes")
    offsets = [0, prefixes[0].shape[0], sum(p.shape[0] for p in prefixes)]
    result = dict(prefixes=prefixes,
                  text_positions=tuple(data["layout"].text_positions[offsets[i]:offsets[i + 1]]
                                       for i in range(2)),
                  images=data["flow"].noisy_latents.detach().float(),
                  timesteps=data["flow"].timesteps.detach().float(),
                  target=data["flow"].target_velocity.detach().float(),
                  image_positions=data["image_positions"])
    if (result["images"].ndim != 3 or result["images"].shape[0] != 2
            or result["target"].shape != result["images"].shape
            or result["timesteps"].shape != (2,)
            or result["image_positions"].shape != (*result["images"].shape[:2], 4)):
        raise ValueError("invalid prepared image, position, timestep or target shapes")
    for prefix, positions in zip(prefixes, result["text_positions"]):
        if positions.shape != (prefix.shape[0], 4):
            raise ValueError("text position shape differs from the valid prefix")
    for tensor in (*prefixes, result["images"], result["timesteps"], result["target"]):
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite comparison inputs")
    return result


def input_identity(data):
    return {label: {name: fingerprint(value) for name, value in {
                "text": data["prefixes"][i], "text_positions": data["text_positions"][i],
                "noisy_latents": data["images"][i], "timestep": data["timesteps"][i:i + 1],
                "target_velocity": data["target"][i], "image_positions": data["image_positions"][i],
            }.items()} for i, label in enumerate(("A", "B"))}


def call_inputs(data, indices, padded_length=None):
    """Build the common batch tensors without resampling or re-encoding."""
    prefixes = tuple(data["prefixes"][i] for i in indices)
    assembled = pad_flux_text_prefixes(prefixes, padded_length=padded_length or max(p.shape[0] for p in prefixes))
    # Preserve original RoPE coordinates, even if future inputs use other positions.
    positions = assembled.text_position_ids.clone()
    for row, index in enumerate(indices):
        positions[row, :prefixes[row].shape[0]] = data["text_positions"][index]
    if len(indices) > 1:
        # Native Diffusers uses the first example's position IDs for the batch.
        # Reject inputs it cannot represent instead of measuring another model.
        for index in indices[1:]:
            common = min(data["prefixes"][indices[0]].shape[0], data["prefixes"][index].shape[0])
            if (not torch.equal(data["image_positions"][indices[0]], data["image_positions"][index])
                    or not torch.equal(data["text_positions"][indices[0]][:common],
                                       data["text_positions"][index][:common])):
                raise ValueError("native Diffusers batch requires shared position IDs")
        # The reference row may be shorter; all positions used by later samples
        # must still match the IDs that Diffusers will broadcast from row zero.
        for row, length in enumerate(p.shape[0] for p in prefixes):
            if not torch.equal(positions[0, :length], positions[row, :length]):
                raise ValueError("native Diffusers batch requires shared position IDs")
    selection = torch.tensor(indices, device=data["images"].device)
    return dict(images=data["images"].index_select(0, selection),
                text=assembled.encoder_hidden_states, text_positions=positions,
                image_positions=data["image_positions"].index_select(0, selection),
                timesteps=data["timesteps"].index_select(0, selection),
                target=data["target"].index_select(0, selection),
                text_mask=assembled.text_attention_mask,
                lengths=tuple(p.shape[0] for p in prefixes), labels=tuple("AB"[i] for i in indices))


def call_identity(inputs):
    return {key: fingerprint(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}


@contextmanager
def controlled_precision(precision, device):
    """Use the same Math SDPA and disable TF32; retain native model dtype rules."""
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


class ForwardCapture:
    """Capture matching boundaries; BFL's direct method calls bypass module hooks."""

    def __init__(self, model, implementation, inputs, references=None):
        self.model, self.implementation, self.inputs = model, implementation, inputs
        self.references = references or {}
        self.snapshots = {label: {} for label in inputs["labels"]}
        self.comparisons = {name: {label: {} for label in inputs["labels"]} for name in self.references}
        self.outputs = {}
        if implementation == "bfl":
            self.double, self.single = model.double_blocks, model.single_blocks
        elif implementation == "diffusers":
            self.double, self.single = model.transformer_blocks, model.single_transformer_blocks
        else:
            raise ValueError("unknown transformer implementation")

    def record(self, stage, streams):
        if stage in self.outputs:
            raise ValueError(f"duplicate observation at {stage}")
        self.outputs[stage] = {}
        for i, label in enumerate(self.inputs["labels"]):
            parts = {name: tensor[i, :self.inputs["lengths"][i]] if name == "text" else tensor[i]
                     for name, tensor in streams.items()}
            snapshots = {name: tensor.detach().cpu().clone() for name, tensor in parts.items()}
            if any(not torch.isfinite(t).all() for t in snapshots.values()):
                raise FloatingPointError(f"non-finite output at {stage}/{label}")
            self.snapshots[label][stage] = snapshots
            self.outputs[stage][label] = {name: dict(shape=list(t.shape), dtype=str(t.dtype))
                                           for name, t in snapshots.items()}
            for reference_name, reference in self.references.items():
                self.comparisons[reference_name][label][stage] = {
                    name: difference(tensor, reference[label][stage][name])
                    for name, tensor in snapshots.items()}

    def block(self, stage, kind, output):
        if kind == "double":
            if self.implementation == "bfl":
                image, text, _ = output
            else:
                text, image = output
        else:
            joint = output[0] if self.implementation == "bfl" else output
            text_length = self.inputs["text"].shape[1]
            text, image = joint[:, :text_length], joint[:, text_length:]
        if (image.shape[:2] != self.inputs["images"].shape[:2]
                or text.shape[:2] != self.inputs["text"].shape[:2]):
            raise ValueError(f"unexpected block output shape at {stage}")
        self.record(stage, {"text": text, "image": image})

    @contextmanager
    def installed(self):
        handles, methods = [], []
        try:
            if self.implementation == "bfl":
                projections = (("time_embedding", "conditioning", self.model.time_in),
                               ("image_projection", "image", self.model.img_in),
                               ("text_projection", "text", self.model.txt_in))
                final_projection = self.model.final_layer.linear
            else:
                projections = (("time_embedding", "conditioning", self.model.time_guidance_embed),
                               ("image_projection", "image", self.model.x_embedder),
                               ("text_projection", "text", self.model.context_embedder))
                final_projection = self.model.proj_out
            for stage, stream, module in projections:
                def observe(module, args, output, stage=stage, stream=stream):
                    self.record(stage, {stream: output})
                handles.append(module.register_forward_hook(observe))
            for kind, blocks in (("double", self.double), ("single", self.single)):
                for i, module in enumerate(blocks):
                    stage = f"{kind}.{i}"
                    if self.implementation == "bfl":
                        method = module.forward_kv_extract
                        existed = "forward_kv_extract" in module.__dict__
                        previous = module.__dict__.get("forward_kv_extract")
                        @wraps(method)
                        def observed(*args, method=method, stage=stage, kind=kind, **kwargs):
                            output = method(*args, **kwargs)
                            self.block(stage, kind, output)
                            return output
                        methods.append((module, existed, previous))
                        module.forward_kv_extract = observed
                    else:
                        def observe(module, args, output, stage=stage, kind=kind):
                            self.block(stage, kind, output)
                        handles.append(module.register_forward_hook(observe))
            handles.append(final_projection.register_forward_pre_hook(
                lambda module, args: self.record("final_norm", {"image": args[0]})))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for module, existed, previous in reversed(methods):
                if existed:
                    module.forward_kv_extract = previous
                else:
                    del module.forward_kv_extract

    def finish(self, output):
        if output.shape != self.inputs["images"].shape:
            raise ValueError("unexpected final velocity shape")
        self.record("velocity", {"image": output})
        if list(self.outputs) != stage_order(len(self.double), len(self.single)):
            raise ValueError("incomplete or reordered transformer boundary observations")


def forward_call(model, implementation, inputs):
    if implementation == "bfl":
        if len(inputs["labels"]) != 1 or not inputs["text_mask"].all():
            raise ValueError("unmodified BFL reference requires an unpadded singleton")
        return model(x=inputs["images"], x_ids=inputs["image_positions"],
                     timesteps=inputs["timesteps"], ctx=inputs["text"],
                     ctx_ids=inputs["text_positions"], guidance=None)
    if implementation != "diffusers":
        raise ValueError("unknown transformer implementation")
    # Singleton controls have no mask. Only the batch exercises our mask route.
    attention_kwargs = {} if len(inputs["labels"]) == 1 else {
        "attention_mask": build_flux_joint_attention_mask(inputs["text_mask"], inputs["images"].shape[1])}
    return model(hidden_states=inputs["images"], encoder_hidden_states=inputs["text"],
                 timestep=inputs["timesteps"], img_ids=inputs["image_positions"],
                 txt_ids=inputs["text_positions"], guidance=None,
                 joint_attention_kwargs=attention_kwargs, return_dict=True).sample


def comparison_summary(comparisons, losses, reference_losses, tolerance):
    result = {}
    for reference, samples in comparisons.items():
        result[reference] = {}
        for label, stages in samples.items():
            rows = [(f"{stage}/{stream}", row) for stage, streams in stages.items()
                    for stream, row in streams.items()]
            failed = [name for name, row in rows if row["relative_l2"] is None
                      or row["relative_l2"] > tolerance["output_relative_l2"]]
            loss_ref = reference_losses[reference][label]
            loss_error = abs(losses[label] - loss_ref)
            relative_loss = loss_error / abs(loss_ref) if loss_ref else (0.0 if loss_error == 0 else None)
            result[reference][label] = dict(
                first_nonidentical=next((name for name, row in rows if not row["bitwise_equal"]), None),
                first_outside_tolerance=next(iter(failed), None),
                maximum_relative_l2=max((row["relative_l2"] for _, row in rows if row["relative_l2"] is not None), default=None),
                velocity_relative_l2=stages["velocity"]["image"]["relative_l2"],
                loss_relative=relative_loss,
                passed=not failed and relative_loss is not None and relative_loss <= tolerance["loss_relative"])
    return result


def validate_attention_operators(operators, implementation, layers):
    """Count actual Math executions; autocast can add nested SDPA API events."""
    minimum_calls = layers * (2 if implementation == "bfl" else 1)
    if (operators.get("aten::scaled_dot_product_attention", 0) < minimum_calls
            or operators.get("aten::_scaled_dot_product_attention_math") != layers
            or any(s in name.lower() for name in operators
                   for s in ("flash_attn", "cudnn", "efficient", "backward"))):
        raise ValueError(f"unexpected Math SDPA execution: {operators}")


def run_case(model, implementation, inputs, precision, tolerance, *, references=None,
             reference_losses=None, profile=False):
    capture = ForwardCapture(model, implementation, inputs, references)
    device = inputs["images"].device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    from contextlib import nullcontext
    with controlled_precision(precision, device), capture.installed(), (
            torch.profiler.profile(activities=activities) if profile else nullcontext()) as profiler:
        output = forward_call(model, implementation, inputs)
        capture.finish(output)
        losses = {label: float((output[i].float() - inputs["target"][i]).square().mean())
                  for i, label in enumerate(inputs["labels"])}
    row = dict(implementation=implementation, precision=precision, labels=list(inputs["labels"]),
               status="complete", sdpa_backend="math", tf32=False, grad_enabled=False,
               outputs=capture.outputs, losses=losses, comparisons=capture.comparisons,
               comparison_summary=comparison_summary(capture.comparisons, losses,
                   reference_losses or {}, tolerance))
    if profile:
        operators = {e.key: e.count for e in profiler.key_averages()
                     if any(s in e.key.lower() for s in ("scaled_dot_product", "flash_attn", "backward"))}
        layers = len(capture.double) + len(capture.single)
        # BFL also calls SDPA for empty reference queries. PyTorch returns those
        # before Math. Conversely autocast can record nested nonempty API calls.
        validate_attention_operators(operators, implementation, layers)
        row["operators"] = operators
        row["cuda_kernel_names"] = sorted({event.name for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA})
        if device.type == "cuda" and not row["cuda_kernel_names"]:
            raise ValueError("CUDA profile has no CUDA kernels")
    return row, capture.snapshots
