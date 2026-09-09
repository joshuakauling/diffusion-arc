"""Compare real FLUX singletons with packing, using forward execution only.

Block snapshots use explicit sequence offsets independently of the packing
permutations. The original model's SDPA singletons provide separate controls.
"""

from contextlib import contextmanager, nullcontext
import hashlib
from itertools import accumulate
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusion_arc.model.contracts import FlowMatchingBatch
from diffusion_arc.training.single_gpu_setup import load_single_gpu_models, seed_single_gpu_process
from scripts.diagnostics import probe_flux_packing as probe
from scripts.diagnostics import _flux_packing_correctness as checks
from scripts.diagnostics._flux_packing import FlashBackend, packed_processors, packing_layout
from scripts.diagnostics._flux_packing_forward_boundaries import boundary_hooks, boundary_stage_order

VARIANTS = ("fa3_a", "fa3_b", "fa3_ab", "fa3_ab_repeat", "fa3_ab_changed_b", "fa3_ab_changed_a",
            "fa3_ba", "fa3_ba_repeat", "fa3_ba_changed_a", "fa3_ba_changed_b", "sdpa_a", "sdpa_b")
WORKER_SECONDS = 480
PERTURBATION = "Selected example's text and noisy latents: -x + 0.125; timestep: 0.83 if t < 0.5 else 0.17"


def case_contract(variant):
    """Declare sample identity and references independently of tensor offsets."""
    if variant in ("fa3_a", "fa3_b", "sdpa_a", "sdpa_b"):
        labels = (variant[-1].upper(),)
        references = ("fa3_singletons", "fa3_ab", "fa3_ba") if variant.startswith("sdpa_") else ()
    else:
        order = variant.split("_")[1]
        labels = tuple(order.upper())
        references = (("fa3_singletons",) if variant == "fa3_ab" else
                      ("fa3_singletons", "fa3_ab") if variant == "fa3_ba" else (f"fa3_{order}",))
    return labels, references


def forward_policy(policy):
    policy = checks.correctness_policy(policy)
    if len(policy.groups[0]["target_indices"]) < 2:
        raise ValueError("forward comparison requires two real targets")
    return policy


def build_plan(policy, training):
    parent = checks.build_plan(forward_policy(policy), training)[0]
    lengths = parent["text_lengths"][:2]
    return [dict(name="short_ab", source_group=parent, sample_labels=["A", "B"],
                 target_indices=parent["target_indices"][:2], text_lengths=lengths,
                 image_tokens_per_sample=900, batch_size=2,
                 cumulative=[0, *accumulate(n + 900 for n in lengths)],
                 reversed_labels=["B", "A"], reversed_text_lengths=lengths[::-1],
                 reversed_cumulative=[0, *accumulate(n + 900 for n in lengths[::-1])],
                 maximum_joint_length=max(lengths) + 900,
                 cases=[dict(variant=v, labels=list(case_contract(v)[0]), references=list(case_contract(v)[1]))
                        for v in VARIANTS],
                 capture_stages=boundary_stage_order([*(f"transformer_blocks.{i}" for i in range(5)),
                                                     *(f"single_transformer_blocks.{i}" for i in range(20))]),
                 singleton_padding="exact valid prefix", perturbation=PERTURBATION)]


def slice_inputs(data, start, stop):
    """Slice already sampled tensors; never re-encode or draw fresh flow inputs."""
    prefixes = data["prefixes"][start:stop]
    if not prefixes or not 0 <= start < stop <= len(data["prefixes"]):
        raise ValueError("invalid example slice")
    return {**data, "prefixes": prefixes, "clean": data["clean"][start:stop],
            "flow": FlowMatchingBatch(**{k: getattr(data["flow"], k)[start:stop]
                for k in ("noisy_latents", "timesteps", "target_velocity")}),
            "image_positions": data["image_positions"][start:stop],
            "layout": packing_layout(tuple(p.shape[0] for p in prefixes),
                                     data["layout"].image_tokens, prefixes[0].device)}


def reordered_inputs(data, order):
    """Permute whole examples, preserving their text, flow draws and positions."""
    if sorted(order) != list(range(len(data["prefixes"]))):
        raise ValueError("order must be a permutation of the prepared examples")
    indices = torch.tensor(order, device=data["clean"].device)
    prefixes = tuple(data["prefixes"][i] for i in order)
    return {**data, "prefixes": prefixes, "clean": data["clean"].index_select(0, indices),
            "flow": FlowMatchingBatch(**{k: getattr(data["flow"], k).index_select(0, indices)
                for k in ("noisy_latents", "timesteps", "target_velocity")}),
            "image_positions": data["image_positions"].index_select(0, indices),
            "layout": packing_layout(tuple(p.shape[0] for p in prefixes), data["layout"].image_tokens,
                                     data["clean"].device)}


def changed_sample(data, index):
    """Perturb one example on new storage, preserving the other's inputs."""
    if len(data["prefixes"]) != 2 or index not in (0, 1):
        raise ValueError("perturbation requires one of two examples")
    prefixes = list(data["prefixes"])
    prefixes[index] = prefixes[index].clone().neg().add_(0.125)
    images, timesteps = (getattr(data["flow"], k).clone() for k in ("noisy_latents", "timesteps"))
    images[index].neg_().add_(0.125)
    timesteps[index] = torch.where(timesteps[index] < 0.5, 0.83, 0.17)
    return {**data, "prefixes": tuple(prefixes),
            "flow": FlowMatchingBatch(images, timesteps, data["flow"].target_velocity)}


def changed_b(data):
    return changed_sample(data, 1)


def fingerprint(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(raw).hexdigest())


def input_identity(data, labels):
    identities, offset = {}, 0
    for i, label in enumerate(labels):
        length = data["prefixes"][i].shape[0]
        identities[label] = {k: fingerprint(t) for k, t in {
            "text": data["prefixes"][i], "noisy_latents": data["flow"].noisy_latents[i],
            "timestep": data["flow"].timesteps[i:i + 1], "image_positions": data["image_positions"][i],
            "text_positions": data["layout"].text_positions[offset:offset + length],
        }.items()}
        offset += length
    return identities


def validate_input_reuse(data, singletons, perturbed, *, changed_label="B"):
    baseline = input_identity(data, ("A", "B"))
    for i, label in enumerate(("A", "B")):
        if input_identity(singletons[i], (label,))[label] != baseline[label]:
            raise ValueError("singleton did not preserve the exact prepared inputs")
    changed = input_identity(perturbed, ("A", "B"))
    unchanged = "A" if changed_label == "B" else "B"
    if changed[unchanged] != baseline[unchanged] or perturbed["layout"] is not data["layout"]:
        raise ValueError("perturbation changed the other example or packing metadata")
    for name, before in baseline[changed_label].items():
        after = changed[changed_label][name]
        if before["shape"] != after["shape"] or before["dtype"] != after["dtype"]:
            raise ValueError("perturbation changed a shape or dtype")
        should_change = name in ("text", "noisy_latents", "timestep")
        if (before["sha256"] != after["sha256"]) != should_change:
            raise ValueError(f"unexpected perturbation for {name}")
    return dict(original=baseline, **{f"changed_{changed_label.lower()}": changed},
                singleton_inputs_identical=True, **{f"{unchanged.lower()}_unchanged": True},
                packing_metadata_unchanged=True)


def split_block_output(kind, output, lengths, image_tokens):
    """Canonicalize block outputs without consulting any PackingLayout index."""
    count, text_total = len(lengths), sum(lengths)
    if kind == "double":
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("double block must return (text, image)")
        text, image = output
        if (text.ndim != 3 or image.ndim != 3 or text.shape[:2] != (1, text_total)
                or image.shape[:2] != (1, count * image_tokens)):
            raise ValueError("unexpected double block shape")
        offsets = (0, *accumulate(lengths))
        return [{"text": text[0, offsets[i]:offsets[i + 1]],
                 "image": image[0, i * image_tokens:(i + 1) * image_tokens]} for i in range(count)]
    if kind == "single":
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[:2] != (1, text_total + count * image_tokens):
            raise ValueError("unexpected single block shape")
        offsets = (0, *accumulate(n + image_tokens for n in lengths))
        return [{"text": output[0, offsets[i]:offsets[i] + n],
                 "image": output[0, offsets[i] + n:offsets[i + 1]]} for i, n in enumerate(lengths)]
    raise ValueError(f"unknown block kind: {kind}")


def difference(actual, reference):
    row = checks.metrics(checks.difference_totals(actual, reference))
    row["bitwise_equal"] = torch.equal(actual.contiguous().view(torch.uint8),
                                      reference.contiguous().view(torch.uint8))
    return row


class BlockCapture:
    """Keep only requested references; compare each candidate as its block ends."""

    def __init__(self, model, lengths, image_tokens, labels, *, references=None, retain=False):
        self.model = model
        self.lengths, self.image_tokens, self.labels = lengths, image_tokens, labels
        self.references, self.retain = references or {}, retain
        self.snapshots = {label: {} for label in labels}
        self.comparisons = {name: {label: {} for label in labels} for name in self.references}
        self.output_metadata = {}
        self.blocks = [(f"{group}.{i}", kind, block) for group, kind, blocks in (
            ("transformer_blocks", "double", model.transformer_blocks),
            ("single_transformer_blocks", "single", model.single_transformer_blocks))
            for i, block in enumerate(blocks)]

    def record(self, name, parts):
        if name in self.output_metadata or len(parts) != len(self.labels):
            raise ValueError("duplicate capture or wrong example count")
        self.output_metadata[name] = {}
        for label, streams in zip(self.labels, parts):
            snapshot = {key: value.detach().cpu().clone() for key, value in streams.items()}
            if any(not torch.isfinite(t).all() for t in snapshot.values()):
                raise FloatingPointError(f"non-finite output at {name}/{label}")
            self.output_metadata[name][label] = {k: dict(shape=list(t.shape), dtype=str(t.dtype))
                                                 for k, t in snapshot.items()}
            if self.retain:
                self.snapshots[label][name] = snapshot
            for reference_name, reference in self.references.items():
                self.comparisons[reference_name][label][name] = {
                    key: difference(value, reference[label][name][key]) for key, value in snapshot.items()}

    @contextmanager
    def installed(self):
        handles = []
        try:
            for name, kind, block in self.blocks:
                def hook(module, args, output, name=name, kind=kind):
                    self.record(name, split_block_output(kind, output, self.lengths, self.image_tokens))
                handles.append(block.register_forward_hook(hook))
            with boundary_hooks(self.model, self, split_block_output):
                yield self
        finally:
            for handle in handles:
                handle.remove()

    def finish(self, output):
        if output.ndim != 3 or output.shape[:2] != (len(self.labels), self.image_tokens):
            raise ValueError("unexpected velocity shape")
        self.record("velocity", [{"image": output[i]} for i in range(len(self.labels))])
        if list(self.output_metadata) != boundary_stage_order([n for n, _, _ in self.blocks]):
            raise ValueError("missing or reordered block captures")


def validate_forward_events(operators, *, packed, layers):
    if any(("backward" in name.lower() or "bwd" in name.lower()) and count
           for name, count in operators.items()):
        raise ValueError("forward-only diagnostic executed backward")
    if packed:
        if (operators.get("flash_attn_3::_flash_attn_forward", 0) != layers
                or operators.get("flux_packing::fa3", 0) != layers
                or any("scaled_dot_product" in name for name in operators)):
            raise ValueError(f"missing FA3 forward evidence or unexpected SDPA: {operators}")
    else:
        checks.observed_sdpa_backend(operators, layers)
        if any("flash_attn_3" in name or "flux_packing::" in name for name in operators):
            raise ValueError("original singleton retained a packed processor")


def model_call(model, data, labels, backend, compute_dtype, *, references=None, retain=False, profile=False):
    """One forward in a common grad/autocast mode, with scoped block observers."""
    lengths = tuple(p.shape[0] for p in data["prefixes"])
    capture = BlockCapture(model, lengths, data["layout"].image_tokens, labels,
                           references=references, retain=retain)
    assembled = (torch.cat(data["prefixes"]) if backend else
                 probe.pad_flux_text_prefixes(data["prefixes"], padded_length=lengths[0]))
    if backend is None and len(labels) != 1:
        raise ValueError("original control must be a singleton")
    processor_context = packed_processors(model, data["layout"], backend) if backend else nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    profiler_context = torch.profiler.profile(activities=activities) if profile else nullcontext()
    previous_observe = getattr(backend, "observe", False)
    if backend is not None:
        backend.observe = profile
    try:
        with torch.no_grad(), processor_context, capture.installed(), profiler_context as profiler:
            output = probe.forward_velocity(model, "fa3" if backend else "sdpa_b2", assembled, data, compute_dtype)
            capture.finish(output)
    finally:
        if backend is not None:
            backend.observe = previous_observe
    row = dict(status="complete", labels=list(labels), comparisons=capture.comparisons,
               outputs=capture.output_metadata, grad_enabled=False, model_training=model.training)
    if profile:
        row["operators"] = checks.attention_operators(profiler)
        validate_forward_events(row["operators"], packed=backend is not None, layers=len(capture.blocks))
        row["cuda_kernel_names"] = sorted({e.name for e in profiler.events()
                                           if e.device_type == torch.autograd.DeviceType.CUDA})
        if not row["cuda_kernel_names"]:
            raise ValueError("forward profile contains no CUDA kernels")
        if backend is None:
            row["sdpa_backend"] = checks.observed_sdpa_backend(row["operators"], len(capture.blocks)).name
    return row, capture.snapshots


def preflight(flash, device, tolerance):
    """Small FA3 forward and fixed-shape isolation check, with no autograd."""
    layout = packing_layout((3, 5), 2, device)
    generator = torch.Generator(device=device).manual_seed(181)
    with torch.no_grad():
        qkv = [torch.randn(12, 2, 128, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3)]
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.cat([F.scaled_dot_product_attention(
                *(t.transpose(0, 1) for t in parts), dropout_p=0.0, is_causal=False).transpose(0, 1)
                for parts in zip(*(t.split((5, 7)) for t in qkv))])
        actual = flash(*qkv, layout)
        changed = [t.clone() for t in qkv]
        for t in changed:
            t[5:].add_(2)
        isolated = flash(*changed, layout)
    delta = difference(actual.cpu(), expected.cpu())
    isolation = difference(isolated[:5].cpu(), actual[:5].cpu())
    if not checks.within(delta["relative_l2"], tolerance) or not isolation["bitwise_equal"]:
        raise ValueError(f"FA3 forward preflight failed: {delta}, isolation: {isolation}")
    return dict(output=delta, unchanged_segment=isolation, passed=True, backward_calls=0)


def comparison_summary(samples, tolerance):
    result = {}
    for label, stages in samples.items():
        blocks = [(name, stream, row) for name, streams in stages.items()
                  if (name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks."))
                  and name.count(".") == 1
                  for stream, row in streams.items()]
        first = next((f"{name}/{stream}" for name, stream, row in blocks if not row["bitwise_equal"]), None)
        worst = max(blocks, key=lambda item: item[2]["relative_l2"] if item[2]["relative_l2"] is not None else float("inf"))
        velocity = stages["velocity"]["image"]
        result[label] = dict(velocity=velocity, velocity_screening_passed=checks.within(velocity["relative_l2"], tolerance),
                             all_outputs_bitwise_equal=all(row["bitwise_equal"] for streams in stages.values() for row in streams.values()),
                             first_nonidentical_block=first,
                             first_nonidentical_capture=next((f"{name}/{stream}" for name, streams in stages.items()
                                 for stream, row in streams.items() if not row["bitwise_equal"]), None),
                             largest_block_relative_l2=dict(block=worst[0], stream=worst[1], value=worst[2]["relative_l2"]))
    return result


def run_comparisons(model, data, backend, compute_dtype, tolerance, *, profile=False,
                    on_case=lambda row: None, on_inputs=lambda identity: None):
    """Twelve fixed calls compare both orders and both isolation directions."""
    a, b = slice_inputs(data, 0, 1), slice_inputs(data, 1, 2)
    altered_a, altered_b = changed_sample(data, 0), changed_sample(data, 1)
    reversed_data = reordered_inputs(data, (1, 0))
    reversed_a, reversed_b = (reordered_inputs(d, (1, 0)) for d in (altered_a, altered_b))
    identity = {**validate_input_reuse(data, (a, b), altered_a, changed_label="A"),
                **validate_input_reuse(data, (a, b), altered_b)}
    for name, forward_data, reverse_data in (("original", data, reversed_data),
            ("changed_a", altered_a, reversed_a), ("changed_b", altered_b, reversed_b)):
        if input_identity(reverse_data, ("B", "A")) != identity[name]:
            raise ValueError("reordering changed a prepared example's contents or positions")
        if reverse_data["layout"].maximum != forward_data["layout"].maximum:
            raise ValueError("reordering changed maximum sequence length")
    identity["reordering_preserves_all_inputs"] = True
    identity["orders"] = {name: dict(labels=list(labels), text_lengths=list(d["layout"].text_lengths),
        cumulative=d["layout"].cumulative.cpu().tolist(), maximum=d["layout"].maximum)
        for name, labels, d in (("ab", ("A", "B"), data), ("ba", ("B", "A"), reversed_data))}
    identity["timesteps"] = dict(original=data["flow"].timesteps.cpu().tolist(),
                                 changed_a=altered_a["flow"].timesteps.cpu().tolist(),
                                 changed_b=altered_b["flow"].timesteps.cpu().tolist())
    on_inputs(identity)
    inputs = dict(fa3_a=a, fa3_b=b, fa3_ab=data, fa3_ab_repeat=data,
                  fa3_ab_changed_a=altered_a, fa3_ab_changed_b=altered_b,
                  fa3_ba=reversed_data, fa3_ba_repeat=reversed_data,
                  fa3_ba_changed_a=reversed_a, fa3_ba_changed_b=reversed_b, sdpa_a=a, sdpa_b=b)
    cases, snapshots = [], {"fa3_singletons": {}, "fa3_ab": {}, "fa3_ba": {}}
    for variant in VARIANTS:
        labels, reference_names = case_contract(variant)
        refs = {name: snapshots[name] for name in reference_names}
        retain = variant in ("fa3_a", "fa3_b", "fa3_ab", "fa3_ba")
        selected_backend = None if variant.startswith("sdpa_") else backend
        on_case(dict(variant=variant, status="running"))
        row, snapshot = model_call(model, inputs[variant], labels, selected_backend, compute_dtype,
                                   references=refs, retain=retain, profile=profile)
        row.update(variant=variant, summary={name: comparison_summary(values, tolerance)
                                           for name, values in row["comparisons"].items()})
        if variant in ("fa3_a", "fa3_b"):
            snapshots["fa3_singletons"].update(snapshot)
        elif retain:
            snapshots[variant].update(snapshot)
        cases.append(row)
        on_case(row)
    if input_identity(data, ("A", "B")) != identity["original"]:
        raise ValueError("the comparison mutated the original inputs")
    by_name = {c["variant"]: c for c in cases}
    main = [s for name in ("fa3_ab", "fa3_ba")
            for s in by_name[name]["summary"]["fa3_singletons"].values()]
    repetition = {name: all(s["all_outputs_bitwise_equal"]
        for s in by_name[f"{name}_repeat"]["summary"][name].values()) for name in ("fa3_ab", "fa3_ba")}
    isolation = {}
    for name in ("fa3_ab", "fa3_ba"):
        for changed in ("A", "B"):
            unchanged = "B" if changed == "A" else "A"
            variant = f"{name}_changed_{changed.lower()}"
            values = by_name[variant]["summary"][name]
            isolation[variant] = dict(unchanged_label=unchanged, changed_label=changed,
                unchanged_outputs_bitwise_equal=values[unchanged]["all_outputs_bitwise_equal"],
                changed_velocity_differs=not values[changed]["velocity"]["bitwise_equal"],
                interpretable=repetition[name])
    repeat_passed = all(repetition.values())
    isolation_passed = all(row["unchanged_outputs_bitwise_equal"] and row["changed_velocity_differs"]
                           for row in isolation.values())
    control_passed = all(s["velocity_screening_passed"] for c in cases if c["variant"].startswith("sdpa_")
                         for comparison in c["summary"].values() for s in comparison.values())
    return dict(cases=cases, input_identity=identity,
                packing_forward_passed=all(s["velocity_screening_passed"] for s in main),
                order_forward_passed=all(s["velocity_screening_passed"] for s in by_name["fa3_ba"]["summary"]["fa3_ab"].values()),
                repeatability_passed=repeat_passed, isolation_passed=isolation_passed,
                repetition_checks=repetition, isolation_checks=isolation,
                isolation_interpretable=repeat_passed, independent_controls_passed=control_passed,
                training_equivalence_established=False)


def run_worker(args):
    result = dict(status="running", phase="configuration", mode="forward_only", experiment_version=2, cases=[])
    def persist(phase=None):
        if phase:
            result["phase"] = phase
        probe.shapes._write_json(args.output_dir / "result.json", result)
        print(probe.PROGRESS_MARKER, result["phase"], flush=True)
    def on_case(row):
        if row["status"] == "complete":
            result["cases"].append(row)
        persist(f"{row['variant']}/{row['status']}")
    def on_inputs(identity):
        result["input_identity"] = identity
        persist("input_identity_verified")
    try:
        policy, training = probe.load_probe_config(args.config)
        policy = forward_policy(policy)
        plan = build_plan(policy, training)[0]
        tolerance = policy.numerical_tolerances["velocity_relative_l2"]
        result.update(plan=plan, velocity_relative_l2_tolerance=tolerance, grad_enabled=False,
                      backward_calls=0, optimizer_initialized=False, optimizer_updates=0,
                      block_counts=dict(double=5, single=20))
        persist("fa3_forward_preflight")
        device = probe.shapes.prepare_single_gpu_device(training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("forward image requires Hopper/SM90")
        flash = FlashBackend("fa3")
        result["backend_preflight"] = preflight(flash, device, tolerance)
        persist("load_models")
        seed_single_gpu_process(training.seed)
        models = load_single_gpu_models(training, device=device)
        model = models.transformer
        if (model.config.attention_head_dim != 128 or len(model.transformer_blocks) != 5
                or len(model.single_transformer_blocks) != 20 or model.gradient_checkpointing
                or models.flux_compute_dtype != torch.bfloat16
                or any(p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError("forward diagnostic requires original 5+20 FLUX blocks, head dim 128, FP32 weights and BF16 compute")
        result.update(environment=probe.shapes._environment_metadata(), flash_binding=flash.binding,
                      flash_version=probe.version("flash-attn-3"), flow_seed=training.seed,
                      parameter_dtype="torch.float32", compute_dtype=str(models.flux_compute_dtype))
        persist("prepare_real_inputs")
        stored = probe.prepare_inputs(policy, training, models)
        parent = plan["source_group"]
        full = probe.group_inputs(stored[parent["name"]], parent, training.seed, device)
        data = slice_inputs(full, 0, 2)
        if (list(data["layout"].text_lengths) != plan["text_lengths"]
                or data["layout"].cumulative.tolist() != plan["cumulative"]):
            raise ValueError("prepared example layout differs from the plan")
        result["input_sha256"] = data["input_sha256"]
        result.update(run_comparisons(model, data, flash, models.flux_compute_dtype, tolerance,
                                      profile=True, on_case=on_case, on_inputs=on_inputs))
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("forward diagnostic populated parameter gradients")
        result.update(status="complete", phase="complete", numerical_passed=all(result[k] for k in (
            "packing_forward_passed", "order_forward_passed", "repeatability_passed", "isolation_passed", "independent_controls_passed")))
        validate_complete(result)
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        persist()
    return result


def validate_complete(result):
    cases = result.get("cases", [])
    expected_stages = boundary_stage_order([*(f"transformer_blocks.{i}" for i in range(5)),
                                          *(f"single_transformer_blocks.{i}" for i in range(20))])
    if [c["variant"] for c in cases] != list(VARIANTS):
        raise ValueError("missing or reordered forward cases")
    for case in cases:
        labels, references = (list(v) for v in case_contract(case["variant"]))
        if (case["status"] != "complete" or list(case["outputs"]) != expected_stages
                or list(case["comparisons"]) != references or case["labels"] != labels):
            raise ValueError("incomplete forward observations")
        for comparison in case["comparisons"].values():
            if list(comparison) != labels or any(list(stages) != expected_stages for stages in comparison.values()):
                raise ValueError("incomplete per-example block comparisons")
        validate_forward_events(case["operators"], packed=case["variant"].startswith("fa3"), layers=25)
    if result.get("backward_calls") != 0 or result.get("optimizer_initialized") is not False:
        raise ValueError("forward-only execution contract violated")


def write_report(path, result):
    def percent(value):
        return f"{100 * value:.6f}%" if value is not None else "undefined (zero reference)"
    lines = ["# FLUX singleton versus packed forward", "", f"Execution: **{result['status']}**.",
             f"Numerical screening passed: **{result.get('numerical_passed', 'incomplete')}**.", "",
             "Plan: two fixed real targets from short_b4; twelve full model calls using no_grad and BF16 autocast with FP32 parameters.",
             "All 900 image tokens remain. There is no backward, optimizer initialization, update, or throughput measurement.",
             "The final velocity screening limit is fixed before execution; block metrics are descriptive, not causal evidence.", "",
             "| Actual | Reference | Example | Velocity relative L2 | Max absolute | First nonidentical capture |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for case in result.get("cases", []):
        for reference, samples in case.get("summary", {}).items():
            for label, row in samples.items():
                lines.append(f"| {case['variant']} | {reference} | {label} | {percent(row['velocity']['relative_l2'])} | "
                             f"{row['velocity']['max_abs']:.8g} | {row['first_nonidentical_capture'] or 'none'} |")
    lines += ["", f"Packing forward screening: **{result.get('packing_forward_passed', 'incomplete')}**.",
              f"Reordered BA versus AB screening: **{result.get('order_forward_passed', 'incomplete')}**.",
              f"Exact repetition: **{result.get('repeatability_passed', 'incomplete')}**.",
              f"Both isolation directions in both orders: **{result.get('isolation_passed', 'incomplete')}**.",
              f"Independent SDPA controls: **{result.get('independent_controls_passed', 'incomplete')}**.", "",
              "A failed repeat makes isolation inconclusive; it does not establish cross-example attention.",
              "Original SDPA singletons independently exercise model assembly but also change the attention backend.",
              "Per-example labels restore original identity before comparing AB and BA; the same perturbations are reused in both orders.",
              "The first differing capture is an observed boundary, not proof of a particular numerical cause.",
              "Forward agreement does not establish backward or training equivalence."]
    for case in result.get("cases", []):
        for reference, samples in case.get("comparisons", {}).items():
            lines += ["", f"## {case['variant']} versus {reference}", "",
                      "| Block | Example | Stream | Relative L2 | RMS | Max absolute | Bitwise equal |",
                      "| --- | --- | --- | ---: | ---: | ---: | --- |"]
            for label, stages in samples.items():
                for stage, streams in stages.items():
                    for stream, row in streams.items():
                        lines.append(f"| {stage} | {label} | {stream} | {percent(row['relative_l2'])} | "
                                     f"{row['rms']:.8g} | {row['max_abs']:.8g} | {row['bitwise_equal']} |")
    if result.get("error"):
        lines += ["", f"Failure during `{result.get('phase')}`: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")
