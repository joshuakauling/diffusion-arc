"""Isolate sequence segmentation with fixed post-RoPE Q/K/V and upstreams.

Flash singletons and pairs use the same public varlen binding. Math and cuDNN
execute one SDPA call per segment; they are correctness controls, not native
ragged throughput candidates. No model packing helper constructs the oracle.
"""

from contextlib import nullcontext
from functools import lru_cache
import inspect
from itertools import accumulate, combinations
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.diagnostics._flux_backend_replay import (
    BACKENDS, SDPA_OPERATORS, fingerprint, replay_precision,
    tensor_metrics, validate_profile,
)
from scripts.diagnostics._flux_packing import FlashBackend


FIELDS = ("output", "dQ", "dK", "dV")
SAMPLES = ("A", "B")


@lru_cache(maxsize=3)
def _flash(name):
    return FlashBackend(name)


def evaluation_plan(seeds, backends=BACKENDS):
    """Static upper bound; deterministic support is resolved from the binding."""
    production = len(backends) * (5 * len(seeds) + 3)
    deterministic = 5 * sum(name.startswith("fa") for name in backends)
    return dict(production_evaluations=production,
                deterministic_evaluations_maximum=deterministic,
                total_evaluations_maximum=production + deterministic,
                pairwise_comparisons=len(seeds) * len(backends) * (len(backends) - 1) // 2,
                samples_per_comparison=2, tensors_per_sample=4,
                production_cases_per_seed=["A", "B", "AB", "BA", "AB_repeat"],
                first_seed_extra_cases=["changed_A", "changed_B", "A_only_upstream"],
                deterministic_cases_first_seed=["A", "B", "AB", "BA", "AB_repeat"])


def attention_forward(name, q, k, v, lengths, *, deterministic=False):
    """Apply noncausal self-attention to flat [sum(S), H, D] inputs."""
    if name.endswith("math") or name == "cudnn":
        if deterministic:
            raise ValueError("SDPA controls do not expose a binding-level deterministic argument")
        selected = SDPBackend.CUDNN_ATTENTION if name == "cudnn" else SDPBackend.MATH
        with sdpa_kernel(selected):
            outputs = [F.scaled_dot_product_attention(
                a.transpose(0, 1).unsqueeze(0), b.transpose(0, 1).unsqueeze(0),
                c.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=False,
                scale=q.shape[-1] ** -0.5,
            ).squeeze(0).transpose(0, 1)
                for a, b, c in zip(q.split(lengths), k.split(lengths), v.split(lengths))]
        return torch.cat(outputs)
    binding = _flash(name)
    cumulative = torch.tensor([0, *accumulate(lengths)], device=q.device, dtype=torch.int32)
    kwargs = dict(cu_seqlens_q=cumulative, cu_seqlens_k=cumulative,
                  max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                  softmax_scale=q.shape[-1] ** -0.5, causal=False)
    # Omission preserves the pinned public binding's production default.
    if deterministic:
        kwargs["deterministic"] = True
    result = binding.kernel(q, k, v, **kwargs)
    return result[0] if name == "fa4" else result


def _profile_evidence(recorded, name, segments, cuda):
    operators = {event.key: event.count for event in recorded.key_averages()}
    kernels = sorted({event.name for event in recorded.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    evidence = _validate_dispatch(operators, kernels, name, segments=segments, cuda=cuda)
    return dict(**evidence, operators=operators, cuda_kernel_names=kernels,
                attention_calls=segments if name.endswith("math") or name == "cudnn" else 1,
                segment_count=segments)


def _validate_dispatch(operators, kernels, name, *, segments, cuda):
    normalized = dict(operators)
    normalized[f"flux_backend_replay::{name}"] = normalized.pop(f"flux_packing_replay::{name}", 0)
    if name.endswith("math") or name == "cudnn":
        counted = (*SDPA_OPERATORS, "aten::_scaled_dot_product_cudnn_attention_backward",
                   "aten::_softmax_backward_data")
        for key in counted:
            if normalized.get(key, 0):
                if normalized[key] % segments:
                    raise RuntimeError(f"{name}: unexpected segment operator count for {key}")
                normalized[key] //= segments
    return validate_profile(normalized, kernels, name, cuda=cuda)


def _evaluate(name, captures, upstreams, order, *, deterministic=False, profile=False):
    dtype = torch.float32 if name == "fp32_math" else torch.bfloat16
    lengths = tuple(captures[index][0].shape[1] for index in order)
    originals = tuple(torch.cat([captures[index][axis].squeeze(0) for index in order])
                      .to(dtype=dtype) for axis in range(3))
    leaves = tuple(value.detach().clone().requires_grad_() for value in originals)
    incoming_original = torch.cat([upstreams[index].squeeze(0) for index in order]).to(dtype=dtype)
    incoming = incoming_original.clone()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if leaves[0].is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities) if profile else nullcontext()
    with profiler as recorded:
        with torch.profiler.record_function(f"flux_packing_replay::{name}"):
            output = attention_forward(name, *leaves, lengths, deterministic=deterministic)
            if output.shape != leaves[0].shape or output.dtype != dtype:
                raise ValueError("attention returned an unexpected shape or dtype")
            gradients = torch.autograd.grad(output, leaves, grad_outputs=incoming)
    for actual, expected in zip(leaves, originals):
        if not torch.equal(actual.detach().view(torch.uint8), expected.view(torch.uint8)):
            raise RuntimeError("attention mutated a Q/K/V input")
    if not torch.equal(incoming.view(torch.uint8), incoming_original.view(torch.uint8)):
        raise RuntimeError("backward mutated the upstream gradient")
    tensors = (output.detach(), *(gradient.detach() for gradient in gradients))
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("non-finite attention output or input gradient")
    values = {SAMPLES[index]: tuple(value for value in group)
              for index, group in zip(order, zip(*(tensor.split(lengths) for tensor in tensors)))}
    evidence = dict(profiled=profile, backend_verified=False, evidence_level="not_profiled",
                    binding=(f"{FlashBackend.MODULES[name]}.flash_attn_varlen_func"
                             if name.startswith("fa") else
                             f"torch.nn.functional.scaled_dot_product_attention/{name}"),
                    layout="native_varlen" if name.startswith("fa") else "segmented_sdpa_control",
                    deterministic_argument=True if deterministic else "binding_default",
                    inputs_unchanged=True, upstream_unchanged=True)
    if profile:
        evidence.update(_profile_evidence(recorded, name, len(order), leaves[0].is_cuda))
    return values, evidence


def _compare(actual, reference):
    return {sample: {field: tensor_metrics(a, b)
                     for field, a, b in zip(FIELDS, actual[sample], reference[sample])}
            for sample in actual}


def _exact(metrics):
    return all(row["max_abs"] == 0 for sample in metrics.values() for row in sample.values())


def _validate(captures, seeds, backends):
    if len(captures) != 2 or any(len(sample) != 3 for sample in captures):
        raise ValueError("replay requires Q/K/V for exactly two samples")
    q = captures[0][0]
    for sample in captures:
        first = sample[0]
        if (first.ndim != 4 or first.shape[0] != 1 or not all(first.shape)
                or first.shape[2:] != q.shape[2:]
                or any(value.shape != first.shape or value.device != q.device
                       or value.dtype != torch.bfloat16 or not value.is_contiguous()
                       for value in sample)):
            raise ValueError("replay requires same-device contiguous BF16 Q/K/V [1, S, H, D]")
        if any(not torch.isfinite(value).all() for value in sample):
            raise FloatingPointError("non-finite replay inputs")
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2 ** 63 for seed in seeds)):
        raise ValueError("seeds must be unique nonnegative 63-bit integers")
    if (not backends or backends[0] != "fp32_math" or len(set(backends)) != len(backends)
            or any(name not in BACKENDS for name in backends)
            or len(backends) > 1 and backends[1] != "bf16_math"):
        raise ValueError("unique backends must start with fp32_math, followed by bf16_math")
    if q.device.type != "cuda" and any(name not in ("fp32_math", "bf16_math") for name in backends):
        raise ValueError("cuDNN and FlashAttention require CUDA")


def replay_pair(captures, *, seeds, profile=False, backends=BACKENDS, on_result=None):
    """Return fixed-input segmentation evidence, saving JSON-safe partial rows.

    Each seed uses A/B singleton references, AB, BA and an unchanged AB repeat.
    The first seed additionally tests both perturbations and disconnected B
    input gradients. Supported Flash deterministic modes repeat the five-case
    comparison once. Failed backends retain their error while independent
    cases continue; numerical differences themselves have no pass threshold.
    """
    seeds, backends = tuple(seeds), tuple(backends)
    _validate(captures, seeds, backends)
    identities = {sample: {field: fingerprint(tensor)
                          for field, tensor in zip(("q", "k", "v"), values)}
                  for sample, values in zip(SAMPLES, captures)}
    result = dict(status="running", plan=evaluation_plan(seeds, backends),
                  evaluations_attempted=0, evaluations_completed=0,
                  segment_lengths=[sample[0].shape[1] for sample in captures],
                  inputs=identities, seeds=list(seeds), input_hashes_unchanged=None,
                  causal=False, dropout=0.0, softmax_scale=captures[0][0].shape[-1] ** -0.5,
                  tf32=False, autocast=False, math_fp16_bf16_reduction=False,
                  model_gradients_measured=False, training_qualified=False,
                  upstream_source="sample-associated seeded BF16 Gaussian, promoted unchanged for FP32",
                  rows=[], pairwise=[], deterministic=[], errors=[])

    def publish():
        if on_result is not None:
            on_result(result)

    def evaluate(name, incoming, order, *, inputs=captures, deterministic=False, observed=False):
        result["evaluations_attempted"] += 1
        values, evidence = _evaluate(name, inputs, incoming, order,
                                     deterministic=deterministic, profile=observed)
        result["evaluations_completed"] += 1
        return values, evidence

    def five_cases(name, incoming, deterministic=False, observed=False):
        singleton = {}
        for index in range(2):
            values, _ = evaluate(name, incoming, (index,), deterministic=deterministic)
            singleton.update(values)
        ab, evidence = evaluate(name, incoming, (0, 1), deterministic=deterministic, observed=observed)
        ba, _ = evaluate(name, incoming, (1, 0), deterministic=deterministic)
        repeated, _ = evaluate(name, incoming, (0, 1), deterministic=deterministic)
        metrics = dict(packed_vs_singletons=_compare(ab, singleton),
                       reversed_vs_singletons=_compare(ba, singleton),
                       reversed_vs_packed=_compare(ba, ab),
                       repeat_vs_packed=_compare(repeated, ab))
        return singleton, ab, dict(**metrics, dispatch=evidence,
                                  repeat_exact=_exact(metrics["repeat_vs_packed"]))

    with replay_precision(captures[0][0].device):
        for seed_index, seed in enumerate(seeds):
            generator = torch.Generator(device=captures[0][0].device).manual_seed(seed)
            incoming = tuple(torch.randn(sample[0].shape, device=sample[0].device,
                                         dtype=torch.bfloat16, generator=generator) for sample in captures)
            upstream_identities = {sample: fingerprint(value) for sample, value in zip(SAMPLES, incoming)}
            # At most one seed's singleton values are retained for direct pairs.
            baselines = {}
            for name in backends:
                row = dict(backend=name, seed=seed, status="running",
                           upstreams=upstream_identities)
                result["rows"].append(row)
                attempted_before, completed_before = result["evaluations_attempted"], result["evaluations_completed"]
                try:
                    singleton, ab, measured = five_cases(name, incoming, observed=profile and seed_index == 0)
                    row.update(measured)
                    baselines[name] = singleton
                    if seed_index == 0:
                        isolation = {}
                        row["isolation"] = isolation
                        for changed in range(2):
                            modified = tuple(tuple((-value + .125).contiguous() for value in sample)
                                             if index == changed else sample
                                             for index, sample in enumerate(captures))
                            perturbed, _ = evaluate(name, incoming, (0, 1), inputs=modified)
                            unchanged = SAMPLES[1 - changed]
                            metrics = _compare({unchanged: perturbed[unchanged]}, {unchanged: ab[unchanged]})
                            output_exact = metrics[unchanged]["output"]["max_abs"] == 0
                            isolation[f"changed_{SAMPLES[changed]}"] = dict(
                                unchanged_sample=unchanged, comparisons=metrics, output_exact=output_exact,
                                all_tensors_exact=_exact(metrics),
                                gradient_variation_with_exact_repeat=any(
                                    metrics[unchanged][field]["max_abs"] != 0
                                    and row["repeat_vs_packed"][unchanged][field]["max_abs"] == 0
                                    for field in FIELDS[1:]),
                                input_gradient_comparison="interpret alongside same-shape backward repeatability")
                            if not output_exact and row["repeat_vs_packed"][unchanged]["output"]["max_abs"] == 0:
                                raise RuntimeError(f"changing {SAMPLES[changed]} affected {unchanged} output")
                        disconnected, _ = evaluate(name, (incoming[0], torch.zeros_like(incoming[1])), (0, 1))
                        zero = {field: dict(exact_zero=bool(torch.count_nonzero(value) == 0),
                                            max_abs=value.abs().max().item())
                                for field, value in zip(FIELDS[1:], disconnected["B"][1:])}
                        isolation["A_only_upstream"] = dict(disconnected_sample="B", gradients=zero,
                                                            exact_zero=all(item["exact_zero"] for item in zero.values()))
                        if not isolation["A_only_upstream"]["exact_zero"]:
                            raise RuntimeError("A-only upstream produced a nonzero B input gradient")
                    row["status"] = "complete"
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                    baselines.pop(name, None)
                    result["errors"].append(dict(backend=name, seed=seed, error=row["error"]))
                row["evaluations_attempted"] = result["evaluations_attempted"] - attempted_before
                row["evaluations_completed"] = result["evaluations_completed"] - completed_before
                publish()

                if seed_index == 0:
                    diagnostic = dict(backend=name, seed=seed)
                    result["deterministic"].append(diagnostic)
                    attempted_before, completed_before = result["evaluations_attempted"], result["evaluations_completed"]
                    if not name.startswith("fa"):
                        diagnostic.update(status="not_applicable", reason="SDPA has no binding-level deterministic argument")
                    elif row["status"] != "complete":
                        diagnostic.update(status="skipped", reason="production case failed")
                    else:
                        try:
                            signature = inspect.signature(_flash(name).kernel)
                            diagnostic["binding_signature"] = str(signature)
                            if "deterministic" not in signature.parameters:
                                diagnostic.update(status="unsupported", reason="public binding has no deterministic parameter")
                            else:
                                _, _, measured = five_cases(name, incoming, deterministic=True, observed=profile)
                                diagnostic.update(status="complete", **measured)
                        except Exception as error:
                            reason = f"{type(error).__name__}: {error}"
                            unsupported = "deterministic" in str(error).lower() and (
                                isinstance(error, NotImplementedError) or "not supported" in str(error).lower()
                                or "unsupported" in str(error).lower())
                            diagnostic.update(status="unsupported" if unsupported else "failed", reason=reason)
                            if not unsupported:
                                result["errors"].append(dict(backend=name, mode="deterministic", error=reason))
                    diagnostic["evaluations_attempted"] = result["evaluations_attempted"] - attempted_before
                    diagnostic["evaluations_completed"] = result["evaluations_completed"] - completed_before
                    publish()

            for reference, actual in combinations(backends, 2):
                if reference in baselines and actual in baselines:
                    result["pairwise"].append(dict(seed=seed, actual=actual, reference=reference,
                                                   layout="independent_singletons",
                                                   comparisons=_compare(baselines[actual], baselines[reference])))
            del baselines
            publish()

    final_identities = {sample: {field: fingerprint(tensor)
                                for field, tensor in zip(("q", "k", "v"), values)}
                        for sample, values in zip(SAMPLES, captures)}
    result["input_hashes_unchanged"] = identities == final_identities
    if not result["input_hashes_unchanged"]:
        result["errors"].append(dict(error="replay modified caller Q/K/V tensors"))
    result["status"] = "complete" if not result["errors"] else "completed_with_failures"
    publish()
    return result


def validate_replay_result(result, *, seeds, profile=False, backends=BACKENDS):
    """Reject incomplete artifacts; return false for recorded execution failures.

    This validates evidence and case accounting, never a numerical tolerance or
    training qualification. Unsupported deterministic modes remain explicit.
    """
    seeds, backends = tuple(seeds), tuple(backends)

    def require(condition, message):
        if not condition:
            raise ValueError(f"invalid packing replay result: {message}")

    def integer(value):
        return type(value) is int and value >= 0

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def identity(value, *, shape=None):
        require(isinstance(value, dict), "missing tensor fingerprint")
        actual_shape = value.get("shape")
        require(isinstance(actual_shape, list) and len(actual_shape) == 4
                and all(type(n) is int and n > 0 for n in actual_shape), "invalid tensor shape")
        require(shape is None or actual_shape == shape, "fingerprint shape mismatch")
        digest = value.get("sha256", "")
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "invalid tensor fingerprint hash")
        require(value.get("dtype") == "torch.bfloat16", "input/upstream fingerprint must be BF16")
        return actual_shape

    require(isinstance(result, dict), "result must be a mapping")
    require(result.get("status") in ("complete", "completed_with_failures"), "execution is incomplete")
    require(result.get("seeds") == list(seeds), "seed schedule mismatch")
    require(result.get("plan") == evaluation_plan(seeds, backends), "static evaluation plan mismatch")
    require(result.get("causal") is False and result.get("dropout") == 0
            and result.get("autocast") is False and result.get("tf32") is False
            and result.get("math_fp16_bf16_reduction") is False,
            "attention or precision contract mismatch")
    require(result.get("training_qualified") is False and result.get("model_gradients_measured") is False,
            "isolated replay mislabelled as model/training validation")
    require(result.get("input_hashes_unchanged") is True, "caller inputs not verified unchanged")
    inputs = result.get("inputs", {})
    require(set(inputs) == set(SAMPLES), "missing sample input identities")
    shapes = {}
    for sample in SAMPLES:
        require(set(inputs[sample]) == {"q", "k", "v"}, "missing Q/K/V identity")
        shapes[sample] = identity(inputs[sample]["q"])
        require(shapes[sample][0] == 1, "input sample is not an unpadded singleton")
        for field in ("k", "v"):
            identity(inputs[sample][field], shape=shapes[sample])
    require(shapes["A"][2:] == shapes["B"][2:], "head shape mismatch")
    require(result.get("segment_lengths") == [shapes[sample][1] for sample in SAMPLES], "segment length mismatch")
    require(result.get("softmax_scale") == shapes["A"][-1] ** -.5, "softmax scale mismatch")

    def metrics(value, samples=SAMPLES):
        require(isinstance(value, dict) and set(value) == set(samples), "missing sample metrics")
        for sample in samples:
            require(isinstance(value[sample], dict) and set(value[sample]) == set(FIELDS), "missing tensor metrics")
            for field in FIELDS:
                row = value[sample][field]
                require(isinstance(row, dict), "tensor metric must be a mapping")
                for key in ("rms", "max_abs", "reference_l2", "actual_l2", "reference_rms", "actual_rms"):
                    require(finite(row.get(key)) and row[key] >= 0, f"missing/non-finite {key}")
                require("relative_l2" in row and (row["relative_l2"] is None and row["reference_l2"] == 0
                        or finite(row["relative_l2"]) and row["relative_l2"] >= 0),
                        "missing/non-finite relative L2")
                require("cosine" in row and (row["cosine"] is None
                        and (row["reference_l2"] == 0 or row["actual_l2"] == 0)
                        or finite(row["cosine"]) and -1 <= row["cosine"] <= 1), "invalid cosine")
                require(type(row.get("count")) is int and row["count"] == math.prod(shapes[sample]),
                        "metric element count mismatch")
                require(row.get("actual_dtype") in ("torch.float32", "torch.bfloat16")
                        and row.get("reference_dtype") in ("torch.float32", "torch.bfloat16"), "missing metric dtype")

    def five(row, *, expected_profile):
        for key in ("packed_vs_singletons", "reversed_vs_singletons", "reversed_vs_packed", "repeat_vs_packed"):
            metrics(row.get(key))
        require(row.get("repeat_exact") is _exact(row["repeat_vs_packed"]), "repeat exactness mismatch")
        dispatch = row.get("dispatch", {})
        name = row["backend"]
        require(dispatch.get("inputs_unchanged") is True and dispatch.get("upstream_unchanged") is True,
                "missing backend mutation checks")
        require(dispatch.get("layout") == ("native_varlen" if name.startswith("fa") else "segmented_sdpa_control"),
                "backend layout mislabelled")
        require(dispatch.get("profiled") is bool(expected_profile), "profile schedule mismatch")
        if expected_profile:
            require(dispatch.get("backend_verified") is True, "backend dispatch not verified")
            try:
                observed = _validate_dispatch(dispatch["operators"], dispatch["cuda_kernel_names"], name,
                                              segments=2, cuda=dispatch["cuda_kernel_capture_requested"])
            except (KeyError, TypeError, RuntimeError) as error:
                raise ValueError(f"invalid packing replay result: missing forward/backward dispatch: {error}") from error
            require(all(dispatch.get(key) == value for key, value in observed.items()), "dispatch evidence mismatch")

    rows = result.get("rows", [])
    expected_keys = {(name, seed) for name in backends for seed in seeds}
    keys = [(row.get("backend"), row.get("seed")) for row in rows]
    require(len(keys) == len(expected_keys) and set(keys) == expected_keys, "missing/duplicate backend-seed case")
    indexed = dict(zip(keys, rows))
    attempted = completed = failures = 0
    for row in rows:
        require(row.get("status") in ("complete", "failed"), "backend case is incomplete")
        require(set(row.get("upstreams", {})) == set(SAMPLES), "missing upstream identities")
        for sample in SAMPLES:
            identity(row["upstreams"][sample], shape=shapes[sample])
        require(row["upstreams"] == indexed[(backends[0], row["seed"])]["upstreams"],
                "upstream differs between backends")
        expected = 8 if row["seed"] == seeds[0] else 5
        count, done = row.get("evaluations_attempted"), row.get("evaluations_completed")
        require(integer(count) and integer(done) and 0 <= done <= count <= expected, "invalid row evaluation count")
        attempted += count
        completed += done
        if row["status"] == "failed":
            require(isinstance(row.get("error"), str) and row["error"], "failed case lacks reason")
            failures += 1
            continue
        require(count == done == expected, "successful case has missing evaluations")
        five(row, expected_profile=profile and row["seed"] == seeds[0])
        if row["seed"] == seeds[0]:
            isolation = row.get("isolation", {})
            require(set(isolation) == {"changed_A", "changed_B", "A_only_upstream"}, "missing isolation cases")
            for changed, unchanged in (("A", "B"), ("B", "A")):
                observed = isolation[f"changed_{changed}"]
                require(observed.get("unchanged_sample") == unchanged, "wrong perturbation ownership")
                metrics(observed.get("comparisons"), (unchanged,))
                require(observed.get("output_exact") is (observed["comparisons"][unchanged]["output"]["max_abs"] == 0)
                        and observed.get("all_tensors_exact") is _exact(observed["comparisons"]), "isolation exactness mismatch")
            disconnected = isolation["A_only_upstream"]
            require(disconnected.get("disconnected_sample") == "B" and disconnected.get("exact_zero") is True,
                    "disconnected gradient check failed")
            require(set(disconnected.get("gradients", {})) == {"dQ", "dK", "dV"}, "missing disconnected gradients")
            for value in disconnected["gradients"].values():
                require(value.get("exact_zero") is True and value.get("max_abs") == 0, "nonzero disconnected gradient")

    deterministic = result.get("deterministic", [])
    require(len(deterministic) == len(backends) and {row.get("backend") for row in deterministic} == set(backends),
            "missing/duplicate deterministic capability case")
    for row in deterministic:
        require(row.get("seed") == seeds[0], "deterministic seed mismatch")
        name, status = row["backend"], row.get("status")
        count, done = row.get("evaluations_attempted"), row.get("evaluations_completed")
        require(integer(count) and integer(done) and 0 <= done <= count <= 5, "invalid deterministic evaluation count")
        attempted += count
        completed += done
        if not name.startswith("fa"):
            require(status == "not_applicable" and count == 0, "SDPA deterministic capability mislabelled")
        elif indexed[(name, seeds[0])]["status"] == "failed":
            require(status == "skipped" and count == 0, "failed production path was not skipped")
        elif status == "complete":
            require(count == done == 5, "missing deterministic evaluations")
            require("deterministic" in row.get("binding_signature", ""), "deterministic binding not identified")
            five(row, expected_profile=profile)
            require(row["dispatch"].get("deterministic_argument") is True, "deterministic argument not used")
        else:
            require(status in ("unsupported", "failed") and isinstance(row.get("reason"), str) and row["reason"],
                    "missing deterministic support/error reason")
            failures += int(status == "failed")

    pairs = result.get("pairwise", [])
    expected_pairs = {(seed, actual, reference) for seed in seeds
                      for reference, actual in combinations(backends, 2)
                      if indexed[(actual, seed)]["status"] == indexed[(reference, seed)]["status"] == "complete"}
    pair_keys = [(row.get("seed"), row.get("actual"), row.get("reference")) for row in pairs]
    require(len(pair_keys) == len(expected_pairs) and set(pair_keys) == expected_pairs, "missing/duplicate direct pairwise case")
    for row in pairs:
        require(row.get("layout") == "independent_singletons", "pairwise reference layout mismatch")
        metrics(row.get("comparisons"))
    require(result.get("evaluations_attempted") == attempted and result.get("evaluations_completed") == completed,
            "evaluation totals mismatch")
    require(attempted <= result["plan"]["total_evaluations_maximum"], "evaluation budget exceeded")
    require(isinstance(result.get("errors"), list) and len(result["errors"]) == failures, "error accounting mismatch")
    require(result["status"] == ("completed_with_failures" if failures else "complete"), "completion status mismatch")
    return not failures
