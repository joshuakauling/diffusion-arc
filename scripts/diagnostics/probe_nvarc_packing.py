#!/usr/bin/env python3
"""Compare frozen NVARC with FA2/FA3/FA4 packing, alignment, and cuDNN ragged.

Only NVARC is loaded. No training, downloads, or optimizer updates are performed.
Each observation processes a complete logical source window, including input
transfer, padding/packing and compact hidden-state extraction. The packed path
is diagnostic code; the production runner and schedulers remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
from itertools import accumulate
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.model.contracts import NVARC_HIDDEN_STATE_INDICES, NVARC_MAX_TOKENS, NVARC_PAD_TOKEN_ID
from diffusion_arc._constants import NVARC_TOKENIZER_MAX_TOKENS
from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner
from diffusion_arc.training.nvarc_schedule import schedule_nvarc_sources
from diffusion_arc.training.dataset import ArcTrainingDataset
from diffusion_arc.training.flux_input import _select_eligible_sequence
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import _validate_nvarc_checkpoint
from scripts.diagnostics.probe_nvarc_batch_capacity import (
    _environment_metadata, _memory, _write_json,
)
from scripts.diagnostics._nvarc_attention_backends import BACKENDS, PackedAttention

PROGRESS_MARKER = "NVARC_PACKING_PROGRESS"


@dataclass(frozen=True)
class Window:
    name: str
    optimizer_step: int
    source_lengths: tuple[int, ...]
    source_keys: tuple[str, ...]


@dataclass(frozen=True)
class ProbeConfig:
    training_config: Path
    warmup_calls: int
    measured_calls: int
    max_duration_seconds: int
    length_source: dict
    windows: tuple[Window, ...]
    backends: tuple[str, ...]
    alignments: tuple[int, ...]


def load_probe_config(path: Path, *, require_checkpoint: bool = True):
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"schema_version", *ProbeConfig.__dataclass_fields__}:
        raise ValueError("unexpected NVARC packing config fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise ValueError("schema_version must be 2")
    for name in ("warmup_calls", "measured_calls", "max_duration_seconds"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(raw["training_config"], str) or not raw["training_config"]:
        raise ValueError("training_config must name a file")
    if not isinstance(raw["length_source"], dict) or not raw["length_source"]:
        raise ValueError("length_source must describe the source-length provenance")
    windows = []
    if not isinstance(raw["windows"], list) or not raw["windows"]:
        raise ValueError("windows must be nonempty")
    for row in raw["windows"]:
        if not isinstance(row, dict) or set(row) != {"name", "optimizer_step", "source_lengths", "source_keys"}:
            raise ValueError("unexpected window fields")
        if not isinstance(row["name"], str) or not row["name"].isidentifier():
            raise ValueError("window name must be an identifier")
        if type(row["optimizer_step"]) is not int or row["optimizer_step"] < 1:
            raise ValueError("optimizer_step must be positive")
        lengths = row["source_lengths"]
        if (not isinstance(lengths, list) or len(lengths) != 16
                or any(type(n) is not int or not 1 <= n <= NVARC_MAX_TOKENS for n in lengths)):
            raise ValueError("each window must contain 16 source lengths in [1, 8400]")
        keys = row["source_keys"]
        if not isinstance(keys, list) or len(keys) != 16 or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("each window must record its 16 original augmentation keys")
        windows.append(Window(row["name"], row["optimizer_step"], tuple(lengths), tuple(keys)))
    if len({w.name for w in windows}) != len(windows):
        raise ValueError("window names must be unique")
    backends, alignments = raw["backends"], raw["alignments"]
    if (not isinstance(backends, list) or not backends
            or any(b not in BACKENDS for b in backends) or len(set(backends)) != len(backends)):
        raise ValueError("backends must be distinct names from fa2, fa3, fa4, cudnn")
    if (not isinstance(alignments, list) or not alignments
            or any(type(a) is not int or a not in (8, 32) for a in alignments)
            or alignments != sorted(set(alignments))):
        raise ValueError("alignments must be a sorted subset of [8, 32]")
    policy = ProbeConfig(
        training_config=(path.resolve().parent / raw["training_config"]).resolve(),
        windows=tuple(windows), length_source=raw["length_source"],
        backends=tuple(backends), alignments=tuple(alignments),
        **{name: raw[name] for name in ("warmup_calls", "measured_calls", "max_duration_seconds")},
    )
    training = load_single_gpu_training_config(policy.training_config)
    if (training.nvarc_scheduling is None or training.loop.nvarc_source_batch_size != 16
            or training.models.nvarc_dtype != "bfloat16" or training.data.shuffle):
        raise ValueError("probe requires the BF16 B2 NVARC scheduler and 16-source windows")
    if require_checkpoint and not training.models.nvarc_checkpoint.is_dir():
        raise FileNotFoundError(training.models.nvarc_checkpoint)
    return policy, training


def packing_layout(lengths, mode="none", alignment=1):
    """Describe physical segments separately from real output lengths."""
    rounded = lambda n: (n + alignment - 1) // alignment * alignment
    if mode == "source":
        segments = tuple(rounded(n) for n in lengths)
    elif mode in ("none", "total"):
        tail = rounded(sum(lengths)) - sum(lengths) if mode == "total" else 0
        segments = tuple(lengths) + ((tail,) if tail else ())
    else:
        raise ValueError(f"unknown packing mode: {mode}")
    offsets = (0, *accumulate(segments))
    return {"segment_lengths": segments, "offsets": offsets,
            "total_tokens": sum(segments), "padding_tokens": sum(segments) - sum(lengths)}


def build_plan(policy, training):
    plan = []
    for index, window in enumerate(policy.windows):
        batches = schedule_nvarc_sources(window.source_lengths, training.nvarc_scheduling)
        variants = [("none", 1), *[(mode, a) for mode in ("total", "source") for a in policy.alignments]]
        cases, seen = [], {}
        for backend in policy.backends:
            for mode, alignment in variants[:1] if backend == "cudnn" else variants:
                layout = packing_layout(window.source_lengths, mode, alignment)
                # Equal total sizes alone are NOT equal layouts: sequence
                # boundaries and real-output locations must also agree.
                identity = (backend, layout["segment_lengths"])
                name = f"{backend}_{mode}" + (f"_{alignment}" if alignment != 1 else "")
                if identity in seen:
                    seen[identity]["aliases"].append(name)
                else:
                    case = {"variant": name, "backend": backend, "mode": mode,
                            "alignment": alignment, "layout": layout, "aliases": []}
                    cases.append(case)
                    seen[identity] = case
        backend_order = policy.backends[index % len(policy.backends):] + policy.backends[:index % len(policy.backends)]
        ordered = []
        for backend in backend_order:
            group = [c for c in cases if c["backend"] == backend]
            shift = index % len(group)
            ordered.extend(group[shift:] + group[:shift])
        plan.append({
            **asdict(window), "real_tokens": sum(window.source_lengths),
            "b2_padded_tokens": sum(len(b.source_indices) * b.padded_length for b in batches),
            "b2_batches": [asdict(b) for b in batches], "cases": ordered,
            "packed_calls": 1, "packed_tokens": sum(window.source_lengths),
        })
    return plan


def real_sources(dataset, window, *, min_context_pairs):
    """Replay the original logical window; reject key/length drift before timing."""
    start = (window.optimizer_step - 1) * 16
    keys = tuple(str(key) for key in dataset.keys[start:start + 16])
    if keys != window.source_keys:
        raise ValueError(f"{window.name}: benchmark source keys differ from the selected dataset")
    sources = []
    for index, length in enumerate(window.source_lengths, start):
        eligible = _select_eligible_sequence(dataset[index], min_context_pairs_before_target=min_context_pairs)
        if eligible is None or len(eligible.input_ids) != length:
            raise ValueError(f"{window.name}: consumed prefix length differs at dataset index {index}")
        sources.append(torch.tensor(eligible.input_ids, dtype=torch.int64))
    return tuple(sources)


def packed_arguments(sources, device, *, layout=None, pad_token_id=NVARC_PAD_TOKEN_ID, cudnn_heads=None):
    """Boundary metadata is computed on CPU, avoiding per-layer scalar GPU reads."""
    lengths = [len(source) for source in sources]
    layout = packing_layout(lengths) if layout is None else layout
    offsets = list(layout["offsets"])
    segments = layout["segment_lengths"]
    ids = torch.full((layout["total_tokens"],), pad_token_id, dtype=torch.int64)
    for start, source in zip(offsets, sources):
        ids[start:start + len(source)] = source
    cumulative = torch.tensor(offsets, dtype=torch.int32, device=device)
    args = {
        "input_ids": ids.unsqueeze(0).to(device),
        "attention_mask": None,
        "position_ids": torch.cat([torch.arange(n) for n in segments]).unsqueeze(0).to(device),
        "cu_seq_lens_q": cumulative, "cu_seq_lens_k": cumulative,
        "max_length_q": max(segments), "max_length_k": max(segments),
    }
    if cudnn_heads is not None:
        q_heads, kv_heads, dim = cudnn_heads
        def element_offsets(heads):
            return torch.tensor([n * heads * dim for n in offsets], dtype=torch.int64,
                                device=device).reshape(-1, 1, 1, 1)
        args["nvarc_cudnn_metadata"] = {
            "batch": len(segments), "q_offsets": element_offsets(q_heads),
            "kv_offsets": element_offsets(kv_heads),
            "lengths": torch.tensor(segments, dtype=torch.int32, device=device).reshape(-1, 1, 1, 1),
        }
    return args, offsets


def run_packed(model, sources, *, layout=None, pad_token_id=NVARC_PAD_TOKEN_ID):
    config = model.config
    heads = (config.num_attention_heads, config.num_key_value_heads, config.head_dim) if (
        config._attn_implementation == "nvarc_probe_cudnn") else None
    args, offsets = packed_arguments(
        sources, model.get_input_embeddings().weight.device, layout=layout,
        pad_token_id=pad_token_id,
        cudnn_heads=heads,
    )
    with torch.no_grad():
        output = model(**args, output_hidden_states=True, use_cache=False, return_dict=True)
    return tuple(
        torch.cat([output.hidden_states[layer][0, start:start + len(source)]
                   for layer in NVARC_HIDDEN_STATE_INDICES], dim=-1)
        for start, source in zip(offsets, sources)
    )


def run_grouped(runner, sources, batches):
    """Execute the real B2 plan and restore logical source order."""
    states = [None] * len(sources)
    for batch in batches:
        selected = [sources[index] for index in batch.source_indices]
        ids = torch.cat(selected)
        offsets = torch.tensor([0, *accumulate(map(len, selected))], dtype=torch.int64)
        output = runner(ids, offsets, padded_length=batch.padded_length)
        for index, state in zip(batch.source_indices, output.sequence_hidden_states):
            states[index] = state
    return tuple(states)


def collect_outputs(call, sources, model):
    """Check the interface once per case; CPU copies are outside timing."""
    states = call()
    if len(states) != len(sources):
        raise ValueError("incorrect number of source outputs")
    expected_width = model.config.hidden_size * len(NVARC_HIDDEN_STATE_INDICES)
    for source, state in zip(sources, states):
        if (state.shape != (len(source), expected_width) or state.requires_grad
                or state.dtype != model.get_input_embeddings().weight.dtype
                or not bool(torch.isfinite(state).all())):
            raise ValueError("invalid frozen NVARC hidden states")
    return tuple(state.cpu().clone() for state in states)


def numerical_difference(actual, reference):
    maximum = squared_error = squared_reference = 0.0
    count = 0
    if len(actual) != len(reference):
        raise ValueError("source output counts differ")
    for left, right in zip(actual, reference):
        if left.shape != right.shape:
            raise ValueError("source output shapes differ")
        # Bound temporary FP32/FP64 allocations when comparing long windows.
        for a, b in zip(left.split(256), right.split(256)):
            delta = a.float() - b.float()
            maximum = max(maximum, delta.abs().max().item())
            squared_error += delta.double().square().sum().item()
            squared_reference += b.double().square().sum().item()
            count += delta.numel()
    return {
        "max_abs": maximum, "rms": (squared_error / count) ** 0.5,
        "relative_l2": (squared_error / squared_reference) ** 0.5 if squared_reference else None,
        "agreement_threshold_applied": False,
    }


def select_backend(model, name):
    model.set_attn_implementation(name)
    if model.config._attn_implementation != name:
        raise RuntimeError(f"model did not select {name}")


def measure_window(call, device, events):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    events[0].record()
    output = call()
    events[1].record()
    events[1].synchronize()
    row = {
        "wall_ms": (time.perf_counter() - started) * 1000,
        "cuda_span_ms": events[0].elapsed_time(events[1]), **_memory(device),
    }
    del output
    return row


def validate_backend_events(counts, backend, *, layers):
    """Reject an unintended attention fallback instead of timing it as a success."""
    if counts.get(f"nvarc_probe::{backend}", 0) != layers:
        raise RuntimeError(f"{backend}: expected {layers} explicit backend calls: {counts}")
    operators = {"fa2": "flash_attn::_flash_attn_varlen_forward",
                 "fa3": "flash_attn_3::_flash_attn_forward"}
    if backend in operators and counts.get(operators[backend], 0) != layers:
        raise RuntimeError(f"{backend}: expected {layers} official CUDA-extension calls: {counts}")
    for name, count in counts.items():
        if count and ("scaled_dot_product" in name or any(
            name == op and backend != other for other, op in operators.items()
        )):
            raise RuntimeError(f"{backend}: unintended attention backend: {name}")


def inspect_backend(call, adapter, *, layers, device):
    torch.cuda.synchronize(device)
    adapter.observe = True
    try:
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as profiler:
            output = call()
            torch.cuda.synchronize(device)
    finally:
        adapter.observe = False
    del output
    counts = {event.key: event.count for event in profiler.key_averages()
              if any(name in event.key for name in ("scaled_dot_product", "flash_attn", "nvarc_probe::"))}
    validate_backend_events(counts, adapter.backend, layers=layers)
    kernels = sorted({event.name for event in profiler.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    if not kernels:
        raise RuntimeError("backend verification recorded no CUDA execution")
    return {"binding": adapter.binding, "operators": counts, "cuda_kernel_names": kernels}


def summarize(samples, *, real_tokens):
    median = statistics.median(s["wall_ms"] for s in samples)
    return {
        "median_wall_ms": median,
        "min_wall_ms": min(s["wall_ms"] for s in samples),
        "max_wall_ms": max(s["wall_ms"] for s in samples),
        "median_cuda_span_ms": statistics.median(s["cuda_span_ms"] for s in samples),
        "tokens_per_second": real_tokens / (median / 1000),
        **{key: max(s[key] for s in samples)
           for key in ("peak_allocated_bytes", "peak_reserved_bytes")},
    }


def load_runtime(training):
    """Load and validate only the real NVARC checkpoint; never download weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("probe requires a CUDA GPU with BF16 support")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(training.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        training.models.nvarc_checkpoint, local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        training.models.nvarc_checkpoint, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa",
    )
    _validate_nvarc_checkpoint(model, tokenizer, expected_device=device, expected_dtype=torch.bfloat16)
    if model.config.head_dim != 128:
        raise ValueError(f"diagnostic kernels require head_dim=128, received {model.config.head_dim}")
    # The checkpoint leaves config.pad_token_id unset; preflight validates
    # tokenizer PAD against NVARC_PAD_TOKEN_ID, also used by the production runner.
    tokenizer.model_max_length = NVARC_TOKENIZER_MAX_TOKENS
    runner = NvarcBatchRunner(model)
    return device, model, runner, tokenizer


def check_isolation(model, sources):
    """Perturb one source and trailing PAD values at fixed shapes, outside timing."""
    small = (sources[0][:31], sources[1][:47])
    layout = packing_layout(tuple(map(len, small)), "source", 32)
    baseline = collect_outputs(lambda: run_packed(model, small, layout=layout), small, model)
    changed = (small[0].roll(1), small[1])
    actual = collect_outputs(lambda: run_packed(model, changed, layout=layout), small, model)
    if not torch.equal(baseline[1], actual[1]):
        raise RuntimeError("source isolation failed: changing source 0 changed source 1")
    alternate_pad = (NVARC_PAD_TOKEN_ID + 1) % model.config.vocab_size
    actual = collect_outputs(lambda: run_packed(model, small, layout=layout,
                                               pad_token_id=alternate_pad), small, model)
    if any(not torch.equal(a, b) for a, b in zip(baseline, actual)):
        raise RuntimeError("causal padding isolation failed: PAD values changed real hidden states")
    return {"source_perturbation": "bitwise_equal_other_source",
            "padding_perturbation": "bitwise_equal_real_outputs", "source_lengths": [31, 47]}


def source_hashes():
    files = [*sorted((REPOSITORY_ROOT / "src").rglob("*.py")),
             Path(__file__), REPOSITORY_ROOT / "scripts/diagnostics/probe_nvarc_batch_capacity.py",
             REPOSITORY_ROOT / "scripts/diagnostics/_nvarc_attention_backends.py",
             REPOSITORY_ROOT / "scripts/cloud/probe_nvarc_packing_modal.py",
             REPOSITORY_ROOT / "pyproject.toml"]
    return {str(p.relative_to(REPOSITORY_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def source_metadata():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPOSITORY_ROOT, text=True).strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": source_hashes()}


def config_hashes(path, policy):
    return {key: hashlib.sha256(p.read_bytes()).hexdigest()
            for key, p in (("probe", path), ("training", policy.training_config))}


def dataset_hashes(training):
    """Hash the existing benchmark bundle before any remote GPU is requested."""
    from diffusion_arc.arc_data._split_layout import resolve_split_layout
    from diffusion_arc.arc_data import load_arc_split_config

    split = training.data.split_directory
    config = load_arc_split_config(split / "config.json")
    _, source = resolve_split_layout(split, dataset=config.dataset, source_split=config.source_split)
    files = sorted({*split.rglob("*.json"), *source.glob("*.json")})
    if not files or not source.is_dir():
        raise FileNotFoundError(f"benchmark dataset is missing: {split}")
    return {str(p.relative_to(split.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def run_worker(args):
    result = {"status": "running", "cases": [], "phase": "configuration"}
    adapters = {}

    def persist(*, notify=True):
        _write_json(args.output_dir / "result.json", result)
        if notify:
            print(PROGRESS_MARKER, result["phase"], flush=True)

    try:
        policy, training = load_probe_config(args.config)
        plan = build_plan(policy, training)
        result.update(plan=plan, phase="load_nvarc")
        persist()
        device, model, runner, tokenizer = load_runtime(training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("this diagnostic image targets Hopper/SM90")
        for backend in policy.backends:
            adapters[backend] = PackedAttention(backend)
            adapters[backend].install()
        dataset = ArcTrainingDataset(training.data.split_directory, training.data.view_name, tokenizer)
        result.update(
            environment={**_environment_metadata(), "packages": {
                name: version(name) for name in ("flash-attn", "flash-attn-3", "flash-attn-4",
                                                 "nvidia-cudnn-frontend", "nvidia-cutlass-dsl",
                                                 "quack-kernels", "apache-tvm-ffi")},
                         "gpu": torch.cuda.get_device_name(device),
                         "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory},
            checkpoint_config=model.config.to_dict(),
            setup_memory=_memory(device), seed=training.seed,
            backend_bindings={b: a.binding for b, a in adapters.items()}, isolation_checks={},
        )
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        seen_backends = set()
        for window, entry in zip(policy.windows, plan):
            sources = real_sources(dataset, window, min_context_pairs=training.loop.min_context_pairs_before_target)
            entry["input_sha256"] = hashlib.sha256(torch.cat(sources).numpy().tobytes()).hexdigest()
            batches = schedule_nvarc_sources(window.source_lengths, training.nvarc_scheduling)
            select_backend(model, "sdpa")
            result["phase"] = f"{window.name}/reference"
            persist()
            reference = collect_outputs(lambda: run_grouped(runner, sources, batches), sources, model)
            for specification in entry["cases"]:
                variant, backend = specification["variant"], specification["backend"]
                adapter = adapters[backend]
                case = {**specification, "window": window.name, "status": "running", "samples": [],
                        "backend_first_use": backend not in seen_backends}
                result["cases"].append(case)
                result["phase"] = f"{window.name}/{variant}/first_call"
                persist()
                select_backend(model, adapter.name)
                call = lambda: run_packed(model, sources, layout=specification["layout"])
                gc.collect()
                torch.cuda.empty_cache()
                case["first_call"] = measure_window(call, device, events)
                seen_backends.add(backend)
                result["phase"] = f"{window.name}/{variant}/warmup"
                persist()
                for _ in range(policy.warmup_calls):
                    output = call()
                    del output
                result["phase"] = f"{window.name}/{variant}/measurement"
                persist()
                for _ in range(policy.measured_calls):
                    case["samples"].append(measure_window(call, device, events))
                    # Save every sample; commit the volume at phase boundaries.
                    persist(notify=False)
                case["summary"] = summarize(case["samples"], real_tokens=entry["real_tokens"])
                result["phase"] = f"{window.name}/{variant}/backend_check"
                persist()
                case["attention_operators"] = inspect_backend(
                    call, adapter, layers=model.config.num_hidden_layers, device=device,
                )
                result["phase"] = f"{window.name}/{variant}/comparison"
                persist()
                actual = collect_outputs(call, sources, model)
                case["hidden_state_difference"] = numerical_difference(actual, reference)
                del actual
                if backend not in result["isolation_checks"]:
                    result["isolation_checks"][backend] = check_isolation(model, sources)
                case["status"] = "complete"
                persist()
            del reference
        result.update(status="complete", phase="complete")
    except Exception as error:
        result.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        persist()
        for adapter in adapters.values():
            adapter.close()
    return result


def launch_worker(command, *, log_path, timeout_seconds, on_progress):
    """Bound the complete worker, including loading, diagnostics, and GPU work."""
    with log_path.open("w") as log, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as process:
        expired = threading.Event()

        def stop():
            if process.poll() is None:
                expired.set()
                process.kill()

        timer = threading.Timer(timeout_seconds, stop)
        timer.start()
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(PROGRESS_MARKER + " "):
                    print(line.strip(), flush=True)
                    on_progress()
            return process.wait(), expired.is_set()
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                process.kill()
                process.wait()


def write_report(path, result):
    rows = ["# NVARC packing probe", "", f"Status: **{result['status']}**.", "",
            "Frozen BF16 NVARC on original ARC benchmark inputs. No pipeline capacity claim.", "",
            "| Window | Variant | PAD tokens | First call ms | Median ms/window | Real tokens/s | Peak allocated GiB | Relative L2 vs SDPA |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in result.get("cases", []):
        if case.get("status") != "complete":
            continue
        s = case["summary"]
        rows.append(f"| {case['window']} | {case['variant']} | {case['layout']['padding_tokens']} | "
                    f"{case['first_call']['wall_ms']:.3f} | {s['median_wall_ms']:.3f} | "
                    f"{s['tokens_per_second']:.1f} | {s['peak_allocated_bytes'] / 2**30:.3f} | "
                    f"{case['hidden_state_difference']['relative_l2']} |")
    rows.extend(["", "Numerical differences are descriptive; no agreement threshold was applied.",
                 "First calls are per case and may reuse kernels from earlier cases; they are not independent cold starts.",
                 "Warm-ups, first calls, backend profiles and output checks are excluded from medians.",
                 "Total alignment adds an isolated tail segment. Source alignment adds causal suffix PAD to each source.",
                 "CUDA spans include dispatch gaps and are not kernel-only execution times.",
                 "These timing samples are a diagnostic, not a formal benchmark."])
    for case in result.get("cases", []):
        if case.get("aliases"):
            rows.append(f"Alias: {', '.join(case['aliases'])} shares the exact layout of {case['window']}/{case['variant']}.")
    if "error" in result:
        rows.extend(["", f"Failure during `{result.get('phase')}`: {result['error']}"])
    path.write_text("\n".join(rows) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy, training = load_probe_config(args.config)
    hashes = config_hashes(args.config, policy)
    metadata = source_metadata() if metadata is None else metadata
    if metadata.get("source_sha256") != source_hashes():
        raise ValueError("uploaded source hashes differ from launch metadata")
    if "config_sha256" in metadata and metadata["config_sha256"] != hashes:
        raise ValueError("uploaded config hashes differ from launch metadata")
    data_hashes = dataset_hashes(training)
    if "dataset_sha256" in metadata and metadata["dataset_sha256"] != data_hashes:
        raise ValueError("uploaded benchmark dataset differs from launch metadata")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        **metadata, "config_sha256": hashes, "config": json.loads(args.config.read_text()),
        "dataset_sha256": data_hashes,
        "training_config": json.loads(policy.training_config.read_text()),
        "plan": build_plan(policy, training), "status": "running",
        "scope": "frozen NVARC only; no FLUX, VAE, optimizer or buffer reserve",
        "checkpoint_weights_hashed": False, "training_policy_validated": False,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    result = {"status": "error", "phase": "worker_start", "cases": []}
    started = time.monotonic()
    try:
        on_progress()
        returncode, timed_out = launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config),
             "--output-dir", str(args.output_dir)], log_path=args.output_dir / "worker.log",
            timeout_seconds=policy.max_duration_seconds, on_progress=on_progress,
        )
        partial = args.output_dir / "result.json"
        if partial.exists():
            result = json.loads(partial.read_text())
        expected = {(entry["name"], c["variant"]) for entry in build_plan(policy, training) for c in entry["cases"]}
        cases = result.get("cases", [])
        valid = (result.get("status") == "complete" and len(cases) == len(expected)
                 and {(c["window"], c["variant"]) for c in cases} == expected
                 and all(c.get("status") == "complete" and len(c["samples"]) == policy.measured_calls
                         for c in cases))
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error="worker exceeded configured time budget")
        elif returncode != 0 or not valid:
            result["status"] = "error"
            result.setdefault("error", "worker failed or returned incomplete results; see worker.log")
    except BaseException as error:
        partial = args.output_dir / "result.json"
        if partial.exists():
            try:
                result = json.loads(partial.read_text())
            except (OSError, ValueError):
                pass  # The original worker file is preserved for inspection.
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = result["status"]
        _write_json(args.output_dir / "summary.json", result)
        _write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", result)
        on_progress()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.dry_run:
        policy, training = load_probe_config(args.config, require_checkpoint=False)
        print(json.dumps(build_plan(policy, training), indent=2))
        return 0
    args.config = args.config.resolve()
    if args.output_dir is None:
        args.output_dir = REPOSITORY_ROOT / "artifacts/nvarc-packing" / datetime.now(
            timezone.utc,
        ).strftime("%Y%m%dT%H%M%S%fZ")
    args.output_dir = args.output_dir.resolve()
    result = run_worker(args) if args.worker else run_probe(args, metadata={
        **source_metadata(), "command": sys.argv,
    })
    print(f"NVARC packing probe {result['status']}: {args.output_dir}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
