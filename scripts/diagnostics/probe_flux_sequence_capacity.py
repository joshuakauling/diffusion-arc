#!/usr/bin/env python3
"""Bound the length of one synthetic FA3 sequence using full FLUX forward/backward."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnostics import probe_flux_packed_capacity as capacity

performance = capacity.performance
packing = capacity.packing
worker_tools = capacity.worker_tools
evidence = capacity.evidence
PROGRESS_MARKER = "FLUX_SEQUENCE_CAPACITY_PROGRESS"


@dataclass(frozen=True)
class ProbeConfig:
    raw: dict
    packing_policy: object
    training: object
    config_paths: tuple
    checkpoint_paths: tuple
    max_duration_seconds: int


def load_probe_config(path, *, require_checkpoints=True):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    fixed = dict(schema_version=1, backend="fa3", modulation_reduction="fp32", image_tokens=900,
        minimum_joint_tokens=1024, start_joint_tokens=8192, maximum_joint_tokens=32768,
        quantum=256, confirmation_calls=3, flow_seed=170)
    if (not isinstance(raw, dict) or set(raw) != {*fixed, "packing_fixture", "max_duration_seconds"}
            or any(type(raw[k]) is not type(v) or raw[k] != v for k, v in fixed.items())
            or not isinstance(raw["packing_fixture"], str) or not raw["packing_fixture"]
            or type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 330):
        raise ValueError("configuration differs from the approved single-sequence length search")
    fixture = (path.parent / raw["packing_fixture"]).resolve()
    parent, training = packing.load_probe_config(fixture, require_checkpoints=require_checkpoints)
    return ProbeConfig(raw, parent, training, (path, fixture, parent.training_config),
        (training.models.nvarc_checkpoint, training.models.flux_checkpoint), raw["max_duration_seconds"])


def build_plan(policy):
    return {**policy.raw, "batch_size": 1, "attention_segments": 1,
        "sequence": "L-900 text tokens followed by 900 image tokens; cu_seqlens=[0,L]",
        "text_construction": "repeat the resident NVARC hidden-state template; continuous text positions",
        "search": "exponential growth then bisection on multiples of 256; confirm lower 3 times and upper once",
        "memory": "resident frozen models, FP32 weights/gradients/AdamW and unchanged preparation reserve",
        "allocator": "unchanged environment; collect garbage and empty cache before every probe",
        "compute": "bfloat16", "tf32": False, "checkpointing": False,
        "optimizer_updates_during_search": 0, "loss_weight": "1/48",
        "capacity_scope": "full-model single-sequence execution; not an abstract FA3 kernel limit",
        "timing_scope": "assembly + forward/loss + backward; excludes input preparation, gradient reset and optimizer step"}


def source_metadata():
    result = evidence.source_metadata()
    path = ROOT / "scripts/cloud/probe_flux_sequence_capacity_modal.py"
    result["source_sha256"][str(path.relative_to(ROOT))] = evidence.file_sha256(path)
    return result


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): evidence.file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return packing.dataset_hashes(policy.training)


def sequence_spec(length):
    return dict(name=f"single_{length}", batch_size=1, text_lengths=[length - 900],
                padded_length=length - 900, real_joint_tokens=length)


def synthetic_data(template, clean, length, seed, *, max_text_tokens):
    """Repeat values, never sequence boundaries: all L tokens share one attention segment."""
    device = template.device
    text_length = length - 900
    if (template.ndim != 2 or template.shape[0] == 0 or clean.shape != (900, 128)
            or not 1 <= text_length <= max_text_tokens):
        raise ValueError("invalid synthetic sequence input contract")
    indices = torch.arange(text_length, device=device) % template.shape[0]
    prefix = template.index_select(0, indices)
    layout = packing.packing_layout((text_length,), 900, device)
    performance.validate_layout(layout, max_text_tokens=max_text_tokens)
    flow = packing.sample_flow_matching_batch(clean.unsqueeze(0),
        generator=torch.Generator(device=device).manual_seed(seed))
    return dict(prefixes=(prefix,), clean=clean.unsqueeze(0), flow=flow,
        image_positions=packing.build_flux_image_position_ids(1, device), layout=layout)


def allocator_state(device):
    stats = torch.cuda.memory_stats(device)
    keys = ("allocated_bytes.all.current", "reserved_bytes.all.current", "inactive_split_bytes.all.current",
            "num_alloc_retries", "num_ooms")
    free, total = torch.cuda.mem_get_info(device)
    return dict(free_bytes=free, total_bytes=total, stats={k: stats[k] for k in keys if k in stats})


def attempt_length(models, optimizer, backend, template, clean, length, policy, *, profile=False):
    """Return only CPU evidence; the exception and failed graph die with this frame."""
    phase = "input_preparation"
    spec = sequence_spec(length)
    try:
        data = synthetic_data(template, clean, length, policy.raw["flow_seed"],
                              max_text_tokens=policy.raw["maximum_joint_tokens"] - 900)
        identity = performance.input_identity(data)
        before = allocator_state(template.device)
        def call():
            return performance.measure_call(models.transformer, optimizer, "fa3", spec, data, backend,
                compute_dtype=torch.bfloat16, modulation_reduction="fp32")
        phase = "forward_backward"
        with performance.observe_modulation_reduction() as observation:
            row, output = call()
        del output
        performance.validate_observation(observation, "fa3")
        phase = "gradient_health"
        health = performance.gradient_health(models.transformer)
        if identity != performance.input_identity(data):
            raise ValueError("synthetic capacity input mutation")
        record = dict(status="passed", spec=spec, calls=[row], gradient_health=health,
            input_identity=identity, inputs_unchanged=True, modulation_observation=observation,
            cu_seqlens=data["layout"].cumulative.tolist(), maximum_segment=data["layout"].maximum,
            allocator_before=before, allocator_after=allocator_state(template.device))
        if profile:
            phase = "backend_profile"
            record["backend_profile"] = performance.profile_backend(call, models.transformer, "fa3", backend)
        return record
    except torch.OutOfMemoryError as error:
        if phase != "forward_backward":
            raise RuntimeError(f"OOM outside model execution ({phase}): {error}") from error
        return dict(status="oom", spec=spec, phase=phase, error=str(error),
                    allocator_at_failure=allocator_state(template.device))


def search_lengths(raw, attempt, recover, publish):
    """Find adjacent tested lengths, with a fresh cache before every independent call."""
    low = high = None
    quantum = raw["quantum"]
    result = dict(status="running", attempts=[], confirmations=[])
    def probe(length, phase):
        recover()
        row = attempt(length)
        if row.get("status") not in ("passed", "oom"):
            raise ValueError("invalid capacity attempt status")
        row = {**row, "search_phase": phase}
        result["attempts"].append(row)
        publish(result)
        return row
    length = raw["start_joint_tokens"]
    while True:
        row = probe(length, "coarse")
        if row["status"] == "oom":
            high = length
            break
        low = length
        if length == raw["maximum_joint_tokens"]:
            break
        length = min(length * 2, raw["maximum_joint_tokens"])
    if low is None:
        row = probe(raw["minimum_joint_tokens"], "minimum_control")
        if row["status"] != "passed":
            raise RuntimeError("minimum sequence did not fit; no lower bound established")
        low = raw["minimum_joint_tokens"]
    while high is not None and high - low > quantum:
        middle = ((low + high) // (2 * quantum)) * quantum
        row = probe(middle, "bisect")
        if row["status"] == "passed":
            low = middle
        else:
            high = middle
    result.update(largest_passed=low, smallest_oom=high)
    for _ in range(raw["confirmation_calls"]):
        row = probe(low, "confirm_lower")
        result["confirmations"].append(row)
        publish(result)
        if row["status"] != "passed":
            raise RuntimeError("largest sequence failed fresh-cache confirmation")
    if high is not None:
        row = probe(high, "confirm_upper")
        result["upper_confirmation"] = row
        publish(result)
        if row["status"] != "oom":
            raise RuntimeError("upper OOM boundary did not reproduce; no stable maximum")
    result.update(status="bounded" if high is not None else "lower_bound_only",
                  resolution_tokens=quantum if high is not None else None)
    publish(result)
    return result


def worker(args):
    policy = load_probe_config(args.config)
    torch.set_num_threads(8)
    started = time.monotonic()
    result = dict(status="running", phase="configuration", plan=build_plan(policy))
    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(PROGRESS_MARKER, phase, flush=True)
    try:
        persist("fa3/preflight")
        device = worker_tools.prepare_single_gpu_device(policy.training)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("cached image requires Hopper/SM90")
        backend = packing.FlashBackend("fa3")
        result["backend_preflight"] = packing.backend_preflight(backend, device, policy.packing_policy.numerical_tolerances)
        persist("load_models_and_training_memory")
        device, models, optimizer, reserve, setup = worker_tools.prepare_runtime(policy.training)
        model = models.transformer
        if model.config.attention_head_dim != 128 or models.flux_compute_dtype != torch.bfloat16:
            raise ValueError("capacity experiment requires BF16 compute and head dimension 128")
        with performance.training_precision(device):
            environment = worker_tools._environment_metadata()
        props = torch.cuda.get_device_properties(device)
        environment.update(gpu_name=props.name, gpu_total_bytes=props.total_memory,
            allocator_backend=torch.cuda.memory.get_allocator_backend(),
            allocator_environment={k: os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")})
        environment["packages"]["flash-attn-3"] = version("flash-attn-3")
        result.update(setup=setup, environment=environment, backend_binding=backend.binding,
            checkpoints={name: evidence.checkpoint_evidence(folder) for name, folder in (
                ("flux", policy.training.models.flux_checkpoint / "transformer"),
                ("vae", policy.training.models.flux_checkpoint / "vae"),
                ("nvarc", policy.training.models.nvarc_checkpoint))},
            synthetic_template=performance.fingerprint(reserve[0]))
        versions = {n: p._version for n, p in model.named_parameters()}
        pointers = {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None}
        persist("single_sequence/backend_profile")
        capacity.recover_memory(device)
        result["control"] = attempt_length(models, optimizer, backend, reserve[0], reserve[-1][0],
                                           policy.raw["minimum_joint_tokens"], policy, profile=True)
        if result["control"]["status"] != "passed":
            raise RuntimeError("initial single-sequence control failed")
        def publish(record):
            result["search"] = record
            latest = record["attempts"][-1]
            persist(f"length/{latest['spec']['real_joint_tokens']}/{latest['search_phase']}/{latest['status']}")
        def attempt(length):
            persist(f"length/{length}/running")
            return attempt_length(models, optimizer, backend, reserve[0], reserve[-1][0], length, policy)
        search_lengths(policy.raw, attempt, lambda: capacity.recover_memory(device), publish)
        result.update(weights_unchanged=versions == {n: p._version for n, p in model.named_parameters()},
            gradient_storage_unchanged=pointers == {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None},
            template_unchanged=result["synthetic_template"] == performance.fingerprint(reserve[0]), status="complete")
        validate_complete(result, policy)
        persist("complete")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def validate_complete(result, policy):
    if result.get("status") != "complete" or result.get("plan") != build_plan(policy):
        raise ValueError("incomplete sequence capacity experiment or changed plan")
    if any(result.get(k) is not True for k in ("weights_unchanged", "gradient_storage_unchanged", "template_unchanged")):
        raise ValueError("model, gradient storage or template changed")
    if (set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"}
            or result.get("backend_preflight", {}).get("passed") is not True
            or result.get("control", {}).get("backend_profile", {}).get("verified") is not True):
        raise ValueError("missing checkpoint or backend evidence")
    search = result.get("search", {})
    # Replay deterministic search decisions against the recorded evidence, rejecting gaps or reordering.
    records = iter(search.get("attempts", []))
    def replay(length):
        row = next(records)
        spec = sequence_spec(length)
        if row.get("spec") != spec:
            raise ValueError("sequence length search evidence differs from the planned search")
        if row.get("status") == "passed":
            capacity.validate_passed(row, spec, 1)
            if row.get("cu_seqlens") != [0, length] or row.get("maximum_segment") != length:
                raise ValueError("probe did not execute one complete attention sequence")
        elif row.get("phase") != "forward_backward" or not row.get("error"):
            raise ValueError("invalid allocation OOM evidence")
        return row
    expected = search_lengths(policy.raw, replay, lambda: None, lambda _: None)
    if next(records, None) is not None or expected != search:
        raise ValueError("incomplete or inconsistent boundary confirmation")


def write_report(path, result):
    search = result.get("search", {})
    lines = ["# FA3 single-sequence capacity", "", f"Status: **{result['status']}**.", "",
        "One attention segment: L-900 synthetic text tokens + 900 image tokens, with continuous text positions.",
        "Full FLUX forward/backward, BF16 compute and FP32 modulation sums; no checkpointing or optimizer update.",
        "FP32 training state, frozen models and preparation reserve remain resident. Cache is cleared before every probe.", "",
        "| Joint tokens | Text tokens | Phase | Outcome | Peak allocated GiB | F+B ms |",
        "| ---: | ---: | --- | --- | ---: | ---: |"]
    for row in search.get("attempts", []):
        length = row["spec"]["real_joint_tokens"]
        calls = row.get("calls", [])
        peak = f"{calls[0]['peak_allocated_bytes'] / 2**30:.3f}" if calls else "—"
        timing = f"{calls[0]['forward_backward_ms']:.3f}" if calls else "—"
        lines.append(f"| {length} | {length - 900} | {row['search_phase']} | {row['status']} | {peak} | {timing} |")
    if search.get("status") == "bounded":
        lines += ["", f"Confirmed fitting length: {search['largest_passed']}; confirmed OOM: {search['smallest_oom']}.",
                  f"Resolution: {search['resolution_tokens']} tokens. This is a runtime-specific memory bound, not a universal FA3 limit."]
    elif search.get("status") == "lower_bound_only":
        lines += ["", f"At least {search['largest_passed']} tokens fit; no upper failure was reached."]
    if result.get("error"):
        lines += ["", f"Error during {result.get('phase')}: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    checks = {**source_metadata(), "config_sha256": config_hashes(args.config, policy), "dataset_sha256": dataset_hashes(policy)}
    if metadata is not None and any(metadata.get(k) != v for k, v in checks.items()):
        raise ValueError("uploaded source/config/data differs from launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**(metadata or {}), **checks, "config": policy.raw, "plan": build_plan(policy), "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start")
    try:
        returncode, timed_out = worker_tools.launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config), "--output-dir", str(args.output_dir)],
            log_path=args.output_dir / "worker.log", timeout_seconds=policy.max_duration_seconds,
            on_progress=on_progress, progress_marker=PROGRESS_MARKER)
        partial = args.output_dir / "result.json"
        if partial.is_file():
            result = json.loads(partial.read_text())
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error=f"worker exceeded {policy.max_duration_seconds} seconds")
        elif returncode != 0 or result.get("status") != "complete":
            result.update(status="error", error=result.get("error", "worker exited without complete evidence"))
        else:
            validate_complete(result, policy)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = result["status"]
        write_report(args.output_dir / "report.md", result)
        worker_tools._write_json(args.output_dir / "summary.json", result)
        worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_sequence_capacity.json")
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(build_plan(load_probe_config(args.config, require_checkpoints=False)), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    result = worker(args) if args.worker else run_probe(args)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
