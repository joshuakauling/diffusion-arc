#!/usr/bin/env python3
"""Bounded first-wave packing and fixed-operand attention diagnostics."""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib.metadata import version
import argparse
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from diffusion_arc.model.flow_matching import sample_flow_matching_batch
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids
from diffusion_arc.training.single_gpu_setup import (
    load_single_gpu_models, prepare_single_gpu_device, seed_single_gpu_process,
)
from scripts.diagnostics import probe_flux_packing as packing
from scripts.diagnostics import probe_flux_batch_shapes as worker_tools
from scripts.diagnostics import probe_flux_backend_conditions as conditions
from scripts.diagnostics import _flux_bfl_forward as observed
from scripts.diagnostics._flux_backend_replay import BACKENDS, fingerprint

PROGRESS_MARKER = "ATTENTION_EXECUTION_PROGRESS"
GROUPS = ("short_b4", "medium_b2", "long_b2")
BLOCKS = ("double.0", "double.4", "single.0", "single.19")


@dataclass(frozen=True)
class ProbeConfig:
    raw: dict
    fixture: Path
    packing_policy: object
    training: object
    config_paths: tuple
    checkpoint_paths: tuple
    max_duration_seconds: int


def load_probe_config(path, *, require_checkpoints=True):
    raw = json.loads(path.read_text())
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "packing_fixture", "groups",
            "capture_blocks", "backends", "upstream_seeds", "max_duration_seconds"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("unexpected attention diagnostic configuration")
    if (raw["groups"] != list(GROUPS) or raw["capture_blocks"] != list(BLOCKS)
            or raw["backends"] != list(BACKENDS)):
        raise ValueError("first-wave groups, blocks and backend scope must match the declared plan")
    if (not isinstance(raw["upstream_seeds"], list) or len(raw["upstream_seeds"]) != 3
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in raw["upstream_seeds"])
            or len(set(raw["upstream_seeds"])) != 3):
        raise ValueError("three distinct integer upstream seeds are required")
    duration = raw["max_duration_seconds"]
    if type(duration) is not int or not 1 <= duration <= 900:
        raise ValueError("first-wave worker budget must be in [1, 900] seconds")
    if not isinstance(raw["packing_fixture"], str) or not raw["packing_fixture"]:
        raise ValueError("packing_fixture must name the recorded input configuration")
    fixture = (path.resolve().parent / raw["packing_fixture"]).resolve()
    parent, training = packing.load_probe_config(fixture, require_checkpoints=require_checkpoints)
    chosen = {g["name"]: g for g in parent.groups}
    if any(name not in chosen for name in GROUPS):
        raise ValueError("missing declared historical input group")
    selected = replace(parent, groups=tuple(chosen[name] for name in GROUPS))
    return ProbeConfig(raw, fixture, selected, training,
        (path.resolve(), fixture, parent.training_config),
        (training.models.nvarc_checkpoint, training.models.flux_checkpoint), duration)


def build_plan(policy):
    from scripts.diagnostics._flux_packing_replay import evaluation_plan
    from scripts.diagnostics._flux_operation_replay import replay_plan

    kernel_plan = evaluation_plan(policy.raw["upstream_seeds"])
    groups = [{**row, "selected_target_indices": row["target_indices"][:2],
               "selected_text_lengths": row["text_lengths"][:2]}
              for row in packing.build_plan(policy.packing_policy, policy.training)]
    return dict(wave=1, groups=groups, capture_blocks=list(BLOCKS), backends=list(BACKENDS),
        upstream_seeds=policy.raw["upstream_seeds"], full_model_singleton_forwards=6,
        kernel_pair_cases=12, production_forward_vjp_evaluations=12 * kernel_plan["production_evaluations"],
        optional_deterministic_forward_vjp_evaluations_max=12 * kernel_plan["deterministic_evaluations_maximum"],
        direct_backend_pair_seed_comparisons=12 * kernel_plan["pairwise_comparisons"],
        kernel_case_plan=kernel_plan, operation_case_plan=replay_plan(policy.raw["upstream_seeds"]),
        operation_replay_groups=[GROUPS[0]], full_model_backward_calls=0, optimizer_updates=0,
        capture_backend="fa3", frozen_conditioning_backend="forced_cudnn_sdpa",
        frozen_vae_dtype="float32", frozen_vae_attention_backend="forced_math_sdpa",
        model_parameters="float32", model_compute="bfloat16",
        sdpa_packing="independent segment calls; not native-ragged throughput",
        max_worker_seconds=policy.max_duration_seconds, max_function_seconds=policy.max_duration_seconds + 120,
        retries=0, model_downloads=False, training_qualified=False)


def source_metadata():
    result = conditions.source_metadata()
    path = ROOT / "scripts/cloud/probe_attention_execution_modal.py"
    result["source_sha256"][str(path.relative_to(ROOT))] = conditions.file_sha256(path)
    return result


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): conditions.file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return packing.dataset_hashes(policy.training)


def prepare_frozen_inputs(policy, training, models):
    """Pin the BF16 encoder and FP32 VAE to their explicit supported backends."""
    def encode_vae(images):
        # cuDNN SDPA cannot consume the production VAE's FP32 Q/K/V.
        with sdpa_kernel(SDPBackend.MATH):
            return models.vae_encoder(images)

    preparation = SimpleNamespace(tokenizer=models.tokenizer, nvarc_runner=models.nvarc_runner,
                                  vae_encoder=encode_vae)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return packing.prepare_inputs(policy, training, preparation)


def singleton_inputs(stored, *, seed, device):
    """Construct independent references without any packing layout or processor."""
    prefixes, clean, _ = stored
    if len(prefixes) < 2 or any(p.ndim != 2 or p.shape[1] != 7680 or not 1 <= len(p) <= 8400
                              or p.dtype != torch.bfloat16 for p in prefixes[:2]):
        raise ValueError("two nonempty prefixes within the 8400-text-token contract are required")
    if (clean.ndim != 3 or clean.shape[0] != len(prefixes) or clean.shape[1:] != (900, 128)
            or clean.dtype != torch.float32):
        raise ValueError("clean latents must be FP32 [number of prefixes, 900, 128]")
    if any(not torch.isfinite(value).all() for value in (*prefixes[:2], clean[:2])):
        raise ValueError("non-finite frozen inputs")
    generator = torch.Generator(device=device).manual_seed(seed)
    # Reproduce the first two draws of the historical group, in its original order.
    draws = [sample_flow_matching_batch(latent.unsqueeze(0).to(device), generator=generator)
             for latent in clean[:2]]
    result = []
    for label, prefix, draw in zip(("A", "B"), prefixes[:2], draws):
        text = prefix.to(device).unsqueeze(0)
        positions = torch.zeros((1, len(prefix), 4), device=device, dtype=torch.int64)
        positions[0, :, -1] = torch.arange(len(prefix), device=device)
        result.append(dict(images=draw.noisy_latents, text=text, timesteps=draw.timesteps,
            target=draw.target_velocity, text_positions=positions,
            image_positions=build_flux_image_position_ids(1, device),
            text_mask=torch.ones((1, len(prefix)), dtype=torch.bool, device=device),
            lengths=(len(prefix),), labels=(label,)))
    return tuple(result)


def capture_pair(model, inputs, *, operations, on_result=None):
    """Observe original unpadded forwards; packing code is absent from this path."""
    from scripts.diagnostics._flux_operation_replay import OperationCapture

    rows, qkv, captures = [], [], []
    for item in inputs:
        identity = observed.call_identity(item)
        capture = OperationCapture(model) if operations else None
        with observed.controlled_precision("bf16", item["images"].device), \
                (capture.installed() if capture is not None else nullcontext()), \
                conditions.attention_route("fa3", capture_qkv=True) as route:
            output = observed.forward_call(model, "diffusers", item)
        if observed.call_identity(item) != identity:
            raise RuntimeError("singleton capture mutated shared inputs")
        rows.append(dict(label=item["labels"][0], inputs=identity, velocity=fingerprint(output),
            loss=float((output.float() - item["target"]).square().mean()),
            timestep=float(item["timesteps"][0]), attention_calls=route["calls"]))
        qkv.append(route["qkv"])
        if capture is not None:
            captures.append(capture)
        if on_result is not None:
            on_result(rows)
    return rows, qkv, tuple(captures)


def validate_complete(result, policy):
    from scripts.diagnostics._flux_packing_replay import validate_replay_result
    from scripts.diagnostics._flux_operation_replay import validate_operation_result

    if result.get("status") != "complete" or result.get("plan") != build_plan(policy):
        raise ValueError("worker did not complete the declared first-wave plan")
    if set(result.get("checkpoints", {})) != {"flux", "nvarc", "vae"}:
        raise ValueError("missing checkpoint evidence")
    groups = result.get("groups", [])
    if [row.get("name") for row in groups] != list(GROUPS):
        raise ValueError("missing or repeated input group")
    for group in groups:
        if [row.get("label") for row in group.get("singletons", [])] != ["A", "B"]:
            raise ValueError("missing independent singleton input evidence")
        for index, row in enumerate(group["singletons"]):
            stages = [*(f"double.{i}" for i in range(5)), *(f"single.{i}" for i in range(20))]
            if row.get("attention_calls") != stages or not row.get("inputs"):
                raise ValueError("incomplete singleton attention capture")
            if (not math.isfinite(row.get("loss", float("nan")))
                    or not 0 <= row.get("timestep", -1) <= 1):
                raise ValueError("invalid singleton loss or timestep")
        if set(group.get("kernels", {})) != set(BLOCKS):
            raise ValueError("missing kernel block evidence")
        for block, kernel in group["kernels"].items():
            if not validate_replay_result(kernel, seeds=tuple(policy.raw["upstream_seeds"]),
                    profile=group["name"] == GROUPS[0] and block == BLOCKS[0]):
                raise ValueError("incomplete kernel replay")
    if not validate_operation_result(groups[0].get("operations", {}), seeds=tuple(policy.raw["upstream_seeds"])):
        raise ValueError("incomplete first-boundary operation replay")
    return True


def worker(args):
    from scripts.diagnostics._flux_packing_replay import replay_pair
    from scripts.diagnostics._flux_operation_replay import operation_replays

    policy = load_probe_config(args.config)
    result = dict(status="running", phase="preflight", plan=build_plan(policy), groups=[], checkpoints={})
    started = time.monotonic()

    def persist(phase):
        result.update(phase=phase, worker_elapsed_seconds=time.monotonic() - started)
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(f"{PROGRESS_MARKER} {phase} elapsed={result['worker_elapsed_seconds']:.1f}s", flush=True)

    try:
        from scripts.diagnostics._flux_packing import packing_layout
        from scripts.diagnostics._flux_packing_preflight import validate_layout

        # Explicit structural preflight; the independent capture path never
        # consumes these layouts or any index produced by the packer.
        for group in result["plan"]["groups"]:
            validate_layout(packing_layout(tuple(group["selected_text_lengths"]), 900, torch.device("cpu")))
        result["packing_preflight"] = "all selected layouts validated independently"
        device = prepare_single_gpu_device(policy.training)
        seed_single_gpu_process(policy.training.seed)
        torch.set_num_threads(8)
        properties = torch.cuda.get_device_properties(device)
        result["environment"] = {**worker_tools._environment_metadata(), "gpu": properties.name,
            "total_memory_bytes": properties.total_memory, "visible_devices": torch.cuda.device_count(),
            "compute_capability": [properties.major, properties.minor],
            "flash_packages": {name: version(name) for name in ("flash-attn", "flash-attn-3", "flash-attn-4")}}
        for name, folder in (("flux", policy.training.models.flux_checkpoint / "transformer"),
                ("nvarc", policy.training.models.nvarc_checkpoint),
                ("vae", policy.training.models.flux_checkpoint / "vae")):
            persist(f"checkpoint/{name}")
            result["checkpoints"][name] = conditions.checkpoint_evidence(folder)
        persist("load_cached_models")
        models = load_single_gpu_models(policy.training, device=device)
        model = models.transformer
        model.eval().requires_grad_(False)
        if model.gradient_checkpointing:
            raise ValueError("checkpointing belongs to the subsequent training wave")
        persist("prepare_frozen_inputs")
        stored = prepare_frozen_inputs(policy.packing_policy, policy.training, models)
        del models
        gc.collect()
        torch.cuda.empty_cache()
        for group_spec in result["plan"]["groups"]:
            name = group_spec["name"]
            group = dict(name=name, source_sha256=stored[name][2], kernels={})
            result["groups"].append(group)
            inputs = singleton_inputs(stored[name], seed=policy.training.seed, device=device)
            persist(f"capture/{name}")
            def capture_progress(rows):
                group["singletons"] = rows
                persist(f"capture_progress/{name}/{rows[-1]['label']}")

            group["singletons"], qkv, captures = capture_pair(model, inputs,
                operations=name == GROUPS[0], on_result=capture_progress)
            if captures:
                persist(f"operations/{name}")
                with observed.controlled_precision("bf16", device):
                    group["operations"] = operation_replays(model, captures, seeds=tuple(policy.raw["upstream_seeds"]))
                del captures
                persist(f"operations_complete/{name}")
            for block in BLOCKS:
                persist(f"kernel/{name}/{block}")

                def progress(partial):
                    group["kernels"][block] = partial
                    persist(f"kernel_progress/{name}/{block}")

                group["kernels"][block] = replay_pair(tuple(sample[block] for sample in qkv),
                    seeds=tuple(policy.raw["upstream_seeds"]),
                    profile=name == GROUPS[0] and block == BLOCKS[0], on_result=progress)
                persist(f"kernel_complete/{name}/{block}")
            del inputs, qkv
            gc.collect()
            torch.cuda.empty_cache()
        result["status"] = "complete"
        validate_complete(result, policy)
        persist("complete")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def write_report(path, result):
    lines = ["# Attention execution diagnostic — wave 1", "", f"Status: **{result['status']}**.", "",
        "Isolated attention input gradients and operation replays; no full-model backward,",
        "optimizer update, training qualification, or throughput ranking.", "",
        f"Elapsed worker/supervisor seconds: {result.get('worker_elapsed_seconds')} / {result.get('elapsed_seconds')}.",
        "", "Detailed per-example metrics, identities and dispatch evidence are in `summary.json`."]
    if result.get("error"):
        lines += ["", "Error: " + result["error"]]
    for group in result.get("groups", []):
        lines += ["", f"- {group['name']}: {len(group.get('kernels', {}))}/4 kernel pairs recorded; "
                  f"operation replay: {group.get('operations', {}).get('status', 'not scheduled')}"]
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    checks = {**source_metadata(), "config_sha256": config_hashes(args.config, policy),
              "dataset_sha256": dataset_hashes(policy)}
    if metadata is not None and any(metadata.get(key) != value for key, value in checks.items()):
        raise ValueError("uploaded source/config/data differs from the approved launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**(metadata or {}), **checks, "config": policy.raw, "plan": build_plan(policy),
                "effective_worker_seconds": policy.max_duration_seconds, "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start", groups=[])
    try:
        returncode, timed_out = worker_tools.launch_worker(
            [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--config", str(args.config),
             "--output-dir", str(args.output_dir)], log_path=args.output_dir / "worker.log",
            timeout_seconds=policy.max_duration_seconds, on_progress=on_progress, progress_marker=PROGRESS_MARKER)
        partial = args.output_dir / "result.json"
        if partial.is_file():
            result = json.loads(partial.read_text())
        result.update(worker_returncode=returncode, timed_out=timed_out)
        if timed_out:
            result.update(status="timeout", error=f"worker exceeded its fixed {policy.max_duration_seconds}-second ceiling")
        elif returncode != 0 or result.get("status") != "complete":
            result.update(status="error", error=result.get("error", "worker exited without complete evidence"))
        else:
            validate_complete(result, policy)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        manifest["status"] = result["status"]
        worker_tools._write_json(args.output_dir / "summary.json", result)
        worker_tools._write_json(args.output_dir / "manifest.json", manifest)
        write_report(args.output_dir / "report.md", result)
        on_progress()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/attention_execution.json")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
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
