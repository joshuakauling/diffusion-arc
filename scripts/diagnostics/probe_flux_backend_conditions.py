#!/usr/bin/env python3
"""Compare attention numerics across native, adapted and new ARC conditions.

Whole-model forwards retain native projections, norms and RoPE. Isolated
attention replays receive identical post-RoPE BF16 Q/K/V and upstream gradients.
Only cached checkpoints and the compact, audited input fixture are used.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from diffusion_arc._third_party.diffusers_flux2 import transformer_flux2 as df
from diffusion_arc.training.single_gpu_setup import _validate_flux_checkpoint, seed_single_gpu_process
from scripts.diagnostics import _flux_backend_inputs as preparation
from scripts.diagnostics import _flux_backend_replay as replay
from scripts.diagnostics import _flux_bfl_forward as observed
from scripts.diagnostics._flux_bfl_weights import build_bfl_model, validate_source
from scripts.diagnostics import probe_flux_batch_shapes as worker_tools


PROGRESS_MARKER = "FLUX_BACKEND_CONDITIONS_PROGRESS"
BLOCKS = ("double.0", "double.4", "single.0", "single.19")
MODEL_VARIANTS = ("bfl_fp32", "bfl_bf16", "diffusers_fp32", "bf16_math", "cudnn", "fa2", "fa3", "fa4")


@dataclass(frozen=True)
class ProbeConfig:
    raw: dict
    checkpoint_paths: tuple[Path, ...]
    config_paths: tuple[Path, ...]
    fixture_paths: tuple[Path, ...]
    max_duration_seconds: int

    @property
    def models(self):
        return dict(zip(("base_checkpoint", "nvarc_checkpoint", "overfit_checkpoint"), self.checkpoint_paths))


def load_probe_config(path, *, require_checkpoints=True):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    expected = {"schema_version", "max_duration_seconds", "seed", "models", "inputs_fixture",
                "timesteps", "backends", "capture_blocks", "thresholds"}
    if not isinstance(raw, dict) or set(raw) != expected or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("invalid backend-conditions schema or fields")
    if type(raw["max_duration_seconds"]) is not int or not 1 <= raw["max_duration_seconds"] <= 1800:
        raise ValueError("worker budget must be an integer in [1, 1800] seconds")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if raw["timesteps"] != [1.0, 0.5] or any(type(t) not in (int, float) for t in raw["timesteps"]):
        raise ValueError("the diagnostic requires t=1 and t=0.5")
    if raw["backends"] != list(replay.BACKENDS[1:]) or raw["capture_blocks"] != list(BLOCKS):
        raise ValueError("all five BF16 backends and four declared blocks are required")
    keys = ("base_checkpoint", "nvarc_checkpoint", "overfit_checkpoint")
    if not isinstance(raw["models"], dict) or set(raw["models"]) != set(keys):
        raise ValueError("exact base, NVARC and overfit checkpoint paths are required")
    if any(not isinstance(raw["models"][key], str) or not raw["models"][key] for key in keys):
        raise ValueError("checkpoint paths must be nonempty strings")
    paths = tuple((path.parent / raw["models"][key]).resolve() for key in keys)
    if len(set(paths)) != 3:
        raise ValueError("base, NVARC and overfit checkpoints must be distinct")
    if not isinstance(raw["inputs_fixture"], str) or not raw["inputs_fixture"]:
        raise ValueError("inputs_fixture must name the compact audited input file")
    fixture_path = (path.parent / raw["inputs_fixture"]).resolve()
    fixture = preparation.load_fixture(fixture_path)
    if not str(paths[2]).endswith("/" + fixture["overfit"]["checkpoint_relative_path"]):
        raise ValueError("overfit path differs from the successful checkpoint manifest")
    limits = raw["thresholds"]
    if (not isinstance(limits, dict) or set(limits) != {
            "fp32_model_relative_l2", "bf16_model_relative_l2", "kernel_output_relative_l2",
            "kernel_gradient_relative_l2"} or any(type(v) not in (int, float)
            or not math.isfinite(v) or not 0 < v < 1 for v in limits.values())):
        raise ValueError("all diagnostic thresholds must be finite fractions in (0, 1)")
    if require_checkpoints:
        for folder in (*paths, paths[0] / "text_encoder", paths[0] / "tokenizer", paths[0] / "vae", paths[0] / "transformer"):
            if not folder.is_dir():
                raise FileNotFoundError(f"cached checkpoint directory missing: {folder}")
    return ProbeConfig(raw, paths, (path,), (fixture_path,), raw["max_duration_seconds"])


def build_plan(policy):
    fixture = preparation.load_fixture(policy.fixture_paths[0])
    arc = [f"{view['name']}_t{t:g}" for view in fixture["arc_views"] for t in policy.raw["timesteps"]]
    conditions = [{"checkpoint": "base", "input": "native_prompt_t1"},
                  *({"checkpoint": checkpoint, "input": name}
                    for checkpoint in ("base", "overfit") for name in arc)]
    return dict(conditions=conditions, full_model_variants=list(MODEL_VARIANTS),
                full_model_forwards=len(conditions) * len(MODEL_VARIANTS),
                capture_blocks=list(BLOCKS), kernel_replays=len(conditions) * len(BLOCKS),
                kernel_backends=list(replay.BACKENDS), kernel_forward_backward_pairs=len(conditions) * len(BLOCKS) * len(replay.BACKENDS),
                native_prompt=fixture["native_prompt"], parameter_dtype="torch.float32",
                image_tokens=900, tf32=False, model_backward_calls=0, optimizer_updates=0,
                kernel_gradient_scope="shared seeded upstream VJP; not model-loss gradients",
                thresholds=policy.raw["thresholds"])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata():
    paths = {*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/diagnostics/*.py"),
             *ROOT.glob("scripts/diagnostics/_third_party/bfl_flux2/*"),
             ROOT / "scripts/cloud/probe_flux_backend_conditions_modal.py",
             ROOT / "scripts/cloud/probe_flux_packing_modal.py", ROOT / "pyproject.toml"}
    return {"source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths) if p.is_file()}}


def config_hashes(path, policy):
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in policy.config_paths}


def dataset_hashes(policy):
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in policy.fixture_paths}


def checkpoint_evidence(path, *, expected=None):
    """Hash actual config and weight files; verify the saved overfit manifest."""
    path = Path(path)
    index_paths = list(path.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(f"ambiguous checkpoint indices in {path}")
    names = {"config.json"}
    if index_paths:
        index = json.loads(index_paths[0].read_text())
        names.add(index_paths[0].name)
        names.update(index["weight_map"].values())
    else:
        names.update(p.name for p in path.glob("*.safetensors"))
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError(f"no checkpoint weights in {path}")
    if any(Path(name).name != name for name in names):
        raise ValueError("checkpoint index contains a nonlocal shard path")
    actual = {name: {"bytes": (path / name).stat().st_size, "sha256": file_sha256(path / name)} for name in sorted(names)}
    if expected is not None and actual != expected:
        raise ValueError("checkpoint files differ from the successful overfit manifest")
    return {"path": str(path), "files": actual, "manifest_verified": expected is not None}


def assembled_inputs(record, device):
    text = record.text.to(device=device, dtype=torch.float32)
    images = record.images.to(device=device, dtype=torch.float32)
    if (text.ndim != 3 or text.shape[0] != 1 or text.shape[2] != 7680
            or tuple(images.shape) != (1, 900, 128)):
        raise ValueError("real FLUX input shape contract violated")
    result = dict(images=images, text=text, image_positions=record.image_ids.to(device),
                  text_positions=record.text_ids.to(device), timesteps=record.timesteps.to(device).float(),
                  target=None if record.target_velocity is None else record.target_velocity.to(device).float(),
                  text_mask=torch.ones((1, text.shape[1]), dtype=torch.bool, device=device),
                  lengths=(text.shape[1],), labels=(record.name,))
    for name in ("images", "text", "timesteps", "target"):
        if result[name] is not None and not torch.isfinite(result[name]).all():
            raise ValueError(f"non-finite prepared {name}")
    if result["image_positions"].shape != (1, 900, 4) or result["text_positions"].shape != (1, text.shape[1], 4):
        raise ValueError("real FLUX position contract violated")
    return result


class SelectedCapture(observed.ForwardCapture):
    """Observe only four declared blocks and velocity to bound diagnostic memory."""

    def record(self, stage, streams):
        if stage in (*BLOCKS, "velocity"):
            super().record(stage, streams)

    def finish(self, output):
        if output.shape != self.inputs["images"].shape:
            raise ValueError("unexpected final velocity shape")
        self.record("velocity", {"image": output})
        if tuple(self.outputs) != (*BLOCKS, "velocity"):
            raise ValueError("incomplete selected model boundary observations")


@contextmanager
def attention_route(backend, capture_qkv=False):
    """Replace only the attention call; all native pre/post arithmetic stays intact."""
    native = df.dispatch_attention_fn
    stages = observed.stage_order(5, 20)[3:-2]
    state = {"calls": [], "qkv": {}}

    def dispatch(q, k, v, *args, **kwargs):
        if args or kwargs.get("attn_mask") is not None:
            raise ValueError("condition comparisons require unpadded singleton attention")
        index = len(state["calls"])
        if index >= len(stages):
            raise ValueError("extra attention call outside the 5+20 FLUX blocks")
        stage = stages[index]
        state["calls"].append(stage)
        common = None
        if capture_qkv or backend not in ("native_math", "fp32_math"):
            common = tuple(t.to(dtype=v.dtype).contiguous() for t in (q, k, v))
        if capture_qkv and stage in BLOCKS:
            if v.dtype != torch.bfloat16:
                raise ValueError("real QKV replay must capture BF16 compute values")
            state["qkv"][stage] = tuple(t.detach().clone() for t in common)
        if backend in ("native_math", "fp32_math"):
            return native(q, k, v, **kwargs)
        return replay.attention_forward(backend, *common)

    with patch.object(df, "dispatch_attention_fn", dispatch):
        yield state
    if state["calls"] != stages:
        raise ValueError("the requested backend did not execute all 25 attention calls")
    if capture_qkv and tuple(state["qkv"]) != BLOCKS:
        raise ValueError("missing real QKV capture")


def model_case(model, implementation, inputs, precision, backend=None, *, capture_qkv=False):
    from contextlib import nullcontext

    capture = SelectedCapture(model, implementation, inputs)
    identity = observed.call_identity(inputs)
    route = attention_route(backend, capture_qkv) if implementation == "diffusers" else nullcontext(None)
    started = time.monotonic()
    with observed.controlled_precision(precision, inputs["images"].device), capture.installed(), route as selected:
        output = observed.forward_call(model, implementation, inputs)
        capture.finish(output)
        loss = None if inputs["target"] is None else float((output.float() - inputs["target"]).square().mean())
    if identity != observed.call_identity(inputs):
        raise ValueError("model forward modified its shared inputs")
    row = dict(implementation=implementation, precision=precision, backend=backend or "native_math",
               loss=loss, outputs=capture.outputs, input_fingerprints=identity,
               elapsed_seconds=time.monotonic() - started,
               attention_calls=25 if selected is None else len(selected["calls"]))
    return row, capture.snapshots, {} if selected is None else selected["qkv"]


def compare_model(values, reference, label):
    rows = {f"{stage}/{stream}": replay.tensor_metrics(tensor, reference[label][stage][stream])
            for stage, streams in values[label].items() for stream, tensor in streams.items()}
    return dict(velocity=rows["velocity/image"], maximum_observed_relative_l2=max(
                row["relative_l2"] for row in rows.values() if row["relative_l2"] is not None),
                stages=rows)


def validate_complete(result, policy):
    expected = [(row["checkpoint"], row["input"]) for row in build_plan(policy)["conditions"]]
    if [(row.get("checkpoint"), row.get("input")) for row in result.get("conditions", [])] != expected:
        raise ValueError("incomplete or reordered condition matrix")
    identities = {}
    comparisons = {"bfl_fp32": set(), "bfl_bf16": {"bfl_fp32"}, "diffusers_fp32": {"bfl_fp32"},
                   "bf16_math": {"diffusers_fp32", "bfl_bf16"},
                   **{name: {"diffusers_fp32", "bf16_math"} for name in ("cudnn", "fa2", "fa3", "fa4")}}
    def check_metric(metric):
        if metric.get("count", 0) <= 0 or any(not math.isfinite(metric[key]) or metric[key] < 0
                for key in ("rms", "max_abs", "reference_l2", "actual_l2")):
            raise ValueError("missing or invalid numerical comparison")
        if metric["relative_l2"] is not None and (not math.isfinite(metric["relative_l2"]) or metric["relative_l2"] < 0):
            raise ValueError("invalid relative numerical comparison")
    for condition in result["conditions"]:
        models = condition.get("models", {})
        kernels = condition.get("kernels", {})
        if set(models) != set(MODEL_VARIANTS) or set(kernels) != set(BLOCKS):
            raise ValueError("incomplete model variants or kernel boundaries")
        identity = models["bfl_fp32"].get("input_fingerprints")
        prior = identities.setdefault(condition["input"], identity)
        if prior != identity:
            raise ValueError("base and overfit checkpoints received different inputs")
        for variant, record in models.items():
            if record["attention_calls"] != 25 or not record.get("input_fingerprints"):
                raise ValueError("invalid full-model execution evidence")
            if record["input_fingerprints"] != identity:
                raise ValueError("model variants received different inputs")
            if set(record.get("comparisons", {})) != comparisons[variant]:
                raise ValueError("incomplete model comparison references")
            for measured in record["comparisons"].values():
                check_metric(measured["velocity"])
                if set(measured["stages"]) != {"velocity/image", *(f"{stage}/{stream}" for stage in BLOCKS for stream in ("text", "image"))}:
                    raise ValueError("incomplete observed model boundaries")
                for metric in measured["stages"].values():
                    check_metric(metric)
        for record in kernels.values():
            if set(record["backends"]) != set(replay.BACKENDS) or not record.get("input_hashes_unchanged"):
                raise ValueError("incomplete backend replay evidence")
            for name, backend in record["backends"].items():
                promoted = name == "fp32_math"
                if (backend.get("input_fingerprints") != record["fp32_inputs" if promoted else "inputs"]
                        or backend.get("upstream_fingerprint") != record["fp32_upstream_gradient" if promoted else "upstream_gradient"]):
                    raise ValueError("kernel replays did not share identical inputs and upstream")
                for metric in ("output", "dQ", "dK", "dV"):
                    check_metric(backend[metric])
                    check_metric(backend["comparisons_vs_bf16_math"][metric])
    first = result["conditions"][0]["kernels"][BLOCKS[0]]["backends"]
    if any(not row.get("profiled") or not row.get("backend_verified") for row in first.values()):
        raise ValueError("missing measured backend dispatch evidence")
    return True


def numerical_screening(result, policy):
    """Summarize configured screening limits separately from execution success."""
    limits, flagged = policy.raw["thresholds"], []
    for condition in result["conditions"]:
        context = {key: condition[key] for key in ("checkpoint", "input")}
        for variant, row in condition["models"].items():
            threshold = limits["fp32_model_relative_l2" if variant == "diffusers_fp32" else "bf16_model_relative_l2"]
            for reference, measured in row["comparisons"].items():
                for boundary, metric in measured["stages"].items():
                    if metric["relative_l2"] is None or metric["relative_l2"] > threshold:
                        flagged.append(dict(**context, variant=variant, reference=reference,
                            metric=boundary, relative_l2=metric["relative_l2"], threshold=threshold))
        for stage, record in condition["kernels"].items():
            for backend, row in record["backends"].items():
                for reference, metrics in (("fp32_math", row), ("bf16_math", row["comparisons_vs_bf16_math"])):
                    for name in ("output", "dQ", "dK", "dV"):
                        threshold = limits["kernel_output_relative_l2" if name == "output" else "kernel_gradient_relative_l2"]
                        value = metrics[name]["relative_l2"]
                        if value is None or value > threshold:
                            flagged.append(dict(**context, stage=stage, backend=backend, reference=reference,
                                                metric=name, relative_l2=value, threshold=threshold))
    return dict(purpose="diagnostic screening only; not calibrated training acceptance",
                thresholds=limits, exceeded_count=len(flagged), exceeded=flagged)


def worker(args):
    policy = load_probe_config(args.config)
    fixture = preparation.load_fixture(policy.fixture_paths[0])
    result = dict(status="running", phase="preflight", plan=build_plan(policy), conditions=[], checkpoints={})

    def persist(phase):
        result["phase"] = phase
        worker_tools._write_json(args.output_dir / "result.json", result)
        print(f"{PROGRESS_MARKER} {phase} conditions={len(result['conditions'])}", flush=True)

    try:
        validate_source()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("this real-input comparison requires a BF16-capable CUDA GPU")
        device = torch.device("cuda", torch.cuda.current_device())
        seed_single_gpu_process(policy.raw["seed"])
        torch.set_num_threads(8)
        result["environment"] = worker_tools._environment_metadata()
        properties = torch.cuda.get_device_properties(device)
        result["environment"].update(gpu=properties.name, total_memory_bytes=properties.total_memory,
            compute_capability=[properties.major, properties.minor], visible_devices=torch.cuda.device_count(),
            flash_packages={name: version(name) for name in ("flash-attn", "flash-attn-3", "flash-attn-4")})
        for name, folder in (("base", policy.models["base_checkpoint"] / "transformer"),
                             ("overfit", policy.models["overfit_checkpoint"]),
                             ("qwen", policy.models["base_checkpoint"] / "text_encoder"),
                             ("nvarc", policy.models["nvarc_checkpoint"]),
                             ("vae", policy.models["base_checkpoint"] / "vae")):
            persist(f"verify_checkpoint/{name}")
            result["checkpoints"][name] = checkpoint_evidence(folder, expected=
                fixture["overfit"]["checkpoint_manifest"] if name == "overfit" else None)
        persist("prepare_native_qwen")
        native = preparation.prepare_native_from_checkpoint(fixture,
            base_checkpoint=policy.models["base_checkpoint"], device=device, seed=policy.raw["seed"])
        persist("prepare_real_nvarc_arc")
        arc = preparation.prepare_arc_from_checkpoints(fixture,
            nvarc_checkpoint=policy.models["nvarc_checkpoint"], base_checkpoint=policy.models["base_checkpoint"],
            device=device, seed=policy.raw["seed"], timesteps=tuple(policy.raw["timesteps"]))
        gc.collect()
        torch.cuda.empty_cache()
        prepared = {record.name: record for record in (native, *arc)}
        if len(prepared) != 7:
            raise ValueError("expected native prompt plus six ARC timestep inputs")
        result["prepared_inputs"] = {name: record.metadata for name, record in prepared.items()}
        profiled = False
        for checkpoint in ("base", "overfit"):
            persist(f"load_flux/{checkpoint}")
            folder = policy.models["base_checkpoint"] / "transformer" if checkpoint == "base" else policy.models["overfit_checkpoint"]
            model = df.Flux2Transformer2DModel.from_pretrained(folder, local_files_only=True,
                use_safetensors=True, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map={"": device})
            _validate_flux_checkpoint(model)
            model.eval().requires_grad_(False)
            bfl, mapping = build_bfl_model(model)
            result["checkpoints"][checkpoint]["mapping"] = mapping
            for item in result["plan"]["conditions"]:
                if item["checkpoint"] != checkpoint:
                    continue
                name = item["input"]
                inputs = assembled_inputs(prepared[name], device)
                condition = dict(checkpoint=checkpoint, input=name, input_metadata=prepared[name].metadata,
                                 models={}, kernels={})
                result["conditions"].append(condition)
                references = {}
                for variant in MODEL_VARIANTS:
                    persist(f"model/{checkpoint}/{name}/{variant}")
                    implementation = "bfl" if variant.startswith("bfl_") else "diffusers"
                    precision = "fp32" if variant.endswith("fp32") else "bf16"
                    backend = None if implementation == "bfl" else "native_math" if variant in ("diffusers_fp32", "bf16_math") else variant
                    row, values, qkv = model_case(bfl if implementation == "bfl" else model,
                        implementation, inputs, precision, backend,
                        capture_qkv=variant == "bf16_math")
                    row["comparisons"] = {key: compare_model(values, baseline, name)
                        for key, baseline in references.items() if key in (
                            "bfl_fp32" if variant in ("bfl_bf16", "diffusers_fp32") else "diffusers_fp32",
                            "bfl_bf16" if variant == "bf16_math" else "bf16_math")}
                    for reference, measured in row["comparisons"].items():
                        previous_loss = condition["models"][reference]["loss"]
                        measured["loss_absolute_difference"] = None if row["loss"] is None else abs(row["loss"] - previous_loss)
                        measured["loss_relative_difference"] = None if not previous_loss else measured["loss_absolute_difference"] / abs(previous_loss)
                    if variant in ("bfl_fp32", "bfl_bf16", "diffusers_fp32", "bf16_math"):
                        references[variant] = values
                    condition["models"][variant] = row
                    if variant == "bf16_math":
                        for index, (stage, tensors) in enumerate(qkv.items()):
                            persist(f"kernel/{checkpoint}/{name}/{stage}")
                            condition["kernels"][stage] = replay.compare_backends(*tensors,
                                seed=policy.raw["seed"] + index, profile=not profiled)
                            profiled = True
                    del qkv, values
                del references, inputs
                gc.collect()
                torch.cuda.empty_cache()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError("unexpected full-model parameter gradients")
            del model, bfl
            gc.collect()
            torch.cuda.empty_cache()
        validate_complete(result, policy)
        result["numerical_screening"] = numerical_screening(result, policy)
        result.update(status="complete", model_parameter_updates=0, model_backward_calls=0)
        persist("complete")
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        persist(result["phase"])
    return result


def write_report(path, result):
    lines = ["# Attention backends across input conditions", "", f"Execution: **{result['status']}**.", "",
             "FP32 parameters, unchanged projection/normalization/RoPE, singleton forwards. "
             "Kernel gradients are isolated Q/K/V vector-Jacobian products with one shared synthetic upstream, "
             "not model-loss gradients. Native prompt uses t=1 Gaussian latents and has no target loss.", "",
             "| Checkpoint | Input | Variant | Reference | Velocity rel. L2 | Velocity abs. RMS | Largest observed state rel. L2 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    def pct(value):
        return "undefined" if value is None else f"{100 * value:.6f}%"
    for condition in result.get("conditions", []):
        for variant, row in condition["models"].items():
            for reference, measured in row["comparisons"].items():
                lines.append(f"| {condition['checkpoint']} | {condition['input']} | {variant} | {reference} | "
                             f"{pct(measured['velocity']['relative_l2'])} | {measured['velocity']['rms']:.6g} | "
                             f"{pct(measured['maximum_observed_relative_l2'])} |")
    lines += ["", "Detailed kernel output/dQ/dK/dV comparisons, norms, cosine similarities, dtype and profiler "
              "evidence are retained in summary.json. Completion does not establish training acceptance, "
              "checkpointing correctness or a throughput advantage."]
    if "numerical_screening" in result:
        lines += ["", f"Configured screening limits were exceeded in {result['numerical_screening']['exceeded_count']} "
                  "comparisons. These limits are diagnostic flags, not calibrated training acceptance criteria."]
    if result.get("error"):
        lines += ["", f"Failure at {result.get('phase')}: {result['error']}"]
    path.write_text("\n".join(lines) + "\n")


def run_probe(args, *, metadata=None, on_progress=lambda: None):
    policy = load_probe_config(args.config)
    identity = source_metadata()
    metadata = identity if metadata is None else metadata
    checks = {"source_sha256": identity["source_sha256"], "config_sha256": config_hashes(args.config, policy),
              "dataset_sha256": dataset_hashes(policy)}
    if any(key in metadata and metadata[key] != value for key, value in checks.items()):
        raise ValueError("uploaded source, configuration or fixture differs from the launch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {**metadata, **checks, "config": policy.raw, "plan": build_plan(policy),
                "bfl_source": validate_source(), "effective_worker_seconds": policy.max_duration_seconds,
                "status": "running"}
    worker_tools._write_json(args.output_dir / "manifest.json", manifest)
    started = time.monotonic()
    result = dict(status="error", phase="worker_start", conditions=[])
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
            result.update(status="timeout", error="worker exceeded its configured time budget")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/diagnostics/flux_backend_conditions.json")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(build_plan(load_probe_config(args.config, require_checkpoints=False)), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    result = worker(args) if args.worker else run_probe(args)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
