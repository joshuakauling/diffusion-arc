#!/usr/bin/env python3
"""Time the agreed validation round with resident B3 models and Adam states."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys
import time

import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY))

from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.setup import build_optimizer, load_models, prepare_device, seed_process
from diffusion_arc.training.run.validation import (
    prepare_validation_cache,
    tokenize_validation_targets as tokenize_targets,
    validate_validation_bundle as validate_bundle,
)


def load_config(path: Path):
    """Validate the bounded experiment before allocating model resources."""
    config = json.loads(path.read_text())
    expected = {
        "schema_version": 1, "timesteps": [.1, .3, .5, .7, .9, 1.],
        "sampling_steps": 40, "warmup_repetitions": 1, "measured_repetitions": 2,
        "cache_device": "cpu", "resident_optimizer_state": True,
    }
    fields = {*expected, "training_config", "bundle", "loss_seed", "sampling_seeds"}
    if set(config) != fields or any(type(config.get(k)) is not type(v) or config.get(k) != v
                                   for k, v in expected.items()):
        raise ValueError("configuration must describe the agreed bounded validation benchmark")
    seeds = [config["loss_seed"], *config["sampling_seeds"]]
    if len(config["sampling_seeds"]) != 2 or len(set(config["sampling_seeds"])) != 2:
        raise ValueError("sampling requires two distinct seeds")
    if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("seeds must be integers in [0, 2**63)")
    runtime = load_training_config((path.parent / config["training_config"]).resolve(), expected_type="baseline")
    if runtime.flux_packing is None or runtime.flux_packing.max_tokens != 12288:
        raise ValueError("the benchmark requires B3 packing with 12288 joint tokens")
    if runtime.nvarc_max_tokens != 134400 or runtime.loop.vae_microbatch_size != 48:
        raise ValueError("the benchmark requires a 134400-token NVARC budget and VAE batches of 48")
    if (runtime.models.nvarc_dtype, runtime.models.vae_dtype,
            runtime.models.flux_parameter_dtype, runtime.models.flux_compute_dtype) != (
                "bfloat16", "float32", "float32", "bfloat16"):
        raise ValueError("the benchmark requires the B3 precision configuration")
    return config, runtime, (path.parent / config["bundle"]).resolve()


def allocate_adam_state(optimizer):
    """Materialize ordinary AdamW's FP32 moments without executing an update."""
    for group in optimizer.param_groups:
        if group.get("amsgrad") or group.get("capturable") or group.get("fused"):
            raise ValueError("resident-state benchmark expects ordinary non-capturable AdamW")
        for parameter in group["params"]:
            if optimizer.state[parameter]:
                raise ValueError("optimizer must be freshly constructed")
            optimizer.state[parameter] = {
                "step": torch.tensor(0., dtype=torch.float32),
                "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format),
            }


def checkpoint_identity(runtime):
    """Hash the actual loaded components, including NVARC tokenizer metadata."""
    components = {
        "nvarc": runtime.models.nvarc_checkpoint,
        "flux_transformer": runtime.models.flux_checkpoint / "transformer",
        "vae": runtime.models.flux_checkpoint / "vae",
    }
    result = {}
    for name, directory in components.items():
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and path.suffix in (".safetensors", ".json", ".txt"))
        if not any(path.suffix == ".safetensors" for path in files):
            raise FileNotFoundError(f"missing local Safetensors component: {directory}")
        hashes = {}
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        result[name] = {"directory": str(directory), "files": hashes}
    return result


def timed(name, operation, device, rows):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    row = {
        "phase": name, "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "end_allocated_bytes": torch.cuda.memory_allocated(device),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    return result


def summarize(rows):
    """Report paired full-round durations, without treating a target as a batch."""
    loss = [row["seconds"] for row in rows if row["phase"].startswith("measure/loss/")]
    sampling = [row["seconds"] for row in rows if row["phase"].startswith("measure/sampling/")]
    if not loss or len(loss) != len(sampling):
        raise ValueError("paired measured loss/sampling repetitions are required")
    return {
        "loss_seconds": statistics.median(loss),
        "sampling_seconds": statistics.median(sampling),
        "validation_round_seconds": statistics.median([a + b for a, b in zip(loss, sampling, strict=True)]),
        "loss_range_seconds": [min(loss), max(loss)],
        "sampling_range_seconds": [min(sampling), max(sampling)],
        "measured_repetitions": len(loss),
    }


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(config_path: Path, output: Path):
    from diffusion_arc.training.run.validation import evaluate_packed_loss, generate_packed_samples, plan_validation_packs
    config, runtime, bundle = load_config(config_path)
    manifest = validate_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {
        "status": "running", "config": config, "runtime_config": asdict(runtime),
        "dataset_manifest": manifest, "timings": rows, "evaluations": [],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("src", "scripts") for path in sorted((REPOSITORY / directory).rglob("*.py"))
        },
        "limitations": [
            "Fresh base transformer measures runtime, not trained-model quality.",
            "Adam moments are resident zero buffers; no optimizer update is performed.",
            "CPU-cache staging, packing, VAE decoding and metric computation are included in recurring timings.",
            "Model loading, CPU-cache preparation, warmup and JSON artifact writes are separate from recurring timings.",
            "The selected public tasks are not established unseen data for the frozen NVARC checkpoint.",
        ],
    }
    # Path-valued runtime settings must be portable in the resulting JSON artifact.
    report["runtime_config"] = json.loads(json.dumps(report["runtime_config"], default=str))
    try:
        device = prepare_device(runtime)
        seed_process(runtime.seed)
        report["environment"] = {
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": sys.version,
            "packages": {name: version(name) for name in ("diffusers", "transformers", "accelerate", "safetensors")},
        }
        report["checkpoints"] = timed("setup/checkpoint_identity", lambda: checkpoint_identity(runtime), device, rows)
        models = timed("setup/models", lambda: load_models(runtime, device=device), device, rows)
        optimizer = build_optimizer(runtime, flux_transformer=models.flux_transformer)
        timed("setup/adam_state", lambda: allocate_adam_state(optimizer), device, rows)
        samples, grids = timed("setup/tokenization", lambda: tokenize_targets(bundle, manifest, models.tokenizer), device, rows)
        report["queries"] = [
            {"task_id": s.task_id, "query_index": s.query_index, "prefix_tokens": len(s.input_ids),
             "grid_height": len(g), "grid_width": len(g[0])} for s, g in zip(samples, grids, strict=True)
        ]
        cache = timed("setup/nvarc_vae_cpu_cache", lambda: prepare_validation_cache(
            samples, grids, nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            nvarc_max_tokens=runtime.nvarc_max_tokens, vae_batch_size=runtime.loop.vae_microbatch_size,
        ), device, rows)
        sampling = tuple(target for target in cache if target.task_id in set(manifest["sampling_task_ids"]))
        packs = plan_validation_packs(cache, max_joint_tokens=runtime.flux_packing.max_tokens)
        sampling_packs = plan_validation_packs(sampling, max_joint_tokens=runtime.flux_packing.max_tokens)
        report["workload"] = {
            "loss_tasks": len({t.task_id for t in cache}), "loss_queries": len(cache),
            "sampling_tasks": len({t.task_id for t in sampling}), "sampling_queries": len(sampling),
            "loss_packs": packs, "sampling_packs": sampling_packs,
            "loss_target_forwards": len(cache) * len(config["timesteps"]),
            "sampling_target_forwards": len(sampling) * len(config["sampling_seeds"]) * config["sampling_steps"],
            "cpu_cache_bytes": sum(t.text.numel() * t.text.element_size() +
                                   t.clean_latent.numel() * t.clean_latent.element_size() for t in cache),
        }
        write_report(output / "report.json", report)
        for repetition in range(config["warmup_repetitions"] + config["measured_repetitions"]):
            phase = "warmup" if repetition < config["warmup_repetitions"] else "measure"
            loss = timed(f"{phase}/loss/{repetition}", lambda: evaluate_packed_loss(
                models.flux_transformer, cache, packs, timesteps=config["timesteps"], seed=config["loss_seed"],
                compute_dtype=models.flux_compute_dtype, backend=models.flux_packing_backend), device, rows)
            generated = timed(f"{phase}/sampling/{repetition}", lambda: generate_packed_samples(
                models.flux_transformer, models.vae_decoder, sampling, sampling_packs, steps=config["sampling_steps"],
                seeds=config["sampling_seeds"], compute_dtype=models.flux_compute_dtype,
                backend=models.flux_packing_backend), device, rows)
            report["evaluations"].append({"phase": phase, "repetition": repetition, "loss": loss, "sampling": generated})
            write_report(output / "report.json", report)
        report["summary"] = summarize(rows)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_report(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / "configs/validation/benchmark_b3.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        _, _, bundle = load_config(args.config.resolve())
        manifest = validate_bundle(bundle)
        print(json.dumps({"status": "bundle_valid", "validation_outputs": manifest["validation_output_count"],
                          "sampling_outputs": manifest["sampling_output_count"]}))
    elif args.output is None:
        parser.error("--output is required for a measurement")
    else:
        run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
