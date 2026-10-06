#!/usr/bin/env python3
"""Compare generation schedules and oracle reconstruction with the packed B3 path."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.cloud import diagnose_runpod as launcher
from scripts.cloud import train_runpod as pod


def diagnostic_variants(config: dict) -> list[tuple[str, float, int]]:
    """Identify each trajectory by spacing, starting time and forward count."""
    if "generation_steps" in config:
        return [("linear", 1.0, steps) for steps in config["generation_steps"]]
    variants = [("linear", t, round(t * config["steps_per_unit"])) for t in config["start_times"]]
    if "cosine_high_noise" in config.get("generation_schedules", []):
        variants.append(("cosine_high_noise", 1.0, config["steps_per_unit"]))
    return variants


def summarize(records: list[dict], queries: list[dict], config: dict) -> list[dict]:
    """Aggregate only a complete paired experiment, with explicit metric denominators."""
    keys = {(q["task_id"], q["query_index"]) for q in queries}
    expected = {(task, query, "clean_roundtrip", None, None, None, None) for task, query in keys}
    expected.update((task, query, kind, schedule, t, steps, seed) for task, query in keys
                    for kind in ("noisy_start", "denoised")
                    for schedule, t, steps in diagnostic_variants(config) for seed in config["sampling_seeds"])
    identities = [(r["task_id"], r["query_index"], r["kind"], r["schedule"], r["start_t"],
                   r["sampling_steps"], r["seed"]) for r in records]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("refusing to aggregate incomplete or duplicated diagnostic records")
    groups = defaultdict(list)
    for record in records:
        groups[record["kind"], record["schedule"], record["start_t"], record["sampling_steps"]].append(record)
    summaries = []
    for (kind, schedule, start_t, sampling_steps), rows in groups.items():
        cells = [row["cell_accuracy"] for row in rows if row["cell_accuracy"] is not None]
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["task_id"], row["seed"]].append(row["exact_grid"])
        summaries.append({
            "kind": kind, "schedule": schedule, "start_t": start_t, "query_samples": len(rows),
            "sampling_steps": sampling_steps,
            "exact_grids": sum(row["exact_grid"] for row in rows),
            "exact_grid_rate": sum(row["exact_grid"] for row in rows) / len(rows),
            "shape_accuracy": sum(row["shape_match"] for row in rows) / len(rows),
            "invalid_grid_rate": sum(not row["prediction_valid"] for row in rows) / len(rows),
            "cell_accuracy_on_correct_shapes": sum(cells) / len(cells) if cells else None,
            "cell_accuracy_query_samples": len(cells),
            "mean_latent_mse": sum(row["latent_mse"] for row in rows) / len(rows),
            "task_samples": len(tasks), "exact_tasks": sum(all(values) for values in tasks.values()),
            "task_exact_rate": sum(all(values) for values in tasks.values()) / len(tasks),
        })
        if kind != "clean_roundtrip":
            task_ids = {task for task, _ in tasks}
            summaries[-1]["task_count"] = len(task_ids)
            for attempts in range(1, len(config["sampling_seeds"]) + 1):
                solved = sum(any(all(tasks[task, seed]) for seed in config["sampling_seeds"][:attempts])
                             for task in task_ids)
                summaries[-1][f"solved_tasks_at_{attempts}"] = solved
                summaries[-1][f"pass_at_{attempts}"] = solved / len(task_ids)
        if kind == "denoised":
            summaries[-1].update(
                mean_initial_velocity_mse=sum(row["initial_v_mse"] for row in rows) / len(rows),
                mean_one_step_target_latent_mse=sum(row["one_step_z0_mse"] for row in rows) / len(rows),
            )
    return summaries


def render_comparisons(output: Path, records: list[dict], targets, inputs, config: dict) -> None:
    """Show full classified canvases, retaining wrong shapes and stray PAD cells."""
    from PIL import Image, ImageDraw, ImageFont
    from diffusion_arc.arc_data.image.palette import PAD_CLASS, TARGET_PALETTE_RGB

    palette = TARGET_PALETTE_RGB.tolist()
    rows = {(r["task_id"], r["query_index"], r["kind"], r["schedule"], r["start_t"],
             r["sampling_steps"], r["seed"]): r for r in records}
    variants = diagnostic_variants(config)
    destination = output / "comparisons"
    destination.mkdir(exist_ok=True)
    size, gap, header = 180, 14, 44
    column = size + gap
    row_height = size + header + gap
    font = ImageFont.load_default(size=13)

    def canvas(grid):
        return [list(values) + [PAD_CLASS] * (30 - len(values)) for values in grid] + [
            [PAD_CLASS] * 30 for _ in range(30 - len(grid))]

    for target in targets:
        key = (target.task_id, target.query_index)
        for seed in config["sampling_seeds"]:
            image = Image.new("RGB", (3 * column + gap, (len(variants) + 1) * row_height + gap), "#f3f4f6")
            draw = ImageDraw.Draw(image)

            def panel(col, row, title, cells, detail=""):
                x, y = gap + col * column, gap + row * row_height
                draw.text((x, y), title, fill="#111827", font=font)
                draw.text((x, y + 20), detail, fill="#4b5563", font=font)
                pixels = bytes(channel for values in cells for value in values for channel in palette[value])
                tile = Image.frombytes("RGB", (30, 30), pixels).resize((size, size), Image.Resampling.NEAREST)
                image.paste(tile, (x, y + header))

            clean = rows[(*key, "clean_roundtrip", None, None, None, None)]
            panel(0, 0, "Input", canvas(inputs[key]), f"{target.task_id} / query {target.query_index}")
            panel(1, 0, "Reference", canvas(target.expected_grid), f"seed {seed}")
            panel(2, 0, "Clean VAE roundtrip", clean["class_canvas"], "exact" if clean["exact_grid"] else "incorrect")
            for index, (schedule, t, steps) in enumerate(variants, start=1):
                before = rows[(*key, "noisy_start", schedule, t, steps, seed)]
                after = rows[(*key, "denoised", schedule, t, steps, seed)]
                y = gap + index * row_height
                draw.text((gap, y), f"Start t = {t:g}", fill="#111827", font=font)
                draw.text((gap, y + 22), f"{after['steps']} Euler steps", fill="#111827", font=font)
                draw.text((gap, y + 56), "Latent MSE:", fill="#4b5563", font=font)
                draw.text((gap, y + 78), f"{before['latent_mse']:.5g} -> {after['latent_mse']:.5g}", fill="#4b5563", font=font)
                draw.text((gap, y + 112), "Oracle diagnostic" if t < 1 else "Pure-noise baseline", fill="#4b5563", font=font)
                draw.text((gap, y + 136), "Linear" if schedule == "linear" else "Half cosine / high noise", fill="#4b5563", font=font)
                for col, title, record in ((1, "Noisy start (VAE)", before), (2, "After denoising", after)):
                    detail = ("exact" if record["exact_grid"] else "incorrect") if record["prediction_valid"] else "invalid grid"
                    panel(col, index, title, record["class_canvas"], detail)
            image.save(destination / f"{target.task_id}-q{target.query_index}-seed{seed}.png")


def run(config_path: Path, workspace: Path, output: Path) -> None:
    import torch
    from diffusion_arc.diagnostics.partial_denoising import diagnostic_schedules, iter_partial_denoising
    from diffusion_arc.training.setup import load_models, prepare_device, seed_process
    from diffusion_arc.training.run.validation import (
        load_sampling_input_grids, plan_validation_packs, prepare_validation_cache,
        tokenize_validation_targets, validate_validation_bundle,
    )

    config = launcher.load_config(config_path)
    started = time.monotonic()
    report = {
        "schema_version": 3, "status": "setting_up", "config": config,
        "command": sys.argv, "records_file": "records.jsonl", "panels_directory": "comparisons",
        "source": (json.loads((ROOT / "_source.json").read_text()) if (ROOT / "_source.json").is_file() else {
            "files": {str(path.relative_to(ROOT)): pod.sha256_file(path)
                      for directory in (ROOT / "src", ROOT / "scripts") for path in sorted(directory.rglob("*.py"))}}),
        "limitations": [
            "t < 1 uses oracle target latents; it is a recovery diagnostic, not direct-sampling benchmark accuracy.",
            "t = 1 uses target-independent query-keyed noise, identical across generation schedules.",
            "Cell accuracy is averaged only over predictions with correct shapes; its denominator is reported.",
            "Latent MSE averages the full 30x30 canvas uniformly, including padding; it is not region-weighted training loss.",
            "Task exact rate averages fixed seed attempts; a task requires every query to be correct under the same seed.",
            "Pass at k counts tasks solved by at least one of the first k fixed seeds, without mixing queries across seeds.",
            "The frozen NVARC checkpoint's prior exposure to these public validation tasks is unknown.",
        ],
    }
    pod.write_json(output / "report.json", report)
    try:
        launcher.prepare_job(config_path, workspace, output)
        prepared = json.loads((output / "prepared.json").read_text())
        report["preparation"] = prepared
        runtime = launcher.load_runtime(config_path, config)
        runtime = replace(runtime, models=replace(runtime.models, **{
            key: Path(value) for key, value in prepared["models"].items()}))
        report["runtime_config"] = runtime.recipe_dict()
        device = prepare_device(runtime)
        report["trajectories"] = [
            {"schedule": name, "start_t": start_t, "steps": len(times) - 1,
             "timesteps": times.tolist(), "solver": "Euler", "oracle_start": start_t < 1.0}
            for name, start_t, times in diagnostic_schedules(
                config.get("start_times", [1.0]), config.get("steps_per_unit"),
                config.get("generation_schedules", ["linear"]),
                generation_steps=config.get("generation_steps"), device=device)
        ]
        seed_process(runtime.seed)
        models = load_models(runtime, device=device, transformer_checkpoint=prepared["transformer"])
        models.flux_transformer.requires_grad_(False)
        torch.cuda.synchronize(device)
        ready_at = time.monotonic()
        pod.write_json(output / "models-ready.json", {"monotonic_seconds": ready_at})
        report["setup_seconds"] = ready_at - started
        report["environment"] = {
            "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda, "python": sys.version,
            "packages": {name: version(name) for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub")},
        }
        bundle = Path(prepared["bundle"])
        manifest = validate_validation_bundle(
            bundle, split_seed=config["split_seed"], verify_payloads=False,
        )
        pool_key = "validation_task_ids" if config.get("task_pool", "sampling") == "validation" else "sampling_task_ids"
        selected_ids = manifest[pool_key][:config["task_count"]]
        if len(selected_ids) != config["task_count"]:
            raise ValueError("selected task pool is smaller than the requested task_count")
        samples, grids = tokenize_validation_targets(bundle, manifest, models.tokenizer)
        selected = [(sample, grid) for sample, grid in zip(samples, grids, strict=True) if sample.task_id in selected_ids]
        targets = prepare_validation_cache(
            [pair[0] for pair in selected], [pair[1] for pair in selected],
            nvarc_encoder=models.nvarc_encoder, vae_encoder=models.vae_encoder,
            nvarc_max_tokens=runtime.nvarc_max_tokens, vae_batch_size=runtime.loop.vae_microbatch_size,
        )
        packs = plan_validation_packs(targets, max_joint_tokens=runtime.flux_packing.max_tokens)
        report.update(status="running", dataset_manifest=manifest, selected_task_ids=selected_ids,
                      queries=[{"task_id": target.task_id, "query_index": target.query_index} for target in targets],
                      packs=packs, cache_seconds=time.monotonic() - ready_at)
        report["planned_flux_forward_calls"] = len(packs) * len(config["sampling_seeds"]) * sum(
            item["steps"] for item in report["trajectories"])
        pod.write_json(output / "report.json", report)
        print(f"Models loaded; {len(selected_ids)} tasks / {len(targets)} queries, {len(packs)} packs. "
              f"Starting {len(report['trajectories'])} paired trajectories "
              f"({report['planned_flux_forward_calls']} transformer forwards).", flush=True)
        records = []
        with (output / "records.jsonl").open("x") as handle:
            for record in iter_partial_denoising(
                models.flux_transformer, models.vae_decoder, targets, packs,
                start_times=config.get("start_times", [1.0]), steps_per_unit=config.get("steps_per_unit"),
                seeds=config["sampling_seeds"], compute_dtype=models.flux_compute_dtype,
                backend=models.flux_packing_backend,
                generation_schedules=config.get("generation_schedules", ["linear"]),
                generation_steps=config.get("generation_steps"),
            ):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
                if record["kind"] == "denoised":
                    print(f"{record['task_id']}/q{record['query_index']} seed={record['seed']} "
                          f"schedule={record['schedule']} steps={record['sampling_steps']} "
                          f"t={record['start_t']:g}: exact={record['exact_grid']} "
                          f"latent_mse={record['latent_mse']:.6g}", flush=True)
        report["summaries"] = summarize(records, report["queries"], config)
        render_comparisons(output, records, targets, load_sampling_input_grids(bundle, selected_ids), config)
        torch.cuda.synchronize(device)
        report.update(status="succeeded", runtime_seconds=time.monotonic() - ready_at,
                      record_count=len(records), total_seconds=time.monotonic() - started)
        pod.write_json(output / "report.json", report)
        print(json.dumps(report["summaries"], indent=2), flush=True)
        print(f"Completed. Report: {output / 'report.json'}; comparisons: {output / 'comparisons'}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", total_seconds=time.monotonic() - started)
        pod.write_json(output / "report.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.workspace.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
