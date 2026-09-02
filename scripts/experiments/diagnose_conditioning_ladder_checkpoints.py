#!/usr/bin/env python3
"""Diagnose saved conditioning-ladder checkpoints across the flow trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

from PIL import Image, ImageDraw
from safetensors import safe_open
import torch
from torch import Tensor

from diffusion_arc.arc_data import (
    ArcEvaluationTask,
    ArcQuery,
    ArcTokenizedTrainingSample,
    PAD_CLASS,
    TARGET_PALETTE_RGB,
    apply_training_augmentation,
    classify_arc_cells,
    decode_arc_grids,
    load_training_task,
    parse_augmentation_key,
    render_evaluation_prompts,
    render_grid_canvas,
    render_training_prompt,
    tokenize_nvarc_evaluation_sample,
    tokenize_nvarc_training_sample,
    validate_augmentation_for_task,
)
from diffusion_arc.inference import (
    continue_arc_batch,
    diagnose_exact_grids,
    linear_timestep_schedule,
)
from diffusion_arc.model import (
    TARGET_LATENT_CHANNELS,
    FluxSpatialShape,
    FluxTextInputs,
    build_flux_image_position_ids,
    pad_flux_text_prefixes,
    predict_flux_velocity,
)
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "single_gpu_0d3d703e_conditioning_ladder.json"
PARENT_RESULT_FAMILY = "single-task-conditioning-overfitting-ladder"
RESULT_FAMILY = "conditioning-ladder-checkpoint-diagnosis"
TASK_ID = "0d3d703e"
MAX_TOKENS = 192
SPATIAL_SHAPE = FluxSpatialShape(height=30, width=30)
STAGE_ORDER = (
    "one-condition",
    "two-condition-routing",
    "eight-condition-overfit",
)
LOCAL_PROBE_TIMESTEPS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
PARTIAL_ROLLOUT_TIMESTEPS = (0.5, 0.9, 0.99, 1.0)
FULL_GENERATION_SEEDS = (167, 168)
LOCAL_PROBE_SEED = 1167
PARTIAL_ROLLOUT_SEED = 3167
INFERENCE_STEPS = 50
MINIMUM_VRAM_GIB = 120.0


REPORT: dict[str, Any]
REPORT_PATH: Path
OUTPUT_DIR: Path
DEVICE: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--diagnosis-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-training-source-commit", required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_report() -> None:
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(REPORT, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT_PATH)


def cuda_memory(label: str) -> dict[str, Any]:
    torch.cuda.synchronize(DEVICE)
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated(DEVICE) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(DEVICE) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(DEVICE) / 2**30,
    }


def _record_failure(stage: str, error: BaseException) -> None:
    REPORT.setdefault("failures", []).append(
        {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    REPORT["status"] = "FAIL"
    REPORT["failed_stage"] = stage
    try:
        save_report()
    except (TypeError, ValueError):
        sanitized = _failure_json_safe(REPORT)
        if not isinstance(sanitized, dict):
            raise AssertionError("Sanitized failure report must remain a dictionary")
        REPORT.clear()
        REPORT.update(sanitized)
        save_report()


def _failure_json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {
            str(key): _failure_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_failure_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _validate_parent_report(
    report: dict[str, Any],
    *,
    training_run_id: str,
    expected_source_commit: str,
) -> None:
    if report.get("status") != "COMPLETE":
        raise RuntimeError("Parent conditioning-ladder report is not COMPLETE")
    if report.get("run_id") != training_run_id:
        raise RuntimeError("Parent report run_id differs from the selected run")
    if report.get("source", {}).get("resolved_commit") != expected_source_commit:
        raise RuntimeError("Parent training source commit differs from the expected commit")
    conditions = report.get("conditions")
    if not isinstance(conditions, dict):
        raise RuntimeError("Parent report has no condition registry")
    order = conditions.get("order")
    items = conditions.get("items")
    if not isinstance(order, list) or len(order) != 17 or len(set(order)) != 17:
        raise RuntimeError("Parent report must contain 17 unique conditions")
    if not isinstance(items, dict) or set(items) != set(order):
        raise RuntimeError("Parent condition items differ from their declared order")
    if "canonical" not in items:
        raise RuntimeError("Parent condition registry has no Original condition")
    protocol_stages = report.get("protocol", {}).get("stages")
    if not isinstance(protocol_stages, list):
        raise RuntimeError("Parent report has no stage protocol")
    stage_specs = {item.get("name"): item for item in protocol_stages}
    if tuple(stage_specs) != STAGE_ORDER:
        raise RuntimeError("Parent stage order differs from the diagnosis protocol")
    stages = report.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_ORDER):
        raise RuntimeError("Parent report stage registry differs")
    for stage_name in STAGE_ORDER:
        stage = stages[stage_name]
        checkpoint = stage.get("saved_checkpoint")
        if stage.get("status") != "COMPLETE" or not isinstance(checkpoint, dict):
            raise RuntimeError(f"Parent stage is incomplete: {stage_name}")
        if checkpoint.get("state_scope") != "transformer_only":
            raise RuntimeError(f"Checkpoint scope differs: {stage_name}")
        if checkpoint.get("step") != stage_specs[stage_name].get("optimizer_steps"):
            raise RuntimeError(f"Checkpoint step differs from stage protocol: {stage_name}")
        manifest = checkpoint.get("manifest")
        if not isinstance(manifest, dict) or not manifest:
            raise RuntimeError(f"Checkpoint manifest is missing: {stage_name}")


def _checkpoint_path(
    volume_root: Path,
    parent_report: dict[str, Any],
    stage_name: str,
) -> Path:
    checkpoint = parent_report["stages"][stage_name]["saved_checkpoint"]
    relative = checkpoint.get("volume_relative_path")
    if not isinstance(relative, str) or not relative.startswith("/"):
        raise RuntimeError(f"Checkpoint has no Volume-relative path: {stage_name}")
    path = (volume_root / relative.lstrip("/")).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {path}")
    return path


def _validate_checkpoint_directory(
    directory: Path,
    expected_manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    actual_files = {
        str(path.relative_to(directory)): path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    if set(actual_files) != set(expected_manifest):
        raise RuntimeError(
            f"Checkpoint file set differs: {set(actual_files) ^ set(expected_manifest)}"
        )
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    config_path = directory / "config.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("Checkpoint lacks config or Safetensors index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Safetensors index contains no weight map")
    shard_names = set(weight_map.values())
    if shard_names != {
        path.name for path in directory.glob("*.safetensors") if path.is_file()
    }:
        raise RuntimeError("Safetensors shard set differs from the index")
    observed_keys: set[str] = set()
    for shard_name in sorted(shard_names):
        with safe_open(directory / shard_name, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if observed_keys.intersection(keys):
            raise RuntimeError(f"Duplicate tensor keys in {shard_name}")
        observed_keys.update(keys)
    if observed_keys != set(weight_map):
        raise RuntimeError("Safetensors tensor keys differ from the index")

    observed_manifest = {}
    for relative, path in actual_files.items():
        expected = expected_manifest[relative]
        size = path.stat().st_size
        if size != expected.get("bytes"):
            raise RuntimeError(f"Checkpoint byte size differs: {relative}")
        digest = sha256_file(path)
        if digest != expected.get("sha256"):
            raise RuntimeError(f"Checkpoint SHA-256 differs: {relative}")
        observed_manifest[relative] = {"bytes": size, "sha256": digest}
        print(f"  verified {relative} ({size / 2**30:.2f} GiB)")
    return observed_manifest


def _json_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise TypeError("Serialized grid must be a non-empty list")
    return tuple(tuple(int(cell) for cell in row) for row in value)


def _pack_final_prefixes(
    samples: tuple[ArcTokenizedTrainingSample, ...],
) -> tuple[Tensor, Tensor]:
    flat_ids: list[int] = []
    offsets = [0]
    for sample in samples:
        if len(sample.target_grids) != 4:
            raise RuntimeError(f"Expected four eligible targets for {sample.key}")
        source_end = sample.prefix_token_ends[3]
        flat_ids.extend(sample.input_ids[:source_end])
        offsets.append(len(flat_ids))
    return (
        torch.tensor(flat_ids, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _build_condition_samples(
    *,
    parent_report: dict[str, Any],
    task: Any,
    tokenizer: Any,
) -> tuple[tuple[str, ...], tuple[ArcTokenizedTrainingSample, ...], tuple[Any, ...]]:
    if len(task.pairs) != 5:
        raise RuntimeError(f"Expected task {TASK_ID} to contain exactly five pairs")
    order = tuple(parent_report["conditions"]["order"])
    items = parent_report["conditions"]["items"]
    samples = []
    grids = []
    for name in order:
        item = items[name]
        key = parse_augmentation_key(item["key"])
        if key.task_id != TASK_ID:
            raise RuntimeError(f"Condition task id differs: {name}")
        validate_augmentation_for_task(key, task)
        augmented = apply_training_augmentation(task, key.augmentation)
        if key.augmentation.pair_indices[-1] != 4:
            raise RuntimeError(f"Condition does not target original pair 4: {name}")
        sample = tokenize_nvarc_training_sample(
            render_training_prompt(augmented),
            tokenizer,
            max_tokens=MAX_TOKENS,
        )
        query_pair = augmented.pairs[-1]
        evaluation_task = ArcEvaluationTask(
            task_id=str(key),
            demonstrations=augmented.pairs[:-1],
            queries=(ArcQuery(query_index=0, input_grid=query_pair.input_grid),),
        )
        evaluation_sample = tokenize_nvarc_evaluation_sample(
            render_evaluation_prompts(evaluation_task)[0],
            tokenizer,
            max_tokens=MAX_TOKENS,
        )
        if evaluation_sample.input_ids != sample.input_ids[: sample.prefix_token_ends[3]]:
            raise RuntimeError(f"Target-free NVARC prefix differs: {name}")
        expected_grid = _json_grid(item["target_grid"])
        if sample.target_grids[3] != expected_grid:
            raise RuntimeError(f"Reconstructed target differs from parent report: {name}")
        samples.append(sample)
        grids.append(expected_grid)
    return order, tuple(samples), tuple(grids)


def _slice_text_inputs(
    inputs: FluxTextInputs,
    indices: Tensor | slice,
) -> FluxTextInputs:
    return FluxTextInputs(
        encoder_hidden_states=inputs.encoder_hidden_states[indices],
        text_position_ids=inputs.text_position_ids[indices],
        text_attention_mask=inputs.text_attention_mask[indices],
    )


def _index_tensor(names: tuple[str, ...], condition_indices: dict[str, int]) -> Tensor:
    return torch.tensor(
        [condition_indices[name] for name in names],
        device=DEVICE,
        dtype=torch.int64,
    )


def _stage_groups(
    parent_report: dict[str, Any],
    stage_name: str,
) -> dict[str, tuple[str, ...]]:
    stage_specs = {
        item["name"]: item for item in parent_report["protocol"]["stages"]
    }
    seen = tuple(stage_specs[stage_name]["condition_names"])
    all_names = tuple(parent_report["conditions"]["order"])
    unseen = tuple(name for name in all_names if name.startswith("unseen-"))
    groups = {"seen": seen, "unseen": unseen}
    if "canonical" not in seen:
        groups["original"] = ("canonical",)
    flattened = tuple(name for names in groups.values() for name in names)
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"Diagnosis panel contains duplicate conditions: {stage_name}")
    return groups


def _alternate_condition_names(
    seen_names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(seen_names) == 1:
        return ("train-id" if seen_names[0] != "train-id" else "canonical",)
    return seen_names[1:] + seen_names[:1]


def _display_name(name: str) -> str:
    if name == "canonical":
        return "Original"
    if name.startswith("train-"):
        return f"Gesehen · {name.removeprefix('train-')}"
    if name.startswith("unseen-"):
        return f"Ungesehen · {name.removeprefix('unseen-')}"
    return name


def _target_class_canvas(grid: Any) -> Tensor:
    canvas = torch.full(
        (SPATIAL_SHAPE.height, SPATIAL_SHAPE.width),
        PAD_CLASS,
        dtype=torch.int64,
    )
    values = torch.tensor(grid, dtype=torch.int64)
    canvas[: values.shape[0], : values.shape[1]] = values
    return canvas


def _shared_standard_noise(
    *,
    batch_size: int,
    latent_shape: tuple[int, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(latent_shape) != 2 or any(dimension <= 0 for dimension in latent_shape):
        raise ValueError("latent_shape must contain two positive dimensions")
    noise = torch.randn(
        (1, *latent_shape),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    return noise.expand(batch_size, -1, -1).clone()


def _require_finite(label: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _encode_targets(grids: tuple[Any, ...], vae_encoder: Any) -> Tensor:
    chunks = []
    for start in range(0, len(grids), 16):
        images = torch.stack(
            tuple(
                render_grid_canvas(grid, canvas_size=SPATIAL_SHAPE.height)
                for grid in grids[start : start + 16]
            )
        )
        chunks.append(vae_encoder(images, spatial_shape=SPATIAL_SHAPE))
    return torch.cat(chunks, dim=0)


def _normalized_images_to_uint8(images: Tensor) -> Tensor:
    return (
        images.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(dtype=torch.uint8)
    )


@torch.no_grad()
def _decode_latents(latents: Tensor, vae_decoder: Any) -> dict[str, Any]:
    rgb_chunks = []
    for start in range(0, latents.shape[0], 8):
        decoded = vae_decoder(
            latents[start : start + 8],
            spatial_shape=SPATIAL_SHAPE,
        )
        rgb_chunks.append(_normalized_images_to_uint8(decoded))
    rgb_images = torch.cat(rgb_chunks, dim=0)
    classes = classify_arc_cells(
        rgb_images,
        canvas_size=SPATIAL_SHAPE.height,
        include_pad=True,
    )
    return {
        "rgb_images": rgb_images,
        "class_canvases": classes,
        "predictions": decode_arc_grids(
            classes,
            canvas_size=SPATIAL_SHAPE.height,
        ),
    }


def _diagnostic_payload(diagnostic: Any) -> dict[str, Any]:
    return asdict(diagnostic)


def _canvas_metrics(
    predicted_canvas: Tensor,
    expected_canvas: Tensor,
    expected_grid: Any,
) -> dict[str, Any]:
    predicted = predicted_canvas.to(device="cpu", dtype=torch.int64)
    expected = expected_canvas.to(device="cpu", dtype=torch.int64)
    if predicted.shape != expected.shape:
        raise ValueError("Predicted and expected class canvases must have equal shapes")
    height = len(expected_grid)
    width = len(expected_grid[0])
    grid_matches = predicted[:height, :width] == expected[:height, :width]
    padding_mask = torch.ones_like(expected, dtype=torch.bool)
    padding_mask[:height, :width] = False
    padding_matches = predicted[padding_mask] == expected[padding_mask]
    all_matches = predicted == expected
    return {
        "grid_cells_correct": int(grid_matches.sum().item()),
        "grid_cells_total": int(grid_matches.numel()),
        "grid_cell_accuracy": float(grid_matches.float().mean().item()),
        "padding_cells_correct": int(padding_matches.sum().item()),
        "padding_cells_total": int(padding_matches.numel()),
        "padding_cell_accuracy": float(padding_matches.float().mean().item()),
        "canvas_cells_correct": int(all_matches.sum().item()),
        "canvas_cells_total": int(all_matches.numel()),
        "canvas_cell_accuracy": float(all_matches.float().mean().item()),
        "exact_class_canvas": bool(all_matches.all().item()),
    }


def _prediction_records(
    *,
    predictions: tuple[Any, ...],
    class_canvases: Tensor,
    expected_grids: tuple[Any, ...],
    expected_canvases: Tensor,
    condition_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    diagnostics = diagnose_exact_grids(predictions, expected_grids)
    return [
        {
            "condition": name,
            "display_name": _display_name(name),
            "prediction": {
                "grid": prediction.grid,
                "height": prediction.height,
                "width": prediction.width,
                "valid": prediction.valid,
                "invalid_reason": prediction.invalid_reason,
            },
            "exact_grid": _diagnostic_payload(diagnostic),
            "regions": _canvas_metrics(
                class_canvas,
                expected_canvas,
                expected_grid,
            ),
        }
        for name, prediction, diagnostic, class_canvas, expected_canvas, expected_grid
        in zip(
            condition_names,
            predictions,
            diagnostics,
            class_canvases,
            expected_canvases,
            expected_grids,
            strict=True,
        )
    ]


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty prediction panel")
    return {
        "count": len(records),
        "exact_grids": sum(item["exact_grid"]["exact_grid"] for item in records),
        "valid_grids": sum(item["prediction"]["valid"] for item in records),
        "shape_matches": sum(item["exact_grid"]["shape_match"] for item in records),
        "mean_grid_cell_accuracy": sum(
            item["regions"]["grid_cell_accuracy"] for item in records
        )
        / len(records),
        "mean_padding_cell_accuracy": sum(
            item["regions"]["padding_cell_accuracy"] for item in records
        )
        / len(records),
        "mean_canvas_cell_accuracy": sum(
            item["regions"]["canvas_cell_accuracy"] for item in records
        )
        / len(records),
    }


def _region_flow_losses(
    squared_error: Tensor,
    expected_grids: tuple[Any, ...],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    spatial = squared_error.reshape(
        squared_error.shape[0],
        SPATIAL_SHAPE.height,
        SPATIAL_SHAPE.width,
        squared_error.shape[2],
    )
    per_condition = []
    for index, grid in enumerate(expected_grids):
        height = len(grid)
        width = len(grid[0])
        padding_mask = torch.ones(
            (SPATIAL_SHAPE.height, SPATIAL_SHAPE.width),
            device=spatial.device,
            dtype=torch.bool,
        )
        padding_mask[:height, :width] = False
        per_condition.append(
            {
                "total_loss": float(squared_error[index].mean().item()),
                "grid_loss": float(
                    spatial[index, :height, :width].mean().item()
                ),
                "padding_loss": float(
                    spatial[index, padding_mask].mean().item()
                ),
            }
        )
    return (
        {
            key: sum(item[key] for item in per_condition) / len(per_condition)
            for key in ("total_loss", "grid_loss", "padding_loss")
        },
        per_condition,
    )


def _class_image(class_canvas: Tensor, *, crop_shape: tuple[int, int] | None) -> Image.Image:
    values = class_canvas.detach().to(device="cpu", dtype=torch.int64)
    if crop_shape is not None:
        values = values[: crop_shape[0], : crop_shape[1]]
    rgb = TARGET_PALETTE_RGB[values].contiguous().numpy()
    return Image.fromarray(rgb)


def _raw_image(rgb_image: Tensor, *, crop_shape: tuple[int, int] | None) -> Image.Image:
    values = rgb_image.detach().to(device="cpu", dtype=torch.uint8)
    if crop_shape is not None:
        values = values[
            :,
            : crop_shape[0] * 16,
            : crop_shape[1] * 16,
        ]
    return Image.fromarray(values.permute(1, 2, 0).contiguous().numpy())


def _error_image(
    predicted_canvas: Tensor,
    expected_canvas: Tensor,
    *,
    crop_shape: tuple[int, int] | None,
) -> Image.Image:
    predicted = predicted_canvas.detach().to(device="cpu", dtype=torch.int64)
    expected = expected_canvas.detach().to(device="cpu", dtype=torch.int64)
    if crop_shape is not None:
        predicted = predicted[: crop_shape[0], : crop_shape[1]]
        expected = expected[: crop_shape[0], : crop_shape[1]]
    matches = predicted == expected
    rgb = torch.empty((*matches.shape, 3), dtype=torch.uint8)
    rgb[matches] = torch.tensor((212, 237, 218), dtype=torch.uint8)
    rgb[~matches] = torch.tensor((220, 53, 69), dtype=torch.uint8)
    return Image.fromarray(rgb.numpy())


def _fit_tile(image: Image.Image, size: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((size, size), Image.Resampling.NEAREST)
    background = Image.new("RGB", (size, size), "white")
    background.paste(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return background


def _save_comparison_sheet(
    *,
    path: Path,
    title: str,
    row_labels: tuple[str, ...],
    expected_grids: tuple[Any, ...],
    expected_canvases: Tensor,
    vae_class_canvases: Tensor,
    predicted_rgb_images: Tensor,
    predicted_class_canvases: Tensor,
    records: list[dict[str, Any]],
    canvas_view: bool,
) -> None:
    count = len(row_labels)
    if not (
        count
        == len(expected_grids)
        == expected_canvases.shape[0]
        == vae_class_canvases.shape[0]
        == predicted_rgb_images.shape[0]
        == predicted_class_canvases.shape[0]
        == len(records)
    ):
        raise ValueError("Comparison sheet inputs have different row counts")
    tile = 168
    label_width = 220
    header_height = 72
    row_height = tile + 58
    column_titles = (
        "Ground truth",
        "VAE round-trip",
        "Prediction raw",
        "Prediction cells",
        "Mismatch map",
    )
    sheet = Image.new(
        "RGB",
        (label_width + len(column_titles) * tile, header_height + count * row_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    view_label = "full 30x30 canvas" if canvas_view else "target grid crop"
    draw.text((8, 30), view_label, fill=(80, 80, 80))
    for column, column_title in enumerate(column_titles):
        draw.text((label_width + column * tile + 6, 48), column_title, fill="black")

    for row, (row_label, expected_grid, record) in enumerate(
        zip(row_labels, expected_grids, records, strict=True)
    ):
        y = header_height + row * row_height
        draw.text((8, y + 8), row_label[:36], fill="black")
        exact = "EXACT" if record["exact_grid"]["exact_grid"] else "not exact"
        grid_correct = record["regions"]["grid_cells_correct"]
        grid_total = record["regions"]["grid_cells_total"]
        pad_correct = record["regions"]["padding_cells_correct"]
        pad_total = record["regions"]["padding_cells_total"]
        draw.text((8, y + 32), f"{exact} · grid {grid_correct}/{grid_total}", fill="black")
        draw.text((8, y + 50), f"padding {pad_correct}/{pad_total}", fill=(80, 80, 80))
        crop_shape = None if canvas_view else (len(expected_grid), len(expected_grid[0]))
        images = (
            _class_image(expected_canvases[row], crop_shape=crop_shape),
            _class_image(vae_class_canvases[row], crop_shape=crop_shape),
            _raw_image(predicted_rgb_images[row], crop_shape=crop_shape),
            _class_image(predicted_class_canvases[row], crop_shape=crop_shape),
            _error_image(
                predicted_class_canvases[row],
                expected_canvases[row],
                crop_shape=crop_shape,
            ),
        )
        for column, image in enumerate(images):
            sheet.paste(_fit_tile(image, tile), (label_width + column * tile, y))
        draw.line((0, y + row_height - 1, sheet.width, y + row_height - 1), fill=(220, 220, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _relative_artifact(path: Path) -> str:
    return str(path.relative_to(OUTPUT_DIR))


def _save_prediction_sheets(
    *,
    directory: Path,
    stem: str,
    title: str,
    row_labels: tuple[str, ...],
    expected_grids: tuple[Any, ...],
    expected_canvases: Tensor,
    vae_class_canvases: Tensor,
    predicted_rgb_images: Tensor,
    predicted_class_canvases: Tensor,
    records: list[dict[str, Any]],
) -> dict[str, str]:
    grid_path = directory / f"{stem}-grid.png"
    canvas_path = directory / f"{stem}-canvas-30x30.png"
    common = {
        "title": title,
        "row_labels": row_labels,
        "expected_grids": expected_grids,
        "expected_canvases": expected_canvases,
        "vae_class_canvases": vae_class_canvases,
        "predicted_rgb_images": predicted_rgb_images,
        "predicted_class_canvases": predicted_class_canvases,
        "records": records,
    }
    _save_comparison_sheet(path=grid_path, canvas_view=False, **common)
    _save_comparison_sheet(path=canvas_path, canvas_view=True, **common)
    return {
        "grid_comparison": _relative_artifact(grid_path),
        "canvas_30x30_comparison": _relative_artifact(canvas_path),
    }


@torch.no_grad()
def _local_flow_probe(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage_name: str,
    groups: dict[str, tuple[str, ...]],
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    all_grids: tuple[Any, ...],
    all_target_canvases: Tensor,
    all_vae_classes: Tensor,
    condition_indices: dict[str, int],
    compute_dtype: torch.dtype,
    image_directory: Path,
) -> dict[str, Any]:
    previous_mode = transformer.training
    transformer.eval()
    group_results: dict[str, Any] = {}
    routing_curve = []
    try:
        for group_name, condition_names in groups.items():
            indices = _index_tensor(condition_names, condition_indices)
            clean_latents = all_latents.index_select(0, indices)
            text_inputs = _slice_text_inputs(all_text_inputs, indices)
            expected_grids = tuple(all_grids[index] for index in indices.tolist())
            expected_canvases = all_target_canvases.index_select(
                0, indices.to(device="cpu")
            )
            vae_classes = all_vae_classes.index_select(0, indices.to(device="cpu"))
            shared_noise = _shared_standard_noise(
                batch_size=len(condition_names),
                latent_shape=tuple(clean_latents.shape[1:]),
                seed=LOCAL_PROBE_SEED,
                device=DEVICE,
                dtype=clean_latents.dtype,
            )
            image_ids = build_flux_image_position_ids(
                len(condition_names),
                DEVICE,
                spatial_shape=SPATIAL_SHAPE,
            )
            assignment_specs = [("correct", text_inputs, condition_names)]
            if group_name == "seen":
                alternate_names = _alternate_condition_names(condition_names)
                alternate_indices = _index_tensor(alternate_names, condition_indices)
                assignment_specs.append(
                    (
                        "swapped",
                        _slice_text_inputs(all_text_inputs, alternate_indices),
                        alternate_names,
                    )
                )
            entries = []
            for timestep_value in LOCAL_PROBE_TIMESTEPS:
                print(
                    f"[{stage_name}/local] {group_name} "
                    f"t={timestep_value:.2f}"
                )
                timestep = torch.full(
                    (len(condition_names),),
                    timestep_value,
                    device=DEVICE,
                    dtype=torch.float32,
                )
                broadcast = timestep.to(dtype=clean_latents.dtype).reshape(-1, 1, 1)
                noisy_latents = (1.0 - broadcast) * clean_latents + broadcast * shared_noise
                target_velocity = shared_noise - clean_latents
                assignment_entries = []
                for assignment, assigned_text, context_names in assignment_specs:
                    predicted_velocity = predict_flux_velocity(
                        transformer,
                        noisy_latents,
                        assigned_text,
                        image_ids,
                        timestep,
                        compute_dtype=compute_dtype,
                    )
                    _require_finite(
                        f"{stage_name}/{group_name}/{assignment}/"
                        f"t={timestep_value:.2f} velocity",
                        predicted_velocity,
                    )
                    squared_error = (
                        predicted_velocity.float() - target_velocity.float()
                    ).square()
                    aggregate_loss, per_condition_loss = _region_flow_losses(
                        squared_error,
                        expected_grids,
                    )
                    reconstructed_latents = noisy_latents - broadcast * predicted_velocity
                    decoded = _decode_latents(reconstructed_latents, vae_decoder)
                    row_target_records = _prediction_records(
                        predictions=decoded["predictions"],
                        class_canvases=decoded["class_canvases"],
                        expected_grids=expected_grids,
                        expected_canvases=expected_canvases,
                        condition_names=condition_names,
                    )
                    context_indices = _index_tensor(context_names, condition_indices)
                    context_grids = tuple(
                        all_grids[index] for index in context_indices.tolist()
                    )
                    context_canvases = all_target_canvases.index_select(
                        0, context_indices.to(device="cpu")
                    )
                    context_records = _prediction_records(
                        predictions=decoded["predictions"],
                        class_canvases=decoded["class_canvases"],
                        expected_grids=context_grids,
                        expected_canvases=context_canvases,
                        condition_names=context_names,
                    )
                    entry = {
                        "assignment": assignment,
                        "context_condition_names": list(context_names),
                        "flow_loss_against_row_target": aggregate_loss,
                        "per_condition_flow_loss": {
                            name: value
                            for name, value in zip(
                                condition_names,
                                per_condition_loss,
                                strict=True,
                            )
                        },
                        "row_target_predictions": row_target_records,
                        "row_target_summary": _aggregate_records(row_target_records),
                        "context_target_predictions": context_records,
                        "context_target_summary": _aggregate_records(context_records),
                        "mean_absolute_z0_latent_error": float(
                            (reconstructed_latents - clean_latents)
                            .abs()
                            .mean()
                            .item()
                        ),
                    }
                    if assignment == "correct":
                        stem = f"local-z0-t-{timestep_value:.2f}".replace(".", "p")
                        entry["artifacts"] = _save_prediction_sheets(
                            directory=image_directory / "local_z0",
                            stem=stem,
                            title=(
                                f"{stage_name} · local z0 reconstruction · "
                                f"t={timestep_value:.2f}"
                            ),
                            row_labels=tuple(
                                f"{group_name}: {_display_name(name)}"
                                for name in condition_names
                            ),
                            expected_grids=expected_grids,
                            expected_canvases=expected_canvases,
                            vae_class_canvases=vae_classes,
                            predicted_rgb_images=decoded["rgb_images"],
                            predicted_class_canvases=decoded["class_canvases"],
                            records=row_target_records,
                        )
                    assignment_entries.append(entry)
                    del predicted_velocity, reconstructed_latents, decoded

                if group_name == "seen" and len(assignment_entries) == 2:
                    correct_loss = assignment_entries[0][
                        "flow_loss_against_row_target"
                    ]["grid_loss"]
                    swapped_loss = assignment_entries[1][
                        "flow_loss_against_row_target"
                    ]["grid_loss"]
                    routing_curve.append(
                        {
                            "timestep": timestep_value,
                            "correct_grid_loss": correct_loss,
                            "swapped_grid_loss": swapped_loss,
                            "correct_to_swapped_grid_loss_ratio": (
                                correct_loss / swapped_loss
                            ),
                            "swapped_context_exact_grids": assignment_entries[1][
                                "context_target_summary"
                            ]["exact_grids"],
                        }
                    )
                entries.append(
                    {
                        "timestep": timestep_value,
                        "assignments": assignment_entries,
                    }
                )
            group_results[group_name] = {
                "condition_names": list(condition_names),
                "shared_noise_across_condition_rows": True,
                "same_noise_across_timesteps": True,
                "same_noise_across_checkpoints": True,
                "identical_noisy_latents_across_rows_at_t1": True,
                "entries": entries,
            }
    finally:
        transformer.train(previous_mode)
    return {
        "timesteps": list(LOCAL_PROBE_TIMESTEPS),
        "groups": group_results,
        "seen_routing_curve": routing_curve,
        "z0_formula": "z0_hat = z_t - t * predicted_velocity",
    }


def _merge_generation_chunks(chunks: list[Any]) -> dict[str, Any]:
    return {
        "rgb_images": torch.cat([chunk.rgb_images for chunk in chunks], dim=0),
        "class_canvases": torch.cat(
            [chunk.class_canvases for chunk in chunks], dim=0
        ),
        "predictions": tuple(
            prediction for chunk in chunks for prediction in chunk.predictions
        ),
    }


@torch.no_grad()
def _full_generation(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage_name: str,
    groups: dict[str, tuple[str, ...]],
    all_text_inputs: FluxTextInputs,
    all_grids: tuple[Any, ...],
    all_target_canvases: Tensor,
    all_vae_classes: Tensor,
    condition_indices: dict[str, int],
    compute_dtype: torch.dtype,
    image_directory: Path,
) -> dict[str, Any]:
    panel_names = tuple(name for names in groups.values() for name in names)
    panel_indices = _index_tensor(panel_names, condition_indices)
    panel_grids = tuple(all_grids[index] for index in panel_indices.tolist())
    panel_canvases = all_target_canvases.index_select(
        0, panel_indices.to(device="cpu")
    )
    panel_vae_classes = all_vae_classes.index_select(
        0, panel_indices.to(device="cpu")
    )
    panel_text = _slice_text_inputs(all_text_inputs, panel_indices)
    group_offsets = {}
    offset = 0
    for group_name, names in groups.items():
        group_offsets[group_name] = (offset, offset + len(names))
        offset += len(names)
    by_seed = []
    schedule = linear_timestep_schedule(INFERENCE_STEPS, device=DEVICE)
    for seed in FULL_GENERATION_SEEDS:
        print(f"[{stage_name}/full] seed={seed}, 50 Euler intervals")
        shared_initial_noise = _shared_standard_noise(
            batch_size=len(panel_names),
            latent_shape=(SPATIAL_SHAPE.token_count, TARGET_LATENT_CHANNELS),
            seed=seed,
            device=DEVICE,
            dtype=torch.float32,
        )
        chunks = []
        for start in range(0, len(panel_names), 8):
            chunks.append(
                continue_arc_batch(
                    transformer,
                    vae_decoder,
                    _slice_text_inputs(panel_text, slice(start, start + 8)),
                    initial_latents=shared_initial_noise[start : start + 8],
                    timestep_schedule=schedule,
                    compute_dtype=compute_dtype,
                    spatial_shape=SPATIAL_SHAPE,
                    infer_padding_boundary=True,
                )
            )
            _require_finite(
                f"{stage_name}/full/seed={seed} final latents",
                chunks[-1].final_latents,
            )
        merged = _merge_generation_chunks(chunks)
        records = _prediction_records(
            predictions=merged["predictions"],
            class_canvases=merged["class_canvases"],
            expected_grids=panel_grids,
            expected_canvases=panel_canvases,
            condition_names=panel_names,
        )
        artifacts = _save_prediction_sheets(
            directory=image_directory / "full_generation",
            stem=f"seed-{seed}",
            title=f"{stage_name} · full generation · seed={seed} · 50 steps",
            row_labels=tuple(
                next(
                    f"{group}: {_display_name(name)}"
                    for group, names in groups.items()
                    if name in names
                )
                for name in panel_names
            ),
            expected_grids=panel_grids,
            expected_canvases=panel_canvases,
            vae_class_canvases=panel_vae_classes,
            predicted_rgb_images=merged["rgb_images"],
            predicted_class_canvases=merged["class_canvases"],
            records=records,
        )
        group_summaries = {
            group_name: _aggregate_records(records[start:end])
            for group_name, (start, end) in group_offsets.items()
        }
        by_seed.append(
            {
                "seed": seed,
                "shared_initial_noise_across_condition_rows": True,
                "same_initial_noise_across_checkpoints": True,
                "condition_names": list(panel_names),
                "predictions": records,
                "groups": group_summaries,
                "artifacts": artifacts,
            }
        )
        del chunks, merged
    pass2 = {}
    for group_name, (start, end) in group_offsets.items():
        pass2[group_name] = {
            "count": end - start,
            "exact_grid_pass1": sum(
                item["exact_grid"]["exact_grid"]
                for item in by_seed[0]["predictions"][start:end]
            ),
            "exact_grid_pass2": sum(
                left["exact_grid"]["exact_grid"]
                or right["exact_grid"]["exact_grid"]
                for left, right in zip(
                    by_seed[0]["predictions"][start:end],
                    by_seed[1]["predictions"][start:end],
                    strict=True,
                )
            ),
        }
    return {
        "inference_steps": INFERENCE_STEPS,
        "seeds": list(FULL_GENERATION_SEEDS),
        "by_seed": by_seed,
        "pass_summary": pass2,
    }


def _partial_timestep_schedule(start_timestep: float) -> Tensor:
    if not 0.0 < start_timestep <= 1.0:
        raise ValueError("start_timestep must lie in (0, 1]")
    intervals = max(1, math.ceil(INFERENCE_STEPS * start_timestep))
    return torch.linspace(
        start_timestep,
        0.0,
        steps=intervals + 1,
        device=DEVICE,
        dtype=torch.float32,
    )


@torch.no_grad()
def _partial_rollouts(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage_name: str,
    seen_names: tuple[str, ...],
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    all_grids: tuple[Any, ...],
    all_target_canvases: Tensor,
    all_vae_classes: Tensor,
    condition_indices: dict[str, int],
    compute_dtype: torch.dtype,
    image_directory: Path,
) -> dict[str, Any]:
    row_indices = _index_tensor(seen_names, condition_indices)
    row_latents = all_latents.index_select(0, row_indices)
    row_grids = tuple(all_grids[index] for index in row_indices.tolist())
    row_canvases = all_target_canvases.index_select(
        0, row_indices.to(device="cpu")
    )
    correct_text = _slice_text_inputs(all_text_inputs, row_indices)
    swapped_names = _alternate_condition_names(seen_names)
    swapped_indices = _index_tensor(swapped_names, condition_indices)
    swapped_text = _slice_text_inputs(all_text_inputs, swapped_indices)
    shared_noise = _shared_standard_noise(
        batch_size=len(seen_names),
        latent_shape=tuple(row_latents.shape[1:]),
        seed=PARTIAL_ROLLOUT_SEED,
        device=DEVICE,
        dtype=row_latents.dtype,
    )
    entries = []
    for start_timestep in PARTIAL_ROLLOUT_TIMESTEPS:
        schedule = _partial_timestep_schedule(start_timestep)
        print(
            f"[{stage_name}/partial] start t={start_timestep:.2f}, "
            f"{len(schedule) - 1} Euler intervals"
        )
        initial_latents = (
            (1.0 - start_timestep) * row_latents + start_timestep * shared_noise
        )
        assignments = []
        for assignment, context_names, text_inputs, context_indices in (
            ("correct", seen_names, correct_text, row_indices),
            ("swapped", swapped_names, swapped_text, swapped_indices),
        ):
            result = continue_arc_batch(
                transformer,
                vae_decoder,
                text_inputs,
                initial_latents=initial_latents,
                timestep_schedule=schedule,
                compute_dtype=compute_dtype,
                spatial_shape=SPATIAL_SHAPE,
                infer_padding_boundary=True,
            )
            _require_finite(
                f"{stage_name}/partial/t={start_timestep:.2f}/"
                f"{assignment} final latents",
                result.final_latents,
            )
            context_grids = tuple(
                all_grids[index] for index in context_indices.tolist()
            )
            context_canvases = all_target_canvases.index_select(
                0, context_indices.to(device="cpu")
            )
            context_vae_classes = all_vae_classes.index_select(
                0, context_indices.to(device="cpu")
            )
            row_records = _prediction_records(
                predictions=result.predictions,
                class_canvases=result.class_canvases,
                expected_grids=row_grids,
                expected_canvases=row_canvases,
                condition_names=seen_names,
            )
            context_records = _prediction_records(
                predictions=result.predictions,
                class_canvases=result.class_canvases,
                expected_grids=context_grids,
                expected_canvases=context_canvases,
                condition_names=context_names,
            )
            stem = (
                f"start-t-{start_timestep:.2f}-{assignment}".replace(".", "p")
            )
            artifacts = _save_prediction_sheets(
                directory=image_directory / "partial_rollouts",
                stem=stem,
                title=(
                    f"{stage_name} · partial rollout · start t={start_timestep:.2f} "
                    f"· {assignment} context · {len(schedule) - 1} steps"
                ),
                row_labels=tuple(
                    (
                        f"Start {_display_name(row_name)} → "
                        f"context {_display_name(context_name)}"
                    )
                    for row_name, context_name in zip(
                        seen_names,
                        context_names,
                        strict=True,
                    )
                ),
                expected_grids=context_grids,
                expected_canvases=context_canvases,
                vae_class_canvases=context_vae_classes,
                predicted_rgb_images=result.rgb_images,
                predicted_class_canvases=result.class_canvases,
                records=context_records,
            )
            assignments.append(
                {
                    "assignment": assignment,
                    "trajectory_kind": (
                        "teacher_forced_row_target_path"
                        if assignment == "correct"
                        else "hybrid_row_target_latent_with_swapped_context"
                    ),
                    "context_condition_names": list(context_names),
                    "row_target_predictions": row_records,
                    "row_target_summary": _aggregate_records(row_records),
                    "context_target_predictions": context_records,
                    "context_target_summary": _aggregate_records(context_records),
                    "artifacts": artifacts,
                }
            )
            del result
        entries.append(
            {
                "start_timestep": start_timestep,
                "euler_intervals": len(schedule) - 1,
                "maximum_step_size": float(
                    (schedule[:-1] - schedule[1:]).max().item()
                ),
                "assignments": assignments,
            }
        )
    return {
        "start_timesteps": list(PARTIAL_ROLLOUT_TIMESTEPS),
        "seed": PARTIAL_ROLLOUT_SEED,
        "condition_names": list(seen_names),
        "shared_noise_across_condition_rows": True,
        "same_noise_across_start_timesteps": True,
        "same_noise_across_checkpoints": True,
        "initial_state_uses_row_target_latent": True,
        "swapped_rollouts_below_t1_are_hybrid_counterfactuals": True,
        "row_target_signal_absent_at_t1": True,
        "initial_state_formula": "z_t = (1 - t) * z0 + t * epsilon",
        "entries": entries,
    }


def _load_transformer(
    checkpoint_path: Path,
    *,
    expected_parameter_count: int,
) -> Flux2Transformer2DModel:
    transformer = Flux2Transformer2DModel.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        device_map={"": DEVICE},
    )
    parameter_count = sum(parameter.numel() for parameter in transformer.parameters())
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Transformer has {parameter_count} parameters, "
            f"expected {expected_parameter_count}"
        )
    if not all(
        parameter.device == DEVICE and parameter.dtype == torch.float32
        for parameter in transformer.parameters()
    ):
        raise RuntimeError("Transformer placement or dtype differs")
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def _local_artifact_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != REPORT_PATH
    }


def _observed_remote_sizes(
    volume: Any,
    *,
    remote_directory: str,
    expected_paths: set[str],
) -> dict[str, int]:
    prefix = remote_directory.strip("/") + "/"
    observed = {}
    for entry in volume.iterdir(remote_directory, recursive=True):
        normalized = entry.path.lstrip("/")
        if normalized.startswith(prefix):
            relative = normalized[len(prefix) :]
            if relative in expected_paths:
                observed[relative] = entry.size
    return observed


def _upload_progress_snapshot(
    *,
    volume_name: str,
    diagnosis_id: str,
    label: str,
    artifact_directory: Path | None = None,
) -> None:
    import modal

    remote_directory = f"/results/{RESULT_FAMILY}/{diagnosis_id}"
    durability = REPORT.setdefault("durability", {})
    snapshots = durability.setdefault("progress_snapshots", [])
    if any(item["label"] == label for item in snapshots):
        raise RuntimeError(f"Duplicate progress snapshot label: {label}")
    snapshot = {
        "label": label,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_directory": (
            None
            if artifact_directory is None
            else str(artifact_directory.relative_to(OUTPUT_DIR))
        ),
        "artifact_file_count": 0,
        "remote_content_verified": False,
    }
    snapshots.append(snapshot)
    durability["remote_directory"] = remote_directory
    durability["remote_report_path"] = f"{remote_directory}/report.json"
    save_report()

    expected_sizes: dict[str, int] = {"report.json": REPORT_PATH.stat().st_size}
    artifact_files: list[Path] = []
    remote_artifact_directory = None
    if artifact_directory is not None:
        artifact_files = [
            path
            for path in sorted(artifact_directory.rglob("*"))
            if path.is_file()
        ]
        relative_root = artifact_directory.relative_to(OUTPUT_DIR)
        remote_artifact_directory = f"{remote_directory}/{relative_root}"
        expected_sizes.update(
            {
                str(relative_root / path.relative_to(artifact_directory)): (
                    path.stat().st_size
                )
                for path in artifact_files
            }
        )
    snapshot["artifact_file_count"] = len(artifact_files)
    save_report()
    expected_sizes["report.json"] = REPORT_PATH.stat().st_size

    volume = modal.Volume.from_name(volume_name)
    with volume.batch_upload(force=True) as batch:
        if artifact_files and remote_artifact_directory is not None:
            batch.put_directory(
                str(artifact_directory),
                remote_artifact_directory,
            )
        batch.put_file(str(REPORT_PATH), f"{remote_directory}/report.json")
    observed_sizes = _observed_remote_sizes(
        volume,
        remote_directory=remote_directory,
        expected_paths=set(expected_sizes),
    )
    if observed_sizes != expected_sizes:
        raise RuntimeError(f"Remote progress snapshot sizes differ: {label}")
    if b"".join(volume.read_file(f"{remote_directory}/report.json")) != REPORT_PATH.read_bytes():
        raise RuntimeError(f"Remote progress report differs: {label}")

    snapshot["remote_content_verified"] = True
    save_report()
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(REPORT_PATH), f"{remote_directory}/report.json")
    if b"".join(volume.read_file(f"{remote_directory}/report.json")) != REPORT_PATH.read_bytes():
        snapshot["remote_content_verified"] = False
        save_report()
        raise RuntimeError(f"Final remote progress report differs: {label}")


def _upload_complete_result(
    *,
    volume_name: str,
    diagnosis_id: str,
) -> None:
    import modal

    remote_directory = f"/results/{RESULT_FAMILY}/{diagnosis_id}"
    REPORT["artifacts"] = _local_artifact_manifest(OUTPUT_DIR)
    durability = REPORT.setdefault("durability", {})
    durability.update(
        {
            "remote_directory": remote_directory,
            "remote_report_path": f"{remote_directory}/report.json",
            "upload_method": "modal.Volume.batch_upload",
            "remote_file_sizes_verified": False,
            "remote_report_content_verified": False,
        }
    )
    save_report()
    volume = modal.Volume.from_name(volume_name)
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(OUTPUT_DIR), remote_directory)

    expected_sizes = {
        relative: details["bytes"]
        for relative, details in REPORT["artifacts"].items()
    }
    expected_sizes["report.json"] = REPORT_PATH.stat().st_size
    observed_sizes = _observed_remote_sizes(
        volume,
        remote_directory=remote_directory,
        expected_paths=set(expected_sizes),
    )
    if observed_sizes != expected_sizes:
        raise RuntimeError(
            "Remote diagnosis artifact sizes differ from local artifacts"
        )

    REPORT["durability"]["remote_file_sizes_verified"] = True
    REPORT["durability"]["remote_report_content_verified"] = True
    save_report()
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(REPORT_PATH), f"{remote_directory}/report.json")
    remote_bytes = b"".join(
        volume.read_file(f"{remote_directory}/report.json")
    )
    if remote_bytes != REPORT_PATH.read_bytes():
        REPORT["durability"]["remote_report_content_verified"] = False
        save_report()
        raise RuntimeError("Remote diagnosis report content differs")


def _upload_failure_report(
    *,
    volume_name: str,
    diagnosis_id: str,
) -> None:
    try:
        import modal

        remote_directory = f"/results/{RESULT_FAMILY}/{diagnosis_id}"
        volume = modal.Volume.from_name(volume_name)
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(OUTPUT_DIR), remote_directory)
    except Exception as upload_error:
        print(f"Failure report upload also failed: {upload_error}")


def run(args: argparse.Namespace) -> None:
    global DEVICE, OUTPUT_DIR, REPORT, REPORT_PATH

    volume_root = args.volume_root.resolve()
    parent_report_path = (
        volume_root
        / "results"
        / PARENT_RESULT_FAMILY
        / args.training_run_id
        / "report.json"
    )
    flux_checkpoint = volume_root / "flux2-klein-base-4b"
    nvarc_checkpoint = volume_root / "nvarc-qwen3-4b-grids15-sft139"
    arc_data_root = volume_root / "data" / "raw" / "arc-agi-2"
    required_paths = (
        parent_report_path,
        flux_checkpoint,
        nvarc_checkpoint,
        arc_data_root,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Volume artifacts are missing: {missing}")

    OUTPUT_DIR = (args.local_output_root / args.diagnosis_id).resolve()
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    REPORT_PATH = OUTPUT_DIR / "report.json"
    REPORT = {
        "schema_version": 1,
        "status": "INITIALIZING",
        "diagnosis_id": args.diagnosis_id,
        "question": (
            "Where along the flow trajectory do the three conditioning-ladder "
            "checkpoints fail, and how strongly does the NVARC context control "
            "their local velocity and partial denoising behavior?"
        ),
        "source": {"resolved_commit": args.source_commit},
        "parent_training_run": {
            "run_id": args.training_run_id,
            "expected_source_commit": args.expected_training_source_commit,
            "report_path": str(parent_report_path),
        },
        "protocol": {
            "task_id": TASK_ID,
            "canvas": asdict(SPATIAL_SHAPE),
            "checkpoint_stages": list(STAGE_ORDER),
            "local_probe_timesteps": list(LOCAL_PROBE_TIMESTEPS),
            "local_probe_noise_pairing": (
                "one shared noise tensor across conditions and checkpoints"
            ),
            "swapped_context_scope": "seen_group_only",
            "partial_rollout_timesteps": list(PARTIAL_ROLLOUT_TIMESTEPS),
            "partial_rollout_seed": PARTIAL_ROLLOUT_SEED,
            "full_generation_seeds": list(FULL_GENERATION_SEEDS),
            "full_generation_steps": INFERENCE_STEPS,
            "full_generation_noise_pairing": (
                "one shared initial noise tensor per seed across conditions "
                "and checkpoints"
            ),
            "local_and_partial_diagnostics_use_target_latents": True,
            "full_generation_is_target_free": True,
            "full_generation_ground_truth_used_only_after_prediction": True,
            "normal_generation_receives_no_ground_truth_dimensions": True,
        },
        "paths": {
            "local_output_directory": str(OUTPUT_DIR),
            "local_report": str(REPORT_PATH),
            "volume_root": str(volume_root),
        },
        "failures": [],
        "checkpoints": {
            stage_name: {"status": "PENDING"} for stage_name in STAGE_ORDER
        },
    }
    save_report()

    try:
        parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
        _validate_parent_report(
            parent_report,
            training_run_id=args.training_run_id,
            expected_source_commit=args.expected_training_source_commit,
        )
        REPORT["parent_training_run"]["resolved_source_commit"] = parent_report[
            "source"
        ]["resolved_commit"]
        REPORT["parent_training_run"]["vae_limitation"] = {
            "failed_condition_names": parent_report["vae_precompute"]["roundtrip"][
                "failed_condition_names"
            ],
            "exact_grids": parent_report["vae_precompute"]["roundtrip"][
                "exact_grids"
            ],
            "total_grids": parent_report["vae_precompute"]["roundtrip"][
                "total_grids"
            ],
        }
        save_report()

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        DEVICE = torch.device("cuda:0")
        torch.cuda.set_device(DEVICE)
        properties = torch.cuda.get_device_properties(DEVICE)
        vram_gib = properties.total_memory / 2**30
        if "H200" not in properties.name.upper() or vram_gib < MINIMUM_VRAM_GIB:
            raise RuntimeError(
                f"Expected H200 with >= {MINIMUM_VRAM_GIB:.0f} GiB, "
                f"received {properties.name} with {vram_gib:.1f} GiB"
            )
        REPORT["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_vram_gib": vram_gib,
            "compute_capability": list(torch.cuda.get_device_capability(DEVICE)),
        }

        checkpoint_paths = {
            stage_name: _checkpoint_path(volume_root, parent_report, stage_name)
            for stage_name in STAGE_ORDER
        }
        save_report()
        _upload_progress_snapshot(
            volume_name=args.volume_name,
            diagnosis_id=args.diagnosis_id,
            label="parent-report-and-runtime-validated",
        )
        for stage_name in STAGE_ORDER:
            print(f"[{stage_name}] validating checkpoint bytes and SHA-256 ...")
            started = time.perf_counter()
            expected_manifest = parent_report["stages"][stage_name][
                "saved_checkpoint"
            ]["manifest"]
            observed_manifest = _validate_checkpoint_directory(
                checkpoint_paths[stage_name],
                expected_manifest,
            )
            REPORT["checkpoints"][stage_name].update(
                {
                    "path": str(checkpoint_paths[stage_name]),
                    "parent_step": parent_report["stages"][stage_name][
                        "saved_checkpoint"
                    ]["step"],
                    "manifest": observed_manifest,
                    "manifest_matches_parent_report": True,
                    "validation_seconds": time.perf_counter() - started,
                    "status": "VALIDATED",
                }
            )
            save_report()
            _upload_progress_snapshot(
                volume_name=args.volume_name,
                diagnosis_id=args.diagnosis_id,
                label=f"{stage_name}-checkpoint-validated",
            )

        source_config = load_single_gpu_training_config(
            REPOSITORY_ROOT / "configs" / "training" / CONFIG_NAME
        )
        run_config = replace(
            source_config,
            models=replace(
                source_config.models,
                flux_checkpoint=flux_checkpoint,
                nvarc_checkpoint=nvarc_checkpoint,
            ),
        )
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        DEVICE = prepare_single_gpu_device(run_config)
        seed_single_gpu_process(run_config.seed)
        torch.cuda.empty_cache()

        first_stage = STAGE_ORDER[0]
        print(f"[{first_stage}] loading frozen NVARC/VAE and transformer ...")
        models = load_single_gpu_models(
            run_config,
            device=DEVICE,
            transformer_checkpoint=checkpoint_paths[first_stage],
        )
        tokenizer = models.tokenizer
        nvarc_runner = models.nvarc_runner
        vae_encoder = models.vae_encoder
        vae_decoder = models.vae_decoder
        transformer = models.transformer
        compute_dtype = models.flux_compute_dtype
        transformer.requires_grad_(False)
        transformer.eval()
        del models

        task = load_training_task(arc_data_root / "training" / f"{TASK_ID}.json")
        condition_order, samples, all_grids = _build_condition_samples(
            parent_report=parent_report,
            task=task,
            tokenizer=tokenizer,
        )
        condition_indices = {
            name: index for index, name in enumerate(condition_order)
        }
        flat_ids, offsets = _pack_final_prefixes(samples)
        print(f"[precompute] NVARC contexts: {len(condition_order)} conditions")
        nvarc_started = time.perf_counter()
        nvarc_output = nvarc_runner(flat_ids, offsets)
        if len(nvarc_output.sequence_hidden_states) != len(condition_order):
            raise RuntimeError("NVARC output count differs from condition count")
        all_text_inputs = pad_flux_text_prefixes(
            nvarc_output.sequence_hidden_states
        )
        _require_finite(
            "NVARC encoder hidden states",
            all_text_inputs.encoder_hidden_states,
        )
        nvarc_seconds = time.perf_counter() - nvarc_started
        del nvarc_output, nvarc_runner, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[precompute] VAE targets: {len(all_grids)} padded 30x30 grids")
        vae_started = time.perf_counter()
        all_latents = _encode_targets(all_grids, vae_encoder)
        _require_finite("VAE target latents", all_latents)
        decoded_targets = _decode_latents(all_latents, vae_decoder)
        all_target_canvases = torch.stack(
            tuple(_target_class_canvas(grid) for grid in all_grids)
        )
        roundtrip_records = _prediction_records(
            predictions=decoded_targets["predictions"],
            class_canvases=decoded_targets["class_canvases"],
            expected_grids=all_grids,
            expected_canvases=all_target_canvases,
            condition_names=condition_order,
        )
        observed_failed = [
            item["condition"]
            for item in roundtrip_records
            if not item["exact_grid"]["exact_grid"]
        ]
        expected_failed = parent_report["vae_precompute"]["roundtrip"][
            "failed_condition_names"
        ]
        if observed_failed != expected_failed:
            raise RuntimeError(
                "Current VAE round-trip failures differ from the parent run: "
                f"{observed_failed} versus {expected_failed}"
            )
        all_vae_classes = decoded_targets["class_canvases"]
        REPORT["precompute"] = {
            "condition_order": list(condition_order),
            "display_names": {
                name: _display_name(name) for name in condition_order
            },
            "nvarc_physical_batch_size": len(condition_order),
            "nvarc_seconds": nvarc_seconds,
            "nvarc_text_shape": list(
                all_text_inputs.encoder_hidden_states.shape
            ),
            "vae_seconds": time.perf_counter() - vae_started,
            "vae_latent_shape": list(all_latents.shape),
            "vae_roundtrip": {
                "records": roundtrip_records,
                "summary": _aggregate_records(roundtrip_records),
                "failed_condition_names": observed_failed,
                "matches_parent_report": True,
            },
        }
        del vae_encoder, decoded_targets
        gc.collect()
        torch.cuda.empty_cache()
        save_report()
        _upload_progress_snapshot(
            volume_name=args.volume_name,
            diagnosis_id=args.diagnosis_id,
            label="conditioning-and-vae-precompute-complete",
        )

        expected_parameter_count = sum(
            parameter.numel() for parameter in transformer.parameters()
        )
        for stage_index, stage_name in enumerate(STAGE_ORDER):
            if stage_index > 0:
                transformer = _load_transformer(
                    checkpoint_paths[stage_name],
                    expected_parameter_count=expected_parameter_count,
                )
            groups = _stage_groups(parent_report, stage_name)
            image_directory = OUTPUT_DIR / "images" / stage_name
            stage_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(DEVICE)
            REPORT["checkpoints"][stage_name].update(
                {
                    "status": "RUNNING",
                    "groups": {
                        group: list(names) for group, names in groups.items()
                    },
                }
            )
            save_report()

            print(f"[{stage_name}] dense local flow probes ...")
            local_probe = _local_flow_probe(
                transformer=transformer,
                vae_decoder=vae_decoder,
                stage_name=stage_name,
                groups=groups,
                all_text_inputs=all_text_inputs,
                all_latents=all_latents,
                all_grids=all_grids,
                all_target_canvases=all_target_canvases,
                all_vae_classes=all_vae_classes,
                condition_indices=condition_indices,
                compute_dtype=compute_dtype,
                image_directory=image_directory,
            )
            REPORT["checkpoints"][stage_name]["local_flow_probe"] = local_probe
            save_report()
            _upload_progress_snapshot(
                volume_name=args.volume_name,
                diagnosis_id=args.diagnosis_id,
                label=f"{stage_name}-local-flow-probe-complete",
                artifact_directory=image_directory,
            )

            print(f"[{stage_name}] partial target-path rollouts ...")
            partial = _partial_rollouts(
                transformer=transformer,
                vae_decoder=vae_decoder,
                stage_name=stage_name,
                seen_names=groups["seen"],
                all_text_inputs=all_text_inputs,
                all_latents=all_latents,
                all_grids=all_grids,
                all_target_canvases=all_target_canvases,
                all_vae_classes=all_vae_classes,
                condition_indices=condition_indices,
                compute_dtype=compute_dtype,
                image_directory=image_directory,
            )
            REPORT["checkpoints"][stage_name]["partial_rollouts"] = partial
            save_report()
            _upload_progress_snapshot(
                volume_name=args.volume_name,
                diagnosis_id=args.diagnosis_id,
                label=f"{stage_name}-partial-rollouts-complete",
                artifact_directory=image_directory,
            )

            print(f"[{stage_name}] normal 50-step generation ...")
            full_generation = _full_generation(
                transformer=transformer,
                vae_decoder=vae_decoder,
                stage_name=stage_name,
                groups=groups,
                all_text_inputs=all_text_inputs,
                all_grids=all_grids,
                all_target_canvases=all_target_canvases,
                all_vae_classes=all_vae_classes,
                condition_indices=condition_indices,
                compute_dtype=compute_dtype,
                image_directory=image_directory,
            )
            REPORT["checkpoints"][stage_name].update(
                {
                    "full_generation": full_generation,
                    "seconds": time.perf_counter() - stage_started,
                    "memory": cuda_memory(f"{stage_name}-complete"),
                    "status": "COMPLETE",
                }
            )
            save_report()
            _upload_progress_snapshot(
                volume_name=args.volume_name,
                diagnosis_id=args.diagnosis_id,
                label=f"{stage_name}-full-generation-complete",
                artifact_directory=image_directory,
            )
            print(
                f"[{stage_name}] complete in "
                f"{REPORT['checkpoints'][stage_name]['seconds']:.1f}s; "
                f"pass@2={full_generation['pass_summary']}"
            )
            del transformer
            gc.collect()
            torch.cuda.empty_cache()

        REPORT["summary"] = {
            stage_name: {
                "full_generation": REPORT["checkpoints"][stage_name][
                    "full_generation"
                ]["pass_summary"],
                "local_seen_routing_curve": REPORT["checkpoints"][stage_name][
                    "local_flow_probe"
                ]["seen_routing_curve"],
                "partial_context_target_exact_grids": {
                    str(entry["start_timestep"]): {
                        assignment["assignment"]: assignment[
                            "context_target_summary"
                        ]["exact_grids"]
                        for assignment in entry["assignments"]
                    }
                    for entry in REPORT["checkpoints"][stage_name][
                        "partial_rollouts"
                    ]["entries"]
                },
            }
            for stage_name in STAGE_ORDER
        }
        REPORT["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT["status"] = "COMPLETE"
        save_report()
        _upload_complete_result(
            volume_name=args.volume_name,
            diagnosis_id=args.diagnosis_id,
        )
        print(json.dumps(REPORT["summary"], indent=2))
        print(f"LOCAL_REPORT={REPORT_PATH}")
        print(
            "REMOTE_REPORT="
            f"{REPORT['durability']['remote_report_path']}"
        )
    except Exception as error:
        _record_failure("diagnosis", error)
        _upload_failure_report(
            volume_name=args.volume_name,
            diagnosis_id=args.diagnosis_id,
        )
        raise


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
