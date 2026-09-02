#!/usr/bin/env python3
"""Run the paired 3x3 versus 30x30 single-task Modal ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm

from diffusion_arc.arc_data import (
    ArcEvaluationTask,
    ArcQuery,
    apply_training_augmentation,
    classify_arc_cells,
    decode_arc_grids,
    decode_fixed_arc_grids,
    load_training_task,
    render_evaluation_prompts,
    render_grid_canvas,
    tokenize_nvarc_evaluation_sample,
)
from diffusion_arc.inference import diagnose_exact_grids, generate_arc_batch
from diffusion_arc.model import (
    FluxSpatialShape,
    FluxTextInputs,
    build_flux_image_position_ids,
    flow_matching_loss,
    pad_flux_text_prefixes,
    predict_flux_velocity,
    sample_flow_matching_batch,
)
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from diffusion_arc.training import ArcTrainingDataset, FluxMicrobatch
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    build_flux_optimizer,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAME = "overfitting_0d3d703e_final_target"
CONFIG_NAME = "single_gpu_0d3d703e_final_target.json"
TASK_ID = "0d3d703e"
TRAIN_SEQUENCE_COUNT = 512
SEEN_PANEL_COUNT = 8
UNSEEN_PANEL_COUNT = 8
PANEL_COUNT = 17
OPTIMIZER_BATCH_SIZE = 8
OPTIMIZER_STEPS = 64
EVALUATION_STEPS = (0, 16, 32, 64)
GENERATION_SEEDS = (167, 168)
PROBE_SEED = 1167
FLOW_SEED = 2167
MAX_GRAD_NORM = 1.0
INFERENCE_STEPS = 50
CHECKPOINT_MAX_SHARD_SIZE = "2GB"
MINIMUM_VRAM_GIB = 120.0
MINIMUM_LOCAL_CHECKPOINT_FREE_GIB = 24.0
VARIANT_ORDER = ("compact-3x3", "padded-30x30")
SPATIAL_SHAPES = {
    "compact-3x3": FluxSpatialShape(height=3, width=3),
    "padded-30x30": FluxSpatialShape(height=30, width=30),
}
VAE_BATCH_SIZES = {"compact-3x3": 512, "padded-30x30": 16}
GROUP_SLICES = (
    ("seen", slice(0, 8)),
    ("unseen", slice(8, 16)),
    ("canonical", slice(16, 17)),
)


REPORT: dict[str, Any]
REPORT_PATH: Path
REPORT_DIR: Path
DEVICE: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuda_memory(label: str) -> dict[str, Any]:
    torch.cuda.synchronize(DEVICE)
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated(DEVICE) / 2**30,
        "reserved_gib": torch.cuda.memory_reserved(DEVICE) / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(DEVICE) / 2**30,
    }


def save_report() -> None:
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(REPORT, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT_PATH)


def record_failure(stage: str, error: BaseException) -> None:
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
    save_report()


def _slice_text_inputs(
    inputs: FluxTextInputs,
    indices: Tensor | slice,
) -> FluxTextInputs:
    return FluxTextInputs(
        encoder_hidden_states=inputs.encoder_hidden_states[indices],
        text_position_ids=inputs.text_position_ids[indices],
        text_attention_mask=inputs.text_attention_mask[indices],
    )


def _permute_text_inputs(inputs: FluxTextInputs) -> FluxTextInputs:
    permutation = torch.roll(
        torch.arange(
            inputs.encoder_hidden_states.shape[0],
            device=inputs.encoder_hidden_states.device,
        ),
        shifts=1,
    )
    return _slice_text_inputs(inputs, permutation)


def _pack_final_prefixes(samples: tuple[Any, ...]) -> tuple[Tensor, Tensor]:
    flat_ids: list[int] = []
    offsets = [0]
    for sample in samples:
        if len(sample.target_grids) != 4:
            raise RuntimeError(
                f"Expected four eligible targets for {sample.key}, "
                f"received {len(sample.target_grids)}"
            )
        source_end = sample.prefix_token_ends[3]
        flat_ids.extend(sample.input_ids[:source_end])
        offsets.append(len(flat_ids))
    return (
        torch.tensor(flat_ids, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _assert_target_free_prefix(
    dataset: ArcTrainingDataset,
    sample: Any,
    tokenizer: Any,
) -> None:
    key = sample.key
    raw_task = load_training_task(dataset.source_directory / f"{key.task_id}.json")
    augmented = apply_training_augmentation(raw_task, key.augmentation)
    query_pair = augmented.pairs[-1]
    evaluation_task = ArcEvaluationTask(
        task_id=str(key),
        demonstrations=augmented.pairs[:-1],
        queries=(ArcQuery(query_index=0, input_grid=query_pair.input_grid),),
    )
    rendered = render_evaluation_prompts(evaluation_task)[0]
    tokenized = tokenize_nvarc_evaluation_sample(
        rendered,
        tokenizer,
        max_tokens=dataset.max_tokens,
    )
    training_prefix = sample.input_ids[: sample.prefix_token_ends[3]]
    if tokenized.input_ids != training_prefix:
        raise RuntimeError(f"Target-free prefix differs for {key}")
    if sample.target_grids[3] != query_pair.output_grid:
        raise RuntimeError(f"Final target differs for {key}")


def _encode_target_grids(
    grids: tuple[Any, ...],
    *,
    variant_name: str,
    vae_encoder: Any,
) -> Tensor:
    spatial_shape = SPATIAL_SHAPES[variant_name]
    batch_size = VAE_BATCH_SIZES[variant_name]
    chunks = []
    progress = tqdm(
        range(0, len(grids), batch_size),
        desc=f"VAE encode {variant_name}",
        unit="batch",
    )
    for start in progress:
        images = torch.stack(
            tuple(
                render_grid_canvas(
                    grid,
                    canvas_size=spatial_shape.height,
                )
                for grid in grids[start : start + batch_size]
            )
        )
        chunks.append(
            vae_encoder(images, spatial_shape=spatial_shape)
        )
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


def _decode_images(
    rgb_images: Tensor,
    *,
    variant_name: str,
) -> tuple[Tensor, tuple[Any, ...]]:
    spatial_shape = SPATIAL_SHAPES[variant_name]
    compact = variant_name == "compact-3x3"
    classes = classify_arc_cells(
        rgb_images,
        canvas_size=spatial_shape.height,
        include_pad=not compact,
    )
    if compact:
        predictions = decode_fixed_arc_grids(
            classes,
            canvas_size=spatial_shape.height,
        )
    else:
        predictions = decode_arc_grids(
            classes,
            canvas_size=spatial_shape.height,
        )
    return classes, predictions


def _vae_roundtrip(
    *,
    variant_name: str,
    latents: Tensor,
    grids: tuple[Any, ...],
    vae_decoder: Any,
) -> dict[str, Any]:
    spatial_shape = SPATIAL_SHAPES[variant_name]
    predictions = []
    for start in range(0, len(grids), 8):
        decoded = vae_decoder(
            latents[start : start + 8],
            spatial_shape=spatial_shape,
        )
        _, batch_predictions = _decode_images(
            _normalized_images_to_uint8(decoded),
            variant_name=variant_name,
        )
        predictions.extend(batch_predictions)
    diagnostics = diagnose_exact_grids(tuple(predictions), grids)
    exact = sum(item.exact_grid for item in diagnostics)
    if exact != len(grids):
        raise RuntimeError(
            f"{variant_name} clean VAE round trip is {exact}/{len(grids)}"
        )
    return {"exact_grids": exact, "total_grids": len(grids)}


def _transformer_fingerprint(transformer: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    parameters = tuple(transformer.named_parameters())
    selected = parameters[:2] + parameters[-2:]
    for name, parameter in selected:
        flat = parameter.detach().reshape(-1)
        sample = torch.cat((flat[:256], flat[-256:])).to(
            device="cpu",
            dtype=torch.float32,
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


def _load_fresh_transformer(
    flux_checkpoint: Path,
    *,
    expected_parameter_count: int,
) -> Flux2Transformer2DModel:
    transformer = Flux2Transformer2DModel.from_pretrained(
        flux_checkpoint,
        subfolder="transformer",
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        device_map={"": DEVICE},
    )
    transformer.requires_grad_(True)
    parameter_count = sum(parameter.numel() for parameter in transformer.parameters())
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Fresh transformer has {parameter_count} parameters, "
            f"expected {expected_parameter_count}"
        )
    if not all(
        parameter.device == DEVICE and parameter.dtype == torch.float32
        for parameter in transformer.parameters()
    ):
        raise RuntimeError("Fresh transformer placement or dtype differs")
    return transformer


def _optimizer_step(
    *,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    text_inputs: FluxTextInputs,
    clean_latents: Tensor,
    spatial_shape: FluxSpatialShape,
    step: int,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    image_ids = build_flux_image_position_ids(
        clean_latents.shape[0],
        clean_latents.device,
        spatial_shape=spatial_shape,
    )
    microbatch = FluxMicrobatch(
        text_inputs=text_inputs,
        clean_latents=clean_latents,
        image_position_ids=image_ids,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(FLOW_SEED + step)
    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    flow_batch = sample_flow_matching_batch(
        microbatch.clean_latents,
        generator=generator,
    )
    predicted_velocity = predict_flux_velocity(
        transformer,
        flow_batch.noisy_latents,
        microbatch.text_inputs,
        microbatch.image_position_ids,
        flow_batch.timesteps,
        compute_dtype=compute_dtype,
    )
    loss = flow_matching_loss(
        predicted_velocity,
        flow_batch.target_velocity.to(device=predicted_velocity.device),
    )
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("Training loss is non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        tuple(
            parameter
            for parameter in transformer.parameters()
            if parameter.requires_grad
        ),
        max_norm=MAX_GRAD_NORM,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.step()
    torch.cuda.synchronize(DEVICE)
    return {
        "step": step,
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "gradient_clipped": bool(gradient_norm > MAX_GRAD_NORM),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30,
    }


@torch.no_grad()
def _fixed_probe(
    *,
    transformer: torch.nn.Module,
    correct_text_inputs: FluxTextInputs,
    shuffled_text_inputs: FluxTextInputs,
    clean_latents: Tensor,
    spatial_shape: FluxSpatialShape,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    previous_mode = transformer.training
    transformer.eval()
    accumulators = {
        "correct": {"total_sum": 0.0, "total_count": 0, "grid_sum": 0.0, "grid_count": 0, "pad_sum": 0.0, "pad_count": 0},
        "shuffled": {"total_sum": 0.0, "total_count": 0, "grid_sum": 0.0, "grid_count": 0, "pad_sum": 0.0, "pad_count": 0},
    }
    try:
        for group_index, (_, group_slice) in enumerate(GROUP_SLICES):
            group_latents = clean_latents[group_slice]
            image_ids = build_flux_image_position_ids(
                group_latents.shape[0],
                group_latents.device,
                spatial_shape=spatial_shape,
            )
            flow = sample_flow_matching_batch(
                group_latents,
                generator=torch.Generator(device=DEVICE).manual_seed(
                    PROBE_SEED + group_index
                ),
            )
            for assignment, all_text in (
                ("correct", correct_text_inputs),
                ("shuffled", shuffled_text_inputs),
            ):
                predicted = predict_flux_velocity(
                    transformer,
                    flow.noisy_latents,
                    _slice_text_inputs(all_text, group_slice),
                    image_ids,
                    flow.timesteps,
                    compute_dtype=compute_dtype,
                )
                squared = (
                    predicted.float()
                    - flow.target_velocity.to(predicted.device).float()
                ).square()
                values = accumulators[assignment]
                values["total_sum"] += float(squared.sum().item())
                values["total_count"] += squared.numel()
                spatial = squared.reshape(
                    squared.shape[0],
                    spatial_shape.height,
                    spatial_shape.width,
                    squared.shape[2],
                )
                grid = spatial[:, :3, :3]
                values["grid_sum"] += float(grid.sum().item())
                values["grid_count"] += grid.numel()
                if spatial_shape.height > 3:
                    pad_mask = torch.ones(
                        (spatial_shape.height, spatial_shape.width),
                        dtype=torch.bool,
                        device=spatial.device,
                    )
                    pad_mask[:3, :3] = False
                    pad = spatial[:, pad_mask]
                    values["pad_sum"] += float(pad.sum().item())
                    values["pad_count"] += pad.numel()
    finally:
        transformer.train(previous_mode)

    result = {}
    for assignment, values in accumulators.items():
        result[assignment] = {
            "total_loss": values["total_sum"] / values["total_count"],
            "grid_loss": values["grid_sum"] / values["grid_count"],
            "padding_loss": (
                None
                if values["pad_count"] == 0
                else values["pad_sum"] / values["pad_count"]
            ),
        }
    result["conditioning_loss_ratio"] = (
        result["correct"]["total_loss"] / result["shuffled"]["total_loss"]
    )
    return result


def _diagnostic_payload(diagnostic: Any) -> dict[str, Any]:
    return {
        "prediction_valid": diagnostic.prediction_valid,
        "invalid_reason": diagnostic.invalid_reason,
        "predicted_shape": diagnostic.predicted_shape,
        "expected_shape": diagnostic.expected_shape,
        "shape_match": diagnostic.shape_match,
        "cell_accuracy": diagnostic.cell_accuracy,
        "exact_grid": diagnostic.exact_grid,
    }


def _save_contact_sheet(
    images: list[Tensor],
    labels: list[str],
    path: Path,
) -> None:
    tile_size = 192
    label_height = 28
    columns = 5
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (tensor, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(index, columns)
        image = Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())
        image.thumbnail((tile_size, tile_size), Image.Resampling.NEAREST)
        x = column * tile_size + (tile_size - image.width) // 2
        y = row * (tile_size + label_height) + (tile_size - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (column * tile_size + 4, row * (tile_size + label_height) + tile_size + 4),
            label[:28],
            fill="black",
        )
    sheet.save(path)


def _generation_assignment(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    variant_name: str,
    step: int,
    assignment: str,
    text_inputs: FluxTextInputs,
    expected_grids: tuple[Any, ...],
    keys: tuple[str, ...],
    compute_dtype: torch.dtype,
    image_directory: Path,
) -> dict[str, Any]:
    spatial_shape = SPATIAL_SHAPES[variant_name]
    by_seed = []
    for generation_seed in GENERATION_SEEDS:
        seed_predictions = []
        seed_diagnostics = []
        seed_latents = []
        seed_classes = []
        seed_images: list[Tensor] = []
        seed_labels: list[str] = []
        for group_index, (group_name, group_slice) in enumerate(GROUP_SLICES):
            result = generate_arc_batch(
                transformer,
                vae_decoder,
                _slice_text_inputs(text_inputs, group_slice),
                timestep_schedule=torch.linspace(
                    1.0,
                    0.0,
                    INFERENCE_STEPS + 1,
                    device=DEVICE,
                    dtype=torch.float32,
                ),
                compute_dtype=compute_dtype,
                generator=torch.Generator(device=DEVICE).manual_seed(
                    generation_seed * 100 + group_index
                ),
                spatial_shape=spatial_shape,
                infer_padding_boundary=variant_name != "compact-3x3",
            )
            group_expected = expected_grids[group_slice]
            diagnostics = diagnose_exact_grids(result.predictions, group_expected)
            seed_predictions.extend(result.predictions)
            seed_diagnostics.extend(diagnostics)
            seed_latents.append(result.final_latents)
            seed_classes.append(result.class_canvases)
            for local_index, image in enumerate(result.rgb_images):
                global_index = group_slice.start + local_index
                filename = (
                    f"step-{step:04d}-{assignment}-seed-{generation_seed}-"
                    f"{global_index:02d}-{group_name}.png"
                )
                Image.fromarray(
                    image.permute(1, 2, 0).contiguous().numpy()
                ).save(image_directory / filename)
                seed_images.append(image)
                seed_labels.append(f"{group_name}:{global_index} {keys[global_index][:18]}")
        contact_path = image_directory / (
            f"step-{step:04d}-{assignment}-seed-{generation_seed}-contact.png"
        )
        _save_contact_sheet(seed_images, seed_labels, contact_path)
        oracle_accuracies = []
        for classes, expected in zip(
            torch.cat(seed_classes, dim=0),
            expected_grids,
            strict=True,
        ):
            expected_tensor = torch.tensor(expected, dtype=torch.int64)
            oracle_accuracies.append(
                float(
                    (classes[:3, :3] == expected_tensor)
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
            )
        by_seed.append(
            {
                "seed": generation_seed,
                "predictions": [prediction.grid for prediction in seed_predictions],
                "diagnostics": [
                    _diagnostic_payload(item) for item in seed_diagnostics
                ],
                "oracle_3x3_accuracies": oracle_accuracies,
                "latents": torch.cat(seed_latents, dim=0),
                "classes": torch.cat(seed_classes, dim=0),
                "contact_sheet": str(contact_path.relative_to(REPORT_DIR)),
            }
        )
    return {"assignment": assignment, "by_seed": by_seed}


def _panel_metrics(by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for group_name, group_slice in GROUP_SLICES:
        first = by_seed[0]["diagnostics"][group_slice]
        second = by_seed[1]["diagnostics"][group_slice]
        result[group_name] = {
            "count": group_slice.stop - group_slice.start,
            "exact_grid_pass1": sum(item["exact_grid"] for item in first),
            "exact_grid_pass2": sum(
                left["exact_grid"] or right["exact_grid"]
                for left, right in zip(first, second, strict=True)
            ),
            "valid_grid_pass2": sum(
                left["prediction_valid"] or right["prediction_valid"]
                for left, right in zip(first, second, strict=True)
            ),
            "shape_pass2": sum(
                left["shape_match"] or right["shape_match"]
                for left, right in zip(first, second, strict=True)
            ),
            "mean_best_cell_accuracy": _available_mean(
                max(
                    value
                    for value in (left["cell_accuracy"], right["cell_accuracy"])
                    if value is not None
                )
                if left["cell_accuracy"] is not None
                or right["cell_accuracy"] is not None
                else None
                for left, right in zip(first, second, strict=True)
            ),
            "mean_best_oracle_3x3_accuracy": sum(
                max(
                    by_seed[0]["oracle_3x3_accuracies"][index],
                    by_seed[1]["oracle_3x3_accuracies"][index],
                )
                for index in range(group_slice.start, group_slice.stop)
            )
            / (group_slice.stop - group_slice.start),
        }
    return result


def _available_mean(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _strip_generation_tensors(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "assignment": payload["assignment"],
        "by_seed": [
            {
                key: value
                for key, value in seed.items()
                if key not in {"latents", "classes"}
            }
            for seed in payload["by_seed"]
        ],
    }


def _evaluate_variant(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    variant_name: str,
    step: int,
    correct_text_inputs: FluxTextInputs,
    shuffled_text_inputs: FluxTextInputs,
    panel_latents: Tensor,
    expected_grids: tuple[Any, ...],
    keys: tuple[str, ...],
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    started = time.perf_counter()
    probe = _fixed_probe(
        transformer=transformer,
        correct_text_inputs=correct_text_inputs,
        shuffled_text_inputs=shuffled_text_inputs,
        clean_latents=panel_latents,
        spatial_shape=SPATIAL_SHAPES[variant_name],
        compute_dtype=compute_dtype,
    )
    image_directory = REPORT_DIR / variant_name / "generated_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    correct = _generation_assignment(
        transformer=transformer,
        vae_decoder=vae_decoder,
        variant_name=variant_name,
        step=step,
        assignment="correct",
        text_inputs=correct_text_inputs,
        expected_grids=expected_grids,
        keys=keys,
        compute_dtype=compute_dtype,
        image_directory=image_directory,
    )
    entry: dict[str, Any] = {
        "step": step,
        "probe": probe,
        "panels": _panel_metrics(correct["by_seed"]),
        "generation": _strip_generation_tensors(correct),
    }
    if step == OPTIMIZER_STEPS:
        shuffled = _generation_assignment(
            transformer=transformer,
            vae_decoder=vae_decoder,
            variant_name=variant_name,
            step=step,
            assignment="shuffled",
            text_inputs=shuffled_text_inputs,
            expected_grids=expected_grids,
            keys=keys,
            compute_dtype=compute_dtype,
            image_directory=image_directory,
        )
        sensitivity = []
        for correct_seed, shuffled_seed in zip(
            correct["by_seed"], shuffled["by_seed"], strict=True
        ):
            sensitivity.append(
                {
                    "seed": correct_seed["seed"],
                    "mean_absolute_final_latent_change": float(
                        (
                            correct_seed["latents"]
                            - shuffled_seed["latents"]
                        ).abs().mean().item()
                    ),
                    "classified_cell_change_fraction": float(
                        (
                            correct_seed["classes"]
                            != shuffled_seed["classes"]
                        ).to(dtype=torch.float32).mean().item()
                    ),
                }
            )
        entry["shuffled_generation"] = _strip_generation_tensors(shuffled)
        entry["conditioning_sensitivity"] = sensitivity
    entry["seconds"] = time.perf_counter() - started
    entry["memory"] = cuda_memory(f"{variant_name}-evaluation-{step}")
    REPORT["variants"][variant_name]["evaluations"].append(entry)
    save_report()
    print(
        variant_name,
        "step",
        step,
        {
            name: values["exact_grid_pass2"]
            for name, values in entry["panels"].items()
        },
        "probe",
        probe,
    )
    return entry


def _checkpoint_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _validate_transformer_checkpoint(directory: Path) -> dict[str, dict[str, Any]]:
    config_path = directory / "config.json"
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError("Checkpoint lacks config or Safetensors index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Safetensors index contains no weight map")
    shard_names = set(weight_map.values())
    actual_shards = {
        path.name for path in directory.glob("*.safetensors") if path.is_file()
    }
    if actual_shards != shard_names:
        raise RuntimeError("Safetensors shard set differs from index")
    observed_keys = set()
    for shard_name in sorted(shard_names):
        with safe_open(directory / shard_name, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        if observed_keys.intersection(shard_keys):
            raise RuntimeError(f"Duplicate tensor keys in {shard_name}")
        observed_keys.update(shard_keys)
    if observed_keys != set(weight_map):
        raise RuntimeError("Safetensors tensor keys differ from index")
    manifest = _checkpoint_manifest(directory)
    if not manifest or any(item["bytes"] <= 0 for item in manifest.values()):
        raise RuntimeError("Checkpoint contains an empty file")
    return manifest


def _save_and_upload_checkpoint(
    *,
    transformer: torch.nn.Module,
    variant_name: str,
    volume_name: str,
    run_id: str,
) -> dict[str, Any]:
    local_root = Path("/tmp") / f"diffusion-arc-checkpoint-{run_id}-{variant_name}"
    if local_root.exists():
        raise FileExistsError(local_root)
    free_gib = shutil.disk_usage(local_root.parent).free / 2**30
    if free_gib < MINIMUM_LOCAL_CHECKPOINT_FREE_GIB:
        raise RuntimeError(
            f"Local checkpoint staging has only {free_gib:.1f} GiB free"
        )
    save_started = time.perf_counter()
    transformer.save_pretrained(
        local_root,
        safe_serialization=True,
        max_shard_size=CHECKPOINT_MAX_SHARD_SIZE,
    )
    manifest = _validate_transformer_checkpoint(local_root)

    import modal

    remote_path = (
        f"/results/single-task-spatial-overfitting-ablation/{run_id}/"
        f"{variant_name}/flux-transformer-step-{OPTIMIZER_STEPS:04d}"
    )
    volume = modal.Volume.from_name(volume_name)
    upload_started = time.perf_counter()
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_root), remote_path)
    entries = tuple(volume.iterdir(remote_path, recursive=True))
    remote_sizes = {
        Path(entry.path).name: entry.size
        for entry in entries
        if Path(entry.path).name in manifest
    }
    expected_sizes = {
        Path(relative).name: details["bytes"]
        for relative, details in manifest.items()
    }
    if remote_sizes != expected_sizes:
        raise RuntimeError(
            f"Remote checkpoint sizes differ: {remote_sizes} versus {expected_sizes}"
        )
    result = {
        "path": str(Path("/mnt") / volume_name / remote_path.lstrip("/")),
        "volume_name": volume_name,
        "volume_relative_path": remote_path,
        "step": OPTIMIZER_STEPS,
        "manifest": manifest,
        "local_validation": True,
        "remote_file_sizes_verified": True,
        "save_seconds": upload_started - save_started,
        "upload_seconds": time.perf_counter() - upload_started,
    }
    shutil.rmtree(local_root)
    return result


def _run_variant(
    *,
    transformer: torch.nn.Module,
    variant_name: str,
    run_config: Any,
    train_order: Tensor,
    training_text_inputs: FluxTextInputs,
    panel_text_inputs: FluxTextInputs,
    shuffled_panel_text_inputs: FluxTextInputs,
    training_latents: Tensor,
    panel_latents: Tensor,
    panel_grids: tuple[Any, ...],
    panel_keys: tuple[str, ...],
    vae_decoder: Any,
    volume_name: str,
    run_id: str,
) -> None:
    spatial_shape = SPATIAL_SHAPES[variant_name]
    optimizer = build_flux_optimizer(run_config, transformer=transformer)
    step_path = REPORT_DIR / variant_name / "optimizer_steps.jsonl"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    _evaluate_variant(
        transformer=transformer,
        vae_decoder=vae_decoder,
        variant_name=variant_name,
        step=0,
        correct_text_inputs=panel_text_inputs,
        shuffled_text_inputs=shuffled_panel_text_inputs,
        panel_latents=panel_latents,
        expected_grids=panel_grids,
        keys=panel_keys,
        compute_dtype=torch.bfloat16,
    )
    progress = tqdm(range(1, OPTIMIZER_STEPS + 1), desc=variant_name, unit="update")
    records = []
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for step in progress:
        index_start = (step - 1) * OPTIMIZER_BATCH_SIZE
        indices = train_order[index_start : index_start + OPTIMIZER_BATCH_SIZE]
        record = _optimizer_step(
            transformer=transformer,
            optimizer=optimizer,
            text_inputs=_slice_text_inputs(training_text_inputs, indices),
            clean_latents=training_latents.index_select(0, indices),
            spatial_shape=spatial_shape,
            step=step,
            compute_dtype=torch.bfloat16,
        )
        records.append(record)
        with step_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress.set_postfix(
            loss=f"{record['loss']:.5f}",
            grad=f"{record['gradient_norm']:.3f}",
            peak_gib=f"{record['peak_allocated_gib']:.1f}",
            refresh=False,
        )
        if step in EVALUATION_STEPS:
            _evaluate_variant(
                transformer=transformer,
                vae_decoder=vae_decoder,
                variant_name=variant_name,
                step=step,
                correct_text_inputs=panel_text_inputs,
                shuffled_text_inputs=shuffled_panel_text_inputs,
                panel_latents=panel_latents,
                expected_grids=panel_grids,
                keys=panel_keys,
                compute_dtype=torch.bfloat16,
            )
    checkpoint = _save_and_upload_checkpoint(
        transformer=transformer,
        variant_name=variant_name,
        volume_name=volume_name,
        run_id=run_id,
    )
    REPORT["variants"][variant_name]["training"] = {
        "completed_steps": len(records),
        "mean_loss": sum(item["loss"] for item in records) / len(records),
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "mean_step_seconds": sum(item["seconds"] for item in records) / len(records),
        "memory": cuda_memory(f"{variant_name}-training-complete"),
    }
    REPORT["variants"][variant_name]["saved_checkpoint"] = checkpoint
    REPORT["variants"][variant_name]["status"] = "COMPLETE"
    save_report()


def _summarize_comparison() -> dict[str, Any]:
    summary = {}
    for variant_name in VARIANT_ORDER:
        variant = REPORT["variants"][variant_name]
        baseline = variant["evaluations"][0]
        final = variant["evaluations"][-1]
        summary[variant_name] = {
            "seen_exact_grid_pass2": final["panels"]["seen"]["exact_grid_pass2"],
            "unseen_exact_grid_pass2": final["panels"]["unseen"]["exact_grid_pass2"],
            "canonical_exact_grid_pass2": final["panels"]["canonical"]["exact_grid_pass2"],
            "seen_oracle_3x3_accuracy": final["panels"]["seen"]["mean_best_oracle_3x3_accuracy"],
            "correct_probe_loss_ratio": (
                final["probe"]["correct"]["total_loss"]
                / baseline["probe"]["correct"]["total_loss"]
            ),
            "final_conditioning_loss_ratio": final["probe"]["conditioning_loss_ratio"],
            "mean_step_seconds": variant["training"]["mean_step_seconds"],
            "peak_allocated_gib": variant["training"]["memory"]["peak_allocated_gib"],
        }
    return summary


def run(args: argparse.Namespace) -> None:
    global REPORT, REPORT_DIR, REPORT_PATH, DEVICE

    volume_root = args.volume_root.resolve()
    flux_checkpoint = volume_root / "flux2-klein-base-4b"
    nvarc_checkpoint = volume_root / "nvarc-qwen3-4b-grids15-sft139"
    model_manifest_path = volume_root / "download-manifest.json"
    arc_data_root = volume_root / "data" / "raw" / "arc-agi-2"
    arc_manifest_path = volume_root / "data" / "arc-agi-2-manifest.json"
    required = (
        flux_checkpoint,
        nvarc_checkpoint,
        model_manifest_path,
        arc_data_root,
        arc_manifest_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Volume artifacts: {missing}")

    REPORT_DIR = (
        volume_root
        / "results"
        / "single-task-spatial-overfitting-ablation"
        / args.run_id
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_PATH = REPORT_DIR / "report.json"
    REPORT = {
        "schema_version": 1,
        "status": "INITIALIZING",
        "run_id": args.run_id,
        "question": (
            "Can FLUX learn the color-permutation rule for ARC task 0d3d703e "
            "from 512 one-pass final-target augmentations, and does removing "
            "the 30x30 padding overhead change the result?"
        ),
        "protocol": {
            "task_id": TASK_ID,
            "train_sequences": TRAIN_SEQUENCE_COUNT,
            "targets_per_sequence": 1,
            "nvarc_physical_batch_size": TRAIN_SEQUENCE_COUNT,
            "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
            "flux_microbatch_size": OPTIMIZER_BATCH_SIZE,
            "optimizer_steps": OPTIMIZER_STEPS,
            "evaluation_steps": list(EVALUATION_STEPS),
            "generation_seeds": list(GENERATION_SEEDS),
            "inference_steps": INFERENCE_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "variant_order": list(VARIANT_ORDER),
            "spatial_shapes": {
                name: asdict(shape) for name, shape in SPATIAL_SHAPES.items()
            },
        },
        "paths": {
            "report": str(REPORT_PATH),
            "volume_root": str(volume_root),
            "flux_checkpoint": str(flux_checkpoint),
            "nvarc_checkpoint": str(nvarc_checkpoint),
        },
        "source": {"resolved_commit": args.source_commit},
        "failures": [],
        "variants": {
            name: {"status": "PENDING", "evaluations": []}
            for name in VARIANT_ORDER
        },
    }
    save_report()

    try:
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

        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if Path(model_manifest["flux"]["path"]).resolve() != flux_checkpoint:
            raise RuntimeError("FLUX checkpoint differs from model manifest")
        if Path(model_manifest["nvarc"]["path"]).resolve() != nvarc_checkpoint:
            raise RuntimeError("NVARC checkpoint differs from model manifest")
        dataset_manifest = json.loads(arc_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest.get("counts") != {"training": 1000, "evaluation": 120}:
            raise RuntimeError("ARC dataset counts differ")
        for relative, expected_hash in dataset_manifest["files"].items():
            path = arc_data_root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"ARC dataset file differs: {relative}")

        temp_root = Path("/tmp") / f"diffusion-arc-spatial-ablation-{args.run_id}"
        temp_data_root = temp_root / "data"
        runtime_split_root = temp_data_root / "splits" / SPLIT_NAME
        runtime_split_root.parent.mkdir(parents=True)
        shutil.copytree(
            REPOSITORY_ROOT / "data" / "splits" / SPLIT_NAME,
            runtime_split_root,
        )
        (temp_data_root / "raw").mkdir()
        os.symlink(
            arc_data_root,
            temp_data_root / "raw" / "arc-agi-2",
            target_is_directory=True,
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
            data=replace(
                source_config.data,
                split_directory=runtime_split_root,
            ),
        )
        if (
            run_config.loop.max_optimizer_steps != OPTIMIZER_STEPS
            or run_config.loop.optimizer_batch_size != OPTIMIZER_BATCH_SIZE
            or run_config.loop.flux_microbatch_size != OPTIMIZER_BATCH_SIZE
            or run_config.loop.min_context_pairs_before_target != 4
        ):
            raise RuntimeError("Training configuration differs from protocol")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        DEVICE = prepare_single_gpu_device(run_config)
        seed_single_gpu_process(run_config.seed)
        torch.cuda.empty_cache()
        models = load_single_gpu_models(run_config, device=DEVICE)
        tokenizer = models.tokenizer
        nvarc_runner = models.nvarc_runner
        vae_encoder = models.vae_encoder
        vae_decoder = models.vae_decoder
        transformer = models.transformer
        compute_dtype = models.flux_compute_dtype
        del models
        parameter_count = sum(
            parameter.numel() for parameter in transformer.parameters()
        )
        base_fingerprint = _transformer_fingerprint(transformer)

        train_dataset = ArcTrainingDataset(runtime_split_root, "train", tokenizer)
        unseen_dataset = ArcTrainingDataset(
            runtime_split_root, "train_unseen", tokenizer
        )
        original_dataset = ArcTrainingDataset(
            runtime_split_root, "train_original", tokenizer
        )
        if (
            len(train_dataset) != TRAIN_SEQUENCE_COUNT
            or len(unseen_dataset) != UNSEEN_PANEL_COUNT
            or len(original_dataset) != 1
        ):
            raise RuntimeError("Dataset view counts differ from protocol")
        training_samples = tuple(
            train_dataset[index] for index in range(len(train_dataset))
        )
        unseen_samples = tuple(
            unseen_dataset[index] for index in range(len(unseen_dataset))
        )
        original_samples = (original_dataset[0],)
        for dataset, samples in (
            (train_dataset, training_samples[:SEEN_PANEL_COUNT]),
            (unseen_dataset, unseen_samples),
            (original_dataset, original_samples),
        ):
            for sample in samples:
                _assert_target_free_prefix(dataset, sample, tokenizer)

        training_ids, training_offsets = _pack_final_prefixes(training_samples)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        nvarc_started = time.perf_counter()
        training_output = nvarc_runner(training_ids, training_offsets)
        nvarc_batch_seconds = time.perf_counter() - nvarc_started
        training_hidden = training_output.sequence_hidden_states
        if len(training_hidden) != TRAIN_SEQUENCE_COUNT:
            raise RuntimeError("NVARC training output count differs")
        single_ids, single_offsets = _pack_final_prefixes((training_samples[0],))
        single_hidden = nvarc_runner(single_ids, single_offsets).sequence_hidden_states[0]
        batch_hidden = training_hidden[0]
        cosine = float(
            F.cosine_similarity(
                single_hidden.float().reshape(1, -1),
                batch_hidden.float().reshape(1, -1),
            ).item()
        )
        mean_absolute_error = float(
            (single_hidden.float() - batch_hidden.float()).abs().mean().item()
        )
        del single_hidden, batch_hidden
        training_text_inputs = pad_flux_text_prefixes(training_hidden)

        auxiliary_samples = unseen_samples + original_samples
        auxiliary_ids, auxiliary_offsets = _pack_final_prefixes(auxiliary_samples)
        auxiliary_hidden = nvarc_runner(
            auxiliary_ids, auxiliary_offsets
        ).sequence_hidden_states
        panel_hidden = training_hidden[:SEEN_PANEL_COUNT] + auxiliary_hidden
        panel_text_inputs = pad_flux_text_prefixes(panel_hidden)
        shuffled_panel_text_inputs = _permute_text_inputs(panel_text_inputs)
        del training_output, training_hidden, auxiliary_hidden, panel_hidden
        torch.cuda.empty_cache()

        panel_samples = (
            training_samples[:SEEN_PANEL_COUNT]
            + unseen_samples
            + original_samples
        )
        panel_keys = tuple(str(sample.key) for sample in panel_samples)
        panel_grids = tuple(sample.target_grids[3] for sample in panel_samples)
        training_grids = tuple(sample.target_grids[3] for sample in training_samples)
        train_order = torch.randperm(
            TRAIN_SEQUENCE_COUNT,
            generator=torch.Generator(device="cpu").manual_seed(run_config.seed),
        ).to(device=DEVICE)

        REPORT["dataset"] = {
            "split_name": SPLIT_NAME,
            "task_id": TASK_ID,
            "train_keys": [str(sample.key) for sample in training_samples],
            "panel_keys": list(panel_keys),
            "panel_groups": {name: [group_slice.start, group_slice.stop] for name, group_slice in GROUP_SLICES},
            "split_file_sha256": {
                str(path.relative_to(runtime_split_root)): sha256_file(path)
                for path in sorted(runtime_split_root.rglob("*.json"))
            },
        }
        REPORT["nvarc_precompute"] = {
            "physical_batch_size": TRAIN_SEQUENCE_COUNT,
            "seconds": nvarc_batch_seconds,
            "batch1_vs_batch512_cosine_similarity": cosine,
            "batch1_vs_batch512_mean_absolute_error": mean_absolute_error,
            "training_text_shape": list(
                training_text_inputs.encoder_hidden_states.shape
            ),
            "panel_text_shape": list(panel_text_inputs.encoder_hidden_states.shape),
            "memory": cuda_memory("after-nvarc-precompute"),
        }
        REPORT["resolved_training_config"] = json.loads(
            json.dumps(asdict(run_config), default=str)
        )

        variant_data = {}
        for variant_name in VARIANT_ORDER:
            encoding_started = time.perf_counter()
            training_latents = _encode_target_grids(
                training_grids,
                variant_name=variant_name,
                vae_encoder=vae_encoder,
            )
            panel_latents = _encode_target_grids(
                panel_grids,
                variant_name=variant_name,
                vae_encoder=vae_encoder,
            )
            roundtrip = _vae_roundtrip(
                variant_name=variant_name,
                latents=panel_latents,
                grids=panel_grids,
                vae_decoder=vae_decoder,
            )
            variant_data[variant_name] = {
                "training_latents": training_latents,
                "panel_latents": panel_latents,
            }
            REPORT["variants"][variant_name].update(
                {
                    "spatial_shape": asdict(SPATIAL_SHAPES[variant_name]),
                    "training_latent_shape": list(training_latents.shape),
                    "panel_latent_shape": list(panel_latents.shape),
                    "vae_roundtrip": roundtrip,
                    "vae_encoding_seconds": time.perf_counter() - encoding_started,
                }
            )

        for variant_name in VARIANT_ORDER:
            first_latent = variant_data[variant_name]["training_latents"][:1]
            first_text = _slice_text_inputs(training_text_inputs, slice(0, 1))
            spatial_shape = SPATIAL_SHAPES[variant_name]
            with torch.no_grad():
                flow = sample_flow_matching_batch(
                    first_latent,
                    generator=torch.Generator(device=DEVICE).manual_seed(PROBE_SEED),
                )
                predicted = predict_flux_velocity(
                    transformer,
                    flow.noisy_latents,
                    first_text,
                    build_flux_image_position_ids(
                        1,
                        DEVICE,
                        spatial_shape=spatial_shape,
                    ),
                    flow.timesteps,
                    compute_dtype=compute_dtype,
                )
                loss = flow_matching_loss(predicted, flow.target_velocity)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"{variant_name} preflight loss is non-finite")
            REPORT["variants"][variant_name]["preflight_loss"] = float(loss.item())
        REPORT["models"] = {
            "flux_parameter_count": parameter_count,
            "base_fingerprint": base_fingerprint,
            "memory_after_preflight": cuda_memory("after-spatial-preflight"),
        }
        save_report()

        REPORT["variants"]["compact-3x3"]["status"] = "TRAINING"
        save_report()
        _run_variant(
            transformer=transformer,
            variant_name="compact-3x3",
            run_config=run_config,
            train_order=train_order,
            training_text_inputs=training_text_inputs,
            panel_text_inputs=panel_text_inputs,
            shuffled_panel_text_inputs=shuffled_panel_text_inputs,
            training_latents=variant_data["compact-3x3"]["training_latents"],
            panel_latents=variant_data["compact-3x3"]["panel_latents"],
            panel_grids=panel_grids,
            panel_keys=panel_keys,
            vae_decoder=vae_decoder,
            volume_name=args.volume_name,
            run_id=args.run_id,
        )
        del transformer
        gc.collect()
        torch.cuda.empty_cache()
        transformer = _load_fresh_transformer(
            flux_checkpoint,
            expected_parameter_count=parameter_count,
        )
        fresh_fingerprint = _transformer_fingerprint(transformer)
        if fresh_fingerprint != base_fingerprint:
            raise RuntimeError("Second transformer does not match the base checkpoint")
        REPORT["variants"]["padded-30x30"]["fresh_base_fingerprint"] = fresh_fingerprint
        REPORT["variants"]["padded-30x30"]["status"] = "TRAINING"
        save_report()
        _run_variant(
            transformer=transformer,
            variant_name="padded-30x30",
            run_config=run_config,
            train_order=train_order,
            training_text_inputs=training_text_inputs,
            panel_text_inputs=panel_text_inputs,
            shuffled_panel_text_inputs=shuffled_panel_text_inputs,
            training_latents=variant_data["padded-30x30"]["training_latents"],
            panel_latents=variant_data["padded-30x30"]["panel_latents"],
            panel_grids=panel_grids,
            panel_keys=panel_keys,
            vae_decoder=vae_decoder,
            volume_name=args.volume_name,
            run_id=args.run_id,
        )
        REPORT["comparison"] = _summarize_comparison()
        REPORT["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT["status"] = "COMPLETE"
        save_report()
        print(json.dumps(REPORT["comparison"], indent=2))
        print(f"FINAL_REPORT={REPORT_PATH}")
    except Exception as error:
        record_failure("experiment", error)
        raise


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
