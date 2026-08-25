#!/usr/bin/env python3
"""Run the padded single-task conditioning-overfitting ladder on Modal."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcEvaluationTask,
    ArcGeometry,
    ArcQuery,
    ArcTokenizedTrainingSample,
    apply_training_augmentation,
    classify_arc_cells,
    decode_arc_grids,
    load_training_task,
    render_evaluation_prompts,
    render_grid_canvas,
    render_training_prompt,
    tokenize_nvarc_evaluation_sample,
    tokenize_nvarc_training_sample,
    validate_augmentation_for_task,
)
from diffusion_arc.inference import diagnose_exact_grids, generate_arc_batch
from diffusion_arc.model import (
    FluxSpatialShape,
    FluxTextInputs,
    build_flux_image_position_ids,
    flow_matching_loss,
    pad_flux_text_prefixes,
    predict_flux_velocity,
    sample_stratified_flow_matching_batch,
)
from diffusion_arc.model.flux_transformer import Flux2Transformer2DModel
from diffusion_arc.training import FluxMicrobatch
from diffusion_arc.training.config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    build_flux_optimizer,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "single_gpu_0d3d703e_conditioning_ladder.json"
TASK_ID = "0d3d703e"
MAX_TOKENS = 192
SPATIAL_SHAPE = FluxSpatialShape(height=30, width=30)
OPTIMIZER_BATCH_SIZE = 8
GENERATION_SEEDS = (167, 168)
PROBE_TIMESTEPS = (0.1, 0.5, 0.9)
PROBE_SEED = 1167
FLOW_SEED = 2167
MAX_GRAD_NORM = 1.0
INFERENCE_STEPS = 50
CHECKPOINT_MAX_SHARD_SIZE = "2GB"
MINIMUM_VRAM_GIB = 120.0
MINIMUM_LOCAL_CHECKPOINT_FREE_GIB = 24.0
RESULT_FAMILY = "single-task-conditioning-overfitting-ladder"


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    geometry: ArcGeometry
    color_mapping: tuple[int, ...]
    pair_indices: tuple[int, ...]

    def key(self) -> ArcAugmentationKey:
        return ArcAugmentationKey(
            task_id=TASK_ID,
            augmentation=ArcAugmentationSpec(
                geometry=self.geometry,
                color_mapping=self.color_mapping,
                pair_indices=self.pair_indices,
            ),
        )


@dataclass(frozen=True)
class StageSpec:
    name: str
    condition_names: tuple[str, ...]
    optimizer_steps: int
    evaluation_steps: tuple[int, ...]

    def batch_condition_names(self) -> tuple[str, ...]:
        if len(self.condition_names) == 1:
            return self.condition_names * OPTIMIZER_BATCH_SIZE
        if len(self.condition_names) == 2:
            return self.condition_names * (OPTIMIZER_BATCH_SIZE // 2)
        if len(self.condition_names) == OPTIMIZER_BATCH_SIZE:
            return self.condition_names
        raise RuntimeError(f"Unsupported stage condition count: {self.name}")


def _cyclic_color_mapping(shift: int) -> tuple[int, ...]:
    return tuple((color + shift) % 10 for color in range(10))


def _affine_color_mapping(offset: int) -> tuple[int, ...]:
    return tuple((3 * color + offset) % 10 for color in range(10))


_IDENTITY_PAIR_ORDER = (0, 1, 2, 3, 4)
_GEOMETRIES = tuple(ArcGeometry)
_TRAIN_CONDITION_NAMES = (
    "train-id",
    "train-r90",
    "train-r180",
    "train-r270",
    "train-fh",
    "train-fv",
    "train-fd1",
    "train-fd2",
)
_UNSEEN_CONDITION_NAMES = (
    "unseen-id",
    "unseen-r90",
    "unseen-r180",
    "unseen-r270",
    "unseen-fh",
    "unseen-fv",
    "unseen-fd1",
    "unseen-fd2",
)
_UNSEEN_PAIR_ORDERS = (
    (3, 2, 1, 0, 4),
    (1, 0, 3, 2, 4),
    (2, 3, 0, 1, 4),
    (0, 2, 3, 1, 4),
    (3, 1, 0, 2, 4),
    (1, 3, 2, 0, 4),
    (2, 0, 1, 3, 4),
    (0, 3, 1, 2, 4),
)


CONDITION_SPECS = tuple(
    ConditionSpec(
        name=name,
        geometry=geometry,
        color_mapping=_cyclic_color_mapping(index + 1),
        pair_indices=_IDENTITY_PAIR_ORDER,
    )
    for index, (name, geometry) in enumerate(
        zip(_TRAIN_CONDITION_NAMES, _GEOMETRIES, strict=True)
    )
) + tuple(
    ConditionSpec(
        name=name,
        geometry=geometry,
        color_mapping=_affine_color_mapping(index),
        pair_indices=_UNSEEN_PAIR_ORDERS[index],
    )
    for index, (name, geometry) in enumerate(
        zip(_UNSEEN_CONDITION_NAMES, _GEOMETRIES, strict=True)
    )
) + (
    ConditionSpec(
        name="canonical",
        geometry=ArcGeometry.IDENTITY,
        color_mapping=tuple(range(10)),
        pair_indices=_IDENTITY_PAIR_ORDER,
    ),
)

STAGES = (
    StageSpec(
        name="one-condition",
        condition_names=("canonical",),
        optimizer_steps=256,
        evaluation_steps=(0, 32, 64, 128, 256),
    ),
    StageSpec(
        name="two-condition-routing",
        condition_names=("canonical", "train-id"),
        optimizer_steps=512,
        evaluation_steps=(0, 32, 64, 128, 256, 512),
    ),
    StageSpec(
        name="eight-condition-overfit",
        condition_names=_TRAIN_CONDITION_NAMES,
        optimizer_steps=512,
        evaluation_steps=(0, 32, 64, 128, 256, 512),
    ),
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


def _pack_final_prefixes(
    samples: tuple[ArcTokenizedTrainingSample, ...],
) -> tuple[Tensor, Tensor]:
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


def _direct_mapping_pairs(pairs: tuple[Any, ...]) -> set[tuple[int, int]]:
    observed: set[tuple[int, int]] = set()
    for pair in pairs:
        for input_row, output_row in zip(
            pair.input_grid,
            pair.output_grid,
            strict=True,
        ):
            observed.update(zip(input_row, output_row, strict=True))
    return observed


def _validate_final_target_condition(
    *,
    task: Any,
    sample: ArcTokenizedTrainingSample,
    tokenizer: Any,
) -> None:
    key = sample.key
    if key.augmentation.pair_indices[-1] != 4:
        raise RuntimeError(f"Condition does not keep original pair 4 as target: {key}")
    if set(key.augmentation.pair_indices[:-1]) != {0, 1, 2, 3}:
        raise RuntimeError(f"Condition demonstrations differ from pairs 0..3: {key}")
    validate_augmentation_for_task(key, task)
    augmented = apply_training_augmentation(task, key.augmentation)
    query_pair = augmented.pairs[-1]
    required_mapping = _direct_mapping_pairs((query_pair,))
    observed_mapping = _direct_mapping_pairs(augmented.pairs[:-1])
    if not required_mapping.issubset(observed_mapping):
        missing = sorted(required_mapping - observed_mapping)
        raise RuntimeError(f"Condition lacks directly demonstrated mappings {missing}: {key}")

    evaluation_task = ArcEvaluationTask(
        task_id=str(key),
        demonstrations=augmented.pairs[:-1],
        queries=(ArcQuery(query_index=0, input_grid=query_pair.input_grid),),
    )
    rendered = render_evaluation_prompts(evaluation_task)[0]
    tokenized = tokenize_nvarc_evaluation_sample(
        rendered,
        tokenizer,
        max_tokens=MAX_TOKENS,
    )
    training_prefix = sample.input_ids[: sample.prefix_token_ends[3]]
    if tokenized.input_ids != training_prefix:
        raise RuntimeError(f"Target-free prefix differs for {key}")
    if sample.target_grids[3] != query_pair.output_grid:
        raise RuntimeError(f"Final target differs for {key}")


def _build_condition_samples(
    task: Any,
    tokenizer: Any,
) -> dict[str, ArcTokenizedTrainingSample]:
    if len(task.pairs) != 5:
        raise RuntimeError(
            f"Expected task {TASK_ID} to contain five pairs, received {len(task.pairs)}"
        )
    result: dict[str, ArcTokenizedTrainingSample] = {}
    serialized_keys: set[str] = set()
    for condition in CONDITION_SPECS:
        key = condition.key()
        serialized = str(key)
        if serialized in serialized_keys:
            raise RuntimeError(f"Duplicate condition key: {serialized}")
        serialized_keys.add(serialized)
        augmented = apply_training_augmentation(task, key.augmentation)
        sample = tokenize_nvarc_training_sample(
            render_training_prompt(augmented),
            tokenizer,
            max_tokens=MAX_TOKENS,
        )
        _validate_final_target_condition(
            task=task,
            sample=sample,
            tokenizer=tokenizer,
        )
        result[condition.name] = sample
    if set(result) != {condition.name for condition in CONDITION_SPECS}:
        raise RuntimeError("Condition registry differs from protocol")
    return result


def _encode_target_grids(grids: tuple[Any, ...], vae_encoder: Any) -> Tensor:
    chunks = []
    progress = tqdm(
        range(0, len(grids), 16),
        desc="VAE encode padded-30x30",
        unit="batch",
    )
    for start in progress:
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


def _decode_latents(latents: Tensor, vae_decoder: Any) -> tuple[Tensor, tuple[Any, ...], Tensor]:
    decoded = vae_decoder(latents, spatial_shape=SPATIAL_SHAPE)
    rgb_images = _normalized_images_to_uint8(decoded)
    classes = classify_arc_cells(
        rgb_images,
        canvas_size=SPATIAL_SHAPE.height,
        include_pad=True,
    )
    return classes, decode_arc_grids(classes, canvas_size=SPATIAL_SHAPE.height), rgb_images


def _vae_roundtrip(
    *,
    latents: Tensor,
    grids: tuple[Any, ...],
    vae_decoder: Any,
) -> dict[str, Any]:
    predictions = []
    for start in range(0, len(grids), 8):
        _, decoded, _ = _decode_latents(
            latents[start : start + 8],
            vae_decoder,
        )
        predictions.extend(decoded)
    diagnostics = diagnose_exact_grids(tuple(predictions), grids)
    exact = sum(item.exact_grid for item in diagnostics)
    if exact != len(grids):
        raise RuntimeError(f"Clean VAE round trip is {exact}/{len(grids)}")
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
    condition_names: tuple[str, ...],
    step: int,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    image_ids = build_flux_image_position_ids(
        clean_latents.shape[0],
        clean_latents.device,
        spatial_shape=SPATIAL_SHAPE,
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
    flow_batch = sample_stratified_flow_matching_batch(
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
    squared_error = (
        predicted_velocity.float() - flow_batch.target_velocity.float()
    ).square().detach()
    loss = flow_matching_loss(predicted_velocity, flow_batch.target_velocity)
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
    spatial_squared = squared_error.reshape(
        squared_error.shape[0],
        SPATIAL_SHAPE.height,
        SPATIAL_SHAPE.width,
        squared_error.shape[2],
    )
    padding_mask = torch.ones(
        (SPATIAL_SHAPE.height, SPATIAL_SHAPE.width),
        dtype=torch.bool,
        device=DEVICE,
    )
    padding_mask[:3, :3] = False
    exposure_metrics = torch.stack(
        (
            flow_batch.timesteps,
            squared_error.mean(dim=(1, 2)),
            spatial_squared[:, :3, :3].mean(dim=(1, 2, 3)),
            spatial_squared[:, padding_mask].mean(dim=(1, 2)),
        ),
        dim=1,
    ).to(device="cpu", dtype=torch.float32)
    per_exposure = []
    for index, condition_name in enumerate(condition_names):
        timestep, total_loss, grid_loss, padding_loss = exposure_metrics[index].tolist()
        per_exposure.append(
            {
                "condition": condition_name,
                "timestep": float(timestep),
                "timestep_stratum": min(int(float(timestep) * 8), 7),
                "total_loss": total_loss,
                "grid_loss": grid_loss,
                "padding_loss": padding_loss,
            }
        )
    return {
        "step": step,
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "gradient_clipped": bool(gradient_norm > MAX_GRAD_NORM),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "timesteps": [item["timestep"] for item in per_exposure],
        "per_exposure": per_exposure,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30,
    }


def _region_losses(squared_error: Tensor) -> dict[str, float]:
    spatial = squared_error.reshape(
        squared_error.shape[0],
        SPATIAL_SHAPE.height,
        SPATIAL_SHAPE.width,
        squared_error.shape[2],
    )
    grid = spatial[:, :3, :3]
    padding_mask = torch.ones(
        (SPATIAL_SHAPE.height, SPATIAL_SHAPE.width),
        dtype=torch.bool,
        device=spatial.device,
    )
    padding_mask[:3, :3] = False
    padding = spatial[:, padding_mask]
    return {
        "total_loss": float(squared_error.mean().item()),
        "grid_loss": float(grid.mean().item()),
        "padding_loss": float(padding.mean().item()),
    }


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


def _oracle_3x3_accuracies(
    classes: Tensor,
    expected_grids: tuple[Any, ...],
) -> list[float]:
    result = []
    for class_canvas, expected in zip(classes, expected_grids, strict=True):
        expected_tensor = torch.tensor(expected, dtype=torch.int64)
        result.append(
            float(
                (class_canvas[:3, :3] == expected_tensor)
                .to(dtype=torch.float32)
                .mean()
                .item()
            )
        )
    return result


def _stage_panel_layout(
    stage: StageSpec,
    condition_indices: dict[str, int],
) -> dict[str, Any]:
    panel_names = stage.condition_names + _UNSEEN_CONDITION_NAMES + ("canonical",)
    seen_end = len(stage.condition_names)
    unseen_end = seen_end + len(_UNSEEN_CONDITION_NAMES)
    groups = (
        ("seen", slice(0, seen_end)),
        ("unseen", slice(seen_end, unseen_end)),
        ("canonical", slice(unseen_end, unseen_end + 1)),
    )
    return {
        "names": panel_names,
        "indices": torch.tensor(
            [condition_indices[name] for name in panel_names],
            device=DEVICE,
            dtype=torch.int64,
        ),
        "groups": groups,
    }


@torch.no_grad()
def _teacher_forced_probe(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage: StageSpec,
    panel_text_inputs: FluxTextInputs,
    panel_latents: Tensor,
    expected_grids: tuple[Any, ...],
    panel_names: tuple[str, ...],
    groups: tuple[tuple[str, slice], ...],
    swapped_seen_text_inputs: FluxTextInputs | None,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    previous_mode = transformer.training
    transformer.eval()
    entries = []
    seen_correct_grid_losses = []
    seen_swapped_grid_losses = []
    try:
        for group_index, (group_name, group_slice) in enumerate(groups):
            group_latents = panel_latents[group_slice]
            group_grids = expected_grids[group_slice]
            group_names = panel_names[group_slice]
            image_ids = build_flux_image_position_ids(
                group_latents.shape[0],
                DEVICE,
                spatial_shape=SPATIAL_SHAPE,
            )
            correct_text = _slice_text_inputs(panel_text_inputs, group_slice)
            assignments = [("correct", correct_text)]
            if group_name == "seen" and swapped_seen_text_inputs is not None:
                assignments.append(("swapped", swapped_seen_text_inputs))

            for timestep_index, timestep_value in enumerate(PROBE_TIMESTEPS):
                noise = torch.randn(
                    group_latents.shape,
                    device=DEVICE,
                    dtype=group_latents.dtype,
                    generator=torch.Generator(device=DEVICE).manual_seed(
                        PROBE_SEED + group_index * 100 + timestep_index
                    ),
                )
                timesteps = torch.full(
                    (group_latents.shape[0],),
                    timestep_value,
                    device=DEVICE,
                    dtype=torch.float32,
                )
                broadcast = timesteps.to(dtype=group_latents.dtype).reshape(-1, 1, 1)
                noisy_latents = (1 - broadcast) * group_latents + broadcast * noise
                target_velocity = noise - group_latents

                for assignment, text_inputs in assignments:
                    predicted = predict_flux_velocity(
                        transformer,
                        noisy_latents,
                        text_inputs,
                        image_ids,
                        timesteps,
                        compute_dtype=compute_dtype,
                    )
                    squared = (predicted.float() - target_velocity.float()).square()
                    losses = _region_losses(squared)
                    per_condition = {
                        name: _region_losses(squared[index : index + 1])
                        for index, name in enumerate(group_names)
                    }
                    reconstructed_latents = noisy_latents - broadcast * predicted
                    classes, predictions, _ = _decode_latents(
                        reconstructed_latents,
                        vae_decoder,
                    )
                    diagnostics = diagnose_exact_grids(predictions, group_grids)
                    entry = {
                        "group": group_name,
                        "assignment": assignment,
                        "timestep": timestep_value,
                        "loss": losses,
                        "per_condition_loss": per_condition,
                        "z0_reconstruction": {
                            "diagnostics": [
                                _diagnostic_payload(item) for item in diagnostics
                            ],
                            "oracle_3x3_accuracies": _oracle_3x3_accuracies(
                                classes,
                                group_grids,
                            ),
                        },
                    }
                    entries.append(entry)
                    if group_name == "seen":
                        if assignment == "correct":
                            seen_correct_grid_losses.append(losses["grid_loss"])
                        else:
                            seen_swapped_grid_losses.append(losses["grid_loss"])
    finally:
        transformer.train(previous_mode)

    routing_summary = None
    if swapped_seen_text_inputs is not None:
        routing_summary = {
            "correct_mean_grid_loss": sum(seen_correct_grid_losses)
            / len(seen_correct_grid_losses),
            "swapped_mean_grid_loss": sum(seen_swapped_grid_losses)
            / len(seen_swapped_grid_losses),
            "correct_to_swapped_grid_loss_ratio": (
                sum(seen_correct_grid_losses)
                / sum(seen_swapped_grid_losses)
            ),
            "same_flow_state_per_target_row": True,
            "condition_order": list(stage.condition_names),
            "swapped_condition_order": list(
                stage.condition_names[1:] + stage.condition_names[:1]
            ),
        }
    return {
        "timesteps": list(PROBE_TIMESTEPS),
        "entries": entries,
        "routing_summary": routing_summary,
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
            (
                column * tile_size + 4,
                row * (tile_size + label_height) + tile_size + 4,
            ),
            label[:28],
            fill="black",
        )
    sheet.save(path)


def _available_mean(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else sum(present) / len(present)


def _panel_metrics(
    by_seed: list[dict[str, Any]],
    groups: tuple[tuple[str, slice], ...],
) -> dict[str, Any]:
    result = {}
    for group_name, group_slice in groups:
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


def _generate_assignment(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage_name: str,
    step: int,
    assignment: str,
    text_inputs: FluxTextInputs,
    expected_grids: tuple[Any, ...],
    condition_names: tuple[str, ...],
    groups: tuple[tuple[str, slice], ...],
    compute_dtype: torch.dtype,
    image_directory: Path,
) -> dict[str, Any]:
    by_seed = []
    schedule = torch.linspace(
        1.0,
        0.0,
        INFERENCE_STEPS + 1,
        device=DEVICE,
        dtype=torch.float32,
    )
    for generation_seed in GENERATION_SEEDS:
        seed_predictions = []
        seed_diagnostics = []
        seed_latents = []
        seed_classes = []
        seed_images: list[Tensor] = []
        seed_labels: list[str] = []
        for group_index, (group_name, group_slice) in enumerate(groups):
            result = generate_arc_batch(
                transformer,
                vae_decoder,
                _slice_text_inputs(text_inputs, group_slice),
                timestep_schedule=schedule,
                compute_dtype=compute_dtype,
                generator=torch.Generator(device=DEVICE).manual_seed(
                    generation_seed * 100 + group_index
                ),
                spatial_shape=SPATIAL_SHAPE,
                infer_padding_boundary=True,
            )
            group_expected = expected_grids[group_slice]
            diagnostics = diagnose_exact_grids(result.predictions, group_expected)
            seed_predictions.extend(result.predictions)
            seed_diagnostics.extend(diagnostics)
            seed_latents.append(result.final_latents)
            seed_classes.append(result.class_canvases)
            for local_index, image in enumerate(result.rgb_images):
                global_index = group_slice.start + local_index
                seed_images.append(image)
                seed_labels.append(
                    f"{group_name}:{global_index} {condition_names[global_index]}"
                )
        contact_path = image_directory / (
            f"step-{step:04d}-{assignment}-seed-{generation_seed}-contact.png"
        )
        _save_contact_sheet(seed_images, seed_labels, contact_path)
        classes = torch.cat(seed_classes, dim=0)
        by_seed.append(
            {
                "seed": generation_seed,
                "predictions": [
                    prediction.grid for prediction in seed_predictions
                ],
                "prediction_objects": tuple(seed_predictions),
                "diagnostics": [
                    _diagnostic_payload(item) for item in seed_diagnostics
                ],
                "oracle_3x3_accuracies": _oracle_3x3_accuracies(
                    classes,
                    expected_grids,
                ),
                "latents": torch.cat(seed_latents, dim=0),
                "classes": classes,
                "contact_sheet": str(contact_path.relative_to(REPORT_DIR)),
            }
        )
    return {
        "stage": stage_name,
        "assignment": assignment,
        "by_seed": by_seed,
    }


def _strip_generation_tensors(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": payload["stage"],
        "assignment": payload["assignment"],
        "by_seed": [
            {
                key: value
                for key, value in seed.items()
                if key not in {"latents", "classes", "prediction_objects"}
            }
            for seed in payload["by_seed"]
        ],
    }


def _evaluate_stage(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    stage: StageSpec,
    step: int,
    panel_text_inputs: FluxTextInputs,
    panel_latents: Tensor,
    expected_grids: tuple[Any, ...],
    panel_names: tuple[str, ...],
    groups: tuple[tuple[str, slice], ...],
    swapped_seen_text_inputs: FluxTextInputs | None,
    swapped_seen_expected_grids: tuple[Any, ...] | None,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    started = time.perf_counter()
    probe = _teacher_forced_probe(
        transformer=transformer,
        vae_decoder=vae_decoder,
        stage=stage,
        panel_text_inputs=panel_text_inputs,
        panel_latents=panel_latents,
        expected_grids=expected_grids,
        panel_names=panel_names,
        groups=groups,
        swapped_seen_text_inputs=swapped_seen_text_inputs,
        compute_dtype=compute_dtype,
    )
    image_directory = REPORT_DIR / stage.name / "generated_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    correct = _generate_assignment(
        transformer=transformer,
        vae_decoder=vae_decoder,
        stage_name=stage.name,
        step=step,
        assignment="correct",
        text_inputs=panel_text_inputs,
        expected_grids=expected_grids,
        condition_names=panel_names,
        groups=groups,
        compute_dtype=compute_dtype,
        image_directory=image_directory,
    )
    entry: dict[str, Any] = {
        "step": step,
        "teacher_forced_probe": probe,
        "panels": _panel_metrics(correct["by_seed"], groups),
        "generation": _strip_generation_tensors(correct),
    }
    if step == stage.optimizer_steps and swapped_seen_text_inputs is not None:
        seen_slice = groups[0][1]
        seen_groups = (("seen", slice(0, len(stage.condition_names))),)
        swapped = _generate_assignment(
            transformer=transformer,
            vae_decoder=vae_decoder,
            stage_name=stage.name,
            step=step,
            assignment="swapped-seen",
            text_inputs=swapped_seen_text_inputs,
            expected_grids=expected_grids[seen_slice],
            condition_names=(
                stage.condition_names[1:] + stage.condition_names[:1]
            ),
            groups=seen_groups,
            compute_dtype=compute_dtype,
            image_directory=image_directory,
        )
        routed_diagnostics = []
        sensitivity = []
        if swapped_seen_expected_grids is None:
            raise AssertionError("Swapped expected grids are required")
        for correct_seed, swapped_seed in zip(
            correct["by_seed"],
            swapped["by_seed"],
            strict=True,
        ):
            routed = diagnose_exact_grids(
                swapped_seed["prediction_objects"],
                swapped_seen_expected_grids,
            )
            routed_diagnostics.append(
                {
                    "seed": swapped_seed["seed"],
                    "diagnostics": [_diagnostic_payload(item) for item in routed],
                }
            )
            sensitivity.append(
                {
                    "seed": correct_seed["seed"],
                    "mean_absolute_final_latent_change": float(
                        (
                            correct_seed["latents"][seen_slice]
                            - swapped_seed["latents"]
                        )
                        .abs()
                        .mean()
                        .item()
                    ),
                    "classified_cell_change_fraction": float(
                        (
                            correct_seed["classes"][seen_slice]
                            != swapped_seed["classes"]
                        )
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                    ),
                }
            )
        entry["swapped_seen_generation"] = _strip_generation_tensors(swapped)
        entry["swapped_seen_routed_target_diagnostics"] = routed_diagnostics
        entry["conditioning_sensitivity"] = sensitivity
    entry["seconds"] = time.perf_counter() - started
    entry["memory"] = cuda_memory(f"{stage.name}-evaluation-{step}")
    REPORT["stages"][stage.name]["evaluations"].append(entry)
    save_report()
    print(
        stage.name,
        "step",
        step,
        {
            name: values["exact_grid_pass2"]
            for name, values in entry["panels"].items()
        },
        "routing",
        probe["routing_summary"],
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


def _validate_transformer_checkpoint(
    directory: Path,
) -> dict[str, dict[str, Any]]:
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
    stage: StageSpec,
    volume_name: str,
    run_id: str,
) -> dict[str, Any]:
    local_root = Path("/tmp") / f"diffusion-arc-checkpoint-{run_id}-{stage.name}"
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
        f"/results/{RESULT_FAMILY}/{run_id}/{stage.name}/"
        f"flux-transformer-step-{stage.optimizer_steps:04d}"
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
        "step": stage.optimizer_steps,
        "state_scope": "transformer_only",
        "manifest": manifest,
        "local_validation": True,
        "remote_file_sizes_verified": True,
        "save_seconds": upload_started - save_started,
        "upload_seconds": time.perf_counter() - upload_started,
    }
    shutil.rmtree(local_root)
    return result


def _upload_and_verify_final_report(
    *,
    volume_name: str,
    run_id: str,
) -> None:
    import modal

    remote_path = f"/results/{RESULT_FAMILY}/{run_id}/report.json"
    REPORT["durability"] = {
        "remote_report_path": remote_path,
        "upload_method": "modal.Volume.batch_upload",
        "remote_content_verified": False,
    }
    save_report()
    staging_path = Path("/tmp") / f"diffusion-arc-report-{run_id}.json"
    volume = modal.Volume.from_name(volume_name)
    for verified_state in (False, True):
        REPORT["durability"]["remote_content_verified"] = verified_state
        save_report()
        local_bytes = REPORT_PATH.read_bytes()
        staging_path.write_bytes(local_bytes)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(staging_path), remote_path)
        remote_bytes = b"".join(volume.read_file(remote_path))
        if remote_bytes != local_bytes:
            raise RuntimeError("Remote report content differs after batch upload")
    staging_path.unlink()


def _summarize_timestep_strata(records: list[dict[str, Any]]) -> dict[str, Any]:
    strata = {index: [] for index in range(8)}
    for record in records:
        for exposure in record["per_exposure"]:
            strata[exposure["timestep_stratum"]].append(exposure)
    result = {}
    for index, exposures in strata.items():
        if len(exposures) != len(records):
            raise RuntimeError(
                f"Timestep stratum {index} contains {len(exposures)} exposures, "
                f"expected {len(records)}"
            )
        result[str(index)] = {
            "interval": [index / 8, (index + 1) / 8],
            "count": len(exposures),
            "mean_total_loss": sum(item["total_loss"] for item in exposures)
            / len(exposures),
            "mean_grid_loss": sum(item["grid_loss"] for item in exposures)
            / len(exposures),
            "mean_padding_loss": sum(item["padding_loss"] for item in exposures)
            / len(exposures),
        }
    return result


def _run_stage(
    *,
    transformer: torch.nn.Module,
    stage: StageSpec,
    run_config: Any,
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    condition_indices: dict[str, int],
    grids_by_name: dict[str, Any],
    vae_decoder: Any,
    compute_dtype: torch.dtype,
    volume_name: str,
    run_id: str,
) -> None:
    batch_names = stage.batch_condition_names()
    batch_indices = torch.tensor(
        [condition_indices[name] for name in batch_names],
        device=DEVICE,
        dtype=torch.int64,
    )
    batch_text_inputs = _slice_text_inputs(all_text_inputs, batch_indices)
    batch_latents = all_latents.index_select(0, batch_indices)

    layout = _stage_panel_layout(stage, condition_indices)
    panel_names = layout["names"]
    panel_indices = layout["indices"]
    groups = layout["groups"]
    panel_text_inputs = _slice_text_inputs(all_text_inputs, panel_indices)
    panel_latents = all_latents.index_select(0, panel_indices)
    expected_grids = tuple(grids_by_name[name] for name in panel_names)

    swapped_seen_text_inputs = None
    swapped_seen_expected_grids = None
    if len(stage.condition_names) > 1:
        swapped_names = stage.condition_names[1:] + stage.condition_names[:1]
        swapped_indices = torch.tensor(
            [condition_indices[name] for name in swapped_names],
            device=DEVICE,
            dtype=torch.int64,
        )
        swapped_seen_text_inputs = _slice_text_inputs(
            all_text_inputs,
            swapped_indices,
        )
        swapped_seen_expected_grids = tuple(
            grids_by_name[name] for name in swapped_names
        )

    stage_report = REPORT["stages"][stage.name]
    stage_report.update(
        {
            "status": "TRAINING",
            "condition_names": list(stage.condition_names),
            "batch_condition_names": list(batch_names),
            "panel_condition_names": list(panel_names),
            "panel_groups": {
                name: [group_slice.start, group_slice.stop]
                for name, group_slice in groups
            },
            "canonical_is_seen": "canonical" in stage.condition_names,
            "fresh_base_fingerprint": _transformer_fingerprint(transformer),
        }
    )
    save_report()

    _evaluate_stage(
        transformer=transformer,
        vae_decoder=vae_decoder,
        stage=stage,
        step=0,
        panel_text_inputs=panel_text_inputs,
        panel_latents=panel_latents,
        expected_grids=expected_grids,
        panel_names=panel_names,
        groups=groups,
        swapped_seen_text_inputs=swapped_seen_text_inputs,
        swapped_seen_expected_grids=swapped_seen_expected_grids,
        compute_dtype=compute_dtype,
    )

    optimizer = build_flux_optimizer(run_config, transformer=transformer)
    step_path = REPORT_DIR / stage.name / "optimizer_steps.jsonl"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(
        range(1, stage.optimizer_steps + 1),
        desc=stage.name,
        unit="update",
    )
    records = []
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for step in progress:
        record = _optimizer_step(
            transformer=transformer,
            optimizer=optimizer,
            text_inputs=batch_text_inputs,
            clean_latents=batch_latents,
            condition_names=batch_names,
            step=step,
            compute_dtype=compute_dtype,
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
        if step in stage.evaluation_steps:
            _evaluate_stage(
                transformer=transformer,
                vae_decoder=vae_decoder,
                stage=stage,
                step=step,
                panel_text_inputs=panel_text_inputs,
                panel_latents=panel_latents,
                expected_grids=expected_grids,
                panel_names=panel_names,
                groups=groups,
                swapped_seen_text_inputs=swapped_seen_text_inputs,
                swapped_seen_expected_grids=swapped_seen_expected_grids,
                compute_dtype=compute_dtype,
            )

    checkpoint = _save_and_upload_checkpoint(
        transformer=transformer,
        stage=stage,
        volume_name=volume_name,
        run_id=run_id,
    )
    stage_report["training"] = {
        "completed_steps": len(records),
        "target_exposures": len(records) * OPTIMIZER_BATCH_SIZE,
        "exposures_per_condition": {
            name: sum(batch_name == name for batch_name in batch_names) * len(records)
            for name in stage.condition_names
        },
        "mean_loss": sum(item["loss"] for item in records) / len(records),
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "gradient_clipped_steps": sum(item["gradient_clipped"] for item in records),
        "mean_step_seconds": sum(item["seconds"] for item in records) / len(records),
        "timestep_strata": _summarize_timestep_strata(records),
        "memory": cuda_memory(f"{stage.name}-training-complete"),
    }
    stage_report["saved_checkpoint"] = checkpoint
    stage_report["status"] = "COMPLETE"
    save_report()
    del optimizer


def _validate_static_protocol() -> None:
    condition_names = tuple(condition.name for condition in CONDITION_SPECS)
    if len(condition_names) != 17 or len(set(condition_names)) != 17:
        raise RuntimeError("Protocol must define 17 unique conditions")
    if set(_TRAIN_CONDITION_NAMES).intersection(_UNSEEN_CONDITION_NAMES):
        raise RuntimeError("Seen and unseen condition names overlap")
    serialized_keys = tuple(str(condition.key()) for condition in CONDITION_SPECS)
    if len(set(serialized_keys)) != len(serialized_keys):
        raise RuntimeError("Protocol condition keys are not unique")
    for condition in CONDITION_SPECS:
        if condition.pair_indices[-1] != 4:
            raise RuntimeError(f"Condition does not target original pair 4: {condition.name}")
        if set(condition.pair_indices[:-1]) != {0, 1, 2, 3}:
            raise RuntimeError(f"Condition demo pairs differ: {condition.name}")
    if {condition.geometry for condition in CONDITION_SPECS[:8]} != set(ArcGeometry):
        raise RuntimeError("Eight-condition stage does not cover every geometry")
    if STAGES[0].condition_names != ("canonical",):
        raise RuntimeError("Stage 1 must use the canonical condition")
    if STAGES[1].condition_names != ("canonical", "train-id"):
        raise RuntimeError("Stage 2 must compare canonical and color-permuted ID")
    for stage in STAGES:
        if len(stage.batch_condition_names()) != OPTIMIZER_BATCH_SIZE:
            raise RuntimeError(f"Stage batch differs from size 8: {stage.name}")
        if stage.evaluation_steps[0] != 0:
            raise RuntimeError(f"Stage lacks step-zero evaluation: {stage.name}")
        if stage.evaluation_steps[-1] != stage.optimizer_steps:
            raise RuntimeError(f"Stage lacks final evaluation: {stage.name}")


def _summarize_stages() -> dict[str, Any]:
    result = {}
    for stage in STAGES:
        payload = REPORT["stages"][stage.name]
        final = payload["evaluations"][-1]
        result[stage.name] = {
            "seen_exact_grid_pass2": final["panels"]["seen"]["exact_grid_pass2"],
            "unseen_exact_grid_pass2": final["panels"]["unseen"]["exact_grid_pass2"],
            "canonical_exact_grid_pass2": final["panels"]["canonical"]["exact_grid_pass2"],
            "seen_mean_best_cell_accuracy": final["panels"]["seen"][
                "mean_best_cell_accuracy"
            ],
            "routing_probe": final["teacher_forced_probe"]["routing_summary"],
            "completed_optimizer_steps": payload["training"]["completed_steps"],
            "mean_step_seconds": payload["training"]["mean_step_seconds"],
            "peak_allocated_gib": payload["training"]["memory"][
                "peak_allocated_gib"
            ],
        }
    return result


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

    REPORT_DIR = volume_root / "results" / RESULT_FAMILY / args.run_id
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_PATH = REPORT_DIR / "report.json"
    REPORT = {
        "schema_version": 1,
        "status": "INITIALIZING",
        "run_id": args.run_id,
        "question": (
            "Can a fresh full FLUX transformer first memorize one fixed ARC "
            "condition, then route two contradictory conditions, and finally "
            "overfit eight repeated conditions when every occurrence receives "
            "fresh stratified flow time and Gaussian noise?"
        ),
        "protocol": {
            "task_id": TASK_ID,
            "target_original_pair_index": 4,
            "canvas": asdict(SPATIAL_SHAPE),
            "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
            "flow_timestep_sampling": "one-jittered-sample-per-batch-stratum",
            "flow_timestep_strata": OPTIMIZER_BATCH_SIZE,
            "probe_timesteps": list(PROBE_TIMESTEPS),
            "generation_seeds": list(GENERATION_SEEDS),
            "inference_steps": INFERENCE_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "stage_order": [stage.name for stage in STAGES],
            "stages": [asdict(stage) for stage in STAGES],
        },
        "paths": {
            "report": str(REPORT_PATH),
            "volume_root": str(volume_root),
            "flux_checkpoint": str(flux_checkpoint),
            "nvarc_checkpoint": str(nvarc_checkpoint),
        },
        "source": {"resolved_commit": args.source_commit},
        "failures": [],
        "stages": {
            stage.name: {"status": "PENDING", "evaluations": []}
            for stage in STAGES
        },
    }
    save_report()

    try:
        _validate_static_protocol()
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
        if (
            run_config.loop.max_optimizer_steps != 512
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

        task = load_training_task(arc_data_root / "training" / f"{TASK_ID}.json")
        samples_by_name = _build_condition_samples(task, tokenizer)
        condition_order = tuple(condition.name for condition in CONDITION_SPECS)
        ordered_samples = tuple(samples_by_name[name] for name in condition_order)
        condition_indices = {
            name: index for index, name in enumerate(condition_order)
        }
        all_ids, all_offsets = _pack_final_prefixes(ordered_samples)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        nvarc_started = time.perf_counter()
        nvarc_output = nvarc_runner(all_ids, all_offsets)
        nvarc_seconds = time.perf_counter() - nvarc_started
        all_hidden = nvarc_output.sequence_hidden_states
        if len(all_hidden) != len(condition_order):
            raise RuntimeError("NVARC condition count differs from protocol")
        single_ids, single_offsets = _pack_final_prefixes((ordered_samples[0],))
        single_hidden = nvarc_runner(single_ids, single_offsets).sequence_hidden_states[0]
        batch_hidden = all_hidden[0]
        cosine = float(
            F.cosine_similarity(
                single_hidden.float().reshape(1, -1),
                batch_hidden.float().reshape(1, -1),
            ).item()
        )
        mean_absolute_error = float(
            (single_hidden.float() - batch_hidden.float()).abs().mean().item()
        )
        all_text_inputs = pad_flux_text_prefixes(all_hidden)
        del nvarc_output, all_hidden, single_hidden, batch_hidden, nvarc_runner
        gc.collect()
        torch.cuda.empty_cache()

        grids_by_name = {
            name: samples_by_name[name].target_grids[3]
            for name in condition_order
        }
        all_grids = tuple(grids_by_name[name] for name in condition_order)
        serialized_targets = {
            json.dumps(grid, separators=(",", ":")) for grid in all_grids
        }
        if len(serialized_targets) != len(all_grids):
            raise RuntimeError("Condition target grids are not all unique")
        routing_cell_differences = sum(
            left != right
            for left_row, right_row in zip(
                grids_by_name["canonical"],
                grids_by_name["train-id"],
                strict=True,
            )
            for left, right in zip(left_row, right_row, strict=True)
        )
        routing_cell_count = sum(
            len(row) for row in grids_by_name["canonical"]
        )
        if routing_cell_differences != routing_cell_count:
            raise RuntimeError(
                "Stage 2 target grids must differ at every cell"
            )
        vae_started = time.perf_counter()
        all_latents = _encode_target_grids(all_grids, vae_encoder)
        vae_roundtrip = _vae_roundtrip(
            latents=all_latents,
            grids=all_grids,
            vae_decoder=vae_decoder,
        )
        del vae_encoder
        gc.collect()
        torch.cuda.empty_cache()

        parameter_count = sum(
            parameter.numel() for parameter in transformer.parameters()
        )
        base_fingerprint = _transformer_fingerprint(transformer)
        first_index = condition_indices["canonical"]
        preflight_indices = torch.full(
            (OPTIMIZER_BATCH_SIZE,),
            first_index,
            device=DEVICE,
            dtype=torch.int64,
        )
        preflight_latents = all_latents.index_select(0, preflight_indices)
        preflight_text = _slice_text_inputs(all_text_inputs, preflight_indices)
        with torch.no_grad():
            preflight_flow = sample_stratified_flow_matching_batch(
                preflight_latents,
                generator=torch.Generator(device=DEVICE).manual_seed(PROBE_SEED),
            )
            predicted = predict_flux_velocity(
                transformer,
                preflight_flow.noisy_latents,
                preflight_text,
                build_flux_image_position_ids(
                    OPTIMIZER_BATCH_SIZE,
                    DEVICE,
                    spatial_shape=SPATIAL_SHAPE,
                ),
                preflight_flow.timesteps,
                compute_dtype=compute_dtype,
            )
            preflight_loss = flow_matching_loss(
                predicted,
                preflight_flow.target_velocity,
            )
        if not bool(torch.isfinite(preflight_loss).item()):
            raise FloatingPointError("Conditioning-ladder preflight loss is non-finite")

        REPORT["conditions"] = {
            "order": list(condition_order),
            "count": len(condition_order),
            "items": {
                condition.name: {
                    "key": str(condition.key()),
                    "geometry": condition.geometry.value,
                    "color_mapping": list(condition.color_mapping),
                    "pair_indices": list(condition.pair_indices),
                    "target_grid": grids_by_name[condition.name],
                }
                for condition in CONDITION_SPECS
            },
            "all_target_original_pair_4": True,
            "all_target_mappings_directly_demonstrated": True,
            "all_target_grids_unique": True,
            "stage_2_target_cell_differences": routing_cell_differences,
            "stage_2_target_cell_count": routing_cell_count,
        }
        REPORT["nvarc_precompute"] = {
            "physical_batch_size": len(condition_order),
            "seconds": nvarc_seconds,
            "batch1_vs_batch17_cosine_similarity": cosine,
            "batch1_vs_batch17_mean_absolute_error": mean_absolute_error,
            "text_shape": list(all_text_inputs.encoder_hidden_states.shape),
            "memory": cuda_memory("after-nvarc-precompute"),
        }
        REPORT["vae_precompute"] = {
            "latent_shape": list(all_latents.shape),
            "seconds": time.perf_counter() - vae_started,
            "roundtrip": vae_roundtrip,
        }
        REPORT["models"] = {
            "flux_parameter_count": parameter_count,
            "base_fingerprint": base_fingerprint,
            "preflight_loss": float(preflight_loss.item()),
            "memory_after_preflight": cuda_memory("after-ladder-preflight"),
        }
        REPORT["resolved_training_config"] = json.loads(
            json.dumps(asdict(run_config), default=str)
        )
        save_report()

        for stage_index, stage in enumerate(STAGES):
            if stage_index > 0:
                transformer = _load_fresh_transformer(
                    flux_checkpoint,
                    expected_parameter_count=parameter_count,
                )
            fresh_fingerprint = _transformer_fingerprint(transformer)
            if fresh_fingerprint != base_fingerprint:
                raise RuntimeError(
                    f"{stage.name} transformer does not match the base checkpoint"
                )
            _run_stage(
                transformer=transformer,
                stage=stage,
                run_config=run_config,
                all_text_inputs=all_text_inputs,
                all_latents=all_latents,
                condition_indices=condition_indices,
                grids_by_name=grids_by_name,
                vae_decoder=vae_decoder,
                compute_dtype=compute_dtype,
                volume_name=args.volume_name,
                run_id=args.run_id,
            )
            del transformer
            gc.collect()
            torch.cuda.empty_cache()

        REPORT["comparison"] = _summarize_stages()
        REPORT["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT["status"] = "COMPLETE"
        save_report()
        _upload_and_verify_final_report(
            volume_name=args.volume_name,
            run_id=args.run_id,
        )
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
