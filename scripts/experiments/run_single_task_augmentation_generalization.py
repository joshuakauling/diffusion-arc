#!/usr/bin/env python3
"""Run or continue the 512-view augmentation-generalization experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import platform
from random import Random
import shutil
import sys
import time
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcEvaluationTask,
    ArcGeometry,
    ArcQuery,
    ArcTokenizedTrainingSample,
    ArcTrainingTask,
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
from diffusion_arc.inference import (
    continue_arc_batch,
    diagnose_exact_grids,
    linear_timestep_schedule,
)
from diffusion_arc.model import (
    FlowMatchingBatch,
    FluxSpatialShape,
    FluxTextInputs,
    TARGET_LATENT_CHANNELS,
    build_flux_image_position_ids,
    flow_matching_loss,
    pad_flux_text_prefixes,
    predict_flux_velocity,
)
from diffusion_arc.training import FluxMicrobatch
from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from diffusion_arc.training.single_gpu_setup import (
    build_flux_optimizer,
    load_single_gpu_models,
    prepare_single_gpu_device,
    seed_single_gpu_process,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "augmentation-generalization-overfit"
CONTINUATION_PROTOCOL = "augmentation-generalization-overfit-continuation"
REGION_WEIGHTED_PROTOCOL = "augmentation-generalization-region-weighted-overfit"
REGION_WEIGHTED_CONTINUATION_PROTOCOL = (
    "augmentation-generalization-region-weighted-overfit-continuation"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL = (
    "augmentation-generalization-region-weighted-endpoint-balanced-continuation"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL = (
    "augmentation-generalization-region-weighted-endpoint-balanced-long-continuation"
)
RESULT_FAMILY = "single-task-augmentation-generalization-overfitting"
STAGE_NAME = "augmentation-generalization-overfit"
CONFIG_NAME = "single_gpu_0d3d703e_augmentation_generalization.json"
CONTINUATION_CONFIG_NAME = (
    "single_gpu_0d3d703e_augmentation_generalization_continuation.json"
)
REGION_WEIGHTED_CONTINUATION_CONFIG_NAME = (
    "single_gpu_0d3d703e_region_weighted_continuation.json"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_CONFIG_NAME = (
    "single_gpu_0d3d703e_region_weighted_endpoint_balanced_continuation.json"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_CONFIG_NAME = (
    "single_gpu_0d3d703e_region_weighted_endpoint_balanced_long_continuation.json"
)
TASK_ID = "0d3d703e"
TARGET_ORIGINAL_PAIR_INDEX = 4
DEMONSTRATION_INDICES = (0, 1, 2, 3)
DEMONSTRATION_ORDERS = tuple(permutations(DEMONSTRATION_INDICES))
EFFECTIVE_GEOMETRIES = (
    ArcGeometry.IDENTITY,
    ArcGeometry.ROTATE_90,
    ArcGeometry.ROTATE_180,
    ArcGeometry.ROTATE_270,
)
REDUNDANT_GEOMETRY_CLASSES = {
    ArcGeometry.IDENTITY: (ArcGeometry.IDENTITY, ArcGeometry.FLIP_HORIZONTAL),
    ArcGeometry.ROTATE_90: (
        ArcGeometry.ROTATE_90,
        ArcGeometry.FLIP_MAIN_DIAGONAL,
    ),
    ArcGeometry.ROTATE_180: (
        ArcGeometry.ROTATE_180,
        ArcGeometry.FLIP_VERTICAL,
    ),
    ArcGeometry.ROTATE_270: (
        ArcGeometry.ROTATE_270,
        ArcGeometry.FLIP_ANTI_DIAGONAL,
    ),
}

TRAIN_PER_GEOMETRY = 128
HOLDOUT_PER_GEOMETRY = 32
TRAIN_VIEW_COUNT = TRAIN_PER_GEOMETRY * len(EFFECTIVE_GEOMETRIES)
HOLDOUT_VIEW_COUNT = HOLDOUT_PER_GEOMETRY * len(EFFECTIVE_GEOMETRIES)
ORIGINAL_VIEW_COUNT = 1
CANDIDATES_PER_GEOMETRY = 320
CANDIDATE_SEED = 3167
CANDIDATE_MAX_ATTEMPTS_PER_GEOMETRY = 100_000
MAX_TOKENS = 192

OPTIMIZER_BATCH_SIZE = 8
OPTIMIZER_STEPS = 512
CONTINUATION_START_STEP = 512
CONTINUATION_FINAL_STEP = 2048
CONTINUATION_OPTIMIZER_STEPS = CONTINUATION_FINAL_STEP - CONTINUATION_START_STEP
CONTINUATION_PHASE_START = CONTINUATION_START_STEP // OPTIMIZER_STEPS
CONTINUATION_PHASE_COUNT = CONTINUATION_OPTIMIZER_STEPS // OPTIMIZER_STEPS
CONTINUATION_PARENT_RUN_ID = "20260826T155954304798Z"
CONTINUATION_PARENT_SOURCE_COMMIT = "6beb492787a5c2c031c8cc2004caa6a7412517f6"
CONTINUATION_PARENT_REPORT_SHA256 = (
    "0480c43fcf7349614a79a4da33f780dd930484cbae7cf06c8b24248d35c7d5cf"
)
CONTINUATION_PARENT_POOL_SHA256 = (
    "f512b3dc4cc5f603c3c764a45b3a4086d4b4837e63a2793a7009c2505e72e0b0"
)
CONTINUATION_PARENT_SCHEDULE_SHA256 = (
    "eac20d584ecb44e78ba7dc22f6fbc7033ee0c5eac12e704affe1287e30636947"
)
REGION_WEIGHTED_CONTINUATION_START_STEP = 512
REGION_WEIGHTED_CONTINUATION_FINAL_STEP = 1536
REGION_WEIGHTED_CONTINUATION_OPTIMIZER_STEPS = (
    REGION_WEIGHTED_CONTINUATION_FINAL_STEP
    - REGION_WEIGHTED_CONTINUATION_START_STEP
)
REGION_WEIGHTED_CONTINUATION_PHASE_START = (
    REGION_WEIGHTED_CONTINUATION_START_STEP // OPTIMIZER_STEPS
)
REGION_WEIGHTED_CONTINUATION_PHASE_COUNT = (
    REGION_WEIGHTED_CONTINUATION_OPTIMIZER_STEPS // OPTIMIZER_STEPS
)
REGION_WEIGHTED_CONTINUATION_PARENT_RUN_ID = "20260826T215440643526Z"
REGION_WEIGHTED_CONTINUATION_PARENT_SOURCE_COMMIT = (
    "ae3c87c68e2ca1e2248bbf614da6d878d8902a57"
)
REGION_WEIGHTED_CONTINUATION_PARENT_REPORT_SHA256 = (
    "c7757b5b4976d1c3f7305641593e000c8c3d87aa149a4df8eb60803e379e5f38"
)
REGION_WEIGHTED_CONTINUATION_PARENT_POOL_SHA256 = (
    "f512b3dc4cc5f603c3c764a45b3a4086d4b4837e63a2793a7009c2505e72e0b0"
)
REGION_WEIGHTED_CONTINUATION_PARENT_SCHEDULE_SHA256 = (
    "eac20d584ecb44e78ba7dc22f6fbc7033ee0c5eac12e704affe1287e30636947"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP = 1536
REGION_WEIGHTED_ENDPOINT_BALANCED_FINAL_STEP = 2048
REGION_WEIGHTED_ENDPOINT_BALANCED_OPTIMIZER_STEPS = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_FINAL_STEP
    - REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_START = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP // OPTIMIZER_STEPS
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_COUNT = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_OPTIMIZER_STEPS // OPTIMIZER_STEPS
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_RUN_ID = "20260826T231644245954Z"
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_SOURCE_COMMIT = (
    "5a1bcf69e7b119f7714396ca6ae6da868a810e9c"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_REPORT_SHA256 = (
    "3011769bad7a007308fd457208c24b56978561064c45414366f5f186067937e7"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_POOL_SHA256 = (
    "f512b3dc4cc5f603c3c764a45b3a4086d4b4837e63a2793a7009c2505e72e0b0"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_SCHEDULE_SHA256 = (
    "74b1055519639ef11e0af0b3bdf8ed43acd6d15f5eff8cd20e576840ae3ee38a"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_SCHEDULE_SHA256 = (
    "f03254d7822fcc2343371f50b6bd3bccd6039543ee35b85f2c68b81dec16b5d4"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP = 2048
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FINAL_STEP = 4096
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_OPTIMIZER_STEPS = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FINAL_STEP
    - REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_START = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP // OPTIMIZER_STEPS
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_COUNT = (
    REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_OPTIMIZER_STEPS // OPTIMIZER_STEPS
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_RUN_ID = "20260827T003543625091Z"
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_SOURCE_COMMIT = (
    "974f5f519a63d56752a80e59318ac2ff548787ab"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_REPORT_SHA256 = (
    "1edcc75976965aeafcf2bc0127e922692a1bbfa29cb139d5075ffce3f875f89d"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_POOL_SHA256 = (
    "f512b3dc4cc5f603c3c764a45b3a4086d4b4837e63a2793a7009c2505e72e0b0"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_SCHEDULE_SHA256 = (
    "f03254d7822fcc2343371f50b6bd3bccd6039543ee35b85f2c68b81dec16b5d4"
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_SCHEDULE_SHA256 = (
    "f54353695a2dbf66dea7bb9d80ade45cb2f40e6f60411e8a3c46658201b82cd7"
)
REGION_WEIGHTED_PARENT_TERMINAL_FAILURE = {
    "stage": "experiment",
    "type": "ExecutionError",
    "message": "Request failed with status 404 Not Found: block not found",
}
UPDATE_CYCLE_LENGTH = 8
ENDPOINT_UPDATE_COUNT = OPTIMIZER_STEPS // UPDATE_CYCLE_LENGTH
REGULAR_UPDATE_COUNT = OPTIMIZER_STEPS - ENDPOINT_UPDATE_COUNT
EXPOSURES_PER_VIEW = 8
DEFAULT_ENDPOINT_CYCLE_POSITIONS = (0,)
ENDPOINT_BALANCED_CYCLE_POSITIONS = (0, 2, 4, 6)
EVALUATION_STEPS = (0, 128, 256, 512)
CONTINUATION_EVALUATION_STEPS = (1024, 2048)
CONTINUATION_FULL_GENERATION_STEPS = (2048,)
REGION_WEIGHTED_EVALUATION_STEPS = EVALUATION_STEPS
REGION_WEIGHTED_FULL_GENERATION_STEPS = (512,)
REGION_WEIGHTED_CONTINUATION_EVALUATION_STEPS = (1024, 1536)
REGION_WEIGHTED_CONTINUATION_FULL_GENERATION_STEPS = (1536,)
REGION_WEIGHTED_ENDPOINT_BALANCED_EVALUATION_STEPS = (1664, 1792, 2048)
REGION_WEIGHTED_ENDPOINT_BALANCED_FULL_GENERATION_STEPS = (2048,)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_EVALUATION_STEPS = (4096,)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FULL_GENERATION_STEPS = (4096,)
UNWEIGHTED_LOSS_POLICY = "unweighted-flow-matching"
REGION_WEIGHTED_LOSS_POLICY = "region-weighted-flow-matching"
REGION_LOSS_WEIGHTS = {
    "grid": 0.6,
    "boundary": 0.3,
    "outer_padding": 0.1,
}
INTERMEDIATE_SEEN_PER_GEOMETRY = 8
FINAL_SEEN_PER_GEOMETRY = 32
INTERMEDIATE_HOLDOUT_PER_GEOMETRY = 8
GENERATION_SEEDS = (167, 168)
INFERENCE_STEPS = 50
GENERATION_BATCH_SIZE = 8
VAE_ENCODE_BATCH_SIZE = 16
VAE_DECODE_BATCH_SIZE = 8
NVARC_TRAIN_BATCH_SIZE = TRAIN_VIEW_COUNT
NVARC_AUXILIARY_BATCH_SIZE = HOLDOUT_VIEW_COUNT + ORIGINAL_VIEW_COUNT
SPATIAL_SHAPE = FluxSpatialShape(height=30, width=30)
PROBE_SEED = 1167
FLOW_SEED = 2167
MAX_GRAD_NORM = 1.0
CHECKPOINT_MAX_SHARD_SIZE = "2GB"
MINIMUM_VRAM_GIB = 120.0
MINIMUM_LOCAL_CHECKPOINT_FREE_GIB = 24.0
PROGRESS_EVERY = 64
SHEET_PAGE_SIZE = 16
FINAL_REPORT_UPLOAD_ATTEMPTS = 3
FINAL_REPORT_RETRY_DELAYS_SECONDS = (1.0, 2.0)


REPORT: dict[str, Any]
REPORT_PATH: Path
REPORT_DIR: Path
DEVICE: torch.device
REGION_MASK_CACHE: dict[
    tuple[int, int, str, int | None],
    tuple[Tensor, Tensor, Tensor, Tensor],
] = {}


@dataclass(frozen=True)
class PreparedView:
    """One fully validated candidate before its VAE eligibility is known."""

    candidate_index: int
    geometry: ArcGeometry
    key: ArcAugmentationKey
    sample: ArcTokenizedTrainingSample
    query_grid: Any
    target_grid: Any
    prefix_hash: str
    query_hash: str
    target_hash: str
    full_view_hash: str

    @property
    def serialized_key(self) -> str:
        return str(self.key)

    @property
    def pair_order(self) -> tuple[int, ...]:
        return self.key.augmentation.pair_indices[:-1]

    @property
    def color_mapping(self) -> tuple[int, ...]:
        return self.key.augmentation.color_mapping


@dataclass(frozen=True)
class TrainingBatchPlan:
    """The exact rows and regular strata used by one optimizer update."""

    step: int
    cycle_index: int
    cycle_position: int
    kind: str
    view_indices: tuple[int, ...]
    timestep_strata: tuple[int, ...] | None


@dataclass(frozen=True)
class ParentRunContract:
    """Immutable lineage requirements for a transformer-only continuation."""

    run_id: str
    source_commit: str
    report_sha256: str
    pool_sha256: str
    schedule_sha256: str
    protocol_name: str
    loss_policy: str
    optimizer_steps: int
    final_step: int
    accept_terminal_report_upload_failure: bool = False


UNWEIGHTED_PARENT_CONTRACT = ParentRunContract(
    run_id=CONTINUATION_PARENT_RUN_ID,
    source_commit=CONTINUATION_PARENT_SOURCE_COMMIT,
    report_sha256=CONTINUATION_PARENT_REPORT_SHA256,
    pool_sha256=CONTINUATION_PARENT_POOL_SHA256,
    schedule_sha256=CONTINUATION_PARENT_SCHEDULE_SHA256,
    protocol_name=PROTOCOL,
    loss_policy=UNWEIGHTED_LOSS_POLICY,
    optimizer_steps=OPTIMIZER_STEPS,
    final_step=CONTINUATION_START_STEP,
)
REGION_WEIGHTED_PARENT_CONTRACT = ParentRunContract(
    run_id=REGION_WEIGHTED_CONTINUATION_PARENT_RUN_ID,
    source_commit=REGION_WEIGHTED_CONTINUATION_PARENT_SOURCE_COMMIT,
    report_sha256=REGION_WEIGHTED_CONTINUATION_PARENT_REPORT_SHA256,
    pool_sha256=REGION_WEIGHTED_CONTINUATION_PARENT_POOL_SHA256,
    schedule_sha256=REGION_WEIGHTED_CONTINUATION_PARENT_SCHEDULE_SHA256,
    protocol_name=REGION_WEIGHTED_PROTOCOL,
    loss_policy=REGION_WEIGHTED_LOSS_POLICY,
    optimizer_steps=OPTIMIZER_STEPS,
    final_step=REGION_WEIGHTED_CONTINUATION_START_STEP,
    accept_terminal_report_upload_failure=True,
)
REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_CONTRACT = ParentRunContract(
    run_id=REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_RUN_ID,
    source_commit=REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_SOURCE_COMMIT,
    report_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_REPORT_SHA256,
    pool_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_POOL_SHA256,
    schedule_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_SCHEDULE_SHA256,
    protocol_name=REGION_WEIGHTED_CONTINUATION_PROTOCOL,
    loss_policy=REGION_WEIGHTED_LOSS_POLICY,
    optimizer_steps=REGION_WEIGHTED_CONTINUATION_OPTIMIZER_STEPS,
    final_step=REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP,
)
REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_CONTRACT = ParentRunContract(
    run_id=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_RUN_ID,
    source_commit=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_SOURCE_COMMIT,
    report_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_REPORT_SHA256,
    pool_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_POOL_SHA256,
    schedule_sha256=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_SCHEDULE_SHA256,
    protocol_name=REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL,
    loss_policy=REGION_WEIGHTED_LOSS_POLICY,
    optimizer_steps=REGION_WEIGHTED_ENDPOINT_BALANCED_OPTIMIZER_STEPS,
    final_step=REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--parent-report", type=Path)
    parser.add_argument("--parent-report-sha256")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _stable_seed(*parts: Any) -> int:
    material = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (2**63 - 1)


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
    samples: Sequence[ArcTokenizedTrainingSample],
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


def _direct_mapping_pairs(pairs: Sequence[Any]) -> set[tuple[int, int]]:
    observed: set[tuple[int, int]] = set()
    for pair in pairs:
        for input_row, output_row in zip(
            pair.input_grid,
            pair.output_grid,
            strict=True,
        ):
            observed.update(zip(input_row, output_row, strict=True))
    return observed


def _candidate_augmentation(
    geometry: ArcGeometry,
    candidate_index: int,
) -> ArcAugmentationSpec:
    if geometry not in EFFECTIVE_GEOMETRIES:
        raise ValueError(f"Unsupported effective geometry: {geometry}")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise TypeError("candidate_index must be an integer")
    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    geometry_index = EFFECTIVE_GEOMETRIES.index(geometry)
    rng = Random(
        _stable_seed(
            "augmentation-generalization-candidate",
            CANDIDATE_SEED,
            geometry.value,
            candidate_index,
        )
    )
    color_mapping = list(range(10))
    rng.shuffle(color_mapping)
    pair_order_index = (candidate_index + geometry_index * 7) % len(
        DEMONSTRATION_ORDERS
    )
    return ArcAugmentationSpec(
        geometry=geometry,
        color_mapping=tuple(color_mapping),
        pair_indices=(
            *DEMONSTRATION_ORDERS[pair_order_index],
            TARGET_ORIGINAL_PAIR_INDEX,
        ),
    )


def _build_tokenized_view(
    task: ArcTrainingTask,
    tokenizer: Any,
    *,
    augmentation: ArcAugmentationSpec,
    candidate_index: int,
) -> PreparedView:
    key = ArcAugmentationKey(task_id=task.task_id, augmentation=augmentation)
    validate_augmentation_for_task(key, task)
    if augmentation.pair_indices[-1] != TARGET_ORIGINAL_PAIR_INDEX:
        raise RuntimeError(f"View does not keep original pair 4 as target: {key}")
    if set(augmentation.pair_indices[:-1]) != set(DEMONSTRATION_INDICES):
        raise RuntimeError(f"View demonstrations differ from pairs 0..3: {key}")
    augmented = apply_training_augmentation(task, augmentation)
    query_pair = augmented.pairs[-1]
    required_mapping = _direct_mapping_pairs((query_pair,))
    demonstrated_mapping = _direct_mapping_pairs(augmented.pairs[:-1])
    if not required_mapping.issubset(demonstrated_mapping):
        missing = sorted(required_mapping - demonstrated_mapping)
        raise RuntimeError(
            f"View lacks directly demonstrated mappings {missing}: {key}"
        )

    sample = tokenize_nvarc_training_sample(
        render_training_prompt(augmented),
        tokenizer,
        max_tokens=MAX_TOKENS,
    )
    evaluation_task = ArcEvaluationTask(
        task_id=str(key),
        demonstrations=augmented.pairs[:-1],
        queries=(ArcQuery(query_index=0, input_grid=query_pair.input_grid),),
    )
    evaluation_tokens = tokenize_nvarc_evaluation_sample(
        render_evaluation_prompts(evaluation_task)[0],
        tokenizer,
        max_tokens=MAX_TOKENS,
    )
    prefix = sample.input_ids[: sample.prefix_token_ends[3]]
    if evaluation_tokens.input_ids != prefix:
        raise RuntimeError(f"Target-free prefix differs for {key}")
    if sample.target_grids[3] != query_pair.output_grid:
        raise RuntimeError(f"Final target differs for {key}")

    full_view_payload = [
        {"input": pair.input_grid, "output": pair.output_grid}
        for pair in augmented.pairs
    ]
    return PreparedView(
        candidate_index=candidate_index,
        geometry=augmentation.geometry,
        key=key,
        sample=sample,
        query_grid=query_pair.input_grid,
        target_grid=query_pair.output_grid,
        prefix_hash=_hash_json(prefix),
        query_hash=_hash_json(query_pair.input_grid),
        target_hash=_hash_json(query_pair.output_grid),
        full_view_hash=_hash_json(full_view_payload),
    )


def _semantic_collision_reason(
    view: PreparedView,
    seen: dict[str, set[Any]],
) -> str | None:
    checks = (
        ("key", view.serialized_key),
        ("color_mapping", view.color_mapping),
        ("prefix", view.prefix_hash),
        ("query", view.query_hash),
        ("target", view.target_hash),
        ("full_view", view.full_view_hash),
    )
    for name, value in checks:
        if value in seen[name]:
            return name
    return None


def _register_semantic_fingerprints(
    view: PreparedView,
    seen: dict[str, set[Any]],
) -> None:
    seen["key"].add(view.serialized_key)
    seen["color_mapping"].add(view.color_mapping)
    seen["prefix"].add(view.prefix_hash)
    seen["query"].add(view.query_hash)
    seen["target"].add(view.target_hash)
    seen["full_view"].add(view.full_view_hash)


def _build_candidate_registry(
    task: ArcTrainingTask,
    tokenizer: Any,
) -> tuple[PreparedView, tuple[PreparedView, ...], dict[str, Any]]:
    if task.task_id != TASK_ID or len(task.pairs) != 5:
        raise RuntimeError(
            f"Expected five-pair task {TASK_ID}, received {task.task_id} "
            f"with {len(task.pairs)} pairs"
        )
    original = _build_tokenized_view(
        task,
        tokenizer,
        augmentation=ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=(*DEMONSTRATION_INDICES, TARGET_ORIGINAL_PAIR_INDEX),
        ),
        candidate_index=-1,
    )
    seen = {
        name: set()
        for name in ("key", "color_mapping", "prefix", "query", "target", "full_view")
    }
    _register_semantic_fingerprints(original, seen)
    candidates: list[PreparedView] = []
    rejection_counts: Counter[str] = Counter()
    attempts_by_geometry: dict[str, int] = {}

    for geometry in EFFECTIVE_GEOMETRIES:
        geometry_started = time.perf_counter()
        accepted = 0
        candidate_index = 0
        while accepted < CANDIDATES_PER_GEOMETRY:
            if candidate_index >= CANDIDATE_MAX_ATTEMPTS_PER_GEOMETRY:
                raise RuntimeError(
                    f"Could not build {CANDIDATES_PER_GEOMETRY} disjoint "
                    f"candidates for {geometry.value}"
                )
            augmentation = _candidate_augmentation(geometry, candidate_index)
            view = _build_tokenized_view(
                task,
                tokenizer,
                augmentation=augmentation,
                candidate_index=candidate_index,
            )
            candidate_index += 1
            collision = _semantic_collision_reason(view, seen)
            if collision is not None:
                rejection_counts[f"semantic_{collision}"] += 1
                continue
            _register_semantic_fingerprints(view, seen)
            candidates.append(view)
            accepted += 1
        attempts_by_geometry[geometry.value] = candidate_index
        print(
            f"[{STAGE_NAME}] candidate pool {geometry.value}: "
            f"{accepted}/{CANDIDATES_PER_GEOMETRY} accepted from "
            f"{candidate_index} attempts in "
            f"{time.perf_counter() - geometry_started:.1f}s"
        )

    return (
        original,
        tuple(candidates),
        {
            "candidate_seed": CANDIDATE_SEED,
            "candidates_per_geometry": CANDIDATES_PER_GEOMETRY,
            "candidate_count": len(candidates),
            "attempts_by_geometry": attempts_by_geometry,
            "semantic_rejection_counts": dict(sorted(rejection_counts.items())),
        },
    )


def _balanced_available_selection(
    candidates: Sequence[PreparedView],
    *,
    count: int,
    rotation: int,
    seed_label: str,
) -> list[PreparedView]:
    if count < 0:
        raise ValueError("count must be non-negative")
    selected: list[PreparedView] = []
    by_order: dict[tuple[int, ...], list[PreparedView]] = defaultdict(list)
    for view in candidates:
        by_order[view.pair_order].append(view)
    for available in by_order.values():
        available.sort(key=lambda view: _stable_seed(seed_label, view.serialized_key))
    rotated_orders = DEMONSTRATION_ORDERS[rotation:] + DEMONSTRATION_ORDERS[:rotation]
    cursors = {order: 0 for order in DEMONSTRATION_ORDERS}
    while len(selected) < count:
        made_progress = False
        for order in rotated_orders:
            cursor = cursors[order]
            available = by_order[order]
            if cursor >= len(available):
                continue
            selected.append(available[cursor])
            cursors[order] += 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise RuntimeError(
                f"Only {len(selected)} VAE-safe candidates are available, "
                f"requires {count}"
            )
    return selected


def _select_safe_pool(
    candidates: Sequence[PreparedView],
    vae_safe_keys: set[str],
) -> tuple[tuple[PreparedView, ...], tuple[PreparedView, ...], dict[str, Any]]:
    train: list[PreparedView] = []
    holdout: list[PreparedView] = []
    safe_counts: dict[str, int] = {}
    for geometry_index, geometry in enumerate(EFFECTIVE_GEOMETRIES):
        safe = [
            view
            for view in candidates
            if view.geometry is geometry and view.serialized_key in vae_safe_keys
        ]
        safe_counts[geometry.value] = len(safe)
        combined = _balanced_available_selection(
            safe,
            count=TRAIN_PER_GEOMETRY + HOLDOUT_PER_GEOMETRY,
            rotation=(geometry_index * 5) % len(DEMONSTRATION_ORDERS),
            seed_label=f"combined-pool-{geometry.value}",
        )
        geometry_holdout = _balanced_available_selection(
            combined,
            count=HOLDOUT_PER_GEOMETRY,
            rotation=(geometry_index * 5 + 11) % len(DEMONSTRATION_ORDERS),
            seed_label=f"holdout-pool-{geometry.value}",
        )
        holdout_keys = {view.serialized_key for view in geometry_holdout}
        geometry_train = [
            view for view in combined if view.serialized_key not in holdout_keys
        ]
        if len(geometry_train) != TRAIN_PER_GEOMETRY:
            raise RuntimeError("Combined train/holdout allocation differs")
        train.extend(geometry_train)
        holdout.extend(geometry_holdout)

    train.sort(
        key=lambda view: (
            EFFECTIVE_GEOMETRIES.index(view.geometry),
            view.candidate_index,
        )
    )
    holdout.sort(
        key=lambda view: (
            EFFECTIVE_GEOMETRIES.index(view.geometry),
            view.candidate_index,
        )
    )
    if len(train) != TRAIN_VIEW_COUNT or len(holdout) != HOLDOUT_VIEW_COUNT:
        raise RuntimeError("Selected pool counts differ from protocol")
    _validate_selected_pool(tuple(train), tuple(holdout))
    return (
        tuple(train),
        tuple(holdout),
        {
            "vae_safe_candidates_by_geometry": safe_counts,
            "train_pair_order_counts": _pair_order_counts(train),
            "holdout_pair_order_counts": _pair_order_counts(holdout),
        },
    )


def _pair_order_counts(views: Sequence[PreparedView]) -> dict[str, int]:
    counts = Counter(
        ",".join(str(index) for index in view.pair_order) for view in views
    )
    return dict(sorted(counts.items()))


def _validate_selected_pool(
    train: tuple[PreparedView, ...],
    holdout: tuple[PreparedView, ...],
) -> None:
    if len(train) != TRAIN_VIEW_COUNT or len(holdout) != HOLDOUT_VIEW_COUNT:
        raise RuntimeError("Pool counts differ from protocol")
    for name, views, expected_per_geometry in (
        ("train", train, TRAIN_PER_GEOMETRY),
        ("holdout", holdout, HOLDOUT_PER_GEOMETRY),
    ):
        geometry_counts = Counter(view.geometry for view in views)
        if geometry_counts != Counter(
            {geometry: expected_per_geometry for geometry in EFFECTIVE_GEOMETRIES}
        ):
            raise RuntimeError(f"{name} geometry counts differ: {geometry_counts}")
        if any(view.key.augmentation.pair_indices[-1] != 4 for view in views):
            raise RuntimeError(f"{name} contains a non-final-pair-4 target")

    combined = train + holdout
    fields = {
        "key": [view.serialized_key for view in combined],
        "color_mapping": [view.color_mapping for view in combined],
        "prefix": [view.prefix_hash for view in combined],
        "query": [view.query_hash for view in combined],
        "target": [view.target_hash for view in combined],
        "full_view": [view.full_view_hash for view in combined],
    }
    for name, values in fields.items():
        if len(set(values)) != len(values):
            raise RuntimeError(f"Selected train/holdout {name} values overlap")


def _balanced_subset_indices(
    views: Sequence[PreparedView],
    *,
    per_geometry: int,
    seed_label: str,
) -> tuple[int, ...]:
    selected: list[int] = []
    for geometry in EFFECTIVE_GEOMETRIES:
        candidates = [
            index for index, view in enumerate(views) if view.geometry is geometry
        ]
        ranked = sorted(
            candidates,
            key=lambda index: _stable_seed(
                seed_label,
                views[index].serialized_key,
            ),
        )
        if len(ranked) < per_geometry:
            raise RuntimeError(
                f"{geometry.value} has only {len(ranked)} evaluation candidates"
            )
        selected.extend(ranked[:per_geometry])
    return tuple(selected)


def _build_training_schedule(
    train: Sequence[PreparedView],
    *,
    phase_start: int = 0,
    phase_count: int = 1,
    endpoint_cycle_positions: tuple[int, ...] = DEFAULT_ENDPOINT_CYCLE_POSITIONS,
    prior_exposure_audit: dict[str, Any] | None = None,
) -> tuple[TrainingBatchPlan, ...]:
    if phase_start < 0:
        raise ValueError("phase_start must be non-negative")
    if phase_count <= 0:
        raise ValueError("phase_count must be positive")
    if (
        not endpoint_cycle_positions
        or tuple(sorted(set(endpoint_cycle_positions))) != endpoint_cycle_positions
        or any(
            position < 0 or position >= UPDATE_CYCLE_LENGTH
            for position in endpoint_cycle_positions
        )
    ):
        raise ValueError(
            "endpoint_cycle_positions must be sorted, unique, and inside the cycle"
        )
    if len(train) != TRAIN_VIEW_COUNT:
        raise ValueError(f"train must contain {TRAIN_VIEW_COUNT} views")
    by_geometry = {
        geometry: [
            index for index, view in enumerate(train) if view.geometry is geometry
        ]
        for geometry in EFFECTIVE_GEOMETRIES
    }
    if any(len(indices) != TRAIN_PER_GEOMETRY for indices in by_geometry.values()):
        raise ValueError("Training geometry counts differ from protocol")

    cumulative_stratum_counts: dict[str, Counter[int]] | None = None
    phase_stratum_counts: dict[str, Counter[int]] | None = None
    if prior_exposure_audit is not None:
        prior_items = prior_exposure_audit.get("items")
        expected_keys = {view.serialized_key for view in train}
        if not isinstance(prior_items, dict) or set(prior_items) != expected_keys:
            raise ValueError("prior_exposure_audit keys differ from training views")
        cumulative_stratum_counts = {
            key: Counter(item.get("regular_strata", ()))
            for key, item in prior_items.items()
        }
        phase_stratum_counts = {key: Counter() for key in prior_items}

    groups: list[tuple[int, ...]] = []
    for group_index in range(ENDPOINT_UPDATE_COUNT):
        slots: list[int | None] = [None] * OPTIMIZER_BATCH_SIZE
        for geometry_index, geometry in enumerate(EFFECTIVE_GEOMETRIES):
            pair = by_geometry[geometry][2 * group_index : 2 * group_index + 2]
            for local_index, view_index in enumerate(pair):
                base_slot = 2 * geometry_index + local_index
                slot = (base_slot + group_index) % OPTIMIZER_BATCH_SIZE
                if slots[slot] is not None:
                    raise AssertionError("Training group slot collision")
                slots[slot] = view_index
        if any(index is None for index in slots):
            raise AssertionError("Training group has an empty slot")
        groups.append(tuple(int(index) for index in slots))

    plans: list[TrainingBatchPlan] = []
    for phase_index in range(phase_start, phase_start + phase_count):
        group_rotation = (17 * phase_index) % ENDPOINT_UPDATE_COUNT
        for local_cycle_index in range(ENDPOINT_UPDATE_COUNT):
            cycle_index = phase_index * ENDPOINT_UPDATE_COUNT + local_cycle_index
            for cycle_position in range(UPDATE_CYCLE_LENGTH):
                step = cycle_index * UPDATE_CYCLE_LENGTH + cycle_position + 1
                group_index = (
                    local_cycle_index
                    + group_rotation
                    + cycle_position * OPTIMIZER_BATCH_SIZE
                ) % ENDPOINT_UPDATE_COUNT
                if cycle_position in endpoint_cycle_positions:
                    plans.append(
                        TrainingBatchPlan(
                            step=step,
                            cycle_index=cycle_index,
                            cycle_position=cycle_position,
                            kind="shared_endpoint",
                            view_indices=groups[group_index],
                            timestep_strata=None,
                        )
                    )
                    continue
                if cumulative_stratum_counts is None:
                    timestep_strata = tuple(
                        (slot + cycle_position + phase_index)
                        % OPTIMIZER_BATCH_SIZE
                        for slot in range(OPTIMIZER_BATCH_SIZE)
                    )
                else:
                    if phase_stratum_counts is None:
                        raise AssertionError("Balanced schedule lacks phase counters")
                    states: dict[
                        int, tuple[int, int, int, tuple[int, ...]]
                    ] = {
                        0: (0, 0, 0, ())
                    }
                    for row, view_index in enumerate(groups[group_index]):
                        key = train[view_index].serialized_key
                        next_states: dict[
                            int, tuple[int, int, int, tuple[int, ...]]
                        ] = {}
                        for mask, (
                            phase_cost,
                            cumulative_cost,
                            tie_cost,
                            assignment,
                        ) in states.items():
                            for stratum in range(OPTIMIZER_BATCH_SIZE):
                                if mask & (1 << stratum):
                                    continue
                                candidate = (
                                    phase_cost + phase_stratum_counts[key][stratum],
                                    cumulative_cost
                                    + cumulative_stratum_counts[key][stratum],
                                    tie_cost
                                    + (
                                        stratum
                                        - row
                                        - cycle_position
                                        - phase_index
                                    )
                                    % OPTIMIZER_BATCH_SIZE,
                                    assignment + (stratum,),
                                )
                                next_mask = mask | (1 << stratum)
                                if (
                                    next_mask not in next_states
                                    or candidate < next_states[next_mask]
                                ):
                                    next_states[next_mask] = candidate
                        states = next_states
                    timestep_strata = states[(1 << OPTIMIZER_BATCH_SIZE) - 1][3]
                    for view_index, stratum in zip(
                        groups[group_index], timestep_strata, strict=True
                    ):
                        cumulative_stratum_counts[
                            train[view_index].serialized_key
                        ][stratum] += 1
                        phase_stratum_counts[
                            train[view_index].serialized_key
                        ][stratum] += 1
                plans.append(
                    TrainingBatchPlan(
                        step=step,
                        cycle_index=cycle_index,
                        cycle_position=cycle_position,
                        kind="stratified",
                        view_indices=groups[group_index],
                        timestep_strata=timestep_strata,
                    )
                )
    result = tuple(plans)
    _validate_training_schedule(
        result,
        train,
        phase_start=phase_start,
        phase_count=phase_count,
        endpoint_cycle_positions=endpoint_cycle_positions,
    )
    return result


def _validate_training_schedule(
    schedule: tuple[TrainingBatchPlan, ...],
    train: Sequence[PreparedView],
    *,
    phase_start: int = 0,
    phase_count: int = 1,
    endpoint_cycle_positions: tuple[int, ...] = DEFAULT_ENDPOINT_CYCLE_POSITIONS,
) -> None:
    expected_updates = OPTIMIZER_STEPS * phase_count
    if len(schedule) != expected_updates:
        raise RuntimeError(
            f"Training schedule contains {len(schedule)} updates, "
            f"expected {expected_updates}"
        )
    exposure_counts = {
        index: {
            "total": 0,
            "shared_endpoint": 0,
            "stratified": 0,
            "strata": Counter(),
        }
        for index in range(len(train))
    }
    omitted_by_geometry: dict[ArcGeometry, Counter[int]] = {
        geometry: Counter() for geometry in EFFECTIVE_GEOMETRIES
    }
    endpoint_steps = []
    first_step = phase_start * OPTIMIZER_STEPS + 1
    for expected_step, batch in enumerate(schedule, start=first_step):
        if batch.step != expected_step:
            raise RuntimeError("Training schedule steps are not contiguous")
        if len(batch.view_indices) != OPTIMIZER_BATCH_SIZE:
            raise RuntimeError("Training batch size differs from eight")
        if len(set(batch.view_indices)) != OPTIMIZER_BATCH_SIZE:
            raise RuntimeError("Training batch repeats a view")
        geometry_counts = Counter(train[index].geometry for index in batch.view_indices)
        if geometry_counts != Counter(
            {geometry: 2 for geometry in EFFECTIVE_GEOMETRIES}
        ):
            raise RuntimeError("Training batch geometry counts differ")
        if batch.kind == "shared_endpoint":
            endpoint_steps.append(batch.step)
            if (
                batch.cycle_position not in endpoint_cycle_positions
                or batch.timestep_strata is not None
            ):
                raise RuntimeError("Endpoint schedule metadata differs")
        elif batch.kind == "stratified":
            if (
                batch.cycle_position in endpoint_cycle_positions
                or batch.timestep_strata is None
            ):
                raise RuntimeError("Regular schedule metadata differs")
            if set(batch.timestep_strata) != set(range(OPTIMIZER_BATCH_SIZE)):
                raise RuntimeError("Regular batch does not cover all eight strata")
        else:
            raise RuntimeError(f"Unknown training batch kind: {batch.kind}")
        for row, view_index in enumerate(batch.view_indices):
            counts = exposure_counts[view_index]
            counts["total"] += 1
            counts[batch.kind] += 1
            if batch.timestep_strata is not None:
                counts["strata"][batch.timestep_strata[row]] += 1

    final_step = (phase_start + phase_count) * OPTIMIZER_STEPS
    expected_endpoint_steps = [
        cycle_start + position
        for cycle_start in range(first_step, final_step + 1, UPDATE_CYCLE_LENGTH)
        for position in endpoint_cycle_positions
    ]
    if endpoint_steps != expected_endpoint_steps:
        raise RuntimeError("Endpoint steps differ from the configured cycle positions")
    endpoint_exposures_per_view = len(endpoint_cycle_positions) * phase_count
    regular_exposures_per_view = (
        UPDATE_CYCLE_LENGTH - len(endpoint_cycle_positions)
    ) * phase_count
    for view_index, counts in exposure_counts.items():
        if (
            counts["total"] != EXPOSURES_PER_VIEW * phase_count
            or counts["shared_endpoint"] != endpoint_exposures_per_view
            or counts["stratified"] != regular_exposures_per_view
        ):
            raise RuntimeError(f"View {view_index} exposure counts differ: {counts}")
        stratum_counts = [counts["strata"][stratum] for stratum in range(8)]
        if max(stratum_counts) - min(stratum_counts) > 1:
            raise RuntimeError(
                f"View {view_index} regular strata are imbalanced: {stratum_counts}"
            )
        for stratum, count in enumerate(stratum_counts):
            omitted_by_geometry[train[view_index].geometry][stratum] += (
                phase_count - count
            )
    for geometry, counts in omitted_by_geometry.items():
        omissions_per_stratum = (
            TRAIN_PER_GEOMETRY
            * (UPDATE_CYCLE_LENGTH - regular_exposures_per_view // phase_count)
            * phase_count
            // UPDATE_CYCLE_LENGTH
        )
        expected_omissions = Counter(
            {stratum: omissions_per_stratum for stratum in range(8)}
        )
        if counts != expected_omissions:
            raise RuntimeError(
                f"Omitted-stratum counts are imbalanced for {geometry.value}: {counts}"
            )


def _schedule_payload(
    schedule: Sequence[TrainingBatchPlan],
    train: Sequence[PreparedView],
) -> list[dict[str, Any]]:
    return [
        {
            "step": batch.step,
            "cycle_index": batch.cycle_index,
            "cycle_position": batch.cycle_position,
            "kind": batch.kind,
            "keys": [train[index].serialized_key for index in batch.view_indices],
            "timestep_strata": batch.timestep_strata,
        }
        for batch in schedule
    ]


def _schedule_exposure_audit(
    schedule: Sequence[TrainingBatchPlan],
    train: Sequence[PreparedView],
) -> dict[str, Any]:
    by_key = {
        view.serialized_key: {
            "total": 0,
            "shared_endpoint": 0,
            "stratified": 0,
            "regular_strata": [],
        }
        for view in train
    }
    for batch in schedule:
        for row, view_index in enumerate(batch.view_indices):
            item = by_key[train[view_index].serialized_key]
            item["total"] += 1
            item[batch.kind] += 1
            if batch.timestep_strata is not None:
                item["regular_strata"].append(batch.timestep_strata[row])
    endpoint_updates = sum(batch.kind == "shared_endpoint" for batch in schedule)
    regular_updates = sum(batch.kind == "stratified" for batch in schedule)
    planned_exposures = len(schedule) * OPTIMIZER_BATCH_SIZE
    endpoint_exposures = endpoint_updates * OPTIMIZER_BATCH_SIZE
    regular_exposures = regular_updates * OPTIMIZER_BATCH_SIZE
    for name, count in (
        ("total", planned_exposures),
        ("shared_endpoint", endpoint_exposures),
        ("stratified", regular_exposures),
    ):
        if count % len(by_key):
            raise RuntimeError(f"{name} exposures do not divide evenly across views")
    expected = {
        "total": planned_exposures // len(by_key),
        "shared_endpoint": endpoint_exposures // len(by_key),
        "stratified": regular_exposures // len(by_key),
    }
    all_exact = all(
        item["total"] == expected["total"]
        and item["shared_endpoint"] == expected["shared_endpoint"]
        and item["stratified"] == expected["stratified"]
        and max(
            Counter(item["regular_strata"])[stratum] for stratum in range(8)
        )
        - min(Counter(item["regular_strata"])[stratum] for stratum in range(8))
        <= 1
        for item in by_key.values()
    )
    return {
        "expected": expected,
        "expected_update_counts": {
            "shared_endpoint": endpoint_updates,
            "stratified": regular_updates,
            "total": len(schedule),
        },
        "view_count": len(by_key),
        "all_exact": all_exact,
        "items": by_key,
    }


def _combine_exposure_audits(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    first_items = first.get("items")
    second_items = second.get("items")
    if not isinstance(first_items, dict) or not isinstance(second_items, dict):
        raise RuntimeError("Cannot combine exposure audits without per-view items")
    if first_items.keys() != second_items.keys():
        raise RuntimeError("Parent and continuation exposure keys differ")
    expected = {
        name: int(first["expected"][name]) + int(second["expected"][name])
        for name in ("total", "shared_endpoint", "stratified")
    }
    items = {}
    for key in first_items:
        left = first_items[key]
        right = second_items[key]
        item = {
            "total": int(left["total"]) + int(right["total"]),
            "shared_endpoint": int(left["shared_endpoint"])
            + int(right["shared_endpoint"]),
            "stratified": int(left["stratified"]) + int(right["stratified"]),
            "regular_strata": list(left["regular_strata"])
            + list(right["regular_strata"]),
        }
        items[key] = item
    all_exact = all(
        item["total"] == expected["total"]
        and item["shared_endpoint"] == expected["shared_endpoint"]
        and item["stratified"] == expected["stratified"]
        and max(
            Counter(item["regular_strata"])[stratum] for stratum in range(8)
        )
        - min(Counter(item["regular_strata"])[stratum] for stratum in range(8))
        <= 1
        for item in items.values()
    )
    return {
        "expected": expected,
        "view_count": len(items),
        "all_exact": all_exact,
        "items": items,
    }


def _subtract_exposure_audits(
    cumulative: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    cumulative_items = cumulative.get("items")
    latest_items = latest.get("items")
    if not isinstance(cumulative_items, dict) or not isinstance(latest_items, dict):
        raise RuntimeError("Cannot subtract exposure audits without per-view items")
    if cumulative_items.keys() != latest_items.keys():
        raise RuntimeError("Cumulative and latest exposure keys differ")
    expected = {
        name: int(cumulative["expected"][name]) - int(latest["expected"][name])
        for name in ("total", "shared_endpoint", "stratified")
    }
    if any(value < 0 for value in expected.values()):
        raise RuntimeError("Latest exposure audit exceeds the cumulative audit")

    items = {}
    for key in cumulative_items:
        total_item = cumulative_items[key]
        latest_item = latest_items[key]
        regular_strata = Counter(total_item["regular_strata"])
        regular_strata.subtract(latest_item["regular_strata"])
        if any(count < 0 for count in regular_strata.values()):
            raise RuntimeError(
                f"Latest regular strata exceed cumulative strata for {key}"
            )
        item = {
            "total": int(total_item["total"]) - int(latest_item["total"]),
            "shared_endpoint": int(total_item["shared_endpoint"])
            - int(latest_item["shared_endpoint"]),
            "stratified": int(total_item["stratified"])
            - int(latest_item["stratified"]),
            "regular_strata": [
                stratum
                for stratum in range(OPTIMIZER_BATCH_SIZE)
                for _ in range(regular_strata[stratum])
            ],
        }
        if any(
            item[name] < 0
            for name in ("total", "shared_endpoint", "stratified")
        ):
            raise RuntimeError(
                f"Latest exposure counts exceed cumulative counts for {key}"
            )
        items[key] = item

    all_exact = all(
        item["total"] == expected["total"]
        and item["shared_endpoint"] == expected["shared_endpoint"]
        and item["stratified"] == expected["stratified"]
        and len(item["regular_strata"]) == expected["stratified"]
        and max(
            Counter(item["regular_strata"])[stratum]
            for stratum in range(OPTIMIZER_BATCH_SIZE)
        )
        - min(
            Counter(item["regular_strata"])[stratum]
            for stratum in range(OPTIMIZER_BATCH_SIZE)
        )
        <= 1
        for item in items.values()
    )
    return {
        "expected": expected,
        "view_count": len(items),
        "all_exact": all_exact,
        "items": items,
    }


def _validate_static_protocol() -> None:
    if TRAIN_VIEW_COUNT != 512 or HOLDOUT_VIEW_COUNT != 128:
        raise RuntimeError("Pool sizes differ from protocol")
    if EFFECTIVE_GEOMETRIES != (
        ArcGeometry.IDENTITY,
        ArcGeometry.ROTATE_90,
        ArcGeometry.ROTATE_180,
        ArcGeometry.ROTATE_270,
    ):
        raise RuntimeError("Effective geometry representatives differ")
    if set(
        geometry for pair in REDUNDANT_GEOMETRY_CLASSES.values() for geometry in pair
    ) != set(ArcGeometry):
        raise RuntimeError("Effective geometry classes do not cover D4")
    if OPTIMIZER_STEPS != 512 or OPTIMIZER_BATCH_SIZE != 8:
        raise RuntimeError("Training allocation differs from protocol")
    if ENDPOINT_UPDATE_COUNT != 64 or REGULAR_UPDATE_COUNT != 448:
        raise RuntimeError("Endpoint allocation differs from protocol")
    if EVALUATION_STEPS != (0, 128, 256, 512):
        raise RuntimeError("Evaluation steps differ from protocol")
    if (
        CONTINUATION_START_STEP != 512
        or CONTINUATION_FINAL_STEP != 2048
        or CONTINUATION_OPTIMIZER_STEPS != 1536
        or CONTINUATION_PHASE_START != 1
        or CONTINUATION_PHASE_COUNT != 3
        or CONTINUATION_EVALUATION_STEPS != (1024, 2048)
        or CONTINUATION_FULL_GENERATION_STEPS != (2048,)
    ):
        raise RuntimeError("Continuation allocation differs from protocol")
    if (
        REGION_WEIGHTED_EVALUATION_STEPS != (0, 128, 256, 512)
        or REGION_WEIGHTED_FULL_GENERATION_STEPS != (512,)
        or REGION_LOSS_WEIGHTS
        != {"grid": 0.6, "boundary": 0.3, "outer_padding": 0.1}
        or abs(sum(REGION_LOSS_WEIGHTS.values()) - 1.0) > 1e-12
    ):
        raise RuntimeError("Region-weighted loss allocation differs from protocol")
    if (
        REGION_WEIGHTED_CONTINUATION_START_STEP != 512
        or REGION_WEIGHTED_CONTINUATION_FINAL_STEP != 1536
        or REGION_WEIGHTED_CONTINUATION_OPTIMIZER_STEPS != 1024
        or REGION_WEIGHTED_CONTINUATION_PHASE_START != 1
        or REGION_WEIGHTED_CONTINUATION_PHASE_COUNT != 2
        or REGION_WEIGHTED_CONTINUATION_EVALUATION_STEPS != (1024, 1536)
        or REGION_WEIGHTED_CONTINUATION_FULL_GENERATION_STEPS != (1536,)
    ):
        raise RuntimeError(
            "Region-weighted continuation allocation differs from protocol"
        )
    if (
        REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP != 1536
        or REGION_WEIGHTED_ENDPOINT_BALANCED_FINAL_STEP != 2048
        or REGION_WEIGHTED_ENDPOINT_BALANCED_OPTIMIZER_STEPS != 512
        or REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_START != 3
        or REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_COUNT != 1
        or ENDPOINT_BALANCED_CYCLE_POSITIONS != (0, 2, 4, 6)
        or REGION_WEIGHTED_ENDPOINT_BALANCED_EVALUATION_STEPS
        != (1664, 1792, 2048)
        or REGION_WEIGHTED_ENDPOINT_BALANCED_FULL_GENERATION_STEPS != (2048,)
    ):
        raise RuntimeError(
            "Endpoint-balanced continuation allocation differs from protocol"
        )
    if (
        REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP != 2048
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FINAL_STEP != 4096
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_OPTIMIZER_STEPS != 2048
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_START != 4
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_COUNT != 4
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_EVALUATION_STEPS != (4096,)
        or REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FULL_GENERATION_STEPS
        != (4096,)
    ):
        raise RuntimeError(
            "Endpoint-balanced long-continuation allocation differs from protocol"
        )
    if (
        FINAL_REPORT_UPLOAD_ATTEMPTS != 3
        or len(FINAL_REPORT_RETRY_DELAYS_SECONDS)
        != FINAL_REPORT_UPLOAD_ATTEMPTS - 1
    ):
        raise RuntimeError("Final report retry policy differs from protocol")


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


def _encode_target_grids(grids: Sequence[Any], vae_encoder: Any) -> Tensor:
    chunks = []
    for start in range(0, len(grids), VAE_ENCODE_BATCH_SIZE):
        images = torch.stack(
            tuple(
                render_grid_canvas(grid, canvas_size=SPATIAL_SHAPE.height)
                for grid in grids[start : start + VAE_ENCODE_BATCH_SIZE]
            )
        )
        chunks.append(vae_encoder(images, spatial_shape=SPATIAL_SHAPE))
    if not chunks:
        raise ValueError("grids must not be empty")
    return torch.cat(chunks, dim=0)


def _decode_latents(
    latents: Tensor,
    vae_decoder: Any,
) -> tuple[Tensor, tuple[Any, ...], Tensor]:
    decoded = vae_decoder(latents, spatial_shape=SPATIAL_SHAPE)
    rgb_images = _normalized_images_to_uint8(decoded)
    classes = classify_arc_cells(
        rgb_images,
        canvas_size=SPATIAL_SHAPE.height,
        include_pad=True,
    )
    predictions = decode_arc_grids(classes, canvas_size=SPATIAL_SHAPE.height)
    return classes, predictions, rgb_images


def _grid_mismatches(
    predicted_grid: Any,
    expected_grid: Any,
) -> list[dict[str, int]]:
    if predicted_grid is None or len(predicted_grid) != len(expected_grid):
        return []
    if any(
        len(predicted_row) != len(expected_row)
        for predicted_row, expected_row in zip(
            predicted_grid,
            expected_grid,
            strict=True,
        )
    ):
        return []
    return [
        {
            "row": row_index,
            "column": column_index,
            "expected": int(expected),
            "predicted": int(predicted),
        }
        for row_index, (predicted_row, expected_row) in enumerate(
            zip(predicted_grid, expected_grid, strict=True)
        )
        for column_index, (predicted, expected) in enumerate(
            zip(predicted_row, expected_row, strict=True)
        )
        if predicted != expected
    ]


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


def _screen_candidate_vae_roundtrips(
    *,
    original: PreparedView,
    candidates: Sequence[PreparedView],
    vae_encoder: Any,
    vae_decoder: Any,
) -> tuple[Tensor, set[str], dict[str, Any]]:
    all_views = tuple(candidates) + (original,)
    grids = tuple(view.target_grid for view in all_views)
    started = time.perf_counter()
    latents = _encode_target_grids(grids, vae_encoder)
    predictions = []
    for start in range(0, len(all_views), VAE_DECODE_BATCH_SIZE):
        _, decoded, _ = _decode_latents(
            latents[start : start + VAE_DECODE_BATCH_SIZE],
            vae_decoder,
        )
        predictions.extend(decoded)
    diagnostics = diagnose_exact_grids(tuple(predictions), grids)
    safe_keys = {
        view.serialized_key
        for view, diagnostic in zip(all_views, diagnostics, strict=True)
        if diagnostic.exact_grid
    }
    failed = []
    exact_by_geometry = Counter()
    total_by_geometry = Counter()
    for view, prediction, diagnostic in zip(
        all_views,
        predictions,
        diagnostics,
        strict=True,
    ):
        group = "original" if view is original else view.geometry.value
        total_by_geometry[group] += 1
        exact_by_geometry[group] += int(diagnostic.exact_grid)
        if view is not original and not diagnostic.exact_grid:
            failed.append(
                {
                    "key": view.serialized_key,
                    "group": group,
                    "diagnostic": _diagnostic_payload(diagnostic),
                    "predicted_grid": prediction.grid,
                    "expected_grid": view.target_grid,
                    "mismatched_cells": _grid_mismatches(
                        prediction.grid,
                        view.target_grid,
                    ),
                }
            )
    original_index = len(candidates)
    original_diagnostic = diagnostics[original_index]
    original_prediction = predictions[original_index]
    report = {
        "candidate_count": len(candidates),
        "candidate_exact_count": sum(int(item.exact_grid) for item in diagnostics[:-1]),
        "candidate_failed_count": sum(
            int(not item.exact_grid) for item in diagnostics[:-1]
        ),
        "by_geometry": {
            geometry.value: {
                "exact": exact_by_geometry[geometry.value],
                "total": total_by_geometry[geometry.value],
            }
            for geometry in EFFECTIVE_GEOMETRIES
        },
        "original": {
            "key": original.serialized_key,
            "exact_grid": original_diagnostic.exact_grid,
            "diagnostic": _diagnostic_payload(original_diagnostic),
            "predicted_grid": original_prediction.grid,
            "expected_grid": original.target_grid,
            "mismatched_cells": _grid_mismatches(
                original_prediction.grid,
                original.target_grid,
            ),
            "score_role": "diagnostic-original-view",
        },
        "failed_candidates": failed,
        "seconds": time.perf_counter() - started,
        "latent_shape": list(latents.shape),
    }
    return latents, safe_keys, report


def _view_payload(view: PreparedView, *, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "key": view.serialized_key,
        "geometry": view.geometry.value,
        "redundant_geometry_class": [
            geometry.value for geometry in REDUNDANT_GEOMETRY_CLASSES[view.geometry]
        ],
        "color_mapping": list(view.color_mapping),
        "pair_indices": list(view.key.augmentation.pair_indices),
        "prefix_sha256": view.prefix_hash,
        "query_grid_sha256": view.query_hash,
        "target_grid_sha256": view.target_hash,
        "full_view_sha256": view.full_view_hash,
        "query_grid": view.query_grid,
        "target_grid": view.target_grid,
    }


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


def _sample_planned_flow_batch(
    clean_latents: Tensor,
    *,
    batch_plan: TrainingBatchPlan,
    generator: torch.Generator,
) -> tuple[FlowMatchingBatch, dict[str, Any]]:
    if batch_plan.kind == "shared_endpoint":
        shared_noise = torch.randn(
            (1, *clean_latents.shape[1:]),
            device=clean_latents.device,
            dtype=clean_latents.dtype,
            generator=generator,
        )
        noise = shared_noise.expand_as(clean_latents).clone()
        timesteps = torch.ones(
            (clean_latents.shape[0],),
            device=clean_latents.device,
            dtype=torch.float32,
        )
        return (
            FlowMatchingBatch(
                noisy_latents=noise,
                timesteps=timesteps,
                target_velocity=noise - clean_latents,
            ),
            {
                "kind": "shared_endpoint",
                "cycle_index": batch_plan.cycle_index,
                "cycle_position": batch_plan.cycle_position,
                "flow_seed": FLOW_SEED + batch_plan.step,
                "shared_noise_across_rows": True,
                "prescribed_timestep_strata": None,
            },
        )
    if batch_plan.kind != "stratified" or batch_plan.timestep_strata is None:
        raise RuntimeError("Regular batch plan is malformed")
    strata = torch.tensor(
        batch_plan.timestep_strata,
        device=clean_latents.device,
        dtype=torch.float32,
    )
    jitter = torch.rand(
        (clean_latents.shape[0],),
        device=clean_latents.device,
        dtype=torch.float32,
        generator=generator,
    )
    lower_bounds = strata / OPTIMIZER_BATCH_SIZE
    upper_bounds = (strata + 1) / OPTIMIZER_BATCH_SIZE
    timesteps = torch.minimum(
        (strata + jitter) / OPTIMIZER_BATCH_SIZE,
        torch.nextafter(upper_bounds, lower_bounds),
    )
    noise = torch.randn(
        clean_latents.shape,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
        generator=generator,
    )
    broadcast = timesteps.to(dtype=clean_latents.dtype).reshape(-1, 1, 1)
    return (
        FlowMatchingBatch(
            noisy_latents=(1 - broadcast) * clean_latents + broadcast * noise,
            timesteps=timesteps,
            target_velocity=noise - clean_latents,
        ),
        {
            "kind": "stratified",
            "cycle_index": batch_plan.cycle_index,
            "cycle_position": batch_plan.cycle_position,
            "flow_seed": FLOW_SEED + batch_plan.step,
            "shared_noise_across_rows": False,
            "prescribed_timestep_strata": list(batch_plan.timestep_strata),
        },
    )


def _optimizer_step(
    *,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    text_inputs: FluxTextInputs,
    clean_latents: Tensor,
    views: Sequence[PreparedView],
    batch_plan: TrainingBatchPlan,
    compute_dtype: torch.dtype,
    loss_policy: str = UNWEIGHTED_LOSS_POLICY,
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
    generator = torch.Generator(device=DEVICE).manual_seed(FLOW_SEED + batch_plan.step)
    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    flow_batch, sampling = _sample_planned_flow_batch(
        microbatch.clean_latents,
        batch_plan=batch_plan,
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
    loss, squared_error, region_losses, weighted_contributions = (
        _training_objective(
            predicted_velocity,
            flow_batch.target_velocity,
            views,
            loss_policy=loss_policy,
        )
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

    row_objectives = (
        weighted_contributions.sum(dim=1)
        if loss_policy == REGION_WEIGHTED_LOSS_POLICY
        else region_losses[:, 0]
    )
    per_row = torch.stack(
        (
            flow_batch.timesteps,
            row_objectives.detach(),
            *tuple(region_losses.detach().unbind(dim=1)),
            *tuple(weighted_contributions.detach().unbind(dim=1)),
        ),
        dim=1,
    ).to(device="cpu", dtype=torch.float32)
    per_exposure = []
    for row, view in enumerate(views):
        (
            timestep,
            objective_loss,
            total_loss,
            grid_loss,
            boundary_loss,
            outer_padding_loss,
            padding_loss,
            weighted_grid_contribution,
            weighted_boundary_contribution,
            weighted_outer_padding_contribution,
        ) = per_row[row].tolist()
        stratum = (
            None
            if batch_plan.timestep_strata is None
            else batch_plan.timestep_strata[row]
        )
        exposure = {
            "key": view.serialized_key,
            "geometry": view.geometry.value,
            "sampling_kind": batch_plan.kind,
            "timestep": float(timestep),
            "timestep_stratum": stratum,
            "loss_policy": loss_policy,
            "objective_loss": objective_loss,
            "total_loss": total_loss,
            "grid_loss": grid_loss,
            "boundary_loss": boundary_loss,
            "outer_padding_loss": outer_padding_loss,
            "padding_loss": padding_loss,
        }
        if loss_policy == REGION_WEIGHTED_LOSS_POLICY:
            exposure.update(
                {
                    "weighted_grid_contribution": weighted_grid_contribution,
                    "weighted_boundary_contribution": (
                        weighted_boundary_contribution
                    ),
                    "weighted_outer_padding_contribution": (
                        weighted_outer_padding_contribution
                    ),
                }
            )
        per_exposure.append(exposure)
    return {
        "step": batch_plan.step,
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "gradient_clipped": bool(gradient_norm > MAX_GRAD_NORM),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "loss_policy": loss_policy,
        "sampling": sampling,
        "per_exposure": per_exposure,
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(DEVICE) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(DEVICE) / 2**30,
    }


def _region_masks(
    target_grid: Any,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    height = len(target_grid)
    width = len(target_grid[0])
    if not 0 < height < SPATIAL_SHAPE.height:
        raise ValueError("Region-weighted pilot requires height in [1, 29]")
    if not 0 < width < SPATIAL_SHAPE.width:
        raise ValueError("Region-weighted pilot requires width in [1, 29]")
    boundary_count = height + width + 1
    outer_padding_count = (
        SPATIAL_SHAPE.token_count - height * width - boundary_count
    )
    if outer_padding_count <= 0:
        raise ValueError("Region-weighted pilot requires outer padding")
    cache_key = (height, width, device.type, device.index)
    cached = REGION_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    grid = torch.zeros(
        (SPATIAL_SHAPE.height, SPATIAL_SHAPE.width),
        dtype=torch.bool,
        device=device,
    )
    grid[:height, :width] = True
    boundary = torch.zeros_like(grid)
    boundary[height, : width + 1] = True
    boundary[:height, width] = True
    padding = ~grid
    outer_padding = padding & ~boundary
    result = (grid, boundary, outer_padding, padding)
    REGION_MASK_CACHE[cache_key] = result
    return result


def _per_row_region_losses(
    squared_error: Tensor,
    views: Sequence[PreparedView],
) -> Tensor:
    if squared_error.shape[0] != len(views):
        raise ValueError("squared_error and views batch sizes differ")
    spatial = squared_error.reshape(
        squared_error.shape[0],
        SPATIAL_SHAPE.height,
        SPATIAL_SHAPE.width,
        squared_error.shape[2],
    )
    rows = []
    for row, view in enumerate(views):
        grid, boundary, outer_padding, padding = _region_masks(
            view.target_grid,
            device=squared_error.device,
        )
        rows.append(
            torch.stack(
                (
                    spatial[row].mean(),
                    spatial[row, grid].mean(),
                    spatial[row, boundary].mean(),
                    spatial[row, outer_padding].mean(),
                    spatial[row, padding].mean(),
                )
            )
        )
    return torch.stack(rows)


def _training_objective(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    views: Sequence[PreparedView],
    *,
    loss_policy: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    squared_error = (
        predicted_velocity.float() - target_velocity.float()
    ).square()
    region_losses = _per_row_region_losses(squared_error, views)
    if loss_policy == UNWEIGHTED_LOSS_POLICY:
        objective = flow_matching_loss(predicted_velocity, target_velocity)
    elif loss_policy == REGION_WEIGHTED_LOSS_POLICY:
        objective = (
            REGION_LOSS_WEIGHTS["grid"] * region_losses[:, 1]
            + REGION_LOSS_WEIGHTS["boundary"] * region_losses[:, 2]
            + REGION_LOSS_WEIGHTS["outer_padding"] * region_losses[:, 3]
        ).mean()
    else:
        raise ValueError(f"Unsupported loss policy: {loss_policy}")
    weighted_contributions = torch.stack(
        (
            REGION_LOSS_WEIGHTS["grid"] * region_losses[:, 1],
            REGION_LOSS_WEIGHTS["boundary"] * region_losses[:, 2],
            REGION_LOSS_WEIGHTS["outer_padding"] * region_losses[:, 3],
        ),
        dim=1,
    )
    return objective, squared_error, region_losses, weighted_contributions


def _keyed_noise(
    keys: Sequence[str],
    *,
    seed: int,
    dtype: torch.dtype,
) -> Tensor:
    rows = []
    for key in keys:
        generator = torch.Generator(device=DEVICE).manual_seed(
            _stable_seed("arc-generation-noise", seed, key)
        )
        rows.append(
            torch.randn(
                (1, SPATIAL_SHAPE.token_count, TARGET_LATENT_CHANNELS),
                device=DEVICE,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.cat(rows, dim=0)


def _group_for_global_index(index: int) -> str:
    if 0 <= index < TRAIN_VIEW_COUNT:
        return "seen"
    if TRAIN_VIEW_COUNT <= index < TRAIN_VIEW_COUNT + HOLDOUT_VIEW_COUNT:
        return "holdout"
    if index == TRAIN_VIEW_COUNT + HOLDOUT_VIEW_COUNT:
        return "original"
    raise IndexError(f"Global view index is out of range: {index}")


def _evaluation_indices(
    step: int,
    train: Sequence[PreparedView],
    holdout: Sequence[PreparedView],
    *,
    evaluation_steps: Sequence[int] = EVALUATION_STEPS,
    final_step: int = OPTIMIZER_STEPS,
    full_final_panel: bool = True,
) -> tuple[int, ...]:
    if step not in evaluation_steps:
        raise ValueError(f"Unsupported evaluation step: {step}")
    if step == final_step and full_final_panel:
        seen_local = _balanced_subset_indices(
            train,
            per_geometry=FINAL_SEEN_PER_GEOMETRY,
            seed_label="seen-evaluation",
        )
        holdout_local = tuple(range(HOLDOUT_VIEW_COUNT))
    else:
        seen_local = _balanced_subset_indices(
            train,
            per_geometry=INTERMEDIATE_SEEN_PER_GEOMETRY,
            seed_label="seen-evaluation",
        )
        holdout_local = _balanced_subset_indices(
            holdout,
            per_geometry=INTERMEDIATE_HOLDOUT_PER_GEOMETRY,
            seed_label="intermediate-holdout-evaluation",
        )
    return (
        *seen_local,
        *(TRAIN_VIEW_COUNT + index for index in holdout_local),
        TRAIN_VIEW_COUNT + HOLDOUT_VIEW_COUNT,
    )


def _summarize_exact_items(
    items: Sequence[dict[str, Any]],
    *,
    pass2: bool,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot summarize an empty diagnostic panel")
    exact = sum(
        bool(item["exact_grid_pass2"] if pass2 else item["diagnostic"]["exact_grid"])
        for item in items
    )
    valid = sum(bool(item["diagnostic"]["prediction_valid"]) for item in items)
    shape = sum(bool(item["diagnostic"]["shape_match"]) for item in items)
    cell_values = [
        item["diagnostic"]["cell_accuracy"]
        for item in items
        if item["diagnostic"]["cell_accuracy"] is not None
    ]
    return {
        "count": len(items),
        "exact_grid_count": exact,
        "exact_grid_fraction": exact / len(items),
        "valid_grid_count": valid,
        "correct_shape_count": shape,
        "mean_cell_accuracy_when_shape_matches": (
            None if not cell_values else sum(cell_values) / len(cell_values)
        ),
    }


def _summarize_t1_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_group = {}
    for group in ("seen", "holdout", "original"):
        selected = [item for item in items if item["group"] == group]
        summary = _summarize_exact_items(selected, pass2=False)
        summary.update(
            {
                "mean_total_loss": sum(item["loss"]["total_loss"] for item in selected)
                / len(selected),
                "mean_grid_loss": sum(item["loss"]["grid_loss"] for item in selected)
                / len(selected),
                "mean_boundary_loss": sum(
                    item["loss"]["boundary_loss"] for item in selected
                )
                / len(selected),
                "mean_outer_padding_loss": sum(
                    item["loss"]["outer_padding_loss"] for item in selected
                )
                / len(selected),
                "mean_padding_loss": sum(
                    item["loss"]["padding_loss"] for item in selected
                )
                / len(selected),
                "mean_region_weighted_loss": sum(
                    item["loss"]["region_weighted_loss"] for item in selected
                )
                / len(selected),
            }
        )
        by_group[group] = summary
    return by_group


@torch.no_grad()
def _t1_diagnostic(
    *,
    transformer: torch.nn.Module,
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    all_views: Sequence[PreparedView],
    compute_dtype: torch.dtype,
    vae_decoder: Any,
) -> dict[str, Any]:
    previous_mode = transformer.training
    transformer.eval()
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for start in range(0, len(all_views), GENERATION_BATCH_SIZE):
            stop = min(start + GENERATION_BATCH_SIZE, len(all_views))
            views = all_views[start:stop]
            indices = torch.arange(start, stop, device=DEVICE, dtype=torch.int64)
            latents = all_latents.index_select(0, indices)
            text_inputs = _slice_text_inputs(all_text_inputs, indices)
            keys = tuple(view.serialized_key for view in views)
            noise = _keyed_noise(keys, seed=PROBE_SEED, dtype=latents.dtype)
            timesteps = torch.ones((len(views),), device=DEVICE, dtype=torch.float32)
            predicted = predict_flux_velocity(
                transformer,
                noise,
                text_inputs,
                build_flux_image_position_ids(
                    len(views), DEVICE, spatial_shape=SPATIAL_SHAPE
                ),
                timesteps,
                compute_dtype=compute_dtype,
            )
            squared = (predicted.float() - (noise - latents).float()).square()
            row_losses = _per_row_region_losses(squared, views).to(
                device="cpu",
                dtype=torch.float32,
            )
            reconstructed = noise - predicted.to(dtype=noise.dtype)
            _, predictions, _ = _decode_latents(reconstructed, vae_decoder)
            expected = tuple(view.target_grid for view in views)
            diagnostics = diagnose_exact_grids(predictions, expected)
            for row, (view, diagnostic) in enumerate(
                zip(views, diagnostics, strict=True)
            ):
                (
                    total_loss,
                    grid_loss,
                    boundary_loss,
                    outer_padding_loss,
                    padding_loss,
                ) = row_losses[row].tolist()
                global_index = start + row
                items.append(
                    {
                        "key": view.serialized_key,
                        "group": _group_for_global_index(global_index),
                        "geometry": view.geometry.value,
                        "loss": {
                            "total_loss": total_loss,
                            "grid_loss": grid_loss,
                            "boundary_loss": boundary_loss,
                            "outer_padding_loss": outer_padding_loss,
                            "padding_loss": padding_loss,
                            "region_weighted_loss": (
                                REGION_LOSS_WEIGHTS["grid"] * grid_loss
                                + REGION_LOSS_WEIGHTS["boundary"] * boundary_loss
                                + REGION_LOSS_WEIGHTS["outer_padding"]
                                * outer_padding_loss
                            ),
                        },
                        "diagnostic": _diagnostic_payload(diagnostic),
                    }
                )
    finally:
        transformer.train(previous_mode)
    return {
        "timestep": 1.0,
        "target_signal_in_model_input": False,
        "noise_policy": "deterministic-per-key",
        "view_count": len(items),
        "summaries": _summarize_t1_items(items),
        "items": items,
        "seconds": time.perf_counter() - started,
    }


def _prediction_label(view: PreparedView, group: str) -> str:
    if group == "original":
        return "Original"
    prefix = "Gesehen" if group == "seen" else "Ungesehen"
    return f"{prefix} · {view.geometry.value} · {view.candidate_index}"


def _uint8_grid_image(grid: Any) -> Tensor:
    canvas_size = max(len(grid), max(len(row) for row in grid))
    normalized = render_grid_canvas(grid, canvas_size=canvas_size)
    return _normalized_images_to_uint8(normalized.unsqueeze(0))[0]


def _save_comparison_pages(
    *,
    records: Sequence[dict[str, Any]],
    prediction_images: dict[str, Tensor],
    all_views_by_key: dict[str, PreparedView],
    image_directory: Path,
    step: int,
    seed: int,
    suffix: str,
) -> list[str]:
    paths = []
    tile_size = 160
    label_height = 28
    row_height = tile_size + label_height
    for page_index, start in enumerate(range(0, len(records), SHEET_PAGE_SIZE)):
        page_records = records[start : start + SHEET_PAGE_SIZE]
        sheet = Image.new(
            "RGB",
            (2 * tile_size, len(page_records) * row_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(page_records):
            key = record["key"]
            view = all_views_by_key[key]
            expected = _uint8_grid_image(view.target_grid)
            predicted_grid = record["prediction_grid"]
            predicted = (
                prediction_images[key]
                if predicted_grid is None
                else _uint8_grid_image(predicted_grid)
            )
            for column, tensor in enumerate((expected, predicted)):
                image = Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())
                scale = min(tile_size / image.width, tile_size / image.height)
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    ),
                    Image.Resampling.NEAREST,
                )
                x = column * tile_size + (tile_size - image.width) // 2
                y = row * row_height + (tile_size - image.height) // 2
                sheet.paste(image, (x, y))
            status = "EXAKT" if record["diagnostic"]["exact_grid"] else "FEHLER"
            label = f"{record['label']} · {status} · GT | Prediction"
            draw.text(
                (4, row * row_height + tile_size + 4),
                label[:74],
                fill="black",
            )
        path = image_directory / (
            f"step-{step:04d}-seed-{seed}-{suffix}-page-{page_index + 1:02d}.png"
        )
        sheet.save(path)
        paths.append(str(path.relative_to(REPORT_DIR)))
    return paths


@torch.no_grad()
def _generate_seed_records(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    all_text_inputs: FluxTextInputs,
    all_views: Sequence[PreparedView],
    global_indices: Sequence[int],
    seed: int,
    compute_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
    schedule = linear_timestep_schedule(INFERENCE_STEPS, device=DEVICE)
    records: list[dict[str, Any]] = []
    images: dict[str, Tensor] = {}
    print(f"[{STAGE_NAME}] generation seed {seed}: " f"0/{len(global_indices)} views")
    for start in range(0, len(global_indices), GENERATION_BATCH_SIZE):
        chunk = tuple(global_indices[start : start + GENERATION_BATCH_SIZE])
        indices = torch.tensor(chunk, device=DEVICE, dtype=torch.int64)
        views = tuple(all_views[index] for index in chunk)
        keys = tuple(view.serialized_key for view in views)
        result = continue_arc_batch(
            transformer,
            vae_decoder,
            _slice_text_inputs(all_text_inputs, indices),
            initial_latents=_keyed_noise(keys, seed=seed, dtype=torch.float32),
            timestep_schedule=schedule,
            compute_dtype=compute_dtype,
            spatial_shape=SPATIAL_SHAPE,
            infer_padding_boundary=True,
        )
        expected = tuple(view.target_grid for view in views)
        diagnostics = diagnose_exact_grids(result.predictions, expected)
        for row, (global_index, view, prediction, diagnostic, image) in enumerate(
            zip(
                chunk,
                views,
                result.predictions,
                diagnostics,
                result.rgb_images,
                strict=True,
            )
        ):
            del row
            group = _group_for_global_index(global_index)
            record = {
                "global_index": global_index,
                "key": view.serialized_key,
                "group": group,
                "geometry": view.geometry.value,
                "label": _prediction_label(view, group),
                "prediction_grid": prediction.grid,
                "diagnostic": _diagnostic_payload(diagnostic),
            }
            records.append(record)
            images[view.serialized_key] = image
        completed = start + len(chunk)
        if completed == len(global_indices) or completed % 64 == 0:
            print(
                f"[{STAGE_NAME}] generation seed {seed}: "
                f"{completed}/{len(global_indices)} views"
            )
    return records, images


def _generation_panel_metrics(
    seed1_records: Sequence[dict[str, Any]],
    seed2_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    seed2_by_key = {record["key"]: record for record in seed2_records}
    combined = []
    for record in seed1_records:
        seed2 = seed2_by_key.get(record["key"])
        pass1 = bool(record["diagnostic"]["exact_grid"])
        pass2 = pass1 or bool(seed2 is not None and seed2["diagnostic"]["exact_grid"])
        combined.append(
            {
                **record,
                "exact_grid_pass1": pass1,
                "exact_grid_pass2": pass2,
                "seed2_evaluated": seed2 is not None,
                "seed2_diagnostic": (None if seed2 is None else seed2["diagnostic"]),
                "seed2_prediction_grid": (
                    None if seed2 is None else seed2["prediction_grid"]
                ),
            }
        )
    panels = {}
    for group in ("seen", "holdout", "original"):
        selected = [item for item in combined if item["group"] == group]
        pass1_summary = _summarize_exact_items(selected, pass2=False)
        pass2_summary = _summarize_exact_items(selected, pass2=True)
        panels[group] = {
            "count": len(selected),
            "exact_grid_pass1": pass1_summary["exact_grid_count"],
            "exact_grid_pass2": pass2_summary["exact_grid_count"],
            "exact_grid_pass1_fraction": pass1_summary["exact_grid_fraction"],
            "exact_grid_pass2_fraction": pass2_summary["exact_grid_fraction"],
            "valid_grid_pass1": pass1_summary["valid_grid_count"],
            "correct_shape_pass1": pass1_summary["correct_shape_count"],
            "mean_cell_accuracy_pass1_when_shape_matches": pass1_summary[
                "mean_cell_accuracy_when_shape_matches"
            ],
            "seed2_evaluated_count": sum(item["seed2_evaluated"] for item in selected),
        }
        if group != "original":
            panels[group]["by_geometry"] = {
                geometry.value: {
                    "count": len(
                        geometry_items := [
                            item
                            for item in selected
                            if item["geometry"] == geometry.value
                        ]
                    ),
                    "exact_grid_pass1": sum(
                        item["exact_grid_pass1"] for item in geometry_items
                    ),
                    "exact_grid_pass2": sum(
                        item["exact_grid_pass2"] for item in geometry_items
                    ),
                }
                for geometry in EFFECTIVE_GEOMETRIES
            }
    return {"panels": panels, "records": combined}


@torch.no_grad()
def _evaluate(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    step: int,
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    train: Sequence[PreparedView],
    holdout: Sequence[PreparedView],
    original: PreparedView,
    compute_dtype: torch.dtype,
    evaluation_steps: Sequence[int] = EVALUATION_STEPS,
    final_step: int = OPTIMIZER_STEPS,
    full_final_panel: bool = True,
    include_t1_diagnostic: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_views = tuple(train) + tuple(holdout) + (original,)
    global_indices = _evaluation_indices(
        step,
        train,
        holdout,
        evaluation_steps=evaluation_steps,
        final_step=final_step,
        full_final_panel=full_final_panel,
    )
    if include_t1_diagnostic:
        print(
            f"[{STAGE_NAME}] evaluation step {step}: starting 641-view t=1 "
            f"diagnostic and {len(global_indices)}-view generation panel"
        )
        t1 = _t1_diagnostic(
            transformer=transformer,
            all_text_inputs=all_text_inputs,
            all_latents=all_latents,
            all_views=all_views,
            compute_dtype=compute_dtype,
            vae_decoder=vae_decoder,
        )
    else:
        print(
            f"[{STAGE_NAME}] evaluation step {step}: starting "
            f"{len(global_indices)}-view generation panel without t=1 diagnostic"
        )
        t1 = None
    seed1_records, seed1_images = _generate_seed_records(
        transformer=transformer,
        vae_decoder=vae_decoder,
        all_text_inputs=all_text_inputs,
        all_views=all_views,
        global_indices=global_indices,
        seed=GENERATION_SEEDS[0],
        compute_dtype=compute_dtype,
    )
    failure_indices = tuple(
        record["global_index"]
        for record in seed1_records
        if not record["diagnostic"]["exact_grid"]
    )
    if failure_indices:
        seed2_records, seed2_images = _generate_seed_records(
            transformer=transformer,
            vae_decoder=vae_decoder,
            all_text_inputs=all_text_inputs,
            all_views=all_views,
            global_indices=failure_indices,
            seed=GENERATION_SEEDS[1],
            compute_dtype=compute_dtype,
        )
    else:
        seed2_records, seed2_images = [], {}
    metrics = _generation_panel_metrics(seed1_records, seed2_records)
    image_directory = REPORT_DIR / STAGE_NAME / "generated_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    views_by_key = {view.serialized_key: view for view in all_views}
    seed1_pages = _save_comparison_pages(
        records=seed1_records,
        prediction_images=seed1_images,
        all_views_by_key=views_by_key,
        image_directory=image_directory,
        step=step,
        seed=GENERATION_SEEDS[0],
        suffix="pass1",
    )
    seed2_pages = (
        _save_comparison_pages(
            records=seed2_records,
            prediction_images=seed2_images,
            all_views_by_key=views_by_key,
            image_directory=image_directory,
            step=step,
            seed=GENERATION_SEEDS[1],
            suffix="pass2-failures",
        )
        if seed2_records
        else []
    )
    entry = {
        "step": step,
        "kind": "full-generation",
        "panel_global_indices": list(global_indices),
        "panel_keys": [all_views[index].serialized_key for index in global_indices],
        "panels": metrics["panels"],
        "generation": {
            "inference_steps": INFERENCE_STEPS,
            "pass1_seed": GENERATION_SEEDS[0],
            "pass2_seed": GENERATION_SEEDS[1],
            "pass2_policy": "only-pass1-failures",
            "pass1_pages": seed1_pages,
            "pass2_failure_pages": seed2_pages,
            "records": metrics["records"],
        },
        "t1_diagnostic": t1,
        "seconds": time.perf_counter() - started,
        "memory": cuda_memory(f"evaluation-step-{step}"),
    }
    REPORT["stages"][STAGE_NAME]["evaluations"].append(entry)
    save_report()
    print(
        f"[{STAGE_NAME}] evaluation step {step}: "
        f"seen={entry['panels']['seen']['exact_grid_pass2']}/"
        f"{entry['panels']['seen']['count']} "
        f"holdout={entry['panels']['holdout']['exact_grid_pass2']}/"
        f"{entry['panels']['holdout']['count']} "
        f"original={entry['panels']['original']['exact_grid_pass2']}/1 "
        f"seconds={entry['seconds']:.1f}"
    )
    return entry


@torch.no_grad()
def _evaluate_t1_only(
    *,
    transformer: torch.nn.Module,
    vae_decoder: Any,
    step: int,
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    train: Sequence[PreparedView],
    holdout: Sequence[PreparedView],
    original: PreparedView,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_views = tuple(train) + tuple(holdout) + (original,)
    print(
        f"[{STAGE_NAME}] evaluation step {step}: starting 641-view t=1 "
        "diagnostic without target-free generation"
    )
    t1 = _t1_diagnostic(
        transformer=transformer,
        all_text_inputs=all_text_inputs,
        all_latents=all_latents,
        all_views=all_views,
        compute_dtype=compute_dtype,
        vae_decoder=vae_decoder,
    )
    entry = {
        "step": step,
        "kind": "t1-only",
        "panels": None,
        "generation": None,
        "t1_diagnostic": t1,
        "seconds": time.perf_counter() - started,
        "memory": cuda_memory(f"evaluation-step-{step}"),
    }
    REPORT["stages"][STAGE_NAME]["evaluations"].append(entry)
    save_report()
    summaries = t1["summaries"]
    print(
        f"[{STAGE_NAME}] evaluation step {step}: t=1 only; "
        f"seen={summaries['seen']['exact_grid_count']}/"
        f"{summaries['seen']['count']} "
        f"holdout={summaries['holdout']['exact_grid_count']}/"
        f"{summaries['holdout']['count']} "
        f"seconds={entry['seconds']:.1f}"
    )
    return entry


def _mean_exposure_metrics(exposures: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not exposures:
        raise RuntimeError("Cannot summarize zero target exposures")
    result = {
        "mean_total_loss": sum(item["total_loss"] for item in exposures)
        / len(exposures),
        "mean_grid_loss": sum(item["grid_loss"] for item in exposures) / len(exposures),
        "mean_padding_loss": sum(item["padding_loss"] for item in exposures)
        / len(exposures),
    }
    optional_metrics = (
        "objective_loss",
        "boundary_loss",
        "outer_padding_loss",
        "weighted_grid_contribution",
        "weighted_boundary_contribution",
        "weighted_outer_padding_contribution",
    )
    for name in optional_metrics:
        if all(name in item for item in exposures):
            result[f"mean_{name}"] = sum(item[name] for item in exposures) / len(
                exposures
            )
    return result


def _summarize_flow_sampling(
    records: Sequence[dict[str, Any]],
    *,
    exposure_audit: dict[str, Any],
    planned_batches: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_updates = exposure_audit.get("expected_update_counts")
    if not isinstance(expected_updates, dict):
        raise RuntimeError("Planned update counts are missing")
    if len(records) != expected_updates.get("total"):
        raise RuntimeError(
            f"Flow summary received {len(records)} updates, expected "
            f"{expected_updates.get('total')}"
        )
    if len(planned_batches) != len(records):
        raise RuntimeError("Planned and observed update counts differ")
    for record, planned in zip(records, planned_batches, strict=True):
        if record.get("step") != planned.get("step"):
            raise RuntimeError("Observed optimizer steps differ from the plan")
        sampling = record.get("sampling", {})
        if sampling.get("kind") != planned.get("kind"):
            raise RuntimeError(
                f"Observed sampling kind differs at step {record.get('step')}"
            )
        if sampling.get("flow_seed") != FLOW_SEED + int(planned["step"]):
            raise RuntimeError(
                f"Observed flow seed differs at step {record.get('step')}"
            )
        observed_keys = [item.get("key") for item in record.get("per_exposure", [])]
        if observed_keys != planned.get("keys"):
            raise RuntimeError(
                f"Observed training keys differ at step {record.get('step')}"
            )
        planned_strata = planned.get("timestep_strata")
        normalized_planned_strata = (
            None if planned_strata is None else list(planned_strata)
        )
        if sampling.get("prescribed_timestep_strata") != normalized_planned_strata:
            raise RuntimeError(
                f"Observed timestep strata differ at step {record.get('step')}"
            )
    endpoint = [
        record for record in records if record["sampling"]["kind"] == "shared_endpoint"
    ]
    stratified = [
        record for record in records if record["sampling"]["kind"] == "stratified"
    ]
    if (
        len(endpoint) != expected_updates.get("shared_endpoint")
        or len(stratified) != expected_updates.get("stratified")
    ):
        raise RuntimeError("Observed endpoint/regular update counts differ")
    endpoint_exposures = [
        exposure for record in endpoint for exposure in record["per_exposure"]
    ]
    regular_exposures = [
        exposure for record in stratified for exposure in record["per_exposure"]
    ]
    expected_by_key = exposure_audit.get("items")
    if not isinstance(expected_by_key, dict):
        raise RuntimeError("Planned per-view exposure audit is missing")
    observed_by_key = {
        key: {
            "total": 0,
            "shared_endpoint": 0,
            "stratified": 0,
            "regular_strata": [],
        }
        for key in expected_by_key
    }
    for record in records:
        kind = record["sampling"]["kind"]
        exposures = record["per_exposure"]
        if len(exposures) != OPTIMIZER_BATCH_SIZE:
            raise RuntimeError(
                f"Observed update {record['step']} has {len(exposures)} exposures"
            )
        for exposure in exposures:
            key = exposure["key"]
            if key not in observed_by_key:
                raise RuntimeError(f"Observed an unplanned training key: {key}")
            if exposure["sampling_kind"] != kind:
                raise RuntimeError(
                    f"Sampling-kind mismatch for {key} at step {record['step']}"
                )
            item = observed_by_key[key]
            item["total"] += 1
            item[kind] += 1
            stratum = exposure["timestep_stratum"]
            timestep = exposure["timestep"]
            if kind == "shared_endpoint":
                if stratum is not None or timestep != 1.0:
                    raise RuntimeError(
                        f"Endpoint exposure differs from exact t=1 for {key}"
                    )
            else:
                if not isinstance(stratum, int) or not 0 <= stratum < 8:
                    raise RuntimeError(f"Invalid regular stratum for {key}: {stratum}")
                lower = stratum / OPTIMIZER_BATCH_SIZE
                upper = (stratum + 1) / OPTIMIZER_BATCH_SIZE
                if not lower <= timestep < upper:
                    raise RuntimeError(
                        f"Regular timestep {timestep} lies outside stratum {stratum}"
                    )
                item["regular_strata"].append(stratum)
    if observed_by_key != expected_by_key:
        mismatched = [
            key
            for key in expected_by_key
            if observed_by_key[key] != expected_by_key[key]
        ]
        raise RuntimeError(
            "Observed per-view exposures differ from the training plan for "
            f"{len(mismatched)} keys"
        )
    strata = {}
    for stratum in range(OPTIMIZER_BATCH_SIZE):
        matching = [
            exposure
            for exposure in regular_exposures
            if exposure["timestep_stratum"] == stratum
        ]
        if len(matching) != len(stratified):
            raise RuntimeError(
                f"Timestep stratum {stratum} has {len(matching)} exposures, "
                f"expected {len(stratified)}"
            )
        strata[str(stratum)] = {
            "interval": [stratum / 8, (stratum + 1) / 8],
            "count": len(matching),
            **_mean_exposure_metrics(matching),
        }
    return {
        "update_counts": {
            "shared_endpoint": len(endpoint),
            "stratified": len(stratified),
        },
        "exposure_counts": {
            "shared_endpoint": len(endpoint_exposures),
            "stratified": len(regular_exposures),
            "total": len(endpoint_exposures) + len(regular_exposures),
        },
        "shared_endpoint": {
            "timestep": 1.0,
            "shared_noise_across_rows": True,
            **_mean_exposure_metrics(endpoint_exposures),
        },
        "stratified_timestep_strata": strata,
        "exposure_plan_matches_observed_keys": True,
    }


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


def _validate_parent_report_payload(
    parent: dict[str, Any],
    *,
    parent_report_path: Path,
    volume_root: Path,
    volume_name: str,
    contract: ParentRunContract = UNWEIGHTED_PARENT_CONTRACT,
) -> tuple[Path, dict[str, Any]]:
    if contract.accept_terminal_report_upload_failure:
        failures = parent.get("failures")
        observed_failure = None if not failures else failures[-1]
        observed_signature = (
            None
            if not isinstance(observed_failure, dict)
            else {
                key: observed_failure.get(key)
                for key in ("stage", "type", "message")
            }
        )
        if (
            parent.get("status") != "FAIL"
            or parent.get("failed_stage") != "experiment"
            or observed_signature != REGION_WEIGHTED_PARENT_TERMINAL_FAILURE
            or not parent.get("completed_at_utc")
            or parent.get("durability", {}).get("remote_content_verified")
            is not False
        ):
            raise RuntimeError(
                "Parent run does not match the accepted terminal report-upload "
                "failure"
            )
    else:
        if parent.get("status") != "COMPLETE":
            raise RuntimeError(
                "Parent augmentation-generalization run is not COMPLETE"
            )
        if not parent.get("durability", {}).get("remote_content_verified"):
            raise RuntimeError("Parent report was not remotely content-verified")
    if parent.get("run_id") != contract.run_id:
        raise RuntimeError("Parent run ID differs from the pinned continuation parent")
    if parent.get("source", {}).get("resolved_commit") != contract.source_commit:
        raise RuntimeError("Parent source commit differs from the pinned run")
    protocol = parent.get("protocol", {})
    if protocol.get("name") != contract.protocol_name:
        raise RuntimeError("Parent report protocol differs from the fresh run")
    if (
        protocol.get("optimizer_steps") != contract.optimizer_steps
        or protocol.get("final_optimizer_step") != contract.final_step
    ):
        raise RuntimeError("Parent report training allocation differs")
    if protocol.get("loss") != contract.loss_policy:
        raise RuntimeError("Parent report loss policy differs")
    if contract.loss_policy == REGION_WEIGHTED_LOSS_POLICY and protocol.get(
        "region_loss", {}
    ).get("weights") != REGION_LOSS_WEIGHTS:
        raise RuntimeError("Parent report region-loss weights differ")
    if parent.get("pool", {}).get("pool_sha256") != contract.pool_sha256:
        raise RuntimeError("Parent report pool hash differs from the pinned run")
    if parent.get("pool", {}).get("counts") != {
        "train": TRAIN_VIEW_COUNT,
        "holdout": HOLDOUT_VIEW_COUNT,
        "original": ORIGINAL_VIEW_COUNT,
    }:
        raise RuntimeError("Parent report pool counts differ")
    stage = parent.get("stages", {}).get(STAGE_NAME)
    if not isinstance(stage, dict) or stage.get("status") != "COMPLETE":
        raise RuntimeError("Parent training stage is not COMPLETE")
    training = stage.get("training")
    if (
        not isinstance(training, dict)
        or training.get("completed_steps") != contract.optimizer_steps
        or training.get("target_exposures")
        != contract.optimizer_steps * OPTIMIZER_BATCH_SIZE
    ):
        raise RuntimeError("Parent training budget differs")
    checkpoint = stage.get("saved_checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Parent report lacks a saved transformer checkpoint")
    if (
        checkpoint.get("step") != contract.final_step
        or checkpoint.get("state_scope") != "transformer_only"
        or not checkpoint.get("remote_file_sizes_verified")
    ):
        raise RuntimeError("Parent transformer checkpoint contract differs")
    checkpoint_path = Path(checkpoint.get("path", "")).resolve()
    if not checkpoint_path.is_relative_to(volume_root):
        raise RuntimeError("Parent transformer checkpoint lies outside the Volume")
    expected_report_path = (
        volume_root
        / "results"
        / RESULT_FAMILY
        / contract.run_id
        / "report.json"
    ).resolve()
    if parent_report_path.resolve() != expected_report_path:
        raise RuntimeError("Parent report path differs from its pinned run ID")
    relative_checkpoint = checkpoint.get("volume_relative_path")
    if not isinstance(relative_checkpoint, str) or not relative_checkpoint.startswith(
        "/results/"
    ):
        raise RuntimeError("Parent checkpoint Volume-relative path is invalid")
    expected_checkpoint_path = (
        volume_root / relative_checkpoint.lstrip("/")
    ).resolve()
    if checkpoint_path != expected_checkpoint_path:
        raise RuntimeError("Parent checkpoint path differs from its Volume path")
    if checkpoint.get("volume_name") != volume_name:
        raise RuntimeError("Parent checkpoint Volume name differs")
    if parent_report_path.resolve() == checkpoint_path:
        raise RuntimeError("Parent report path unexpectedly equals checkpoint path")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("Parent checkpoint manifest is missing")
    return checkpoint_path, checkpoint


def _load_and_validate_parent_report(
    *,
    parent_report_path: Path,
    expected_sha256: str,
    volume_root: Path,
    volume_name: str,
    contract: ParentRunContract = UNWEIGHTED_PARENT_CONTRACT,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("parent-report-sha256 must be a lowercase SHA-256 digest")
    if expected_sha256 != contract.report_sha256:
        raise RuntimeError("Parent report SHA-256 differs from the pinned run")
    resolved_report = parent_report_path.resolve()
    if not resolved_report.is_relative_to(volume_root):
        raise RuntimeError("Parent report lies outside the mounted Volume")
    if not resolved_report.is_file():
        raise FileNotFoundError(resolved_report)
    actual_sha256 = sha256_file(resolved_report)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Parent report SHA-256 differs: {actual_sha256} versus {expected_sha256}"
        )
    parent = json.loads(resolved_report.read_text(encoding="utf-8"))
    checkpoint_path, checkpoint = _validate_parent_report_payload(
        parent,
        parent_report_path=resolved_report,
        volume_root=volume_root,
        volume_name=volume_name,
        contract=contract,
    )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    observed_manifest = _validate_transformer_checkpoint(checkpoint_path)
    if observed_manifest != checkpoint["manifest"]:
        raise RuntimeError("Parent transformer checkpoint hashes differ from report")
    return parent, checkpoint_path, {
        "report_path": str(resolved_report),
        "report_sha256": actual_sha256,
        "run_id": parent.get("run_id"),
        "source_commit": parent.get("source", {}).get("resolved_commit"),
        "pool_sha256": parent["pool"]["pool_sha256"],
        "report_status": parent.get("status"),
        "remote_report_content_verified": parent.get("durability", {}).get(
            "remote_content_verified"
        ),
        "accepted_terminal_report_upload_failure": (
            contract.accept_terminal_report_upload_failure
        ),
        "checkpoint": checkpoint,
        "optimizer_state_restored": False,
        "optimizer_reset_reason": "parent checkpoint contains transformer weights only",
    }


def _save_and_upload_checkpoint(
    *,
    transformer: torch.nn.Module,
    volume_name: str,
    run_id: str,
    final_step: int = OPTIMIZER_STEPS,
) -> dict[str, Any]:
    local_root = Path("/tmp") / (f"diffusion-arc-checkpoint-{run_id}-{STAGE_NAME}")
    if local_root.exists():
        raise FileExistsError(local_root)
    free_gib = shutil.disk_usage(local_root.parent).free / 2**30
    if free_gib < MINIMUM_LOCAL_CHECKPOINT_FREE_GIB:
        raise RuntimeError(f"Local checkpoint staging has only {free_gib:.1f} GiB free")
    print(
        f"[{STAGE_NAME}] saving final transformer checkpoint to local staging "
        f"({free_gib:.1f} GiB free)"
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
        f"/results/{RESULT_FAMILY}/{run_id}/{STAGE_NAME}/"
        f"flux-transformer-step-{final_step:04d}"
    )
    volume = modal.Volume.from_name(volume_name)
    upload_started = time.perf_counter()
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_root), remote_path)
    entries = tuple(volume.iterdir(remote_path, recursive=True))
    remote_sizes = {
        Path(entry.path).name: entry.size
        for entry in entries
        if Path(entry.path).name in {Path(relative).name for relative in manifest}
    }
    expected_sizes = {
        Path(relative).name: details["bytes"] for relative, details in manifest.items()
    }
    if remote_sizes != expected_sizes:
        raise RuntimeError(
            f"Remote checkpoint sizes differ: {remote_sizes} versus {expected_sizes}"
        )
    result = {
        "path": str(Path("/mnt") / volume_name / remote_path.lstrip("/")),
        "volume_name": volume_name,
        "volume_relative_path": remote_path,
        "step": final_step,
        "state_scope": "transformer_only",
        "manifest": manifest,
        "local_validation": True,
        "remote_file_sizes_verified": True,
        "save_seconds": upload_started - save_started,
        "upload_seconds": time.perf_counter() - upload_started,
    }
    shutil.rmtree(local_root)
    print(
        f"[{STAGE_NAME}] final checkpoint uploaded and size-verified in "
        f"{result['upload_seconds']:.1f}s"
    )
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
        "upload_attempts": 0,
    }
    save_report()
    staging_path = Path("/tmp") / f"diffusion-arc-report-{run_id}.json"
    volume = modal.Volume.from_name(volume_name)
    REPORT["durability"]["remote_content_verified"] = True
    save_report()
    local_bytes = REPORT_PATH.read_bytes()
    staging_path.write_bytes(local_bytes)
    last_error: Exception | None = None
    try:
        for attempt in range(1, FINAL_REPORT_UPLOAD_ATTEMPTS + 1):
            REPORT["durability"]["upload_attempts"] = attempt
            save_report()
            local_bytes = REPORT_PATH.read_bytes()
            staging_path.write_bytes(local_bytes)
            try:
                with volume.batch_upload(force=True) as batch:
                    batch.put_file(str(staging_path), remote_path)
                remote_bytes = b"".join(volume.read_file(remote_path))
                if remote_bytes != local_bytes:
                    raise RuntimeError(
                        "Remote report content differs after batch upload"
                    )
            except Exception as error:
                last_error = error
                if attempt == FINAL_REPORT_UPLOAD_ATTEMPTS:
                    break
                delay = FINAL_REPORT_RETRY_DELAYS_SECONDS[attempt - 1]
                print(
                    f"[{STAGE_NAME}] final report upload attempt {attempt} "
                    f"failed: {error}; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            return
        raise RuntimeError(
            "Final report upload failed after "
            f"{FINAL_REPORT_UPLOAD_ATTEMPTS} attempts"
        ) from last_error
    except Exception:
        REPORT["durability"]["remote_content_verified"] = False
        save_report()
        raise
    finally:
        staging_path.unlink(missing_ok=True)


def _run_training(
    *,
    transformer: torch.nn.Module,
    run_config: Any,
    train_text_inputs: FluxTextInputs,
    train_latents: Tensor,
    train: Sequence[PreparedView],
    schedule: Sequence[TrainingBatchPlan],
    all_text_inputs: FluxTextInputs,
    all_latents: Tensor,
    holdout: Sequence[PreparedView],
    original: PreparedView,
    vae_decoder: Any,
    compute_dtype: torch.dtype,
    volume_name: str,
    run_id: str,
    exposure_audit: dict[str, Any],
    initial_step: int = 0,
    final_step: int = OPTIMIZER_STEPS,
    evaluation_steps: Sequence[int] = EVALUATION_STEPS,
    full_generation_steps: Sequence[int] = EVALUATION_STEPS,
    evaluate_initial: bool = True,
    checkpoint_before_final_evaluation: bool = False,
    loss_policy: str = UNWEIGHTED_LOSS_POLICY,
    full_final_generation_panel: bool = True,
    include_t1_diagnostic: bool = True,
) -> None:
    stage_report = REPORT["stages"][STAGE_NAME]
    stage_report["status"] = "TRAINING"
    save_report()
    optimizer = build_flux_optimizer(run_config, transformer=transformer)
    step_path = REPORT_DIR / STAGE_NAME / "optimizer_steps.jsonl"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    deferred_final_evaluation = False
    torch.cuda.reset_peak_memory_stats(DEVICE)
    if evaluate_initial:
        evaluation_arguments = {
            "transformer": transformer,
            "vae_decoder": vae_decoder,
            "step": initial_step,
            "all_text_inputs": all_text_inputs,
            "all_latents": all_latents,
            "train": train,
            "holdout": holdout,
            "original": original,
            "compute_dtype": compute_dtype,
        }
        if initial_step in full_generation_steps:
            _evaluate(
                **evaluation_arguments,
                evaluation_steps=evaluation_steps,
                final_step=final_step,
                full_final_panel=full_final_generation_panel,
                include_t1_diagnostic=include_t1_diagnostic,
            )
        else:
            _evaluate_t1_only(**evaluation_arguments)
    training_started = time.perf_counter()
    with step_path.open("a", encoding="utf-8") as handle:
        for batch_plan in schedule:
            indices = torch.tensor(
                batch_plan.view_indices,
                device=DEVICE,
                dtype=torch.int64,
            )
            batch_views = tuple(train[index] for index in batch_plan.view_indices)
            record = _optimizer_step(
                transformer=transformer,
                optimizer=optimizer,
                text_inputs=_slice_text_inputs(train_text_inputs, indices),
                clean_latents=train_latents.index_select(0, indices),
                views=batch_views,
                batch_plan=batch_plan,
                compute_dtype=compute_dtype,
                loss_policy=loss_policy,
            )
            records.append(record)
            completed_updates = len(records)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            if (
                batch_plan.step % PROGRESS_EVERY == 0
                or batch_plan.step in evaluation_steps
                or completed_updates == 1
            ):
                handle.flush()
                os.fsync(handle.fileno())
            if completed_updates == 1 or batch_plan.step % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - training_started
                remaining = (
                    elapsed
                    / completed_updates
                    * (len(schedule) - completed_updates)
                )
                print(
                    f"[{STAGE_NAME}] global step {batch_plan.step}/{final_step} "
                    f"phase update {completed_updates}/{len(schedule)} "
                    f"loss={record['loss']:.5f} "
                    f"grad={record['gradient_norm']:.3f} "
                    f"peak={record['peak_allocated_gib']:.1f}GiB "
                    f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m"
                )
            if batch_plan.step in evaluation_steps:
                if (
                    checkpoint_before_final_evaluation
                    and batch_plan.step == final_step
                ):
                    deferred_final_evaluation = True
                    continue
                evaluation_arguments = {
                    "transformer": transformer,
                    "vae_decoder": vae_decoder,
                    "step": batch_plan.step,
                    "all_text_inputs": all_text_inputs,
                    "all_latents": all_latents,
                    "train": train,
                    "holdout": holdout,
                    "original": original,
                    "compute_dtype": compute_dtype,
                }
                if batch_plan.step in full_generation_steps:
                    _evaluate(
                        **evaluation_arguments,
                        evaluation_steps=evaluation_steps,
                        final_step=final_step,
                        full_final_panel=full_final_generation_panel,
                        include_t1_diagnostic=include_t1_diagnostic,
                    )
                else:
                    _evaluate_t1_only(**evaluation_arguments)

    flow_summary = _summarize_flow_sampling(
        records,
        exposure_audit=exposure_audit,
        planned_batches=_schedule_payload(schedule, train),
    )
    stage_report["training"] = {
        "loss_policy": loss_policy,
        "completed_steps": len(records),
        "initial_global_step": initial_step,
        "final_global_step": final_step,
        "cumulative_completed_steps": final_step,
        "target_exposures": len(records) * OPTIMIZER_BATCH_SIZE,
        "cumulative_target_exposures": final_step * OPTIMIZER_BATCH_SIZE,
        "mean_loss": sum(record["loss"] for record in records) / len(records),
        "first_loss": records[0]["loss"],
        "last_loss": records[-1]["loss"],
        "mean_gradient_norm_before_clipping": sum(
            record["gradient_norm"] for record in records
        )
        / len(records),
        "gradient_clipped_steps": sum(
            bool(record["gradient_clipped"]) for record in records
        ),
        "gradient_clipped_fraction": sum(
            bool(record["gradient_clipped"]) for record in records
        )
        / len(records),
        "mean_step_seconds": sum(record["seconds"] for record in records)
        / len(records),
        "flow_sampling": flow_summary,
        "per_view_exposure_audit": exposure_audit,
        "memory": cuda_memory("training-complete"),
    }
    save_report()
    checkpoint = _save_and_upload_checkpoint(
        transformer=transformer,
        volume_name=volume_name,
        run_id=run_id,
        final_step=final_step,
    )
    stage_report["saved_checkpoint"] = checkpoint
    if deferred_final_evaluation:
        stage_report["status"] = "FINAL_EVALUATION"
        save_report()
        evaluation_arguments = {
            "transformer": transformer,
            "vae_decoder": vae_decoder,
            "step": final_step,
            "all_text_inputs": all_text_inputs,
            "all_latents": all_latents,
            "train": train,
            "holdout": holdout,
            "original": original,
            "compute_dtype": compute_dtype,
        }
        if final_step in full_generation_steps:
            _evaluate(
                **evaluation_arguments,
                evaluation_steps=evaluation_steps,
                final_step=final_step,
                full_final_panel=full_final_generation_panel,
                include_t1_diagnostic=include_t1_diagnostic,
            )
        else:
            _evaluate_t1_only(**evaluation_arguments)
    stage_report["status"] = "COMPLETE"
    save_report()


def _validate_source_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError("source-commit must be an exact 40-character commit")


def run(args: argparse.Namespace) -> None:
    global REPORT, REPORT_DIR, REPORT_PATH, DEVICE

    _validate_static_protocol()
    supported_protocols = (
        PROTOCOL,
        CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_PROTOCOL,
        REGION_WEIGHTED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL,
    )
    if args.protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported protocol {args.protocol!r}; expected one of "
            f"{supported_protocols!r}"
        )
    _validate_source_commit(args.source_commit)
    continuation = args.protocol in (
        CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL,
    )
    region_weighted = args.protocol in (
        REGION_WEIGHTED_PROTOCOL,
        REGION_WEIGHTED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL,
        REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL,
    )
    parent_contract: ParentRunContract | None = None
    endpoint_cycle_positions = DEFAULT_ENDPOINT_CYCLE_POSITIONS
    include_t1_diagnostic = True
    if args.protocol == CONTINUATION_PROTOCOL:
        if args.parent_report is None or args.parent_report_sha256 is None:
            raise ValueError(
                "Continuation requires --parent-report and --parent-report-sha256"
            )
        parent_contract = UNWEIGHTED_PARENT_CONTRACT
        initial_step = CONTINUATION_START_STEP
        final_step = CONTINUATION_FINAL_STEP
        training_update_count = CONTINUATION_OPTIMIZER_STEPS
        phase_start = CONTINUATION_PHASE_START
        phase_count = CONTINUATION_PHASE_COUNT
        evaluation_steps = CONTINUATION_EVALUATION_STEPS
        full_generation_steps = CONTINUATION_FULL_GENERATION_STEPS
        config_name = CONTINUATION_CONFIG_NAME
        loss_policy = UNWEIGHTED_LOSS_POLICY
        full_final_generation_panel = True
        checkpoint_before_final_evaluation = True
        question = (
            "Does warm-starting from the verified step-512 transformer and giving "
            "the same 512 training augmentations three additional endpoint/regular "
            "sampling phases through global step 2048 improve seen fitting and "
            "transfer to the unchanged 128-view holdout?"
        )
    elif (
        args.protocol
        == REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL
    ):
        if args.parent_report is None or args.parent_report_sha256 is None:
            raise ValueError(
                "Endpoint-balanced long continuation requires --parent-report "
                "and --parent-report-sha256"
            )
        parent_contract = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PARENT_CONTRACT
        initial_step = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_START_STEP
        final_step = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FINAL_STEP
        training_update_count = (
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_OPTIMIZER_STEPS
        )
        phase_start = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_START
        phase_count = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_PHASE_COUNT
        endpoint_cycle_positions = ENDPOINT_BALANCED_CYCLE_POSITIONS
        evaluation_steps = REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_EVALUATION_STEPS
        full_generation_steps = (
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_FULL_GENERATION_STEPS
        )
        config_name = (
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_CONFIG_NAME
        )
        loss_policy = REGION_WEIGHTED_LOSS_POLICY
        full_final_generation_panel = False
        checkpoint_before_final_evaluation = True
        include_t1_diagnostic = False
        question = (
            "Does continuing the verified step-2048 transformer for 2048 more "
            "updates with the unchanged 50/50 endpoint-balanced schedule improve "
            "seen fitting or same-task augmentation transfer by global step 4096?"
        )
    elif args.protocol == REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL:
        if args.parent_report is None or args.parent_report_sha256 is None:
            raise ValueError(
                "Endpoint-balanced continuation requires --parent-report and "
                "--parent-report-sha256"
            )
        parent_contract = REGION_WEIGHTED_ENDPOINT_BALANCED_PARENT_CONTRACT
        initial_step = REGION_WEIGHTED_ENDPOINT_BALANCED_START_STEP
        final_step = REGION_WEIGHTED_ENDPOINT_BALANCED_FINAL_STEP
        training_update_count = REGION_WEIGHTED_ENDPOINT_BALANCED_OPTIMIZER_STEPS
        phase_start = REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_START
        phase_count = REGION_WEIGHTED_ENDPOINT_BALANCED_PHASE_COUNT
        endpoint_cycle_positions = ENDPOINT_BALANCED_CYCLE_POSITIONS
        evaluation_steps = REGION_WEIGHTED_ENDPOINT_BALANCED_EVALUATION_STEPS
        full_generation_steps = (
            REGION_WEIGHTED_ENDPOINT_BALANCED_FULL_GENERATION_STEPS
        )
        config_name = REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_CONFIG_NAME
        loss_policy = REGION_WEIGHTED_LOSS_POLICY
        full_final_generation_panel = False
        checkpoint_before_final_evaluation = True
        question = (
            "Does a 50/50 endpoint-balanced continuation from the verified "
            "step-1536 transformer improve condition-dependent color prediction "
            "without discarding regular trajectory training through step 2048?"
        )
    elif args.protocol == REGION_WEIGHTED_CONTINUATION_PROTOCOL:
        if args.parent_report is None or args.parent_report_sha256 is None:
            raise ValueError(
                "Region-weighted continuation requires --parent-report and "
                "--parent-report-sha256"
            )
        parent_contract = REGION_WEIGHTED_PARENT_CONTRACT
        initial_step = REGION_WEIGHTED_CONTINUATION_START_STEP
        final_step = REGION_WEIGHTED_CONTINUATION_FINAL_STEP
        training_update_count = REGION_WEIGHTED_CONTINUATION_OPTIMIZER_STEPS
        phase_start = REGION_WEIGHTED_CONTINUATION_PHASE_START
        phase_count = REGION_WEIGHTED_CONTINUATION_PHASE_COUNT
        evaluation_steps = REGION_WEIGHTED_CONTINUATION_EVALUATION_STEPS
        full_generation_steps = (
            REGION_WEIGHTED_CONTINUATION_FULL_GENERATION_STEPS
        )
        config_name = REGION_WEIGHTED_CONTINUATION_CONFIG_NAME
        loss_policy = REGION_WEIGHTED_LOSS_POLICY
        full_final_generation_panel = False
        checkpoint_before_final_evaluation = True
        question = (
            "Do two complete additional region-weighted sampling phases from "
            "the verified step-512 transformer through global step 1536 "
            "produce exact seen grids or improved same-task holdout behavior?"
        )
    elif region_weighted:
        if args.parent_report is not None or args.parent_report_sha256 is not None:
            raise ValueError("Fresh region-weighted protocol must not receive a parent")
        initial_step = 0
        final_step = OPTIMIZER_STEPS
        training_update_count = OPTIMIZER_STEPS
        phase_start = 0
        phase_count = 1
        evaluation_steps = REGION_WEIGHTED_EVALUATION_STEPS
        full_generation_steps = REGION_WEIGHTED_FULL_GENERATION_STEPS
        config_name = CONFIG_NAME
        loss_policy = REGION_WEIGHTED_LOSS_POLICY
        full_final_generation_panel = False
        checkpoint_before_final_evaluation = True
        question = (
            "Does a fresh full FLUX transformer learn the 3x3 content and retain "
            "a valid padding boundary more reliably when the flow-matching MSE "
            "is averaged separately over grid, immediate boundary, and outer "
            "padding with weights 0.6/0.3/0.1?"
        )
    else:
        if args.parent_report is not None or args.parent_report_sha256 is not None:
            raise ValueError("Fresh protocol must not receive a parent report")
        initial_step = 0
        final_step = OPTIMIZER_STEPS
        training_update_count = OPTIMIZER_STEPS
        phase_start = 0
        phase_count = 1
        evaluation_steps = EVALUATION_STEPS
        full_generation_steps = EVALUATION_STEPS
        config_name = CONFIG_NAME
        loss_policy = UNWEIGHTED_LOSS_POLICY
        full_final_generation_panel = True
        checkpoint_before_final_evaluation = False
        question = (
            "Can a fresh full FLUX transformer trained on 512 VAE-safe "
            "augmentations of ARC task 0d3d703e fit a representative seen "
            "panel and generalize to 128 semantically disjoint augmentations "
            "when every training view receives one exact t=1 endpoint contact "
            "and seven regular trajectory contacts?"
        )
    intermediate_generation_enabled = any(
        step != final_step and step in full_generation_steps
        for step in evaluation_steps
    )
    final_seen_per_geometry = (
        FINAL_SEEN_PER_GEOMETRY
        if full_final_generation_panel
        else INTERMEDIATE_SEEN_PER_GEOMETRY
    )
    final_holdout_count = (
        HOLDOUT_VIEW_COUNT
        if full_final_generation_panel
        else INTERMEDIATE_HOLDOUT_PER_GEOMETRY * len(EFFECTIVE_GEOMETRIES)
    )
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

    parent_report = None
    parent_checkpoint = None
    parent_lineage = None
    if continuation:
        if parent_contract is None:
            raise AssertionError("Continuation lacks a parent contract")
        parent_report, parent_checkpoint, parent_lineage = (
            _load_and_validate_parent_report(
                parent_report_path=args.parent_report,
                expected_sha256=args.parent_report_sha256,
                volume_root=volume_root,
                volume_name=args.volume_name,
                contract=parent_contract,
            )
        )

    REPORT_DIR = volume_root / "results" / RESULT_FAMILY / args.run_id
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_PATH = REPORT_DIR / "report.json"
    REPORT = {
        "schema_version": 1,
        "status": "INITIALIZING",
        "run_id": args.run_id,
        "question": question,
        "protocol": {
            "name": args.protocol,
            "task_id": TASK_ID,
            "target_original_pair_index": TARGET_ORIGINAL_PAIR_INDEX,
            "effective_geometries": [
                geometry.value for geometry in EFFECTIVE_GEOMETRIES
            ],
            "redundant_geometry_classes": {
                representative.value: [geometry.value for geometry in members]
                for representative, members in REDUNDANT_GEOMETRY_CLASSES.items()
            },
            "pool_counts": {
                "train": TRAIN_VIEW_COUNT,
                "holdout": HOLDOUT_VIEW_COUNT,
                "original": ORIGINAL_VIEW_COUNT,
            },
            "optimizer_steps": training_update_count,
            "initial_optimizer_step": initial_step,
            "final_optimizer_step": final_step,
            "optimizer_batch_size": OPTIMIZER_BATCH_SIZE,
            "endpoint_cycle_positions": list(endpoint_cycle_positions),
            "endpoint_update_fraction": (
                len(endpoint_cycle_positions) / UPDATE_CYCLE_LENGTH
            ),
            "evaluation_steps": list(evaluation_steps),
            "full_generation_steps": list(full_generation_steps),
            "t1_only_steps": [
                step for step in evaluation_steps if step not in full_generation_steps
            ],
            "checkpoint_saved_before_final_evaluation": (
                checkpoint_before_final_evaluation
            ),
            "intermediate_generation_counts": {
                "seen": (
                    INTERMEDIATE_SEEN_PER_GEOMETRY * len(EFFECTIVE_GEOMETRIES)
                    if intermediate_generation_enabled
                    else 0
                ),
                "holdout": (
                    INTERMEDIATE_HOLDOUT_PER_GEOMETRY * len(EFFECTIVE_GEOMETRIES)
                    if intermediate_generation_enabled
                    else 0
                ),
                "original": 1 if intermediate_generation_enabled else 0,
            },
            "final_generation_counts": {
                "seen": final_seen_per_geometry * len(EFFECTIVE_GEOMETRIES),
                "holdout": final_holdout_count,
                "original": 1,
            },
            "t1_diagnostic_counts": {
                "seen": TRAIN_VIEW_COUNT if include_t1_diagnostic else 0,
                "holdout": HOLDOUT_VIEW_COUNT if include_t1_diagnostic else 0,
                "original": 1 if include_t1_diagnostic else 0,
            },
            "generation_seeds": list(GENERATION_SEEDS),
            "pass2_policy": "second-seed-only-for-pass1-failures",
            "inference_steps": INFERENCE_STEPS,
            "checkpoint_policy": "only-final-transformer-checkpoint",
            "loss": loss_policy,
            "padding_loss_reweighting": region_weighted,
            "region_loss": (
                {
                    "weights": REGION_LOSS_WEIGHTS,
                    "aggregation": "per-view-region-mean-then-weighted-sum",
                    "boundary": "one-cell L immediately below and right of grid",
                }
                if region_weighted
                else None
            ),
            "unweighted_control": (
                {
                    "run_id": CONTINUATION_PARENT_RUN_ID,
                    "report_sha256": CONTINUATION_PARENT_REPORT_SHA256,
                    "pool_sha256": CONTINUATION_PARENT_POOL_SHA256,
                    "schedule_sha256": CONTINUATION_PARENT_SCHEDULE_SHA256,
                }
                if region_weighted
                else None
            ),
            "canvas": asdict(SPATIAL_SHAPE),
        },
        "paths": {
            "report": str(REPORT_PATH),
            "volume_root": str(volume_root),
            "flux_checkpoint": str(flux_checkpoint),
            "nvarc_checkpoint": str(nvarc_checkpoint),
            "initial_transformer_checkpoint": str(
                parent_checkpoint if continuation else flux_checkpoint
            ),
        },
        "source": {"resolved_commit": args.source_commit},
        "parent_lineage": parent_lineage,
        "failures": [],
        "pool": {"status": "PENDING"},
        "stages": {STAGE_NAME: {"status": "PENDING", "evaluations": []}},
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

        source_config = load_single_gpu_training_config(
            REPOSITORY_ROOT / "configs" / "training" / config_name
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
            run_config.loop.max_optimizer_steps != training_update_count
            or run_config.loop.optimizer_batch_size != OPTIMIZER_BATCH_SIZE
            or run_config.loop.flux_microbatch_size != OPTIMIZER_BATCH_SIZE
            or run_config.loop.min_context_pairs_before_target != 4
            or run_config.optimizer.learning_rate != 1e-5
        ):
            raise RuntimeError("Training configuration differs from protocol")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        DEVICE = prepare_single_gpu_device(run_config)
        seed_single_gpu_process(run_config.seed)
        torch.cuda.empty_cache()
        transformer_source = parent_checkpoint if continuation else None
        source_label = (
            f"verified parent transformer at global step {initial_step}"
            if continuation
            else "fresh FLUX transformer"
        )
        print(f"[{STAGE_NAME}] loading frozen NVARC/VAE and {source_label}")
        model_load_started = time.perf_counter()
        models = load_single_gpu_models(
            run_config,
            device=DEVICE,
            transformer_checkpoint=transformer_source,
        )
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
        initial_transformer_fingerprint = _transformer_fingerprint(transformer)
        print(
            f"[{STAGE_NAME}] models loaded in "
            f"{time.perf_counter() - model_load_started:.1f}s"
        )

        task = load_training_task(arc_data_root / "training" / f"{TASK_ID}.json")
        REPORT["pool"]["status"] = "BUILDING_CANDIDATES"
        save_report()
        pool_started = time.perf_counter()
        print(
            f"[{STAGE_NAME}] building {CANDIDATES_PER_GEOMETRY} candidates for "
            f"each of {len(EFFECTIVE_GEOMETRIES)} effective geometries"
        )
        original, candidates, candidate_audit = _build_candidate_registry(
            task,
            tokenizer,
        )
        print(
            f"[{STAGE_NAME}] screening {len(candidates)} candidates plus Original "
            "through the 30x30 VAE round trip"
        )
        vae_screen_started = time.perf_counter()
        candidate_latents, vae_safe_keys, vae_audit = _screen_candidate_vae_roundtrips(
            original=original,
            candidates=candidates,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
        )
        print(
            f"[{STAGE_NAME}] VAE screen complete: "
            f"{vae_audit['candidate_exact_count']}/{len(candidates)} safe in "
            f"{time.perf_counter() - vae_screen_started:.1f}s; "
            f"Original exact={vae_audit['original']['exact_grid']}"
        )
        train, holdout, selection_audit = _select_safe_pool(
            candidates,
            vae_safe_keys,
        )
        selected_keys = {view.serialized_key for view in tuple(train) + tuple(holdout)}
        if not selected_keys.issubset(vae_safe_keys):
            raise RuntimeError("Selected pool contains a VAE-ineligible candidate")

        candidate_positions = {
            view.serialized_key: index for index, view in enumerate(candidates)
        }
        selected_candidate_indices = torch.tensor(
            [
                candidate_positions[view.serialized_key]
                for view in tuple(train) + tuple(holdout)
            ]
            + [len(candidates)],
            device=DEVICE,
            dtype=torch.int64,
        )
        all_latents = candidate_latents.index_select(
            0,
            selected_candidate_indices,
        )
        train_latents = all_latents[:TRAIN_VIEW_COUNT]
        del candidate_latents, selected_candidate_indices, vae_encoder
        gc.collect()
        torch.cuda.empty_cache()

        all_views = tuple(train) + tuple(holdout) + (original,)
        all_samples = tuple(view.sample for view in all_views)
        training_ids, training_offsets = _pack_final_prefixes(
            all_samples[:TRAIN_VIEW_COUNT]
        )
        auxiliary_ids, auxiliary_offsets = _pack_final_prefixes(
            all_samples[TRAIN_VIEW_COUNT:]
        )
        torch.cuda.reset_peak_memory_stats(DEVICE)
        nvarc_started = time.perf_counter()
        print(
            f"[{STAGE_NAME}] precomputing frozen NVARC conditions in batches "
            f"{NVARC_TRAIN_BATCH_SIZE} and {NVARC_AUXILIARY_BATCH_SIZE}"
        )
        training_output = nvarc_runner(training_ids, training_offsets)
        training_hidden = training_output.sequence_hidden_states
        if len(training_hidden) != NVARC_TRAIN_BATCH_SIZE:
            raise RuntimeError("NVARC training output count differs")
        auxiliary_output = nvarc_runner(auxiliary_ids, auxiliary_offsets)
        auxiliary_hidden = auxiliary_output.sequence_hidden_states
        if len(auxiliary_hidden) != NVARC_AUXILIARY_BATCH_SIZE:
            raise RuntimeError("NVARC auxiliary output count differs")
        single_ids, single_offsets = _pack_final_prefixes((all_samples[0],))
        single_hidden = nvarc_runner(
            single_ids,
            single_offsets,
        ).sequence_hidden_states[0]
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
        all_hidden = training_hidden + auxiliary_hidden
        all_text_inputs = pad_flux_text_prefixes(all_hidden)
        train_text_inputs = _slice_text_inputs(
            all_text_inputs,
            slice(0, TRAIN_VIEW_COUNT),
        )
        nvarc_seconds = time.perf_counter() - nvarc_started
        print(f"[{STAGE_NAME}] NVARC precompute complete in {nvarc_seconds:.1f}s")
        del (
            training_output,
            auxiliary_output,
            training_hidden,
            auxiliary_hidden,
            all_hidden,
            single_hidden,
            batch_hidden,
            nvarc_runner,
        )
        gc.collect()
        torch.cuda.empty_cache()

        prior_exposure_audit = None
        if args.protocol in (
            REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL,
            REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL,
        ):
            prior_exposure_audit = parent_report.get("training_schedule", {}).get(
                "cumulative_per_view_exposure_audit"
            )
            if not isinstance(prior_exposure_audit, dict):
                raise RuntimeError(
                    "Endpoint-balanced parent lacks its cumulative exposure audit"
                )
        schedule = _build_training_schedule(
            train,
            phase_start=phase_start,
            phase_count=phase_count,
            endpoint_cycle_positions=endpoint_cycle_positions,
            prior_exposure_audit=prior_exposure_audit,
        )
        schedule_payload = _schedule_payload(schedule, train)
        schedule_hash = _hash_json(schedule_payload)
        if (
            args.protocol
            == REGION_WEIGHTED_ENDPOINT_BALANCED_CONTINUATION_PROTOCOL
            and schedule_hash != REGION_WEIGHTED_ENDPOINT_BALANCED_SCHEDULE_SHA256
        ):
            raise RuntimeError(
                "Endpoint-balanced training schedule differs from its pinned hash"
            )
        if (
            args.protocol
            == REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_CONTINUATION_PROTOCOL
            and schedule_hash
            != REGION_WEIGHTED_ENDPOINT_BALANCED_LONG_SCHEDULE_SHA256
        ):
            raise RuntimeError(
                "Endpoint-balanced long training schedule differs from its pinned "
                "hash"
            )
        exposure_audit = _schedule_exposure_audit(schedule, train)
        if not exposure_audit["all_exact"]:
            raise RuntimeError("Training exposure audit is not exact")

        pool_payload = {
            "counts": {
                "train": len(train),
                "holdout": len(holdout),
                "original": 1,
            },
            "train": [_view_payload(view, split="train") for view in train],
            "holdout": [_view_payload(view, split="holdout") for view in holdout],
            "original": _view_payload(original, split="original"),
        }
        pool_hash = _hash_json(pool_payload)
        cumulative_exposure_audit = exposure_audit
        parent_schedule_hash = None
        if region_weighted:
            if pool_hash != CONTINUATION_PARENT_POOL_SHA256:
                raise RuntimeError(
                    "Region-weighted pool differs from the unweighted baseline"
                )
            if (
                not continuation
                and schedule_hash != CONTINUATION_PARENT_SCHEDULE_SHA256
            ):
                raise RuntimeError(
                    "Region-weighted schedule differs from the unweighted baseline"
                )
        if continuation:
            if parent_contract is None:
                raise AssertionError("Continuation lacks a parent contract")
            parent_pool_payload = {
                name: parent_report["pool"][name]
                for name in ("counts", "train", "holdout", "original")
            }
            recorded_parent_pool_hash = parent_report["pool"]["pool_sha256"]
            if _hash_json(parent_pool_payload) != recorded_parent_pool_hash:
                raise RuntimeError("Parent pool payload does not match its hash")
            normalized_pool_payload = json.loads(
                json.dumps(pool_payload, sort_keys=True)
            )
            if (
                pool_hash != recorded_parent_pool_hash
                or normalized_pool_payload != parent_pool_payload
            ):
                raise RuntimeError("Regenerated pool differs from the parent run")

            parent_schedule = parent_report.get("training_schedule", {})
            parent_batches = parent_schedule.get("batches")
            parent_schedule_hash = parent_schedule.get("sha256")
            if (
                not isinstance(parent_batches, list)
                or parent_schedule_hash != parent_contract.schedule_sha256
                or _hash_json(parent_batches) != parent_schedule_hash
            ):
                raise RuntimeError("Parent training schedule does not match its hash")
            parent_global_step_range = parent_schedule.get("global_step_range")
            parent_update_count = parent_schedule.get("updates")
            if (
                not isinstance(parent_global_step_range, list)
                or len(parent_global_step_range) != 2
                or not isinstance(parent_update_count, int)
                or parent_update_count <= 0
                or parent_update_count % OPTIMIZER_STEPS
                or parent_global_step_range[0] < 1
                or parent_global_step_range[1]
                != parent_global_step_range[0] + parent_update_count - 1
                or (parent_global_step_range[0] - 1) % OPTIMIZER_STEPS
            ):
                raise RuntimeError("Parent schedule allocation is invalid")
            parent_phase_start = (
                parent_global_step_range[0] - 1
            ) // OPTIMIZER_STEPS
            parent_phase_count = parent_update_count // OPTIMIZER_STEPS
            parent_endpoint_cycle_positions = tuple(
                parent_report.get("protocol", {}).get(
                    "endpoint_cycle_positions",
                    DEFAULT_ENDPOINT_CYCLE_POSITIONS,
                )
            )
            parent_prior_exposure_audit = None
            if parent_endpoint_cycle_positions != DEFAULT_ENDPOINT_CYCLE_POSITIONS:
                parent_cumulative_exposure_audit = parent_schedule.get(
                    "cumulative_per_view_exposure_audit"
                )
                parent_local_exposure_audit = parent_schedule.get(
                    "per_view_exposure_audit"
                )
                if not isinstance(
                    parent_cumulative_exposure_audit, dict
                ) or not isinstance(parent_local_exposure_audit, dict):
                    raise RuntimeError(
                        "Balanced parent schedule lacks cumulative or local audits"
                    )
                parent_prior_exposure_audit = _subtract_exposure_audits(
                    parent_cumulative_exposure_audit,
                    parent_local_exposure_audit,
                )
            regenerated_parent_schedule = _schedule_payload(
                _build_training_schedule(
                    train,
                    phase_start=parent_phase_start,
                    phase_count=parent_phase_count,
                    endpoint_cycle_positions=parent_endpoint_cycle_positions,
                    prior_exposure_audit=parent_prior_exposure_audit,
                ),
                train,
            )
            if _hash_json(regenerated_parent_schedule) != parent_schedule_hash:
                raise RuntimeError("Regenerated base schedule differs from parent")
            parent_exposure_audit = parent_schedule.get(
                "cumulative_per_view_exposure_audit"
            ) or parent_schedule.get("per_view_exposure_audit")
            if not isinstance(parent_exposure_audit, dict):
                raise RuntimeError("Parent schedule lacks a per-view exposure audit")
            cumulative_exposure_audit = _combine_exposure_audits(
                parent_exposure_audit,
                exposure_audit,
            )
            if not cumulative_exposure_audit["all_exact"]:
                raise RuntimeError("Cumulative exposure audit is not exact")

        REPORT["pool"] = {
            "status": "COMPLETE",
            **pool_payload,
            "pool_sha256": pool_hash,
            "matches_parent_exactly": continuation,
            "matches_unweighted_baseline_exactly": region_weighted,
            "candidate_generation": candidate_audit,
            "selection": selection_audit,
            "vae_roundtrip_selection": vae_audit,
            "disjointness": {
                "augmentation_keys": True,
                "full_color_mappings": True,
                "target_free_nvarc_prefixes": True,
                "query_grids": True,
                "target_grids": True,
                "full_views": True,
                "original_reserved_before_selection": True,
            },
            "seconds": time.perf_counter() - pool_started,
        }
        REPORT["nvarc_precompute"] = {
            "physical_batch_sizes": [
                NVARC_TRAIN_BATCH_SIZE,
                NVARC_AUXILIARY_BATCH_SIZE,
            ],
            "seconds": nvarc_seconds,
            "batch1_vs_batch512_cosine_similarity": cosine,
            "batch1_vs_batch512_mean_absolute_error": mean_absolute_error,
            "all_text_shape": list(all_text_inputs.encoder_hidden_states.shape),
            "memory": cuda_memory("after-nvarc-precompute"),
        }
        REPORT["training_schedule"] = {
            "sha256": schedule_hash,
            "parent_sha256": parent_schedule_hash,
            "unweighted_baseline_sha256": (
                CONTINUATION_PARENT_SCHEDULE_SHA256 if region_weighted else None
            ),
            "matches_unweighted_baseline_exactly": (
                region_weighted and not continuation
            ),
            "updates": len(schedule),
            "global_step_range": [schedule[0].step, schedule[-1].step],
            "endpoint_steps": [
                batch.step for batch in schedule if batch.kind == "shared_endpoint"
            ],
            "update_counts": exposure_audit["expected_update_counts"],
            "exposure_counts": {
                "shared_endpoint": sum(
                    batch.kind == "shared_endpoint" for batch in schedule
                )
                * OPTIMIZER_BATCH_SIZE,
                "stratified": sum(
                    batch.kind == "stratified" for batch in schedule
                )
                * OPTIMIZER_BATCH_SIZE,
                "total": len(schedule) * OPTIMIZER_BATCH_SIZE,
            },
            "flow_seed_range": [
                FLOW_SEED + schedule[0].step,
                FLOW_SEED + schedule[-1].step,
            ],
            "parent_flow_seed_range": (
                [FLOW_SEED + 1, FLOW_SEED + initial_step]
                if continuation
                else None
            ),
            "parent_flow_seeds_disjoint": (
                schedule[0].step > initial_step
                if continuation
                else None
            ),
            "per_view_exposure_audit": exposure_audit,
            "cumulative_per_view_exposure_audit": cumulative_exposure_audit,
            "batches": schedule_payload,
        }
        REPORT["resolved_training_config"] = json.loads(
            json.dumps(asdict(run_config), default=str)
        )

        first_plan = schedule[0]
        first_indices = torch.tensor(
            first_plan.view_indices,
            device=DEVICE,
            dtype=torch.int64,
        )
        first_views = tuple(train[index] for index in first_plan.view_indices)
        with torch.no_grad():
            preflight_flow, _ = _sample_planned_flow_batch(
                train_latents.index_select(0, first_indices),
                batch_plan=first_plan,
                generator=torch.Generator(device=DEVICE).manual_seed(
                    FLOW_SEED + first_plan.step
                ),
            )
            preflight_prediction = predict_flux_velocity(
                transformer,
                preflight_flow.noisy_latents,
                _slice_text_inputs(train_text_inputs, first_indices),
                build_flux_image_position_ids(
                    OPTIMIZER_BATCH_SIZE,
                    DEVICE,
                    spatial_shape=SPATIAL_SHAPE,
                ),
                preflight_flow.timesteps,
                compute_dtype=compute_dtype,
            )
            (
                preflight_loss,
                _,
                preflight_region_losses,
                preflight_weighted_contributions,
            ) = _training_objective(
                preflight_prediction,
                preflight_flow.target_velocity,
                first_views,
                loss_policy=loss_policy,
            )
        if not bool(torch.isfinite(preflight_loss).item()):
            raise FloatingPointError("Preflight flow loss is non-finite")
        print(
            f"[{STAGE_NAME}] preflight passed with {loss_policy} loss "
            f"{float(preflight_loss.item()):.6f}; starting training"
        )
        mean_preflight_regions = preflight_region_losses.mean(dim=0).tolist()
        mean_preflight_contributions = preflight_weighted_contributions.mean(
            dim=0
        ).tolist()
        REPORT["models"] = {
            "flux_parameter_count": parameter_count,
            "initial_transformer_fingerprint": initial_transformer_fingerprint,
            "fresh_full_transformer": not continuation,
            "warm_start_from_parent_transformer": continuation,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "scheduler_state_restored": False,
            "optimizer_state_policy": (
                "fresh AdamW because the parent checkpoint is transformer-only"
                if continuation
                else "fresh AdamW for fresh transformer training"
            ),
            "nvarc_frozen": True,
            "vae_frozen": True,
            "loss_policy": loss_policy,
            "preflight_loss": float(preflight_loss.item()),
            "preflight_region_losses": {
                "raw_total": mean_preflight_regions[0],
                "grid": mean_preflight_regions[1],
                "boundary": mean_preflight_regions[2],
                "outer_padding": mean_preflight_regions[3],
                "combined_padding": mean_preflight_regions[4],
            },
            "preflight_weighted_contributions": {
                "grid": mean_preflight_contributions[0],
                "boundary": mean_preflight_contributions[1],
                "outer_padding": mean_preflight_contributions[2],
            }
            if region_weighted
            else None,
            "memory_after_preflight": cuda_memory("after-preflight"),
        }
        REPORT["stages"][STAGE_NAME]["schedule_sha256"] = schedule_hash
        REPORT["stages"][STAGE_NAME]["evaluation_steps"] = list(evaluation_steps)
        save_report()

        _run_training(
            transformer=transformer,
            run_config=run_config,
            train_text_inputs=train_text_inputs,
            train_latents=train_latents,
            train=train,
            schedule=schedule,
            all_text_inputs=all_text_inputs,
            all_latents=all_latents,
            holdout=holdout,
            original=original,
            vae_decoder=vae_decoder,
            compute_dtype=compute_dtype,
            volume_name=args.volume_name,
            run_id=args.run_id,
            exposure_audit=exposure_audit,
            initial_step=initial_step,
            final_step=final_step,
            evaluation_steps=evaluation_steps,
            full_generation_steps=full_generation_steps,
            evaluate_initial=not continuation,
            checkpoint_before_final_evaluation=(
                checkpoint_before_final_evaluation
            ),
            loss_policy=loss_policy,
            full_final_generation_panel=full_final_generation_panel,
            include_t1_diagnostic=include_t1_diagnostic,
        )
        final = REPORT["stages"][STAGE_NAME]["evaluations"][-1]
        REPORT["comparison"] = {
            "step": final["step"],
            "seen": final["panels"]["seen"],
            "holdout": final["panels"]["holdout"],
            "original": {
                **final["panels"]["original"],
                "interpretation": "diagnostic-original-view",
                "clean_vae_exact": vae_audit["original"]["exact_grid"],
            },
            "primary_same_task_generalization_metric": (
                "holdout exact-grid pass@1 and pass@2"
            ),
        }
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
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
