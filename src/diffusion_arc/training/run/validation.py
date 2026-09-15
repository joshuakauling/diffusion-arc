"""Prepare held-out CPU caches, measure flow losses, and score generated grids.

Supervision stays in this module. Packed generation receives only query identities
and cached conditioning; packing changes execution without changing noise draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random

from torch import Tensor, nn
import torch

from diffusion_arc._constants import ARC_AGI_1_ARCHIVE_SHA256
from diffusion_arc.arc_data import (
    align_evaluation_targets, load_evaluation_targets, load_evaluation_task,
    render_evaluation_prompts, render_grid_canvas, tokenize_nvarc_evaluation_sample,
)
from diffusion_arc.arc_data._grid import validate_arc_grid
from diffusion_arc.arc_data.tokenization import ArcTokenizedEvaluationSample
from diffusion_arc.arc_data.canvas import ArcGridPrediction
from diffusion_arc.arc_data.types import Grid
from diffusion_arc.inference.conditioning import (
    PackedGenerationInput,
    evaluation_mode,
    plan_conditioning_packs,
    sample_conditioning_noise,
    stage_packed_conditioning,
    validate_packed_execution,
    validate_sampling_seed,
)
from diffusion_arc.inference.generation import generate_packed_arc_samples
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.model.flux_packing import predict_packed_flux_velocity
from diffusion_arc.model.flux_vae import FluxVaeEncoder
from diffusion_arc.model.nvarc_encoder import NvarcEncoder

from ..step.encoding import plan_nvarc_batches
from ..step.loss import RegionLossWeights, build_region_loss_reduction


VALIDATION_TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


@dataclass(frozen=True)
class ValidationTarget:
    """One original query with detached CPU caches and scoring supervision.

    Conditioning is validated once at cache construction and can be handed to
    inference independently of clean latents and the expected output grid.
    """

    task_id: str
    query_index: int
    text: Tensor
    clean_latent: Tensor
    expected_grid: Grid
    conditioning: PackedGenerationInput = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditioning", PackedGenerationInput(
            self.task_id, self.query_index, self.text,
        ))
        if not isinstance(self.clean_latent, Tensor):
            raise TypeError("clean_latent must be a torch.Tensor")
        if self.clean_latent.device.type != "cpu" or self.clean_latent.requires_grad:
            raise ValueError("clean_latent must be a detached CPU cache")
        if self.clean_latent.shape != TARGET_LATENT_SHAPE:
            raise ValueError(f"clean_latent must have shape {TARGET_LATENT_SHAPE}")
        if self.clean_latent.dtype != torch.float32:
            raise ValueError("clean_latent must preserve the FP32 VAE output")
        if not bool(torch.isfinite(self.clean_latent).all().item()):
            raise ValueError("clean_latent must contain only finite values")
        validate_arc_grid(self.expected_grid)


def validate_validation_bundle(bundle: Path, *, split_seed: int = 167) -> dict:
    """Verify the frozen identities and payload hashes without reading test payloads."""

    validate_sampling_seed(split_seed)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("archive_sha256") != ARC_AGI_1_ARCHIVE_SHA256
            or type(manifest.get("split_seed")) is not int or manifest["split_seed"] != split_seed):
        raise ValueError("validation bundle has an unexpected source or split seed")
    validation = manifest["validation_task_ids"]
    sampling = manifest["sampling_task_ids"]
    final_test = manifest["final_test_task_ids"]
    if len(validation) != 100 or len(final_test) != 300 or len(set(validation + final_test)) != 400:
        raise ValueError("the split must contain 100/300 disjoint unique tasks")
    if sampling != validation[:32]:
        raise ValueError("sampling must use the fixed first 32 validation tasks")
    if any(not isinstance(key, str) or len(key) != 8 or any(c not in "0123456789abcdef" for c in key)
           for key in validation + final_test):
        raise ValueError("task identities must be eight lowercase hexadecimal characters")
    ordered = sorted(validation + final_test)
    random.Random(manifest["split_seed"]).shuffle(ordered)
    if ordered != validation + final_test:
        raise ValueError("task partition differs from the seeded split")
    if set(manifest["task_sha256"]) != set(validation):
        raise ValueError("payload hash inventory must match validation tasks")
    if {p.name for p in (bundle / "tasks").iterdir()} != {f"{key}.json" for key in validation}:
        raise ValueError("only the 100 validation payloads may be mounted")
    counts = {}
    for key in validation:
        payload = (bundle / "tasks" / f"{key}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["task_sha256"][key]:
            raise ValueError(f"validation task payload changed: {key}")
        counts[key] = len(json.loads(payload)["test"])
    if (sum(counts.values()) != manifest["validation_output_count"]
            or sum(counts[key] for key in sampling) != manifest["sampling_output_count"]):
        raise ValueError("query counts disagree with the frozen bundle manifest")
    return manifest


def tokenize_validation_targets(
    bundle: Path, manifest: dict, tokenizer,
) -> tuple[list[ArcTokenizedEvaluationSample], list[Grid]]:
    """Keep model-facing prompts and scoring-only output grids separate."""

    samples, grids, errors = [], [], []
    for key in manifest["validation_task_ids"]:
        path = bundle / "tasks" / f"{key}.json"
        task = load_evaluation_task(path)
        targets = align_evaluation_targets(task, load_evaluation_targets(path))
        for rendered, target in zip(render_evaluation_prompts(task), targets, strict=True):
            try:
                sample = tokenize_nvarc_evaluation_sample(rendered, tokenizer)
            except ValueError as error:
                errors.append(f"{key}/query-{target.query_index}: {error}")
            else:
                samples.append(sample)
                grids.append(target.output_grid)
    if errors:
        raise ValueError("Complete evaluation prompts failed preflight; no truncation or replacement:\n"
                         + "\n".join(errors))
    if len(samples) != manifest["validation_output_count"]:
        raise ValueError("tokenized query count differs from the frozen split")
    return samples, grids


def load_sampling_input_grids(
    bundle: Path, task_ids: Sequence[str],
) -> dict[tuple[str, int], Grid]:
    """Load original query inputs for presentation from the selected bundle tasks."""

    inputs = {}
    for task_id in task_ids:
        task = load_evaluation_task(bundle / "tasks" / f"{task_id}.json")
        for query in task.queries:
            inputs[task.task_id, query.query_index] = query.input_grid
    return inputs


@torch.no_grad()
def prepare_validation_cache(
    samples: Sequence[ArcTokenizedEvaluationSample],
    grids: Sequence[Grid],
    *,
    nvarc_encoder: NvarcEncoder,
    vae_encoder: FluxVaeEncoder,
    nvarc_max_tokens: int,
    vae_batch_size: int,
) -> tuple[ValidationTarget, ...]:
    """Encode complete queries once and retain only detached CPU tensors.

    Preflight every query before either frozen model runs. NVARC groups whole
    prompts by token sum; independent VAE batches encode scoring-only outputs.
    Both caches retain the original task/query order without query outputs ever
    entering the text conditioning.
    """

    values, expected_grids = tuple(samples), tuple(grids)
    if not values or len(values) != len(expected_grids):
        raise ValueError("samples and grids must be non-empty and have equal lengths")
    for name, size in (("nvarc_max_tokens", nvarc_max_tokens), ("vae_batch_size", vae_batch_size)):
        if type(size) is not int or size <= 0:
            raise ValueError(f"{name} must be a positive integer")
    identities = tuple((sample.task_id, sample.query_index) for sample in values)
    if len(set(identities)) != len(identities):
        raise ValueError("validation target identities must be unique")
    lengths = tuple(len(sample.input_ids) for sample in values)
    if any(length == 0 for length in lengths):
        raise ValueError("validation prompts must contain at least one token")
    batches = plan_nvarc_batches(lengths, max_tokens=nvarc_max_tokens)
    for grid in expected_grids:
        validate_arc_grid(grid)
    for name, encoder in (("nvarc_encoder", nvarc_encoder), ("vae_encoder", vae_encoder)):
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError(f"{name} must be frozen and in evaluation mode")

    text, latents = [], []
    for indices in batches:
        input_ids, offsets = [], [0]
        for index in indices:
            input_ids.extend(values[index].input_ids)
            offsets.append(len(input_ids))
        output = nvarc_encoder(torch.tensor(input_ids, dtype=torch.int64),
                               torch.tensor(offsets, dtype=torch.int64))
        for index, state in zip(indices, output.sequence_hidden_states, strict=True):
            if state.shape[0] != lengths[index]:
                raise RuntimeError("NVARC cache length differs from the complete query prompt")
            text.append(state.detach().to(device="cpu", copy=True))
        # Drop all GPU output references before preparing the next group.
        del state, output
    for start in range(0, len(expected_grids), vae_batch_size):
        images = torch.stack(tuple(render_grid_canvas(grid)
                                   for grid in expected_grids[start:start + vae_batch_size]))
        encoded = vae_encoder(images).detach().cpu()
        if len(encoded) != len(images):
            raise RuntimeError("VAE cache count differs from its target image batch")
        latents.extend(value.clone() for value in encoded)

    return tuple(ValidationTarget(sample.task_id, sample.query_index, state, latent, grid)
                 for sample, state, latent, grid in zip(values, text, latents, expected_grids, strict=True))


@dataclass(frozen=True)
class ExactGridDiagnostic:
    """Exact-grid diagnostic computed after generation without fallback behavior."""

    prediction_valid: bool
    invalid_reason: str | None
    predicted_shape: tuple[int, int] | None
    expected_shape: tuple[int, int]
    shape_match: bool
    cell_accuracy: float | None
    exact_grid: bool


def plan_validation_packs(
    targets: Sequence[ValidationTarget], *, max_joint_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Plan equal-target validation packs using only conditioning lengths."""

    values = _target_sequence(targets)
    return plan_conditioning_packs(
        tuple(target.conditioning for target in values), max_joint_tokens=max_joint_tokens,
    )


def evaluate_packed_loss(
    transformer: nn.Module,
    targets: Sequence[ValidationTarget],
    packs: Sequence[Sequence[int]],
    *,
    timesteps: Sequence[float],
    seed: int,
    compute_dtype: torch.dtype,
    backend: object,
    weights: RegionLossWeights = RegionLossWeights(),
) -> dict:
    """Measure equal-target region losses at five inner times and the endpoint."""

    values = _target_sequence(targets)
    _, groups, device = validate_packed_execution(
        transformer, tuple(target.conditioning for target in values), packs, compute_dtype, backend,
    )
    times = tuple(timesteps)
    if times != VALIDATION_TIMESTEPS or any(isinstance(t, bool) for t in times):
        raise ValueError(f"timesteps must be {VALIDATION_TIMESTEPS}")
    validate_sampling_seed(seed)
    # Preserve original target order for aggregation, including smaller final packs.
    results = torch.empty((len(values), len(times), 4), dtype=torch.float32)
    boundary_count = sum(
        len(target.expected_grid) < 30 or len(target.expected_grid[0]) < 30
        for target in values
    )
    with evaluation_mode(transformer):
        for indices in groups:
            batch = tuple(values[index] for index in indices)
            text, layout, positions = stage_packed_conditioning(
                tuple(target.conditioning for target in batch), device,
            )
            clean = torch.stack(tuple(target.clean_latent for target in batch)).to(device)
            reduction = build_region_loss_reduction(
                tuple((len(target.expected_grid), len(target.expected_grid[0])) for target in batch),
                device=device, weights=weights,
            )
            measurements = []
            for time in times:
                noise = torch.stack(tuple(
                    sample_conditioning_noise(target.conditioning, seed, ("loss", time), device)
                    for target in batch
                ))
                state = (1.0 - time) * clean + time * noise
                velocity = predict_packed_flux_velocity(
                    transformer, state, text, positions,
                    torch.full((len(batch),), time, device=device, dtype=torch.float32),
                    layout, compute_dtype=compute_dtype, backend=backend,
                )
                loss = reduction.compute(velocity, noise - clean)
                measurements.append(torch.cat((loss.region_means, loss.per_target[:, None]), dim=1))
            pack_results = torch.stack(measurements, dim=1).cpu()
            if not bool(torch.isfinite(pack_results).all().item()):
                raise RuntimeError("validation flow loss is non-finite")
            results[list(indices)] = pack_results

    means = results.double().mean(dim=0)
    per_timestep = []
    for time, row in zip(times, means.tolist(), strict=True):
        per_timestep.append({
            "timestep": time, "canvas_loss": row[0], "grid_loss": row[1],
            # Boundary diagnostics average only targets with an existing boundary.
            "boundary_loss": row[2] * len(values) / boundary_count if boundary_count else None,
            "weighted_loss": row[3],
        })
    inner = float(means[:-1, 3].mean().item())
    endpoint = float(means[-1, 3].item())
    text_tokens = sum(target.text.shape[0] for target in values) * len(times)
    image_tokens = len(values) * TARGET_LATENT_SHAPE[0] * len(times)
    return {
        "target_count": len(values), "boundary_target_count": boundary_count,
        "pack_count": len(groups), "actual_forward_calls": len(groups) * len(times),
        "target_forward_evaluations": len(values) * len(times),
        "nvarc_tokens": 0, "flux_text_tokens": text_tokens,
        "flux_image_tokens": image_tokens, "flux_joint_tokens": text_tokens + image_tokens,
        "flux_forward_calls": len(groups) * len(times),
        "per_timestep": per_timestep, "uniform_inner_loss": inner,
        "endpoint_loss": endpoint, "combined_loss": (11.0 * inner + endpoint) / 12.0,
    }


def generate_packed_samples(
    transformer: nn.Module,
    vae_decoder: nn.Module,
    targets: Sequence[ValidationTarget],
    packs: Sequence[Sequence[int]],
    *,
    steps: int,
    seeds: Sequence[int],
    compute_dtype: torch.dtype,
    backend: object,
) -> dict:
    """Score two independent trajectories after target-free generation completes.

    A multi-query task is solved only if one shared sample index solves every
    query; independently choosing the best sample per query is excluded. Cell
    accuracy averages shape-correct predictions equally, regardless of grid
    area. Full classified canvases remain available even for invalid predictions.
    """

    values = _target_sequence(targets)
    groups = tuple(tuple(pack) for pack in packs)
    sample_seeds = tuple(seeds)
    for seed in sample_seeds:
        validate_sampling_seed(seed)
    if len(sample_seeds) != 2 or len(set(sample_seeds)) != 2:
        raise ValueError("seeds must contain exactly two distinct integers")
    predictions = generate_packed_arc_samples(
        transformer, vae_decoder, tuple(target.conditioning for target in values), groups,
        steps=steps, seeds=sample_seeds, compute_dtype=compute_dtype, backend=backend,
    )

    # Ground truth first becomes visible here, after every trajectory and decode.
    queries = []
    task_samples: dict[str, list[bool]] = {}
    shape_correct_count, invalid_count = 0, 0
    cell_accuracy_sum = 0.0
    for target, query_predictions in zip(values, predictions, strict=True):
        diagnostics = diagnose_exact_grids(query_predictions, (target.expected_grid,) * 2)
        task_flags = task_samples.setdefault(target.task_id, [True, True])
        for sample_index, diagnostic in enumerate(diagnostics):
            task_flags[sample_index] &= diagnostic.exact_grid
            invalid_count += not diagnostic.prediction_valid
            if diagnostic.cell_accuracy is not None:
                shape_correct_count += 1
                cell_accuracy_sum += diagnostic.cell_accuracy
        queries.append({
            "task_id": target.task_id, "query_index": target.query_index,
            "samples": [dict(sample_index=i, seed=sample_seeds[i],
                             class_canvas=prediction.class_canvas.tolist(), **asdict(diagnostic))
                        for i, (prediction, diagnostic) in enumerate(
                            zip(query_predictions, diagnostics, strict=True))],
        })
    tasks = [{"task_id": task_id, "sample_exact": flags,
              "pass_at_1": flags[0], "pass_at_2": any(flags)}
             for task_id, flags in task_samples.items()]
    solved_1 = sum(task["pass_at_1"] for task in tasks)
    solved_2 = sum(task["pass_at_2"] for task in tasks)
    prediction_count = len(values) * 2
    forward_calls = len(groups) * 2 * steps
    text_tokens = sum(target.text.shape[0] for target in values) * 2 * steps
    image_tokens = len(values) * TARGET_LATENT_SHAPE[0] * 2 * steps
    return {
        "target_count": len(values), "task_count": len(tasks), "pack_count": len(groups),
        "steps": steps, "seeds": list(sample_seeds),
        "actual_forward_calls": forward_calls,
        "target_forward_evaluations": len(values) * 2 * steps,
        "nvarc_tokens": 0, "flux_text_tokens": text_tokens,
        "flux_image_tokens": image_tokens, "flux_joint_tokens": text_tokens + image_tokens,
        "flux_forward_calls": forward_calls,
        "solved_at_1": solved_1, "solved_at_2": solved_2,
        "pass_at_1": solved_1 / len(tasks), "pass_at_2": solved_2 / len(tasks),
        "prediction_count": prediction_count, "shape_correct_count": shape_correct_count,
        "shape_accuracy": shape_correct_count / prediction_count,
        "invalid_grid_rate": invalid_count / prediction_count,
        "cell_accuracy": cell_accuracy_sum / shape_correct_count if shape_correct_count else None,
        "queries": queries, "tasks": tasks,
    }


def diagnose_exact_grids(
    predictions: Sequence[ArcGridPrediction],
    expected_grids: Sequence[Grid],
) -> tuple[ExactGridDiagnostic, ...]:
    """Compare completed predictions with targets without affecting generation."""

    prediction_batch = tuple(predictions)
    expected_batch = tuple(expected_grids)
    if len(prediction_batch) != len(expected_batch):
        raise ValueError("predictions and expected_grids must have equal lengths")

    diagnostics: list[ExactGridDiagnostic] = []
    for prediction, expected_grid in zip(
        prediction_batch,
        expected_batch,
        strict=True,
    ):
        if not isinstance(prediction, ArcGridPrediction):
            raise TypeError("predictions must contain ArcGridPrediction values")
        validate_arc_grid(expected_grid)
        expected_shape = (len(expected_grid), len(expected_grid[0]))
        predicted_shape = (
            None
            if prediction.height is None or prediction.width is None
            else (prediction.height, prediction.width)
        )
        shape_match = bool(
            prediction.valid and predicted_shape == expected_shape
        )
        cell_accuracy: float | None = None
        exact_grid = False
        if shape_match:
            if prediction.grid is None:
                raise AssertionError("a valid shape-matched prediction must expose a grid")
            predicted_tensor = torch.tensor(prediction.grid, dtype=torch.int64)
            expected_tensor = torch.tensor(expected_grid, dtype=torch.int64)
            matches = predicted_tensor == expected_tensor
            cell_accuracy = float(matches.to(dtype=torch.float32).mean().item())
            exact_grid = bool(matches.all().item())

        diagnostics.append(
            ExactGridDiagnostic(
                prediction_valid=prediction.valid,
                invalid_reason=prediction.invalid_reason,
                predicted_shape=predicted_shape,
                expected_shape=expected_shape,
                shape_match=shape_match,
                cell_accuracy=cell_accuracy,
                exact_grid=exact_grid,
            )
        )
    return tuple(diagnostics)


def _target_sequence(targets):
    values = tuple(targets)
    if not values or any(not isinstance(target, ValidationTarget) for target in values):
        raise ValueError("targets must be a non-empty sequence of ValidationTarget values")
    identities = [(target.task_id, target.query_index) for target in values]
    if len(set(identities)) != len(identities):
        raise ValueError("validation target identities must be unique")
    return values


__all__ = [
    "VALIDATION_TIMESTEPS", "ValidationTarget", "plan_validation_packs",
    "validate_validation_bundle", "tokenize_validation_targets", "prepare_validation_cache",
    "load_sampling_input_grids",
    "evaluate_packed_loss", "generate_packed_samples", "ExactGridDiagnostic", "diagnose_exact_grids",
]
