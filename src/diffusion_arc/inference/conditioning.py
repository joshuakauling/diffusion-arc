"""Target-free conditioning caches and packed execution primitives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import Tensor, nn

from diffusion_arc._constants import NVARC_CONTEXT_WIDTH, NVARC_MAX_TOKENS

from diffusion_arc.arc_data import ArcTokenizedEvaluationSample
from diffusion_arc.model import FluxTextInputs, NvarcEncoder
from diffusion_arc.model.flux_conditioning import pad_flux_text_prefixes
from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.model.flux_packing import PackingLayout, packing_layout
from diffusion_arc.model.flux_velocity import build_flux_image_position_ids, flux_transformer_device


def prepare_evaluation_text_inputs(
    samples: Sequence[ArcTokenizedEvaluationSample],
    *,
    nvarc_encoder: NvarcEncoder,
) -> FluxTextInputs:
    """Encode complete target-free evaluation prompts for FLUX conditioning."""

    evaluation_samples = tuple(samples)
    if not evaluation_samples:
        raise ValueError("samples must not be empty")
    if any(
        not isinstance(sample, ArcTokenizedEvaluationSample)
        for sample in evaluation_samples
    ):
        raise TypeError("samples must contain ArcTokenizedEvaluationSample values")

    input_ids = torch.tensor(
        tuple(
            token_id
            for sample in evaluation_samples
            for token_id in sample.input_ids
        ),
        dtype=torch.int64,
    )
    offsets = [0]
    for sample in evaluation_samples:
        offsets.append(offsets[-1] + len(sample.input_ids))
    sequence_offsets = torch.tensor(offsets, dtype=torch.int64)

    nvarc_output = nvarc_encoder(input_ids, sequence_offsets)
    if len(nvarc_output.sequence_hidden_states) != len(evaluation_samples):
        raise RuntimeError(
            "NVARC output count does not match target-free evaluation samples"
        )
    return pad_flux_text_prefixes(nvarc_output.sequence_hidden_states)


@dataclass(frozen=True)
class PackedGenerationInput:
    """One query identity and its detached CPU conditioning, without supervision."""

    task_id: str
    query_index: int
    text: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if type(self.query_index) is not int or self.query_index < 0:
            raise ValueError("query_index must be a non-negative integer")
        if not isinstance(self.text, Tensor):
            raise TypeError("text must be a torch.Tensor")
        if self.text.device.type != "cpu" or self.text.requires_grad:
            raise ValueError("text must be a detached CPU cache")
        if self.text.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("text must have BF16 or FP32 dtype")
        if (self.text.ndim != 2 or self.text.shape[1] != NVARC_CONTEXT_WIDTH
                or not 1 <= self.text.shape[0] <= NVARC_MAX_TOKENS):
            raise ValueError(f"text must have shape [1..{NVARC_MAX_TOKENS}, {NVARC_CONTEXT_WIDTH}]")
        if not bool(torch.isfinite(self.text).all().item()):
            raise ValueError("text must contain only finite values")


def plan_conditioning_packs(
    inputs: Sequence[PackedGenerationInput], *, max_joint_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Best-fit decreasing packs, resolving equal lengths/bins by input order."""

    values = _conditioning_sequence(inputs)
    if type(max_joint_tokens) is not int or max_joint_tokens <= 0:
        raise ValueError("max_joint_tokens must be a positive integer")
    lengths = tuple(query.text.shape[0] + TARGET_LATENT_SHAPE[0] for query in values)
    for query, length in zip(values, lengths, strict=True):
        if length > max_joint_tokens:
            raise ValueError(
                f"{query.task_id}:{query.query_index} requires {length} joint tokens, "
                f"exceeding budget {max_joint_tokens}"
            )
    packs: list[list[int]] = []
    remaining: list[int] = []
    for index in sorted(range(len(values)), key=lambda index: (-lengths[index], index)):
        candidates = [i for i, free in enumerate(remaining) if free >= lengths[index]]
        if candidates:
            selected = min(candidates, key=lambda i: (remaining[i], i))
            packs[selected].append(index)
            remaining[selected] -= lengths[index]
        else:
            packs.append([index])
            remaining.append(max_joint_tokens - lengths[index])
    return tuple(tuple(pack) for pack in packs)


def _conditioning_sequence(
    inputs: Sequence[PackedGenerationInput],
) -> tuple[PackedGenerationInput, ...]:
    values = tuple(inputs)
    if not values or any(not isinstance(query, PackedGenerationInput) for query in values):
        raise ValueError("inputs must be a non-empty sequence of PackedGenerationInput values")
    identities = [(query.task_id, query.query_index) for query in values]
    if len(set(identities)) != len(identities):
        raise ValueError("query identities must be unique")
    return values


def validate_packed_execution(
    transformer: nn.Module,
    inputs: Sequence[PackedGenerationInput],
    packs: Sequence[Sequence[int]],
    compute_dtype: torch.dtype,
    backend: object,
) -> tuple[tuple[PackedGenerationInput, ...], tuple[tuple[int, ...], ...], torch.device]:
    """Validate pack coverage and the execution contract before model forwards."""

    values = _conditioning_sequence(inputs)
    groups = tuple(tuple(pack) for pack in packs)
    flattened = [index for pack in groups for index in pack]
    if (not groups or any(not pack for pack in groups)
            or any(type(index) is not int for index in flattened)
            or sorted(flattened) != list(range(len(values)))):
        raise ValueError("packs must contain every query index exactly once, without empty packs")
    if compute_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("compute_dtype must be BF16 or FP32")
    if backend is None:
        raise ValueError("an explicitly preflighted packing backend is required")
    return values, groups, flux_transformer_device(transformer)


def stage_packed_conditioning(
    batch: Sequence[PackedGenerationInput], device: torch.device,
) -> tuple[Tensor, PackingLayout, Tensor]:
    """Stage one already validated pack, preserving its query and token order."""

    text = torch.cat(tuple(query.text for query in batch)).to(device)
    layout = packing_layout(tuple(query.text.shape[0] for query in batch), 900, device)
    return text, layout, build_flux_image_position_ids(len(batch), device)


def validate_sampling_seed(seed: int) -> None:
    """Check the integer seed range accepted by deterministic query noise."""

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seeds must be integers in [0, 2**63)")


def sample_conditioning_noise(
    query: PackedGenerationInput,
    seed: int,
    purpose: tuple[str, int | float],
    device: torch.device | str,
) -> Tensor:
    """Draw FP32 canvas noise from a validated query identity without global RNG."""

    material = json.dumps((query.task_id, query.query_index, seed, purpose), separators=(",", ":"))
    derived = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % 2**63
    generator = torch.Generator(device=device).manual_seed(derived)
    return torch.randn(TARGET_LATENT_SHAPE, device=device, dtype=torch.float32, generator=generator)


@contextmanager
def evaluation_mode(transformer: nn.Module) -> Iterator[None]:
    """Disable gradients and temporarily evaluate the model, restoring its mode."""

    previous = transformer.training
    transformer.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        transformer.train(previous)


__all__ = [
    "PackedGenerationInput", "prepare_evaluation_text_inputs", "plan_conditioning_packs",
    "validate_packed_execution", "stage_packed_conditioning", "sample_conditioning_noise",
    "validate_sampling_seed", "evaluation_mode",
]
