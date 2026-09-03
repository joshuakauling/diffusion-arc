"""NVARC tokenization of rendered ARC training and evaluation prompts."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from diffusion_arc._constants import NVARC_MAX_TOKENS

from .augmentation import ArcAugmentationKey
from .prompt import ArcRenderedEvaluationSample, ArcRenderedTrainingSample
from .types import Grid


@dataclass(frozen=True)
class ArcTokenizedTrainingSample:
    """One NVARC token sequence with aligned target-prefix boundaries."""

    key: ArcAugmentationKey
    input_ids: tuple[int, ...]
    prefix_token_ends: tuple[int, ...]
    target_grids: tuple[Grid, ...]


@dataclass(frozen=True)
class ArcTokenizedEvaluationSample:
    """One target-free NVARC token sequence for an evaluation query."""

    task_id: str
    query_index: int
    input_ids: tuple[int, ...]


def tokenize_nvarc_training_sample(
    sample: ArcRenderedTrainingSample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = NVARC_MAX_TOKENS,
) -> ArcTokenizedTrainingSample:
    """Keep the full prompt and limit the final consumed NVARC prefix."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    input_ids = _encode(tokenizer, sample.prompt)

    prefix_token_ends: list[int] = []
    target_grids: list[Grid] = []
    for target in sample.targets:
        character_end = target.prefix_character_end
        if not 0 < character_end <= len(sample.prompt):
            raise ValueError("prefix character boundary lies outside the prompt")

        prefix_ids = _encode(tokenizer, sample.prompt[:character_end])
        token_end = len(prefix_ids)
        if input_ids[:token_end] != prefix_ids:
            raise ValueError("prefix character boundary is not a stable token boundary")

        prefix_token_ends.append(token_end)
        target_grids.append(target.target_grid)

    source_length = prefix_token_ends[-1]
    if source_length > max_tokens:
        raise ValueError(
            f"NVARC source prefix contains {source_length} tokens, "
            f"exceeding {max_tokens}"
        )

    return ArcTokenizedTrainingSample(
        key=sample.key,
        input_ids=input_ids,
        prefix_token_ends=tuple(prefix_token_ends),
        target_grids=tuple(target_grids),
    )


def tokenize_nvarc_evaluation_sample(
    sample: ArcRenderedEvaluationSample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = NVARC_MAX_TOKENS,
) -> ArcTokenizedEvaluationSample:
    """Tokenize one complete evaluation prompt without padding or truncation."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    input_ids = _encode(tokenizer, sample.prompt)
    if len(input_ids) > max_tokens:
        raise ValueError(
            f"NVARC prompt contains {len(input_ids)} tokens, exceeding {max_tokens}"
        )

    return ArcTokenizedEvaluationSample(
        task_id=sample.task_id,
        query_index=sample.query_index,
        input_ids=input_ids,
    )


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text, add_special_tokens=False))


__all__ = [
    "ArcTokenizedEvaluationSample",
    "ArcTokenizedTrainingSample",
    "NVARC_MAX_TOKENS",
    "tokenize_nvarc_evaluation_sample",
    "tokenize_nvarc_training_sample",
]
