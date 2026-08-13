"""NVARC prompt rendering for augmented ARC training samples."""

from __future__ import annotations

from dataclasses import dataclass

from .augmentation import ArcAugmentationKey, ArcAugmentedTrainingSample
from .types import Grid


_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


@dataclass(frozen=True)
class ArcPromptTarget:
    """One target grid and its end-exclusive character prefix boundary."""

    prefix_character_end: int
    target_grid: Grid


@dataclass(frozen=True)
class ArcRenderedTrainingSample:
    """One complete teacher-forced NVARC training prompt and its targets."""

    key: ArcAugmentationKey
    prompt: str
    targets: tuple[ArcPromptTarget, ...]


def render_training_prompt(
    sample: ArcAugmentedTrainingSample,
) -> ArcRenderedTrainingSample:
    """Render all pairs and record target boundaries after the first pair."""

    if not isinstance(sample, ArcAugmentedTrainingSample):
        raise TypeError("sample must be an ArcAugmentedTrainingSample")
    if len(sample.pairs) < 2:
        raise ValueError("a training prompt requires at least two pairs")

    parts: list[str] = []
    character_count = 0
    targets: list[ArcPromptTarget] = []

    for pair_position, pair in enumerate(sample.pairs):
        user_turn = (
            f"{_IM_START}user\n{_serialize_grid(pair.input_grid)}{_IM_END}"
        )
        assistant_header = f"{_IM_START}assistant\n"
        parts.extend((user_turn, assistant_header))
        character_count += len(user_turn) + len(assistant_header)

        if pair_position > 0:
            targets.append(
                ArcPromptTarget(
                    prefix_character_end=character_count,
                    target_grid=pair.output_grid,
                )
            )

        assistant_output = f"{_serialize_grid(pair.output_grid)}{_IM_END}"
        parts.append(assistant_output)
        character_count += len(assistant_output)

    return ArcRenderedTrainingSample(
        key=sample.key,
        prompt="".join(parts),
        targets=tuple(targets),
    )


def _serialize_grid(grid: Grid) -> str:
    return "\n".join("".join(str(value) for value in row) for row in grid)


__all__ = [
    "ArcPromptTarget",
    "ArcRenderedTrainingSample",
    "render_training_prompt",
]
