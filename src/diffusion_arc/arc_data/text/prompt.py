"""NVARC prompt rendering for ARC training and evaluation samples."""

from __future__ import annotations

from dataclasses import dataclass

from ..augmentation import ArcAugmentationKey, ArcAugmentedTrainingSample
from ..types import ArcEvaluationTask, Grid


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


@dataclass(frozen=True)
class ArcRenderedEvaluationSample:
    """One target-free NVARC evaluation prompt for a single query."""

    task_id: str
    query_index: int
    prompt: str


def render_training_prompt(
    sample: ArcAugmentedTrainingSample,
) -> ArcRenderedTrainingSample:
    """Render all pairs and record target boundaries after the first pair."""

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


def render_evaluation_prompts(
    task: ArcEvaluationTask,
) -> tuple[ArcRenderedEvaluationSample, ...]:
    """Render one target-free prompt per query using every demonstration."""

    demonstration_parts: list[str] = []
    for pair in task.demonstrations:
        demonstration_parts.extend(
            (
                f"{_IM_START}user\n{_serialize_grid(pair.input_grid)}{_IM_END}",
                f"{_IM_START}assistant\n"
                f"{_serialize_grid(pair.output_grid)}{_IM_END}",
            )
        )
    demonstration_prompt = "".join(demonstration_parts)

    return tuple(
        ArcRenderedEvaluationSample(
            task_id=task.task_id,
            query_index=query.query_index,
            prompt=(
                f"{demonstration_prompt}"
                f"{_IM_START}user\n{_serialize_grid(query.input_grid)}{_IM_END}"
                f"{_IM_START}assistant\n"
            ),
        )
        for query in task.queries
    )


def _serialize_grid(grid: Grid) -> str:
    return "\n".join("".join(str(value) for value in row) for row in grid)


__all__ = [
    "ArcPromptTarget",
    "ArcRenderedEvaluationSample",
    "ArcRenderedTrainingSample",
    "render_evaluation_prompts",
    "render_training_prompt",
]
