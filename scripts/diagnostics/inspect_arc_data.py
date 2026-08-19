#!/usr/bin/env python3
"""Print the training and evaluation data views of one local ARC task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random

from transformers import PreTrainedTokenizerFast


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

from diffusion_arc.arc_data import (  # noqa: E402
    NVARC_MAX_TOKENS,
    ArcAugmentationSpec,
    ArcTrainingTask,
    align_evaluation_targets,
    apply_training_augmentation,
    load_evaluation_targets,
    load_evaluation_task,
    load_training_task,
    make_identity_training_augmentation,
    render_evaluation_prompts,
    render_grid_canvas,
    render_training_prompt,
    render_training_target_canvases,
    sample_training_augmentation,
    tokenize_nvarc_evaluation_sample,
    tokenize_nvarc_training_sample,
)
from _nvarc_tokenizer import make_nvarc_tokenizer  # noqa: E402


def print_training_view(
    label: str,
    task: ArcTrainingTask,
    augmentation: ArcAugmentationSpec,
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_tokens: int,
) -> None:
    augmented = apply_training_augmentation(task, augmentation)
    rendered = render_training_prompt(augmented)
    tokenized = tokenize_nvarc_training_sample(
        rendered,
        tokenizer,
        max_tokens=max_tokens,
    )
    canvases = render_training_target_canvases(rendered)

    print(f"\n=== {label} ===")
    print(f"key: {rendered.key}")
    print(f"tokens: {len(tokenized.input_ids)}")
    print(f"prefix_token_ends: {tokenized.prefix_token_ends}")
    print(f"target_canvases: {tuple(canvases.shape)}")
    print("prompt:")
    print(rendered.prompt)


def inspect_task(args: argparse.Namespace) -> None:
    tokenizer = make_nvarc_tokenizer(model_max_length=NVARC_MAX_TOKENS)
    training_task = load_training_task(args.task_path)

    identity = make_identity_training_augmentation(
        training_task,
        max_pairs=args.max_pairs,
        max_cells=args.max_cells,
    )
    print_training_view(
        "TRAINING: IDENTITY",
        training_task,
        identity,
        tokenizer,
        max_tokens=args.max_tokens,
    )

    rng = Random(args.seed)
    for index in range(args.augmentations):
        augmentation = sample_training_augmentation(
            training_task,
            augmentation_index=index,
            max_pairs=args.max_pairs,
            max_cells=args.max_cells,
            rng=rng,
        )
        print_training_view(
            f"TRAINING: AUGMENTATION {index}",
            training_task,
            augmentation,
            tokenizer,
            max_tokens=args.max_tokens,
        )

    evaluation_task = load_evaluation_task(args.task_path)
    evaluation_targets = load_evaluation_targets(args.task_path)
    aligned_targets = align_evaluation_targets(
        evaluation_task,
        evaluation_targets,
    )

    for rendered in render_evaluation_prompts(evaluation_task):
        tokenized = tokenize_nvarc_evaluation_sample(
            rendered,
            tokenizer,
            max_tokens=args.max_tokens,
        )
        print(f"\n=== EVALUATION: QUERY {rendered.query_index} ===")
        print(f"tokens: {len(tokenized.input_ids)}")
        print("prompt:")
        print(rendered.prompt)

    print("\n=== HELD-OUT TARGETS (NOT PART OF THE PROMPTS) ===")
    for target in aligned_targets:
        canvas = render_grid_canvas(target.output_grid)
        print(f"query {target.query_index}: target_canvas={tuple(canvas.shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_path", type=Path, help="path to one ARC task JSON file")
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=167)
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument("--max-cells", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=NVARC_MAX_TOKENS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.augmentations < 0:
        parser.error("--augmentations must be non-negative")

    try:
        inspect_task(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
