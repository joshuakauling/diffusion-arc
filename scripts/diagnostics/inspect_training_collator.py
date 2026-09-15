#!/usr/bin/env python3
"""Inspect collation of local ARC tasks without loading model checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

from diffusion_arc.arc_data import (  # noqa: E402
    NVARC_MAX_TOKENS,
    ArcTokenizedTrainingSample,
    apply_training_augmentation,
    load_training_task,
    make_identity_training_augmentation,
    render_training_prompt,
    tokenize_nvarc_training_sample,
)
from diffusion_arc.training.data import validate_packed_training_batch  # noqa: E402
from diffusion_arc.training.data import CollatedTrainingBatch, TrainingCollator  # noqa: E402
from _nvarc_tokenizer import make_nvarc_tokenizer  # noqa: E402


def build_tokenized_samples(
    task_paths: tuple[Path, ...],
    *,
    max_pairs: int,
    max_cells: int,
    max_tokens: int,
) -> tuple[ArcTokenizedTrainingSample, ...]:
    """Build deterministic identity samples independently of target selection."""

    tokenizer = make_nvarc_tokenizer(model_max_length=max_tokens)
    tokenized_samples: list[ArcTokenizedTrainingSample] = []

    for task_path in task_paths:
        task = load_training_task(task_path)
        augmentation = make_identity_training_augmentation(
            task,
            max_pairs=max_pairs,
            max_cells=max_cells,
        )

        augmented = apply_training_augmentation(task, augmentation)
        rendered = render_training_prompt(augmented)
        tokenized_samples.append(
            tokenize_nvarc_training_sample(
                rendered,
                tokenizer,
                max_tokens=max_tokens,
            )
        )

    return tuple(tokenized_samples)


def inspect_collator(args: argparse.Namespace) -> CollatedTrainingBatch:
    """Build, validate, and print one host-side collated training batch."""

    task_paths = tuple(args.task_paths)
    collator = TrainingCollator(
        max_targets_per_source=args.max_targets_per_source
    )
    samples = build_tokenized_samples(
        task_paths,
        max_pairs=args.max_pairs,
        max_cells=args.max_cells,
        max_tokens=args.max_tokens,
    )
    batch = collator(samples)
    validate_packed_training_batch(batch.tensors)

    tensors = batch.tensors
    offsets = tensors.sequence_offsets.tolist()
    target_sources = tensors.target_sequence_indices.tolist()
    target_prefixes = tensors.target_prefix_ends.tolist()

    print("=== COLLATOR CONFIGURATION ===")
    print(f"sources: {len(samples)}")
    print(f"max_pairs_per_view: {args.max_pairs}")
    target_limit = args.max_targets_per_source
    print(f"max_targets_per_source: {target_limit if target_limit is not None else 'all'}")
    print(f"total_targets: {len(target_sources)}")

    target_start = 0
    for source_index, (task_path, sample, key) in enumerate(
        zip(task_paths, samples, batch.metadata.source_keys, strict=True)
    ):
        selected_target_count = target_sources.count(source_index)
        target_end = target_start + selected_target_count
        collated_token_count = offsets[source_index + 1] - offsets[source_index]
        selected_candidate_start = len(sample.target_grids) - selected_target_count
        selected_candidate_positions = tuple(
            range(selected_candidate_start, len(sample.target_grids))
        )
        print(f"\n=== SOURCE {source_index} ===")
        print(f"task_path: {task_path}")
        print(f"key: {key}")
        print(f"token_span: [{offsets[source_index]}, {offsets[source_index + 1]})")
        print(f"full_token_count: {len(sample.input_ids)}")
        print(f"collated_token_count: {collated_token_count}")
        print(f"available_targets: {len(sample.target_grids)}")
        print(f"selected_targets: {selected_target_count}")
        print(f"candidate_prefix_token_ends: {sample.prefix_token_ends}")
        print(f"selected_candidate_positions: {selected_candidate_positions}")
        print(
            "selected_prefix_token_ends: "
            f"{tuple(target_prefixes[target_start:target_end])}"
        )
        print(
            "target_pair_indices: "
            f"{batch.metadata.target_pair_indices[target_start:target_end]}"
        )
        target_start = target_end

    print("\n=== PACKED TRAINING BATCH ===")
    _print_tensor("input_ids", tensors.input_ids)
    print(f"sequence_offsets: {offsets}")
    print(f"target_sequence_indices: {target_sources}")
    print(f"target_prefix_ends: {target_prefixes}")
    _print_tensor("target_images", tensors.target_images)
    image_minimum = tensors.target_images.min().item()
    image_maximum = tensors.target_images.max().item()
    print(f"target_image_range: [{image_minimum}, {image_maximum}]")
    print("contract_validation: valid")
    return batch


def _print_tensor(name: str, tensor: Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task_paths",
        nargs="+",
        type=Path,
        help="paths to one or more ARC task JSON files",
    )
    parser.add_argument("--max-pairs", type=int, default=6)
    parser.add_argument(
        "--max-targets-per-source",
        type=int,
        default=None,
        help="select at most this many suffix targets per source; default: all",
    )
    parser.add_argument("--max-cells", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=NVARC_MAX_TOKENS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        inspect_collator(args)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
