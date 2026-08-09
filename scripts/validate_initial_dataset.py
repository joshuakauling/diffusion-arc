#!/usr/bin/env python3
"""Validate and summarize the pinned initial ARC-AGI-2 dataset subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.dataset import (  # noqa: E402
    load_dataset_manifest,
    resolve_arc_dataset,
    sha256_file,
    summarize_samples,
)


DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs/datasets/arc_agi_2_initial.json"
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data/benchmarks/kaggle-arc-prize-2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the pinned initial ARC-AGI-2 smoke and tiny subsets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--challenges",
        type=Path,
        default=DEFAULT_DATA_ROOT / "arc-agi_training_challenges.json",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=DEFAULT_DATA_ROOT / "arc-agi_training_solutions.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_dataset_manifest(args.manifest)
    largest_subset, largest_size = max(
        manifest.subset_prefix_lengths,
        key=lambda item: item[1],
    )
    samples = resolve_arc_dataset(
        manifest,
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        subset=largest_subset,
    )
    if len(samples) != largest_size:
        raise AssertionError("resolved largest subset has an unexpected size")

    report = {
        "status": "passed",
        "manifest": {
            "path": str(args.manifest.expanduser().resolve()),
            "schema_version": manifest.schema_version,
            "name": manifest.name,
            "benchmark": manifest.benchmark,
            "source_split": manifest.source_split,
        },
        "sources": {
            "challenges": {
                "path": str(args.challenges.expanduser().resolve()),
                "sha256": sha256_file(args.challenges),
            },
            "solutions": {
                "path": str(args.solutions.expanduser().resolve()),
                "sha256": sha256_file(args.solutions),
            },
        },
        "subsets": {
            subset_name: summarize_samples(samples[:subset_size])
            for subset_name, subset_size in manifest.subset_prefix_lengths
        },
        "limitations": [
            "Prompt lengths are measured in characters, not exact NVARC tokenizer tokens.",
            "All targets still use the fixed 30x30 padded canvas and 900 FLUX image tokens.",
            "These curated debug subsets are not representative validation splits.",
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
