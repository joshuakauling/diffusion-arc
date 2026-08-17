#!/usr/bin/env python3
"""Generate configured ARC training-view augmentation key files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.arc_data import generate_augmented_key_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible augmented-key files for ARC training views."
    )
    parser.add_argument("split_directory", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    views = generate_augmented_key_files(
        args.split_directory,
        data_root=args.data_root,
        overwrite=args.overwrite,
    )
    report = {
        "split_directory": str(args.split_directory.resolve()),
        "views": {name: len(keys) for name, keys in views.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
