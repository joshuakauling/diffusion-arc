#!/usr/bin/env python3
"""Print a copy-ready provider-neutral prompt for one local ARC-AGI-2 task."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from diffusion_arc.serialization import serialize_arc_task_for_llm  # noqa: E402


DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data/raw/arc-agi-2"
SPLITS = ("training", "evaluation")
TASK_ID_PATTERN = re.compile(r"[0-9a-f]{8}")


def _task_id(value: str) -> str:
    normalized = value.strip().lower()
    if not TASK_ID_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "task ID must contain exactly eight hexadecimal characters"
        )
    return normalized


def load_arc_task(
    task_id: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[Mapping[str, Any], str]:
    """Load one uniquely matching ARC task from the training or evaluation split."""

    normalized_id = _task_id(task_id)
    matches = [
        (split, data_root / split / f"{normalized_id}.json")
        for split in SPLITS
        if (data_root / split / f"{normalized_id}.json").is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"task {normalized_id!r} was not found in training or evaluation under "
            f"{data_root}"
        )
    if len(matches) > 1:
        found_splits = ", ".join(split for split, _ in matches)
        raise ValueError(f"task {normalized_id!r} exists in multiple splits: {found_splits}")

    split, task_path = matches[0]
    with task_path.open("r", encoding="utf-8") as handle:
        task = json.load(handle)
    if not isinstance(task, Mapping):
        raise TypeError(f"task file must contain a JSON object: {task_path}")
    return task, split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a provider-neutral LLM prompt for a local ARC-AGI-2 task. "
            "When TASK_ID is omitted, it is requested interactively."
        )
    )
    parser.add_argument("task_id", nargs="?", type=_task_id, metavar="TASK_ID")
    parser.add_argument(
        "--space-separated",
        action="store_true",
        help="insert one ASCII space between adjacent grid cells",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="directory containing training/ and evaluation/ task folders",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    task_id = args.task_id
    if task_id is None:
        try:
            task_id = _task_id(input("ARC task ID: "))
        except EOFError:
            parser.error("no task ID was provided")
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))

    try:
        task, _split = load_arc_task(task_id, data_root=args.data_root)
        prompt = serialize_arc_task_for_llm(
            task,
            space_separated=args.space_separated,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
