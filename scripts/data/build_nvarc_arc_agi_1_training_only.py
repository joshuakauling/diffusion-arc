#!/usr/bin/env python3
"""Build the NVARC subset whose two parents are ARC-AGI-1 training tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "nvarc_data"
DEFAULT_SOURCE_TASKS = DEFAULT_DATA_ROOT / "nvarc_train_tasks.jsonl"
DEFAULT_OFFICIAL_KEYS = DEFAULT_DATA_ROOT / "arc_agi_1_training_keys.json"
DEFAULT_OUTPUT_TASKS = (
    DEFAULT_DATA_ROOT / "nvarc_train_arc_agi_1_training_only_tasks.jsonl"
)
DEFAULT_OUTPUT_KEYS = (
    DEFAULT_DATA_ROOT / "nvarc_train_arc_agi_1_training_only_keys.json"
)
DERIVED_GROUP = "nvarc_train_arc_agi_1_training_only"
EXPECTED_SOURCE_TASKS = 47_337
EXPECTED_OFFICIAL_KEYS = 400
TASK_ID_PATTERN = re.compile(r"[0-9a-f]{8}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path, default=DEFAULT_SOURCE_TASKS)
    parser.add_argument("--official-keys", type=Path, default=DEFAULT_OFFICIAL_KEYS)
    parser.add_argument("--output-tasks", type=Path, default=DEFAULT_OUTPUT_TASKS)
    parser.add_argument("--output-keys", type=Path, default=DEFAULT_OUTPUT_KEYS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files after successful validation.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_jsonl_record(record: dict[str, object]) -> bytes:
    return (
        json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_official_keys(path: Path) -> tuple[set[str], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("group") != "arc_agi_1_training":
        raise ValueError(f"Unexpected official key payload: {path}")
    keys = payload.get("keys")
    if not isinstance(keys, list) or any(
        not isinstance(key, str) or TASK_ID_PATTERN.fullmatch(key) is None
        for key in keys
    ):
        raise ValueError(f"Invalid ARC-AGI-1 training keys: {path}")
    key_set = set(keys)
    if (
        len(keys) != EXPECTED_OFFICIAL_KEYS
        or len(key_set) != len(keys)
        or keys != sorted(keys)
        or payload.get("key_count") != len(keys)
    ):
        raise ValueError(f"ARC-AGI-1 training key inventory differs: {path}")
    return key_set, payload


def make_temporary_output(path: Path, *, overwrite: bool) -> tuple[Path, BinaryIO]:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {path}; pass --overwrite intentionally."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.fchmod(descriptor, 0o644)
    return Path(temporary_name), os.fdopen(descriptor, "wb")


def validate_source_record(record: object, *, line_number: int) -> dict[str, object]:
    if not isinstance(record, dict):
        raise TypeError(f"Task line {line_number} is not a JSON object")
    required = {
        "archive_path",
        "parent1_arc_agi_1_split",
        "parent1_id",
        "parent2_arc_agi_1_split",
        "parent2_id",
        "source_subset",
        "task_key",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Task line {line_number} is missing fields: {sorted(missing)}")
    parent1 = record["parent1_id"]
    parent2 = record["parent2_id"]
    if (
        not isinstance(parent1, str)
        or TASK_ID_PATTERN.fullmatch(parent1) is None
        or not isinstance(parent2, str)
        or TASK_ID_PATTERN.fullmatch(parent2) is None
    ):
        raise ValueError(f"Task line {line_number} has invalid parent IDs")
    if record["task_key"] != f"{parent1}_{parent2}":
        raise ValueError(f"Task line {line_number} has an inconsistent task key")
    if record["source_subset"] != "nvarc_training":
        raise ValueError(f"Task line {line_number} is not from nvarc_training")
    return record


def build_subset(
    *,
    source_tasks: Path,
    official_keys_path: Path,
    output_tasks: Path,
    output_keys: Path,
    overwrite: bool,
) -> dict[str, object]:
    if output_tasks.resolve() == output_keys.resolve():
        raise ValueError("Task and key outputs must use different paths")
    official_keys, official_payload = load_official_keys(official_keys_path)
    source_tasks_sha256 = sha256_file(source_tasks)
    official_keys_sha256 = sha256_file(official_keys_path)

    temporary_tasks, tasks_output = make_temporary_output(
        output_tasks, overwrite=overwrite
    )
    temporary_keys: Path | None = None
    source_task_count = 0
    selected_task_count = 0
    selected_parent_keys: set[str] = set()
    selected_task_keys: set[str] = set()
    selected_archive_paths: set[str] = set()
    tasks_digest = hashlib.sha256()

    try:
        with tasks_output:
            with source_tasks.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    if not raw_line.strip():
                        raise ValueError(f"Empty task line: {line_number}")
                    source_task_count += 1
                    record = validate_source_record(
                        json.loads(raw_line), line_number=line_number
                    )
                    if not (
                        record["parent1_arc_agi_1_split"] == "training"
                        and record["parent2_arc_agi_1_split"] == "training"
                    ):
                        continue

                    parent_keys = {record["parent1_id"], record["parent2_id"]}
                    if not parent_keys <= official_keys:
                        invalid = sorted(parent_keys - official_keys)
                        raise ValueError(
                            f"Task line {line_number} uses non-training parents: {invalid}"
                        )
                    task_key = record["task_key"]
                    archive_path = record["archive_path"]
                    if task_key in selected_task_keys:
                        raise ValueError(f"Duplicate selected task key: {task_key}")
                    if archive_path in selected_archive_paths:
                        raise ValueError(
                            f"Duplicate selected archive path: {archive_path}"
                        )

                    encoded = canonical_jsonl_record(record)
                    tasks_output.write(encoded)
                    tasks_digest.update(encoded)
                    selected_task_count += 1
                    selected_parent_keys.update(parent_keys)
                    selected_task_keys.add(task_key)
                    selected_archive_paths.add(archive_path)
            tasks_output.flush()
            os.fsync(tasks_output.fileno())

        if source_task_count != EXPECTED_SOURCE_TASKS:
            raise ValueError(
                f"Source task count differs: {source_task_count} != "
                f"{EXPECTED_SOURCE_TASKS}"
            )
        missing_official_keys = sorted(official_keys - selected_parent_keys)
        key_payload = {
            "schema_version": 1,
            "group": DERIVED_GROUP,
            "definition": (
                "Tasks from nvarc_training for which both parent keys are in the "
                "official pinned ARC-AGI-1 training split."
            ),
            "selection": {
                "source_subset": "nvarc_training",
                "parent1_arc_agi_1_split": "training",
                "parent2_arc_agi_1_split": "training",
            },
            "source": {
                "tasks_file": source_tasks.name,
                "tasks_sha256": source_tasks_sha256,
                "official_keys_file": official_keys_path.name,
                "official_keys_sha256": official_keys_sha256,
                "official_keys_group": official_payload["group"],
            },
            "source_task_count": source_task_count,
            "task_count": selected_task_count,
            "task_file": output_tasks.name,
            "task_file_sha256": tasks_digest.hexdigest(),
            "official_key_count": len(official_keys),
            "key_count": len(selected_parent_keys),
            "official_keys_not_used_count": len(missing_official_keys),
            "official_keys_not_used": missing_official_keys,
            "keys": sorted(selected_parent_keys),
        }
        key_bytes = canonical_json_bytes(key_payload)
        temporary_keys, keys_output = make_temporary_output(
            output_keys, overwrite=overwrite
        )
        with keys_output:
            keys_output.write(key_bytes)
            keys_output.flush()
            os.fsync(keys_output.fileno())

        temporary_tasks.replace(output_tasks)
        temporary_keys.replace(output_keys)
    finally:
        if temporary_tasks.exists():
            temporary_tasks.unlink()
        if temporary_keys is not None and temporary_keys.exists():
            temporary_keys.unlink()

    return {
        "source_tasks": source_task_count,
        "selected_tasks": selected_task_count,
        "selected_parent_keys": len(selected_parent_keys),
        "official_keys_not_used": len(official_keys - selected_parent_keys),
        "output_tasks": str(output_tasks),
        "output_tasks_sha256": tasks_digest.hexdigest(),
        "output_keys": str(output_keys),
        "output_keys_sha256": sha256_file(output_keys),
    }


def main() -> None:
    args = parse_args()
    summary = build_subset(
        source_tasks=args.source_tasks,
        official_keys_path=args.official_keys,
        output_tasks=args.output_tasks,
        output_keys=args.output_keys,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
