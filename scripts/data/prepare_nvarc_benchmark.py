#!/usr/bin/env python3
"""Prepare a fixed, single-variant NVARC benchmark from the local archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import sys
import tarfile
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from diffusion_arc.arc_data import (
    ARC_AUGMENTATION_GENERATOR_VERSION,
    ARC_SPLIT_SCHEMA_VERSION,
    NVARC_MAX_TOKENS,
    TARGET_PALETTE_ID,
    generate_augmented_key_files,
)
from scripts.data.build_nvarc_arc_agi_1_training_only import (
    DERIVED_GROUP,
    load_official_keys,
    sha256_file,
    validate_source_record,
)


DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "data/nvarc_data"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data/benchmarks/single_gpu_b0"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_index(path: Path, official_keys: set[str]) -> list[dict]:
    """Accept only unique, training-only source records; never filter bad rows."""

    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = validate_source_record(json.loads(line), line_number=line_number)
            if (
                record["parent1_arc_agi_1_split"] != "training"
                or record["parent2_arc_agi_1_split"] != "training"
                or not {record["parent1_id"], record["parent2_id"]} <= official_keys
            ):
                raise ValueError(f"Task line {line_number} has non-training parents")
            task_id = record["task_key"]
            expected_path = f"tasks/nvarc_training/{record['parent1_id']}/{task_id}.json"
            if record["archive_path"] != expected_path:
                raise ValueError(f"Unexpected archive path for {task_id}")
            if task_id in records:
                raise ValueError(f"Duplicate task ID: {task_id}")
            pair_count = record["pair_count"]
            if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 5:
                raise ValueError(f"Task {task_id} must have at least five pairs")
            records[task_id] = record
    return [records[task_id] for task_id in sorted(records)]


def prepare_benchmark(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    source_count: int = 960,
    seed: int = 167,
) -> dict:
    """Build a new bundle atomically, preserving all existing output directories."""

    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count <= 0:
        raise ValueError("source_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark bundle: {output_directory}")
    archive_path = source_root / "nvarc-all-classified.tar.gz"
    index_path = source_root / f"{DERIVED_GROUP}_tasks.jsonl"
    official_keys_path = source_root / "arc_agi_1_training_keys.json"
    official_keys, _ = load_official_keys(official_keys_path)
    records = _load_index(index_path, official_keys)
    if source_count > len(records):
        raise ValueError(f"Requested {source_count} tasks from a pool of {len(records)}")
    # This RNG selects task membership/order only. The existing key generator
    # derives an independent augmentation RNG from the seed and each task ID.
    Random(seed).shuffle(records)
    selected = records[:source_count]
    remaining = {record["archive_path"]: record for record in selected}
    input_hashes = {
        path.name: sha256_file(path)
        for path in (archive_path, index_path, official_keys_path)
    }
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_directory.name}-", dir=output_directory.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        raw_directory = bundle / "raw" / DERIVED_GROUP / "training"
        converted_hashes = {}
        print(f"Reading {source_count} selected tasks from {archive_path.name}", flush=True)
        # Read members as bytes, never extract arbitrary archive paths or links.
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                record = remaining.get(member.name)
                if record is None:
                    continue
                if not member.isfile():
                    raise ValueError(f"Expected a regular task file: {member.name}")
                with archive.extractfile(member) as handle:
                    content = handle.read()
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise ValueError(f"Task checksum mismatch: {member.name}")
                pairs = json.loads(content)
                if not isinstance(pairs, list) or len(pairs) != record["pair_count"]:
                    raise ValueError(f"Task pair bank differs from its index: {member.name}")
                destination = raw_directory / f"{record['task_key']}.json"
                # These are synthetic labelled pair banks, not evaluation splits.
                _write_json(destination, {"train": pairs, "test": []})
                converted_hashes[record["task_key"]] = sha256_file(destination)
                del remaining[member.name]
                if not remaining:
                    break
        if remaining:
            raise FileNotFoundError(f"Selected task files missing from archive: {sorted(remaining)}")

        split = bundle / "splits/train"
        _write_json(split / "config.json", {
            "schema_version": ARC_SPLIT_SCHEMA_VERSION,
            "generator_version": ARC_AUGMENTATION_GENERATOR_VERSION,
            "source": {"dataset": DERIVED_GROUP, "split": "training"},
            "palette": TARGET_PALETTE_ID,
            "limits": {"max_pairs": 5, "max_cells": 9000, "max_tokens": NVARC_MAX_TOKENS},
            "task_splits": {"train": "train.json", "validation": "validation.json"},
            "views": {"train": {
                "task_split": "train", "output": "augmented_keys/train.json",
                "mode": "sampled", "seed": seed, "augmentations_per_task": 1, "exclude": [],
            }},
        })
        _write_json(split / "train.json", [record["task_key"] for record in selected])
        _write_json(split / "validation.json", [])
        # Existing generation validates the selected grids and samples five
        # pairs without size exclusions: 5 * 2 * 30 * 30 = 9000 cells.
        keys = generate_augmented_key_files(split)["train"]
        manifest = {
            "schema_version": 1,
            "purpose": "performance_benchmark_only",
            "dataset": DERIVED_GROUP,
            "eligible_task_count": len(records),
            "source_count": len(keys),
            "target_count": 3 * len(keys),
            "seed": seed,
            "selection": "sort task IDs, seeded shuffle, take first source_count",
            "augmentation": {
                "variants_per_task": 1, "pairs": 5, "geometry": "ID",
                "colors": "random_permutation", "generator_version": ARC_AUGMENTATION_GENERATOR_VERSION,
            },
            "command": sys.argv,
            "preparation_script_sha256": sha256_file(Path(__file__)),
            "python": sys.version,
            "inputs_sha256": input_hashes,
            "keys_sha256": sha256_file(split / "augmented_keys/train.json"),
            "sources": [{**record, "converted_sha256": converted_hashes[record["task_key"]]}
                        for record in selected],
        }
        _write_json(split / "manifest.json", manifest)
        if output_directory.exists():
            raise FileExistsError(f"Benchmark bundle appeared during preparation: {output_directory}")
        bundle.rename(output_directory)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--source-count", type=int, default=960)
    parser.add_argument("--seed", type=int, default=167)
    args = parser.parse_args(argv)
    manifest = prepare_benchmark(
        source_root=args.source_root, output_directory=args.output_dir,
        source_count=args.source_count, seed=args.seed,
    )
    print(f"Prepared {manifest['source_count']} sources / {manifest['target_count']} targets")
    print(f"Key SHA-256: {manifest['keys_sha256']}")
    print(f"Benchmark bundle: {args.output_dir}")


if __name__ == "__main__":
    main()
