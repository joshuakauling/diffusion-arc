#!/usr/bin/env python3
"""Freeze an ARC-AGI-1 split from verified local tasks or the pinned archive."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import sys
import tarfile


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from diffusion_arc._constants import ARC_AGI_1_ARCHIVE_SHA256 as ARCHIVE_SHA256


ARCHIVE_URL = "https://codeload.github.com/fchollet/ARC-AGI/tar.gz/refs/tags/v1.0.2"
# Canonical filename -> file-SHA256 inventory from the pinned baseline archive.
TASK_FILES_SHA256 = "b9e4932b457de4bce75200fbc2772e17e8f8d2feeee58c9b62471c1fb38016a4"


def prepare(archive: Path, destination: Path, *, seed: int = 167) -> dict:
    """Copy only validation payloads; retain final-test identities in the manifest."""
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("archive is not the pinned ARC-AGI-1 v1.0.2 release")
    if destination.exists():
        raise FileExistsError(f"refusing to replace an existing split: {destination}")
    with tarfile.open(archive, "r:gz") as source:
        members = {}
        for member in source:
            path = PurePosixPath(member.name)
            if member.isfile() and path.suffix == ".json" and path.parts[-3:-1] == ("data", "evaluation"):
                if path.stem in members:
                    raise ValueError("duplicate evaluation task in archive")
                members[path.stem] = member
        if len(members) != 400:
            raise ValueError("the pinned evaluation inventory must contain 400 tasks")
        ordered = sorted(members)
        random.Random(seed).shuffle(ordered)
        validation, final_test = ordered[:100], ordered[100:]
        payloads = {key: source.extractfile(members[key]).read() for key in validation}
    return _write_bundle(destination, validation, final_test, payloads, seed=seed,
                         verification={"method": "archive_sha256", "sha256": archive_hash})


def prepare_directory(evaluation: Path, destination: Path, *, seed: int = 167) -> dict:
    """Verify all 400 original files against existing baseline evidence, without downloading."""
    if destination.exists():
        raise FileExistsError(f"refusing to replace an existing split: {destination}")
    paths = sorted(evaluation.glob("*.json"))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(paths) != 400 or digest != TASK_FILES_SHA256:
        raise ValueError("local evaluation files differ from the pinned ARC-AGI-1 baseline inventory")
    ordered = sorted(path.stem for path in paths)
    random.Random(seed).shuffle(ordered)
    validation, final_test = ordered[:100], ordered[100:]
    payloads = {key: (evaluation / f"{key}.json").read_bytes() for key in validation}
    return _write_bundle(destination, validation, final_test, payloads, seed=seed,
                         verification={"method": "task_files_sha256", "sha256": digest,
                                       "directory": str(evaluation.resolve())})


def _write_bundle(destination, validation, final_test, payloads, *, seed, verification):
    # The split is decided using task identities before parsing any validation output.
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()}
    output_count = sum(len(json.loads(value)["test"]) for value in payloads.values())
    sampling_count = sum(len(json.loads(payloads[key])["test"]) for key in validation[:32])
    manifest = {
        "schema_version": 1,
        "dataset": "ARC-AGI-1",
        "release": "v1.0.2",
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "source_verification": verification,
        "split_seed": seed,
        "selection": "Python Random(seed).shuffle(sorted(task_ids)); first 100 validation, first 32 sampling",
        "validation_task_ids": validation,
        "sampling_task_ids": validation[:32],
        "final_test_task_ids": final_test,
        "validation_output_count": output_count,
        "sampling_output_count": sampling_count,
        "task_sha256": hashes,
    }
    (destination / "tasks").mkdir(parents=True)
    for key, payload in payloads.items():
        (destination / "tasks" / f"{key}.json").write_bytes(payload)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--evaluation-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = (prepare(args.archive, args.output) if args.archive is not None
                else prepare_directory(args.evaluation_directory, args.output))
    print(json.dumps({key: manifest[key] for key in ("validation_output_count", "sampling_output_count")}, indent=2))


if __name__ == "__main__":
    main()
