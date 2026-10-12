#!/usr/bin/env python3
"""Prepare CPU prefix-length metadata for token-balanced rank assignment."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from transformers import AutoTokenizer

from diffusion_arc._constants import NVARC_TOKENIZER_MAX_TOKENS
from diffusion_arc.training.benchmark_config import load_training_benchmark_config
from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.data import ArcTrainingDataset
from diffusion_arc.training.source_lengths import write_source_token_lengths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_prefix_lengths_from_steps(dataset, steps_jsonl: Path, *, min_context_pairs_before_target: int):
    """Reuse a complete measured view, checking source identities before saving."""
    steps_jsonl = Path(steps_jsonl).resolve()
    manifest_path = steps_jsonl.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    if manifest.get("data_fingerprint") != dataset.fingerprint:
        raise ValueError("benchmark data fingerprint differs from the prepared view")
    training = manifest.get("effective_training")
    if not isinstance(training, dict) or not isinstance(training.get("loop"), dict):
        raise ValueError("benchmark manifest is missing its target selection")
    recorded_context = training["loop"].get("min_context_pairs_before_target")
    if type(recorded_context) is not int or recorded_context != min_context_pairs_before_target:
        raise ValueError("benchmark target selection differs from min_context_pairs_before_target")
    lengths_by_source = {}
    for line in steps_jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get("event") != "step"
                or not isinstance(row.get("ranks"), list)):
            raise ValueError("steps JSONL must contain benchmark step records")
        for rank in row["ranks"]:
            if not isinstance(rank, dict):
                raise ValueError("benchmark ranks must contain JSON objects")
            indices, keys, prefixes = (rank.get(name) for name in
                                      ("source_indices", "source_keys", "target_prefix_tokens"))
            if (not all(isinstance(value, list) for value in (indices, keys, prefixes))
                    or len(indices) != len(keys)):
                raise ValueError("benchmark source indices, keys or prefixes are malformed")
            offset = 0
            for index, key in zip(indices, keys, strict=True):
                if type(index) is not int or not 0 <= index < len(dataset.keys):
                    raise ValueError("benchmark source index lies outside the prepared view")
                if index in lengths_by_source:
                    raise ValueError(f"benchmark source {index} appears more than once")
                expected_key = dataset.keys[index]
                if key != str(expected_key):
                    raise ValueError(f"benchmark source {index} key differs from the prepared view")
                count = len(expected_key.augmentation.pair_indices) - min_context_pairs_before_target
                if count <= 0:
                    raise ValueError(f"source {index} has no eligible targets")
                lengths_by_source[index] = tuple(prefixes[offset:offset + count])
                offset += count
            if offset != len(prefixes) or rank.get("targets") != offset:
                raise ValueError("benchmark prefix count differs from its eligible target count")
    if set(lengths_by_source) != set(range(len(dataset.keys))):
        raise ValueError("benchmark steps must cover every prepared source exactly once")
    models = training.get("models")
    provenance = {
        "method": "recorded_benchmark_prefixes",
        "steps_jsonl": str(steps_jsonl),
        "steps_sha256": _sha256(steps_jsonl),
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": _sha256(manifest_path),
        "benchmark_source_identity": manifest.get("source_identity"),
        "benchmark_package_versions": manifest.get("package_versions"),
        "nvarc_checkpoint": models.get("nvarc_checkpoint") if isinstance(models, dict) else None,
    }
    return tuple(lengths_by_source[index] for index in range(len(dataset.keys))), provenance


def prepare_training_token_lengths(config_path: Path, *, steps_jsonl: Path | None = None) -> Path:
    """Create only length metadata, loading a cached tokenizer when needed."""
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = (load_training_benchmark_config(config_path).training
              if payload.get("config_type") == "ddp_benchmark"
              else load_training_config(config_path, expected_type="training_run"))
    tokenizer = None
    if steps_jsonl is None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.models.nvarc_checkpoint, local_files_only=True, trust_remote_code=False,
        )
        tokenizer.model_max_length = NVARC_TOKENIZER_MAX_TOKENS
    dataset = ArcTrainingDataset(config.data.split_directory, config.data.view_name, tokenizer=tokenizer)
    context = config.loop.min_context_pairs_before_target
    if steps_jsonl is not None:
        lengths, provenance = source_prefix_lengths_from_steps(
            dataset, steps_jsonl, min_context_pairs_before_target=context,
        )
    else:
        # Only this explicit preparation reads and tokenizes the full view.
        lengths = tuple(tuple(dataset[index].prefix_token_ends[context - 1:])
                        for index in range(len(dataset)))
        provenance = {
            "method": "cpu_tokenization",
            "nvarc_checkpoint": str(config.models.nvarc_checkpoint),
            "transformers_version": version("transformers"),
            "tokenizer_max_tokens": NVARC_TOKENIZER_MAX_TOKENS,
        }
    provenance.update(config_path=str(config_path), config_sha256=_sha256(config_path),
                      preparation_script_sha256=_sha256(Path(__file__)))
    return write_source_token_lengths(
        dataset, lengths, min_context_pairs_before_target=context, provenance=provenance,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps-jsonl", type=Path,
                        help="Reuse a complete benchmark view instead of tokenizing its sources.")
    args = parser.parse_args(argv)
    destination = prepare_training_token_lengths(args.config, steps_jsonl=args.steps_jsonl)
    print(f"Prepared source token lengths: {destination}")


if __name__ == "__main__":
    main()
