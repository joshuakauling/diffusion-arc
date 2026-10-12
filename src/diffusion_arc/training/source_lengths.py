"""Read explicitly prepared source-prefix lengths without loading source tasks."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import ArcTrainingDataset


def source_lengths_path(dataset: ArcTrainingDataset) -> Path:
    """Locate the length metadata belonging to this prepared training view."""
    return dataset.split_directory / f"{dataset.view_name}.token_lengths.json"


def _validate_source_prefix_lengths(dataset, prefix_lengths, min_context_pairs_before_target):
    if type(min_context_pairs_before_target) is not int or min_context_pairs_before_target <= 0:
        raise ValueError("min_context_pairs_before_target must be a positive integer")
    if not isinstance(prefix_lengths, (list, tuple)) or len(prefix_lengths) != len(dataset.keys):
        raise ValueError("token-length metadata must contain one prefix list per source")
    validated = []
    for index, (key, lengths) in enumerate(zip(dataset.keys, prefix_lengths, strict=True)):
        eligible_count = len(key.augmentation.pair_indices) - min_context_pairs_before_target
        if (eligible_count <= 0 or not isinstance(lengths, (list, tuple))
                or len(lengths) != eligible_count):
            raise ValueError(f"source {index} prefix count differs from its eligible target count")
        previous = 0
        for length in lengths:
            if type(length) is not int or not previous < length <= dataset.max_tokens:
                raise ValueError(
                    f"source {index} prefix lengths must be positive, strictly increasing "
                    f"integers within max_tokens={dataset.max_tokens}"
                )
            previous = length
        validated.append(tuple(lengths))
    return tuple(validated)


def load_source_token_lengths(
    dataset: ArcTrainingDataset, *, min_context_pairs_before_target: int,
) -> tuple[tuple[int, ...], ...]:
    """Require matching prepared metadata; never tokenize or scan source tasks."""
    path = source_lengths_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing source token lengths: {path}; run "
            "scripts/data/prepare_training_token_lengths.py explicitly first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "data_fingerprint": dataset.fingerprint,
        "view_name": dataset.view_name,
        "source_keys": [str(key) for key in dataset.keys],
        "min_context_pairs_before_target": min_context_pairs_before_target,
    }
    if (not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or type(payload.get("min_context_pairs_before_target")) is not int
            or any(payload.get(name) != value for name, value in expected.items())):
        raise ValueError(f"Stale or mismatched source token-length metadata: {path}")
    return _validate_source_prefix_lengths(
        dataset, payload.get("source_prefix_token_lengths"), min_context_pairs_before_target,
    )


def write_source_token_lengths(
    dataset: ArcTrainingDataset,
    prefix_lengths: Sequence[Sequence[int]],
    *,
    min_context_pairs_before_target: int,
    provenance: dict,
) -> Path:
    """Save validated eligible prefix lengths during explicit CPU preparation."""
    lengths = _validate_source_prefix_lengths(
        dataset, prefix_lengths, min_context_pairs_before_target,
    )
    payload = {
        "schema_version": 1,
        "data_fingerprint": dataset.fingerprint,
        "view_name": dataset.view_name,
        "source_keys": [str(key) for key in dataset.keys],
        "min_context_pairs_before_target": min_context_pairs_before_target,
        "source_prefix_token_lengths": lengths,
        "provenance": provenance,
    }
    path = source_lengths_path(dataset)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


__all__ = ["source_lengths_path", "load_source_token_lengths", "write_source_token_lengths"]
