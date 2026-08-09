"""Strict loading for the initial, manifest-defined ARC training samples."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import serialize_arc_task, serialize_grid


MANIFEST_SCHEMA_VERSION = 1
_TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_CONSTRAINTS = {
    "test_examples_per_task",
    "min_train_examples",
    "max_train_examples",
    "max_prompt_characters",
    "max_grid_dimension",
}

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ArcDatasetSource:
    """Pinned identities for separate ARC challenge and solution files."""

    challenges_file: str
    challenges_sha256: str
    solutions_file: str
    solutions_sha256: str


@dataclass(frozen=True)
class ArcSampleSpec:
    """One selected ARC test example."""

    task_id: str
    test_index: int
    label: str


@dataclass(frozen=True)
class ArcDatasetManifest:
    """Validated configuration for nested initial dataset subsets."""

    schema_version: int
    name: str
    benchmark: str
    source_split: str
    source: ArcDatasetSource
    samples: tuple[ArcSampleSpec, ...]
    subset_prefix_lengths: tuple[tuple[str, int], ...]
    selection_constraints: tuple[tuple[str, int], ...]

    def subset_size(self, subset: str) -> int:
        for name, size in self.subset_prefix_lengths:
            if name == subset:
                return size
        known = ", ".join(name for name, _ in self.subset_prefix_lengths)
        raise KeyError(f"Unknown dataset subset {subset!r}; expected one of: {known}")

    def specs_for_subset(self, subset: str) -> tuple[ArcSampleSpec, ...]:
        return self.samples[: self.subset_size(subset)]


@dataclass(frozen=True)
class ArcTrainingSample:
    """Leakage-safe model input and target for one selected ARC example."""

    task_id: str
    test_index: int
    label: str
    prompt: str
    target_grid: Grid
    train_example_count: int
    test_example_count: int
    test_input_shape: tuple[int, int]
    target_shape: tuple[int, int]
    max_grid_dimension: int

    @property
    def prompt_characters(self) -> int:
        return len(self.prompt)


def _read_json(path: Path, *, name: str) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected {name} JSON file: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected local file for hashing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_string(value, name=name).lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal characters")
    return digest


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return value


def load_dataset_manifest(path: Path) -> ArcDatasetManifest:
    """Load and validate a versioned initial-dataset manifest."""

    payload = _require_mapping(_read_json(path, name="dataset manifest"), name="manifest")
    schema_version = _require_nonnegative_int(
        payload.get("schema_version"),
        name="manifest.schema_version",
    )
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest schema {schema_version}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )

    source_payload = _require_mapping(payload.get("source"), name="manifest.source")
    source = ArcDatasetSource(
        challenges_file=_require_string(
            source_payload.get("challenges_file"),
            name="manifest.source.challenges_file",
        ),
        challenges_sha256=_require_sha256(
            source_payload.get("challenges_sha256"),
            name="manifest.source.challenges_sha256",
        ),
        solutions_file=_require_string(
            source_payload.get("solutions_file"),
            name="manifest.source.solutions_file",
        ),
        solutions_sha256=_require_sha256(
            source_payload.get("solutions_sha256"),
            name="manifest.source.solutions_sha256",
        ),
    )

    sample_payloads = _require_sequence(payload.get("samples"), name="manifest.samples")
    if not sample_payloads:
        raise ValueError("manifest.samples must not be empty")
    samples: list[ArcSampleSpec] = []
    identities: set[tuple[str, int]] = set()
    for index, raw_sample in enumerate(sample_payloads):
        sample = _require_mapping(raw_sample, name=f"manifest.samples[{index}]")
        task_id = _require_string(
            sample.get("task_id"),
            name=f"manifest.samples[{index}].task_id",
        )
        if not _TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(
                f"manifest.samples[{index}].task_id must be eight lowercase hex characters"
            )
        test_index = _require_nonnegative_int(
            sample.get("test_index"),
            name=f"manifest.samples[{index}].test_index",
        )
        label = _require_string(
            sample.get("label"),
            name=f"manifest.samples[{index}].label",
        )
        identity = (task_id, test_index)
        if identity in identities:
            raise ValueError(f"Duplicate manifest sample {task_id}:{test_index}")
        identities.add(identity)
        samples.append(ArcSampleSpec(task_id=task_id, test_index=test_index, label=label))

    raw_subsets = _require_mapping(
        payload.get("subset_prefix_lengths"),
        name="manifest.subset_prefix_lengths",
    )
    if not raw_subsets:
        raise ValueError("manifest.subset_prefix_lengths must not be empty")
    subsets: list[tuple[str, int]] = []
    previous_size = 0
    for raw_name, raw_size in raw_subsets.items():
        name = _require_string(raw_name, name="manifest subset name")
        size = _require_nonnegative_int(
            raw_size,
            name=f"manifest.subset_prefix_lengths[{name!r}]",
        )
        if size <= previous_size:
            raise ValueError("dataset subset prefix lengths must be strictly increasing")
        if size > len(samples):
            raise ValueError(
                f"dataset subset {name!r} requests {size} of only {len(samples)} samples"
            )
        subsets.append((name, size))
        previous_size = size
    if previous_size != len(samples):
        raise ValueError("the largest dataset subset must include every manifest sample")

    raw_constraints = _require_mapping(
        payload.get("selection_constraints"),
        name="manifest.selection_constraints",
    )
    unknown_constraints = set(raw_constraints) - _SELECTION_CONSTRAINTS
    if unknown_constraints:
        names = ", ".join(sorted(unknown_constraints))
        raise ValueError(f"Unknown selection constraints: {names}")
    constraints = tuple(
        (
            _require_string(raw_name, name="selection constraint name"),
            _require_nonnegative_int(
                raw_value,
                name=f"manifest.selection_constraints[{raw_name!r}]",
            ),
        )
        for raw_name, raw_value in raw_constraints.items()
    )

    return ArcDatasetManifest(
        schema_version=schema_version,
        name=_require_string(payload.get("name"), name="manifest.name"),
        benchmark=_require_string(payload.get("benchmark"), name="manifest.benchmark"),
        source_split=_require_string(
            payload.get("source_split"),
            name="manifest.source_split",
        ),
        source=source,
        samples=tuple(samples),
        subset_prefix_lengths=tuple(subsets),
        selection_constraints=constraints,
    )


def _normalize_grid(grid: Any, *, name: str) -> Grid:
    serialize_grid(grid, name=name)
    return tuple(tuple(row) for row in grid)


def _verify_source(path: Path, *, expected_sha256: str, name: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )


def _resolve_sample(
    spec: ArcSampleSpec,
    *,
    challenges: Mapping[str, Any],
    solutions: Mapping[str, Any],
) -> ArcTrainingSample:
    if spec.task_id not in challenges:
        raise KeyError(f"Task {spec.task_id!r} is absent from the challenge source")
    task = _require_mapping(
        challenges[spec.task_id],
        name=f"challenge[{spec.task_id!r}]",
    )
    train = _require_sequence(
        task.get("train"),
        name=f"challenge[{spec.task_id!r}]['train']",
    )
    tests = _require_sequence(
        task.get("test"),
        name=f"challenge[{spec.task_id!r}]['test']",
    )
    if spec.test_index >= len(tests):
        raise IndexError(
            f"test index {spec.test_index} is outside task {spec.task_id!r} "
            f"with {len(tests)} test examples"
        )

    all_grids: list[Grid] = []
    for example_index, raw_example in enumerate(train):
        example = _require_mapping(
            raw_example,
            name=f"challenge[{spec.task_id!r}]['train'][{example_index}]",
        )
        if "input" not in example or "output" not in example:
            raise ValueError(
                f"challenge[{spec.task_id!r}]['train'][{example_index}] "
                "must contain input and output"
            )
        all_grids.append(
            _normalize_grid(
                example["input"],
                name=f"challenge[{spec.task_id!r}]['train'][{example_index}]['input']",
            )
        )
        all_grids.append(
            _normalize_grid(
                example["output"],
                name=f"challenge[{spec.task_id!r}]['train'][{example_index}]['output']",
            )
        )

    normalized_test_inputs: list[Grid] = []
    for example_index, raw_example in enumerate(tests):
        example = _require_mapping(
            raw_example,
            name=f"challenge[{spec.task_id!r}]['test'][{example_index}]",
        )
        if "input" not in example:
            raise ValueError(
                f"challenge[{spec.task_id!r}]['test'][{example_index}] "
                "must contain input"
            )
        if "output" in example:
            raise ValueError(
                f"challenge[{spec.task_id!r}]['test'][{example_index}] "
                "must not contain output; load targets from the solution source"
            )
        normalized = _normalize_grid(
            example["input"],
            name=f"challenge[{spec.task_id!r}]['test'][{example_index}]['input']",
        )
        normalized_test_inputs.append(normalized)
        all_grids.append(normalized)

    if spec.task_id not in solutions:
        raise KeyError(f"Task {spec.task_id!r} is absent from the solution source")
    raw_targets = _require_sequence(
        solutions[spec.task_id],
        name=f"solutions[{spec.task_id!r}]",
    )
    if len(raw_targets) != len(tests):
        raise ValueError(
            f"Task {spec.task_id!r} has {len(tests)} test inputs but "
            f"{len(raw_targets)} solutions"
        )
    targets = [
        _normalize_grid(
            target,
            name=f"solutions[{spec.task_id!r}][{target_index}]",
        )
        for target_index, target in enumerate(raw_targets)
    ]
    all_grids.extend(targets)

    prompt = serialize_arc_task(task, test_index=spec.test_index)
    target = targets[spec.test_index]
    test_input = normalized_test_inputs[spec.test_index]
    maximum_dimension = max(
        max(len(grid), len(grid[0]))
        for grid in all_grids
    )
    return ArcTrainingSample(
        task_id=spec.task_id,
        test_index=spec.test_index,
        label=spec.label,
        prompt=prompt,
        target_grid=target,
        train_example_count=len(train),
        test_example_count=len(tests),
        test_input_shape=(len(test_input), len(test_input[0])),
        target_shape=(len(target), len(target[0])),
        max_grid_dimension=maximum_dimension,
    )


def validate_selection_constraints(
    manifest: ArcDatasetManifest,
    samples: Sequence[ArcTrainingSample],
) -> None:
    """Fail if resolved samples no longer satisfy their curated size contract."""

    constraints = dict(manifest.selection_constraints)
    for sample in samples:
        checks = {
            "test_examples_per_task": sample.test_example_count,
            "max_prompt_characters": sample.prompt_characters,
            "max_grid_dimension": sample.max_grid_dimension,
        }
        exact_tests = constraints.get("test_examples_per_task")
        if exact_tests is not None and checks["test_examples_per_task"] != exact_tests:
            raise ValueError(
                f"Task {sample.task_id!r} has {sample.test_example_count} test examples; "
                f"expected {exact_tests}"
            )
        minimum_train = constraints.get("min_train_examples")
        if minimum_train is not None and sample.train_example_count < minimum_train:
            raise ValueError(
                f"Task {sample.task_id!r} has only {sample.train_example_count} "
                f"training examples; expected at least {minimum_train}"
            )
        maximum_train = constraints.get("max_train_examples")
        if maximum_train is not None and sample.train_example_count > maximum_train:
            raise ValueError(
                f"Task {sample.task_id!r} has {sample.train_example_count} "
                f"training examples; expected at most {maximum_train}"
            )
        maximum_prompt = constraints.get("max_prompt_characters")
        if maximum_prompt is not None and sample.prompt_characters > maximum_prompt:
            raise ValueError(
                f"Task {sample.task_id!r} prompt has {sample.prompt_characters} "
                f"characters; expected at most {maximum_prompt}"
            )
        maximum_grid = constraints.get("max_grid_dimension")
        if maximum_grid is not None and sample.max_grid_dimension > maximum_grid:
            raise ValueError(
                f"Task {sample.task_id!r} uses grid dimension "
                f"{sample.max_grid_dimension}; expected at most {maximum_grid}"
            )


def resolve_arc_dataset(
    manifest: ArcDatasetManifest,
    *,
    challenges_path: Path,
    solutions_path: Path,
    subset: str,
) -> tuple[ArcTrainingSample, ...]:
    """Resolve one strict manifest subset from separately stored sources."""

    _verify_source(
        challenges_path,
        expected_sha256=manifest.source.challenges_sha256,
        name="challenge source",
    )
    _verify_source(
        solutions_path,
        expected_sha256=manifest.source.solutions_sha256,
        name="solution source",
    )
    challenges = _require_mapping(
        _read_json(challenges_path, name="challenge source"),
        name="challenge source",
    )
    solutions = _require_mapping(
        _read_json(solutions_path, name="solution source"),
        name="solution source",
    )
    samples = tuple(
        _resolve_sample(spec, challenges=challenges, solutions=solutions)
        for spec in manifest.specs_for_subset(subset)
    )
    validate_selection_constraints(manifest, samples)
    return samples


def load_arc_dataset(
    manifest_path: Path,
    *,
    challenges_path: Path,
    solutions_path: Path,
    subset: str,
) -> tuple[ArcTrainingSample, ...]:
    """Load a manifest and resolve one named subset."""

    manifest = load_dataset_manifest(manifest_path)
    return resolve_arc_dataset(
        manifest,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        subset=subset,
    )


def summarize_samples(samples: Sequence[ArcTrainingSample]) -> dict[str, Any]:
    """Build a JSON-serializable audit summary."""

    if not samples:
        raise ValueError("Cannot summarize an empty dataset")
    prompt_lengths = [sample.prompt_characters for sample in samples]
    train_counts = [sample.train_example_count for sample in samples]
    return {
        "sample_count": len(samples),
        "task_count": len({sample.task_id for sample in samples}),
        "prompt_characters": {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
        },
        "train_examples": {
            "min": min(train_counts),
            "max": max(train_counts),
        },
        "max_grid_dimension": max(sample.max_grid_dimension for sample in samples),
        "samples": [
            {
                "task_id": sample.task_id,
                "test_index": sample.test_index,
                "label": sample.label,
                "train_examples": sample.train_example_count,
                "test_examples": sample.test_example_count,
                "prompt_characters": sample.prompt_characters,
                "test_input_shape": list(sample.test_input_shape),
                "target_shape": list(sample.target_shape),
                "max_grid_dimension": sample.max_grid_dimension,
            }
            for sample in samples
        ],
    }


__all__ = [
    "ArcDatasetManifest",
    "ArcDatasetSource",
    "ArcSampleSpec",
    "ArcTrainingSample",
    "load_arc_dataset",
    "load_dataset_manifest",
    "resolve_arc_dataset",
    "sha256_file",
    "summarize_samples",
    "validate_selection_constraints",
]
