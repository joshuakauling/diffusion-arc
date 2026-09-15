"""Internal path resolution for the canonical ARC data layout."""

from __future__ import annotations

from pathlib import Path


def resolve_split_layout(
    split_directory: str | Path,
    *,
    dataset: str,
    source_split: str,
) -> tuple[Path, Path]:
    """Resolve one split directory and its configured raw-task directory."""

    split_root = Path(split_directory).resolve()
    if split_root.parent.name != "splits":
        raise ValueError(
            "split_directory must match "
            "<data_directory>/splits/<split_name>"
        )

    data_directory = split_root.parent.parent
    source_directory = data_directory / "raw" / dataset / source_split
    return split_root, source_directory
