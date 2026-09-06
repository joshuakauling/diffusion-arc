"""Validated fixed sequence-length buckets shared by execution and telemetry."""

from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceLengthBuckets:
    """Round lengths up to immutable boundaries, or keep exact lengths with None."""

    boundaries: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.boundaries is None:
            return
        if not isinstance(self.boundaries, tuple):
            raise TypeError("bucket boundaries must be an immutable tuple or None")
        if not self.boundaries:
            raise ValueError("bucket boundaries must not be empty")
        previous = 0
        for boundary in self.boundaries:
            if isinstance(boundary, bool) or not isinstance(boundary, int):
                raise TypeError("bucket boundaries must be integers")
            if boundary <= previous:
                raise ValueError("bucket boundaries must be positive and strictly increasing")
            previous = boundary

    def padded_length(self, length: int) -> int:
        """Select a boundary for an actual tensor length without truncation."""

        if self.boundaries is None:
            return length
        index = bisect_left(self.boundaries, length)
        if index == len(self.boundaries):
            raise ValueError(
                f"sequence length {length} exceeds the largest bucket {self.boundaries[-1]}"
            )
        return self.boundaries[index]
