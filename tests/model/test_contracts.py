"""Tests for packed model input contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from diffusion_arc.model import (
    NVARC_CONTEXT_WIDTH,
    NVARC_HIDDEN_SIZE,
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_MAX_TOKENS,
    NVARC_PAD_TOKEN_ID,
    NVARC_VOCAB_SIZE,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_SHAPE,
    TARGET_IMAGE_WIDTH,
    NvarcBatchOutput,
    PackedTrainingBatch,
    validate_packed_source_sequences,
    validate_packed_training_batch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _make_training_batch(
    source_lengths: tuple[int, ...] = (8,),
    targets: tuple[tuple[int, int], ...] = ((0, 2), (0, 4), (0, 6)),
) -> PackedTrainingBatch:
    offsets = [0]
    for source_length in source_lengths:
        offsets.append(offsets[-1] + source_length)

    target_count = len(targets)
    return PackedTrainingBatch(
        input_ids=torch.arange(offsets[-1], dtype=torch.int64).remainder(
            NVARC_VOCAB_SIZE
        ),
        sequence_offsets=torch.tensor(offsets, dtype=torch.int64),
        target_sequence_indices=torch.tensor(
            [source_index for source_index, _ in targets], dtype=torch.int64
        ),
        target_prefix_ends=torch.tensor(
            [prefix_end for _, prefix_end in targets], dtype=torch.int64
        ),
        target_images=torch.zeros(
            (target_count, *TARGET_IMAGE_SHAPE), dtype=torch.float32
        ),
    )


def _replace(batch: PackedTrainingBatch, **changes: torch.Tensor) -> PackedTrainingBatch:
    values = {
        "input_ids": batch.input_ids,
        "sequence_offsets": batch.sequence_offsets,
        "target_sequence_indices": batch.target_sequence_indices,
        "target_prefix_ends": batch.target_prefix_ends,
        "target_images": batch.target_images,
    }
    values.update(changes)
    return PackedTrainingBatch(**values)


class PackedContractTests(unittest.TestCase):
    def test_nvarc_execution_constants_match_the_fixed_interface(self) -> None:
        self.assertEqual(NVARC_PAD_TOKEN_ID, 13)
        self.assertEqual(NVARC_HIDDEN_SIZE, 2560)
        self.assertEqual(NVARC_HIDDEN_STATE_INDICES, (9, 18, 27))
        self.assertEqual(NVARC_CONTEXT_WIDTH, 7680)

    def test_packed_source_validation_returns_individual_lengths(self) -> None:
        input_ids = torch.arange(9, dtype=torch.int64).remainder(NVARC_VOCAB_SIZE)
        sequence_offsets = torch.tensor([0, 2, 7, 9], dtype=torch.int64)

        source_lengths = validate_packed_source_sequences(
            input_ids=input_ids,
            sequence_offsets=sequence_offsets,
        )

        self.assertEqual(source_lengths, (2, 5, 2))

    def test_nvarc_batch_output_contract_is_frozen(self) -> None:
        output = NvarcBatchOutput(
            sequence_hidden_states=(torch.zeros((2, NVARC_CONTEXT_WIDTH)),),
        )

        with self.assertRaises(FrozenInstanceError):
            output.sequence_hidden_states = ()  # type: ignore[misc]

    def test_one_source_can_expose_multiple_targets(self) -> None:
        validate_packed_training_batch(_make_training_batch())

    def test_variable_sources_and_target_counts_are_valid(self) -> None:
        batch = _make_training_batch(
            source_lengths=(5, 8, 3),
            targets=((0, 1), (0, 4), (1, 2), (2, 1), (2, 2)),
        )

        validate_packed_training_batch(batch)

    def test_contracts_are_frozen(self) -> None:
        batch = _make_training_batch()

        with self.assertRaises(FrozenInstanceError):
            batch.input_ids = torch.zeros(1, dtype=torch.int64)  # type: ignore[misc]

    def test_invalid_offsets_are_rejected(self) -> None:
        batch = _make_training_batch()
        invalid_offsets = (
            torch.tensor([1, 8], dtype=torch.int64),
            torch.tensor([0, 7], dtype=torch.int64),
            torch.tensor([0, 4, 4, 8], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

        for offsets in invalid_offsets:
            with self.subTest(offsets=offsets.tolist()), self.assertRaises(ValueError):
                validate_packed_training_batch(_replace(batch, sequence_offsets=offsets))

    def test_empty_and_too_long_sources_are_rejected(self) -> None:
        batch = _make_training_batch()

        with self.assertRaises(ValueError):
            validate_packed_training_batch(
                _replace(
                    batch,
                    input_ids=torch.empty(0, dtype=torch.int64),
                    sequence_offsets=torch.tensor([0], dtype=torch.int64),
                )
            )

        with self.assertRaises(ValueError):
            validate_packed_training_batch(
                _replace(
                    batch,
                    input_ids=torch.zeros(NVARC_MAX_TOKENS + 1, dtype=torch.int64),
                    sequence_offsets=torch.tensor([0, NVARC_MAX_TOKENS + 1]),
                )
            )

    def test_invalid_token_ids_and_dtypes_are_rejected(self) -> None:
        batch = _make_training_batch()
        invalid_ids = (
            torch.tensor([0, -1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
            torch.tensor([0, NVARC_VOCAB_SIZE, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
        )

        for input_ids in invalid_ids:
            with self.subTest(input_ids=input_ids.tolist()), self.assertRaises(ValueError):
                validate_packed_training_batch(_replace(batch, input_ids=input_ids))

        with self.assertRaises(TypeError):
            validate_packed_training_batch(
                _replace(batch, input_ids=batch.input_ids.to(torch.int32))
            )
        with self.assertRaises(TypeError):
            validate_packed_training_batch(
                _replace(batch, sequence_offsets=batch.sequence_offsets.to(torch.int32))
            )
        with self.assertRaises(TypeError):
            validate_packed_training_batch(  # type: ignore[arg-type]
                _replace(batch, input_ids=[0, 1, 2])
            )

    def test_target_indices_must_be_ordered_in_range_and_cover_sources(self) -> None:
        batch = _make_training_batch(
            source_lengths=(6, 6),
            targets=((0, 2), (1, 2)),
        )
        invalid_indices = (
            torch.tensor([1, 0], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
            torch.tensor([0, 0], dtype=torch.int64),
        )

        for indices in invalid_indices:
            with self.subTest(indices=indices.tolist()), self.assertRaises(ValueError):
                validate_packed_training_batch(
                    _replace(batch, target_sequence_indices=indices)
                )

    def test_target_prefixes_must_be_ordered_and_not_beyond_source(self) -> None:
        batch = _make_training_batch(
            source_lengths=(8,),
            targets=((0, 2), (0, 4), (0, 8)),
        )
        validate_packed_training_batch(batch)

        invalid_prefixes = (
            torch.tensor([0, 4, 8], dtype=torch.int64),
            torch.tensor([2, 2, 8], dtype=torch.int64),
            torch.tensor([2, 8, 4], dtype=torch.int64),
            torch.tensor([2, 4, 9], dtype=torch.int64),
        )

        for prefix_ends in invalid_prefixes:
            with self.subTest(prefix_ends=prefix_ends.tolist()), self.assertRaises(
                ValueError
            ):
                validate_packed_training_batch(
                    _replace(batch, target_prefix_ends=prefix_ends)
                )

    def test_target_metadata_lengths_must_match_images(self) -> None:
        batch = _make_training_batch()

        with self.assertRaises(ValueError):
            validate_packed_training_batch(
                _replace(
                    batch,
                    target_prefix_ends=torch.tensor([2, 4], dtype=torch.int64),
                )
            )

        empty = _make_training_batch(source_lengths=(8,), targets=())
        with self.assertRaises(ValueError):
            validate_packed_training_batch(empty)

    def test_invalid_target_images_are_rejected(self) -> None:
        batch = _make_training_batch(source_lengths=(4,), targets=((0, 2),))
        invalid_images = (
            torch.zeros((1, 3, 479, 480), dtype=torch.float32),
            torch.zeros((1, *TARGET_IMAGE_SHAPE), dtype=torch.float64),
            torch.full((1, *TARGET_IMAGE_SHAPE), float("nan"), dtype=torch.float32),
            torch.full((1, *TARGET_IMAGE_SHAPE), 1.01, dtype=torch.float32),
            torch.full((1, *TARGET_IMAGE_SHAPE), -1.01, dtype=torch.float32),
        )

        for target_images in invalid_images:
            expected_error = TypeError if target_images.dtype != torch.float32 else ValueError
            with self.subTest(
                shape=tuple(target_images.shape), dtype=target_images.dtype
            ), self.assertRaises(expected_error):
                validate_packed_training_batch(
                    _replace(batch, target_images=target_images)
                )

    def test_contract_constants_match_arc_data(self) -> None:
        from diffusion_arc.arc_data.canvas import CANVAS_SIZE, CELL_SIZE
        from diffusion_arc.arc_data.tokenization import (
            NVARC_MAX_TOKENS as ARC_DATA_NVARC_MAX_TOKENS,
        )

        self.assertEqual(TARGET_IMAGE_HEIGHT, CANVAS_SIZE * CELL_SIZE)
        self.assertEqual(TARGET_IMAGE_WIDTH, CANVAS_SIZE * CELL_SIZE)
        self.assertEqual(NVARC_MAX_TOKENS, ARC_DATA_NVARC_MAX_TOKENS)

    def test_importing_model_does_not_import_heavy_model_dependencies(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import diffusion_arc.model; "
                    "assert not any(name == 'transformers' or "
                    "name.startswith('transformers.') for name in sys.modules); "
                    "assert not any(name.startswith('diffusion_arc._vendor') "
                    "or name.startswith('diffusion_arc._third_party') "
                    "or 'bfl_flux' in name for name in sys.modules)"
                ),
            ],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
