"""Tests for CPU collation of tokenized ARC training samples."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from diffusion_arc.arc_data.augmentation import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcGeometry,
)
from diffusion_arc.arc_data.canvas import render_grid_canvas
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model import PackedTrainingBatch, validate_packed_training_batch
from diffusion_arc.training import CollatedTrainingBatch, TrainingCollator


def _make_key(
    task_id: str,
    pair_indices: tuple[int, ...],
) -> ArcAugmentationKey:
    return ArcAugmentationKey(
        task_id=task_id,
        augmentation=ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=pair_indices,
        ),
    )


def _make_sample(
    *,
    task_id: str = "abc123ef",
    input_ids: tuple[int, ...] = (14, 11, 10, 0, 15, 14, 12, 10, 7, 15),
    prefix_token_ends: tuple[int, ...] = (2, 5, 8),
    pair_indices: tuple[int, ...] = (4, 1, 3, 2),
    target_values: tuple[int, ...] = (2, 7, 9),
) -> ArcTokenizedTrainingSample:
    return ArcTokenizedTrainingSample(
        key=_make_key(task_id, pair_indices),
        input_ids=input_ids,
        prefix_token_ends=prefix_token_ends,
        target_grids=tuple(((value,),) for value in target_values),
    )


class TrainingCollatorTests(unittest.TestCase):
    def test_selects_last_targets_and_truncates_at_final_prefix(self) -> None:
        sample = _make_sample()

        collated = TrainingCollator(max_targets_per_source=2)([sample])

        self.assertIsInstance(collated, CollatedTrainingBatch)
        self.assertIsInstance(collated.tensors, PackedTrainingBatch)
        torch.testing.assert_close(
            collated.tensors.input_ids,
            torch.tensor(sample.input_ids[:8], dtype=torch.int64),
        )
        torch.testing.assert_close(
            collated.tensors.sequence_offsets,
            torch.tensor([0, 8], dtype=torch.int64),
        )
        torch.testing.assert_close(
            collated.tensors.target_sequence_indices,
            torch.tensor([0, 0], dtype=torch.int64),
        )
        torch.testing.assert_close(
            collated.tensors.target_prefix_ends,
            torch.tensor([5, 8], dtype=torch.int64),
        )
        torch.testing.assert_close(
            collated.tensors.target_images[0],
            render_grid_canvas(sample.target_grids[1]),
        )
        self.assertEqual(collated.metadata.source_keys, (sample.key,))
        self.assertEqual(collated.metadata.target_pair_indices, (3, 2))
        validate_packed_training_batch(collated.tensors)

    def test_accepts_fewer_than_maximum_and_variable_source_counts(self) -> None:
        first = _make_sample()
        second = _make_sample(
            task_id="deadbeef",
            input_ids=(14, 11, 10, 4, 15, 12),
            prefix_token_ends=(4,),
            pair_indices=(8, 6),
            target_values=(5,),
        )

        collated = TrainingCollator(max_targets_per_source=2)((first, second))

        expected_ids = first.input_ids[:8] + second.input_ids[:4]
        torch.testing.assert_close(
            collated.tensors.input_ids,
            torch.tensor(expected_ids, dtype=torch.int64),
        )
        self.assertEqual(collated.tensors.sequence_offsets.tolist(), [0, 8, 12])
        self.assertEqual(
            collated.tensors.target_sequence_indices.tolist(),
            [0, 0, 1],
        )
        self.assertEqual(collated.tensors.target_prefix_ends.tolist(), [5, 8, 4])
        self.assertEqual(collated.tensors.target_images.shape, (3, 3, 480, 480))
        expected_images = torch.stack(
            (
                render_grid_canvas(first.target_grids[1]),
                render_grid_canvas(first.target_grids[2]),
                render_grid_canvas(second.target_grids[0]),
            )
        )
        torch.testing.assert_close(collated.tensors.target_images, expected_images)
        self.assertEqual(collated.metadata.source_keys, (first.key, second.key))
        self.assertEqual(collated.metadata.target_pair_indices, (3, 2, 6))
        validate_packed_training_batch(collated.tensors)

    def test_none_selects_all_available_targets(self) -> None:
        sample = _make_sample()

        collated = TrainingCollator(max_targets_per_source=None)([sample])

        self.assertEqual(collated.tensors.input_ids.tolist(), list(sample.input_ids[:8]))
        self.assertEqual(collated.tensors.target_prefix_ends.tolist(), [2, 5, 8])
        self.assertEqual(collated.tensors.target_sequence_indices.tolist(), [0, 0, 0])
        self.assertEqual(collated.metadata.target_pair_indices, (1, 3, 2))

    def test_works_as_a_pytorch_dataloader_collate_function(self) -> None:
        samples = [
            _make_sample(),
            _make_sample(
                task_id="deadbeef",
                input_ids=(14, 11, 10, 1, 2, 15, 12),
                prefix_token_ends=(2, 5),
                pair_indices=(2, 5, 7),
                target_values=(4, 8),
            ),
        ]
        loader = DataLoader(
            samples,
            batch_size=2,
            collate_fn=TrainingCollator(max_targets_per_source=2),
        )

        collated = next(iter(loader))

        self.assertIsInstance(collated, CollatedTrainingBatch)
        self.assertEqual(collated.tensors.sequence_offsets.tolist(), [0, 8, 13])

    def test_rejects_invalid_maximum_configuration(self) -> None:
        TrainingCollator(max_targets_per_source=None)
        for value, error_type in ((True, TypeError), (1.5, TypeError), (0, ValueError)):
            with self.subTest(value=value), self.assertRaises(error_type):
                TrainingCollator(  # type: ignore[arg-type]
                    max_targets_per_source=value
                )

    def test_rejects_empty_or_invalid_sample_collections(self) -> None:
        collator = TrainingCollator(max_targets_per_source=2)

        with self.assertRaises(ValueError):
            collator([])
        with self.assertRaises(TypeError):
            collator(iter((_make_sample(),)))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            collator([object()])  # type: ignore[list-item]

    def test_rejects_empty_or_misaligned_target_candidates(self) -> None:
        collator = TrainingCollator(max_targets_per_source=2)
        no_targets = _make_sample(
            prefix_token_ends=(),
            pair_indices=(4,),
            target_values=(),
        )
        mismatched_targets = ArcTokenizedTrainingSample(
            key=_make_key("abc123ef", (4, 1, 3)),
            input_ids=(14, 11, 10, 0, 15, 12),
            prefix_token_ends=(2, 4),
            target_grids=(((2,),),),
        )

        for sample in (no_targets, mismatched_targets):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                collator([sample])

    def test_rejects_pair_metadata_not_aligned_with_all_candidates(self) -> None:
        sample = _make_sample(pair_indices=(4, 1, 3))

        with self.assertRaises(ValueError):
            TrainingCollator(max_targets_per_source=2)([sample])

    def test_rejects_values_that_would_be_silently_cast(self) -> None:
        invalid_ids = _make_sample(
            input_ids=(14, 11, 1.5, 15),  # type: ignore[arg-type]
            prefix_token_ends=(1, 3),
            pair_indices=(4, 1, 3),
            target_values=(2, 7),
        )
        invalid_prefixes = _make_sample(
            prefix_token_ends=(1, True),  # type: ignore[arg-type]
            pair_indices=(4, 1, 3),
            target_values=(2, 7),
        )

        for sample in (invalid_ids, invalid_prefixes):
            with self.subTest(sample=sample), self.assertRaises(TypeError):
                TrainingCollator(max_targets_per_source=2)([sample])

    def test_rejects_invalid_token_ids_and_full_input_prefixes(self) -> None:
        invalid_high_id = _make_sample(
            input_ids=(14, 16, 10, 0, 15, 14, 12, 10, 7, 15),
        )
        invalid_negative_id = _make_sample(
            input_ids=(14, -1, 10, 0, 15, 14, 12, 10, 7, 15),
        )
        prefix_at_full_end = _make_sample(
            prefix_token_ends=(2, 10),
            pair_indices=(4, 1, 3),
            target_values=(2, 7),
        )
        unordered_prefixes = _make_sample(
            prefix_token_ends=(5, 2),
            pair_indices=(4, 1, 3),
            target_values=(2, 7),
        )

        for sample in (
            invalid_high_id,
            invalid_negative_id,
            prefix_at_full_end,
            unordered_prefixes,
        ):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                TrainingCollator(max_targets_per_source=2)([sample])

    def test_rejects_selected_source_prefixes_beyond_nvarc_limit(self) -> None:
        sample = _make_sample(
            input_ids=(0,) * 8194,
            prefix_token_ends=(8193,),
            pair_indices=(0, 1),
            target_values=(2,),
        )

        with self.assertRaisesRegex(ValueError, "exceeds 8192 tokens"):
            TrainingCollator(max_targets_per_source=1)([sample])


if __name__ == "__main__":
    unittest.main()
