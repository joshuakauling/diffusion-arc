"""Tests for data."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from diffusion_arc.arc_data.augmentation import ArcAugmentationKey, ArcAugmentationSpec, ArcGeometry
from diffusion_arc.arc_data.image.canvas import render_grid_canvas
from diffusion_arc.arc_data.text.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.contracts import NVARC_MAX_TOKENS, NVARC_VOCAB_SIZE, TARGET_IMAGE_SHAPE
from diffusion_arc.training.config import DataConfig, load_training_config
from diffusion_arc.training.data import (
    ArcTrainingDataset,
    CollatedTrainingBatch,
    PackedTrainingBatch,
    TrainingCollator,
    build_dataloader,
    iter_optimizer_batches,
    validate_packed_training_batch,
)


TASK_A_KEY = "task-a_ID_0123456789_0,1,2,3"


TASK_B_KEY = "task-b_R90_9876543210_2,0,3"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_nvarc_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        **{str(value): value for value in range(10)},
        "\n": 10,
        "user": 11,
        "assistant": 12,
        "<|endoftext|>": 13,
        "<|im_start|>": 14,
        "<|im_end|>": 15,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<|endoftext|>"))
    backend.pre_tokenizer = Split(
        Regex(r"(<\|im_start\|>|<\|im_end\|>|assistant|user|\n|[0-9])"),
        behavior="isolated",
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


def make_task_payload(pairs: tuple[tuple[int, int], ...]) -> dict[str, object]:
    examples = [
        {"input": [[input_value]], "output": [[output_value]]}
        for input_value, output_value in pairs
    ]
    return {"train": examples[:3], "test": examples[3:]}


def make_split_config(*, max_tokens: int = 256) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator_version": 1,
        "source": {"dataset": "arc-agi-2", "split": "training"},
        "palette": "greedy_rgb_maximin_v1",
        "limits": {
            "max_pairs": 4,
            "max_cells": 8,
            "max_tokens": max_tokens,
        },
        "task_splits": {"train": "train.json", "validation": "validation.json"},
        "views": {
            "train": {
                "task_split": "train",
                "output": "augmented_keys/train.json",
                "mode": "sampled",
                "seed": 167,
                "augmentations_per_task": 1,
                "exclude": [],
            }
        },
    }


def build_layout(
    root: Path,
    *,
    keys: object = (TASK_B_KEY, TASK_A_KEY),
    max_tokens: int = 256,
) -> tuple[Path, Path]:
    data_root = root / "data"
    split_root = data_root / "splits/run"
    source_root = data_root / "raw/arc-agi-2/training"

    write_json(split_root / "config.json", make_split_config(max_tokens=max_tokens))
    write_json(split_root / "train.json", ["task-a", "task-b"])
    write_json(split_root / "validation.json", [])
    write_json(split_root / "augmented_keys/train.json", keys)
    write_json(
        source_root / "task-a.json",
        make_task_payload(((0, 1), (2, 3), (4, 5), (6, 7))),
    )
    write_json(
        source_root / "task-b.json",
        make_task_payload(((1, 2), (3, 4), (5, 6), (7, 8))),
    )
    return split_root, data_root


class ArcTrainingDatasetTest(unittest.TestCase):
    def test_fingerprint_tracks_order_augmentation_and_all_source_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(Path(temporary_directory))
            tokenizer = make_nvarc_tokenizer()

            def dataset():
                return ArcTrainingDataset(split_root, "train", tokenizer)

            original = dataset()
            self.assertRegex(original.fingerprint, r"^[0-9a-f]{64}$")
            self.assertEqual(original.fingerprint, dataset().fingerprint)
            key_path = split_root / "augmented_keys/train.json"
            write_json(key_path, [TASK_A_KEY, TASK_B_KEY])
            self.assertNotEqual(original.fingerprint, dataset().fingerprint)
            write_json(key_path, [TASK_B_KEY.replace("R90", "ID"), TASK_A_KEY])
            self.assertNotEqual(original.fingerprint, dataset().fingerprint)
            write_json(key_path, [TASK_B_KEY, TASK_A_KEY])

            # Pair 1 of task-b is not selected, but still belongs to the source.
            source_path = data_root / "raw/arc-agi-2/training/task-b.json"
            original_sample = original[0]
            changed_payload = make_task_payload(((1, 2), (3, 9), (5, 6), (7, 8)))
            write_json(source_path, changed_payload)
            changed = dataset()
            self.assertEqual(original_sample, changed[0])
            self.assertNotEqual(original.fingerprint, changed.fingerprint)
            # Formatting and object-key order do not change task meaning.
            source_path.write_text(json.dumps(changed_payload, indent=4, sort_keys=True))
            self.assertEqual(changed.fingerprint, dataset().fingerprint)

            write_json(split_root / "config.json", make_split_config(max_tokens=512))
            self.assertNotEqual(changed.fingerprint, dataset().fingerprint)

    def test_preserves_key_order_and_materializes_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root, data_root = build_layout(root)
            dataset = ArcTrainingDataset(
                split_root,
                "train",
                make_nvarc_tokenizer(),
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                tuple(key.task_id for key in dataset.keys),
                ("task-b", "task-a"),
            )
            self.assertEqual(dataset.split_directory, split_root.resolve())
            self.assertEqual(
                dataset.source_directory,
                (data_root / "raw/arc-agi-2/training").resolve(),
            )
            self.assertEqual(dataset.max_tokens, 256)

            first = dataset[0]
            self.assertEqual(first, dataset[0])
            self.assertEqual(first.key, dataset.keys[0])
            self.assertEqual(len(first.prefix_token_ends), 2)
            self.assertEqual(first.target_grids, (((7,),), ((1,),)))

            second = dataset[1]
            self.assertEqual(len(second.prefix_token_ends), 3)
            self.assertEqual(second.target_grids, (((3,),), ((5,),), ((7,),)))

    def test_uses_configured_token_limit_at_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(
                Path(temporary_directory),
                max_tokens=1,
            )
            dataset = ArcTrainingDataset(
                split_root,
                "train",
                make_nvarc_tokenizer(),
            )

            with self.assertRaisesRegex(ValueError, "exceeding 1"):
                dataset[0]

    def test_reloads_the_referenced_task_for_each_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(Path(temporary_directory))
            dataset = ArcTrainingDataset(
                split_root,
                "train",
                make_nvarc_tokenizer(),
            )
            task_a_path = data_root / "raw/arc-agi-2/training/task-a.json"

            original = dataset[1]
            write_json(
                task_a_path,
                make_task_payload(((0, 1), (2, 9), (4, 8), (6, 7))),
            )
            reloaded = dataset[1]

            self.assertEqual(original.target_grids, (((3,),), ((5,),), ((7,),)))
            self.assertEqual(reloaded.target_grids, (((9,),), ((8,),), ((7,),)))

    def test_integrates_with_standard_dataloader_and_training_collator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(Path(temporary_directory))
            dataset = ArcTrainingDataset(
                split_root,
                "train",
                make_nvarc_tokenizer(),
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                collate_fn=TrainingCollator(max_targets_per_source=2),
            )

            collated = next(iter(loader))

            self.assertEqual(collated.metadata.source_keys, dataset.keys)
            self.assertEqual(collated.metadata.target_pair_indices, (0, 3, 2, 3))
            self.assertEqual(
                collated.tensors.target_sequence_indices.tolist(),
                [0, 0, 1, 1],
            )
            self.assertEqual(collated.tensors.sequence_offsets.shape, (3,))
            self.assertEqual(collated.tensors.target_images.shape, (4, 3, 480, 480))

    def test_rejects_unknown_view_and_invalid_split_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root, data_root = build_layout(root)
            tokenizer = make_nvarc_tokenizer()

            with self.assertRaisesRegex(ValueError, "unknown training view"):
                ArcTrainingDataset(split_root, "missing", tokenizer)

            invalid_config = make_split_config()
            invalid_config["schema_version"] = 2
            write_json(split_root / "config.json", invalid_config)
            with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                ArcTrainingDataset(split_root, "train", tokenizer)

    def test_rejects_split_directory_outside_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root = Path(temporary_directory) / "data/configurations/run"
            write_json(split_root / "config.json", make_split_config())

            with self.assertRaisesRegex(ValueError, "data_directory"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )

    def test_rejects_malformed_or_duplicate_key_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root, data_root = build_layout(root)
            key_path = split_root / "augmented_keys/train.json"
            tokenizer = make_nvarc_tokenizer()

            write_json(key_path, {"key": TASK_A_KEY})
            with self.assertRaisesRegex(TypeError, "JSON array"):
                ArcTrainingDataset(split_root, "train", tokenizer)

            write_json(key_path, [TASK_A_KEY, TASK_A_KEY])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                ArcTrainingDataset(split_root, "train", tokenizer)

    def test_rejects_keys_outside_the_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(
                Path(temporary_directory),
                keys=["task-c_ID_0123456789_0,1"],
            )

            with self.assertRaisesRegex(ValueError, "outside its 'train' task split"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )

    def test_rejects_missing_sources_and_invalid_pair_selections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root, data_root = build_layout(root, keys=[TASK_A_KEY])
            source_path = data_root / "raw/arc-agi-2/training/task-a.json"
            source_path.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "task-a"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            split_root, data_root = build_layout(
                Path(temporary_directory),
                keys=["task-a_ID_0123456789_0,99"],
            )

            with self.assertRaisesRegex(ValueError, "out of range"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )

    def test_rejects_one_pair_keys_and_overlapping_task_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_root, data_root = build_layout(
                root,
                keys=["task-a_ID_0123456789_0"],
            )

            with self.assertRaisesRegex(ValueError, "at least two pairs"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )

            write_json(split_root / "augmented_keys/train.json", [TASK_A_KEY])
            write_json(split_root / "validation.json", ["task-a"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                ArcTrainingDataset(
                    split_root,
                    "train",
                    make_nvarc_tokenizer(),
                )


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


class ResumedDataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.keys = (
            TASK_A_KEY,
            TASK_A_KEY.replace("ID", "R90"),
            TASK_A_KEY.replace("task-a", "task-b"),
            TASK_A_KEY.replace("ID", "R180"),
            TASK_A_KEY.replace("ID", "R270"),
        )
        split_root, _ = build_layout(Path(self.temporary_directory.name), keys=self.keys)
        base_config = load_training_config(
            Path(__file__).resolve().parents[2] / "configs/training/arc_agi_1_epoch.json"
        )
        self.config = replace(
            base_config,
            data=DataConfig(split_root, "train", shuffle=False, num_workers=0),
        )
        self.tokenizer = make_nvarc_tokenizer()

    def test_resumes_exact_suffix_and_final_tail_without_materializing_prefix(self):
        uninterrupted = tuple(build_dataloader(self.config, tokenizer=self.tokenizer))
        full_batches = tuple(iter_optimizer_batches(
            uninterrupted, optimizer_batch_size=6, min_context_pairs_before_target=1,
        ))
        visited = []
        original_getitem = ArcTrainingDataset.__getitem__

        def materialize(dataset, index):
            visited.append(index)
            return original_getitem(dataset, index)

        with patch.object(ArcTrainingDataset, "__getitem__", materialize):
            resumed = build_dataloader(self.config, tokenizer=self.tokenizer, start_index=2)
            remaining_batches = tuple(iter_optimizer_batches(
                resumed, optimizer_batch_size=6, min_context_pairs_before_target=1,
            ))

        self.assertIsInstance(resumed.dataset, ArcTrainingDataset)
        self.assertEqual(len(resumed.dataset.keys), len(self.keys))
        self.assertEqual(visited, [2, 3, 4])
        self.assertEqual(remaining_batches, full_batches[1:])
        self.assertEqual([len(batch) for batch in remaining_batches], [2, 1])
        exhausted = build_dataloader(
            self.config, tokenizer=self.tokenizer, start_index=len(self.keys),
        )
        self.assertEqual(tuple(exhausted), ())

    def test_rejects_invalid_cursors_and_shuffled_resume(self):
        for cursor, error in ((True, TypeError), (1.0, TypeError), (-1, ValueError),
                              (len(self.keys) + 1, ValueError)):
            with self.subTest(cursor=cursor), self.assertRaisesRegex(error, "start_index"):
                build_dataloader(self.config, tokenizer=self.tokenizer, start_index=cursor)
        shuffled = replace(self.config, data=replace(self.config.data, shuffle=True))
        with self.assertRaisesRegex(ValueError, "shuffle=False"):
            build_dataloader(shuffled, tokenizer=self.tokenizer, start_index=1)
        # The existing seeded shuffle path remains available at the beginning.
        first = tuple(build_dataloader(shuffled, tokenizer=self.tokenizer))
        second = tuple(build_dataloader(shuffled, tokenizer=self.tokenizer, start_index=0))
        self.assertEqual(first, second)


class OptimizerBatchTests(unittest.TestCase):
    def test_uses_actual_target_counts_and_keeps_final_partial(self):
        samples = (
            _make_sample(),
            _make_sample(task_id="short", prefix_token_ends=(2, 5),
                         pair_indices=(0, 1, 2), target_values=(1, 2)),
            _make_sample(task_id="last"),
        )
        batches = tuple(iter_optimizer_batches(
            samples, optimizer_batch_size=3, min_context_pairs_before_target=2,
        ))
        self.assertEqual(batches, (samples[:2], samples[2:]))

    def test_limit_never_requests_another_source(self):
        consumed = []

        def samples():
            for index in range(4):
                consumed.append(index)
                yield _make_sample(task_id=f"s{index}")
            self.fail("must not consume past the step limit")

        batches = tuple(iter_optimizer_batches(
            samples(), optimizer_batch_size=4, min_context_pairs_before_target=2,
            max_optimizer_steps=1,
        ))
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(len(batches), 1)

    def test_rejects_overshoot_instead_of_splitting_or_discarding(self):
        with self.assertRaisesRegex(ValueError, "would exceed.*1 pending.*2 eligible"):
            tuple(iter_optimizer_batches(
                (_make_sample(prefix_token_ends=(2, 5), pair_indices=(0, 1, 2),
                              target_values=(1, 2)), _make_sample()),
                optimizer_batch_size=2, min_context_pairs_before_target=2,
            ))

    def test_skips_sources_without_eligible_targets_without_a_phantom_batch(self):
        sample = _make_sample()
        self.assertEqual(tuple(iter_optimizer_batches(
            (sample,), optimizer_batch_size=2, min_context_pairs_before_target=4,
        )), ())

    def test_rejects_invalid_limits(self):
        for field in ("optimizer_batch_size", "min_context_pairs_before_target", "max_optimizer_steps"):
            for value in (0, -1, True, 1.5):
                kwargs = dict(optimizer_batch_size=3, min_context_pairs_before_target=1)
                kwargs[field] = value
                with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                    tuple(iter_optimizer_batches((), **kwargs))


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

    def test_rejects_empty_sample_collection(self) -> None:
        collator = TrainingCollator(max_targets_per_source=2)

        with self.assertRaises(ValueError):
            collator([])


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


class PackedTrainingContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
