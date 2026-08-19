from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from diffusion_arc.training import ArcTrainingDataset, TrainingCollator


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
        "palette": "arc_standard_v1",
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


if __name__ == "__main__":
    unittest.main()
