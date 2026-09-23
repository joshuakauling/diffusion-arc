"""Global batch boundaries and disjoint rank ownership before tokenization."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from diffusion_arc.training.config import load_training_config
from diffusion_arc.training.data import iter_optimizer_batches
from diffusion_arc.training.distributed_data import (
    DistributedBatch,
    build_distributed_dataloader,
    plan_distributed_batches,
)


class DistributedBatchPlanningTests(unittest.TestCase):
    def test_preserves_global_batches_and_covers_sources_and_targets_once(self):
        # A full six-source batch followed by a partial five-source batch.
        counts = (1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1)
        samples = tuple(SimpleNamespace(key=i, target_grids=(None,) * count)
                        for i, count in enumerate(counts))
        reference = tuple(iter_optimizer_batches(
            samples, optimizer_batch_size=8, min_context_pairs_before_target=1,
        ))
        for world_size in (1, 2, 4):
            with self.subTest(world_size=world_size):
                ranks = tuple(plan_distributed_batches(
                    counts, optimizer_batch_size=8, rank=rank, world_size=world_size,
                ) for rank in range(world_size))
                self.assertEqual({len(plans) for plans in ranks}, {len(reference)})
                all_sources = []
                for step, batch in enumerate(reference):
                    source_indices = tuple(sample.key for sample in batch)
                    global_count = sum(counts[index] for index in source_indices)
                    targets_by_source = {}
                    offset = 0
                    for source in source_indices:
                        targets_by_source[source] = tuple(range(offset, offset + counts[source]))
                        offset += counts[source]
                    actual_sources = tuple(index for plans in ranks
                                           for index in plans[step].source_indices)
                    actual_targets = tuple(index for plans in ranks
                                           for index in plans[step].target_indices)
                    self.assertEqual(actual_sources, source_indices)
                    self.assertEqual(actual_targets, tuple(range(global_count)))
                    for plans in ranks:
                        plan = plans[step]
                        self.assertEqual(plan.step, step)
                        self.assertEqual(plan.global_target_count, global_count)
                        self.assertTrue(plan.source_indices)
                        self.assertEqual(plan.target_indices, tuple(
                            target for source in plan.source_indices
                            for target in targets_by_source[source]
                        ))
                    all_sources.extend(actual_sources)
                self.assertEqual(all_sources, list(range(len(counts))))
                self.assertEqual(ranks[0][-1].global_target_count, 6)

    def test_assigns_remainder_sources_to_earlier_ranks(self):
        counts = (1, 2, 3, 1, 1)
        plans = tuple(plan_distributed_batches(
            counts, optimizer_batch_size=8, rank=rank, world_size=2,
        )[0] for rank in range(2))
        self.assertEqual(plans, (
            DistributedBatch(0, (0, 1, 2), tuple(range(6)), 8),
            DistributedBatch(0, (3, 4), (6, 7), 8),
        ))
        with self.assertRaises(FrozenInstanceError):
            plans[0].step = 1

    def test_rejects_overshoot_and_insufficient_full_or_final_batches(self):
        for counts, budget, world_size, message in (
            ((2, 3), 4, 1, "would exceed.*2 pending.*3 eligible"),
            ((5,), 4, 1, "would exceed"),
            ((2, 2), 4, 4, "global batch 0 has 2 sources"),
            ((1,) * 7, 4, 4, "global batch 1 has 3 sources"),
        ):
            for rank in range(world_size):
                with self.subTest(counts=counts, rank=rank), self.assertRaisesRegex(ValueError, message):
                    plan_distributed_batches(
                        counts, optimizer_batch_size=budget, rank=rank, world_size=world_size,
                    )

    def test_rejects_empty_or_invalid_counts_and_rank_configuration(self):
        for counts, error in (((), ValueError), ((0,), ValueError), ((-1,), ValueError),
                              ((True,), TypeError), ((1.0,), TypeError)):
            with self.subTest(counts=counts), self.assertRaisesRegex(error, "target_counts"):
                plan_distributed_batches(counts, optimizer_batch_size=4, rank=0, world_size=1)
        for field, value, error in (
            ("optimizer_batch_size", 0, ValueError), ("optimizer_batch_size", True, TypeError),
            ("world_size", 0, ValueError), ("world_size", 1.5, TypeError),
            ("rank", -1, ValueError), ("rank", 1, ValueError), ("rank", False, TypeError),
        ):
            kwargs = dict(optimizer_batch_size=4, rank=0, world_size=1)
            kwargs[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(error, field):
                plan_distributed_batches((1,), **kwargs)


class RecordingDataset(Dataset):
    def __init__(self, source_count=10, pair_count=5):
        self.keys = tuple(SimpleNamespace(
            task_id=f"task-{index}",
            augmentation=SimpleNamespace(pair_indices=tuple(range(pair_count))),
        ) for index in range(source_count))
        self.visited = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self.visited.append(index)
        return f"sample-{index}"


class DistributedDataLoaderTests(unittest.TestCase):
    def setUp(self):
        base = load_training_config(
            Path(__file__).resolve().parents[2] / "configs/training/arc_agi_1_epoch.json"
        )
        recipe = base.recipe_dict()
        recipe["data"]["expected_task_count"] = 10
        self.config = replace(
            base, recipe=recipe, data=replace(base.data, num_workers=0),
            loop=replace(base.loop, optimizer_batch_size=18),
        )

    def test_loads_only_owned_sources_in_step_order_without_consuming_global_rng(self):
        all_visited = []
        for rank in range(2):
            dataset = RecordingDataset()
            before = torch.random.get_rng_state().clone()
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                loader, plans = build_distributed_dataloader(
                    self.config, tokenizer=object(), rank=rank, world_size=2,
                )
                self.assertEqual(dataset.visited, [])
                result = tuple(loader)
            expected = ([0, 1, 2, 6, 7], [3, 4, 5, 8, 9])[rank]
            self.assertIs(loader.dataset, dataset)
            self.assertEqual(dataset.visited, expected)
            self.assertEqual(result, tuple(f"sample-{index}" for index in expected))
            self.assertEqual(tuple(index for plan in plans for index in plan.source_indices), tuple(expected))
            self.assertEqual([plan.global_target_count for plan in plans], [18, 12])
            torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
            all_visited.extend(dataset.visited)
        self.assertEqual(sorted(all_visited), list(range(10)))

    def test_rejects_recipe_mismatches_before_loading_any_sample(self):
        duplicate_tasks = RecordingDataset()
        duplicate_tasks.keys[1].task_id = duplicate_tasks.keys[0].task_id
        for dataset, message in (
            (RecordingDataset(source_count=9), "exactly 10 distinct tasks"),
            (duplicate_tasks, "exactly 10 distinct tasks"),
            (RecordingDataset(pair_count=4), "pair counts"),
        ):
            with self.subTest(message=message), patch(
                "diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset,
            ), self.assertRaisesRegex(ValueError, message):
                build_distributed_dataloader(self.config, tokenizer=object(), rank=0, world_size=2)
            self.assertEqual(dataset.visited, [])

    def test_resume_keeps_original_plan_identity_and_does_not_load_completed_sources(self):
        for world_size in (1, 2):
            for rank in range(world_size):
                dataset = RecordingDataset()
                with self.subTest(world_size=world_size, rank=rank), patch(
                    "diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset,
                ):
                    _, original = build_distributed_dataloader(
                        self.config, tokenizer=object(), rank=rank, world_size=world_size,
                    )
                    loader, resumed = build_distributed_dataloader(
                        self.config, tokenizer=object(), rank=rank, world_size=world_size, start_step=1,
                    )
                    self.assertEqual(resumed, original[1:])
                    self.assertEqual(dataset.visited, [])
                    tuple(loader)
                    self.assertEqual(tuple(dataset.visited), original[1].source_indices)
                    complete, plans = build_distributed_dataloader(
                        self.config, tokenizer=object(), rank=rank, world_size=world_size, start_step=2,
                    )
                    self.assertEqual(plans, ())
                    self.assertEqual(tuple(complete), ())

    def test_rejects_invalid_resume_cursor_before_reading_any_source(self):
        for settings in (dict(start_step=-1), dict(start_step=True), dict(start_step=1.5),
                         dict(start_step=3), dict(start_step=1, diagnostic_batch=2)):
            dataset = RecordingDataset()
            with self.subTest(settings=settings), patch(
                "diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset,
            ), self.assertRaisesRegex(ValueError, "start_step"):
                build_distributed_dataloader(self.config, tokenizer=object(), rank=0, world_size=2, **settings)
            self.assertEqual(dataset.visited, [])

    def test_diagnostic_repeats_only_selected_sources_with_original_step_identity(self):
        for rank, expected in ((0, [6, 7]), (1, [8, 9])):
            dataset = RecordingDataset()
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                loader, plans = build_distributed_dataloader(
                    self.config, tokenizer=object(), rank=rank, world_size=2,
                    diagnostic_batch=2, diagnostic_repeats=3,
                )
                tuple(loader)
            self.assertEqual(dataset.visited, expected * 3)
            self.assertEqual([p.step for p in plans], [1, 1, 1])
        for batch in (0, 3, True):
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=RecordingDataset()), \
                    self.assertRaisesRegex(ValueError, "one-based global batch"):
                build_distributed_dataloader(self.config, tokenizer=object(), rank=0, world_size=2,
                                             diagnostic_batch=batch)

    def test_rejects_shuffle_and_missing_recipe(self):
        for config, message in (
            (replace(self.config, data=replace(self.config.data, shuffle=True)), "shuffle=False"),
            (replace(self.config, config_type="baseline", recipe=None), "training_run recipe"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2)


if __name__ == "__main__":
    unittest.main()
