"""Global batch boundaries and disjoint rank ownership before tokenization."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from diffusion_arc.training.config import load_training_config, parse_training_config
from diffusion_arc.training.data import iter_optimizer_batches
from diffusion_arc.training.distributed_data import (
    RankBatchPlan,
    build_distributed_dataloader,
    plan_distributed_batches,
)
from diffusion_arc.training.source_lengths import write_source_token_lengths


class DistributedBatchPlanningTests(unittest.TestCase):
    def test_token_balancing_keeps_batches_whole_sources_and_original_target_ids(self):
        counts = (3,) * 16 + (2,) * 9
        costs = (12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000,
                 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000) + (1500,) * 9
        for world_size in (1, 2, 4, 8):
            with self.subTest(world_size=world_size):
                balanced = [plan_distributed_batches(counts, optimizer_batch_size=48,
                    rank=rank, world_size=world_size, source_joint_tokens=costs) for rank in range(world_size)]
                contiguous = [plan_distributed_batches(counts, optimizer_batch_size=48,
                    rank=rank, world_size=world_size) for rank in range(world_size)]
                for step, (start, end) in enumerate(((0, 16), (16, 25))):
                    self.assertEqual(sorted(source for plans in balanced for source in plans[step].source_indices), list(range(start, end)))
                    self.assertEqual(sorted(target for plans in balanced for target in plans[step].target_indices), list(range(sum(counts[start:end]))))
                    for plans in balanced:
                        plan = plans[step]
                        self.assertTrue(plan.source_indices)
                        self.assertEqual(plan.epoch_batch_index, step)
                        self.assertEqual(plan.source_indices, tuple(sorted(plan.source_indices)))
                        self.assertEqual(plan.target_indices, tuple(target for source in plan.source_indices
                            for target in range(sum(counts[start:source]), sum(counts[start:source + 1]))))
                    if step == 0:
                        heavier = lambda plans: max(sum(costs[index] for index in plan[step].source_indices) for plan in plans)
                        self.assertLessEqual(heavier(balanced), heavier(contiguous))

    def test_token_balancing_can_assign_unequal_source_counts_and_is_deterministic(self):
        kwargs = dict(optimizer_batch_size=12, world_size=2, source_joint_tokens=(12000, 1000, 1000, 1000))
        first = plan_distributed_batches((3,) * 4, rank=0, **kwargs)
        second = plan_distributed_batches((3,) * 4, rank=1, **kwargs)
        self.assertEqual(first, (RankBatchPlan(0, (0,), (0, 1, 2), 12),))
        self.assertEqual(second, (RankBatchPlan(0, (1, 2, 3), tuple(range(3, 12)), 12),))
        self.assertEqual(plan_distributed_batches((3,) * 4, rank=0, **kwargs), first)
        for costs in ((1,), (1, 0, 2, 3), (1, True, 2, 3), (1, 1.5, 2, 3)):
            with self.subTest(costs=costs), self.assertRaisesRegex(ValueError, "source_joint_tokens"):
                plan_distributed_batches((3,) * 4, optimizer_batch_size=12, rank=0, world_size=2, source_joint_tokens=costs)

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
                        self.assertEqual(plan.epoch_batch_index, step)
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
            RankBatchPlan(0, (0, 1, 2), tuple(range(6)), 8),
            RankBatchPlan(0, (3, 4), (6, 7), 8),
        ))
        with self.assertRaises(FrozenInstanceError):
            plans[0].epoch_batch_index = 1

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
        recipe.update(validation=None, logging=None, checkpointing=None)
        recipe["data"].update(num_workers=0, prefetch_factor=None)
        recipe["loop"]["optimizer_batch_size"] = 18
        recipe["scheduler"].update(warmup_steps=2, first_update_warmup_fraction=0.5)
        self.config = parse_training_config(recipe, config_path=base.path)

    def test_precomputed_packs_select_only_owned_sources_and_reuse_replayed_batch(self):
        recipe = self.config.recipe_dict()
        recipe["execution"].update(rank_batch_assignment="token_balanced_v1",
                                   flux_packing_algorithm="legacy_bfd_v1")
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        config = parse_training_config(recipe, config_path=self.config.path)
        with tempfile.TemporaryDirectory() as directory:
            prefixes = ((5000, 6000, 7000),) + ((10, 20, 30),) * 9
            ranks, metadata = [], []
            for rank in range(2):
                dataset = RecordingDataset()
                dataset.split_directory = Path(directory)
                dataset.view_name, dataset.fingerprint, dataset.max_tokens = "train", "prepared", 8400
                write_source_token_lengths(dataset, prefixes, min_context_pairs_before_target=2, provenance={})
                with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                    before = torch.random.get_rng_state().clone()
                    loader, plans = build_distributed_dataloader(config, tokenizer=object(), rank=rank,
                        world_size=2, precompute_packing=True, max_batches=1)
                    torch.testing.assert_close(torch.random.get_rng_state(), before, atol=0, rtol=0)
                    self.assertEqual(dataset.visited, [])
                    self.assertEqual(len(plans), 1)
                    plan = plans[0]
                    self.assertIsNotNone(plan.flux_plan)
                    self.assertEqual(plan.flux_plan.target_prefix_tokens,
                        tuple(prefix for source in plan.source_indices for prefix in prefixes[source]))
                    self.assertEqual(sorted(index for pack in plan.flux_plan.microbatch_plans
                                            for index in pack.target_indices), list(range(len(plan.target_indices))))
                    tuple(loader)
                    self.assertEqual(tuple(dataset.visited), plan.source_indices)
                    ranks.append(plan)
                    metadata.append(dataset.rank_assignment_metadata["precomputed_packing"])
                    dataset.visited.clear()
                    from diffusion_arc.training import batch_packing
                    with patch.object(batch_packing, "plan_batch", wraps=batch_packing.plan_batch) as search:
                        repeated_loader, repeated = build_distributed_dataloader(config, tokenizer=object(),
                            rank=rank, world_size=2, precompute_packing=True,
                            diagnostic_batch_number=2, diagnostic_repeats=3)
                    self.assertEqual(search.call_count, 1)
                    self.assertEqual(repeated, (repeated[0],) * 3)
                    self.assertEqual(repeated[0].epoch_batch_index, 1)
                    tuple(repeated_loader)
                    self.assertEqual(tuple(dataset.visited), repeated[0].source_indices * 3)
            self.assertEqual(metadata[0], metadata[1])
            self.assertEqual(metadata[0]["scheduled_batches"], 1)
            self.assertEqual(sorted(source for plan in ranks for source in plan.source_indices), list(range(6)))
            self.assertEqual(sorted(target for plan in ranks for target in plan.target_indices), list(range(18)))

    def test_prepared_balancing_loads_only_owned_sources_and_keeps_resume_and_replay_ids(self):
        recipe = self.config.recipe_dict()
        recipe["execution"]["rank_batch_assignment"] = "token_balanced_v1"
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        config = parse_training_config(recipe, config_path=self.config.path)
        with tempfile.TemporaryDirectory() as directory:
            dataset = RecordingDataset()
            dataset.split_directory = Path(directory)
            dataset.view_name, dataset.fingerprint, dataset.max_tokens = "train", "prepared", 8400
            prefixes = ((5000, 6000, 7000),) + ((10, 20, 30),) * 9
            write_source_token_lengths(dataset, prefixes, min_context_pairs_before_target=2, provenance={})
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=dataset):
                loader, plans = build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2)
                self.assertEqual(dataset.visited, [])
                self.assertEqual(plans[0].source_indices, (0,))
                self.assertEqual(plans[0].target_indices, (0, 1, 2))
                tuple(loader)
                self.assertEqual(tuple(dataset.visited), tuple(index for plan in plans for index in plan.source_indices))
                dataset.visited.clear()
                resumed, remaining = build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2, start_step=1)
                self.assertEqual(remaining, plans[1:])
                tuple(resumed)
                self.assertEqual(tuple(dataset.visited), plans[1].source_indices)
                dataset.visited.clear()
                replay, repeated = build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2,
                    diagnostic_batch_number=2, diagnostic_repeats=3)
                self.assertEqual(repeated, (plans[1],) * 3)
                tuple(replay)
                self.assertEqual(tuple(dataset.visited), plans[1].source_indices * 3)
                self.assertEqual(dataset.rank_assignment_metadata["algorithm"], "token_balanced_v1")
                self.assertEqual(len(dataset.rank_assignment_metadata["source_token_lengths_sha256"]), 64)


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
                         dict(start_step=3), dict(start_step=1, diagnostic_batch_number=2)):
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
                    diagnostic_batch_number=2, diagnostic_repeats=3,
                )
                tuple(loader)
            self.assertEqual(dataset.visited, expected * 3)
            self.assertEqual([p.epoch_batch_index for p in plans], [1, 1, 1])
        for batch in (0, 3, True):
            with patch("diffusion_arc.training.distributed_data.ArcTrainingDataset", return_value=RecordingDataset()), \
                    self.assertRaisesRegex(ValueError, "one-based global batch"):
                build_distributed_dataloader(self.config, tokenizer=object(), rank=0, world_size=2,
                                             diagnostic_batch_number=batch)

    def test_rejects_shuffle_and_missing_recipe(self):
        for config, message in (
            (replace(self.config, data=replace(self.config.data, shuffle=True)), "shuffle=False"),
            (replace(self.config, config_type="baseline", recipe=None), "training_run recipe"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_distributed_dataloader(config, tokenizer=object(), rank=0, world_size=2)


if __name__ == "__main__":
    unittest.main()
