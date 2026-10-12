"""Benchmark selection preserves the current recipe and validates timing contracts."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diffusion_arc.training.benchmark_config import load_training_benchmark_config
from diffusion_arc.training.config import load_training_config


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"
BENCHMARK = ROOT / "configs/training/benchmarks/ddp_training.json"
B3 = ROOT / "configs/training/benchmarks/single_gpu_b3.json"


class DdpBenchmarkConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "benchmark.json"
        self.payload = json.loads(BENCHMARK.read_text())
        self.payload["training_recipe"] = str(RECIPE)

    def load(self, payload=None):
        self.path.write_text(json.dumps(self.payload if payload is None else payload))
        return load_training_benchmark_config(self.path)

    def test_token_balancing_variant_changes_only_source_assignment_from_b9(self):
        baseline = load_training_benchmark_config(BENCHMARK.parent / "b9_fsdp2_12k_no_forward_reshard.json")
        candidate = load_training_benchmark_config(BENCHMARK.parent / "b10_fsdp2_12k_token_balanced.json")
        expected = baseline.training.recipe_dict()
        expected["execution"]["rank_batch_assignment"] = "token_balanced_v1"
        self.assertEqual(candidate.training.recipe_dict(), expected)
        self.assertEqual(candidate.training.data, baseline.training.data)
        self.assertEqual((candidate.repetitions, candidate.warmup_steps, candidate.measured_steps), (1, 10, 50))
        self.payload["execution"]["rank_batch_assignment"] = "token_balanced_v1"
        self.assertEqual(self.load().training.rank_batch_assignment, "token_balanced_v1")

    def test_joint_packing_assignment_can_override_benchmark_execution(self):
        self.payload["execution"].update(rank_batch_assignment="joint_packing_v1",
                                           flux_packing_algorithm="legacy_bfd_v1")
        for strategy, gpu_counts in (("single_process", (1,)), ("ddp", (1, 2, 4, 8)),
                                     ("fsdp2", (1, 2, 4, 8))):
            for gpu_count in gpu_counts:
                with self.subTest(strategy=strategy, gpu_count=gpu_count):
                    self.payload["execution"].update(distributed_strategy=strategy, gpu_count=gpu_count)
                    config = self.load()
                    self.assertEqual(config.training.rank_batch_assignment, "joint_packing_v1")
                    self.assertFalse(config.precompute_packing)
                    self.assertFalse(config.training.data.shuffle)
        for field, value, error in (
            ("rng_scheme", "legacy_stream_v1", "joint_packing_v1 assignment requires flow.rng_scheme"),
            ("flux_packing_algorithm", "balanced_release_v1",
             "joint_packing_v1 assignment requires execution.flux_packing_algorithm"),
        ):
            payload = copy.deepcopy(self.payload)
            payload["execution"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                self.load(payload)

    def test_repository_config_uses_b3_data_with_the_unchanged_training_recipe(self):
        config = load_training_benchmark_config(BENCHMARK)
        original = load_training_config(RECIPE)
        b3 = load_training_config(B3)
        self.assertEqual(config.path, BENCHMARK)
        self.assertEqual(config.training.path, RECIPE)
        self.assertEqual(config.training.name, original.name)
        self.assertEqual(config.training.models, original.models)
        self.assertEqual(config.training.optimizer, original.optimizer)
        self.assertEqual(config.training.scheduler, original.scheduler)
        self.assertEqual(config.training.data.split_directory, b3.data.split_directory)
        self.assertEqual(config.training.recipe["execution"]["distributed_strategy"], "ddp")
        self.assertEqual(config.training.recipe["deployment"]["gpu_count"], 2)
        self.assertEqual(config.training.flux_packing.algorithm, "balanced_release_v1")
        self.assertEqual(config.training.recipe["flow"]["rng_scheme"], "global_step_target_v1")
        self.assertEqual(config.expected_source_count, 960)
        self.assertEqual(config.repetitions, 3)
        self.assertEqual((config.smoke_steps, config.warmup_steps, config.measured_steps), (3, 10, 50))
        self.assertEqual(config.benchmark_steps, 60)
        self.assertEqual(config.training.loop.max_optimizer_steps, 906)
        self.assertEqual(config.training.scheduler.warmup_steps, 50)
        self.assertEqual(config.training.recipe["loss"]["type"], "region_weighted_velocity_mse")
        self.assertEqual(config.training.recipe["flow"]["endpoint_targets_per_full_batch"], 4)
        self.assertIsNone(config.training.benchmark)

    def test_resolves_paths_at_the_declaring_file_without_requiring_artifacts(self):
        recipe_directory = self.path.parent / "recipes"
        recipe_directory.mkdir()
        recipe_path = recipe_directory / "recipe.json"
        recipe_path.write_text(RECIPE.read_text())
        self.payload["training_recipe"] = "recipes/recipe.json"
        self.payload["data"]["split_directory"] = "missing/benchmark/splits/train"
        config = self.load()
        self.assertEqual(config.training.path, recipe_path)
        self.assertEqual(config.training.data.split_directory,
                         self.path.parent / "missing/benchmark/splits/train")
        original = load_training_config(recipe_path)
        self.assertEqual(config.training.models, original.models)
        self.assertFalse(config.training.models.flux_checkpoint.exists())
        self.assertFalse(config.training.data.split_directory.exists())
        self.assertEqual(config.training.recipe["data"]["prepared_split_directory"],
                         original.data.split_directory)

    def test_b6_changes_only_pack_capacity_from_the_fsdp_baseline(self):
        directory = BENCHMARK.parent
        baseline = load_training_benchmark_config(directory / "fsdp2_training.json")
        candidate = load_training_benchmark_config(directory / "b6_fsdp2_16k.json")
        self.assertEqual(baseline.training.flux_packing.max_tokens, 12288)
        self.assertEqual(candidate.training.flux_packing.max_tokens, 16384)
        expected = baseline.training.recipe_dict()
        expected["execution"]["flux_max_joint_tokens"] = 16384
        self.assertEqual(candidate.training.recipe_dict(), expected)
        self.assertEqual(candidate.training.data, baseline.training.data)
        self.assertEqual((candidate.repetitions, candidate.warmup_steps, candidate.measured_steps), (3, 10, 50))
        self.assertEqual(candidate.training.loop.optimizer_batch_size, 48)
        self.assertEqual(load_training_config(RECIPE).flux_packing.max_tokens, 12288)

    def test_b7_pair_is_matched_except_forward_resharding(self):
        baseline = load_training_benchmark_config(BENCHMARK.parent / "fsdp2_training.json")
        expected = baseline.training.recipe_dict()
        expected["execution"]["flux_max_joint_tokens"] = 14336
        for filename, reshard in (("b7_fsdp2_14k_forward_reshard.json", True),
                                 ("b7_fsdp2_14k_no_forward_reshard.json", False)):
            with self.subTest(filename=filename):
                candidate = load_training_benchmark_config(BENCHMARK.parent / filename)
                expected["execution"]["fsdp_reshard_after_forward"] = reshard
                self.assertEqual(candidate.training.recipe_dict(), expected)
                self.assertIs(candidate.training.fsdp_reshard_after_forward, reshard)
                self.assertEqual(candidate.training.flux_packing.max_tokens, 14336)
                self.assertEqual(candidate.training.data, baseline.training.data)
                self.assertEqual((candidate.repetitions, candidate.smoke_steps, candidate.warmup_steps,
                                  candidate.measured_steps), (3, 3, 10, 50))
        self.assertTrue(baseline.training.fsdp_reshard_after_forward)
        self.assertTrue(load_training_benchmark_config(BENCHMARK.parent / "b6_fsdp2_16k.json")
                        .training.fsdp_reshard_after_forward)

    def test_fsdp_resharding_can_inherit_or_override_the_recipe(self):
        recipe = load_training_config(RECIPE).recipe_dict()
        recipe["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        recipe_path = self.path.parent / "recipe.json"
        self.payload["training_recipe"] = str(recipe_path)
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        for value in (False, True):
            with self.subTest(value=value):
                recipe["execution"]["fsdp_reshard_after_forward"] = value
                recipe_path.write_text(json.dumps(recipe))
                self.payload["execution"].pop("fsdp_reshard_after_forward", None)
                self.assertIs(self.load().training.fsdp_reshard_after_forward, value)
                self.payload["execution"]["fsdp_reshard_after_forward"] = not value
                self.assertIs(self.load().training.fsdp_reshard_after_forward, not value)
        for value in (None, 0, 1, "false"):
            self.payload["execution"]["fsdp_reshard_after_forward"] = value
            with self.subTest(value=value), self.assertRaisesRegex((TypeError, ValueError), "fsdp_reshard_after_forward"):
                self.load()
        self.payload["execution"].update(distributed_strategy="ddp", fsdp_reshard_after_forward=False)
        with self.assertRaisesRegex(ValueError, "requires distributed_strategy=fsdp2"):
            self.load()

    def test_b14_changes_only_fsdp_grouping_from_b13(self):
        baseline = load_training_benchmark_config(BENCHMARK.parent / "b13_fsdp2_14k_token_balanced.json")
        candidate = load_training_benchmark_config(BENCHMARK.parent / "b14_fsdp2_14k_grouped_blocks.json")
        expected = baseline.training.recipe_dict()
        expected["execution"]["fsdp_blocks_per_group"] = 5
        self.assertEqual(candidate.training.recipe_dict(), expected)
        self.assertEqual(candidate.training.data, baseline.training.data)
        self.assertEqual(candidate.training.fsdp_blocks_per_group, 5)
        self.assertEqual(baseline.training.fsdp_blocks_per_group, 1)
        self.assertEqual((candidate.repetitions, candidate.smoke_steps, candidate.warmup_steps,
                          candidate.measured_steps), (1, 3, 10, 50))

    def test_b15_precomputes_packing_without_changing_the_b13_training_recipe(self):
        baseline = load_training_benchmark_config(BENCHMARK.parent / "b13_fsdp2_14k_token_balanced.json")
        candidate = load_training_benchmark_config(BENCHMARK.parent / "b15_fsdp2_14k_precomputed_packing.json")
        self.assertFalse(baseline.precompute_packing)
        self.assertTrue(candidate.precompute_packing)
        self.assertEqual(candidate.training.recipe_dict(), baseline.training.recipe_dict())
        self.assertEqual(candidate.training.data, baseline.training.data)
        self.assertEqual((candidate.repetitions, candidate.smoke_steps, candidate.warmup_steps,
                          candidate.measured_steps), (1, 3, 10, 50))

    def test_precomputed_packing_validates_its_execution_contract(self):
        self.payload["precompute_packing"] = True
        self.payload["execution"].update(flux_packing_algorithm="legacy_bfd_v1",
                                           rank_batch_assignment="token_balanced_v1")
        self.assertTrue(self.load().precompute_packing)
        for strategy in ("ddp", "fsdp2"):
            self.payload["execution"]["distributed_strategy"] = strategy
            self.assertTrue(self.load().precompute_packing)
        for value in (None, 0, 1, "true"):
            payload = copy.deepcopy(self.payload)
            payload["precompute_packing"] = value
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "precompute_packing"):
                self.load(payload)
        for updates, error in (
            (dict(distributed_strategy="single_process", gpu_count=1), "ddp or fsdp2"),
            (dict(rng_scheme="legacy_stream_v1"), "global_step_target_v1"),
            (dict(rank_batch_assignment="contiguous_v1"), "token_balanced_v1"),
            (dict(rank_batch_assignment="joint_packing_v1"),
             "precompute_packing requires rank_batch_assignment=token_balanced_v1"),
            (dict(flux_packing_algorithm="balanced_release_v1"), "legacy_bfd_v1"),
        ):
            payload = copy.deepcopy(self.payload)
            payload["execution"].update(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, error):
                self.load(payload)
        self.payload["precompute_packing"] = False
        self.payload["execution"].update(distributed_strategy="ddp", rank_batch_assignment="contiguous_v1",
                                           flux_packing_algorithm="balanced_release_v1")
        self.assertFalse(self.load().precompute_packing)

    def test_benchmark_cli_passes_precomputed_packing_and_records_the_flag(self):
        from scripts.benchmark_training import main

        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "0"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            output = Path(directory) / "benchmark"
            self.assertEqual(main([
                "--config", str(BENCHMARK.parent / "b15_fsdp2_14k_precomputed_packing.json"),
                "--mode", "benchmark", "--output-dir", str(output),
            ]), 0)
        options = run.call_args.kwargs
        self.assertTrue(options["precompute_packing"])
        self.assertTrue(options["source_identity"]["benchmark_config"]["precompute_packing"])
        self.assertEqual(options["max_steps"], 60)
        self.assertEqual(options["measurement"].measured_steps, 50)

    def test_per_microbatch_experiment_is_benchmark_only_and_reaches_cli(self):
        from scripts.benchmark_training import main

        path = BENCHMARK.parent / "fsdp2_16k_per_microbatch_sync_precomputed_packing.json"
        candidate = load_training_benchmark_config(path)
        self.assertEqual(candidate.fsdp_gradient_sync, "per_microbatch")
        self.assertTrue(candidate.precompute_packing)
        self.assertTrue(candidate.training.fsdp_reshard_after_forward)
        self.assertEqual(candidate.training.fsdp_blocks_per_group, 1)
        self.assertEqual(candidate.training.flux_packing.max_tokens, 16384)
        self.assertEqual(candidate.training.loop.optimizer_batch_size, 48)
        self.assertEqual((candidate.repetitions, candidate.warmup_steps, candidate.measured_steps), (1, 10, 50))
        self.assertNotIn("fsdp_gradient_sync", candidate.training.recipe_dict()["execution"])
        self.assertEqual(load_training_benchmark_config(BENCHMARK.parent / "b6_fsdp2_16k.json").fsdp_gradient_sync,
                         "last_microbatch")
        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "0"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            self.assertEqual(main(["--config", str(path), "--mode", "benchmark",
                                   "--output-dir", str(Path(directory) / "run")]), 0)
        self.assertEqual(run.call_args.kwargs["fsdp_gradient_sync"], "per_microbatch")
        self.assertEqual(run.call_args.kwargs["source_identity"]["benchmark_config"]["fsdp_gradient_sync"],
                         "per_microbatch")

    def test_gradient_sync_control_rejects_invalid_values_and_non_fsdp(self):
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        for value in (None, True, 0, "off", "per_batch"):
            self.payload["fsdp_gradient_sync"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.load()
        self.payload["fsdp_gradient_sync"] = "per_microbatch"
        self.assertEqual(self.load().fsdp_gradient_sync, "per_microbatch")
        self.payload["execution"]["distributed_strategy"] = "ddp"
        with self.assertRaisesRegex(ValueError, "fsdp_gradient_sync requires"):
            self.load()

    def test_fsdp_grouping_can_inherit_or_override_the_recipe(self):
        recipe = load_training_config(RECIPE).recipe_dict()
        recipe["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1",
                                   fsdp_blocks_per_group=5)
        recipe["flow"]["rng_scheme"] = "global_step_target_v1"
        recipe_path = self.path.parent / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        self.payload["training_recipe"] = str(recipe_path)
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        self.assertEqual(self.load().training.fsdp_blocks_per_group, 5)
        self.payload["execution"]["fsdp_blocks_per_group"] = 1
        self.assertEqual(self.load().training.fsdp_blocks_per_group, 1)
        for value in (None, True, 0, -1, 5.0, "5"):
            self.payload["execution"]["fsdp_blocks_per_group"] = value
            with self.subTest(value=value), self.assertRaisesRegex((TypeError, ValueError), "fsdp_blocks_per_group"):
                self.load()
        self.payload["execution"].update(distributed_strategy="ddp", fsdp_blocks_per_group=5)
        with self.assertRaisesRegex(ValueError, "requires distributed_strategy=fsdp2"):
            self.load()

    def test_removed_gradient_sync_override_is_rejected(self):
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        for value in (False, True):
            with self.subTest(value=value):
                self.payload["execution"]["fsdp_sync_gradients_at_batch_end"] = value
                with self.assertRaisesRegex(ValueError, "fsdp_sync_gradients_at_batch_end"):
                    self.load()

    def test_pack_capacity_override_rejects_invalid_types_and_out_of_range_limits(self):
        for value in (None, True, 16384.0, "16384", 0, 9299, 1 << 31):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.payload["execution"]["flux_max_joint_tokens"] = value
                self.load()

    def test_rejects_wrong_config_types_unknown_missing_and_duplicate_fields(self):
        for change in (
            lambda p: p.update(config_type="training_run"),
            lambda p: p.update(schema_version=2),
            lambda p: p.update(schema_version=True),
            lambda p: p.update(extra=True),
            lambda p: p.pop("training_recipe"),
            lambda p: p.pop("repetitions"),
            lambda p: p.pop("execution"),
            lambda p: p["execution"].pop("rng_scheme"),
            lambda p: p["data"].update(num_workers=0),
            lambda p: p["data"].pop("expected_source_count"),
        ):
            payload = copy.deepcopy(self.payload)
            change(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.load(payload)
        for raw in ('{"schema_version": 1, "schema_version": 1}',
                    '{"data": {"view_name": "train", "view_name": "train"}}'):
            self.path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "duplicate config field"):
                load_training_benchmark_config(self.path)
        with self.assertRaisesRegex(TypeError, "JSON object"):
            self.load([])

    def test_rejects_invalid_field_types_and_values(self):
        cases = (
            (None, "training_recipe", ""), (None, "training_recipe", 5),
            (None, "smoke_steps", True), (None, "smoke_steps", 0),
            (None, "warmup_steps", False), (None, "warmup_steps", -1),
            (None, "measured_steps", 0), (None, "measured_steps", 50.0),
            (None, "repetitions", True), (None, "repetitions", 0), (None, "repetitions", 3.0),
            (None, "data", []), ("data", "expected_source_count", True),
            ("data", "expected_source_count", 0), ("data", "split_directory", None),
            ("data", "view_name", ""),
        )
        for section, field, value in cases:
            payload = copy.deepcopy(self.payload)
            (payload if section is None else payload[section])[field] = value
            with self.subTest(section=section, field=field, value=value):
                with self.assertRaises((ValueError, TypeError)):
                    self.load(payload)

    def test_timing_windows_cover_exactly_the_full_view_without_changing_the_recipe(self):
        self.payload["smoke_steps"] = 61
        with self.assertRaisesRegex(ValueError, "smoke_steps must not exceed"):
            self.load()
        self.payload["smoke_steps"] = 3
        for count in (959, 961):
            self.payload["data"]["expected_source_count"] = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "complete view"):
                self.load()
        self.payload.update(warmup_steps=0, measured_steps=907)
        self.payload["data"]["expected_source_count"] = 907 * 16
        with self.assertRaisesRegex(ValueError, "max_optimizer_steps"):
            self.load()
        self.payload.update(warmup_steps=0, measured_steps=60)
        self.payload["data"]["expected_source_count"] = 960
        config = self.load()
        self.assertEqual(config.benchmark_steps, 60)
        self.assertEqual(config.training.scheduler.warmup_steps, 50)

    def test_requires_a_fresh_training_run_recipe(self):
        self.payload["training_recipe"] = str(B3)
        baseline_objective = self.load()
        self.assertEqual(baseline_objective.training.recipe["loss"]["type"], "velocity_mse")
        self.payload["training_recipe"] = str(RECIPE)
        recipe = json.loads(RECIPE.read_text())
        recipe["models"]["resume_from"] = "checkpoints/prior-run"
        recipe_path = self.path.parent / "resume.json"
        recipe_path.write_text(json.dumps(recipe))
        self.payload["training_recipe"] = str(recipe_path)
        with self.assertRaisesRegex(ValueError, "resume_from must be null"):
            self.load()

    def test_loading_needs_no_tensor_or_model_runtime(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; sys.path.insert(0, 'src'); "
                "from diffusion_arc.training.benchmark_config import load_training_benchmark_config; "
                "config = load_training_benchmark_config('configs/training/benchmarks/ddp_training.json'); "
                "assert config.benchmark_steps == 60; "
                "assert not {'torch', 'transformers', 'diffusers'}.intersection(sys.modules)"
            )], cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
