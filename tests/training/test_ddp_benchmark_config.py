"""Benchmark selection preserves the current recipe and validates timing contracts."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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
