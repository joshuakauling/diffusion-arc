"""Canonical config contracts for baseline workloads and finite training recipes."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from diffusion_arc.training.config import (
    CosineSchedulerConfig,
    LinearWarmupConstantConfig,
    TrainingConfig,
    load_training_config,
)


def _payload() -> dict:
    return {
        "schema_version": 10,
        "nvarc_max_tokens": 134400,
        "seed": 17,
        "device": "cuda:1",
        "models": {
            "nvarc_checkpoint": "models/nvarc",
            "flux_checkpoint": "/models/flux",
            "nvarc_dtype": "bfloat16",
            "vae_dtype": "float32",
            "flux_parameter_dtype": "float32",
            "flux_compute_dtype": "bfloat16",
        },
        "data": {
            "split_directory": "data/split",
            "view_name": "train",
            "shuffle": True,
            "num_workers": 0,
            "prefetch_factor": None,
        },
        "loop": {
            "max_optimizer_steps": 3,
            "optimizer_batch_size": 4,
            "min_context_pairs_before_target": 1,
            "vae_microbatch_size": 8,
        },
        "scheduler": None, "benchmark": None,
        "flux_packing": {"max_tokens": 12288},
        "optimizer": {
            "learning_rate": 1e-5,
            "beta1": 0.9,
            "beta2": 0.99,
            "epsilon": 1e-8,
            "weight_decay": 0.01,
        },
    }


def _write_config(directory: Path, payload: dict) -> Path:
    path = directory / "training.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class BaselineConfigTests(unittest.TestCase):
    def test_packing_spread_has_a_default_and_accepts_an_explicit_bound(self):
        payload = _payload()
        with tempfile.TemporaryDirectory() as directory:
            config = load_training_config(_write_config(Path(directory), payload))
            self.assertEqual(config.flux_packing.max_spread_tokens, 1000)
            payload["flux_packing"]["max_spread_tokens"] = 500
            config = load_training_config(_write_config(Path(directory), payload))
            self.assertEqual(config.flux_packing.max_spread_tokens, 500)
            for invalid in (True, -1, 1.5):
                payload["flux_packing"]["max_spread_tokens"] = invalid
                with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                    load_training_config(_write_config(Path(directory), payload))

    def test_b3_recipes_preserve_workload_and_change_only_duration(self):
        root = Path(__file__).resolve().parents[2] / "configs/training"
        benchmark = load_training_config(root / "benchmarks/single_gpu_b3.json")
        profiling = load_training_config(root / "profiling/single_gpu_b3_torch_profile.json")
        smoke = load_training_config(root / "benchmarks/single_gpu_b3_smoke.json")
        self.assertEqual((benchmark.nvarc_max_tokens, benchmark.loop.vae_microbatch_size,
                          benchmark.loop.optimizer_batch_size), (134400, 48, 48))
        self.assertEqual(benchmark.flux_packing.max_tokens, 12288)
        self.assertIsNone(benchmark.optimizer.max_grad_norm)
        self.assertEqual((benchmark.benchmark.warmup_steps, benchmark.benchmark.measured_steps,
                          benchmark.benchmark.repetitions), (10, 50, 3))
        self.assertEqual(profiling, replace(benchmark, benchmark=None,
                         loop=replace(benchmark.loop, max_optimizer_steps=7)))
        self.assertEqual((smoke.benchmark.warmup_steps, smoke.benchmark.measured_steps,
                          smoke.benchmark.repetitions), (1, 2, 1))
        self.assertEqual(smoke, replace(benchmark, benchmark=smoke.benchmark,
                         loop=replace(benchmark.loop, max_optimizer_steps=3)))

    def test_schema_v10_requires_packing_and_rejects_retired_fields(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original = json.loads((root / "configs/training/benchmarks/single_gpu_b3.json").read_text())
        changes = (lambda p: p.pop("flux_packing"),
                   lambda p: p.update(flux_packing=None),
                   lambda p: p["flux_packing"].update(max_tokens=True),
                   lambda p: p["flux_packing"].update(max_tokens=9299),
                   lambda p: p["flux_packing"].update(backend="sdpa"),
                   lambda p: p["loop"].update(flux_microbatch_size=1),
                   lambda p: p["loop"].update(flux_microbatch_size=None),
                   lambda p: p["loop"].update(nvarc_source_batch_size=16),
                   lambda p: p.update(padding=None),
                   lambda p: p.update(flux_scheduling=None),
                   lambda p: p.update(padding={"flux_bucket_boundaries": [8400]}),
                   lambda p: p.update(flux_scheduling={"shapes": []}))
        with tempfile.TemporaryDirectory() as directory:
            for change in changes:
                payload = json.loads(json.dumps(original))
                change(payload)
                with self.assertRaises((ValueError, TypeError)):
                    load_training_config(_write_config(Path(directory), payload))


    def test_nvarc_budget_is_explicit_and_independent_of_optimizer_batch_size(self):
        payload = _payload()
        payload["nvarc_max_tokens"] = 8400
        with tempfile.TemporaryDirectory() as directory:
            config = load_training_config(_write_config(Path(directory), payload))
            self.assertEqual(config.nvarc_max_tokens, 8400)
            self.assertEqual(config.loop.optimizer_batch_size, 4)
            for value in (0, -1, True, 1.5):
                payload["nvarc_max_tokens"] = value
                with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                    load_training_config(_write_config(Path(directory), payload))

    def test_prefetch_contracts(self):
        path = Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b3.json"
        original = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            config = load_training_config(_write_config(Path(directory), original))
        self.assertEqual((config.data.num_workers, config.data.prefetch_factor), (2, 8))
        for fields in ({"prefetch_factor": None}, {"num_workers": 0},
                       {"prefetch_factor": 0}, {"prefetch_factor": True}):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as directory:
                payload = copy.deepcopy(original)
                payload["data"].update(fields)
                with self.assertRaises((ValueError, TypeError)):
                    load_training_config(_write_config(Path(directory), payload))


    def test_rejects_retired_nvarc_options_and_precision(self):
        mutations = (
            lambda p: p.update(schema_version=6),
            lambda p: p.update(nvarc_scheduling=None),
            lambda p: p.update(padding={"nvarc_bucket_boundaries": None}),
            lambda p: p["models"].update(nvarc_dtype="float32"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for mutate in mutations:
                payload = _payload()
                mutate(payload)
                with self.assertRaises(ValueError):
                    load_training_config(_write_config(Path(directory), payload))

    def test_loads_checked_in_packed_reference_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        config = load_training_config(
            repository_root / "configs/training/benchmarks/single_gpu_b3.json"
        )
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.models.nvarc_dtype, "bfloat16")
        self.assertEqual(config.models.vae_dtype, "float32")
        self.assertEqual(config.models.flux_parameter_dtype, "float32")
        self.assertEqual(config.models.flux_compute_dtype, "bfloat16")
        self.assertEqual(config.data.split_directory,
                         repository_root / "data/benchmarks/single_gpu_b0/splits/train")
        self.assertEqual(config.flux_packing.max_tokens, 12288)


    def test_loads_valid_config_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config = load_training_config(
                _write_config(directory, _payload())
            )

        self.assertEqual(config.seed, 17)
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(
            config.models.nvarc_checkpoint,
            (directory / "models/nvarc").resolve(),
        )
        self.assertEqual(config.models.flux_checkpoint, Path("/models/flux"))
        self.assertEqual(
            config.data.split_directory,
            (directory / "data/split").resolve(),
        )
        self.assertEqual(config.loop.optimizer_batch_size, 4)
        self.assertEqual(config.nvarc_max_tokens, 134400)
        self.assertEqual(config.loop.vae_microbatch_size, 8)
        self.assertEqual(config.optimizer.beta2, 0.99)


    def test_rejects_missing_and_extra_fields(self) -> None:
        cases = []
        missing = _payload()
        del missing["seed"]
        cases.append(missing)
        extra = _payload()
        extra["epoch_count"] = 1
        cases.append(extra)
        nested_extra = _payload()
        nested_extra["data"]["batch_size"] = 1
        cases.append(nested_extra)

        for payload in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    load_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_unsupported_schema_device_and_dtype(self) -> None:
        cases = (
            ("schema_version", 6, "unsupported schema_version"),
            ("schema_version", 7, "unsupported schema_version"),
            ("schema_version", 8, "unsupported schema_version"),
            ("schema_version", 9, "unsupported schema_version"),
            ("device", "cpu", "device must be"),
        )
        for field, value, message in cases:
            payload = _payload()
            payload[field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    load_training_config(
                        _write_config(Path(temp), payload)
                    )

        payload = _payload()
        payload["models"]["vae_dtype"] = "float16"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "models.vae_dtype"):
                load_training_config(
                    _write_config(Path(temporary_directory), payload)
                )

        precision_cases = (
            ("flux_parameter_dtype", "bfloat16", "must be 'float32'"),
            ("flux_compute_dtype", "float32", "must be 'bfloat16'"),
        )
        for field, value, message in precision_cases:
            payload = _payload()
            payload["models"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    load_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_boolean_and_nonpositive_integer_fields(self) -> None:
        cases = (
            ("seed", True, TypeError),
            ("seed", 2**64, ValueError),
            ("num_workers", -1, ValueError),
            ("num_workers", 1, TypeError),
            ("max_optimizer_steps", 0, ValueError),
            ("optimizer_batch_size", True, TypeError),
            ("vae_microbatch_size", True, TypeError),
        )
        for field, value, error_type in cases:
            payload = _payload()
            if field == "seed":
                payload[field] = value
            elif field == "num_workers":
                payload["data"][field] = value
            else:
                payload["loop"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(error_type):
                    load_training_config(
                        _write_config(Path(temp), payload)
                    )

        payload = _payload()
        payload["data"]["shuffle"] = 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(TypeError, "data.shuffle"):
                load_training_config(
                    _write_config(Path(temporary_directory), payload)
                )

    def test_rejects_invalid_optimizer_values(self) -> None:
        cases = (
            ("learning_rate", 0, "learning_rate"),
            ("beta1", -0.1, "beta values"),
            ("beta2", 1.0, "beta values"),
            ("epsilon", 0, "epsilon"),
            ("weight_decay", -1, "weight_decay"),
            ("learning_rate", float("inf"), "finite"),
        )
        for field, value, message in cases:
            payload = _payload()
            payload["optimizer"][field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    load_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_invalid_scheduler_and_benchmark(self) -> None:
        cases = (
            ("scheduler", "type", "linear"),
            ("scheduler", "warmup_steps", -1),
            ("scheduler", "warmup_steps", True),
            ("scheduler", "warmup_steps", 10),
            ("scheduler", "total_steps", 2),
            ("benchmark", "warmup_steps", -1),
            ("benchmark", "measured_steps", 1),
            ("benchmark", "repetitions", 0),
        )
        for section, field, value in cases:
            payload = _payload()
            payload.update(
                schema_version=10,
                scheduler={"type": "cosine", "warmup_steps": 2, "total_steps": 10},
                benchmark={"warmup_steps": 1, "measured_steps": 2, "repetitions": 3},
            )
            payload[section][field] = value
            with self.subTest(section=section, field=field, value=value), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises((ValueError, TypeError)):
                    load_training_config(_write_config(Path(temp), payload))


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"


class TrainingRunConfigTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(RECIPE.read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run.json"

    def load(self, payload=None):
        self.path.write_text(json.dumps(self.payload if payload is None else payload))
        return load_training_config(self.path)

    def test_recipe_records_effective_spread_and_preserves_explicit_override(self):
        config = self.load()
        self.assertEqual(config.flux_packing.max_spread_tokens, 1000)
        self.assertEqual(config.recipe["execution"]["flux_max_spread_tokens"], 1000)
        self.payload["execution"]["flux_max_spread_tokens"] = 0
        config = self.load()
        self.assertEqual(config.flux_packing.max_spread_tokens, 0)
        self.assertEqual(config.recipe["execution"]["flux_max_spread_tokens"], 0)
        for invalid in (True, -1, 1.5):
            self.payload["execution"]["flux_max_spread_tokens"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                self.load()

    def test_recipe_reuses_b3_components_and_retains_complete_epoch(self):
        config = load_training_config(RECIPE)
        benchmark = load_training_config(ROOT / "configs/training/benchmarks/single_gpu_b3.json")
        self.assertEqual(config.models, benchmark.models)
        self.assertEqual(config.optimizer, replace(benchmark.optimizer, max_grad_norm=1.0))
        self.assertEqual(config.optimizer.max_grad_norm, config.recipe["optimizer"]["max_grad_norm"])
        self.assertEqual(config.flux_packing, benchmark.flux_packing)
        self.assertFalse(config.data.shuffle)
        self.assertEqual(config.nvarc_max_tokens, 134400)
        self.assertEqual(config.targets_per_sequence, 3)
        self.assertEqual(config.loop.max_optimizer_steps, 906)
        self.assertEqual(config.final_batch_target_count, 30)
        self.assertEqual(config.endpoint_target_count, 3623)
        batch_counts = [48] * 905 + [30]
        self.assertEqual(sum(batch_counts), config.recipe["loop"]["expected_target_count"])
        self.assertEqual(config.validation_steps, (0, 100, 200, 300, 400, 500, 600, 700, 800, 906))
        self.assertEqual(config.sampling_steps, (0, 300, 600, 906))
        self.assertEqual(config.checkpoint_steps, (300, 600, 906))
        self.assertAlmostEqual(config.optimizer.learning_rate * config.recipe["scheduler"]["first_update_warmup_fraction"], 2e-7)

    def test_recipe_batches_keep_whole_sequences_with_an_independent_nvarc_budget(self):
        self.payload["execution"]["nvarc_max_tokens"] = 8400
        self.assertEqual(self.load().nvarc_max_tokens, 8400)
        self.payload["loop"]["optimizer_batch_size"] = 49
        with self.assertRaisesRegex(ValueError, "fit whole training sequences"):
            self.load()

    def test_resolves_all_paths_without_requiring_deployment_or_files(self):
        self.payload["models"]["flux_checkpoint"] = "/models/flux"
        self.payload["deployment"]["output_directory"] = "outputs"
        config = self.load()
        self.assertEqual(config.models.flux_checkpoint, Path("/models/flux"))
        self.assertEqual(config.recipe["deployment"]["output_directory"], (self.path.parent / "outputs").resolve())
        for section, field in (
            ("models", "nvarc_checkpoint"), ("data", "task_index"),
            ("data", "source_archive"), ("data", "official_training_keys"),
            ("data", "prepared_split_directory"), ("validation", "bundle"),
        ):
            with self.subTest(section=section, field=field):
                self.assertEqual(config.recipe[section][field], (self.path.parent / self.payload[section][field]).resolve())
        self.assertIsNone(config.recipe["logging"]["entity"])
        self.assertIsNone(config.recipe["logging"]["project"])
        self.assertIsNone(config.recipe["checkpointing"]["keep_last"])

    def test_resume_checkpoint_is_optional_and_resolved_without_loading_state(self):
        self.assertIsNone(self.load().recipe["models"]["resume_from"])
        self.payload["models"]["resume_from"] = "outputs/checkpoints/run/step-000300"
        config = self.load()
        self.assertEqual(config.recipe["models"]["resume_from"],
                         (self.path.parent / self.payload["models"]["resume_from"]).resolve())
        self.assertFalse(config.recipe["models"]["resume_from"].exists())
        with self.assertRaises(TypeError):
            config.recipe["loop"]["drop_last"] = True
        self.assertIsInstance(config.recipe["data"]["geometries"], tuple)

    def test_entrypoints_can_reject_unsupported_experiments(self):
        benchmark = ROOT / "configs/training/benchmarks/single_gpu_b3.json"
        with self.assertRaisesRegex(ValueError, "expected 'baseline'"):
            load_training_config(RECIPE, expected_type="baseline")
        with self.assertRaisesRegex(ValueError, "expected 'training_run'"):
            load_training_config(benchmark, expected_type="training_run")
        self.assertEqual(load_training_config(RECIPE, expected_type="training_run").config_type, "training_run")
        self.assertEqual(load_training_config(benchmark, expected_type="baseline").config_type, "baseline")
        with self.assertRaisesRegex(ValueError, "expected_type"):
            load_training_config(RECIPE, expected_type="anything")

    def test_recipe_dict_is_json_compatible_and_independent_of_the_validated_config(self):
        config = load_training_config(RECIPE)
        serialized = config.recipe_dict()
        self.assertEqual(json.loads(json.dumps(serialized, allow_nan=False)), serialized)
        self.assertEqual(serialized["models"]["flux_checkpoint"], str(config.models.flux_checkpoint))
        self.assertIsInstance(serialized["data"]["geometries"], list)
        serialized["data"]["geometries"].clear()
        serialized["scheduler"]["warmup_steps"] = 999
        self.assertTrue(config.recipe["data"]["geometries"])
        self.assertEqual(config.recipe_dict()["scheduler"]["warmup_steps"], 50)

    def test_formats_share_runtime_type_and_expose_their_own_schedule(self):
        recipe = load_training_config(RECIPE)
        baseline = load_training_config(ROOT / "configs/training/benchmarks/single_gpu_b3.json")
        self.assertIs(type(recipe), TrainingConfig)
        self.assertIs(type(baseline), TrainingConfig)
        self.assertIsInstance(recipe.scheduler, LinearWarmupConstantConfig)
        self.assertEqual(recipe.scheduler.warmup_steps, 50)
        self.assertIsInstance(baseline.scheduler, CosineSchedulerConfig)
        self.assertIsNone(baseline.recipe)
        self.assertEqual(recipe.device, baseline.device)
        self.assertEqual(recipe.path, RECIPE)
        self.assertEqual(recipe.name, "arc_agi_1_epoch")
        with self.assertRaisesRegex(ValueError, "requires a training_run"):
            _ = baseline.validation_steps

    def test_full_final_batch_uses_full_endpoint_count_and_no_duplicate_final_event(self):
        self.payload["data"]["expected_task_count"] = 14400
        self.payload["loop"].update(max_optimizer_steps=900, expected_target_count=43200,
                                    expected_final_batch_target_count=48)
        self.payload["flow"]["endpoint_targets_in_final_batch"] = 4
        config = self.load()
        self.assertEqual(config.endpoint_target_count, 3600)
        self.assertEqual(config.sampling_steps, (0, 300, 600, 900))
        self.assertEqual(config.checkpoint_steps, (300, 600, 900))
        self.payload["flow"]["endpoint_targets_in_final_batch"] = 3
        with self.assertRaisesRegex(ValueError, "full final batch"):
            self.load()

    def test_epoch_shorter_than_interval_keeps_initial_and_final_events(self):
        self.payload["data"]["expected_task_count"] = 10
        self.payload["loop"].update(max_optimizer_steps=1, expected_target_count=30,
                                    expected_final_batch_target_count=30)
        self.payload["scheduler"].update(warmup_steps=1, first_update_warmup_fraction=1)
        config = self.load()
        self.assertEqual(config.validation_steps, (0, 1))
        self.assertEqual(config.sampling_steps, (0, 1))
        self.assertEqual(config.checkpoint_steps, (1,))
        self.assertEqual(config.endpoint_target_count, 3)

    def test_rejects_inconsistent_counts_and_batch_policy(self):
        for section, field, value in (
            ("data", "expected_task_count", 14494),
            ("loop", "expected_target_count", 43471),
            ("loop", "max_optimizer_steps", 905),
            ("loop", "expected_final_batch_target_count", 48),
            ("loop", "nvarc_source_batch_size", 8),
            ("loop", "drop_last", True),
            ("loop", "epochs", 2),
            ("flow", "endpoint_targets_per_full_batch", 49),
            ("flow", "endpoint_targets_in_final_batch", 31),
        ):
            with self.subTest(section=section, field=field):
                payload = copy.deepcopy(self.payload)
                payload[section][field] = value
                with self.assertRaises(ValueError):
                    self.load(payload)

    def test_rejects_invalid_loss_schedule_and_validation_relationships(self):
        changes = (
            lambda p: p["loss"]["region_weights"].update(boundary=0.3),
            lambda p: p["loss"]["region_weights"].update(canvas=0, grid=0, boundary=1),
            lambda p: p["scheduler"].update(warmup_steps=51),
            lambda p: p["scheduler"].update(warmup_steps=1000, first_update_warmup_fraction=0.001),
            lambda p: p["validation"]["loss"].update(endpoint_weight=0.5),
            lambda p: p["validation"]["loss"].update(at_end=False),
            lambda p: p["validation"].update(expected_query_count=99),
            lambda p: p["validation"]["sampling"].update(expected_task_count=101, expected_query_count=101),
            lambda p: p["validation"]["sampling"].update(expected_query_count=105),
            lambda p: p["validation"]["sampling"].update(seeds=[20167, 20167]),
            lambda p: p["checkpointing"].update(every_optimizer_steps=400),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                payload = copy.deepcopy(self.payload)
                change(payload)
                with self.assertRaises(ValueError):
                    self.load(payload)

    def test_rejects_unsupported_semantics_and_unsafe_boundaries(self):
        for section, field, value in (
            ("data", "require_both_parents_in_official_training", False),
            ("data", "geometry_sampling", "cycle"),
            ("data", "vae_safety_filter", True),
            ("models", "trainable_modules", ["nvarc", "flux_transformer"]),
            ("execution", "activation_checkpointing", True),
            ("execution", "distributed_strategy", "ddp"),
            ("execution", "device", "cpu"),
            ("execution", "flux_max_joint_tokens", 9299),
            ("scheduler", "type", "cosine"),
            ("logging", "automatic_system_metrics", True),
            ("logging", "model_watch", True),
            ("checkpointing", "save_full_state", False),
            ("final_test", "enabled", True),
        ):
            with self.subTest(section=section, field=field):
                payload = copy.deepcopy(self.payload)
                payload[section][field] = value
                with self.assertRaises(ValueError):
                    self.load(payload)

    def test_rejects_unknown_missing_and_duplicate_fields(self):
        changes = (
            lambda p: p.update(unknown=True),
            lambda p: p["flow"].update(endpoint_fraction=0.5),
            lambda p: p["logging"]["hardware"].update(automatic=True),
            lambda p: p["scheduler"].pop("warmup_steps"),
        )
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                payload = copy.deepcopy(self.payload)
                change(payload)
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    self.load(payload)
        self.path.write_text('{"seed": 167, "seed": 42}')
        with self.assertRaisesRegex(ValueError, "duplicate config field"):
            load_training_config(self.path)

    def test_rejects_wrong_types_nonfinite_numbers_and_invalid_shared_options(self):
        for section, field, value in (
            ("deployment", "gpu_count", True),
            ("loop", "optimizer_batch_size", True),
            ("data", "num_workers", 0),
            ("data", "prefetch_factor", None),
            ("flow", "uniform_min", False),
            ("flow", "endpoint_time", float("nan")),
            ("optimizer", "learning_rate", float("inf")),
            ("optimizer", "beta2", 1),
            ("optimizer", "weight_decay", -1),
            ("optimizer", "max_grad_norm", 0),
            ("validation", "split_seed", -1),
            ("checkpointing", "keep_last", 0),
        ):
            with self.subTest(section=section, field=field):
                payload = copy.deepcopy(self.payload)
                payload[section][field] = value
                with self.assertRaises((ValueError, TypeError)):
                    self.load(payload)
        self.payload["data"].update(num_workers=0, prefetch_factor=None)
        self.assertEqual(self.load().data.num_workers, 0)

    def test_metric_selection_is_explicit_and_bounded(self):
        for names in (["train/loss_task_123"], ["train/loss", "train/loss"], [], [1]):
            with self.subTest(names=names):
                self.payload["logging"]["train_metrics"] = names
                with self.assertRaises((ValueError, TypeError)):
                    self.load()
        self.payload["logging"]["train_metrics"] = ["train/loss", "progress/optimizer_step"]
        self.assertEqual(self.load().recipe["logging"]["train_metrics"], ("train/loss", "progress/optimizer_step"))


if __name__ == "__main__":
    unittest.main()
