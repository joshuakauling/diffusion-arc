"""Strict shared configuration contracts for training and measurement runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from diffusion_arc.training.benchmark_config import load_training_benchmark_config
from diffusion_arc.training.config import (
    CosineSchedulerConfig,
    LinearWarmupConstantConfig,
    load_training_config,
    parse_training_config,
)

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"
BENCHMARK = ROOT / "configs/training/benchmarks/single_gpu_b3.json"


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(RECIPE.read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run.json"

    def load(self, payload=None):
        self.path.write_text(json.dumps(self.payload if payload is None else payload))
        return load_training_config(self.path)

    def test_token_assignment_round_trips_and_requires_indexed_rng(self):
        self.assertEqual(self.load().rank_batch_assignment, "contiguous_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        for algorithm in ("contiguous_v1", "token_balanced_v1"):
            self.payload["execution"]["rank_batch_assignment"] = algorithm
            config = self.load()
            self.assertEqual(config.rank_batch_assignment, algorithm)
            self.assertEqual(parse_training_config(config.recipe_dict(), config_path=self.path), config)
        self.payload["flow"]["rng_scheme"] = "legacy_stream_v1"
        with self.assertRaisesRegex(ValueError, "assignment requires flow.rng_scheme"):
            self.load()
        for value in (None, True, 1, "unknown"):
            self.payload["execution"]["rank_batch_assignment"] = value
            with self.subTest(value=value), self.assertRaisesRegex((ValueError, TypeError), "rank_batch_assignment"):
                self.load()

    def test_joint_packing_assignment_round_trips_for_single_and_distributed_execution(self):
        self.payload["execution"].update(rank_batch_assignment="joint_packing_v1",
                                         flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        for strategy, gpu_counts in (("single_process", (1,)), ("ddp", (1, 2, 4, 8)),
                                     ("fsdp2", (1, 2, 4, 8))):
            for gpu_count in gpu_counts:
                with self.subTest(strategy=strategy, gpu_count=gpu_count):
                    self.payload["execution"]["distributed_strategy"] = strategy
                    self.payload["deployment"]["gpu_count"] = gpu_count
                    config = self.load()
                    self.assertEqual(config.rank_batch_assignment, "joint_packing_v1")
                    self.assertFalse(config.data.shuffle)
                    self.assertEqual(parse_training_config(config.recipe_dict(), config_path=self.path), config)

    def test_joint_packing_assignment_rejects_incompatible_rng_packing_and_shuffle(self):
        self.payload["execution"].update(rank_batch_assignment="joint_packing_v1",
                                         flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        for section, field, value, error in (
            ("flow", "rng_scheme", "legacy_stream_v1", "joint_packing_v1 assignment requires flow.rng_scheme"),
            ("execution", "flux_packing_algorithm", "balanced_release_v1",
             "joint_packing_v1 assignment requires execution.flux_packing_algorithm"),
        ):
            payload = copy.deepcopy(self.payload)
            payload[section][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                self.load(payload)
        data = self.payload["data"]
        self.payload["data"] = {name: data[name] for name in (
            "prepared_split_directory", "view_name", "expected_task_count", "pairs_per_sequence",
            "num_workers", "prefetch_factor",
        )}
        self.payload["data"].update(source="prepared_split", shuffle=False)
        self.assertFalse(self.load().data.shuffle)
        self.payload["data"]["shuffle"] = True
        with self.assertRaisesRegex(ValueError, "joint_packing_v1 assignment requires data.shuffle=false"):
            self.load()

    def test_all_checked_in_workloads_resolve_to_explicit_training_schema(self):
        for path in (ROOT / "configs/training").rglob("*.json"):
            if path.name == "ddp_numerics.json":
                continue
            with self.subTest(path=path):
                if json.loads(path.read_text()).get("config_type") == "ddp_benchmark":
                    config = load_training_benchmark_config(path).training
                else:
                    config = load_training_config(path)
                self.assertEqual(config.config_type, "training_run")
                self.assertIsNotNone(config.recipe)
                self.assertEqual(config.recipe["schema_version"], 2)

    def test_every_experiment_choice_is_required_without_silent_defaults(self):
        for section, field in ((None, "config_type"), (None, "benchmark"),
                               ("execution", "distributed_strategy"),
                               ("execution", "nvarc_hidden_layers"),
                               ("execution", "flux_max_spread_tokens"),
                               ("execution", "flux_packing_algorithm"),
                               ("flow", "rng_scheme"), ("loop", "max_steps"),
                               (None, "scheduler"), (None, "validation"),
                               (None, "logging"), (None, "checkpointing")):
            payload = copy.deepcopy(self.payload)
            del (payload if section is None else payload[section])[field]
            with self.subTest(section=section, field=field), self.assertRaisesRegex(ValueError, "missing="):
                self.load(payload)

    def test_epoch_limit_derives_counts_without_redundant_input(self):
        config = self.load()
        self.assertEqual(config.targets_per_sequence, 3)
        self.assertEqual(config.loop.max_optimizer_steps, 906)
        self.assertEqual(config.epoch_optimizer_steps, 906)
        self.assertEqual(config.recipe["loop"]["expected_target_count"], 43470)
        self.assertEqual(config.final_batch_target_count, 30)
        self.assertEqual(config.endpoint_target_count, 3623)
        self.assertEqual(config.validation_steps, (0, 100, 200, 300, 400, 500, 600, 700, 800, 906))
        self.assertEqual(config.sampling_steps, (0, 300, 600, 906))
        self.assertEqual(config.checkpoint_steps, (300, 600, 906))

    def test_configured_nvarc_layers_reach_runtime_and_serialized_recipe(self):
        for layers in ([9, 18, 27], [18, 27, 36], [9, 18, 36], [1, 2, 3]):
            with self.subTest(layers=layers):
                self.payload["execution"].update(
                    nvarc_hidden_layers=layers, nvarc_early_exit_layer=layers[-1],
                )
                config = self.load()
                self.assertEqual(config.nvarc_hidden_layers, tuple(layers))
                serialized = config.recipe_dict()
                self.assertEqual(serialized["execution"]["nvarc_hidden_layers"], layers)
                self.assertEqual(parse_training_config(serialized, config_path=self.path), config)

    def test_nvarc_layers_require_three_distinct_ordered_block_indices(self):
        for layers in ([], [9, 18], [9, 18, 27, 36], [0, 18, 27], [-1, 18, 27],
                       [9, 18, 37], [9, 9, 27], [27, 18, 9], [True, 18, 27],
                       [9.0, 18, 27], ["9", 18, 27], "9,18,27", None):
            with self.subTest(layers=layers), self.assertRaisesRegex((TypeError, ValueError), "nvarc_hidden_layers"):
                self.payload["execution"]["nvarc_hidden_layers"] = layers
                self.load()

    def test_nvarc_early_exit_must_match_the_selected_layers(self):
        for layers, early_exit in (([18, 27, 36], 27), ([9, 18, 27], 36)):
            self.payload["execution"].update(nvarc_hidden_layers=layers, nvarc_early_exit_layer=early_exit)
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "highest selected NVARC layer"):
                self.load()

    def test_step_limit_is_a_prefix_not_a_completed_epoch(self):
        self.payload["loop"].update(epochs=None, max_steps=50)
        config = self.load()
        self.assertEqual(config.loop.max_optimizer_steps, 50)
        self.assertEqual(config.epoch_optimizer_steps, 906)
        self.assertEqual(config.recipe["loop"]["expected_target_count"], 2400)
        self.assertEqual(config.final_batch_target_count, 30)
        self.assertEqual(config.recipe["loop"]["expected_final_batch_target_count"], 48)
        self.assertEqual(config.endpoint_target_count, 200)
        self.assertIsNone(config.recipe["loop"]["epochs"])

    def test_short_prefix_preserves_warmup_and_epoch_tail_flow_quotas(self):
        for path in (RECIPE, ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json"):
            payload = json.loads(path.read_text())
            full = self.load(payload)
            payload["loop"].update(epochs=None, max_steps=3)
            prefix = self.load(payload)
            self.assertEqual(prefix.scheduler, full.scheduler)
            self.assertEqual(prefix.recipe["flow"], full.recipe["flow"])
            self.assertEqual(prefix.final_batch_target_count, full.final_batch_target_count)
            self.assertEqual(prefix.endpoint_target_count,
                             3 * full.recipe["flow"]["endpoint_targets_per_full_batch"])

    def test_loop_limit_requires_exactly_one_and_respects_prepared_epoch(self):
        for values in ({"epochs": None, "max_steps": None}, {"epochs": 1, "max_steps": 50},
                       {"epochs": 2, "max_steps": None}, {"epochs": None, "max_steps": 907},
                       {"epochs": None, "max_steps": 0}, {"epochs": True, "max_steps": None}):
            payload = copy.deepcopy(self.payload)
            payload["loop"].update(values)
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                self.load(payload)

    def test_unknown_prepared_size_only_allows_bounded_uncheckpointed_runs(self):
        payload = json.loads(BENCHMARK.read_text())
        payload["data"]["expected_task_count"] = None
        config = self.load(payload)
        self.assertIsNone(config.epoch_optimizer_steps)
        self.assertEqual(config.loop.max_optimizer_steps, 60)
        payload["checkpointing"] = self.payload["checkpointing"]
        with self.assertRaisesRegex(ValueError, "checkpointing requires"):
            self.load(payload)
        payload["checkpointing"] = None
        payload["loop"].update(epochs=1, max_steps=None)
        with self.assertRaisesRegex(ValueError, "epoch runs require"):
            self.load(payload)

    def test_weights_continuation_requires_known_epoch_size_only_for_nonzero_offsets(self):
        payload = json.loads(BENCHMARK.read_text())
        source = json.loads((ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json").read_text())
        payload["models"] = source["models"]
        payload["models"]["completed_epochs"] = 0
        payload["data"]["expected_task_count"] = None
        self.assertIsNone(self.load(payload).epoch_optimizer_steps)
        payload["models"]["completed_epochs"] = 2
        with self.assertRaisesRegex(ValueError, "completed epoch offsets require"):
            self.load(payload)

    def test_b3_preserves_precision_packing_and_training_math(self):
        config = load_training_config(BENCHMARK)
        self.assertEqual((config.nvarc_max_tokens, config.loop.vae_microbatch_size,
                          config.loop.optimizer_batch_size), (134400, 48, 48))
        self.assertEqual(config.flux_packing.max_tokens, 12288)
        self.assertEqual(config.flux_packing.algorithm, "legacy_bfd_v1")
        self.assertIsNone(config.optimizer.max_grad_norm)
        self.assertEqual(config.recipe["loss"]["type"], "velocity_mse")
        self.assertEqual(config.recipe["flow"]["time_distribution"], "uniform")
        self.assertEqual(config.endpoint_target_count, 0)
        self.assertIsInstance(config.scheduler, CosineSchedulerConfig)
        self.assertEqual((config.scheduler.warmup_steps, config.scheduler.total_steps), (213, 7245))
        self.assertEqual((config.benchmark.warmup_steps, config.benchmark.measured_steps,
                          config.benchmark.repetitions), (10, 50, 3))
        for name, steps in (("benchmarks/single_gpu_b3_smoke.json", 3),
                            ("profiling/single_gpu_b3_torch_profile.json", 7)):
            other = load_training_config(ROOT / "configs/training" / name)
            self.assertEqual(other.loop.max_optimizer_steps, steps)
            self.assertEqual(other.epoch_optimizer_steps, 60)
            self.assertEqual((other.models, other.data, other.optimizer, other.scheduler,
                              other.flux_packing, other.recipe["loss"], other.recipe["flow"]),
                             (config.models, config.data, config.optimizer, config.scheduler,
                              config.flux_packing, config.recipe["loss"], config.recipe["flow"]))

    def test_services_can_be_disabled_independently(self):
        for section in ("validation", "logging", "checkpointing", "scheduler"):
            payload = copy.deepcopy(self.payload)
            payload[section] = None
            config = self.load(payload)
            self.assertIsNone(config.recipe[section])
        self.payload.update(validation=None, logging=None, checkpointing=None, scheduler=None)
        config = self.load()
        self.assertEqual((config.validation_steps, config.sampling_steps, config.checkpoint_steps), ((), (), ()))
        self.assertIsNone(config.scheduler)

    def test_checkpoint_cadence_is_independent_of_sampling(self):
        self.payload["checkpointing"]["every_optimizer_steps"] = 400
        self.assertEqual(self.load().checkpoint_steps, (400, 906))

    def test_explicit_distributed_choices_have_no_strategy_defaults(self):
        for strategy, algorithm in (("ddp", "balanced_release_v1"), ("fsdp2", "legacy_bfd_v1")):
            payload = copy.deepcopy(self.payload)
            payload["deployment"]["gpu_count"] = 2
            payload["execution"].update(distributed_strategy=strategy, flux_packing_algorithm=algorithm)
            payload["flow"]["rng_scheme"] = "global_step_target_v1"
            config = self.load(payload)
            self.assertEqual(config.flux_packing.algorithm, algorithm)
            payload["flow"]["rng_scheme"] = "legacy_stream_v1"
            with self.assertRaisesRegex(ValueError, "requires flow.rng_scheme"):
                self.load(payload)
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="balanced_release_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        with self.assertRaisesRegex(ValueError, "fsdp2 execution requires execution.flux_packing_algorithm"):
            self.load()

    def test_fsdp_resharding_round_trips_and_preserves_legacy_default(self):
        self.payload["deployment"]["gpu_count"] = 2
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        self.assertTrue(self.load().fsdp_reshard_after_forward)
        for value in (False, True):
            with self.subTest(value=value):
                self.payload["execution"]["fsdp_reshard_after_forward"] = value
                config = self.load()
                self.assertIs(config.fsdp_reshard_after_forward, value)
                serialized = config.recipe_dict()
                self.assertIs(serialized["execution"]["fsdp_reshard_after_forward"], value)
                self.assertEqual(parse_training_config(serialized, config_path=self.path), config)

    def test_fsdp_resharding_rejects_non_booleans_and_other_strategies(self):
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        for value in (None, 0, 1, "false", [], {}):
            self.payload["execution"]["fsdp_reshard_after_forward"] = value
            with self.subTest(value=value), self.assertRaisesRegex((TypeError, ValueError), "fsdp_reshard_after_forward"):
                self.load()
        self.payload["execution"]["fsdp_reshard_after_forward"] = False
        for strategy in ("single_process", "ddp"):
            self.payload["execution"]["distributed_strategy"] = strategy
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, "requires distributed_strategy=fsdp2"):
                self.load()

    def test_fsdp_block_grouping_round_trips_and_rejects_invalid_settings(self):
        self.payload["deployment"]["gpu_count"] = 2
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        self.assertEqual(self.load().fsdp_blocks_per_group, 1)
        for value in (1, 5):
            with self.subTest(value=value):
                self.payload["execution"]["fsdp_blocks_per_group"] = value
                config = self.load()
                self.assertEqual(config.fsdp_blocks_per_group, value)
                self.assertEqual(parse_training_config(config.recipe_dict(), config_path=self.path), config)
        for value in (None, True, False, 0, -1, 5.0, "5", [], {}):
            self.payload["execution"]["fsdp_blocks_per_group"] = value
            with self.subTest(value=value), self.assertRaisesRegex((TypeError, ValueError), "fsdp_blocks_per_group"):
                self.load()
        self.payload["execution"]["fsdp_blocks_per_group"] = 5
        for strategy in ("single_process", "ddp"):
            self.payload["execution"]["distributed_strategy"] = strategy
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, "requires distributed_strategy=fsdp2"):
                self.load()

    def test_removed_gradient_sync_option_is_rejected(self):
        self.payload["deployment"]["gpu_count"] = 2
        self.payload["execution"].update(distributed_strategy="fsdp2", flux_packing_algorithm="legacy_bfd_v1")
        self.payload["flow"]["rng_scheme"] = "global_step_target_v1"
        for value in (False, True):
            with self.subTest(value=value):
                self.payload["execution"]["fsdp_sync_gradients_at_batch_end"] = value
                with self.assertRaisesRegex(ValueError, "fsdp_sync_gradients_at_batch_end"):
                    self.load()

    def test_models_and_paths_resolve_without_requiring_files(self):
        config = self.load()
        self.assertEqual(config.models.flux_parameter_dtype, "float32")
        self.assertEqual(config.models.flux_compute_dtype, "bfloat16")
        self.assertIsInstance(config.scheduler, LinearWarmupConstantConfig)
        for section, field in (("models", "nvarc_checkpoint"), ("data", "task_index"),
                               ("data", "prepared_split_directory"), ("validation", "bundle")):
            self.assertEqual(config.recipe[section][field], (self.path.parent / self.payload[section][field]).resolve())
        self.payload["models"]["resume_from"] = "checkpoints/step-000050"
        self.assertEqual(self.load().recipe["models"]["resume_from"], (self.path.parent / "checkpoints/step-000050").resolve())

    def test_serialization_round_trips_without_derived_inputs(self):
        config = self.load()
        serialized = config.recipe_dict()
        for field in ("max_optimizer_steps", "expected_target_count", "expected_final_batch_target_count", "epoch_steps"):
            self.assertNotIn(field, serialized["loop"])
        self.assertEqual(parse_training_config(serialized, config_path=self.path), config)
        with self.assertRaises(TypeError):
            config.recipe["loop"]["epochs"] = None
        serialized["data"]["geometries"].clear()
        self.assertTrue(config.recipe["data"]["geometries"])

    def test_high_noise_initialization_and_evaluation_are_preserved(self):
        config = load_training_config(ROOT / "configs/training/arc_agi_1_epoch3_high_noise.json")
        self.assertEqual((config.seed, config.endpoint_target_count), (169, 21735))
        self.assertEqual(config.recipe["models"]["completed_epochs"], 2)
        self.assertEqual(config.recipe["validation"]["loss"], self.load().recipe["validation"]["loss"])
        payload = config.recipe_dict()
        for change in (lambda p: p["flow"].update(high_noise_targets_per_full_batch=25),
                       lambda p: p["flow"].update(high_noise_min=1),
                       lambda p: p["models"]["transformer_source"].update(revision="main"),
                       lambda p: p["models"]["transformer_source"].update(subfolder="../transformer"),
                       lambda p: p["models"].update(completed_epochs=-1)):
            invalid = copy.deepcopy(payload)
            change(invalid)
            with self.assertRaises(ValueError):
                self.load(invalid)

    def test_invalid_values_and_research_boundaries_are_rejected(self):
        for section, field, value in (
            ("execution", "device", "cpu"), ("execution", "flux_max_joint_tokens", 9299),
            ("execution", "flux_max_spread_tokens", -1), ("execution", "activation_checkpointing", True),
            ("models", "trainable_modules", ["nvarc", "flux_transformer"]),
            ("models", "flux_parameter_dtype", "bfloat16"), ("models", "nvarc_dtype", "float32"),
            ("data", "require_both_parents_in_official_training", False),
            ("data", "geometry_sampling", "cycle"), ("data", "vae_safety_filter", True),
            ("data", "num_workers", -1), ("data", "prefetch_factor", None),
            ("loop", "optimizer_batch_size", 49), ("loop", "vae_microbatch_size", True),
            ("loop", "drop_last", True), ("optimizer", "learning_rate", float("inf")),
            ("optimizer", "beta2", 1), ("optimizer", "weight_decay", -1),
            ("optimizer", "max_grad_norm", 0), ("flow", "endpoint_targets_per_full_batch", 49),
            ("flow", "endpoint_targets_in_final_batch", 31), ("flow", "uniform_min", False),
            ("flow", "endpoint_time", float("nan")), ("validation", "split_seed", -1),
            ("final_test", "enabled", True), ("checkpointing", "keep_last", 0)):
            invalid = copy.deepcopy(self.payload)
            invalid[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises((ValueError, TypeError)):
                self.load(invalid)

    def test_relationships_and_dynamic_schemas_are_strict(self):
        for change in (
            lambda p: p["loss"]["region_weights"].update(boundary=0.3),
            lambda p: p["loss"]["region_weights"].update(canvas=0, grid=0, boundary=1),
            lambda p: p["scheduler"].update(warmup_steps=51),
            lambda p: p["scheduler"].update(type="cosine"),
            lambda p: p["validation"]["loss"].update(endpoint_weight=0.5),
            lambda p: p["validation"]["loss"].update(at_end=False),
            lambda p: p["validation"].update(expected_query_count=99),
            lambda p: p["validation"]["sampling"].update(seeds=[20167, 20167]),
            lambda p: p["logging"].update(train_metrics=[]),
            lambda p: p["logging"].update(train_metrics=["train/loss_task_123"]),
            lambda p: p["loss"].update(type="velocity_mse"),
            lambda p: p["flow"].update(time_distribution="uniform"),
        ):
            invalid = copy.deepcopy(self.payload)
            change(invalid)
            with self.assertRaises((ValueError, TypeError)):
                self.load(invalid)

    def test_cosine_and_benchmark_constraints(self):
        payload = json.loads(BENCHMARK.read_text())
        for section, key, value in (("scheduler", "warmup_steps", -1), ("scheduler", "total_steps", 10),
                                     ("scheduler", "warmup_steps", 7245), ("benchmark", "measured_steps", 49),
                                     ("benchmark", "repetitions", 0)):
            invalid = copy.deepcopy(payload)
            invalid[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises((TypeError, ValueError)):
                self.load(invalid)

    def test_old_missing_unknown_and_duplicate_fields_are_rejected(self):
        for change in (lambda p: p.update(schema_version=1), lambda p: p.update(config_type="baseline"),
                       lambda p: p.update(unknown=True), lambda p: p["loop"].update(max_optimizer_steps=906)):
            invalid = copy.deepcopy(self.payload)
            change(invalid)
            with self.assertRaises(ValueError):
                self.load(invalid)
        with self.assertRaisesRegex(ValueError, "baseline configs are retired"):
            load_training_config(RECIPE, expected_type="baseline")
        self.path.write_text('{"seed": 1, "seed": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate config field"):
            load_training_config(self.path)


if __name__ == "__main__":
    unittest.main()
