"""The training benchmark CLI validates inputs before loading the training runtime."""

import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.benchmark_training import main


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"
BENCHMARK = ROOT / "configs/training/benchmarks/ddp_training.json"


class TrainingBenchmarkScriptTests(unittest.TestCase):
    def test_benchmark_entry_preserves_explicit_training_choices(self):
        from diffusion_arc.training.config import load_training_config
        from diffusion_arc.training.benchmark_runner import run_training_benchmark
        config = load_training_config(ROOT / "configs/training/benchmarks/single_gpu_b3.json")
        original = config.recipe_dict()
        with patch("diffusion_arc.training.benchmark_runner.run_benchmark", return_value="progress") as run:
            result = run_training_benchmark(config, max_steps=2, emit=lambda record: None)
        self.assertEqual(result, "progress")
        self.assertIs(run.call_args.args[0], config)
        self.assertEqual(run.call_args.kwargs["experiment"].max_steps, 2)
        self.assertEqual(config.recipe_dict(), original)
        self.assertEqual(config.flux_packing.algorithm, "legacy_bfd_v1")

    def test_passes_recipe_step_limit_and_source_identity(self):
        run = Mock()
        with patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}), \
                patch("scripts.benchmark_training.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="revision\n")):
            self.assertEqual(main(["--config", str(RECIPE), "--max-steps", "2"]), 0)
        config = run.call_args.args[0]
        self.assertEqual(config.config_type, "training_run")
        self.assertEqual(config.loop.optimizer_batch_size, 48)
        self.assertEqual(run.call_args.kwargs["max_steps"], 2)
        self.assertEqual(run.call_args.kwargs["source_identity"]["git_commit"], "revision")

    def test_single_process_recipe_remains_single_process(self):
        run = Mock()
        with patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            main(["--config", str(ROOT / "configs/training/benchmarks/single_gpu_b3.json"), "--max-steps", "2"])
        self.assertEqual(run.call_args.args[0].recipe["execution"]["distributed_strategy"], "single_process")

    def test_rejects_bad_recipe_or_step_limit_before_runner_import(self):
        with patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": None}):
            for args in ([], ["--config", str(RECIPE), "--max-steps", "0"],
                         ["--config", str(RECIPE), "--max-steps", "907"]):
                with self.subTest(args=args), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    main(args)

    def test_prepared_smoke_and_benchmark_preserve_training_recipe(self):
        for mode, steps, measurement in (("smoke", 3, None), ("benchmark", 60, (10, 50))):
            run = Mock()
            with tempfile.TemporaryDirectory() as directory, \
                    patch.dict("os.environ", {"RANK": "0"}), \
                    patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
                output = Path(directory) / mode
                self.assertEqual(main(["--config", str(BENCHMARK), "--mode", mode,
                                       "--output-dir", str(output)]), 0)
                self.assertTrue((output / "steps.jsonl").exists())
            config = run.call_args.args[0]
            options = run.call_args.kwargs
            self.assertEqual(config.data.split_directory, ROOT / "data/benchmarks/single_gpu_b0/splits/train")
            self.assertEqual(config.recipe["loss"]["type"], "region_weighted_velocity_mse")
            self.assertEqual(config.recipe["flow"]["endpoint_targets_per_full_batch"], 4)
            self.assertEqual(config.scheduler.warmup_steps, 50)
            self.assertEqual(config.loop.max_optimizer_steps, 906)
            self.assertEqual(options["max_steps"], steps)
            self.assertEqual(options["expected_source_count"], 960)
            if measurement is None:
                self.assertNotIn("measurement", options)
            else:
                self.assertEqual((options["measurement"].warmup_steps, options["measurement"].measured_steps), measurement)
            self.assertEqual(len(options["source_identity"]["source_sha256"]), 64)

    def test_prepared_mode_requires_output_and_rejects_independent_step_override(self):
        invalid = (["--config", str(BENCHMARK), "--mode", "benchmark"],
                   ["--config", str(BENCHMARK), "--mode", "smoke", "--output-dir", "unused", "--max-steps", "2"])
        with patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": None}):
            for args in invalid:
                with self.subTest(args=args), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    main(args)

    def test_diagnostic_selection_is_separate_from_benchmark_measurement(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "0"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            output = Path(directory) / "diagnostic"
            main(["--config", str(BENCHMARK), "--mode", "diagnostic", "--diagnostic-batch", "29",
                  "--output-dir", str(output)])
        options = run.call_args.kwargs
        self.assertEqual((options["diagnostic_batch_number"], options["max_steps"]), (29, 3))
        self.assertEqual(options["trace_directory"], output)
        self.assertNotIn("measurement", options)
        for extra in (["--mode", "diagnostic"], ["--mode", "smoke", "--diagnostic-batch", "29"],
                      ["--mode", "diagnostic", "--diagnostic-batch", "0"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                main(["--config", str(BENCHMARK), "--output-dir", "unused", *extra])

    def test_secondary_rank_never_creates_artifacts(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "1"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            output = Path(directory) / "not-written"
            main(["--config", str(BENCHMARK), "--mode", "smoke", "--output-dir", str(output)])
            self.assertFalse(output.exists())
            self.assertNotIn("emit", run.call_args.kwargs)

    def test_memory_mode_traverses_full_window_without_benchmark_statistics(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "0"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            output = Path(directory) / "memory"
            main(["--config", str(BENCHMARK), "--mode", "memory", "--output-dir", str(output)])
        options = run.call_args.kwargs
        self.assertEqual(options["max_steps"], 60)
        self.assertTrue(options["memory_history"])
        self.assertEqual(options["trace_directory"], output)
        self.assertNotIn("measurement", options)
        self.assertNotIn("diagnostic_batch_number", options)
        self.assertEqual(run.call_args.args[0].flux_packing.max_tokens, 12288)

    def test_profile_records_one_step_after_benchmark_warmup_without_repetitions(self):
        run = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "0"}), \
                patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}):
            output = Path(directory) / "profile"
            main(["--config", str(BENCHMARK), "--mode", "profile", "--output-dir", str(output)])
        run.assert_called_once()
        options = run.call_args.kwargs
        self.assertEqual(options["max_steps"], 11)
        self.assertEqual(options["profile_warmup_steps"], 10)
        self.assertEqual(options["trace_directory"], output)
        self.assertNotIn("measurement", options)
        self.assertNotIn("memory_history", options)

    def test_removed_stage_snapshot_option_is_rejected_before_training(self):
        run = Mock()
        with patch.dict(sys.modules, {"diffusion_arc.training.benchmark_runner": SimpleNamespace(run_training_benchmark=run)}), \
                patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            main(["--config", str(BENCHMARK), "--mode", "memory", "--output-dir", "unused",
                  "--memory-snapshot-step", "54"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
