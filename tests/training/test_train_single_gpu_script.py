"""The training CLI validates configuration before importing the run implementation."""

import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.train import main
from scripts.train_single_gpu import main as compatibility_main
from diffusion_arc.training.config import TrainingConfig


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "configs/training/arc_agi_1_epoch.json"


class TrainSingleGpuScriptTests(unittest.TestCase):
    def test_compatibility_entrypoint_uses_the_same_training_command(self):
        self.assertIs(compatibility_main, main)

    def test_ddp_recipe_reaches_the_same_runner_with_explicit_strategy_and_gpu_count(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = json.loads(RECIPE.read_text())
            payload["execution"]["distributed_strategy"] = "ddp"
            payload["deployment"]["gpu_count"] = 2
            path = Path(temp) / "ddp.json"
            path.write_text(json.dumps(payload))
            run = Mock()
            with patch.dict(sys.modules, {"diffusion_arc.training.runner": SimpleNamespace(run_training=run)}):
                self.assertEqual(main(["--config", str(path)]), 0)
            config = run.call_args.args[0]
            self.assertEqual(config.recipe["execution"]["distributed_strategy"], "ddp")
            self.assertEqual(config.recipe["deployment"]["gpu_count"], 2)
            self.assertEqual(config.recipe["flow"]["rng_scheme"], "global_step_target_v1")

    def test_passes_the_loaded_typed_config_with_resolved_paths_to_runner(self):
        run = Mock()
        with patch.dict(sys.modules, {"diffusion_arc.training.runner": SimpleNamespace(run_training=run)}):
            self.assertEqual(main(["--config", str(RECIPE)]), 0)
        run.assert_called_once()
        config = run.call_args.args[0]
        self.assertIsInstance(config, TrainingConfig)
        self.assertEqual(config.path, RECIPE)
        self.assertEqual(config.models.flux_checkpoint, ROOT / "checkpoints/flux2-klein-base-4b")
        self.assertEqual(config.loop.optimizer_batch_size, 48)
        self.assertEqual(config.validation_steps, (0, 100, 200, 300, 400, 500, 600, 700, 800, 906))
        self.assertEqual(config.sampling_steps, (0, 300, 600, 906))
        self.assertEqual(config.checkpoint_steps, (300, 600, 906))

    def test_baseline_or_invalid_recipe_is_rejected_before_importing_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp) / "invalid.json"
            payload = json.loads(RECIPE.read_text())
            payload["loop"]["optimizer_batch_size"] = 0
            invalid.write_text(json.dumps(payload))
            for path in (ROOT / "configs/training/benchmarks/single_gpu_b3.json", invalid):
                with self.subTest(path=path), patch.dict(sys.modules, {"diffusion_arc.training.runner": None}):
                    with self.assertRaises(ValueError):
                        main(["--config", str(path)])

    def test_profiler_flags_and_missing_config_are_rejected_by_cli(self):
        for args in ([], ["--config", str(RECIPE), "--profiler", "torch"]):
            with self.subTest(args=args), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(args)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
