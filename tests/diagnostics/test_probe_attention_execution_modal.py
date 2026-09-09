"""Exercise the cached-image launcher without network or GPU requests."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.diagnostics.test_probe_nvarc_batch_capacity_modal import _App

ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.diagnostics.probe_attention_execution"
CONFIG = "configs/diagnostics/attention_execution.json"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.volume = SimpleNamespace(commit=Mock())
        self.image = SimpleNamespace(object_id="cached-image")
        infrastructure = SimpleNamespace(LOCAL_REPOSITORY_ROOT=ROOT,
            REMOTE_REPOSITORY_ROOT=Path("/opt/diffusion-arc"), VOLUME_ROOT=Path("/mnt/diffusion-arc-models"),
            image=self.image, model_volume=self.volume)
        source = ROOT / "scripts/cloud/probe_attention_execution_modal.py"
        spec = importlib.util.spec_from_file_location("_attention_execution_launcher_test", source)
        self.launcher = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", list(sys.path)), patch.dict(sys.modules, {
                "modal": SimpleNamespace(App=_App, is_local=lambda: True, __version__="test"),
                "scripts.cloud.probe_flux_packing_modal": infrastructure}):
            spec.loader.exec_module(self.launcher)

    def test_resources_are_exactly_one_bounded_h200_without_retries(self):
        options = self.launcher.app.function_options
        self.assertIs(options["image"], self.image)
        self.assertEqual(options["timeout"], 870)
        self.assertEqual(options["gpu"], "H200")
        self.assertEqual(options["max_containers"], 1)
        self.assertEqual(options["retries"], 0)

    def test_local_preflight_rejects_paths_and_budget_before_remote_request(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            valid = dict(config_paths=(repository / CONFIG,),
                checkpoint_paths=(repository / "checkpoints/model",), max_duration_seconds=750,
                training=SimpleNamespace(data=SimpleNamespace(
                    split_directory=repository / "data/benchmarks/single_gpu_b0/splits/train")))
            variants = [dict(config_paths=(Path("/tmp/outside.json"),)),
                        dict(checkpoint_paths=(Path("/tmp/outside-checkpoint"),)),
                        dict(max_duration_seconds=751), dict(max_duration_seconds=True)]
            for change in variants:
                with self.subTest(change=change), patch.dict(sys.modules, {MODULE: SimpleNamespace(
                        load_probe_config=Mock(return_value=SimpleNamespace(**{**valid, **change})))}):
                    with self.assertRaises(ValueError):
                        self.launcher._validate_layout(repository, CONFIG,
                            checkpoint_root=repository / "checkpoints", require_checkpoints=False)

    def test_launch_sends_source_config_and_data_identity(self):
        policy = object()
        probe = SimpleNamespace(source_metadata=Mock(return_value={"source_sha256": {"source": "sha"}}),
            config_hashes=Mock(return_value={"config": "sha"}), dataset_hashes=Mock(return_value={"data": "sha"}))
        with patch.dict(sys.modules, {MODULE: probe}), \
                patch.object(self.launcher, "_validate_layout", return_value=(ROOT / CONFIG, policy)):
            self.launcher.main()
        config, metadata = self.launcher.probe_attention_execution.remote.call_args.args
        self.assertEqual(config, CONFIG)
        self.assertEqual(metadata["dataset_sha256"], {"data": "sha"})
        self.assertEqual(metadata["config_sha256"], {"config": "sha"})
        self.assertEqual(metadata["source_sha256"], {"source": "sha"})


if __name__ == "__main__":
    unittest.main()
