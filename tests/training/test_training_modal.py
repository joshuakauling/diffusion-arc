"""Offline checks that training launchers install FA3 before GPU execution."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.cloud._training_image import FA3_REVISION


class _App:
    def __init__(self, _name):
        self.function_options = None

    def function(self, **options):
        self.function_options = options
        return lambda function: function

    def local_entrypoint(self):
        return lambda function: function


class RecordingImage:
    object_id = "image-test"

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return record


class TrainingModalTests(unittest.TestCase):
    def test_launchers_share_pinned_backward_capable_image_and_existing_volume(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("benchmark_single_gpu_modal", "profile_single_gpu_modal",
                     "probe_nvarc_batch_capacity_modal"):
            with self.subTest(name=name):
                image, volume = RecordingImage(), Mock()
                factory = Mock(return_value=image)
                volume_factory = Mock(return_value=volume)
                fake = SimpleNamespace(
                    App=_App, is_local=lambda: True,
                    Image=SimpleNamespace(from_registry=factory),
                    Volume=SimpleNamespace(from_name=volume_factory),
                )
                spec = importlib.util.spec_from_file_location(name, root / "scripts/cloud" / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {"modal": fake}), patch.object(sys, "path", list(sys.path)):
                    spec.loader.exec_module(module)
                self.assertEqual(factory.call_count, 1)
                self.assertIn("@sha256:", factory.call_args.args[0])
                volume_factory.assert_called_once_with("diffusion-arc-models", create_if_missing=False)
                self.assertEqual(module.app.function_options["gpu"], "H200")
                self.assertEqual(module.app.function_options["retries"], 0)
                commands = "\n".join(arg for call, args, _ in image.calls if call == "run_commands" for arg in args)
                self.assertIn(FA3_REVISION + "#subdirectory=hopper", commands)
                self.assertIn("flash_attn_3.flash_attn_interface", commands)
                self.assertNotIn("flash-attn==", commands)
                environment = {key: value for call, args, _ in image.calls if call == "env" for key, value in args[0].items()}
                self.assertEqual(environment["FLASH_ATTENTION_DISABLE_BACKWARD"], "FALSE")
                self.assertEqual(environment["FLASH_ATTENTION_DISABLE_VARLEN"], "FALSE")
                self.assertEqual(environment["FLASH_ATTENTION_DISABLE_HDIM128"], "FALSE")
                self.assertEqual(environment["FLASH_ATTENTION_DISABLE_FP16"], "TRUE")
                for feature in ("BACKWARD", "VARLEN", "HDIM128"):
                    self.assertIn(f"assert not CONFIG['build_flags']['FLASHATTENTION_DISABLE_{feature}']", commands)
                self.assertEqual(environment["FLASH_ATTN_CUDA_ARCHS"], "90")
                self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
