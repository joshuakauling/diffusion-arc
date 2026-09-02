"""Tests for the strict single-GPU training JSON configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config


def _payload() -> dict:
    return {
        "schema_version": 1,
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
        },
        "loop": {
            "max_optimizer_steps": 3,
            "optimizer_batch_size": 4,
            "flux_microbatch_size": 2,
            "min_context_pairs_before_target": 1,
        },
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


class SingleGpuTrainingConfigTests(unittest.TestCase):
    def test_loads_checked_in_reference_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]

        config = load_single_gpu_training_config(
            repository_root / "configs/training/single_gpu.json"
        )

        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.models.nvarc_dtype, "bfloat16")
        self.assertEqual(config.models.vae_dtype, "float32")
        self.assertEqual(config.models.flux_parameter_dtype, "float32")
        self.assertEqual(config.models.flux_compute_dtype, "bfloat16")
        self.assertEqual(
            config.data.split_directory,
            repository_root / "data/splits/overfitting_run_1",
        )

    def test_loads_valid_config_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config = load_single_gpu_training_config(
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
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_unsupported_schema_device_and_dtype(self) -> None:
        cases = (
            ("schema_version", 2, "unsupported schema_version"),
            ("device", "cpu", "device must be"),
        )
        for field, value, message in cases:
            payload = _payload()
            payload[field] = value
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError, message):
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )

        payload = _payload()
        payload["models"]["vae_dtype"] = "float16"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "models.vae_dtype"):
                load_single_gpu_training_config(
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
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_boolean_and_nonpositive_integer_fields(self) -> None:
        cases = (
            ("seed", True, TypeError),
            ("seed", 2**64, ValueError),
            ("num_workers", -1, ValueError),
            ("num_workers", 1, ValueError),
            ("max_optimizer_steps", 0, ValueError),
            ("optimizer_batch_size", True, TypeError),
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
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )

        payload = _payload()
        payload["data"]["shuffle"] = 1
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(TypeError, "data.shuffle"):
                load_single_gpu_training_config(
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
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )


if __name__ == "__main__":
    unittest.main()
