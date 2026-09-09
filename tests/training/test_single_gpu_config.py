"""Tests for the strict single-GPU training JSON configuration."""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from diffusion_arc.training.single_gpu_config import load_single_gpu_training_config
from scripts import train_single_gpu


def _payload() -> dict:
    return {
        "schema_version": 7,
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
            "flux_microbatch_size": 2,
            "min_context_pairs_before_target": 1,
            "nvarc_source_batch_size": 8,
            "vae_microbatch_size": 8,
        },
        "scheduler": None, "benchmark": None,
        "padding": {"flux_bucket_boundaries": None}, "flux_scheduling": None,
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
    def test_current_config_preserves_workload_and_profile_changes_only_duration(self):
        root = Path(__file__).resolve().parents[2] / "configs/training"
        benchmark = load_single_gpu_training_config(root / "benchmarks/single_gpu_b3_nvarc.json")
        profiling = load_single_gpu_training_config(root / "profiling/single_gpu_b3_nvarc_torch_profile.json")
        self.assertEqual((benchmark.loop.nvarc_source_batch_size, benchmark.loop.vae_microbatch_size,
                          benchmark.loop.optimizer_batch_size), (16, 48, 48))
        self.assertEqual(len(benchmark.flux_scheduling.shapes), 24)
        self.assertEqual((benchmark.benchmark.warmup_steps, benchmark.benchmark.measured_steps,
                          benchmark.benchmark.repetitions), (10, 50, 3))
        self.assertEqual(profiling, replace(benchmark, benchmark=None,
                         loop=replace(benchmark.loop, max_optimizer_steps=7)))

    def test_prefetch_and_flux_policy_contracts(self):
        path = Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b3_nvarc.json"
        original = json.loads(path.read_text())
        mutations = (
            lambda p: p["data"].update(prefetch_factor=None),
            lambda p: p["data"].update(num_workers=0),
            lambda p: p["loop"].update(flux_microbatch_size=1),
            lambda p: p["padding"].update(flux_bucket_boundaries=[8400]),
            lambda p: p["flux_scheduling"].update(shapes=[]),
            lambda p: p["flux_scheduling"].update(shapes={}),
            lambda p: p["flux_scheduling"].update(shapes=[{"batch_size": 1, "max_tokens": 8399, "forward_backward_ms": 1}]),
            lambda p: p["flux_scheduling"]["shapes"][0].update(forward_backward_ms=True),
            lambda p: p["flux_scheduling"]["shapes"].append(p["flux_scheduling"]["shapes"][0]),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(mutations):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                    load_single_gpu_training_config(_write_config(Path(directory), payload))

    def test_rejects_retired_nvarc_options_and_precision(self):
        mutations = (
            lambda p: p.update(schema_version=6),
            lambda p: p.update(nvarc_scheduling=None),
            lambda p: p["padding"].update(nvarc_bucket_boundaries=None),
            lambda p: p["models"].update(nvarc_dtype="float32"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for mutate in mutations:
                payload = _payload()
                mutate(payload)
                with self.assertRaises(ValueError):
                    load_single_gpu_training_config(_write_config(Path(directory), payload))

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

        profiling = load_single_gpu_training_config(
            repository_root
            / "configs/training/profiling/single_gpu_torch_profile.json"
        )
        self.assertEqual(profiling.loop.nvarc_source_batch_size, 8)
        self.assertEqual(profiling.loop.vae_microbatch_size, 8)

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
        self.assertEqual(config.loop.nvarc_source_batch_size, 8)
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
                    load_single_gpu_training_config(
                        _write_config(Path(temp), payload)
                    )

    def test_rejects_unsupported_schema_device_and_dtype(self) -> None:
        cases = (
            ("schema_version", 6, "unsupported schema_version"),
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
            ("num_workers", 1, TypeError),
            ("max_optimizer_steps", 0, ValueError),
            ("optimizer_batch_size", True, TypeError),
            ("nvarc_source_batch_size", 0, ValueError),
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
                schema_version=7,
                scheduler={"type": "cosine", "warmup_steps": 2, "total_steps": 10},
                benchmark={"warmup_steps": 1, "measured_steps": 2, "repetitions": 3},
            )
            payload[section][field] = value
            with self.subTest(section=section, field=field, value=value), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises((ValueError, TypeError)):
                    load_single_gpu_training_config(_write_config(Path(temp), payload))


if __name__ == "__main__":
    unittest.main()
