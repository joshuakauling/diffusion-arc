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
        "schema_version": 2,
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
            "nvarc_source_batch_size": 8,
            "vae_microbatch_size": 8,
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
    def test_b2_adds_exact_approved_flux_catalog_and_preserves_preparation(self) -> None:
        root = Path(__file__).resolve().parents[2] / "configs/training/benchmarks"
        preparation = load_single_gpu_training_config(root / "single_gpu_b2_preparation.json")
        b2 = load_single_gpu_training_config(root / "single_gpu_b2.json")
        self.assertEqual(replace(b2, loop=preparation.loop, padding=preparation.padding,
                                 flux_scheduling=None), preparation)
        self.assertEqual(replace(b2.loop, flux_microbatch_size=1), preparation.loop)
        self.assertIsNone(b2.loop.flux_microbatch_size)
        self.assertIsNone(b2.padding.flux_bucket_boundaries)
        expected = {
            1: [487, 2422, 3325, 4210, 4837, 6100, 8400],
            2: [487, 1348, 2126, 2422, 2749, 2990, 3325, 3714, 4210, 4837],
            4: [487, 778, 1090, 1348, 1578, 1818, 2126],
        }
        self.assertEqual(len(b2.flux_scheduling.shapes), 24)
        for size, bounds in expected.items():
            self.assertEqual([s.max_tokens for s in b2.flux_scheduling.shapes if s.batch_size == size], bounds)

    def test_schema_v6_rejects_ambiguous_or_incomplete_flux_policies(self) -> None:
        path = Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b2.json"
        original = json.loads(path.read_text())
        mutations = (
            lambda p: p.pop("flux_scheduling"),
            lambda p: p.update(flux_scheduling=None),
            lambda p: p["loop"].update(flux_microbatch_size=1),
            lambda p: p["padding"].update(flux_bucket_boundaries=[8400]),
            lambda p: p["flux_scheduling"].update(shapes={}),
            lambda p: p["flux_scheduling"].update(shapes=[]),
            lambda p: p["flux_scheduling"].update(call_overhead_tokens=1),
            lambda p: p["flux_scheduling"].update(shapes=[{"batch_size": 1, "max_tokens": 8399, "forward_backward_ms": 1}]),
            lambda p: p["flux_scheduling"]["shapes"][0].pop("forward_backward_ms"),
            lambda p: p["flux_scheduling"]["shapes"][0].update(forward_backward_ms=True),
            lambda p: p["flux_scheduling"]["shapes"].append(p["flux_scheduling"]["shapes"][0]),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(mutations):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                    load_single_gpu_training_config(_write_config(Path(directory), payload))
            original["flux_scheduling"] = None
            original["loop"]["flux_microbatch_size"] = 1
            self.assertIsNone(load_single_gpu_training_config(_write_config(Path(directory), original)).flux_scheduling)

    def test_b2_preparation_preserves_b1_training_and_flux_policy(self) -> None:
        root = Path(__file__).resolve().parents[2] / "configs/training/benchmarks"
        b1 = load_single_gpu_training_config(root / "single_gpu_b1.json")
        b2 = load_single_gpu_training_config(root / "single_gpu_b2_preparation.json")
        self.assertEqual(replace(b2, data=b1.data, padding=b1.padding, nvarc_scheduling=None), b1)
        self.assertEqual(b2.padding.flux_bucket_boundaries, b1.padding.flux_bucket_boundaries)
        self.assertIsNone(b2.padding.nvarc_bucket_boundaries)
        self.assertEqual((b2.data.num_workers, b2.data.prefetch_factor), (2, 8))
        self.assertEqual(replace(b2.data, num_workers=0, prefetch_factor=None), b1.data)
        self.assertEqual(len(b2.nvarc_scheduling.shapes), 20)
        self.assertEqual(b2.nvarc_scheduling.call_overhead_tokens, 2560)
        self.assertEqual(set(b2.nvarc_scheduling.__dataclass_fields__), {"shapes", "call_overhead_tokens"})

    def test_schema_v5_validates_prefetch_and_explicit_shape_catalog(self) -> None:
        path = Path(__file__).resolve().parents[2] / "configs/training/benchmarks/single_gpu_b2_preparation.json"
        original = json.loads(path.read_text())
        mutations = (
            lambda p: p["data"].update(num_workers=-1),
            lambda p: p["data"].update(num_workers=True),
            lambda p: p["data"].update(prefetch_factor=0),
            lambda p: p["data"].update(prefetch_factor=True),
            lambda p: p["data"].update(prefetch_factor=None),
            lambda p: p["data"].update(num_workers=0),
            lambda p: p["nvarc_scheduling"].update(shapes=[]),
            lambda p: p["nvarc_scheduling"].update(shapes={}),
            lambda p: p["nvarc_scheduling"].update(odd_source_bucket_length=8400),
            lambda p: p["nvarc_scheduling"].update(shapes=[{"batch_size": 2, "max_tokens": 8399}]),
            lambda p: p["nvarc_scheduling"].update(call_overhead_tokens=-1),
            lambda p: p["nvarc_scheduling"]["shapes"].append(p["nvarc_scheduling"]["shapes"][0]),
            lambda p: p["nvarc_scheduling"]["shapes"][0].update(batch_size=3),
            lambda p: p["nvarc_scheduling"]["shapes"][0].update(dummy_sources=True),
            lambda p: p["padding"].update(nvarc_bucket_boundaries=[8400]),
            lambda p: p.pop("nvarc_scheduling"),
            lambda p: p["data"].pop("prefetch_factor"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(mutations):
                payload = json.loads(json.dumps(original))
                mutate(payload)
                with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                    load_single_gpu_training_config(_write_config(Path(directory), payload))
            original["data"].update(num_workers=0, prefetch_factor=None)
            original["nvarc_scheduling"] = None
            config = load_single_gpu_training_config(_write_config(Path(directory), original))
            self.assertEqual(config.data.num_workers, 0)
            self.assertIsNone(config.nvarc_scheduling)

    def test_b1_changes_only_fixed_preparation_batches_and_padding(self) -> None:
        root = Path(__file__).resolve().parents[2]
        b0 = load_single_gpu_training_config(root / "configs/training/benchmarks/single_gpu_b0.json")
        b1 = load_single_gpu_training_config(root / "configs/training/benchmarks/single_gpu_b1.json")
        self.assertEqual((b1.loop.nvarc_source_batch_size, b1.loop.vae_microbatch_size), (16, 48))
        self.assertEqual(b1.padding.nvarc_bucket_boundaries,
                         (4096, 4608, 5120, 5632, 6144, 7168, 8400))
        self.assertEqual(b1.padding.flux_bucket_boundaries,
                         (*range(256, 3073, 256), 3584, 4096, 4608, 5120, 6144, 7168, 8400))
        self.assertIsNone(b0.padding.nvarc_bucket_boundaries)
        self.assertIsNone(b0.padding.flux_bucket_boundaries)
        self.assertEqual(replace(b1, padding=b0.padding, loop=replace(
            b1.loop, nvarc_source_batch_size=1, vae_microbatch_size=1,
        )), b0)

    def test_schema_v4_validates_independent_bucket_lists(self) -> None:
        payload = _payload()
        payload.update(schema_version=4, scheduler=None, benchmark=None, padding={
            "nvarc_bucket_boundaries": [256, 512, 8400],
            "flux_bucket_boundaries": None,
        })
        with tempfile.TemporaryDirectory() as directory:
            config = load_single_gpu_training_config(_write_config(Path(directory), payload))
        self.assertEqual(config.padding.nvarc_bucket_boundaries, (256, 512, 8400))
        self.assertIsNone(config.padding.flux_bucket_boundaries)
        for field in payload["padding"]:
            for value, error in (([], ValueError), ([256, 256, 8400], ValueError),
                                 ([512, 256, 8400], ValueError), ([0, 8400], ValueError),
                                 ([-1, 8400], ValueError), ([8192], ValueError),
                                 ([8401], ValueError), ([True, 8400], TypeError),
                                 ([256.0, 8400], TypeError), (["256", 8400], TypeError),
                                 ("256,8400", TypeError)):
                invalid = {**payload, "padding": {**payload["padding"], field: value}}
                with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(error):
                        load_single_gpu_training_config(_write_config(Path(directory), invalid))
        for section in ({"nvarc_bucket_boundaries": None},
                        {**payload["padding"], "sort_by_length": True}):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "fields differ"):
                    load_single_gpu_training_config(_write_config(
                        Path(directory), {**payload, "padding": section},
                    ))

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

    def test_loads_schema_v1_with_original_single_source_preparation(self) -> None:
        payload = _payload()
        payload["schema_version"] = 1
        del payload["loop"]["nvarc_source_batch_size"]
        del payload["loop"]["vae_microbatch_size"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = load_single_gpu_training_config(
                _write_config(Path(temporary_directory), payload)
            )

        self.assertEqual(config.loop.nvarc_source_batch_size, 1)
        self.assertIsNone(config.loop.vae_microbatch_size)
        self.assertIsNone(config.padding.nvarc_bucket_boundaries)
        self.assertIsNone(config.padding.flux_bucket_boundaries)

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
            ("schema_version", 7, "unsupported schema_version"),
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

    def test_b0_config_and_legacy_scheduler_defaults(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = load_single_gpu_training_config(root / "configs/training/benchmarks/single_gpu_b0.json")
        self.assertEqual(config.loop.optimizer_batch_size, 48)
        self.assertEqual(config.loop.max_optimizer_steps, 60)
        self.assertEqual(config.loop.nvarc_source_batch_size, 1)
        self.assertEqual(config.loop.vae_microbatch_size, 1)
        self.assertEqual(config.loop.flux_microbatch_size, 1)
        self.assertEqual(config.scheduler.warmup_steps, 213)
        self.assertEqual(config.scheduler.total_steps, 7245)
        self.assertEqual(config.optimizer.learning_rate, 1e-5)
        self.assertEqual(config.benchmark.warmup_steps, 10)
        self.assertEqual(config.benchmark.measured_steps, 50)
        self.assertEqual(config.benchmark.repetitions, 3)
        legacy = load_single_gpu_training_config(root / "configs/training/single_gpu.json")
        self.assertIsNone(legacy.scheduler)
        self.assertIsNone(legacy.benchmark)

    def test_profiles_change_only_duration_and_disable_benchmark(self) -> None:
        root = Path(__file__).resolve().parents[2]
        profile_steps = (
            train_single_gpu.PROFILER_WAIT_STEPS
            + train_single_gpu.PROFILER_WARMUP_STEPS
            + train_single_gpu.PROFILER_ACTIVE_STEPS
        )
        self.assertEqual(profile_steps, 7)
        for variant in ("b0", "b1", "b2"):
            with self.subTest(variant=variant):
                benchmark = load_single_gpu_training_config(
                    root / f"configs/training/benchmarks/single_gpu_{variant}.json"
                )
                profiling = load_single_gpu_training_config(
                    root / f"configs/training/profiling/single_gpu_{variant}_torch_profile.json"
                )
                self.assertEqual(profiling, replace(
                    benchmark,
                    loop=replace(benchmark.loop, max_optimizer_steps=profile_steps),
                    benchmark=None,
                ))

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
                schema_version=3,
                scheduler={"type": "cosine", "warmup_steps": 2, "total_steps": 10},
                benchmark={"warmup_steps": 1, "measured_steps": 2, "repetitions": 3},
            )
            payload[section][field] = value
            with self.subTest(section=section, field=field, value=value), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises((ValueError, TypeError)):
                    load_single_gpu_training_config(_write_config(Path(temp), payload))


if __name__ == "__main__":
    unittest.main()
