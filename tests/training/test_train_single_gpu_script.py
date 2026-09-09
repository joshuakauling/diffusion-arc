"""Tests for the thin single-GPU training script orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, patch

import torch

from diffusion_arc.training.single_gpu_config import (
    AdamWConfig,
    SingleGpuDataConfig,
    SingleGpuLoopConfig,
    SingleGpuModelConfig,
    SingleGpuPaddingConfig,
    SingleGpuTrainingConfig,
)
from scripts.train_single_gpu import _ProfiledModule, main, train_from_config
from diffusion_arc.training.flux_schedule import FluxBatchShape, FluxSchedulingPolicy


class _AcceptAnyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        return args + tuple(kwargs.values())


def _config(root: Path) -> SingleGpuTrainingConfig:
    return SingleGpuTrainingConfig(
        seed=23,
        device="cuda:0",
        models=SingleGpuModelConfig(
            nvarc_checkpoint=root / "nvarc",
            flux_checkpoint=root / "flux",
            nvarc_dtype="bfloat16",
            vae_dtype="float32",
            flux_parameter_dtype="float32",
            flux_compute_dtype="bfloat16",
        ),
        data=SingleGpuDataConfig(
            split_directory=root / "split",
            view_name="train",
            shuffle=True,
            num_workers=0,
        ),
        loop=SingleGpuLoopConfig(
            max_optimizer_steps=7,
            optimizer_batch_size=5,
            flux_microbatch_size=None,
            min_context_pairs_before_target=2,
            nvarc_source_batch_size=8,
            vae_microbatch_size=8,
        ),
        optimizer=AdamWConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        ),
        padding=SingleGpuPaddingConfig(),
        flux_scheduling=FluxSchedulingPolicy((FluxBatchShape(1, 8400, 747.171),)),
    )


class TrainSingleGpuScriptTests(unittest.TestCase):
    def test_forwards_constructed_runtime_and_every_loop_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "training.json"
            config = _config(root)

            device = torch.device("cuda:0")
            models = SimpleNamespace(
                tokenizer=object(),
                nvarc_runner=_AcceptAnyModule(),
                vae_encoder=_AcceptAnyModule(),
                transformer=_AcceptAnyModule(),
                flux_compute_dtype=torch.bfloat16,
            )
            samples = object()
            optimizer = object()
            scheduler = Mock()
            flow_generator = object()
            profiler_context = MagicMock()
            profiler = Mock()
            profiler_context.__enter__.return_value = profiler
            profiler_schedule = object()

            def run_training(*_args, **kwargs) -> None:
                kwargs["on_optimizer_step"](torch.tensor(1.0), 5, 3)

            with (
                patch(
                    "scripts.train_single_gpu.load_single_gpu_training_config",
                    return_value=config,
                ) as load_config,
                patch(
                    "scripts.train_single_gpu.prepare_single_gpu_device",
                    return_value=device,
                ) as prepare_device,
                patch("scripts.train_single_gpu.seed_single_gpu_process") as seed,
                patch(
                    "scripts.train_single_gpu.load_single_gpu_models",
                    return_value=models,
                ) as load_models,
                patch(
                    "scripts.train_single_gpu.build_single_sample_loader",
                    return_value=samples,
                ) as build_loader,
                patch(
                    "scripts.train_single_gpu.build_flux_optimizer",
                    return_value=optimizer,
                ) as build_optimizer,
                patch(
                    "scripts.train_single_gpu.build_flux_scheduler",
                    return_value=scheduler,
                ) as build_scheduler,
                patch(
                    "scripts.train_single_gpu.build_flow_generator",
                    return_value=flow_generator,
                ) as build_generator,
                patch(
                    "scripts.train_single_gpu.run_single_gpu_training",
                    side_effect=run_training,
                ) as run_training_mock,
                patch(
                    "scripts.train_single_gpu.torch.profiler.schedule",
                    return_value=profiler_schedule,
                ) as build_schedule,
                patch(
                    "scripts.train_single_gpu.torch.profiler.profile",
                    return_value=profiler_context,
                ) as build_profiler,
                patch(
                    "scripts.train_single_gpu.PROFILER_OUTPUT_ROOT",
                    root / "profiling",
                ),
            ):
                train_from_config(config_path)

                export_trace = build_profiler.call_args.kwargs["on_trace_ready"]
                export_trace(profiler)

        load_config.assert_called_once_with(config_path)
        prepare_device.assert_called_once_with(config)
        seed.assert_called_once_with(config.seed)
        load_models.assert_called_once_with(config, device=device)
        build_loader.assert_called_once_with(config, tokenizer=models.tokenizer)
        build_optimizer.assert_called_once_with(
            config,
            transformer=models.transformer,
        )
        build_generator.assert_called_once_with(config, device=device)
        build_scheduler.assert_called_once_with(config, optimizer=optimizer)
        scheduler.step.assert_called_once_with()
        build_schedule.assert_called_once_with(
            wait=2,
            warmup=2,
            active=3,
            repeat=1,
        )
        build_profiler.assert_called_once_with(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            schedule=profiler_schedule,
            on_trace_ready=ANY,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        run_training_mock.assert_called_once()
        training_args, training_kwargs = run_training_mock.call_args
        self.assertEqual(training_args, (samples,))
        self.assertEqual(training_kwargs["optimizer"], optimizer)
        self.assertEqual(training_kwargs["flux_compute_dtype"], torch.bfloat16)
        self.assertEqual(training_kwargs["max_optimizer_steps"], 7)
        self.assertEqual(training_kwargs["optimizer_batch_size"], 5)
        self.assertIsNone(training_kwargs["flux_microbatch_size"])
        self.assertEqual(training_kwargs["min_context_pairs_before_target"], 2)
        self.assertEqual(training_kwargs["nvarc_source_batch_size"], 8)
        self.assertEqual(training_kwargs["vae_microbatch_size"], 8)
        self.assertIsNone(training_kwargs["flux_bucket_boundaries"])

        self.assertIs(training_kwargs["flux_scheduling"], config.flux_scheduling)
        self.assertEqual(training_kwargs["generator"], flow_generator)
        self.assertIsNotNone(training_kwargs["on_optimizer_step"])
        expected_profiled_modules = (
            ("nvarc_runner", models.nvarc_runner, "prepare/nvarc"),
            ("vae_encoder", models.vae_encoder, "prepare/vae"),
            ("transformer", models.transformer, "train/flux_forward"),
        )
        for argument_name, wrapped_module, range_name in expected_profiled_modules:
            profiled_module = training_kwargs[argument_name]
            self.assertIsInstance(profiled_module, _ProfiledModule)
            self.assertIs(profiled_module.wrapped_module, wrapped_module)
            self.assertEqual(profiled_module.range_name, range_name)
        profiler.step.assert_called_once_with()
        profiler.export_chrome_trace.assert_called_once_with(
            str(root / "profiling/training/trace.json")
        )

    def test_profiled_module_records_its_named_forward_range(self) -> None:
        module = _AcceptAnyModule()
        profiled_module = _ProfiledModule(module, range_name="prepare/nvarc")
        range_context = MagicMock()

        with patch(
            "scripts.train_single_gpu.torch.profiler.record_function",
            return_value=range_context,
        ) as record_function:
            result = profiled_module("input", flag="value")

        self.assertEqual(result, ("input", "value"))
        record_function.assert_called_once_with("prepare/nvarc")
        range_context.__enter__.assert_called_once_with()
        range_context.__exit__.assert_called_once()
        self.assertEqual(
            tuple(profiled_module.parameters()),
            tuple(module.parameters()),
        )
        profiled_module.eval()
        self.assertFalse(module.training)
        profiled_module.train()
        self.assertTrue(module.training)

    def test_cli_accepts_only_the_required_config_option(self) -> None:
        with patch("scripts.train_single_gpu.train_from_config") as train:
            result = main(["--config", "run.json"])

        self.assertEqual(result, 0)
        train.assert_called_once_with(Path("run.json"))

        stderr = Mock()
        with self.assertRaises(SystemExit), patch("sys.stderr", stderr):
            main([])


if __name__ == "__main__":
    unittest.main()
