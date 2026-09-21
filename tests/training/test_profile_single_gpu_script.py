"""Tests for the single-GPU profiling script orchestration."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, MagicMock, Mock, patch

from scripts.profile_single_gpu import _ProfiledModule, main, profile_from_config
import torch

from diffusion_arc.training.config import (
    AdamWConfig,
    DataConfig,
    LoopConfig,
    ModelConfig,
    TrainingConfig,
)
from diffusion_arc.training.setup import TrainingModels
from diffusion_arc.training.step.flux_batching import (
    FluxPackingPolicy,
)


class _AcceptAnyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        return args + tuple(kwargs.values())


def _config(root: Path) -> TrainingConfig:
    return TrainingConfig(
        seed=23,
        device="cuda:0",
        models=ModelConfig(
            nvarc_checkpoint=root / "nvarc",
            flux_checkpoint=root / "flux",
            nvarc_dtype="bfloat16",
            vae_dtype="float32",
            flux_parameter_dtype="float32",
            flux_compute_dtype="bfloat16",
        ),
        data=DataConfig(
            split_directory=root / "split",
            view_name="train",
            shuffle=True,
            num_workers=0,
        ),
        loop=LoopConfig(
            max_optimizer_steps=7,
            optimizer_batch_size=5,
            min_context_pairs_before_target=2,
            vae_microbatch_size=8,
        ),
        optimizer=AdamWConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        ),
        flux_packing=FluxPackingPolicy(),
    )


class ProfileSingleGpuScriptTests(unittest.TestCase):
    def test_binds_config_and_frozen_profile_wrappers_without_advancing_lr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "training.json"
            config = _config(root)

            device = torch.device("cuda:0")
            models = TrainingModels(vae_decoder=object(), flux_packing_backend=object(),
                tokenizer=object(),
                nvarc_encoder=_AcceptAnyModule(),
                vae_encoder=_AcceptAnyModule(),
                flux_transformer=_AcceptAnyModule(),
                flux_compute_dtype=torch.bfloat16,
            )
            samples = [SimpleNamespace(target_grids=(None,) * 6, key=f"s{i}") for i in range(7)]
            optimizer = object()
            scheduler = Mock()
            flow_generator = object()
            flow_sampler = object()
            loss_function = object()
            step = Mock(return_value=SimpleNamespace(loss=1.0, target_count=5, flux_forward_calls=3))
            profiler_context = MagicMock()
            profiler = Mock()
            profiler_context.__enter__.return_value = profiler
            profiler_schedule = object()

            def run_training(**kwargs):
                return SimpleNamespace(step=step)

            with (
                patch(
                    "scripts.profile_single_gpu.load_training_config",
                    return_value=config,
                ) as load_config,
                patch(
                    "scripts.profile_single_gpu.prepare_device",
                    return_value=device,
                ) as prepare_device,
                patch("scripts.profile_single_gpu.seed_process") as seed,
                patch(
                    "scripts.profile_single_gpu.load_models",
                    return_value=models,
                ) as load_models,
                patch(
                    "scripts.profile_single_gpu.build_dataloader",
                    return_value=samples,
                ) as build_loader,
                patch(
                    "scripts.profile_single_gpu.build_optimizer",
                    return_value=optimizer,
                ) as build_optimizer,
                patch(
                    "scripts.profile_single_gpu.build_lr_scheduler",
                    return_value=scheduler,
                ) as build_scheduler,
                patch(
                    "scripts.profile_single_gpu.build_flow_generator",
                    return_value=flow_generator,
                ) as build_generator,
                patch("scripts.profile_single_gpu.build_flow_sampler", return_value=flow_sampler) as build_sampler,
                patch("scripts.profile_single_gpu.build_velocity_loss", return_value=loss_function) as build_loss,
                patch(
                    "scripts.profile_single_gpu.Trainer",
                    side_effect=run_training,
                ) as run_training_mock,
                patch(
                    "scripts.profile_single_gpu.torch.profiler.schedule",
                    return_value=profiler_schedule,
                ) as build_schedule,
                patch(
                    "scripts.profile_single_gpu.torch.profiler.profile",
                    return_value=profiler_context,
                ) as build_profiler,
                patch(
                    "scripts.profile_single_gpu.PROFILER_OUTPUT_ROOT",
                    root / "profiling",
                ),
            ):
                profile_from_config(config_path)

                export_trace = build_profiler.call_args.kwargs["on_trace_ready"]
                export_trace(profiler)

        load_config.assert_called_once_with(config_path, expected_type="baseline")
        prepare_device.assert_called_once_with(config)
        seed.assert_called_once_with(config.seed)
        load_models.assert_called_once_with(config, device=device)
        build_loader.assert_called_once_with(config, tokenizer=models.tokenizer)
        build_optimizer.assert_called_once_with(
            config,
            flux_transformer=models.flux_transformer,
        )
        build_generator.assert_called_once_with(config, device=device)
        build_scheduler.assert_called_once_with(config, optimizer=optimizer)
        scheduler.step.assert_not_called()
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
        self.assertEqual(training_args, ())
        self.assertIs(training_kwargs["optimizer"], optimizer)
        self.assertIs(training_kwargs["scheduler"], scheduler)
        build_sampler.assert_called_once_with(config, generator=flow_generator)
        build_loss.assert_called_once_with(config)
        self.assertIs(training_kwargs["flow_sampler"], flow_sampler)
        self.assertIs(training_kwargs["loss_function"], loss_function)
        self.assertIs(training_kwargs["flux_transformer"], models.flux_transformer)
        self.assertIs(training_kwargs["flux_packing_backend"], models.flux_packing_backend)
        self.assertEqual(training_kwargs["nvarc_max_tokens"], config.nvarc_max_tokens)
        self.assertEqual(training_kwargs["flux_max_spread_tokens"], config.flux_packing.max_spread_tokens)
        self.assertNotIn("config", training_kwargs)
        self.assertFalse(any(name.startswith("on_") for name in training_kwargs))
        for name, range_name in (("nvarc_encoder", "prepare/nvarc"), ("vae_encoder", "prepare/vae")):
            module = training_kwargs[name]
            self.assertIsInstance(module, _ProfiledModule)
            self.assertIs(module.wrapped_module, getattr(models, name))
            self.assertEqual(module.range_name, range_name)
        self.assertEqual(profiler.step.call_count, 7)
        self.assertEqual(step.call_count, 7)
        self.assertEqual([call.args[0] for call in step.call_args_list], [(sample,) for sample in samples])
        profiler.export_chrome_trace.assert_called_once_with(
            str(root / "profiling/training/trace.json")
        )

    def test_profiled_module_records_its_named_forward_range(self) -> None:
        module = _AcceptAnyModule()
        profiled_module = _ProfiledModule(module, range_name="prepare/nvarc")
        range_context = MagicMock()

        with patch(
            "scripts.profile_single_gpu.torch.profiler.record_function",
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
        with patch("scripts.profile_single_gpu.profile_from_config") as train:
            result = main(["--config", "run.json"])

        self.assertEqual(result, 0)
        train.assert_called_once_with(Path("run.json"))

        stderr = Mock()
        with self.assertRaises(SystemExit), patch("sys.stderr", stderr):
            main([])


if __name__ == "__main__":
    unittest.main()
