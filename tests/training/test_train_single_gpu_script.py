"""Tests for the thin single-GPU training script orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from diffusion_arc.training.single_gpu_config import (
    AdamWConfig,
    SingleGpuDataConfig,
    SingleGpuLoopConfig,
    SingleGpuModelConfig,
    SingleGpuTrainingConfig,
)
from scripts.train_single_gpu import main, train_from_config


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
            flux_microbatch_size=2,
            min_context_pairs_before_target=2,
        ),
        optimizer=AdamWConfig(
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
        ),
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
                nvarc_runner=object(),
                vae_encoder=object(),
                transformer=object(),
                flux_compute_dtype=torch.bfloat16,
            )
            samples = object()
            optimizer = object()
            flow_generator = object()
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
                    "scripts.train_single_gpu.build_flow_generator",
                    return_value=flow_generator,
                ) as build_generator,
                patch(
                    "scripts.train_single_gpu.run_single_gpu_training"
                ) as run_training,
            ):
                train_from_config(config_path)

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
        run_training.assert_called_once_with(
            samples,
            nvarc_runner=models.nvarc_runner,
            vae_encoder=models.vae_encoder,
            transformer=models.transformer,
            optimizer=optimizer,
            flux_compute_dtype=torch.bfloat16,
            max_optimizer_steps=7,
            optimizer_batch_size=5,
            flux_microbatch_size=2,
            min_context_pairs_before_target=2,
            generator=flow_generator,
        )

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
