"""Tests for offline model and runtime construction on one GPU."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch
from torch import nn

from diffusion_arc.model import NVARC_HIDDEN_SIZE, NVARC_VOCAB_SIZE
from diffusion_arc.training.single_gpu_config import (
    AdamWConfig,
    CosineSchedulerConfig,
    SingleGpuDataConfig,
    SingleGpuLoopConfig,
    SingleGpuModelConfig,
    SingleGpuTrainingConfig,
)
from diffusion_arc.training.single_gpu_setup import (
    build_flow_generator,
    build_flux_optimizer,
    build_flux_scheduler,
    build_single_sample_loader,
    load_single_gpu_models,
    prepare_single_gpu_device,
    _validate_parameter_placement,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.model_max_length = 0
        self.pad_token_id = 13
        self.eos_token_id = 15

    def __len__(self) -> int:
        return 16

    def get_vocab(self) -> dict[str, int]:
        return {
            **{str(value): value for value in range(10)},
            "Ċ": 10,
            "user": 11,
            "assistant": 12,
            "<|endoftext|>": 13,
            "<|im_start|>": 14,
            "<|im_end|>": 15,
        }

    def encode(
        self,
        _text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        if add_special_tokens:
            raise AssertionError("the checkpoint probe must not add tokens")
        return [14, 11, 10, 0, 15, 14, 12, 10]


class _FakeNvarc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(NVARC_VOCAB_SIZE, NVARC_HIDDEN_SIZE)
        self.config = SimpleNamespace(
            vocab_size=NVARC_VOCAB_SIZE,
            hidden_size=NVARC_HIDDEN_SIZE,
            num_hidden_layers=36,
            max_position_embeddings=262144,
            _attn_implementation="sdpa",
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Embedding:
        return self.embedding


class _FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(128)
        self.config = SimpleNamespace(batch_norm_eps=1e-5)

    def encode(self, _images: torch.Tensor) -> None:
        raise AssertionError("model loading must not execute the VAE")

    def decode(self, _latents: torch.Tensor) -> None:
        raise AssertionError("model loading must not execute the VAE")


class _FakeFluxTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.out_channels = 128
        self.config = SimpleNamespace(
            in_channels=128,
            joint_attention_dim=7680,
            attention_head_dim=128,
            num_attention_heads=24,
            num_layers=5,
            num_single_layers=20,
            guidance_embeds=False,
            axes_dims_rope=(32, 32, 32, 32),
        )


def _config(root: Path, *, shuffle: bool = False) -> SingleGpuTrainingConfig:
    return SingleGpuTrainingConfig(
        seed=17,
        device="cuda",
        models=SingleGpuModelConfig(
            nvarc_checkpoint=root / "nvarc",
            flux_checkpoint=root / "flux",
            nvarc_dtype="float32",
            vae_dtype="float32",
            flux_parameter_dtype="float32",
            flux_compute_dtype="bfloat16",
        ),
        data=SingleGpuDataConfig(
            split_directory=root / "split",
            view_name="train",
            shuffle=shuffle,
            num_workers=0,
        ),
        loop=SingleGpuLoopConfig(
            max_optimizer_steps=2,
            optimizer_batch_size=4,
            flux_microbatch_size=2,
            min_context_pairs_before_target=1,
        ),
        optimizer=AdamWConfig(
            learning_rate=1e-5,
            beta1=0.8,
            beta2=0.9,
            epsilon=1e-7,
            weight_decay=0.02,
        ),
    )


class SingleGpuSetupTests(unittest.TestCase):
    def test_cosine_schedule_and_legacy_constant_lr(self) -> None:
        config = _config(Path("/unused"))
        optimizer = build_flux_optimizer(config, transformer=nn.Linear(1, 1))
        self.assertIsNone(build_flux_scheduler(config, optimizer=optimizer))
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-5)
        config = replace(config, scheduler=CosineSchedulerConfig(warmup_steps=213, total_steps=7245))
        scheduler = build_flux_scheduler(config, optimizer=optimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        checkpoints = {1: 1e-5 / 213, 213: 1e-5, 3729: 0.5e-5, 7245: 0.0}
        for step in range(1, 7246):
            optimizer.step()
            scheduler.step()
            if step in checkpoints:
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], checkpoints[step])

    def test_missing_checkpoint_directory_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
            with patch(
                "diffusion_arc.training.single_gpu_setup."
                "AutoTokenizer.from_pretrained"
            ) as load_tokenizer:
                with self.assertRaisesRegex(FileNotFoundError, "NVARC checkpoint"):
                    load_single_gpu_models(config, device=torch.device("cpu"))

        load_tokenizer.assert_not_called()

    def test_loads_only_local_checkpoints_and_sets_parameter_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config.models.nvarc_checkpoint.mkdir()
            config.models.flux_checkpoint.mkdir()

            tokenizer = _FakeTokenizer()
            nvarc = _FakeNvarc()
            vae = _FakeVae()
            transformer = _FakeFluxTransformer()
            with (
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoModelForCausalLM.from_pretrained",
                    return_value=nvarc,
                ) as load_nvarc,
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoencoderKLFlux2.from_pretrained",
                    return_value=vae,
                ) as load_vae,
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "Flux2Transformer2DModel.from_pretrained",
                    return_value=transformer,
                ) as load_flux,
            ):
                models = load_single_gpu_models(
                    config,
                    device=torch.device("cpu"),
                )

        load_tokenizer.assert_called_once_with(
            config.models.nvarc_checkpoint,
            local_files_only=True,
            trust_remote_code=False,
        )
        load_nvarc.assert_called_once_with(
            config.models.nvarc_checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map={"": torch.device("cpu")},
            attn_implementation="sdpa",
        )
        for loader, subfolder in ((load_vae, "vae"), (load_flux, "transformer")):
            loader.assert_called_once_with(
                config.models.flux_checkpoint,
                subfolder=subfolder,
                local_files_only=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                device_map={"": torch.device("cpu")},
            )

        self.assertIs(models.tokenizer, tokenizer)
        self.assertIs(models.nvarc_runner.model, nvarc)
        self.assertIs(models.vae_encoder.vae, vae)
        self.assertIs(models.vae_decoder.vae, vae)
        self.assertIs(models.transformer, transformer)
        self.assertEqual(models.flux_compute_dtype, torch.bfloat16)
        self.assertFalse(models.nvarc_runner.training)
        self.assertFalse(models.vae_encoder.training)
        self.assertFalse(models.vae_decoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in nvarc.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in vae.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in transformer.parameters()))

    def test_loads_standalone_transformer_while_reusing_base_vae(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config.models.nvarc_checkpoint.mkdir()
            config.models.flux_checkpoint.mkdir()
            transformer_checkpoint = root / "trained-transformer"
            transformer_checkpoint.mkdir()

            with (
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoTokenizer.from_pretrained",
                    return_value=_FakeTokenizer(),
                ),
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoModelForCausalLM.from_pretrained",
                    return_value=_FakeNvarc(),
                ),
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "AutoencoderKLFlux2.from_pretrained",
                    return_value=_FakeVae(),
                ) as load_vae,
                patch(
                    "diffusion_arc.training.single_gpu_setup."
                    "Flux2Transformer2DModel.from_pretrained",
                    return_value=_FakeFluxTransformer(),
                ) as load_flux,
            ):
                load_single_gpu_models(
                    config,
                    device=torch.device("cpu"),
                    transformer_checkpoint=transformer_checkpoint,
                )

        load_vae.assert_called_once_with(
            config.models.flux_checkpoint,
            subfolder="vae",
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map={"": torch.device("cpu")},
        )
        load_flux.assert_called_once_with(
            transformer_checkpoint,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map={"": torch.device("cpu")},
        )

    def test_missing_standalone_transformer_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config.models.nvarc_checkpoint.mkdir()
            config.models.flux_checkpoint.mkdir()
            missing_transformer = root / "missing-transformer"

            with patch(
                "diffusion_arc.training.single_gpu_setup."
                "AutoTokenizer.from_pretrained"
            ) as load_tokenizer:
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "standalone FLUX transformer",
                ):
                    load_single_gpu_models(
                        config,
                        device=torch.device("cpu"),
                        transformer_checkpoint=missing_transformer,
                    )

        load_tokenizer.assert_not_called()

    def test_builds_unbatched_dataloader_and_independent_generators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
            samples = tuple(object() for _ in range(8))
            with patch(
                "diffusion_arc.training.single_gpu_setup.ArcTrainingDataset",
                return_value=samples,
            ) as build_dataset:
                loader = build_single_sample_loader(
                    config,
                    tokenizer=_FakeTokenizer(),
                )
            flow_generator = build_flow_generator(
                config,
                device=torch.device("cpu"),
            )

        build_dataset.assert_called_once_with(
            split_directory=config.data.split_directory,
            view_name="train",
            tokenizer=ANY,
        )
        self.assertIsNone(loader.batch_size)
        self.assertEqual(tuple(loader), samples)
        self.assertIsNot(loader.generator, flow_generator)
        self.assertEqual(loader.generator.initial_seed(), config.seed)
        self.assertEqual(flow_generator.initial_seed(), config.seed)

    def test_seeded_shuffle_repeats_the_same_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory), shuffle=True)
            samples = tuple(object() for _ in range(8))
            with patch(
                "diffusion_arc.training.single_gpu_setup.ArcTrainingDataset",
                return_value=samples,
            ):
                first_loader = build_single_sample_loader(
                    config,
                    tokenizer=_FakeTokenizer(),
                )
                second_loader = build_single_sample_loader(
                    config,
                    tokenizer=_FakeTokenizer(),
                )

        first_order = tuple(first_loader)
        second_order = tuple(second_loader)
        self.assertEqual(first_order, second_order)
        self.assertNotEqual(first_order, samples)

    def test_builds_adamw_over_exactly_the_transformer_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
        transformer = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

        optimizer = build_flux_optimizer(config, transformer=transformer)

        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimized, {id(parameter) for parameter in transformer.parameters()})
        self.assertEqual(optimizer.defaults["lr"], 1e-5)
        self.assertEqual(optimizer.defaults["betas"], (0.8, 0.9))
        self.assertEqual(optimizer.defaults["eps"], 1e-7)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.02)
        self.assertFalse(optimizer.defaults["foreach"])

    def test_adamw_preserves_small_updates_in_fp32_parameters_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
        transformer = nn.Linear(1, 1, bias=False, dtype=torch.float32)
        with torch.no_grad():
            transformer.weight.fill_(0.02)
        optimizer = build_flux_optimizer(config, transformer=transformer)
        before = transformer.weight.detach().clone()
        transformer.weight.grad = torch.ones_like(transformer.weight)

        optimizer.step()

        state = optimizer.state[transformer.weight]
        self.assertEqual(transformer.weight.dtype, torch.float32)
        self.assertEqual(transformer.weight.grad.dtype, torch.float32)
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        self.assertFalse(torch.equal(transformer.weight.detach(), before))

    def test_prepares_canonical_cuda_device_and_rejects_missing_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))

        with patch(
            "diffusion_arc.training.single_gpu_setup.torch.cuda.is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires an available CUDA"):
                prepare_single_gpu_device(config)

        with (
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.set_device"
            ) as set_device,
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.current_device",
                return_value=2,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup."
                "torch.cuda.is_bf16_supported",
                return_value=True,
            ),
        ):
            device = prepare_single_gpu_device(config)

        set_device.assert_not_called()
        self.assertEqual(device, torch.device("cuda:2"))

        indexed_config = replace(config, device="cuda:1")
        with (
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.set_device"
            ) as set_indexed_device,
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.current_device",
                return_value=1,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup."
                "torch.cuda.is_bf16_supported",
                return_value=True,
            ),
        ):
            indexed_device = prepare_single_gpu_device(indexed_config)

        set_indexed_device.assert_called_once_with(torch.device("cuda:1"))
        self.assertEqual(indexed_device, torch.device("cuda:1"))

    def test_rejects_cuda_without_required_bfloat16_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config = _config(Path(temporary_directory))
        with (
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup.torch.cuda.current_device",
                return_value=0,
            ),
            patch(
                "diffusion_arc.training.single_gpu_setup."
                "torch.cuda.is_bf16_supported",
                return_value=False,
            ) as supports_bfloat16,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not support bfloat16"):
                prepare_single_gpu_device(config)

        supports_bfloat16.assert_called_once_with()

    def test_rejects_mixed_parameter_placement(self) -> None:
        model = nn.Module()
        model.first = nn.Parameter(torch.ones((), dtype=torch.float32))
        model.second = nn.Parameter(torch.ones((), dtype=torch.bfloat16))

        with self.assertRaisesRegex(RuntimeError, "must use only"):
            _validate_parameter_placement(
                model,
                name="mixed model",
                expected_device=torch.device("cpu"),
                expected_dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
