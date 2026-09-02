"""Tests for the initial consumer-first single-GPU training policy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn

from diffusion_arc.arc_data.augmentation import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcGeometry,
)
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model import (
    NvarcBatchOutput,
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)
from diffusion_arc.training import run_single_gpu_training


class RecordingNvarcRunner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []

    def forward(
        self,
        input_ids: Tensor,
        sequence_offsets: Tensor,
    ) -> NvarcBatchOutput:
        self.calls.append((input_ids.clone(), sequence_offsets.clone()))
        hidden_states = torch.arange(
            input_ids.numel() * 4,
            dtype=torch.float32,
        ).reshape(input_ids.numel(), 4)
        return NvarcBatchOutput(sequence_hidden_states=(hidden_states,))


class NumberedVaeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Tensor] = []
        self.next_target_value = 1

    def forward(self, target_images: Tensor) -> Tensor:
        self.calls.append(target_images.clone())
        clean_latents = []
        for _ in range(target_images.shape[0]):
            clean_latents.append(
                torch.full(
                    (TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS),
                    float(self.next_target_value),
                )
            )
            self.next_target_value += 1
        return torch.stack(clean_latents)


class ScalarTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class RecordingLoss:
    def __init__(self) -> None:
        self.target_values: list[tuple[float, ...]] = []
        self.generators: list[torch.Generator | None] = []

    def __call__(
        self,
        transformer: ScalarTransformer,
        microbatch,
        *,
        compute_dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if compute_dtype != torch.bfloat16:
            raise AssertionError("single-GPU tests require BF16 compute")
        values = microbatch.clean_latents[:, 0, 0]
        self.target_values.append(tuple(float(value) for value in values))
        self.generators.append(generator)
        return transformer.weight * values.mean()


def _sample(target_count: int) -> ArcTokenizedTrainingSample:
    prefix_ends = tuple(2 * (index + 1) for index in range(target_count))
    return ArcTokenizedTrainingSample(
        key=ArcAugmentationKey(
            task_id="abc123ef",
            augmentation=ArcAugmentationSpec(
                geometry=ArcGeometry.IDENTITY,
                color_mapping=tuple(range(10)),
                pair_indices=tuple(range(target_count + 1)),
            ),
        ),
        input_ids=tuple(range(prefix_ends[-1] + 1)),
        prefix_token_ends=prefix_ends,
        target_grids=tuple(((index,),) for index in range(target_count)),
    )


class SingleGpuTrainingTests(unittest.TestCase):
    def test_scales_fifo_microbatches_and_flushes_stream_tail(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer().eval()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)
        generator = torch.Generator().manual_seed(17)
        updates: list[tuple[Tensor, int, int]] = []
        recording_loss = RecordingLoss()

        with patch(
            "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
            side_effect=recording_loss,
        ):
            run_single_gpu_training(
                (_sample(2), _sample(3)),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=2,
                optimizer_batch_size=4,
                flux_microbatch_size=2,
                generator=generator,
                on_optimizer_step=lambda loss, targets, microbatches: updates.append(
                    (loss, targets, microbatches)
                ),
            )

        self.assertTrue(transformer.training)
        self.assertEqual(len(nvarc.calls), 2)
        self.assertEqual(len(vae.calls), 2)
        self.assertEqual(
            recording_loss.target_values,
            [(1.0, 2.0), (3.0, 4.0), (5.0,)],
        )
        self.assertTrue(all(value is generator for value in recording_loss.generators))
        torch.testing.assert_close(
            transformer.weight.detach(),
            torch.tensor(-6.5),
        )

        self.assertEqual([(targets, batches) for _, targets, batches in updates], [(4, 2), (1, 1)])
        torch.testing.assert_close(updates[0][0], torch.tensor(2.5))
        torch.testing.assert_close(updates[1][0], torch.tensor(-7.5))
        self.assertTrue(all(loss.grad_fn is None for loss, _, _ in updates))

    def test_stops_exactly_at_step_limit_without_draining_buffer(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)
        updates: list[tuple[Tensor, int, int]] = []
        recording_loss = RecordingLoss()

        with patch(
            "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
            side_effect=recording_loss,
        ):
            run_single_gpu_training(
                (_sample(5), _sample(1)),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=1,
                optimizer_batch_size=2,
                flux_microbatch_size=1,
                on_optimizer_step=lambda loss, targets, microbatches: updates.append(
                    (loss, targets, microbatches)
                ),
            )

        self.assertEqual(len(nvarc.calls), 1)
        self.assertEqual(len(vae.calls), 1)
        self.assertEqual(recording_loss.target_values, [(1.0,), (2.0,)])
        torch.testing.assert_close(
            transformer.weight.detach(),
            torch.tensor(-0.5),
        )
        self.assertEqual(len(updates), 1)
        torch.testing.assert_close(updates[0][0], torch.tensor(1.5))
        self.assertEqual(updates[0][1:], (2, 2))

    def test_reports_and_clips_accumulated_gradient_before_update(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)
        metrics = []
        recording_loss = RecordingLoss()

        with patch(
            "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
            side_effect=recording_loss,
        ):
            run_single_gpu_training(
                (_sample(2),),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=1,
                optimizer_batch_size=2,
                flux_microbatch_size=1,
                max_grad_norm=0.5,
                on_optimizer_step_metrics=metrics.append,
            )

        self.assertEqual(len(metrics), 1)
        step = metrics[0]
        torch.testing.assert_close(step.loss, torch.tensor(1.5))
        self.assertEqual(step.target_count, 2)
        self.assertEqual(step.microbatch_count, 2)
        self.assertAlmostEqual(step.gradient_norm, 1.5)
        self.assertTrue(step.gradient_clipped)
        self.assertEqual(step.learning_rates, (1.0,))
        self.assertGreaterEqual(step.step_seconds, 0.0)
        torch.testing.assert_close(transformer.weight.detach(), torch.tensor(0.5))

    def test_rejects_non_finite_loss_before_optimizer_update(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)

        with (
            patch(
                "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
                return_value=transformer.weight * torch.tensor(float("nan")),
            ),
            self.assertRaisesRegex(FloatingPointError, "non-finite"),
        ):
            run_single_gpu_training(
                (_sample(1),),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=1,
                optimizer_batch_size=1,
            )

        torch.testing.assert_close(transformer.weight.detach(), torch.tensor(1.0))

    def test_rejects_non_finite_accumulated_gradient_before_update(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        with torch.no_grad():
            transformer.weight.zero_()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)

        with (
            patch(
                "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
                side_effect=lambda model, *_args, **_kwargs: torch.sqrt(model.weight),
            ),
            self.assertRaisesRegex(RuntimeError, "non-finite"),
        ):
            run_single_gpu_training(
                (_sample(1),),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=1,
                optimizer_batch_size=1,
                max_grad_norm=1.0,
            )

        torch.testing.assert_close(transformer.weight.detach(), torch.tensor(0.0))

    def test_rejects_invalid_run_configuration_before_model_work(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=1.0)
        base_arguments = {
            "max_optimizer_steps": 1,
            "optimizer_batch_size": 1,
            "flux_microbatch_size": 1,
            "min_context_pairs_before_target": 1,
        }

        for name in base_arguments:
            arguments = dict(base_arguments)
            arguments[name] = 0
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_single_gpu_training(
                    (),
                    nvarc_runner=nvarc,  # type: ignore[arg-type]
                    vae_encoder=vae,  # type: ignore[arg-type]
                    transformer=transformer,
                    optimizer=optimizer,
                    flux_compute_dtype=torch.bfloat16,
                    **arguments,
                )

        with self.assertRaises(TypeError):
            run_single_gpu_training(
                (),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                generator=object(),  # type: ignore[arg-type]
                **base_arguments,
            )

        for invalid_norm in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(max_grad_norm=invalid_norm), self.assertRaises(ValueError):
                run_single_gpu_training(
                    (),
                    nvarc_runner=nvarc,  # type: ignore[arg-type]
                    vae_encoder=vae,  # type: ignore[arg-type]
                    transformer=transformer,
                    optimizer=optimizer,
                    flux_compute_dtype=torch.bfloat16,
                    max_grad_norm=invalid_norm,
                    **base_arguments,
                )

        self.assertEqual(nvarc.calls, [])
        self.assertEqual(vae.calls, [])


if __name__ == "__main__":
    unittest.main()
