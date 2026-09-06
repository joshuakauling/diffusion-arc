"""Tests for the initial consumer-first single-GPU training policy."""

from __future__ import annotations

import unittest
from dataclasses import replace
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
    NvarcBatchRunner,
    NvarcBatchOutput,
    TARGET_LATENT_CHANNELS,
    TARGET_LATENT_TOKEN_COUNT,
)
from diffusion_arc.training import run_single_gpu_training
from diffusion_arc.sequence_buckets import SequenceLengthBuckets
from tests.model.test_nvarc_encoder import FakeNvarc


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
        offsets = sequence_offsets.tolist()
        return NvarcBatchOutput(
            sequence_hidden_states=tuple(
                hidden_states[start:end].clone()
                for start, end in zip(offsets, offsets[1:])
            )
        )


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
    def test_b1_preserves_two_optimizer_batches_and_per_target_loss_weights(self) -> None:
        samples = []
        for index in range(32):
            sample = _sample(4)
            ends = tuple(end + index % 3 for end in sample.prefix_token_ends)
            samples.append(replace(
                sample,
                key=replace(sample.key, task_id=f"{index:08x}"),
                input_ids=(index % 10,) * (ends[-1] + 1),
                prefix_token_ends=ends,
                target_grids=tuple(((index // 10, index % 10, target),) for target in range(4)),
            ))

        def run(source_batch, vae_batch, nvarc_bounds, flux_bounds):
            events, targets, prefixes, padded_lengths, weights, steps = [], [], [], [], [], []

            class Nvarc(FakeNvarc):
                def forward(self, **kwargs):
                    events.append(("nvarc", kwargs["input_ids"].shape[0]))
                    return super().forward(**kwargs)

            class Vae(nn.Module):
                def forward(self, images):
                    events.append(("vae", images.shape[0]))
                    return images[:, 0, 0, 0, None, None].expand(
                        -1, TARGET_LATENT_TOKEN_COUNT, TARGET_LATENT_CHANNELS,
                    ).clone()

            def loss(model, microbatch, **_kwargs):
                target = int(microbatch.clean_latents[0, 0, 0])
                targets.append(target)
                prefixes.append(int(microbatch.text_inputs.text_attention_mask.sum()))
                padded_lengths.append(microbatch.text_inputs.encoder_hidden_states.shape[1])
                events.append(("flux", target))
                value = model.weight * microbatch.clean_latents[:, 0, 0].mean()
                value.register_hook(lambda grad: weights.append(float(grad)))
                return value

            def on_step(loss, target_count, microbatch_count):
                steps.append((tuple(targets[-48:]), float(loss), float(model.weight.grad),
                              target_count, microbatch_count))
                events.append(("optimizer", len(steps)))

            def render(grid):
                # Keep the pipeline test small while identifying every source/target pair.
                tens, ones, target = grid[0]
                return torch.tensor([[[float(30 * tens + 3 * ones + target)]]])

            model = ScalarTransformer()
            with (
                patch("diffusion_arc.training.flux_input._render_valid_grid_canvas", side_effect=render),
                patch("diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss", side_effect=loss),
            ):
                run_single_gpu_training(
                    samples,
                    nvarc_runner=NvarcBatchRunner(Nvarc(), length_buckets=SequenceLengthBuckets(nvarc_bounds)),
                    vae_encoder=Vae(), transformer=model,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                    flux_compute_dtype=torch.bfloat16, max_optimizer_steps=2,
                    optimizer_batch_size=48, flux_microbatch_size=1,
                    min_context_pairs_before_target=2,
                    nvarc_source_batch_size=source_batch, vae_microbatch_size=vae_batch,
                    flux_bucket_boundaries=flux_bounds, on_optimizer_step=on_step,
                )
            return events, targets, prefixes, padded_lengths, weights, steps, model.weight.detach()

        b0 = run(1, 1, None, None)
        b1 = run(16, 48, (16,), (8, 16))
        self.assertEqual(b1[1], list(range(1, 97)))
        self.assertEqual(b1[1:3], b0[1:3])
        self.assertEqual(b1[3], [8 if length <= 8 else 16 for length in b1[2]])
        self.assertEqual(b0[3], b0[2])
        self.assertEqual(b1[5], b0[5])
        self.assertEqual([row[0] for row in b1[5]], [tuple(range(1, 49)), tuple(range(49, 97))])
        for step, expected_gradient in zip(b1[5], (24.5, 72.5)):
            self.assertAlmostEqual(step[2], expected_gradient, places=4)
            self.assertEqual(step[3:], (48, 48))
        self.assertEqual(len(b1[4]), 96)
        for weight in b1[4]:
            self.assertAlmostEqual(weight, 1 / 48, places=8)
        torch.testing.assert_close(b1[6], b0[6])
        torch.testing.assert_close(b1[6], torch.tensor(0.03), atol=1e-6, rtol=1e-6)
        expected_events = []
        for step in range(2):
            expected_events.extend([("nvarc", 16), ("vae", 48)])
            expected_events.extend(("flux", target) for target in range(48 * step + 1, 48 * step + 49))
            expected_events.append(("optimizer", step + 1))
        self.assertEqual(b1[0], expected_events)

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

    def test_batches_eight_sources_and_uses_bounded_vae_batches(self) -> None:
        nvarc = RecordingNvarcRunner()
        vae = NumberedVaeEncoder()
        transformer = ScalarTransformer()
        optimizer = torch.optim.SGD(transformer.parameters(), lr=0.0)
        recording_loss = RecordingLoss()

        with patch(
            "diffusion_arc.training.single_gpu_loop.compute_flux_microbatch_loss",
            side_effect=recording_loss,
        ):
            run_single_gpu_training(
                tuple(_sample(1) for _ in range(10)),
                nvarc_runner=nvarc,  # type: ignore[arg-type]
                vae_encoder=vae,  # type: ignore[arg-type]
                transformer=transformer,
                optimizer=optimizer,
                flux_compute_dtype=torch.bfloat16,
                max_optimizer_steps=3,
                optimizer_batch_size=4,
                flux_microbatch_size=2,
                nvarc_source_batch_size=8,
                vae_microbatch_size=8,
            )

        self.assertEqual(len(nvarc.calls), 2)
        self.assertEqual(nvarc.calls[0][1].tolist(), list(range(0, 17, 2)))
        self.assertEqual(nvarc.calls[1][1].tolist(), [0, 2, 4])
        self.assertEqual([call.shape[0] for call in vae.calls], [8, 2])
        self.assertEqual(
            recording_loss.target_values,
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 10.0)],
        )

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
            "nvarc_source_batch_size": 1,
            "vae_microbatch_size": 1,
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
