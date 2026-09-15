"""Numerical checks for region weighting, clipped boundaries, and accumulation."""

import unittest

import torch

from diffusion_arc.model.contracts import TARGET_LATENT_SHAPE
from diffusion_arc.training.step.loss import (
    RegionLossWeights,
    flow_matching_loss,
    build_region_loss_reduction,
    compute_region_velocity_loss,
)


SHAPES = ((1, 1), (2, 7), (7, 2), (29, 29), (30, 29), (29, 30), (30, 30))


class RegionLossTests(unittest.TestCase):
    def test_hand_computed_region_means_and_contributions(self):
        residuals, expected_means, expected_losses = [], [], []
        for height, width in SHAPES:
            # Distinct integer residuals make the expected MSE analytical.
            residual = torch.full((30, 30), 5.0)
            extent_h, extent_w = min(height + 1, 30), min(width + 1, 30)
            residual[:extent_h, :extent_w] = 3.0
            residual[:height, :width] = 2.0
            residuals.append(residual.flatten()[:, None].expand(-1, 4))
            grid_count = height * width
            boundary_count = extent_h * extent_w - grid_count
            canvas = (4 * grid_count + 9 * boundary_count
                      + 25 * (900 - grid_count - boundary_count)) / 900
            expected_means.append((canvas, 4.0, 9.0 if boundary_count else 0.0))
            numerator = 0.1 * canvas + 0.8 * 4 + (0.1 * 9 if boundary_count else 0.0)
            expected_losses.append(numerator / (1.0 if boundary_count else 0.9))

        predictions = torch.stack(residuals)
        result = compute_region_velocity_loss(predictions, torch.zeros_like(predictions), SHAPES)
        torch.testing.assert_close(result.region_means, torch.tensor(expected_means))
        torch.testing.assert_close(result.per_target, torch.tensor(expected_losses))
        torch.testing.assert_close(result.contributions.sum(dim=-1), result.per_target)
        torch.testing.assert_close(result.has_boundary, torch.tensor([True] * 6 + [False]))
        self.assertEqual(result.contributions[-1, 2].item(), 0.0)
        self.assertEqual(result.region_means[-1, 2].item(), 0.0)
        torch.testing.assert_close(result.contributions[-1, :2], torch.tensor([4 / 9, 32 / 9]))

    def test_constant_error_has_same_target_weight_for_all_shapes(self):
        predictions = torch.full((len(SHAPES), 900, 3), 2.0)
        for weights in (RegionLossWeights(), RegionLossWeights(.4, .1, .5),
                        RegionLossWeights(1, 0, 0), RegionLossWeights(0, 1, 0)):
            with self.subTest(weights=weights):
                result = compute_region_velocity_loss(predictions, torch.zeros_like(predictions),
                                              SHAPES, weights=weights)
                torch.testing.assert_close(result.per_target, torch.full((len(SHAPES),), 4.0))

    def test_boundary_contains_exactly_bottom_right_and_corner_cells(self):
        reduction = build_region_loss_reduction(SHAPES, device="cpu")
        for shape, masks in zip(SHAPES, reduction.masks, strict=True):
            with self.subTest(shape=shape):
                height, width = shape
                expected = torch.zeros((30, 30), dtype=torch.bool)
                if height < 30:
                    expected[height, :min(width + 1, 30)] = True
                if width < 30:
                    expected[:height, width] = True
                torch.testing.assert_close(masks[2].reshape(30, 30) > 0, expected)
                torch.testing.assert_close(masks[:2].sum(dim=-1), torch.ones(2))

    def test_subtraction_and_reductions_use_fp32(self):
        prediction = torch.full((1, 900, 3), 101.0, dtype=torch.bfloat16)
        target = torch.full_like(prediction, -100.0)
        result = compute_region_velocity_loss(prediction, target, ((3, 4),))
        expected = (prediction.float() - target.float()).square().mean()
        for tensor in (result.per_target, result.region_means, result.contributions):
            self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(result.per_target, expected[None])

    def test_unequal_microbatches_preserve_loss_and_parameter_gradient(self):
        generator = torch.Generator().manual_seed(41)
        inputs = torch.randn((len(SHAPES), 900, 4), generator=generator)
        targets = torch.randn(inputs.shape, generator=generator)
        parameter = torch.tensor((.3, .7, -.1, 1.2), requires_grad=True)
        reference = compute_region_velocity_loss(inputs * parameter, targets, SHAPES).per_target.mean()
        reference.backward()
        expected_gradient = parameter.grad.clone()
        parameter.grad = None

        accumulated = 0.0
        # Reordered packs of three, one, and three targets exercise target weighting.
        for indices in ((5, 1, 4), (6,), (0, 3, 2)):
            result = compute_region_velocity_loss(inputs[list(indices)] * parameter,
                                          targets[list(indices)], tuple(SHAPES[i] for i in indices))
            contribution = result.per_target.mean() * (len(indices) / len(SHAPES))
            contribution.backward()
            accumulated += contribution.detach()
        torch.testing.assert_close(accumulated, reference.detach())
        torch.testing.assert_close(parameter.grad, expected_gradient)

    def test_reusable_reduction_matches_convenience_api_and_gradients(self):
        predictions = torch.randn((2, 900, 2), generator=torch.Generator().manual_seed(6),
                                  requires_grad=True)
        targets = torch.zeros_like(predictions)
        reduction = build_region_loss_reduction(SHAPES[:2], device="cpu")
        direct = compute_region_velocity_loss(predictions, targets, SHAPES[:2])
        prepared = reduction.compute(predictions, targets)
        torch.testing.assert_close(prepared.per_target, direct.per_target, atol=0, rtol=0)
        direct_grad = torch.autograd.grad(direct.per_target.sum(), predictions)[0]
        prepared_grad = torch.autograd.grad(prepared.per_target.sum(), predictions)[0]
        torch.testing.assert_close(prepared_grad, direct_grad, atol=0, rtol=0)

    def test_invalid_weights_and_shapes_fail_at_preparation(self):
        for weights in ((-.1, 1, .1), (float("nan"), .8, .1), (float("inf"), 0, 0),
                        (True, 0, 0), (0, 0, 1), (.1, .7, .1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                RegionLossWeights(*weights)
        for shapes in ((), ((0, 1),), ((31, 1),), ((1, True),), ((1, 1.5),), ((3,),)):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                build_region_loss_reduction(shapes, device="cpu")

    def test_shape_contract_prevents_broadcasting_and_nonspatial_tokens(self):
        reduction = build_region_loss_reduction(((1, 1),), device="cpu")
        for prediction, target in (
            (torch.zeros((1, 899, 2)), torch.zeros((1, 899, 2))),
            (torch.zeros((1, 900, 2)), torch.zeros((1, 900, 1))),
            (torch.zeros((1, 900, 0)), torch.zeros((1, 900, 0))),
            (torch.zeros((1, 900, 2), dtype=torch.int64), torch.zeros((1, 900, 2))),
        ):
            with self.subTest(shape=prediction.shape, dtype=prediction.dtype), self.assertRaises(ValueError):
                reduction.compute(prediction, target)


class PlainFlowLossTests(unittest.TestCase):
    def test_loss_is_unweighted_float32_mse_and_backpropagates(self) -> None:
        predicted_velocity = torch.empty(
            (2, *TARGET_LATENT_SHAPE),
            dtype=torch.bfloat16,
        )
        predicted_velocity[0].fill_(1)
        predicted_velocity[1].fill_(3)
        predicted_velocity.requires_grad_()
        target_velocity = torch.zeros(
            (2, *TARGET_LATENT_SHAPE),
            dtype=torch.float32,
        )

        loss = flow_matching_loss(predicted_velocity, target_velocity)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 5.0)
        loss.backward()
        self.assertIsNotNone(predicted_velocity.grad)
        self.assertTrue(bool(torch.isfinite(predicted_velocity.grad).all()))

    def test_loss_rejects_mismatched_target_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            flow_matching_loss(
                torch.zeros((2, *TARGET_LATENT_SHAPE)),
                torch.zeros((1, *TARGET_LATENT_SHAPE)),
            )


if __name__ == "__main__":
    unittest.main()
