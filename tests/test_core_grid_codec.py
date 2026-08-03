from __future__ import annotations

import unittest

import torch

from diffusion_arc.grid_codec import (
    ARC_PALETTE_RGB,
    CANVAS_SIZE,
    PAD_CLASS,
    decode_arc_canvas,
    decode_arc_image,
    pad_arc_grids,
    render_arc_canvas,
)


class GridCodecTests(unittest.TestCase):
    def test_top_left_padding_and_palette_rendering(self) -> None:
        grid = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        canvas = pad_arc_grids(grid)

        self.assertEqual(canvas.shape, (1, 30, 30))
        torch.testing.assert_close(canvas[0, :2, :2], grid)
        self.assertTrue(bool((canvas[0, 2:, :] == PAD_CLASS).all()))
        self.assertTrue(bool((canvas[0, :2, 2:] == PAD_CLASS).all()))

        image = render_arc_canvas(grid)
        self.assertEqual(image.shape, (1, 3, 480, 480))
        expected_black = ARC_PALETTE_RGB[0].float().div(127.5).sub(1.0)
        expected_blue = ARC_PALETTE_RGB[1].float().div(127.5).sub(1.0)
        expected_white = ARC_PALETTE_RGB[PAD_CLASS].float().div(127.5).sub(1.0)
        torch.testing.assert_close(image[0, :, 0, 0], expected_black)
        torch.testing.assert_close(image[0, :, 0, 16], expected_blue)
        torch.testing.assert_close(image[0, :, 32, 32], expected_white)

    def test_exact_render_decode_round_trip(self) -> None:
        grid = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        result = decode_arc_image(render_arc_canvas(grid))
        self.assertTrue(result.valid)
        self.assertIsNone(result.reason)
        self.assertEqual((result.height, result.width), (2, 3))
        self.assertIsNotNone(result.grid)
        torch.testing.assert_close(result.grid, grid)

    def test_all_pad_is_invalid(self) -> None:
        canvas = torch.full(
            (CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        result = decode_arc_canvas(canvas)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "all_pad")
        self.assertIsNone(result.grid)
        self.assertIsNone(result.height)
        self.assertIsNone(result.width)

    def test_pad_inside_inferred_extent_is_invalid_without_repair(self) -> None:
        canvas = torch.full(
            (CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        canvas[0, 0] = 1
        canvas[1, 1] = 2
        result = decode_arc_canvas(canvas)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "pad_inside_extent")
        self.assertEqual((result.height, result.width), (2, 2))
        self.assertIsNone(result.grid)

    def test_variable_size_batch_is_supported(self) -> None:
        canvases = pad_arc_grids([[[1]], [[2, 3], [4, 5]]])
        self.assertEqual(canvases.shape, (2, 30, 30))
        self.assertEqual(canvases[0, 0, 0].item(), 1)
        self.assertEqual(canvases[1, 1, 1].item(), 5)

    def test_rejects_pad_as_ground_truth_color(self) -> None:
        with self.assertRaises(ValueError):
            render_arc_canvas(torch.tensor([[PAD_CLASS]]))

    def test_rejects_empty_tensor_batch(self) -> None:
        with self.assertRaises(ValueError):
            pad_arc_grids(torch.empty((0, 1, 1), dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
