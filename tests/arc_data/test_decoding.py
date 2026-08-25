"""Tests for deterministic generated RGB canvas decoding."""

from __future__ import annotations

import unittest

import torch

from diffusion_arc.arc_data import (
    CANVAS_SIZE,
    CELL_SIZE,
    PAD_CLASS,
    TARGET_PALETTE_RGB,
    classify_arc_cells,
    decode_arc_grid,
    decode_arc_grids,
    decode_fixed_arc_grids,
)


def _render_class_canvases(class_canvases: torch.Tensor) -> torch.Tensor:
    rgb = TARGET_PALETTE_RGB[class_canvases]
    rgb = rgb.repeat_interleave(CELL_SIZE, dim=1).repeat_interleave(
        CELL_SIZE,
        dim=2,
    )
    return rgb.permute(0, 3, 1, 2).contiguous()


class ArcDecodingTests(unittest.TestCase):
    def test_classifies_all_arc_and_pad_prototypes(self) -> None:
        values = torch.arange(CANVAS_SIZE * CANVAS_SIZE, dtype=torch.int64)
        class_canvas = values.remainder(PAD_CLASS + 1).reshape(
            CANVAS_SIZE,
            CANVAS_SIZE,
        )
        rgb_images = _render_class_canvases(class_canvas.unsqueeze(0))

        result = classify_arc_cells(rgb_images)

        torch.testing.assert_close(result, class_canvas.unsqueeze(0))

    def test_nearest_palette_tie_uses_lowest_palette_index(self) -> None:
        class_canvas = torch.full(
            (1, CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        rgb_images = _render_class_canvases(class_canvas)
        rgb_images[0, :, :CELL_SIZE, :CELL_SIZE] = torch.tensor(
            (0, 0, 85),
            dtype=torch.uint8,
        ).reshape(3, 1, 1)

        result = classify_arc_cells(rgb_images)

        self.assertEqual(result[0, 0, 0].item(), 0)

    def test_classifies_and_decodes_known_full_compact_grid(self) -> None:
        compact = torch.tensor(
            (((1, 2, 3), (4, 5, 6), (7, 8, 9)),),
            dtype=torch.int64,
        )
        rgb_images = _render_class_canvases(compact)

        classified = classify_arc_cells(
            rgb_images,
            canvas_size=3,
            include_pad=False,
        )
        prediction = decode_fixed_arc_grids(
            classified,
            canvas_size=3,
        )[0]

        torch.testing.assert_close(classified, compact)
        self.assertTrue(prediction.valid)
        self.assertEqual((prediction.height, prediction.width), (3, 3))
        self.assertEqual(
            prediction.grid,
            ((1, 2, 3), (4, 5, 6), (7, 8, 9)),
        )

    def test_decodes_valid_top_left_grid_from_pad_boundary(self) -> None:
        canvas = torch.full(
            (CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        canvas[:2, :3] = torch.tensor(((1, 2, 3), (4, 5, 6)))

        prediction = decode_arc_grid(canvas)

        self.assertTrue(prediction.valid)
        self.assertIsNone(prediction.invalid_reason)
        self.assertEqual((prediction.height, prediction.width), (2, 3))
        self.assertEqual(prediction.grid, ((1, 2, 3), (4, 5, 6)))

    def test_rejects_all_pad_and_pad_inside_inferred_extent(self) -> None:
        all_pad = torch.full(
            (CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        interior_pad = all_pad.clone()
        interior_pad[0, 0] = 1
        interior_pad[1, 1] = 2

        all_pad_prediction, interior_prediction = decode_arc_grids(
            torch.stack((all_pad, interior_pad))
        )

        self.assertFalse(all_pad_prediction.valid)
        self.assertEqual(all_pad_prediction.invalid_reason, "all_pad")
        self.assertIsNone(all_pad_prediction.grid)
        self.assertEqual(
            (all_pad_prediction.height, all_pad_prediction.width),
            (None, None),
        )

        self.assertFalse(interior_prediction.valid)
        self.assertEqual(interior_prediction.invalid_reason, "pad_inside_extent")
        self.assertEqual(
            (interior_prediction.height, interior_prediction.width),
            (2, 2),
        )
        self.assertIsNone(interior_prediction.grid)

    def test_stray_non_pad_cell_expands_extent_without_fallback(self) -> None:
        canvas = torch.full(
            (CANVAS_SIZE, CANVAS_SIZE),
            PAD_CLASS,
            dtype=torch.int64,
        )
        canvas[:2, :2] = 0
        canvas[5, 5] = 1

        prediction = decode_arc_grid(canvas)

        self.assertFalse(prediction.valid)
        self.assertEqual(prediction.invalid_reason, "pad_inside_extent")
        self.assertEqual((prediction.height, prediction.width), (6, 6))
        self.assertIsNone(prediction.grid)

    def test_rejects_non_uint8_images_and_invalid_classes(self) -> None:
        with self.assertRaises(TypeError):
            classify_arc_cells(
                torch.zeros(
                    (1, 3, CANVAS_SIZE * CELL_SIZE, CANVAS_SIZE * CELL_SIZE),
                    dtype=torch.float32,
                )
            )
        with self.assertRaises(ValueError):
            decode_arc_grid(
                torch.full(
                    (CANVAS_SIZE, CANVAS_SIZE),
                    PAD_CLASS + 1,
                    dtype=torch.int64,
                )
            )


if __name__ == "__main__":
    unittest.main()
