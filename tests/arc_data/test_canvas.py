from __future__ import annotations

import unittest

import torch

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcGeometry,
    ArcPromptTarget,
    ArcRenderedTrainingSample,
    PAD_CLASS,
    TARGET_PALETTE_ID,
    TARGET_PALETTE_RGB,
    render_grid_canvas,
    render_training_target_canvases,
)


def normalized_color(color_index: int) -> torch.Tensor:
    return TARGET_PALETTE_RGB[color_index].float().div(127.5).sub(1.0)


class ArcCanvasTest(unittest.TestCase):
    def test_uses_fixed_greedy_rgb_maximin_palette(self) -> None:
        self.assertEqual(TARGET_PALETTE_ID, "greedy_rgb_maximin_v1")
        self.assertEqual(PAD_CLASS, 10)
        torch.testing.assert_close(
            TARGET_PALETTE_RGB,
            torch.tensor(
                [
                    (0x00, 0x00, 0x00),
                    (0x00, 0x55, 0xFF),
                    (0x55, 0xFF, 0x00),
                    (0xFF, 0x00, 0x55),
                    (0x00, 0xFF, 0xAA),
                    (0xAA, 0x00, 0xFF),
                    (0xFF, 0xAA, 0x00),
                    (0xAA, 0xAA, 0xAA),
                    (0x55, 0x55, 0x55),
                    (0x00, 0x00, 0xAA),
                    (0xFF, 0xFF, 0xFF),
                ],
                dtype=torch.uint8,
            ),
        )

    def test_grid_uses_fixed_palette_and_white_padding(self) -> None:
        image = render_grid_canvas(((0, 1), (2, 3)))

        self.assertEqual(image.shape, (3, 480, 480))
        self.assertEqual(image.dtype, torch.float32)
        torch.testing.assert_close(image[:, 0, 0], normalized_color(0))
        torch.testing.assert_close(image[:, 0, 16], normalized_color(1))
        torch.testing.assert_close(image[:, 16, 0], normalized_color(2))
        torch.testing.assert_close(image[:, 16, 16], normalized_color(3))
        torch.testing.assert_close(image[:, 32, 32], normalized_color(10))

    def test_training_targets_are_rendered_in_prompt_cut_order(self) -> None:
        augmentation = ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=(0, 1, 2),
        )
        sample = ArcRenderedTrainingSample(
            key=ArcAugmentationKey("training-task", augmentation),
            prompt="prompt",
            targets=(
                ArcPromptTarget(prefix_character_end=1, target_grid=((1,),)),
                ArcPromptTarget(prefix_character_end=2, target_grid=((2,),)),
            ),
        )

        images = render_training_target_canvases(sample)

        self.assertEqual(images.shape, (2, 3, 480, 480))
        torch.testing.assert_close(images[0, :, 0, 0], normalized_color(1))
        torch.testing.assert_close(images[1, :, 0, 0], normalized_color(2))

    def test_rejects_grids_larger_than_the_canvas(self) -> None:
        oversized_grids = (
            tuple((0,) for _ in range(31)),
            (tuple(0 for _ in range(31)),),
        )

        for grid in oversized_grids:
            with self.subTest(shape=(len(grid), len(grid[0]))), self.assertRaises(
                ValueError
            ):
                render_grid_canvas(grid)

    def test_accepts_a_grid_at_the_thirty_by_thirty_boundary(self) -> None:
        grid = tuple(tuple(0 for _ in range(30)) for _ in range(30))

        image = render_grid_canvas(grid)

        self.assertEqual(image.shape, (3, 480, 480))


if __name__ == "__main__":
    unittest.main()
