from __future__ import annotations

import unittest

import torch

from diffusion_arc.arc_data import (
    ARC_PALETTE_RGB,
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcGeometry,
    ArcPromptTarget,
    ArcRenderedTrainingSample,
    render_grid_canvas,
    render_training_target_canvases,
)


def normalized_color(color_index: int) -> torch.Tensor:
    return ARC_PALETTE_RGB[color_index].float().div(127.5).sub(1.0)


class ArcCanvasTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
