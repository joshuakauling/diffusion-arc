from __future__ import annotations

import unittest

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcAugmentedTrainingSample,
    ArcEvaluationTask,
    ArcGeometry,
    ArcPair,
    ArcQuery,
    render_evaluation_prompts,
    render_training_prompt,
)


class ArcTrainingPromptTest(unittest.TestCase):
    def test_rendering_includes_all_outputs_and_records_prefix_cuts(self) -> None:
        augmentation = ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=(0, 1, 2),
        )
        sample = ArcAugmentedTrainingSample(
            key=ArcAugmentationKey("training-task", augmentation),
            pairs=(
                ArcPair(input_grid=((1, 2),), output_grid=((3,),)),
                ArcPair(input_grid=((4,),), output_grid=((5, 6),)),
                ArcPair(input_grid=((7,),), output_grid=((8,),)),
            ),
        )

        rendered = render_training_prompt(sample)

        first_target_prefix = (
            "<|im_start|>user\n12<|im_end|>"
            "<|im_start|>assistant\n3<|im_end|>"
            "<|im_start|>user\n4<|im_end|>"
            "<|im_start|>assistant\n"
        )
        second_target_prefix = (
            f"{first_target_prefix}56<|im_end|>"
            "<|im_start|>user\n7<|im_end|>"
            "<|im_start|>assistant\n"
        )
        expected_prompt = f"{second_target_prefix}8<|im_end|>"

        self.assertEqual(rendered.key, sample.key)
        self.assertEqual(rendered.prompt, expected_prompt)
        self.assertEqual(
            tuple(target.target_grid for target in rendered.targets),
            (((5, 6),), ((8,),)),
        )
        self.assertEqual(
            tuple(
                rendered.prompt[: target.prefix_character_end]
                for target in rendered.targets
            ),
            (first_target_prefix, second_target_prefix),
        )


class ArcEvaluationPromptTest(unittest.TestCase):
    def test_rendering_creates_one_target_free_prompt_per_query(self) -> None:
        task = ArcEvaluationTask(
            task_id="evaluation-task",
            demonstrations=(
                ArcPair(input_grid=((1, 2),), output_grid=((3,),)),
                ArcPair(input_grid=((4,),), output_grid=((5, 6),)),
            ),
            queries=(
                ArcQuery(query_index=0, input_grid=((7,),)),
                ArcQuery(query_index=1, input_grid=((8, 9),)),
            ),
        )

        rendered = render_evaluation_prompts(task)

        demonstration_prefix = (
            "<|im_start|>user\n12<|im_end|>"
            "<|im_start|>assistant\n3<|im_end|>"
            "<|im_start|>user\n4<|im_end|>"
            "<|im_start|>assistant\n56<|im_end|>"
        )
        self.assertEqual(
            tuple((sample.task_id, sample.query_index) for sample in rendered),
            (("evaluation-task", 0), ("evaluation-task", 1)),
        )
        self.assertEqual(
            tuple(sample.prompt for sample in rendered),
            (
                f"{demonstration_prefix}<|im_start|>user\n7<|im_end|>"
                "<|im_start|>assistant\n",
                f"{demonstration_prefix}<|im_start|>user\n89<|im_end|>"
                "<|im_start|>assistant\n",
            ),
        )


if __name__ == "__main__":
    unittest.main()
