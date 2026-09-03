from __future__ import annotations

import unittest

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from diffusion_arc.arc_data import (
    ArcAugmentationKey,
    ArcAugmentationSpec,
    ArcAugmentedTrainingSample,
    ArcEvaluationTask,
    ArcGeometry,
    ArcPair,
    ArcQuery,
    NVARC_MAX_TOKENS,
    render_evaluation_prompts,
    render_training_prompt,
    tokenize_nvarc_evaluation_sample,
    tokenize_nvarc_training_sample,
)


def make_nvarc_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        **{str(value): value for value in range(10)},
        "\n": 10,
        "user": 11,
        "assistant": 12,
        "<|endoftext|>": 13,
        "<|im_start|>": 14,
        "<|im_end|>": 15,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<|endoftext|>"))
    backend.pre_tokenizer = Split(
        Regex(r"(<\|im_start\|>|<\|im_end\|>|assistant|user|\n|[0-9])"),
        behavior="isolated",
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


class ArcNvarcTokenizationTest(unittest.TestCase):
    def test_tokenization_aligns_training_cuts_and_enforces_context_limit(self) -> None:
        augmentation = ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=(0, 1, 2),
        )
        augmented = ArcAugmentedTrainingSample(
            key=ArcAugmentationKey("training-task", augmentation),
            pairs=(
                ArcPair(input_grid=((1,),), output_grid=((2,),)),
                ArcPair(input_grid=((3,),), output_grid=((4,),)),
                ArcPair(input_grid=((5,),), output_grid=((6,),)),
            ),
        )
        rendered = render_training_prompt(augmented)
        tokenizer = make_nvarc_tokenizer()

        tokenized = tokenize_nvarc_training_sample(rendered, tokenizer, max_tokens=28)

        self.assertEqual(len(tokenizer), 16)
        self.assertEqual(tokenized.prefix_token_ends, (18, 28))
        self.assertEqual(tokenized.target_grids, (((4,),), ((6,),)))
        self.assertEqual(len(tokenized.input_ids), 30)

        with self.assertRaisesRegex(
            ValueError, "source prefix contains 28 tokens, exceeding 27"
        ):
            tokenize_nvarc_training_sample(rendered, tokenizer, max_tokens=27)

    def test_five_maximum_pairs_keep_full_prompt_and_fit_8400_tokens(self) -> None:
        grid = tuple(
            tuple((row + column) % 10 for column in range(30))
            for row in range(30)
        )
        augmentation = ArcAugmentationSpec(
            geometry=ArcGeometry.IDENTITY,
            color_mapping=tuple(range(10)),
            pair_indices=tuple(range(5)),
        )
        augmented = ArcAugmentedTrainingSample(
            key=ArcAugmentationKey("maximum-size-task", augmentation),
            pairs=tuple(ArcPair(input_grid=grid, output_grid=grid) for _ in range(5)),
        )
        rendered = render_training_prompt(augmented)
        tokenizer = make_nvarc_tokenizer()

        tokenized = tokenize_nvarc_training_sample(rendered, tokenizer)

        self.assertEqual(NVARC_MAX_TOKENS, 8400)
        self.assertEqual(len(tokenized.input_ids), 9330)
        self.assertEqual(tokenized.prefix_token_ends, (2802, 4668, 6534, 8400))
        self.assertEqual(tokenized.target_grids, (grid,) * 4)
        self.assertEqual(
            tokenized.input_ids,
            tuple(tokenizer.encode(rendered.prompt, add_special_tokens=False)),
        )

        with self.assertRaisesRegex(
            ValueError, "source prefix contains 8400 tokens, exceeding 8399"
        ):
            tokenize_nvarc_training_sample(rendered, tokenizer, max_tokens=8399)

    def test_tokenization_preserves_evaluation_identity_and_context_limit(self) -> None:
        task = ArcEvaluationTask(
            task_id="evaluation-task",
            demonstrations=(
                ArcPair(input_grid=((1,),), output_grid=((2,),)),
            ),
            queries=(ArcQuery(query_index=3, input_grid=((4,),)),),
        )
        rendered = render_evaluation_prompts(task)[0]
        tokenizer = make_nvarc_tokenizer()

        tokenized = tokenize_nvarc_evaluation_sample(rendered, tokenizer)

        self.assertEqual(tokenized.task_id, "evaluation-task")
        self.assertEqual(tokenized.query_index, 3)
        self.assertEqual(
            tokenized.input_ids,
            tuple(tokenizer.encode(rendered.prompt, add_special_tokens=False)),
        )

        with self.assertRaises(ValueError):
            tokenize_nvarc_evaluation_sample(
                rendered,
                tokenizer,
                max_tokens=len(tokenized.input_ids) - 1,
            )


if __name__ == "__main__":
    unittest.main()
