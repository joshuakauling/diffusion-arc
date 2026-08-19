"""Checkpoint-free tokenizer used only by local diagnostic scripts."""

from __future__ import annotations

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


def make_nvarc_tokenizer(*, model_max_length: int) -> PreTrainedTokenizerFast:
    """Build the complete 16-token NVARC tokenizer without a checkpoint."""

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
        model_max_length=model_max_length,
    )


__all__ = ["make_nvarc_tokenizer"]
