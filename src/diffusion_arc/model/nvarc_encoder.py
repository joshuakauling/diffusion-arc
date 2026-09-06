"""Frozen batched NVARC encoding of packed complete source sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion_arc.sequence_buckets import SequenceLengthBuckets

from .contracts import (
    NVARC_HIDDEN_STATE_INDICES,
    NVARC_PAD_TOKEN_ID,
    NvarcBatchOutput,
)


def _module_input_device(module: nn.Module) -> torch.device:
    """Return the device on which a module expects token IDs."""

    get_input_embeddings = getattr(module, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        raise RuntimeError(
            "NVARC must expose input embeddings through get_input_embeddings()"
        )

    embeddings = get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if not isinstance(weight, Tensor):
        raise RuntimeError("NVARC input embeddings must expose a tensor weight")
    return weight.device


class NvarcBatchRunner(nn.Module):
    """Run frozen NVARC once for a batch of complete packed sources.

    The public forward interface intentionally accepts no target metadata. It
    right-pads the source sequences only for the NVARC call and removes those
    padded positions before returning the selected hidden states.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        length_buckets: SequenceLengthBuckets = SequenceLengthBuckets(),
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        self.model = model
        self.length_buckets = length_buckets
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> NvarcBatchRunner:
        """Keep both the runner and its frozen NVARC model in evaluation mode."""

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self,
        input_ids: Tensor,
        sequence_offsets: Tensor,
    ) -> NvarcBatchOutput:
        """Encode all complete sources into independent compact tensors."""

        offsets = sequence_offsets.tolist()
        source_lengths = tuple(
            end - start for start, end in zip(offsets, offsets[1:])
        )
        source_count = len(source_lengths)
        max_length = self.length_buckets.padded_length(max(source_lengths))
        model_device = _module_input_device(self.model)

        flat_input_ids = input_ids.to(device=model_device)
        padded_input_ids = torch.full(
            (source_count, max_length),
            NVARC_PAD_TOKEN_ID,
            dtype=torch.int64,
            device=model_device,
        )
        attention_mask = torch.zeros(
            (source_count, max_length),
            dtype=torch.bool,
            device=model_device,
        )

        source_start = 0
        for source_index, source_length in enumerate(source_lengths):
            source_end = source_start + source_length
            padded_input_ids[source_index, :source_length] = flat_input_ids[
                source_start:source_end
            ]
            attention_mask[source_index, :source_length] = True
            source_start = source_end

        with torch.no_grad():
            output = self.model(
                input_ids=padded_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        sequence_hidden_states = tuple(
            torch.cat(
                tuple(
                    output.hidden_states[hidden_state_index][
                        source_index, :source_length
                    ]
                    for hidden_state_index in NVARC_HIDDEN_STATE_INDICES
                ),
                dim=-1,
            )
            for source_index, source_length in enumerate(source_lengths)
        )
        return NvarcBatchOutput(
            sequence_hidden_states=sequence_hidden_states,
        )


__all__ = ["NvarcBatchRunner"]
