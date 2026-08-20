"""Target-aware preparation of frozen NVARC prefixes for FLUX text input."""

from __future__ import annotations

import torch
from torch import Tensor

from .contracts import FluxTextInputs, PackedNvarcHiddenStates


def prepare_flux_text_inputs(
    packed_hidden_states: PackedNvarcHiddenStates,
    target_sequence_indices: Tensor,
    target_prefix_ends: Tensor,
) -> FluxTextInputs:
    """Build a dynamically right-padded FLUX text microbatch.

    Inputs are trusted internal contracts produced by the validated collation
    and frozen-NVARC paths. Target order is preserved exactly.
    """

    offsets = packed_hidden_states.sequence_offsets.tolist()
    source_indices = target_sequence_indices.tolist()
    prefix_ends = target_prefix_ends.tolist()
    target_count = len(prefix_ends)
    max_prefix_length = max(prefix_ends)

    hidden_states = packed_hidden_states.hidden_states
    encoder_hidden_states = hidden_states.new_zeros(
        (target_count, max_prefix_length, hidden_states.shape[-1])
    )

    for target_index, (source_index, prefix_end) in enumerate(
        zip(source_indices, prefix_ends, strict=True)
    ):
        source_start = offsets[source_index]
        encoder_hidden_states[target_index, :prefix_end] = hidden_states[
            source_start : source_start + prefix_end
        ]

    positions = torch.arange(max_prefix_length, device=hidden_states.device)
    prefix_lengths = target_prefix_ends.to(device=hidden_states.device)
    text_attention_mask = positions.unsqueeze(0) < prefix_lengths.unsqueeze(1)

    text_position_ids = torch.zeros(
        (target_count, max_prefix_length, 4),
        dtype=torch.int64,
        device=hidden_states.device,
    )
    text_position_ids[:, :, 3] = positions

    return FluxTextInputs(
        encoder_hidden_states=encoder_hidden_states,
        text_position_ids=text_position_ids,
        text_attention_mask=text_attention_mask,
    )


def build_flux_joint_attention_mask(
    text_attention_mask: Tensor,
    image_token_count: int,
) -> Tensor:
    """Append valid image keys and return a mask ready for FLUX attention."""

    image_attention_mask = torch.ones(
        (text_attention_mask.shape[0], image_token_count),
        dtype=torch.bool,
        device=text_attention_mask.device,
    )
    key_attention_mask = torch.cat(
        (text_attention_mask, image_attention_mask),
        dim=1,
    )
    return key_attention_mask[:, None, None, :]


__all__ = ["build_flux_joint_attention_mask", "prepare_flux_text_inputs"]
