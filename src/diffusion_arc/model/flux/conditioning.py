"""Exact or bucketed padding and attention masks for FLUX text prefixes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from diffusion_arc.sequence_buckets import SequenceLengthBuckets

from ..contracts import FluxTextInputs


def pad_flux_text_prefixes(
    hidden_state_prefixes: Sequence[Tensor],
    *,
    length_buckets: SequenceLengthBuckets = SequenceLengthBuckets(),
    padded_length: int | None = None,
) -> FluxTextInputs:
    """Right-pad exact NVARC prefixes into one FLUX text batch.

    Prefix selection and ownership remain training-side concerns. This pure
    tensor operation trusts a non-empty sequence of compatible exact prefixes
    and preserves their order. A validated scheduler can supply an explicit
    ``padded_length`` instead of rounding to ``length_buckets``.
    """

    prefix_lengths = tuple(prefix.shape[0] for prefix in hidden_state_prefixes)
    target_count = len(hidden_state_prefixes)
    max_prefix_length = (
        length_buckets.padded_length(max(prefix_lengths))
        if padded_length is None else padded_length
    )

    reference_prefix = hidden_state_prefixes[0]
    encoder_hidden_states = reference_prefix.new_zeros(
        (target_count, max_prefix_length, reference_prefix.shape[-1])
    )

    for target_index, prefix in enumerate(hidden_state_prefixes):
        encoder_hidden_states[target_index, : prefix.shape[0]] = prefix

    positions = torch.arange(max_prefix_length, device=reference_prefix.device)
    prefix_length_tensor = torch.tensor(
        prefix_lengths,
        dtype=torch.int64,
        device=reference_prefix.device,
    )
    text_attention_mask = positions.unsqueeze(0) < prefix_length_tensor.unsqueeze(1)

    text_position_ids = torch.zeros(
        (target_count, max_prefix_length, 4),
        dtype=torch.int64,
        device=reference_prefix.device,
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


__all__ = ["build_flux_joint_attention_mask", "pad_flux_text_prefixes"]
