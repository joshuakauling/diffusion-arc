"""Frozen NVARC encoding of sources concatenated end to end without padding."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import NVARC_ATTENTION_IMPLEMENTATION

from ..contracts import NvarcBatchOutput


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


class NvarcEncoder(nn.Module):
    """Encode sources concatenated end to end in one frozen NVARC forward.

    The input contains all tokens of source 1, followed by source 2, and so on,
    without padding between sources or at the end. FA3 Varlen uses their
    boundaries to keep causal attention within each source. Position IDs restart
    at zero for each source. Selected hidden states are returned as separate
    compact tensors in the original source order. The validated one-based
    hidden-state indices select three increasing block outputs. Execution stops
    after the highest selected block, always before final normalization.
    """

    def __init__(self, model: nn.Module, *, hidden_state_indices: tuple[int, int, int]) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if getattr(model.config, "_attn_implementation", None) != NVARC_ATTENTION_IMPLEMENTATION:
            raise ValueError("NvarcEncoder requires the explicit nvarc_fa3 attention binding")
        if hidden_state_indices[-1] > len(model.base_model.layers):
            raise ValueError("selected NVARC layer exceeds the checkpoint's decoder block count")
        self.hidden_state_indices = tuple(hidden_state_indices)
        self.model = model
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> NvarcEncoder:
        """Keep both the encoder and its frozen NVARC model in evaluation mode."""

        super().train(False)
        self.model.eval()
        return self

    def forward(self, input_ids: Tensor, sequence_offsets: Tensor) -> NvarcBatchOutput:
        """Encode flat IDs and validated CPU source offsets without padding.

        Offsets include zero and the total token count. The model receives
        [1, total_tokens] IDs, reset position IDs, and int32 GPU boundaries;
        its maximum attention length is the longest individual source. Original
        Qwen blocks run directly, skipping later blocks, final norm and LM head.
        """

        offsets = sequence_offsets.tolist()
        lengths = tuple(end - start for start, end in zip(offsets, offsets[1:]))
        device = _module_input_device(self.model)
        positions = torch.cat(tuple(torch.arange(length) for length in lengths))
        cumulative_lengths = sequence_offsets.to(device=device, dtype=torch.int32)
        position_ids = positions.to(device=device).unsqueeze(0)
        backbone = self.model.base_model
        selected_states: list[Tensor] = []
        with torch.no_grad():
            hidden_states = backbone.embed_tokens(input_ids.to(device=device).unsqueeze(0))
            position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
            # Read raw block outputs, including block 36. HF's final model
            # output applies a separate normalization that FLUX does not expect.
            for index, block in enumerate(
                backbone.layers[:self.hidden_state_indices[-1]], start=1,
            ):
                hidden_states = block(
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    cu_seq_lens_q=cumulative_lengths,
                    cu_seq_lens_k=cumulative_lengths,
                    max_length_q=max(lengths),
                    max_length_k=max(lengths),
                    use_cache=False,
                )
                if index in self.hidden_state_indices:
                    selected_states.append(hidden_states)
        return NvarcBatchOutput(sequence_hidden_states=tuple(
            torch.cat(tuple(states[0, start:end] for states in selected_states), dim=-1)
            for start, end in zip(offsets, offsets[1:])
        ))


__all__ = ["NvarcEncoder"]
