"""Prepare reusable frozen-model outputs for individual FLUX targets."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch
from torch import Tensor

from diffusion_arc.arc_data.canvas import _render_valid_grid_canvas
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.model.flux_image import FluxVaeEncoder
from diffusion_arc.model.nvarc import NvarcBatchRunner


class NvarcSequenceState:
    """Own one sequence's hidden states until every dependent input is built.

    Prefix ends are unique within one ARC sequence, so the owner can use them
    directly for consumption bookkeeping without global training identities.
    """

    def __init__(
        self,
        hidden_states: Tensor,
        pending_prefix_ends: Collection[int],
    ) -> None:
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
            raise ValueError("hidden_states must have shape [L, D] with L > 0")
        if hidden_states.requires_grad or hidden_states.grad_fn is not None:
            raise ValueError("hidden_states must not retain an autograd graph")

        prefix_ends = tuple(pending_prefix_ends)
        unique_prefix_ends = set(prefix_ends)
        if not prefix_ends:
            raise ValueError("pending_prefix_ends must not be empty")
        if any(
            isinstance(prefix_end, bool) or not isinstance(prefix_end, int)
            for prefix_end in prefix_ends
        ):
            raise ValueError("pending_prefix_ends must contain integers")
        if len(unique_prefix_ends) != len(prefix_ends):
            raise ValueError("pending_prefix_ends must be unique")
        if any(
            not 0 < prefix_end <= hidden_states.shape[0]
            for prefix_end in prefix_ends
        ):
            raise ValueError(
                "pending_prefix_ends must lie inside the NVARC sequence state"
            )

        self._hidden_states: Tensor | None = hidden_states
        self._sequence_length = hidden_states.shape[0]
        self._pending_prefix_ends = unique_prefix_ends

    @property
    def sequence_length(self) -> int:
        """Return the number of real hidden-state positions."""

        return self._sequence_length

    @property
    def pending_prefix_ends(self) -> frozenset[int]:
        """Return an immutable snapshot of prefixes that still need this state."""

        return frozenset(self._pending_prefix_ends)

    @property
    def is_released(self) -> bool:
        """Return whether this owner has dropped its hidden-state tensor."""

        return self._hidden_states is None

    @property
    def allocated_bytes(self) -> int:
        """Return tensor bytes still retained directly by this owner."""

        hidden_states = self._hidden_states
        if hidden_states is None:
            return 0
        return hidden_states.numel() * hidden_states.element_size()

    def _validate_pending_prefixes(
        self,
        prefix_ends: Collection[int],
    ) -> None:
        requested_prefixes = tuple(prefix_ends)
        unique_requested_prefixes = set(requested_prefixes)
        if len(unique_requested_prefixes) != len(requested_prefixes):
            raise ValueError(
                "prefix ends must be unique within one NVARC sequence state"
            )
        unknown_prefixes = unique_requested_prefixes.difference(
            self._pending_prefix_ends
        )
        if unknown_prefixes:
            raise RuntimeError(
                "FLUX prefixes are not pending for this NVARC sequence state: "
                f"{sorted(unknown_prefixes)}"
            )
        if self._hidden_states is None:
            raise RuntimeError("NVARC sequence hidden states have already been released")

    def _prefix_for(self, prefix_end: int) -> Tensor:
        self._validate_pending_prefixes((prefix_end,))
        hidden_states = self._hidden_states
        if hidden_states is None:  # Established by _validate_pending_prefixes.
            raise AssertionError("validated NVARC sequence state unexpectedly released")
        return hidden_states[:prefix_end]

    def _commit_prefixes(
        self,
        prefix_ends: Collection[int],
    ) -> None:
        """Drop already validated prefixes without another failure point."""

        self._pending_prefix_ends.difference_update(prefix_ends)
        if not self._pending_prefix_ends:
            self._hidden_states = None


@dataclass(frozen=True)
class FluxInput:
    """One prepared target waiting for a physical FLUX microbatch.

    Inputs from the same ARC sequence share ``nvarc_sequence_state`` and retain
    only the exact prefix boundary and clean target latent needed by FLUX.
    """

    nvarc_sequence_state: NvarcSequenceState
    prefix_end: int
    clean_latent: Tensor


def prepare_flux_inputs(
    sequence: ArcTokenizedTrainingSample,
    *,
    nvarc_runner: NvarcBatchRunner,
    vae_encoder: FluxVaeEncoder,
    min_context_pairs_before_target: int = 1,
) -> tuple[FluxInput, ...]:
    """Run frozen preparation once and return every eligible FLUX input.

    Eligibility is decided before either model runs. NVARC receives the single
    complete conditioning sequence ending at the final eligible prefix. The
    VAE receives every eligible target image in one physical batch. Flow noise
    and timesteps are intentionally deferred until FLUX consumes an input.
    """

    if (
        isinstance(min_context_pairs_before_target, bool)
        or not isinstance(min_context_pairs_before_target, int)
        or min_context_pairs_before_target <= 0
    ):
        raise ValueError("min_context_pairs_before_target must be a positive integer")

    first_target_position = min_context_pairs_before_target - 1
    target_count = len(sequence.target_grids)
    if first_target_position >= target_count:
        return ()

    target_positions = tuple(range(first_target_position, target_count))
    prefix_ends = tuple(
        sequence.prefix_token_ends[position] for position in target_positions
    )
    target_grids = tuple(sequence.target_grids[position] for position in target_positions)

    source_end = prefix_ends[-1]
    input_ids = torch.tensor(sequence.input_ids[:source_end], dtype=torch.int64)
    sequence_offsets = torch.tensor((0, source_end), dtype=torch.int64)
    nvarc_output = nvarc_runner(input_ids, sequence_offsets)
    if len(nvarc_output.sequence_hidden_states) != 1:
        raise RuntimeError("single-sequence preparation requires one NVARC output")

    target_images = torch.stack(
        tuple(_render_valid_grid_canvas(target_grid) for target_grid in target_grids)
    )
    clean_latents = vae_encoder(target_images)
    sequence_state = NvarcSequenceState(
        hidden_states=nvarc_output.sequence_hidden_states[0],
        pending_prefix_ends=prefix_ends,
    )

    return tuple(
        FluxInput(
            nvarc_sequence_state=sequence_state,
            prefix_end=prefix_end,
            clean_latent=clean_latents[prepared_index],
        )
        for prepared_index, prefix_end in enumerate(prefix_ends)
    )


__all__ = [
    "FluxInput",
    "NvarcSequenceState",
    "prepare_flux_inputs",
]
