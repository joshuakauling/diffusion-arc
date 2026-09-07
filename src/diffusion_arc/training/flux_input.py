"""Prepare reusable frozen-model outputs for individual FLUX targets."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from diffusion_arc.arc_data.canvas import _render_valid_grid_canvas
from diffusion_arc.arc_data.tokenization import ArcTokenizedTrainingSample
from diffusion_arc.arc_data.types import Grid
from diffusion_arc.model.flux_vae import FluxVaeEncoder
from diffusion_arc.model.nvarc_encoder import NvarcBatchRunner

from .nvarc_schedule import NvarcSchedulingPolicy, NvarcSourceBatch, schedule_nvarc_sources


# Benchmark metadata: source identities and the physical batches used to prepare them.
@dataclass(frozen=True)
class PreparedSource:
    """CPU-only identity and eligible prefix lengths in logical source order."""

    key: str
    prefix_ends: tuple[int, ...]


@dataclass(frozen=True)
class PreparationMetrics:
    """Sources and physical NVARC calls actually executed for one window."""

    sources: tuple[PreparedSource, ...]
    nvarc_batches: tuple[NvarcSourceBatch, ...]


class NvarcSequenceState:
    """Own full source states until all needed prefixes enter FLUX microbatches.

    Logical ``FluxInput`` objects share this owner. ``build_flux_microbatch``
    copies their prefixes into a physical batch before marking them consumed.
    After the final prefix is consumed, the owner drops its full-source tensor
    reference; the physical microbatch owns the copied conditioning states.
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


@dataclass(frozen=True)
class _EligibleSequence:
    """Training targets with enough preceding demonstration pairs and their input.

    ``prefix_ends`` and ``target_grids`` contain only those targets.
    ``input_ids`` stops before the final target output; earlier outputs remain
    available as context.
    """

    input_ids: tuple[int, ...]
    prefix_ends: tuple[int, ...]
    target_grids: tuple[Grid, ...]


def _require_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _select_eligible_sequence(
    sequence: ArcTokenizedTrainingSample,
    *,
    min_context_pairs_before_target: int,
) -> _EligibleSequence | None:
    """Select targets with enough demonstrations and trim IDs to their last prefix."""

    first_target_position = min_context_pairs_before_target - 1
    target_count = len(sequence.target_grids)
    if first_target_position >= target_count:
        return None

    prefix_ends = sequence.prefix_token_ends[first_target_position:]
    target_grids = sequence.target_grids[first_target_position:]
    source_end = prefix_ends[-1]
    return _EligibleSequence(
        input_ids=sequence.input_ids[:source_end],
        prefix_ends=prefix_ends,
        target_grids=target_grids,
    )


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

    return prepare_flux_inputs_batch(
        (sequence,),
        nvarc_runner=nvarc_runner,
        vae_encoder=vae_encoder,
        min_context_pairs_before_target=min_context_pairs_before_target,
        vae_microbatch_size=None,
    )


def prepare_flux_inputs_batch(
    sequences: Sequence[ArcTokenizedTrainingSample],
    *,
    nvarc_runner: NvarcBatchRunner,
    vae_encoder: FluxVaeEncoder,
    min_context_pairs_before_target: int = 1,
    vae_microbatch_size: int | None = None,
    nvarc_scheduling: NvarcSchedulingPolicy | None = None,
    on_preparation: Callable[[PreparationMetrics], None] | None = None,
) -> tuple[FluxInput, ...]:
    """Prepare one source window with independent NVARC and VAE batches.

    First select eligible targets and trim each source to its last needed
    prefix. Then execute the optional NVARC schedule, indexing returned states
    by original eligible-source position. Without scheduling, all eligible
    sources execute in one physical NVARC batch.

    VAE chunks span the whole window's targets in original source-major order,
    independently of NVARC groups. ``vae_microbatch_size=None`` encodes them
    in one call. Finally, pair each latent with its source state and prefix in
    that same order, preserving the target FIFO and optimizer-batch membership.
    Flow noise and timesteps are sampled later, when FLUX consumes the inputs.
    """

    source_sequences = tuple(sequences)
    if not source_sequences:
        raise ValueError("sequences must contain at least one training sample")
    _require_positive_integer(
        min_context_pairs_before_target,
        name="min_context_pairs_before_target",
    )
    if vae_microbatch_size is not None:
        _require_positive_integer(
            vae_microbatch_size,
            name="vae_microbatch_size",
        )

    eligible_sequences_list: list[_EligibleSequence] = []
    prepared_sources: list[PreparedSource] = []
    for sequence in source_sequences:
        eligible = _select_eligible_sequence(
            sequence,
            min_context_pairs_before_target=min_context_pairs_before_target,
        )
        if eligible is not None:
            eligible_sequences_list.append(eligible)
            # Benchmark metadata: retain logical source order before scheduling.
            if on_preparation is not None:
                prepared_sources.append(PreparedSource(str(sequence.key), eligible.prefix_ends))
    eligible_sequences = tuple(eligible_sequences_list)
    if not eligible_sequences:
        return ()

    source_lengths = tuple(len(eligible.input_ids) for eligible in eligible_sequences)
    if nvarc_scheduling is None:
        # The legacy runner owns padding. Read that same policy for reporting,
        # without introducing a second scheduler in the benchmark entrypoint.
        padded_length = (
            nvarc_runner.length_buckets.padded_length(max(source_lengths))
            if on_preparation is not None else max(source_lengths)
        )
        batches = (NvarcSourceBatch(tuple(range(len(eligible_sequences))), padded_length),)
    else:
        batches = schedule_nvarc_sources(source_lengths, nvarc_scheduling)

    # Keep outputs indexed by logical source position so scheduling cannot
    # change the alignment between conditioning states and VAE targets.
    source_hidden_states: dict[int, Tensor] = {}
    for batch in batches:
        packed_input_ids: list[int] = []
        sequence_offsets = [0]
        for index in batch.source_indices:
            packed_input_ids.extend(eligible_sequences[index].input_ids)
            sequence_offsets.append(len(packed_input_ids))
        padding_arguments = (
            {} if nvarc_scheduling is None else {"padded_length": batch.padded_length}
        )
        nvarc_output = nvarc_runner(
            torch.tensor(packed_input_ids, dtype=torch.int64),
            torch.tensor(sequence_offsets, dtype=torch.int64),
            **padding_arguments,
        )
        if len(nvarc_output.sequence_hidden_states) != len(batch.source_indices):
            raise RuntimeError(
                "batched preparation requires one NVARC output per eligible source"
            )
        source_hidden_states.update(zip(batch.source_indices, nvarc_output.sequence_hidden_states))

    # VAE chunks span the logical window, not the physical NVARC groups.
    target_grids = tuple(
        target_grid
        for eligible in eligible_sequences
        for target_grid in eligible.target_grids
    )
    physical_vae_batch_size = (
        len(target_grids) if vae_microbatch_size is None else vae_microbatch_size
    )
    clean_latents: list[Tensor] = []
    for start in range(0, len(target_grids), physical_vae_batch_size):
        image_batch = torch.stack(
            tuple(
                _render_valid_grid_canvas(target_grid)
                for target_grid in target_grids[start : start + physical_vae_batch_size]
            )
        )
        latent_batch = vae_encoder(image_batch)
        if latent_batch.shape[0] != image_batch.shape[0]:
            raise RuntimeError("VAE output count does not match its target image batch")
        clean_latents.extend(latent_batch.unbind(dim=0))

    flux_inputs: list[FluxInput] = []
    latent_index = 0
    for index, eligible in enumerate(eligible_sequences):
        sequence_state = NvarcSequenceState(
            hidden_states=source_hidden_states[index],
            pending_prefix_ends=eligible.prefix_ends,
        )
        for prefix_end in eligible.prefix_ends:
            flux_inputs.append(
                FluxInput(
                    nvarc_sequence_state=sequence_state,
                    prefix_end=prefix_end,
                    clean_latent=clean_latents[latent_index],
                )
            )
            latent_index += 1

    # Benchmark callback: report this window after frozen preparation completes.
    if on_preparation is not None:
        on_preparation(PreparationMetrics(tuple(prepared_sources), batches))
    return tuple(flux_inputs)


__all__ = [
    "FluxInput",
    "NvarcSequenceState",
    "PreparationMetrics",
    "PreparedSource",
    "prepare_flux_inputs",
    "prepare_flux_inputs_batch",
]
