"""Target-free NVARC conditioning preparation for FLUX generation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from diffusion_arc.arc_data import ArcTokenizedEvaluationSample
from diffusion_arc.model import FluxTextInputs, NvarcBatchRunner
from diffusion_arc.model.flux_text import pad_flux_text_prefixes


def prepare_evaluation_text_inputs(
    samples: Sequence[ArcTokenizedEvaluationSample],
    *,
    nvarc_runner: NvarcBatchRunner,
) -> FluxTextInputs:
    """Encode complete target-free evaluation prompts for FLUX conditioning."""

    evaluation_samples = tuple(samples)
    if not evaluation_samples:
        raise ValueError("samples must not be empty")
    if any(
        not isinstance(sample, ArcTokenizedEvaluationSample)
        for sample in evaluation_samples
    ):
        raise TypeError("samples must contain ArcTokenizedEvaluationSample values")

    input_ids = torch.tensor(
        tuple(
            token_id
            for sample in evaluation_samples
            for token_id in sample.input_ids
        ),
        dtype=torch.int64,
    )
    offsets = [0]
    for sample in evaluation_samples:
        offsets.append(offsets[-1] + len(sample.input_ids))
    sequence_offsets = torch.tensor(offsets, dtype=torch.int64)

    nvarc_output = nvarc_runner(input_ids, sequence_offsets)
    if len(nvarc_output.sequence_hidden_states) != len(evaluation_samples):
        raise RuntimeError(
            "NVARC output count does not match target-free evaluation samples"
        )
    return pad_flux_text_prefixes(nvarc_output.sequence_hidden_states)


__all__ = ["prepare_evaluation_text_inputs"]
