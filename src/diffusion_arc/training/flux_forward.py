"""Execute one prepared FLUX microbatch and compute its flow loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion_arc.model.flow import flow_matching_loss, sample_flow_matching_batch
from diffusion_arc.model.flux_text import build_flux_joint_attention_mask

from .flux_microbatch import FluxMicrobatch


def compute_flux_microbatch_loss(
    transformer: nn.Module,
    microbatch: FluxMicrobatch,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one flow state, execute FLUX, and return its unscaled mean loss."""

    flow_batch = sample_flow_matching_batch(
        microbatch.clean_latents,
        generator=generator,
    )
    joint_attention_mask = build_flux_joint_attention_mask(
        microbatch.text_inputs.text_attention_mask,
        image_token_count=microbatch.clean_latents.shape[1],
    )
    output = transformer(
        hidden_states=flow_batch.noisy_latents,
        encoder_hidden_states=microbatch.text_inputs.encoder_hidden_states,
        timestep=flow_batch.timesteps,
        img_ids=microbatch.image_position_ids,
        txt_ids=microbatch.text_inputs.text_position_ids,
        guidance=None,
        joint_attention_kwargs={"attention_mask": joint_attention_mask},
        return_dict=True,
    )
    return flow_matching_loss(
        output.sample,
        flow_batch.target_velocity,
    )


__all__ = ["compute_flux_microbatch_loss"]
