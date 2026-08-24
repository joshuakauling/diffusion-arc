"""Execute one prepared FLUX microbatch and compute its flow loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from diffusion_arc.model.flow import flow_matching_loss, sample_flow_matching_batch
from diffusion_arc.model.flux_text import build_flux_joint_attention_mask

from .flux_microbatch import FluxMicrobatch


def _transformer_device(transformer: nn.Module) -> torch.device:
    """Return the device holding the persistent FLUX parameters."""

    try:
        parameter = next(transformer.parameters())
    except StopIteration as error:
        raise RuntimeError("FLUX transformer must expose parameters") from error
    if parameter.dtype != torch.float32:
        raise RuntimeError("persistent FLUX parameters must use float32")
    return parameter.device


def compute_flux_microbatch_loss(
    transformer: nn.Module,
    microbatch: FluxMicrobatch,
    *,
    compute_dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Run one BF16-autocast FLUX forward and return its FP32 mean loss."""

    transformer_device = _transformer_device(transformer)
    flow_batch = sample_flow_matching_batch(
        microbatch.clean_latents,
        generator=generator,
    )
    joint_attention_mask = build_flux_joint_attention_mask(
        microbatch.text_inputs.text_attention_mask,
        image_token_count=microbatch.clean_latents.shape[1],
    )
    with torch.autocast(
        device_type=transformer_device.type,
        dtype=compute_dtype,
    ):
        output = transformer(
            hidden_states=flow_batch.noisy_latents.to(device=transformer_device),
            encoder_hidden_states=(
                microbatch.text_inputs.encoder_hidden_states.to(
                    device=transformer_device
                )
            ),
            timestep=flow_batch.timesteps.to(device=transformer_device),
            img_ids=microbatch.image_position_ids.to(device=transformer_device),
            txt_ids=microbatch.text_inputs.text_position_ids.to(
                device=transformer_device
            ),
            guidance=None,
            joint_attention_kwargs={
                "attention_mask": joint_attention_mask.to(
                    device=transformer_device
                )
            },
            return_dict=True,
        )
    return flow_matching_loss(
        output.sample,
        flow_batch.target_velocity.to(device=transformer_device),
    )


__all__ = ["compute_flux_microbatch_loss"]
