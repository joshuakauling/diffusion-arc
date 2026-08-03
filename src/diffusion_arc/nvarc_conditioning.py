"""Frozen NVARC conditioning for the FLUX.2 context interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _module_input_device(module: nn.Module) -> torch.device:
    get_input_embeddings = getattr(module, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


class NVARCConditioner(nn.Module):
    """Extract and concatenate frozen NVARC hidden states.

    FLUX.2 Klein 4B expects three Qwen3 hidden states with width 2,560,
    concatenated in layer order into a context width of 7,680.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_indices: Sequence[int] = (9, 18, 27),
        hidden_size: int = 2560,
        context_dim: int = 7680,
        max_length: int = 8192,
    ) -> None:
        super().__init__()
        if not layer_indices:
            raise ValueError("At least one NVARC hidden layer is required.")
        if any(index < 0 for index in layer_indices):
            raise ValueError(f"Hidden-layer indices must be non-negative: {layer_indices}")
        if len(layer_indices) * hidden_size != context_dim:
            raise ValueError(
                "The selected hidden layers do not match the FLUX context width: "
                f"{len(layer_indices)} * {hidden_size} != {context_dim}."
            )
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")

        self.model = model
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.hidden_size = int(hidden_size)
        self.context_dim = int(context_dim)
        self.max_length = int(max_length)

        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> NVARCConditioner:
        """Keep the frozen encoder in evaluation mode."""

        super().train(False)
        self.model.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, L], got {input_ids.shape}.")
        if input_ids.shape[0] != 1:
            raise ValueError(
                "The initial Diffusion-ARC path requires batch size 1 so every "
                "FLUX context has its exact, unpadded length."
            )
        if input_ids.shape[1] == 0:
            raise ValueError("NVARC context must contain at least one token.")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must match input_ids exactly, got "
                f"{attention_mask.shape} and {input_ids.shape}."
            )
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"NVARC context length {input_ids.shape[1]} exceeds the explicit "
                f"limit of {self.max_length}; truncation is forbidden."
            )
        if input_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}.")
        if attention_mask.dtype != torch.bool and attention_mask.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(
                f"attention_mask must be boolean or integer, got {attention_mask.dtype}."
            )
        if not bool(torch.all(attention_mask == 1).item()):
            raise ValueError(
                "Padded or masked NVARC contexts are forbidden because FLUX.2 "
                "has no text attention-mask input. Use the exact sequence length."
            )

        model_device = _module_input_device(self.model)
        input_ids = input_ids.to(device=model_device)
        attention_mask = attention_mask.to(device=model_device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("NVARC did not return hidden states.")
        if len(hidden_states) <= max(self.layer_indices):
            raise RuntimeError(
                f"NVARC returned {len(hidden_states)} hidden states, but layer "
                f"{max(self.layer_indices)} was requested."
            )

        selected: list[Tensor] = []
        for index in self.layer_indices:
            state = hidden_states[index]
            expected_shape = (*input_ids.shape, self.hidden_size)
            if tuple(state.shape) != expected_shape:
                raise RuntimeError(
                    f"NVARC hidden state {index} has shape {tuple(state.shape)}; "
                    f"expected {expected_shape}."
                )
            selected.append(state)

        context = torch.cat(selected, dim=-1)
        if context.shape[-1] != self.context_dim:
            raise RuntimeError(
                f"Conditioning width is {context.shape[-1]}, expected {self.context_dim}."
            )
        if context.requires_grad:
            raise RuntimeError("Frozen NVARC conditioning unexpectedly requires gradients.")
        if not bool(torch.isfinite(context).all().item()):
            raise FloatingPointError("NVARC conditioning contains non-finite values.")
        return context
