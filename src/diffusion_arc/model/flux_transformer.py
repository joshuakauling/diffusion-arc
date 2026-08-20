"""Project-facing access to the pinned local FLUX.2 transformer source.

Importing this module intentionally loads Diffusers. The lightweight
``diffusion_arc.model`` package does not import it eagerly.
"""

from diffusion_arc._third_party.diffusers_flux2.transformer_flux2 import (
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
)

__all__ = ["Flux2Transformer2DModel", "Flux2Transformer2DModelOutput"]
