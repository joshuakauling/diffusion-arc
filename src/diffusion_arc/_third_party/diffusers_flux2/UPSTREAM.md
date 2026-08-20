# Diffusers FLUX.2 transformer source

This directory contains the FLUX.2 transformer implementation used by
Diffusion-ARC.

## Provenance

- Project: Hugging Face Diffusers
- Release: `v0.39.0`
- Upstream file:
  [`src/diffusers/models/transformers/transformer_flux2.py`](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_flux2.py)
- Retrieval date: 2026-08-20
- Upstream SHA-256:
  `0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006`
- Local SHA-256:
  `20a84b9cd636e5a69ef4d82591f3c964427bb5da3afa1f0a6c933aed463eb50f`

The upstream file is licensed under Apache License 2.0. The complete license
text is stored in `LICENSE` next to the source.

## Local changes

The source is unchanged except that its package-relative imports were rewritten
as absolute `diffusers` imports so that the module can live below
`diffusion_arc._third_party`. Shared Diffusers infrastructure such as model
configuration, checkpoint loading, attention dispatch, and normalization is not
copied and remains supplied by the pinned `diffusers==0.39.0` dependency.

Project code should import the transformer through
`diffusion_arc.model.flux_transformer`, not from this internal package.
