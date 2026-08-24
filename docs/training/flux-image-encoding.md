# FLUX Target-Image Encoding

## Status and scope

`FluxVaeEncoder` is the implemented boundary between canonical ARC target
canvases and clean transformer-facing FLUX latents. It executes the frozen
FLUX.2 VAE in evaluation mode, uses the posterior mode, applies FLUX.2's native
2×2 packing and frozen BatchNorm normalization, and returns the clean latents.

The implementation lives in
[`src/diffusion_arc/model/flux_image.py`](../../src/diffusion_arc/model/flux_image.py).

This component does not choose the palette, sample flow noise or timesteps,
construct image position IDs or a noisy state, execute the FLUX transformer, or
compute a loss.

## Fixed input representation

The collator produces one target canvas for every target exposure:

```text
[K, 3, 480, 480], torch.float32, values in [-1, 1]
```

ARC IDs use the fixed `greedy_rgb_maximin_v1` mapping defined in
[`arc_data/palette.py`](../../src/diffusion_arc/arc_data/palette.py), with class
`10` reserved for RGB-white PAD. Palette selection is established before this
model boundary; the VAE encoder consumes trusted rendered tensors and does not
reclassify their pixels.

## Encoding path

For `K` target exposures, the deterministic path is:

```text
[K, 3, 480, 480]      normalized RGB canvases
        |
        v  frozen VAE posterior mode
[K, 32, 60, 60]       VAE latents
        |
        v  lossless 2x2 spatial-to-channel packing
[K, 128, 30, 30]      packed latents
        |
        v  frozen VAE BatchNorm running statistics
[K, 128, 30, 30]      normalized clean latents
        |
        v  row-major spatial flattening
[K, 900, 128]         clean transformer-facing targets
```

The encoder moves input canvases to the VAE parameter device and dtype before
encoding. It uses `torch.no_grad()` so VAE activations and parameters never
participate in autograd, while the returned ordinary tensors remain valid
inputs for computing downstream FLUX parameter gradients.

The VAE is frozen at construction and remains in evaluation mode even when a
parent training module calls `train(True)`.

## Output contract

The returned value is one tensor:

```python
clean_latents  # [K, 900, 128], VAE floating dtype
```

Target order is unchanged from the supplied target images. The fixed
target-image coordinates are independent of VAE encoding and are therefore
generated later by `build_flux_microbatch`. They follow FLUX's four-axis target
convention:

```text
(t, h, w, l) = (0, h, w, 0),  h,w in [0,29]
```

`clean_latents` is the clean flow endpoint `z_0`.
[`sample_flow_matching_batch`](flow-matching.md) draws noise and timesteps and
constructs `z_t`; calling the clean output the final FLUX input would conflate
these two contracts.

## Validation boundaries

Construction checks the stable structural interface needed from the injected
VAE: an `encode()` method, 128 frozen BatchNorm running statistics, a positive
`batch_norm_eps`, and floating-point parameters. The forward hot path trusts
the already validated target-image batch and does not repeat full tensor scans.

Unit tests establish:

- exact 2×2 channel order and row-major token order;
- BatchNorm normalization and `[K,900,128]` output shape;
- posterior-mode execution under `no_grad()`;
- permanent VAE freezing and evaluation mode;
- VAE dtype transfer; and
- downstream trainable-parameter gradients without VAE gradients.

The microbatch tests independently establish the fixed `(0,h,w,0)` image
coordinates.

## Measured real-checkpoint preflight

On 2026-08-19, the production encoder was compared offline with the native
packing helpers from `diffusers==0.39.0` using the pinned VAE revision
`8c44a2fbef88fae175da65df054db0f901aa9747`. The run used CPU FP32 and two
canvases rendered with `greedy_rgb_maximin_v1` from the existing local model
cache; no download occurred.

Measured results:

- clean latent shape: `[2,900,128]`;
- all clean latents finite;
- maximum absolute difference from the pinned Diffusers reference: `0.0`;
- maximum absolute difference across two encoder calls: `0.0`;
- every VAE parameter frozen and VAE evaluation mode active.

Before training, the same explicit comparison must be rerun under the selected
CUDA precision and attention/training environment. The checkpoint check must
remain an external preflight rather than a hidden first-forward branch.
