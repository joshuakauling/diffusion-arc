# FLUX.2 VAE smoke test

## Objective

This smoke test checks whether the frozen FLUX.2 autoencoder preserves exact
ARC cell colors and spatial structure when a 30x30 ARC canvas is rendered at
8, 16, or 32 pixels per cell. It isolates the VAE from NVARC and the FLUX
transformer.

FLUX.2 applies a fixed 16x spatial reduction between pixels and packed,
transformer-facing latents:

1. the VAE reduces each image dimension by 8x and produces 32 channels;
2. FLUX.2 packs each 2x2 VAE-latent patch into 128 channels;
3. the packed latents are normalized with the VAE's frozen BatchNorm statistics.

## Reproducibility

- Model: `black-forest-labs/FLUX.2-klein-base-4B`
- Component: `vae`
- Model revision: `8c44a2fbef88fae175da65df054db0f901aa9747`
- Expected weight SHA256:
  `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04`
- Posterior statistic: mode, without sampling
- Precision: FP32
- Seed: `20260722`
- ARC padding diagnostic: class 10 rendered as RGB white `(255, 255, 255)`

Create the environment and run the test:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python scripts/diagnostics/vae_smoke_test.py --device mps --cell-size 16
```

After the first run, run the three-way comparison without network access:

```bash
.venv/bin/python scripts/diagnostics/vae_smoke_test.py --device mps --cell-size 8 --local-files-only
.venv/bin/python scripts/diagnostics/vae_smoke_test.py --device mps --cell-size 16 --local-files-only
.venv/bin/python scripts/diagnostics/vae_smoke_test.py --device mps --cell-size 32 --local-files-only
```

Generated metrics and PNG comparisons are written below
`artifacts/vae-smoke/<width>x<height>/` and are intentionally excluded from
version control.

## Diagnostic patterns

- all ARC colors and padding as vertical bands;
- isolated single cells on the padding color;
- one-cell-wide horizontal and vertical lines;
- a high-frequency multicolor checkerboard;
- a seeded, class-balanced random grid.

The visual comparison shows the source image, raw VAE reconstruction, nearest
palette reconstruction, and amplified absolute RGB error. A separate PCA image
maps every 128-dimensional transformer-facing spatial token to RGB for visual
inspection. PCA colors are diagnostic coordinates, not ARC colors.

## Acceptance criteria

- the exact model revision and weight hash load successfully;
- every VAE parameter is frozen and the module is in evaluation mode;
- tensor shapes and FP32 dtype match the documented interface;
- all critical tensors are finite;
- repeated deterministic encoding agrees within `1e-6`;
- cell-level and exact-grid accuracy are reported without assuming success;
- a second local-files-only run reproduces the measurements.

## Measured results

Measured on 2026-07-22 and 2026-07-23 with an Apple M1, 8 GB unified memory,
macOS 15.5, Python 3.11.9, PyTorch 2.13.0, and Diffusers 0.39.0 using the MPS
device without VAE tiling.

- The downloaded VAE contained 84,046,115 parameters and occupied 168,120,878
  bytes. Its SHA256 matched the expected model artifact.
- All parameters were frozen, evaluation mode was active, and all checked
  tensors were finite FP32 values.
- Repeated encoding had a maximum absolute difference of `0.0`.
- Patchify/normalize/denormalize/unpatchify introduced at most
  `4.76837158203125e-7` absolute error.
- Every resolution classified all 4,500 cells correctly and reconstructed all
  five grids exactly after nearest-palette classification. Across the three
  resolutions this is 13,500/13,500 cells and 15/15 exact grids per run.
- A second complete `--local-files-only` run reproduced all non-timing JSON
  metrics and every generated PNG byte-for-byte.

| Image | Pixels per cell | Raw VAE latent | Packed FLUX latent | FLUX tokens per ARC cell | Cell accuracy | Pixel PSNR range | Pixel MAE range |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 240x240 | 8 | `[1,32,30,30]` | `[1,128,15,15]` | 0.25 | 100% | 24.91-36.07 dB | 0.00517-0.03727 |
| 480x480 | 16 | `[1,32,60,60]` | `[1,128,30,30]` | 1 | 100% | 32.08-39.64 dB | 0.00455-0.01422 |
| 960x960 | 32 | `[1,32,120,120]` | `[1,128,60,60]` | 4 | 100% | 34.91-40.60 dB | 0.00443-0.01013 |

Timing was diagnostic rather than benchmark-grade and varied with system load.
The mean encode/decode time across the five patterns ranged from 0.91-2.20
seconds at 240x240, 3.53-7.64 seconds at 480x480, and 73.86-83.78 seconds at
960x960 across the two comparison runs. The untiled 960x960 runs completed
without an MPS out-of-memory error or CPU fallback.

The images show that the remaining RGB error is concentrated around cell
boundaries and isolated shapes. The error is visibly strongest at 240x240 and
weakest at 960x960, but the extra pixel fidelity did not improve cell-level or
exact-grid accuracy on this suite.

The comparison supports 480x480 for the initial architecture. At 240x240 one
FLUX token covers a 2x2 block of ARC cells, so downstream grid decoding would
need four spatial predictions per token or an upsampling stage. At 960x960 each
ARC cell produces a 2x2 block of FLUX tokens, requiring pooling or a
higher-resolution readout while increasing the token count from 900 to 3,600.
480x480 is the only tested size with direct one-token-per-cell alignment.

## Limitations

This test does not validate NVARC conditioning, the FLUX transformer, flow
matching, or end-to-end ARC solving. Synthetic grids test interface fidelity
but do not estimate ARC task-solving quality. White padding is the sentinel
selected for the initial flow-only architecture and remains an experimental
variable for later variants.

The timing measurements do not control background load, warm-up state, or
thermal throttling and must not be treated as a formal performance benchmark.

The run emitted a non-failing deprecation warning because Diffusers passes the
deprecated `local_dir_use_symlinks` argument to Hugging Face Hub 1.24.0. It did
not affect the cached artifact, hash verification, offline execution, or
measured outputs.

## Primary sources

- [FLUX.2 official inference repository](https://github.com/black-forest-labs/flux2)
- [FLUX.2 Klein Base 4B model repository](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [Diffusers FLUX.2 Klein pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
- [ARC-AGI-2 task repository](https://github.com/arcprize/ARC-AGI-2)
