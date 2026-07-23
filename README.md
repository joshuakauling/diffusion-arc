# diffusion-arc

## Overview

Diffusion-ARC investigates whether a pretrained flow-matching image model can be adapted into a spatial solver for ARC-AGI-2.

The system combines two complementary components:

1. **NVARC/Qwen3-4B**, a language model specialized in serialized ARC tasks, provides task-level reasoning representations.
2. **FLUX.2-klein-base-4B**, a pretrained flow transformer, processes the task conditioning and generates a spatial representation of the target grid.

NVARC remains frozen, while the complete FLUX transformer is fine-tuned using
full-parameter distributed training. The initial architecture uses only FLUX's
original velocity head and flow-matching objective. It adds no conditioning
adapter, color head, or shape head.

The project has two goals:

- Evaluate whether visual flow-transformer pretraining provides useful spatial priors for ARC reasoning.
- Build and analyze a reproducible full-parameter distributed training system for a 4B flow transformer.

## Research Questions

The primary research question is:

> Can a pretrained flow transformer, conditioned by an ARC-specialized language model, solve ARC-AGI-2 tasks more reliably than purely autoregressive or task-specific spatial decoders?

Supporting questions include:

- Does NVARC conditioning outperform the native FLUX text encoder?
- Can the original flow-matching objective recover exact discrete grids without
  a task-specific prediction head?
- Can output dimensions be recovered from a generated padded canvas without a
  separate shape predictor?
- Does FLUX provide value beyond a substantially smaller 2D transformer decoder?
- What are the memory, throughput, and numerical trade-offs of DDP and FSDP2 for full-parameter FLUX fine-tuning?

## Architecture

```text
Serialized ARC demonstrations and test input
                    │
                    ▼
       Frozen NVARC / Qwen3-4B
                    │
       Hidden states from selected layers
                    │
       concatenate directly to width 7680
                    │
                    ▼
        existing FLUX text projection
                 7680 → 3072
                    │
                    ▼
        FLUX.2-klein-base-4B
          full-parameter training
                    │
          Original Flow Head
                 3072 → 128
                    │
                    ▼
          Packed-latent velocity
                    │
             ODE integration
                    │
                    ▼
          Frozen FLUX.2 VAE decoder
                    │
                    ▼
      Deterministic palette and PAD decoding
```

The FLUX VAE and NVARC model remain frozen. The complete FLUX transformer,
including its existing input projections and final velocity head, is trained.
No newly initialized trainable module is required.

The initial implementation uses the NVARC hidden states corresponding to
FLUX's standard Qwen conditioning layers, subject to checkpoint and tensor
compatibility verification. Three width-2,560 hidden states are expected to
concatenate to FLUX's native context width of 7,680 and are passed directly to
the existing `txt_in` projection. The expected ARC serialization may require
roughly 8,000 context tokens rather than the 512-token reference path.

The detailed initial design is specified in
[`docs/initial-flow-only-architecture.md`](docs/initial-flow-only-architecture.md).

## Grid Representation

ARC grids are represented on a fixed 30×30 canvas, corresponding to the maximum
ARC grid dimensions. The actual output is aligned to the top-left. Every
remaining cell is filled with RGB white `(255, 255, 255)`, an eleventh visual
sentinel outside the ten ARC colors.

Each cell is rendered as a solid 16×16 pixel block, producing a 480×480 image.
The frozen VAE and its 2×2 latent packing transform this into a 30×30 grid of
128-dimensional tokens:

\[
[B,3,480,480]
\rightarrow
[B,128,30,30]
\rightarrow
[B,900,128].
\]

This gives one FLUX image token per ARC cell. Output dimensions are not supplied
to the model. They are represented by the generated PAD boundary and recovered
deterministically after VAE decoding.

## Training Objective

For the clean packed target latent `z_0`, sample Gaussian noise `epsilon` and a
flow time `t`, then construct:

\[
z_t=(1-t)z_0+t\epsilon.
\]

The straight-path velocity target is:

\[
u_t=\epsilon-z_0.
\]

The initial and only loss is:

\[
\mathcal{L}_{\text{flow}}
=
\mathbb{E}
\left[
\left\|
v_\theta(z_t,t,c)-(\epsilon-z_0)
\right\|_2^2
\right].
\]

There is no color cross-entropy, auxiliary decoder, adapter loss, or shape
loss in the initial architecture.

## Inference and Grid Decoding

Inference starts from a Gaussian packed latent of shape `[B,900,128]` and
integrates the learned velocity field backward from noise to data. The frozen
VAE decodes the final latent once into a 480×480 RGB image.

The image is divided into 30×30 cells. Each cell's mean RGB value is assigned
to the nearest prototype among the ten ARC colors plus white PAD. The last row
and column containing any non-PAD cell determine the predicted output extent,
and the top-left rectangle is returned as the ARC answer.

An all-PAD canvas or PAD inside the inferred output rectangle is an invalid
prediction. Ground-truth output dimensions must not be used as a fallback.

## Distributed Training

FLUX is fine-tuned with full trainable parameters rather than LoRA. The training system should support:

- single-GPU execution,
- DistributedDataParallel,
- FSDP2 with transformer-block-level sharding,
- BF16 mixed precision,
- activation checkpointing,
- distributed data loading,
- gradient clipping and non-finite detection,
- distributed checkpoint save and resume,
- and profiling with `torch.profiler` and, if practical, Nsight Systems.

The published benchmark should compare 1, 2, and 4 GPUs using:

- training throughput,
- median and p95 step time,
- peak GPU memory,
- scaling efficiency,
- communication overhead,
- and numerical agreement between training strategies.

## Evaluation

Model quality is evaluated using:

- exact grid accuracy,
- per-cell accuracy,
- pass@1 and pass@2 accuracy,
- output-height and output-width accuracy,
- flow loss,
- performance by grid size and transformation category,
- and sensitivity to the number of denoising steps.

All train/validation separation must happen at the ARC task level. Augmented variants of held-out tasks must not occur in the training set.

## Required Baselines

The project should include at least:

1. the original autoregressive NVARC solver,
2. native FLUX-Qwen conditioning versus NVARC conditioning,
3. NVARC conditioning with a small task-specific 2D decoder,
4. pretrained FLUX versus a capacity-controlled non-pretrained spatial model,
5. LoRA versus full-parameter FLUX fine-tuning.

These baselines are necessary to determine whether improvements come from
visual pretraining, the flow objective, the NVARC encoder, or simply additional
model capacity.

Color heads, combined flow-and-color heads, explicit shape heads, and
conditioning adapters are optional later architecture experiments. They are
not required for the initial implementation or its first evaluation.

## Milestones

### Milestone 1 — Interface Validation

- Load and freeze NVARC.
- Verify tokenizer and checkpoint compatibility.
- Extract the required hidden states.
- Pass the concatenated hidden states directly through FLUX's existing text
  projection.
- Pass one ARC example through the complete pipeline.
- Compute flow loss and backpropagate through FLUX only.

### Milestone 2 — Single-GPU Correctness

- Implement the top-left-aligned padded canvas and deterministic grid decoder.
- Overfit a tiny set of ARC examples.
- Add deterministic tests, gradient checks, and NaN detection.
- Confirm that NVARC and VAE remain frozen.

### Milestone 3 — Distributed Training

- Implement DDP and FSDP2.
- Add activation checkpointing and distributed checkpoints.
- Validate loss and gradient agreement.
- Publish initial 1/2/4/8-GPU memory and throughput results.

This is the first application-ready public milestone.

### Milestone 4 — Research Evaluation

- Run the first meaningful training experiments.
- Evaluate exact grid and output-shape accuracy.
- Compare against the required baselines.
- Determine whether flow matching and visual pretraining provide measurable value.

### Milestone 5 — Extensions

Only after the core hypothesis is validated:

- conditioning adapters,
- discrete color heads,
- combined flow and color heads,
- explicit output-shape heads,
- compact visual reference canvases,
- test-time adaptation,
- FP8 experiments,
- Triton kernels,
- and multi-node training.

## Initial Non-Goals

The first version will not:

- fine-tune NVARC,
- add a NVARC-to-FLUX adapter,
- add a color, shape, or other task-specific learned output head,
- use visual reference-image tokens,
- receive ground-truth output dimensions,
- target state-of-the-art ARC performance,
- use FP8 or custom kernels,
- or claim large-scale/multi-node training.

The initial goal is a correct, measurable, and reproducible full-parameter distributed training system, followed by a controlled evaluation of the central research hypothesis.
