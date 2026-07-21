# diffusion-arc

## Overview

Diffusion-ARC investigates whether a pretrained flow-matching image model can be adapted into a spatial solver for ARC-AGI-2.

The system combines two complementary components:

1. **NVARC/Qwen3-4B**, a language model specialized in serialized ARC tasks, provides task-level reasoning representations.
2. **FLUX.2-klein-base-4B**, a pretrained flow transformer, processes the task conditioning and generates a spatial representation of the target grid.

NVARC remains frozen, while the complete FLUX transformer is fine-tuned using full-parameter distributed training. A discrete color-classification head is added alongside the original flow-matching output head to produce exact ARC grids.

The project has two goals:

- Evaluate whether visual flow-transformer pretraining provides useful spatial priors for ARC reasoning.
- Build and analyze a reproducible full-parameter distributed training system for a 4B flow transformer.

## Research Questions

The primary research question is:

> Can a pretrained flow transformer, conditioned by an ARC-specialized language model, solve ARC-AGI-2 tasks more reliably than purely autoregressive or task-specific spatial decoders?

Supporting questions include:

- Does NVARC conditioning outperform the native FLUX text encoder?
- Does the flow-matching objective improve exact grid prediction beyond a discrete color loss?
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
        Trainable residual adapter
                    │
                    ▼
        FLUX.2-klein-base-4B
          full-parameter training
                    │
        Internal target-token features
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Original Flow Head    ARC Color Head
          │                   │
      Velocity          10 colors + PAD
```

The FLUX VAE and NVARC model remain frozen. The following components are trained:

- the complete FLUX transformer,
- the NVARC-to-FLUX conditioning adapter,
- the ARC color head,
- and, in a later stage, an output-shape head.

The initial implementation uses the NVARC hidden states corresponding to FLUX’s standard Qwen conditioning layers, subject to checkpoint compatibility verification.

## Grid Representation

ARC grids are represented on a fixed 30×30 canvas, corresponding to the maximum ARC grid dimensions.

Each position is assigned one of eleven classes:

- the ten ARC colors,
- one additional padding/outside class.

Initially, the correct output dimensions are provided to the model. Shape prediction is intentionally postponed until the transformation itself works reliably.

The target canvas is also rendered as an image and encoded into FLUX latent space for flow-matching training.

## Training Objective

The initial loss is:

\[
L = L_{\text{flow}} + \lambda_{\text{color}} L_{\text{color}}
\]

where:

- `L_flow` is the original FLUX flow-matching loss,
- `L_color` is cross-entropy over the discrete ARC color classes.

A later version may add:

\[
\lambda_{\text{shape}} L_{\text{shape}}
\]

for height and width prediction.

An important ablation removes `L_flow` to determine whether the iterative flow process contributes to solving the task or whether the discrete head is doing all meaningful work.

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
- flow and color loss,
- performance by grid size and transformation category,
- and sensitivity to the number of denoising steps.

All train/validation separation must happen at the ARC task level. Augmented variants of held-out tasks must not occur in the training set.

## Required Baselines

The project should include at least:

1. the original autoregressive NVARC solver,
2. native FLUX-Qwen conditioning versus NVARC conditioning,
3. NVARC conditioning with a small task-specific 2D decoder,
4. FLUX with color loss only,
5. FLUX with combined flow and color loss,
6. LoRA versus full-parameter FLUX fine-tuning.

These baselines are necessary to determine whether improvements come from visual pretraining, the flow objective, the NVARC encoder, or simply additional model capacity.

## Milestones

### Milestone 1 — Interface Validation

- Load and freeze NVARC.
- Verify tokenizer and checkpoint compatibility.
- Extract the required hidden states.
- Pass one ARC example through the complete pipeline.
- Compute both losses and backpropagate through FLUX only.

### Milestone 2 — Single-GPU Correctness

- Implement the fixed canvas and discrete color head.
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
- Evaluate exact grid accuracy.
- Compare against the required baselines.
- Determine whether flow matching and visual pretraining provide measurable value.

### Milestone 5 — Extensions

Only after the core hypothesis is validated:

- output-shape prediction,
- compact visual reference canvases,
- test-time adaptation,
- FP8 experiments,
- Triton kernels,
- and multi-node training.

## Initial Non-Goals

The first version will not:

- fine-tune NVARC,
- use multiple full-resolution reference images,
- predict output dimensions,
- target state-of-the-art ARC performance,
- use FP8 or custom kernels,
- or claim large-scale/multi-node training.

The initial goal is a correct, measurable, and reproducible full-parameter distributed training system, followed by a controlled evaluation of the central research hypothesis.
