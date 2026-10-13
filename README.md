# Diffusion-ARC

**Distributed training for a pretrained flow transformer, applied to ARC reasoning.**

Diffusion-ARC develops a reproducible training system for full-parameter
fine-tuning of **FLUX.2-klein-base-4B**, conditioned by a frozen
**NVARC/Qwen3-4B** model.

The project connects two goals: building and measuring an efficient training
pipeline across single-GPU, DDP, and FSDP2 execution, and investigating whether a
flow transformer can turn NVARC's representations into exact ARC output grids.

ARC-AGI-2 is the target research setting. Current engineering benchmarks and
preliminary learning experiments also use ARC-AGI-1. The training system,
numerical validation, and performance findings are project results independently
of the eventual ARC accuracy.

## How it works

![Diffusion-ARC architecture](docs/architecture/diagrams/architecture.svg)

NVARC reads the serialized demonstrations and test input. Its hidden states
condition FLUX, which generates the output grid in latent image space. The VAE
decodes the generated latents into an image, which is converted back into ARC
colors and dimensions.

| Component | Role | Training |
| --- | --- | --- |
| NVARC / Qwen3-4B | Encode ARC demonstrations and inputs | Frozen |
| FLUX.2-klein-base-4B transformer | Generate output-grid latents through flow matching | Fully trainable |
| FLUX.2 VAE | Encode training targets and decode generated latents | Frozen |

The architecture reuses FLUX's native Qwen3 conditioning interface. Hidden states
from NVARC layers 9, 18, and 27 match the existing input projection, so no separate
conditioning adapter is needed.

Target grids occupy the top-left corner of a fixed 30×30 canvas, with white
padding outside the grid. Rendering at 480×480 pixels and applying the VAE's
latent packing produces 900 image tokens—one spatial token per ARC cell. Output
dimensions are inferred from the generated canvas.

See the [architecture description](docs/architecture/initial-flow-only-architecture.md)
for the training objective, representations, and decoding procedure.

## Current status

The project is under active development. Existing work includes:

- **Data and model integration:** reproducible task augmentation, task-level
  split handling, frozen NVARC conditioning, target rendering, VAE encoding, and
  full-parameter FLUX training.
- **Single-GPU training and profiling:** a measured training pipeline using
  FlashAttention-3 sequence packing, with documented optimization experiments
  and end-to-end benchmarks.
- **Distributed training:** completed DDP and FSDP2 benchmarks on two, four,
  and eight H200 GPUs. DDP with background joint packing is the preferred
  strategy for the measured workload; FSDP2 remains a memory-capacity option.
- **Training services:** configuration-driven validation, sampling, W&B logging,
  checkpointing, and resume. Validation of the complete distributed GPU
  lifecycle remains ongoing.

## Performance highlights

These measurements cover full-parameter FLUX training with frozen NVARC and VAE
encoders on NVIDIA H200 GPUs. Throughput counts training targets across all GPUs.

| Finding | Measured result |
| --- | --- |
| [Single-GPU pipeline, B0 → B3](docs/performance/README.md#results) | **69.7% higher throughput**, from 3.094 to 5.250 targets/s; median step time fell from 15.45 to 9.06 s. |
| [DDP on four and eight GPUs](docs/performance/fsdp2.md#four-eight-gpu-results) | **20.747 and 40.845 targets/s**: 1.97× throughput when both GPU count and global batch doubled, from 96 to 192 targets. |
| [DDP versus FSDP2](docs/performance/fsdp2.md#implications) | FSDP2 saved peak allocated memory but delivered **1.7% and 2.9% less throughput** on four and eight GPUs, despite larger packs. |

B0 and B3 each combine three repetitions of 50 measured updates; the four/eight-GPU
configurations each use one repetition of 25. Every repetition excludes ten
warm-up updates. Timing includes data retrieval, encoder preparation, and the
complete training/optimizer step; model loading, validation, sampling, logging,
and checkpoints are excluded. The four-to-eight-GPU result measures scaling at
24 targets per GPU, rather than a fixed global batch.

These bounded benchmarks do not establish long-run numerical equivalence.
The initial single-GPU/FSDP2 comparison failed its loss tolerance, and the latest
planner/strategy combinations still need dedicated GPU numerical validation.
See the [numerical evidence](docs/performance/fsdp2.md#numerical-evidence-and-limits)
and [benchmark reports](docs/performance/README.md) for methods and limitations.

## Preliminary ARC result

Broader ARC-AGI-1 training completed five epochs and continued partway through
epoch six. In the documented checkpoint comparison, checkpoint 4830 with four
Euler steps achieved **8/100 exact-grid pass@1 and 9/100 pass@2** on the fixed
100-task validation set, without test-time training, augmentation, or voting.

This is a validation result used to compare checkpoints and sampling budgets;
the 300 reserved final-test tasks were not used. NVARC's prior benchmark exposure
limits claims of unseen-task generalization for the complete system. See the
[checkpoint comparison](docs/training/arc-agi-1-recipe.md#checkpoint-comparison-4530-and-4830-2026-09-22)
for the sampling sensitivity, controls, and remaining failure modes.

## Getting started

The Python package supports Python 3.11 and 3.12. The measured training setup
uses NVIDIA H200 GPUs and a CUDA environment with FlashAttention-3. Model
checkpoints and prepared datasets must be available at the paths specified by
the selected configuration.

Start with the [training guide](docs/training/README.md) and
[data preparation documentation](docs/data/training-data-assembly.md). Training
uses one entrypoint with an explicit configuration:

```bash
# Single GPU
python scripts/train.py --config <training-config.json>

# Two GPUs, using a matching DDP configuration
torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config <training-config.json>
```

Configurations define model and data paths, execution strategy, batch planning,
optimization, validation, and checkpoint behavior.

Further documentation:

- [Architecture and model interfaces](docs/architecture/initial-flow-only-architecture.md)
- [ARC data pipeline](docs/data/data-pipeline.md)
- [Training system](docs/training/README.md)
- [Performance benchmarks](docs/performance/README.md)
- [Research experiments](docs/experiments/README.md)
- [Full documentation index](docs/_README.md)

## Planned evaluation

FLUX and direct autoregressive sampling from NVARC will be compared on the same
held-out tasks using **exact-grid pass@1 and pass@2**, without test-time training,
test-time augmentation, or voting.

Held-out tasks and their augmentations are excluded from FLUX training. Final
test tasks are also excluded from tuning and checkpoint selection.
Interpretation will account for
[NVARC's prior training exposure](docs/architecture/nvarc-overview.md).

## Future work

- **Distributed validation:** complete GPU lifecycle checks for validation,
  sampling, checkpointing, interrupted runs, and reproducible continuation.
- **Numerical validation:** check the final planner/strategy combinations and
  resolve the initial FSDP2 loss-tolerance failure before claiming equivalence.
- **Scaling and memory:** extend the bounded results with repeated runs and
  comparisons at fixed global batches and seeds. Revisit FSDP2 and activation
  checkpointing when a concrete memory constraint requires them.
- **ARC learning:** improve the broader-task training recipe through bounded
  experiments, including sampling-budget and checkpoint comparisons. Fix the
  recipe, compute budget, and checkpoint-selection rule before final training.
- **Final evaluation:** compare FLUX and autoregressive NVARC on held-out tasks,
  including exact-grid accuracy, failure modes, and sampling cost.

FP8 and custom kernel experiments are possible later extensions, after the core
training and evaluation work.

## Scope

The project focuses on full-parameter FLUX fine-tuning within a single node.
NVARC and the VAE remain frozen.

The initial scope excludes conditioning adapters, task-specific output heads,
visual reference images, ground-truth output dimensions at inference, and
multi-node training. State-of-the-art ARC performance is not a project
requirement.
