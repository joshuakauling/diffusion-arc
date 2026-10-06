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
- **Two-GPU DDP:** completed training-step benchmarks and a three-update
  numerical comparison against single-GPU execution that passed the predefined
  loss and gradient tolerances.
- **Training services:** configuration-driven validation, sampling, W&B logging,
  checkpointing, and resume. Validation of the complete distributed GPU
  lifecycle remains ongoing.

The [performance reports](docs/performance/README.md) and
[DDP investigation](docs/performance/ddp.md) describe the methods, results, and
limitations.

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
- **Sharding and memory:** validate FSDP2 against the DDP reference and measure
  the effects of sharding and activation checkpointing.
- **Scaling and performance:** compare execution strategies at fixed global
  batches and seeds, extending to 4 and 8 GPUs where hardware permits. Use
  complete-step measurements to guide optimization.
- **Broader ARC training:** test whether the single-task findings transfer to a
  wider task distribution, develop the training recipe through bounded
  experiments, and fix the recipe, compute budget, and checkpoint-selection
  rule before final training.
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
