# Training System Overview

## Scope

This page explains the overall Diffusion-ARC training flow and links to the
relevant code and detailed documentation. The root
[`README.md`](../../README.md) remains authoritative for project scope,
milestones, evaluation, and non-goals.

## Overall workflow

**Encode each source once. Train its eligible targets from the shared state.**

The ordered CPU loader can prefetch tokenized samples in worker processes. The
training loop takes a fixed window of consecutive sources (16 in the current
benchmark), selects eligible targets, and trims each source before its final
target output. NVARC encodes the entire window in one unpadded FA3 Varlen call.
Sources have separate causal attention boundaries and reset position IDs.

The VAE then encodes the targets in configured chunks (48 in the benchmark).
Each `FluxInput` selects an exact prefix from its shared NVARC sequence state.
Prepared targets enter a FIFO buffer; the next 48 define one optimizer update.
The FLUX scheduler groups only those selected targets into physical microbatches
under a 12,288-token budget, counting each target's exact text prefix and 900
image tokens. It may reorder execution within that fixed batch, while
noise/timesteps remain assigned in logical order and losses are weighted by
target count.

The [FA3 sequence-packing recipe](flux-sequence-packing.md) gives each FLUX
target a separate noncausal attention segment and preserves its own timestep
modulation. Backward sums modulation contributions in FP32. NVARC uses causal
segments and stops after block 27, retaining the raw states after 9, 18 and 27.

Neither component uses length buckets or alignment padding. NVARC has no
scheduler or cross-window lookahead: the logical source window bounds
preparation and retained hidden states. FLUX groups only the already selected
optimizer batch; it does not change membership across steps.

The entrypoint is [`scripts/train_single_gpu.py`](../../scripts/train_single_gpu.py).
Its `DataLoader` uses `batch_size=None`; workers perform loading/tokenization
in advance, while the consumer groups consecutive items into each source window.
The auxiliary `TrainingCollator` remains available to diagnostics but is not
part of this entrypoint.

## Model boundaries

| Component | State | Role |
| --- | --- | --- |
| NVARC/Qwen3-4B | Frozen | Encode complete ARC source sequences. |
| FLUX.2 VAE | Frozen | Encode target canvases into clean image latents. |
| FLUX.2 transformer | Fully trainable | Predict flow velocity from noisy image latents and NVARC conditioning. |
| AdamW | FLUX parameters only | Update the complete FLUX transformer. |

NVARC and the VAE run without training autograd graphs. FLUX parameters and
optimizer state remain FP32, the FLUX forward uses BF16 autocast, and the loss
is reduced in FP32.

## Core objects

| Object | Meaning |
| --- | --- |
| `ArcTokenizedTrainingSample` | One complete tokenized ARC source sequence with its recorded target positions and grids. |
| `NvarcSequenceState` | One source's NVARC hidden states, shared by all of its eligible target inputs. |
| `FluxInput` | One target, its exact prefix boundary, its clean latent, and a reference to the shared sequence state. |
| Private FIFO queue | Prepared target inputs waiting for optimizer-batch selection. |
| `OptimizerBatch` | All targets that contribute to one parameter update. |
| `FluxMicrobatch` | A physical subset of the optimizer batch processed in one FLUX forward and backward pass. |

The shared hidden-state tensor is released after its last target prefix has been
copied into a physical microbatch. Python later collects the small
`NvarcSequenceState` object when no `FluxInput` references it anymore.

Microbatch losses are weighted by target count, so changing the physical
grouping preserves the mathematical objective; finite-precision gradients can
still differ. The detailed batching rules and
loss normalization are documented in the
[`batching-and-efficiency-roadmap.md`](batching-and-efficiency-roadmap.md).

## Training module map

| Module | Responsibility |
| --- | --- |
| [`dataset.py`](../../src/diffusion_arc/training/dataset.py) | Build tokenized ARC training samples from configured task views. |
| [`collation.py`](../../src/diffusion_arc/training/collation.py) | Provide the auxiliary packed CPU batch used by tests and diagnostics. |
| [`single_gpu_config.py`](../../src/diffusion_arc/training/single_gpu_config.py) | Validate model, data, loop, and optimizer settings. |
| [`single_gpu_setup.py`](../../src/diffusion_arc/training/single_gpu_setup.py) | Select the device, seed RNGs, load models, and construct the training components. |
| [`flux_input.py`](../../src/diffusion_arc/training/flux_input.py) | Create target-level inputs from frozen NVARC and VAE outputs. |
| [`flux_microbatch.py`](../../src/diffusion_arc/training/flux_microbatch.py) | Copy exact prefixes and build physical FLUX microbatches. |
| [`flux_schedule.py`](../../src/diffusion_arc/training/flux_schedule.py) | Group one optimizer window by packed token budget or the padded reference cost catalog. |
| [`flux_loss.py`](../../src/diffusion_arc/training/flux_loss.py) | Sample flow states, run FLUX, and compute the velocity loss. |
| [`single_gpu_loop.py`](../../src/diffusion_arc/training/single_gpu_loop.py) | Keep FIFO membership, execute scheduled FLUX batches, and update FLUX. |
| [`flow_probes.py`](../../src/diffusion_arc/training/flow_probes.py) | Measure comparable fixed-noise flow losses. |
| [`__init__.py`](../../src/diffusion_arc/training/__init__.py) | Define the public training-package exports. |

## Further reading

| Document | Details |
| --- | --- |
| [`../data/data-pipeline.md`](../data/data-pipeline.md) | ARC loading, task splits, augmentation, prompts, tokenization, and target canvases. |
| [`../data/training-data-assembly.md`](../data/training-data-assembly.md) | The auxiliary `TrainingCollator` and `PackedTrainingBatch` contract. |
| [`nvarc-batch-runner.md`](nvarc-batch-runner.md) | Frozen NVARC packing, causal boundaries, early exit, and checkpoint evidence. |
| [`flux-text-conditioning.md`](flux-text-conditioning.md) | Exact-prefix and ownership contracts, including the padded reference interface. |
| [`flux-image-encoding.md`](flux-image-encoding.md) | Frozen VAE target encoding, latent packing, and normalization. |
| [`flow-matching.md`](flow-matching.md) | Flow sampling, velocity supervision, loss computation, and validation. |
| [`batching-and-efficiency-roadmap.md`](batching-and-efficiency-roadmap.md) | Historical batching design and the retained FIFO/loss invariants. |
| [`flux-sequence-packing.md`](flux-sequence-packing.md) | Production FA3 packing, FP32 modulation sums, token budgeting, and validation limits. |

## Validation

Run the local unit-test suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B -m unittest discover -s tests
```

The local tests cover tensor contracts, hidden-state ownership, loss weighting,
step limits, and tail handling. Real checkpoints and backend-specific behavior
remain explicit smoke or preflight tests.

## Current boundaries

- Training is single-process and single-GPU; DDP and FSDP2 are not implemented.
- Schema v8 is the combined configuration contract. Historical schemas and
  NVARC scheduler/bucket options are rejected rather than silently translated.
- The [B3 benchmark](../../configs/training/benchmarks/single_gpu_b3.json),
  [three-step smoke](../../configs/training/benchmarks/single_gpu_b3_smoke.json),
  and [profiler](../../configs/training/profiling/single_gpu_b3_torch_profile.json)
  all combine NVARC and FLUX packing.
- The shared Modal image requires Hopper/SM90 and BF16 FA3, with head dimension
  128 and both forward and backward support.
- Combined B3 numerical and performance validation on real GPU checkpoints
  remains outstanding. Separate component diagnoses do not establish full-step
  capacity, throughput, or learning equivalence.
