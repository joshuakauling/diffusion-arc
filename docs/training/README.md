# Training System Overview

## Scope

This page explains the overall Diffusion-ARC training flow and links to the
relevant code and detailed documentation. The root
[`README.md`](../../README.md) remains authoritative for project scope,
milestones, evaluation, and non-goals.

The [first ARC-AGI-1 training recipe](arc-agi-1-recipe.md) records the agreed
dataset, augmentation, objective, optimizer and validation setup for the next
full-epoch run, including logging and checkpoint continuation.

## Overall workflow

**Encode each source once. Train its eligible targets from the shared state.**

The flow below describes the [B3 single-GPU baseline](../performance/b3.md).

The ordered CPU loader prefetches tokenized samples. An outer loop groups whole
sequences by their eligible target count using `iter_optimizer_batches`. In the
current recipe, 16 five-pair sequences provide 48 targets; the final ten sources
provide 30. Each group is passed to `Trainer.step(sequences)` for one update.
No prepared targets are carried between steps.

The step trims each source before its final target output and calls frozen
NVARC under a **134,400-token budget** (8,400 × 16). Smaller budgets split whole
sources into consecutive packs while preserving logical order. Each source has
separate causal attention and reset position IDs. NVARC stops after block 27,
retaining raw states after blocks 9, 18 and 27. The VAE then encodes target
canvases in independent image batches (48 in the current recipe).

Flow noise/times are assigned in logical target order. FLUX packs those targets
under its separate **12,288 joint-token budget**, counting each prefix plus 900
image tokens. Backward weights each pack by its share of the actual target
count. One optimizer update and one scheduler advance follow all packs.
Gradients are cleared with `set_to_none=True` before the step returns, leaving
memory available for validation or sampling. Construction clears gradients once
before the first step and sets FLUX to training mode; NVARC/VAE remain frozen.

`TrainStepResult` contains total loss, gradient/LR metrics, total step time
including encoding and workload counters. Its `targets` tuple holds CPU-only
measurements in source order, then eligible target order: loss, sampled time,
demonstration count, boundary presence, regional MSEs and weighted contributions.
The tuple index also identifies the target in the packing diagnostics. Regional
measurements are `None` when the objective does not provide them; a 30×30 target
has `has_boundary=False`, no boundary MSE and zero regional boundary contribution.
Detailed preparation and packing metadata live separately under `diagnostics`.
No tensors or autograd graphs leave the step.

The outer loop owns global progress, logging and event timing. Logging derives
endpoint/uniform and demonstration groups from individual target measurements,
including their counts and combinations across steps. The
[training entrypoint](../../scripts/train_single_gpu.py) loads a validated
`TrainingConfig` and passes it to
[`run_training`](../../src/diffusion_arc/training/runner.py). The runner builds
the components, consumes one complete epoch and checks its progress totals.
[`logging.py`](../../src/diffusion_arc/training/run/logging.py) derives the configured
training scalars, reads selected hardware counters and sends one combined W&B
record per update. Validation prepares fixed CPU caches once and evaluates them
at the configured steps, including the initial model. Its losses and workload
counters join the same update record. Fixed-seed sampling reuses a subset of
that cache, calls the existing inference pipeline and adds scores plus a W&B
comparison table. Full-state checkpoints follow sampling at the configured
steps. A resume restores the optimizer, scheduler, RNGs and committed data cursor,
then continues the same W&B run without replaying existing history records.
The separate [profiling entrypoint](../../scripts/profile_single_gpu.py) retains
the B3 Torch/Nsight workflow. The auxiliary `TrainingCollator` serves diagnostics
and is not part of the production step.

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
| `EncodedTrainingBatch` | Prepared targets plus CPU metadata for their NVARC/VAE execution. |
| `TrainStepResult` | Update scalars, workload counters, target measurements and separate diagnostics returned to the outer loop. |
| `TargetMetrics` | One target's loss, flow time, demonstration count and regional measurements, before logging aggregation. |
| `StepDiagnostics` | CPU-only preparation details, physical FLUX packs and logical target prefix lengths. |
| `FluxMicrobatch` | Exact text prefixes, attention-segment boundaries and image positions for one packed FLUX call; flow tensors are supplied separately. |

The shared hidden-state tensor is released after its last target prefix has been
copied into a physical microbatch. Python later collects the small
`NvarcSequenceState` object when no `FluxInput` references it anymore.

Microbatch losses are weighted by target count, so changing the physical
grouping preserves the mathematical objective; finite-precision gradients can
still differ. The [FLUX packing guide](flux-sequence-packing.md) documents
physical grouping and loss normalization, including a smaller final batch.

## Package responsibilities

`arc_data` defines ARC tasks, augmentation, prompts, and tokenization. `model`
defines neural components, forward operations, and their tensor interfaces.
`training` composes those components with targets, objectives, optimizer steps,
and validation. `inference` generates predictions from target-free conditioning.
Neither `model` nor `inference` imports training code; validation passes only
conditioning to generation and scores the returned predictions afterward.

Within `model`, `nvarc/` contains the frozen encoder and explicit attention
binding. `flux/` contains the transformer, text conditioning, velocity prediction,
packing and modulation. The independent VAE encoder/decoder lives in
[`model/vae.py`](../../src/diffusion_arc/model/vae.py); shared tensor interfaces
remain in [`model/contracts.py`](../../src/diffusion_arc/model/contracts.py).
The component packages import no backends on their own; model setup still
explicitly registers NVARC attention and selects the FLUX packing backend.

## Training module map

`trainer.py` and `runner.py` are the two entrypoints. `step/` contains the
components of one optimizer update; `run/` contains validation, logging and
checkpoint services. Shared configuration, setup, data loading and optional
profiling remain at the package root.

| Module | Responsibility |
| --- | --- |
| [`config.py`](../../src/diffusion_arc/training/config.py) | Parse baseline and run JSON formats directly into one `TrainingConfig`. |
| [`setup.py`](../../src/diffusion_arc/training/setup.py) | Select the device, seed RNGs, load frozen/trainable models, and construct AdamW. |
| [`data.py`](../../src/diffusion_arc/training/data.py) | Build the dataset, DataLoader and whole-sequence optimizer batches; retain auxiliary CPU collation. |
| [`trainer.py`](../../src/diffusion_arc/training/trainer.py) | Execute one complete update from supplied sequences and return aggregated step metrics. |
| [`runner.py`](../../src/diffusion_arc/training/runner.py) | Build or restore components, iterate the remaining epoch and run scheduled evaluation/checkpoint events. |
| [`profiling.py`](../../src/diffusion_arc/training/profiling.py) | Record explicit benchmark measurements and optional profiler ranges. |
| [`step/encoding.py`](../../src/diffusion_arc/training/step/encoding.py) | Encode source sequences and target grids through frozen NVARC and VAE models. |
| [`step/flux_microbatch.py`](../../src/diffusion_arc/training/step/flux_microbatch.py) | Build physical microbatches, execute FLUX, and apply the selected loss. |
| [`step/flux_batching.py`](../../src/diffusion_arc/training/step/flux_batching.py) | Group one optimizer window by packed token budget. |
| [`step/flow_sampling.py`](../../src/diffusion_arc/training/step/flow_sampling.py) | Draw noise/times and construct straight-path supervision, including endpoint quotas. |
| [`step/loss.py`](../../src/diffusion_arc/training/step/loss.py) | Define plain MSE and equal-target canvas/grid/boundary reductions. |
| [`step/lr_schedule.py`](../../src/diffusion_arc/training/step/lr_schedule.py) | Construct linear-warmup constant or cosine learning-rate schedules. |
| [`run/logging.py`](../../src/diffusion_arc/training/run/logging.py) | Reduce measurements, render prediction tables, collect selected hardware counters and own the W&B run. |
| [`run/validation.py`](../../src/diffusion_arc/training/run/validation.py) | Verify the held-out bundle, prepare fixed CPU caches, evaluate loss and score generated predictions. |
| [`run/checkpointing.py`](../../src/diffusion_arc/training/run/checkpointing.py) | Save HF transformer weights and training state, verify checkpoint contracts, restore RNG/optimizer state and apply retention. |

Import components from their owning modules. Loading `training.config` requires
neither PyTorch nor model initialization. The outer caller selects and constructs
models, optimizer, sampler, loss and scheduler. `Trainer` accepts those components
and execution limits directly, without a run config or logging callbacks. Its
only public execution method is `step`; it does not own epochs or stop criteria.
Logging and checkpoints serialize the recipe through `TrainingConfig.recipe_dict()`.
Training and validation share `plan_nvarc_batches` from `step.encoding`.

## Further reading

| Document | Details |
| --- | --- |
| [`../data/data-pipeline.md`](../data/data-pipeline.md) | ARC loading, task splits, augmentation, prompts, tokenization, and target canvases. |
| [`../data/training-data-assembly.md`](../data/training-data-assembly.md) | The auxiliary `TrainingCollator` and `PackedTrainingBatch` contract. |
| [`nvarc-batch-runner.md`](nvarc-batch-runner.md) | Frozen NVARC packing, causal boundaries, early exit, and checkpoint evidence. |
| [`flux-text-conditioning.md`](flux-text-conditioning.md) | Training prefix ownership and the separate dense inference interface. |
| [`flux-image-encoding.md`](flux-image-encoding.md) | Frozen VAE target encoding, latent packing, and normalization. |
| [`flow-matching.md`](flow-matching.md) | Flow sampling, velocity supervision, loss computation, and validation. |
| [`batching-and-efficiency-roadmap.md`](batching-and-efficiency-roadmap.md) | Superseded design note for the padded pipeline and proposed lookahead schedulers. |
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
- Baseline JSON uses schema v10; the finite epoch recipe uses schema v1 with
  `config_type="training_run"`. Both normalize to `TrainingConfig`; entrypoints
  explicitly restrict the supported type. Older baseline schemas and NVARC
  scheduler/bucket options and `loop.nvarc_source_batch_size` remain rejected.
  NVARC execution uses `nvarc_max_tokens`; optimizer size remains a target count.
- Training always uses FLUX packing with an explicit token budget. Padded
  training, fixed FLUX microbatch sizes, and measured padding-cost catalogs
  have been removed; retired fields are rejected even when set to null.
- The [B3 benchmark](../../configs/training/benchmarks/single_gpu_b3.json),
  [three-step smoke](../../configs/training/benchmarks/single_gpu_b3_smoke.json),
  and [profiler](../../configs/training/profiling/single_gpu_b3_torch_profile.json)
  all combine NVARC and FLUX packing.
- The shared Modal image requires Hopper/SM90 and BF16 FA3, with head dimension
  128 and both forward and backward support.
- The [B3 benchmark](../performance/b3.md#benchmark-result) measures combined
  throughput, peak memory, and step-loss differences from B2 on H200.
  Gradient/update equivalence and long-term learning quality remain open.
