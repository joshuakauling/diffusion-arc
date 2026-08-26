# Single-Task Conditioning Overfitting Ladder

## Status and question

This document records the preregistration and one pre-training protocol
amendment for a three-stage diagnostic on ARC task `0d3d703e`. It keeps the
primary fixed 30x30 canvas, frozen NVARC, frozen FLUX VAE, full
FLUX-transformer training, native velocity head, and unweighted flow-matching
loss.

The question is:

> Can a fresh FLUX transformer first memorize one fixed target, then route two
> contradictory NVARC conditions, and finally overfit eight repeated
> conditions when every occurrence receives fresh flow noise and a stratified
> timestep?

The experiment is a same-task trainability diagnostic. It does not establish
held-out ARC-task generalization or replace the project's primary evaluation.

## Motivation

The preceding spatial ablation optimized 512 different conditions exactly
once, producing only 64 optimizer updates. It learned the shared stripe
structure of `0d3d703e`, but correct and shuffled NVARC conditioning produced
nearly equal fixed-flow loss. Some of those views also moved an incompletely
demonstrated pair into the target position.

This ladder changes one central variable: a small fixed set of conditions is
repeated across many optimizer steps with independently sampled Gaussian noise
and broad flow-time coverage. It also fixes the original fifth pair as the
target in every condition. Only original pairs 0 through 3 appear as
demonstrations.

## Pre-training VAE protocol amendment

The first execution, run `20260825T233807503991Z` from source commit
`d7cc6632ad6bde7bef177a959670fcb40ad499ec`, stopped before any optimizer was
constructed because the clean frozen-VAE encode/decode round trip was exact
for 16 of the 17 conditions. No FLUX learning result or stage-zero generation
was observed before this amendment.

The fixed palette and all 17 deterministic conditions remain unchanged. The
clean-VAE check now records the expected grid, decoded grid, validity, inferred
shape, cell accuracy, and mismatched cells for every condition. The run may
continue only when canonical is exact and at least 16 of 17 conditions are
exact. Thus at most one noncanonical clean-VAE failure is tolerated and made
visible in every relevant panel summary. Two failures or any canonical failure
still stop the run before training.

This tolerance isolates the requested FLUX/NVARC overfitting diagnostic while
the broader VAE representation issue is investigated separately. It does not
reclassify a VAE-limited condition as an exact model success, regenerate data,
or change the target palette after inspecting the failure.

## Conditions

The runner constructs 17 deterministic conditions directly from the raw task:

- eight training conditions, one for each square-grid geometry, with distinct
  cyclic color permutations and original demonstration order;
- eight unseen conditions with the same geometry coverage, disjoint affine
  color permutations, and deterministic permutations of demonstrations 0--3;
- one canonical identity condition.

Every augmentation key ends in pair index 4. Runtime gates verify that the
four preceding indices are exactly `{0, 1, 2, 3}`, that every target color
mapping is directly demonstrated, that all target grids are unique, and that
the two routing targets differ in every cell.

All 17 target-free prefixes are encoded in one physical NVARC batch. Their
hidden states and the 17 frozen-VAE target latents are cached before any FLUX
optimizer is constructed. The detailed clean-target-latent round trip is saved
before its acceptance gate is evaluated, so an aborted run still identifies
the exact failing conditions.

## Independent stages

Every stage loads a fresh complete FLUX transformer from the same verified
basis checkpoint. Parameter fingerprints must match before training.

| Stage | Conditions per optimizer batch | Updates | Target exposures |
| --- | --- | ---: | ---: |
| `one-condition` | canonical repeated 8 times | 256 | 2,048 |
| `two-condition-routing` | canonical and one ID color permutation, 4 each | 512 | 4,096 |
| `eight-condition-overfit` | eight fixed training conditions, 1 each | 512 | 4,096 |

Stage 1 is an unconditional mechanics control: success does not prove that the
model uses NVARC because it may ignore the single repeated condition. Stage 2
is the decisive conditioning test. Its conditions have identical geometry and
demonstration order but contradictory color targets. Stage 3 asks whether the
same direct conditioning path can memorize eight fixed views.

For uniform reporting, canonical is also emitted as a separate panel in every
stage. It therefore duplicates the seen condition in stages 1 and 2; the
report records `canonical_is_seen=true` for those stages. Stage 1 has no
meaningful swapped-condition probe and records routing as unavailable.

## Stratified flow sampling

Every optimizer batch contains exactly one jittered timestep from each of
eight equal intervals:

```text
t_i = (i + u_i) / 8,  u_i ~ Uniform(0, 1),  i = 0, ..., 7
```

The eight values are randomly assigned to batch rows, and every row receives
independent Gaussian noise. The training loss remains the unweighted mean
squared velocity error. Per-exposure total, 3x3-grid, and padding losses are
recorded with the corresponding timestep stratum. Runtime summarization
requires every stratum to contain exactly one exposure per optimizer step.

## Evaluation

Stage 1 evaluates at updates `0, 32, 64, 128, 256`. Stages 2 and 3 additionally
evaluate at update `512`.

Every evaluation contains the stage's seen conditions, all eight unseen
conditions, and canonical, using two generation seeds and 50 linear Euler
steps. It records:

- exact-grid pass@1 and pass@2;
- valid-grid, shape, per-cell, and oracle top-left 3x3 diagnostics;
- total, true-grid, and padding velocity loss;
- teacher-forced clean-latent reconstruction at `t = 0.1, 0.5, 0.9`;
- per-stratum training loss across all eight timestep intervals;
- contact sheets for correct generation at every evaluation;
- final same-noise swapped generation for stages 2 and 3.

Each stage report additionally stores the clean-VAE exact reference for every
condition panel, and the final cross-stage comparison places those reference
counts beside the raw generation results.

For the routing probe, each target row keeps exactly the same `z_t`, timestep,
and Gaussian noise while only its NVARC condition is exchanged. Final correct
and swapped generations reinitialize the generator with the same seed, batch
shape, and row order. Swapped outputs are diagnosed both against the original
row target and the target belonging to the exchanged condition.

## Interpretation gates

- Stage 1 failure localizes the immediate problem before conditional routing:
  flow optimization, the padded objective, or training/sampling consistency.
- Stage 1 success with stage 2 failure localizes the bottleneck to the
  NVARC-to-FLUX conditioning path or its representation distribution.
- Stage 2 exact success demonstrates that direct NVARC conditioning can route
  contradictory outputs for one ARC task.
- Stage 2 success with stage 3 failure indicates interference or insufficient
  repeated exposure across the larger fixed condition set.
- Seen success without unseen success is presentation memorization, which is
  sufficient for this overfitting diagnostic but not augmentation transfer.
- A model failure on a condition whose clean target latent already fails the
  VAE round trip cannot be attributed solely to FLUX. Raw exact-grid results
  remain unchanged, while the report exposes the corresponding clean-VAE
  reference count beside every stage summary.

Run status `COMPLETE` means the technical protocol, checkpoint uploads, and
report upload completed. It never implies that a learning criterion passed.

## Artifacts and non-goals

The executable notebook is
[`experiments/training/single-task-conditioning-overfitting-ladder-modal.ipynb`](../../experiments/training/single-task-conditioning-overfitting-ladder-modal.ipynb).
The source-controlled runner is
[`scripts/experiments/run_single_task_conditioning_ladder.py`](../../scripts/experiments/run_single_task_conditioning_ladder.py).

Results are written under:

```text
/mnt/diffusion-arc-models/results/single-task-conditioning-overfitting-ladder/<run-id>/
```

One locally validated Safetensors checkpoint is uploaded per stage. Remote
checkpoint file sizes are verified, and the final JSON report is uploaded with
Modal's batch API and read back byte-for-byte before automatic shutdown.

This experiment does not add a compact-canvas run, region-balanced loss,
learning-rate sweep, conditioning adapter, alternative sampler, test-time
training, augmentation voting, or held-out-task claim.
