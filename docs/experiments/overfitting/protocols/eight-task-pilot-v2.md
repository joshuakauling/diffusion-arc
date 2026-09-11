# Single-GPU Overfitting Pilot v2

This historical protocol is retained with the [findings and archived source](../README.md).

## Status and question

This document preregisters the second eight-task single-GPU overfitting pilot.
It replaces canonical-only generation as the primary memorization diagnostic
without reinterpreting the first pilot's recorded results.

The primary question is:

> Can the complete trainable FLUX transformer exactly reconstruct fixed ARC
> targets from conditioning prefixes that occurred verbatim during training?

The experiment remains a same-task memorization diagnostic. Canonical and
unseen augmented views are secondary transfer diagnostics and are not held-out
task evaluation.

## Fixed data and optimization protocol

The training data remains the immutable `train` view from
`overfitting_run_1`: eight ARC tasks, 64 augmentations per task, 512 sequences,
and 1,664 eligible target exposures per epoch. Exact identity keys remain
excluded from this optimization view.

The optimizer recipe remains AdamW with learning rate `1e-5`, betas
`(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and global gradient
clipping at `1.0`. The effective optimizer batch remains eight target
exposures. The physical FLUX microbatch increases from one to eight, removing
gradient accumulation without changing the effective batch size. The measured
uncheckpointed H200 capacity sweep passed physical batch eight at sequence
length 256; the pilot fails rather than silently changing this contract.

One epoch remains exactly 208 optimizer updates. The notebook permits at most
12 manually triggered epochs. The user may stop after any completed epoch and
explicitly finalize the selected checkpoint.

## Evaluation panels

All panels contain one deterministic final query per pilot task.

### Seen training panel

The first key for every task in the immutable `train` view is selected before
training. The notebook reconstructs the target-free evaluation prompt and
asserts that its token IDs equal the exact prefix used by the training sample.
Full pass@1/pass@2 generation runs at step zero and after every epoch.

This panel exclusively defines the overfitting evidence levels:

1. optimization signal: final fixed seen-panel flow loss is at most 50% of its
   step-zero value;
2. positive memorization evidence: at least one of eight tasks is exact-grid
   pass@2;
3. strong memorization evidence: at least six of eight tasks are exact-grid
   pass@2;
4. complete memorization: all eight tasks are exact-grid pass@2.

### Same-task transfer panels

The canonical `train_original` panel is generated at step zero and after
epochs 4, 8, and 12. The separately seeded `train_unseen` panel is generated
after epochs 4, 8, and 12. These panels measure transfer between presentations
of already known tasks. They must not be described as held-out ARC
generalization.

If a checkpoint is explicitly finalized after another epoch, canonical
generation is first recorded at that same optimizer step so the reusable
checkpoint evaluator can make a step-aligned comparison.

## Runtime diagnostics

The notebook retains deterministic fixed-flow probes for the seen, unseen, and
canonical panels. It also requires an exact clean VAE encode/decode round trip
for all eight selected seen-panel targets before optimization. Training epochs
and generation seeds expose progress bars and elapsed-time estimates.

The recorded report includes selected augmentation keys, exact-prefix checks,
dataset and checkpoint hashes, model and package revisions, seeds, precision,
hardware, optimizer-step telemetry, generated images, and all panel metrics.

## Durable checkpoint contract

Checkpoint finalization is explicit and does not include AdamW resume state.
The transformer is first saved to local ephemeral storage with 2 GB
Safetensors shards. The notebook validates the index, shards, sizes, and
SHA-256 hashes before copying them through a unique staging directory into the
mounted Volume.

The Volume is explicitly committed and reloaded. The final checkpoint and its
manifest are revalidated from the reloaded mount before `saved_checkpoint` is
written to `report.json`. The updated report is then committed, reloaded, and
checked once more. Automatic container shutdown is allowed only after this
contract succeeds.

## Interpretation and non-goals

- Exact seen-panel grids with poor canonical or unseen results demonstrate
  memorization without presentation transfer.
- Improvements on unseen or canonical panels provide same-task transfer
  evidence only.
- No exact seen-panel grids after the bounded run, despite a very low fixed
  probe loss, motivates investigation of sampling, conditioning, and flow
  parameterization before substantially longer training.

This pilot does not provide held-out task generalization, an NVARC baseline,
distributed-training evidence, optimizer-resume validation, or a learning-rate
sweep. A later one-augmentation-per-task experiment remains an optional,
separate pipeline sanity check.
