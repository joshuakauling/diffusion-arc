# Single-GPU Overfitting Pilot

## Status and question

This document preregisters the first Milestone 2 learning pilot. It does not
claim completion of Milestone 2 and must not be interpreted as held-out ARC
evaluation.

The pilot asks:

> Can the complete trainable FLUX transformer learn a measurable flow signal
> from frozen NVARC conditioning and produce increasingly correct ARC grids on
> a fixed eight-task training set?

A positive result supports basic learnability of the initial flow-only path. A
negative bounded run does not by itself reject the central Diffusion-ARC
hypothesis.

## Fixed data protocol

The pilot uses all eight task IDs in
[`data/splits/overfitting_run_1/train.json`](../../data/splits/overfitting_run_1/train.json).
No task-level validation split is created for this memorization experiment.

The existing generated views remain unchanged:

| View | Per task | Total sequences | Total eligible FLUX targets | Role |
| --- | ---: | ---: | ---: | --- |
| `train` | 64 | 512 | 1,664 | Optimization |
| `train_original` | 1 | 8 | 26 | Identity inspection only |
| `train_unseen` | 16 | 128 | 416 | New views of the same tasks |

The sampled training view cycles evenly through the eight square-grid
symmetries and independently samples color permutations and pair orders. The
exact identity keys are excluded from `train`; `train_unseen` is generated from
a separate seed and excludes both training and identity keys.

The canonical inference panel contains one target-free query for each of the
eight tasks. Query outputs and output dimensions are absent from NVARC and FLUX
conditioning. They enter only the scoring and fixed-flow-probe paths.

## Fixed training recipe

The source configuration is
[`configs/training/single_gpu_overfitting_pilot.json`](../../configs/training/single_gpu_overfitting_pilot.json).
The Modal notebook resolves only the checkpoint paths to the mounted Volume.
The complete raw dataset is installed separately by
[`experiments/download_arc_agi_2_to_modal.ipynb`](../../experiments/download_arc_agi_2_to_modal.ipynb)
and verified against its durable per-file manifest before model loading.

- one H200 and one process;
- NVARC BF16 and frozen;
- FLUX VAE FP32 and frozen;
- complete FLUX transformer trainable with persistent FP32 parameters;
- BF16 FLUX compute under autocast;
- AdamW, learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, and
  weight decay `0.01`;
- physical FLUX microbatch size `1`;
- optimizer batch size `8` target exposures, implemented as eight accumulated
  microbatches;
- global gradient norm measured once per optimizer update and clipped to
  `1.0`;
- uniform flow timesteps and fresh Gaussian noise for every training exposure;
- seed `167` for the process and flow generator;
- four manually started epochs of exactly `208` optimizer steps each, for a
  maximum of `832` updates and `6,656` target exposures.

The notebook uses a distinct deterministic DataLoader shuffle seed for each
epoch while retaining the same immutable 512-key training view. Each epoch is a
separate cell so a user explicitly controls every additional paid-compute
segment.

## Instrumentation and evaluation

Every optimizer update records:

- global and epoch-local step;
- mean flow loss over eight target exposures;
- target and physical microbatch counts;
- learning rate;
- pre-clip global gradient norm and whether clipping occurred;
- synchronized optimizer-step time;
- current, reserved, and peak CUDA memory;
- non-finite failures.

Fixed-flow probes use cached frozen conditioning and target latents with the
same probe seed on every invocation. They run at step 0 and every 52 optimizer
steps. This makes their loss values comparable despite stochastic training
timesteps and noise.

Full canonical inference runs at step 0 and after steps 208, 416, 624, and 832.
It uses:

- a fixed linear 50-step Euler schedule from `t=1` to `t=0`;
- two independent, fixed generation seeds, `167` and `168`;
- exact PAD-boundary dimension inference with no ground-truth fallback;
- exact-grid pass@1 and pass@2, valid-grid rate, shape accuracy, and conditional
  cell accuracy;
- saved RGB images, classified canvases, decoded grids, and invalid reasons.

The durable JSON report records the repository revision, configuration and
split hashes, checkpoint identities, environment, full step log, probe series,
generation metrics, and failure details. Large checkpoints are not saved
automatically. The notebook provides a separate explicit cell that saves the
current FP32 FLUX transformer when the user chooses to retain a checkpoint.

## Preregistered interpretation

The pilot uses three evidence levels:

1. **Optimization signal:** final fixed train-probe loss is no more than 50% of
   its step-0 value.
2. **Structural learning signal:** at least four of eight canonical tasks have
   a valid pass@2 prediction with the correct output shape, accompanied by an
   improvement in cell-level diagnostics over step 0.
3. **Positive overfitting evidence:** at least one of eight canonical tasks is
   exact in pass@2. Six of eight is strong evidence; eight of eight is complete
   overfitting of this panel.

Loss improvement without an exact grid is partial evidence: the flow objective
is learnable, while sampling, VAE reconstruction, palette classification, or
PAD-boundary decoding remains limiting. No fixed-probe improvement is evidence
against the current implementation or recipe, not a conclusive rejection of
the broader research hypothesis.

## Explicit non-goals

This pilot does not provide:

- held-out task generalization;
- comparison with NVARC autoregressive decoding;
- hyperparameter selection;
- DDP, FSDP2, activation-checkpointing, or efficiency conclusions;
- `torch.profiler` or Nsight Systems traces;
- complete checkpoint/resume validation;
- Milestone 2 completion.
