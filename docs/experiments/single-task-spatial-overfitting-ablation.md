# Single-Task Spatial Overfitting Ablation

## Status and question

This document preregisters a paired diagnostic on ARC task `0d3d703e`. It is
not a replacement for the primary fixed-30x30 architecture and does not change
the project's task-level evaluation protocol.

The question is:

> Can the complete FLUX transformer learn the task's color-permutation rule
> from 512 unique one-pass augmentations when only the final target is
> supervised, and does removing the 30x30 padding overhead change the result?

NVARC/Qwen3-4B and the FLUX VAE remain frozen. Each spatial variant trains the
complete FLUX transformer from the same verified basis checkpoint with the
original velocity head and unweighted flow-matching loss.

## Why the final target is isolated

Every selected augmentation contains all five task pairs. The rendered
training sequence therefore exposes four possible targets after one, two,
three, or four preceding pairs. For `0d3d703e`, early prefixes can omit color
mappings required by the next query. The experiment sets
`min_context_pairs_before_target = 4`, producing exactly one target after four
complete demonstrations. This removes prefix underdetermination without
introducing ground-truth information at inference.

The split under
[`data/splits/overfitting_0d3d703e_final_target`](../../data/splits/overfitting_0d3d703e_final_target)
contains:

| View | Count | Role |
| --- | ---: | --- |
| `train` | 512 | Unique optimization augmentations |
| `train_unseen` | 8 | Disjoint same-task presentation transfer |
| `train_original` | 1 | Canonical same-task presentation |

All keys are deterministic and unique. The identity key is excluded from
training, and the unseen keys are excluded from both training and identity
views. This is same-task transfer, not held-out ARC generalization.

## Paired spatial variants

The two runs execute sequentially in one notebook but do not share trainable
state:

| Variant | RGB target | Packed FLUX tokens | Decode contract |
| --- | ---: | ---: | --- |
| `compact-3x3` | 48x48 | 3x3 = 9 | Known full 3x3 grid; no PAD boundary |
| `padded-30x30` | 480x480 | 30x30 = 900 | Strict inferred PAD boundary |

After the compact checkpoint is saved, its transformer and optimizer are
released. The padded transformer is loaded again from the basis checkpoint,
and a deterministic parameter fingerprint must equal the compact run's
initial fingerprint. Both variants use the same key order, optimizer settings,
and per-step timestep seeds. Their Gaussian tensors cannot be elementwise
identical because their spatial shapes differ, but each optimizer step starts
from an independently reset generator with the same seed.

The compact variant is an explicit diagnostic scope exception. The default
canvas, model, flow, and inference APIs remain 30x30, and the primary approach
continues to infer output dimensions from PAD rather than using known shapes.

## Preparation and optimization

The 512 final conditioning prefixes are packed into one physical NVARC batch.
Their hidden states are cached before optimization and shared by the two
spatial runs. A batch-size-one reference for the first sequence is compared
against its batch-512 representation using cosine similarity and mean absolute
error. The experiment records the resolved source revision, package versions,
split hashes, keys, seeds, model manifests, tensor shapes, hardware, timings,
and peak memory.

Each run uses:

- 512 final targets consumed exactly once;
- optimizer and physical FLUX batch size 8;
- 64 optimizer updates;
- AdamW learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, and weight
  decay `0.01`;
- global gradient clipping at `1.0`;
- FP32 persistent FLUX parameters with BF16 autocast compute.

There are no epochs, repeated targets, learning-rate schedules, or automatic
batch-size fallbacks.

## Evaluation

Evaluation runs at optimizer steps 0, 16, 32, and 64. Each regular generation
contains 17 views and two seeds:

- eight fixed keys drawn from the 512 training augmentations;
- all eight disjoint unseen augmentations;
- the canonical identity view.

The first generation seed defines pass@1. A grid solved by either seed defines
pass@2. Exact-grid results are reported separately for seen, unseen, and
canonical panels.

The compact variant classifies the complete known 3x3 grid against the ten ARC
colors. The padded variant uses the normal eleven-class palette and strict PAD
boundary inference. A padded result also records accuracy inside the true
top-left 3x3 window as a diagnostic, but this oracle-window measurement never
counts as an exact-grid success.

At every evaluation step, a fixed-noise flow probe compares correct conditioning
with a deterministic one-row cyclic permutation. It records total MSE and the
MSE over the nine real grid tokens. The padded variant additionally records MSE
over the remaining 891 PAD tokens. At step 64, complete Euler generation is
repeated under permuted conditioning with the same initial noise as the correct
run. Mean absolute final-latent change and classified-cell change measure
whether FLUX uses NVARC context.

## Interpretation

- Compact seen and unseen success with padded failure supports padding or loss
  imbalance as a major limitation.
- Seen success without unseen success demonstrates presentation memorization,
  not the shared augmentation rule.
- Comparable correct and shuffled conditioning indicates context neglect even
  if unconditional output quality improves.
- Failure of both variants despite falling fixed-flow loss shifts attention
  toward flow/timestep parameterization or inference rather than padding alone.
- Padded success would show that the previous eight-task failure was more
  likely caused by multi-task interference or ambiguous early targets.

This experiment does not establish held-out task generalization, choose a
production canvas policy, test alternative learning rates, compare samplers,
or provide an NVARC autoregressive baseline.

## Notebook and outputs

The executable entrypoint is
[`experiments/training/single-task-spatial-overfitting-ablation-modal.ipynb`](../../experiments/training/single-task-spatial-overfitting-ablation-modal.ipynb).
It runs the source-controlled experiment runner, streams progress, writes a
single comparison report, and uploads one validated Safetensors checkpoint per
variant through the Modal Volume batch-upload API.

Results are stored under:

```text
/mnt/diffusion-arc-models/results/single-task-spatial-overfitting-ablation/<run-id>/
```

Automatic notebook shutdown is permitted only after both remote checkpoint
file sets have matching sizes and the report status is `COMPLETE`.
