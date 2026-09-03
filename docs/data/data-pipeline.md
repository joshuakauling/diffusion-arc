# ARC Data Pipeline

This document describes the implemented ARC data path from one-file-per-task
JSON files to model-facing prompts, token IDs, and target canvases. The project
scope and milestone definitions in [`README.md`](../../README.md) remain
authoritative.

## Goals and invariants

The data layer is designed around four rules:

1. Every ARC task is stored in one JSON file and identified by its filename.
2. Training and evaluation use different types and rendering functions.
3. Evaluation targets never enter a model-facing task, prompt, or token
   sequence.
4. Generated training views are reproducible and retain their source task and
   augmentation choices in a canonical key.

The public API lives in
[`src/diffusion_arc/arc_data`](../../src/diffusion_arc/arc_data).

## On-disk layout

Raw ARC files use the standard `train` and `test` arrays:

```text
data/raw/arc-agi-2/
├── training/
│   └── <task_id>.json
└── evaluation/
    └── <task_id>.json
```

Raw datasets are intentionally ignored by Git. Reproducible task selections
and augmentation-key files are stored separately:

```text
data/splits/<split_name>/
├── config.json
├── train.json
├── validation.json
└── augmented_keys/
    ├── train.json
    ├── train_original.json
    └── train_unseen.json
```

`train.json` and `validation.json` contain source task IDs. Their intersection
must be empty, so augmented variants of a held-out task cannot enter training.

### Modal Volume mirror

[`experiments/setup/download_arc_agi_2_to_modal.ipynb`](../../experiments/setup/download_arc_agi_2_to_modal.ipynb)
installs the complete public ARC-AGI-2 raw dataset once at:

```text
/mnt/diffusion-arc-models/data/raw/arc-agi-2/
├── training/    # 1,000 task JSON files
└── evaluation/  # 120 task JSON files
```

The setup pins official repository commit
`f3283f727488ad98fe575ea6a5ac981e4a188e49`, verifies the downloaded archive
SHA-256, validates every task JSON, and writes a durable per-file hash manifest
at `/mnt/diffusion-arc-models/data/arc-agi-2-manifest.json`. A matching install
is reused; an incomplete or differently identified target is not overwritten.
Training notebooks consume this Volume mirror through an ephemeral canonical
data layout and perform no implicit dataset download.

## Core data types

All core data objects are frozen dataclasses containing immutable tuples.

| Type | Purpose |
| --- | --- |
| `ArcPair` | One labelled input/output grid pair. |
| `ArcTrainingTask` | All labelled pairs used to construct training views. |
| `ArcQuery` | One evaluation input with no output field. |
| `ArcEvaluationTask` | Demonstrations plus target-free queries. |
| `ArcTarget` | One separately held evaluation output. |
| `ArcEvaluationTargets` | The separately held outputs for one task. |

The same labelled ARC file can be loaded through two intentionally different
paths:

- `load_training_task` concatenates the file's `train` and labelled `test`
  examples into one tuple of pairs, in that order.
- `load_evaluation_task` converts `train` examples into demonstrations and
  exposes only the inputs from `test` as queries.
- `load_evaluation_targets` loads the `test` outputs into a separate object.

Grid loading is strict: grids must be non-empty, rectangular, at most 30×30,
and contain integer color IDs from 0 through 9.

## Training path

```text
task JSON
  → load_training_task
  → choose an identity or sampled augmentation
  → apply_training_augmentation
  → render_training_prompt
  ├─→ tokenize_nvarc_training_sample
  └─→ render_training_target_canvases
```

### 1. Select an augmentation

`make_identity_training_augmentation` keeps the original geometry, colors, and pair order. `sample_training_augmentation` samples:

- one of the eight square-grid symmetries,
- a permutation of all ten ARC colors,
- an ordered subset of source pairs.

The pair selector respects `max_pairs` and `max_cells`. Cell usage is the sum
of input and output cells for each selected pair. A pair that does not fit is
skipped while later candidates are still considered. Sampling fails if fewer
than two pairs fit.

Geometry codes are:

| Code | Transformation |
| --- | --- |
| `ID` | Identity |
| `R90`, `R180`, `R270` | Clockwise rotations |
| `FH` | Reflection across the horizontal axis; top and bottom swap |
| `FV` | Reflection across the vertical axis; left and right swap |
| `FD1` | Reflection across the main diagonal |
| `FD2` | Reflection across the anti-diagonal |

Geometry selection cycles deterministically through these eight choices by
augmentation candidate index. Color permutations and pair order are random but
reproducible for a given generator configuration.

### 2. Preserve the view in an augmentation key

Every training view has the canonical form:

```text
<task_id>_<geometry>_<color_mapping>_<pair_indices>
```

For example:

```text
00576224_R90_3214078965_1,2,0
```

This means:

- source task `00576224`,
- a clockwise 90-degree rotation,
- source color `0` maps to `3`, source color `1` maps to `2`, and so on,
- source pairs are selected and presented in the order `1, 2, 0`.

Pair indices use comma-separated decimal integers, so tasks with ten or more
pairs remain unambiguous. Keys round-trip through `format_augmentation_key`
and `parse_augmentation_key`.

### 3. Render the NVARC training prompt

`render_training_prompt` serializes every selected pair using the NVARC chat
format. Grid cells are adjacent digits and rows are separated by newlines:

```text
<|im_start|>user
<input grid><|im_end|><|im_start|>assistant
<output grid><|im_end|>
```

The first pair provides context. Every pair after the first becomes a training
target. For each such pair, the renderer records the end-exclusive character
position immediately after its assistant header. These prefix cuts allow the
training path to extract the autoregressive hidden state for each target from
one complete prompt.

### 4. Tokenize without changing the prompt

`tokenize_nvarc_training_sample` tokenizes the complete prompt with
`add_special_tokens=False`. It does not pad or truncate. It converts every
character prefix cut into a token prefix cut and verifies that the boundary is
stable in the full token sequence.

If the result exceeds `max_tokens`, tokenization fails instead of silently
truncating. The split generator records `max_tokens`, but key generation itself
currently uses only `max_pairs` and `max_cells`; the caller must pass the
configured token limit when tokenizing a generated view.

### 5. Render target canvases

`render_training_target_canvases` renders the target grids in the same order as the prefix cuts. Each grid is:

- placed at the top-left of a fixed 30×30 logical canvas,
- padded with the white PAD sentinel,
- rendered with the fixed `greedy_rgb_maximin_v1` target RGB palette,
- expanded to 480×480 pixels using 16×16 pixels per ARC cell,
- returned as a float tensor in `[-1, 1]` with shape `[3, 480, 480]`.

The stacked training result has shape
`[number_of_targets, 3, 480, 480]`.

The data layer exposes every pair after the first as an aligned target
candidate. It does not apply the training run's target budget. The host-side
training collator may later select a suffix of these candidates while retaining
the preceding pairs as NVARC context.

## Evaluation path

Evaluation keeps model inputs and scoring targets on separate branches:

```text
task JSON
  ├─→ load_evaluation_task
  │     → render_evaluation_prompts
  │     → tokenize_nvarc_evaluation_sample
  │
  └─→ load_evaluation_targets
        → align_evaluation_targets
        → scoring only
```

`render_evaluation_prompts` creates one prompt per query. Each prompt contains all demonstrations in their original order, followed by only the selected
query input. It ends immediately after:

```text
<|im_start|>assistant
```

There is no query target in `ArcEvaluationTask`,
`ArcRenderedEvaluationSample`, or `ArcTokenizedEvaluationSample`.

`align_evaluation_targets` is used only after the model-facing path. It checks
that task IDs match, rejects duplicate query indices, requires exactly one
target per query, and returns targets in query order.

Evaluation currently uses the canonical task without augmentation. Model
sampling, generated-canvas decoding, and exact-grid metrics are outside this
data-preparation layer.

## Reproducible split generation

[`data/splits/overfitting_run_1/config.json`](../../data/splits/overfitting_run_1/config.json)
is the first concrete split configuration. It defines:

- source dataset and source split,
- the required `greedy_rgb_maximin_v1` palette identifier,
- pair, cell, and token limits,
- task-level train and validation files,
- named generated training views.

The current views are:

| View | Meaning |
| --- | --- |
| `train` | Sampled augmentations used for optimization. |
| `train_original` | One identity view per training task. |
| `train_unseen` | New augmentations of the same training tasks, excluded from the other views. |

`train_unseen` does not contain unseen tasks and is not a held-out task-level
evaluation split. It diagnoses whether an overfitting run generalizes to new
views of tasks it has already seen.

For sampled views, each candidate RNG seed is derived with SHA-256 from the
generator version, view name, configured seed, task ID, and candidate index.
This avoids dependence on Python's process-randomized hash and makes output
stable across runs. Views can explicitly exclude keys generated by other
views, and duplicate keys are rejected.

Generate all configured key files with:

```bash
.venv/bin/python scripts/generate_augmented_keys.py \
  data/splits/overfitting_run_1
```

The current `overfitting_run_1/validation.json` is empty. A real held-out
validation run still requires selecting task IDs for that file.

## PyTorch training assembly

`ArcTrainingDataset` builds a map-style PyTorch dataset directly from one
configured split view. The current single-GPU entrypoint uses a standard
`DataLoader` with automatic batching disabled:

```python
from pathlib import Path

from torch.utils.data import DataLoader

from diffusion_arc.training import ArcTrainingDataset


dataset = ArcTrainingDataset(
    split_directory=Path("data/splits/overfitting_run_1"),
    view_name="train",
    tokenizer=tokenizer,
)
loader = DataLoader(
    dataset,
    batch_size=None,
    shuffle=False,
)
```

Each loader item therefore remains one `ArcTokenizedTrainingSample`. The
single-GPU loop groups consecutive items into a configured source batch and
sends that group to `prepare_flux_inputs_batch`. That function performs batched
frozen NVARC preparation, bounded VAE preparation, and appends the resulting
targets in source-major order to the internal `FluxInput` queue. The complete
runtime path is documented in the [training system overview](../training/README.md).

`TrainingCollator` and `PackedTrainingBatch` provide a separate implemented
host-side contract for diagnostics, tests, and alternate source batching. They are
not invoked by the current single-GPU entrypoint. Their suffix-target
selection, packed tensor fields, metadata sidecar, CPU boundary, and
responsibility split are documented in
[`training-data-assembly.md`](training-data-assembly.md).

## Manual inspection

Use the local playground to print identity, augmented, and evaluation prompts,
plus token counts, prefix cuts, augmentation keys, and canvas shapes:

```bash
.venv/bin/python scripts/diagnostics/inspect_arc_data.py \
  data/raw/arc-agi-2/training/00576224.json \
  --augmentations 2 \
  --seed 167
```

The playground rebuilds the small 16-token NVARC tokenizer locally so prompt
inspection requires neither a model checkpoint nor network access. Production
code should use the tokenizer from the verified NVARC checkpoint. The separate
collator inspection command is documented in
[`training-data-assembly.md`](training-data-assembly.md#inspection).

## Current boundaries

The data-pipeline layer documented here deliberately does not provide:

- a custom sampler or distributed `DataLoader` setup,
- NVARC hidden-state extraction,
- VAE encoding,
- flow-noise or timestep sampling,
- generated-canvas boundary decoding,
- validation sampling or accuracy metrics.

Those responsibilities belong to the training and evaluation runners. They
should compose the immutable objects produced here rather than duplicating the
loading, augmentation, prompt, tokenization, or target-alignment logic.

The training package provides both the config-driven map-style dataset and the
auxiliary host-side collator. Higher training layers consume the dataset
objects and add model execution, accelerator placement, flow sampling, and
optimizer updates; those operations are deliberately outside this data-pipeline
document rather than absent from the repository.
