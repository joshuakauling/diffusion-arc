# Training Data Assembly

## Status and scope

This document describes the implemented dataset contract and the auxiliary
CPU-side packed-collation path. The packed path is useful for diagnostics,
tests, and alternate source batching, but it is not the input path used by the
current `scripts/train_single_gpu.py` entrypoint.

The current trainer instead configures `DataLoader(batch_size=None)` and groups
consecutive `ArcTokenizedTrainingSample` items inside the training loop. It
passes each fixed source batch to `prepare_flux_inputs_batch`; the auxiliary
collator still does not participate. See the [training system
overview](../training/README.md) for that end-to-end runtime path.

The project scope and milestones in the root [`README.md`](../../README.md) remain
authoritative. The upstream loading, augmentation, prompt, and tokenization
rules are documented in [`data-pipeline.md`](data-pipeline.md). Current physical
grouping and loss weighting are documented in the
[FLUX packing guide](../training/flux-sequence-packing.md).
The target-blind execution of packed sources through frozen NVARC is documented
in [`nvarc-batch-runner.md`](../training/nvarc-batch-runner.md).
Exact-prefix FLUX conditioning and training-side ownership are documented in
[`flux-text-conditioning.md`](../training/flux-text-conditioning.md).
Frozen target-image encoding is documented in
[`flux-image-encoding.md`](../training/flux-image-encoding.md).
Per-target flow-noise and timestep sampling is documented in
[`flow-matching.md`](../training/flow-matching.md).

The auxiliary packed path is:

```text
split config + augmentation keys + raw ARC tasks
  -> ArcTrainingDataset[index]
  -> ArcTokenizedTrainingSample
  -> PyTorch DataLoader
  -> TrainingCollator(samples)
  -> CollatedTrainingBatch
       |- PackedTrainingBatch tensors
       `- TrainingBatchMetadata sidecar
```

This is a host transport contract. It prepares aligned CPU tensors but does
not define how NVARC or FLUX execute them on an accelerator. The current
single-GPU runner follows the separate fixed-source-batch path described above.

The implemented dataset and collator live in
[`src/diffusion_arc/training`](../../src/diffusion_arc/training). Their tensor
contract lives in
[`src/diffusion_arc/model/contracts.py`](../../src/diffusion_arc/model/contracts.py).

## Responsibility boundaries

| Component | Responsible for | Not responsible for |
| --- | --- | --- |
| `ArcTrainingDataset` | Selecting a configured view, validating split, key, and task boundaries, loading the referenced task for each item, applying stored augmentations, rendering prompts, tokenizing complete samples | Task caching, shuffling, batching, target budgeting, canvas rendering, model execution |
| PyTorch `DataLoader` | Selecting dataset indices, shuffle/sampler policy, and workers; optionally forming source-level batches and invoking a collator | ARC semantics, target selection, packed tensor meaning |
| `TrainingCollator` | Assembling trusted tokenized samples by selecting suffix targets, truncating sources, rendering target canvases, and building the packed tensor contract and metadata sidecar | Validating alternate sample producers, sampling dataset items, padding, attention masks, device transfer, noise, timesteps, model execution |

## `ArcTrainingDataset`

### Construction

The map-style dataset is constructed from a split directory, a configured
view name, and an already loaded tokenizer:

```python
from pathlib import Path

from diffusion_arc.training import ArcTrainingDataset


dataset = ArcTrainingDataset(
    split_directory=Path("data/splits/overfitting_run_1"),
    view_name="train",
    tokenizer=tokenizer,
)
```

The split directory must match:

```text
<data_directory>/splits/<split_name>
```

The dataset reads `config.json` from that directory and derives the raw source
directory without a separate root argument:

```text
<data_directory>/raw/<config.source.dataset>/<config.source.split>
```

`view_name` is a lookup key in the configuration's `views` mapping. Names such
as `train`, `train_original`, and `train_unseen` are current configuration
choices rather than names hard-coded into the dataset. The selected view tells
the dataset which augmentation-key file and task split to use. The current
split schema only permits generated training views over the `train` task split;
a future held-out validation or inference dataset is a separate concern.

The tokenizer is injected by the caller. Dataset construction never downloads
or chooses a checkpoint.

### Construction-time checks and retained state

During construction, the dataset:

1. strictly loads the split configuration;
2. selects the named view;
3. loads the task-level train and validation lists and rejects overlap;
4. loads the selected view's augmentation keys in stored order;
5. rejects malformed keys, duplicates, and task IDs outside the configured
   task split;
6. derives the raw source directory and verifies every referenced task file;
7. loads every referenced task for validation;
8. validates every key against its task and requires at least two selected
   pairs.

The dataset retains the parsed keys and source location, but it does not cache
parsed ARC tasks. Item materialization reloads the task referenced by the
selected key. Its length is the number of keys, not the number of source ARC
tasks. One task can therefore appear in multiple dataset items through
different augmentation keys.

### Item materialization

For one index, the dataset performs a deterministic transformation:

```text
stored augmentation key
  -> load its referenced raw ARC task
  -> apply its geometry, color mapping, and pair order
  -> render the complete teacher-forced NVARC prompt
  -> tokenize without padding or truncation
  -> ArcTokenizedTrainingSample
```

`ArcTokenizedTrainingSample` contains:

| Field | Meaning |
| --- | --- |
| `key` | Source task plus the exact augmentation and ordered original pair indices. |
| `input_ids` | The complete teacher-forced token sequence for the selected pairs. |
| `prefix_token_ends` | End-exclusive token boundaries immediately after each target pair's assistant header. Boundaries are relative to this one source sequence. |
| `target_grids` | Target output grids aligned one-to-one with `prefix_token_ends`. |

The first selected pair is context only. Every later selected pair is exposed
as a target candidate. If the key contains original pair indices `(4, 1, 3)`,
the sample has two target candidates:

```text
pair 4: complete initial context
pair 1: prefix boundary + target grid 0
pair 3: prefix boundary + target grid 1
```

The dataset exposes all available candidates. It does not apply the training
run's target budget, because doing so would couple reusable data preparation to
one batching policy.

## PyTorch `DataLoader`

When the auxiliary packed path is selected, the standard PyTorch loader
determines which dataset items form one source-level batch and passes them as a
sequence to the collator:

```python
from torch.utils.data import DataLoader

from diffusion_arc.training import TrainingCollator


loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=TrainingCollator(max_targets_per_source=2),
)
batch = next(iter(loader))
```

Here, `batch_size=2` means two source samples. It does not guarantee two or
four target exposures: the number of available targets may differ by source.
Shuffle policy, a future length-aware sampler, worker count, and distributed
sampling remain explicit loader or runner decisions.

The current single-GPU entrypoint deliberately uses `batch_size=None` and no
`collate_fn`, so each loader item remains one `ArcTokenizedTrainingSample`.
The training loop groups those items into fixed consecutive source batches and
sends them to `prepare_flux_inputs_batch`; neither `TrainingCollator` nor
`PackedTrainingBatch` participates in the current training loop.

The current returned batch uses custom dataclasses without a `pin_memory()`
method. Consequently, `DataLoader(pin_memory=True)` does not automatically pin
the contained tensors. The initial reference path remains ordinary CPU memory;
pinning can be added later when the accelerator transfer path is implemented
and measured.

## `TrainingCollator`

### Input and target selection

The collator receives a non-empty sequence of trusted
`ArcTokenizedTrainingSample` objects produced by `ArcTrainingDataset`. Another
producer must establish the same sample contract before invoking the collator.
It accepts:

```python
TrainingCollator(max_targets_per_source=K)
```

or:

```python
TrainingCollator(max_targets_per_source=None)
```

For a source with `A` available target candidates, it selects:

```text
last min(K, A) candidates        when K is an integer
all A candidates                when K is None
```

Selection is a suffix. With four available candidates and `K=2`, candidates
two and three are retained. With one available candidate and `K=2`, that one
candidate remains valid.

Suffix selection keeps the final available target. The collator can therefore
truncate the source after that target's assistant header:

```python
source_input_end = selected_prefix_ends[-1]
```

The unused output tokens of the final target are removed. Outputs of earlier
pairs remain available as causal context for later selected targets. Pair
selection itself has already been stored in the augmentation key; the collator
only chooses which aligned target candidates produce training exposures.

### Source-major assembly

Sources and their selected targets retain DataLoader order. The collator:

1. selects the target suffix for each source;
2. truncates each source at its final selected prefix;
3. appends the truncated token sequences into one flat tensor;
4. renders selected target grids as fixed 480x480 canvases;
5. records source and target lookup information;
6. returns tensors and audit metadata together.

It does not add padding or an `attention_mask`. Flat concatenation is a compact
host representation, not permission for attention to cross source boundaries.

## Output contract

`TrainingCollator` returns the public `PackedTrainingBatch` contract together
with its metadata sidecar:

```python
CollatedTrainingBatch(
    tensors=PackedTrainingBatch(...),
    metadata=TrainingBatchMetadata(...),
)
```

For `S` source sequences, `T` total collated tokens, and `K` selected target
exposures, the tensor fields are:

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `input_ids` | `[T]`, `torch.int64` | All truncated source token sequences concatenated without padding. |
| `sequence_offsets` | `[S + 1]`, `torch.int64` | End-exclusive boundaries of each source inside `input_ids`; starts at zero and ends at `T`. |
| `target_sequence_indices` | `[K]`, `torch.int64` | Source index belonging to each target exposure. Entries are source-major. |
| `target_prefix_ends` | `[K]`, `torch.int64` | End-exclusive conditioning boundary relative to the corresponding source, not the flat token tensor. |
| `target_images` | `[K, 3, 480, 480]`, `torch.float32` | Selected target canvases in `[-1, 1]`, aligned with the two target lookup tensors. |

The sidecar fields are:

| Field | Length | Meaning |
| --- | ---: | --- |
| `source_keys` | `S` | Exact augmentation key for every source. |
| `target_pair_indices` | `K` | Original ARC pair index belonging to each selected target. |

The sidecar carries identities required for diagnostics, provenance, and error
analysis but not for model arithmetic. Keeping it separate avoids encoding
strings or audit-only values in the model tensor interface.

### Two-source example

Assume the collator produces two truncated sources:

```text
source 0 length: 120, selected prefix ends: (66, 120)
source 1 length: 105, selected prefix ends: (48, 105)
```

The canonical batch is:

```python
input_ids.shape          == (225,)
sequence_offsets         == [0, 120, 225]
target_sequence_indices  == [0, 0, 1, 1]
target_prefix_ends        == [66, 120, 48, 105]
target_images.shape       == (4, 3, 480, 480)
```

For target exposure 2:

```text
source index             = target_sequence_indices[2] = 1
flat source span         = input_ids[120:225]
relative prefix boundary = target_prefix_ends[2] = 48
conditioning tokens      = input_ids[120:120 + 48]
target image             = target_images[2]
original pair index      = metadata.target_pair_indices[2]
```

This two-stage lookup is why source indices and prefix ends are separate. The
prefix boundary alone cannot identify a source when several unpadded sequences
are stored in one flat tensor.

## CPU boundary and validation

The collator creates ordinary CPU tensors. Split loading, raw-task parsing,
augmentation construction, prompt rendering, and tokenization establish the
sample invariants before this hot path. The collator therefore trusts its
immutable input objects and the values it constructs; it does not repeat type,
alignment, token-range, grid, or full-tensor scans for every batch.

`validate_packed_training_batch` remains the explicit full validator for
diagnostics, tests, and alternate batch producers. Such producers must invoke
it at their own ingestion or preflight boundary instead of relying on the
collator to validate arbitrary objects.

Model execution is a separate boundary. The current
[NVARC runner](../training/nvarc-batch-runner.md) consumes flat token IDs and
source offsets using unpadded causal FA3 attention, stops after block 27, and
returns one owning `[L_s, 7680]` tensor per source.

B3 training creates these inputs directly from consecutive samples, without
the collator. Target prefixes share source storage until copied into
[FLUX packed microbatches](../training/flux-sequence-packing.md).
The [training overview](../training/README.md) describes preparation, ownership,
FIFO selection, and model execution together.

## Where noise enters

Noise and flow timesteps are not data-loading or collation concerns. They enter
after this host batch reaches the model-side flow sampler:

```text
target_images
  -> frozen FLUX VAE encoding
  -> target latents
  -> per-target noise and flow-timestep sampling
  -> noisy/interpolated FLUX input
  -> FLUX velocity prediction and flow-matching loss
```

Each selected target exposure receives its own target image and therefore its
own latent, noise sample, and uniform timestep. The optional generator lets the
training runner control and record sampling reproducibly.

## Inspection

The collator can be inspected without loading NVARC or FLUX:

```bash
.venv/bin/python scripts/diagnostics/inspect_training_collator.py \
  data/raw/arc-agi-2/training/00576224.json \
  --max-pairs 6 \
  --max-targets-per-source 2
```

The script constructs deterministic identity samples, prints the candidate and
selected prefix cuts, shows source offsets and target mappings, reports tensor
properties, and explicitly runs `validate_packed_training_batch`.

## Responsibility boundary

The implemented dataset and collator intentionally do not own:

- a validation or inference dataset;
- a custom length-aware or cost-aware sampler;
- accelerator transfer or automatic pinned-memory traversal;
- accelerator attention backends or model-level batch selection;
- alternate flow-timestep densities or timestep-dependent loss weighting;
- FLUX input buffering, optimizer-batch scheduling, or model execution;
- distributed sampling or optimizer-batch accounting.

These responsibilities remain separate from the host contract. B3 already
provides packed model execution, a single-GPU FIFO training loop, benchmark
records, and profiling. Distributed sampling and checkpoint/resume remain
separate implementation work.
