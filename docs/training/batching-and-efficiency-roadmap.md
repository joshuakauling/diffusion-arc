# Batching and Efficiency Roadmap

## Status and purpose

This document records the staged batching strategy for Diffusion-ARC. It is a
supporting engineering plan, not a change to the project hypothesis,
architecture, milestone order, evaluation protocol, or initial non-goals in
the root [`README.md`](../../README.md).

This roadmap applies only to the training path. Inference contracts, batching,
prefix reuse, and autoregressive caching are intentionally designed and
validated independently when the inference path is implemented.

The implementation should begin with the smallest correct reference path and
become more efficient only through measured, numerically validated changes.
This progression is itself a project result: it should make the costs and
correctness requirements of variable-length NVARC conditioning and FLUX
training explicit rather than hiding them behind a prematurely optimized
batching implementation.

## Training units and terminology

Diffusion-ARC training has more than one useful notion of a batch.

### Source sequence

A training view contains one complete tokenized NVARC training prompt with an
ordered set of ARC input/output pairs. The first pair provides an initial
demonstration, while each later pair may produce a target for FLUX training.
The collated source sequence retains the view through the assistant header of
its final selected target. It omits that target's output suffix because no
later conditioning prefix can use it.

One source sequence can therefore produce multiple target exposures. If a
sequence contains four pairs, for example, it may provide three targets:

```text
prefix through the assistant header for pair 2 -> output image for pair 2
prefix through the assistant header for pair 3 -> output image for pair 3
prefix through the assistant header for pair 4 -> output image for pair 4
```

### Target exposure

A target exposure is one conditioning prefix paired with one rendered target
image. It is the semantic unit on which FLUX predicts a velocity field and
receives a flow-matching loss.

Multiple target exposures from the same source sequence share an NVARC
forward pass in the intended implementation, but they remain separate FLUX
training cases because they have different conditioning lengths, target
images, noise tensors, and possibly flow timesteps.

### Physical microbatch

A physical microbatch is a collection of source sequences or target exposures
executed together because they fit within a bounded device workload. NVARC and
FLUX need not use the same physical grouping:

- NVARC can group collated source sequences by their actual token lengths.
- FLUX can regroup the resulting target exposures by their prefix lengths.

The physical microbatch is an execution detail and may contain a variable
number of source sequences and target exposures.

### Optimizer batch

An optimizer batch is the complete set of target exposures contributing to one
optimizer update across all ranks and gradient-accumulation microsteps. Its
normalization is defined in target-exposure weight, not in source-sequence
count or padded token count.

In the initial reference implementation, every target exposure has weight
`1`. Its flow loss is the mean squared error over its `900x128` velocity
elements, and the optimizer-batch loss is the arithmetic mean of those
per-exposure losses. Non-unit target weights are deferred and would require an
explicitly documented sidecar and normalization rule rather than an implicit
change to the tensor contract.

These distinctions prevent device scheduling decisions from silently changing
the statistical weighting of the training objective.

## Contract and execution layout

[`PackedTrainingBatch`](../../src/diffusion_arc/model/contracts.py) is a host-side
transport contract. Its flat `input_ids` and `sequence_offsets` preserve
unpadded logical source sequences, but they do not require or authorize one
concatenated attention operation. A runner handling more than one source must
either execute the sources separately or convert them to a correctly masked
padded or true variable-length NVARC layout.

The implemented dataset-to-collator reference path and every public batch field
are documented in
[`training-data-assembly.md`](../data/training-data-assembly.md). This roadmap starts
from that reference contract and describes only later execution and scheduling
stages.

The implemented target-blind padded NVARC execution boundary is documented in
[`nvarc-batch-runner.md`](nvarc-batch-runner.md). It provides the correctness
primitive used before adding length buckets or target-aware staging.

The public target metadata is source-major so it can be validated against the
source boundaries. Later staging pools may build an internal reordered view for
FLUX execution without changing the canonical contract. Task IDs,
augmentation keys, split identities, and other non-tensor audit metadata stay
in a sidecar owned by the dataset or runner and must remain aligned with every
source and target exposure.

The initial `TrainingCollator` is the CPU-side adapter that creates this
contract and its `TrainingBatchMetadata` sidecar from tokenized ARC samples. It
keeps training-view pair selection independent from target selection. It takes
at most the configured number of suffix targets, renders only their canvases,
and truncates each packed source at its final selected prefix. It trusts the
immutable samples produced by the validated dataset and the offsets, tensor
layouts, and palette values it constructs itself. The complete public tensor
validator remains available to diagnostics, tests, and alternate batch
producers, which must establish the contract before entering the hot path. The
collator does not sample data, pad tokens, create an attention mask, move
tensors to an accelerator, or draw flow noise or timesteps.

## Variable structure and cost signals

For source sequence `s`, define:

- `L_s`: the collated NVARC source length after terminal-output truncation;
- `K_s`: the number of target exposures produced by the sequence;
- `p_sj`: the conditioning-prefix length for target `j`;
- `q_sj = p_sj + 900`: the joint FLUX length for that target, including the
  fixed 900 image tokens.

No single scalar fully describes the cost of a physical microbatch. Useful
first-order signals include:

| Component | Approximate variable cost signal |
| --- | ---: |
| NVARC activation storage | `sum_s L_s` |
| NVARC attention compute | `sum_s L_s^2` |
| FLUX activation storage | `sum_s sum_j q_sj` |
| FLUX attention compute | `sum_s sum_j q_sj^2` |
| VAE work | total target exposures `sum_s K_s` |
| Loss weighting | total target-exposure weight |

These expressions are planning signals, not a substitute for measurements.
Actual memory and runtime depend on attention kernels, precision, activation
checkpointing, distributed strategy, and hardware. The fixed memory occupied
by model parameters, gradients, and optimizer state is separate from the
variable costs above.

Packing only by source-sequence count ignores both token length and target
count. Packing only by text tokens ignores the 900 image tokens and repeated
FLUX use of different prefixes. Packing only by target count ignores context
length. The later optimized scheduler should therefore enforce multiple
resource limits or use an empirically fitted cost model.

## Correct reference strategy

The first implementation deliberately favors transparency over utilization.

### Bounded suffix targets per source sequence

The training-view `max_pairs` and `max_cells` settings determine which ordered
pairs enter a source independently of its FLUX target budget. Every pair after
the first remains an available target candidate.

The initial run configuration may declare `max_targets_per_source`. The
collator selects the last `min(max_targets_per_source, available_targets)`
candidates while preserving their source order. A view with fewer candidates
is accepted, and `None` selects all available targets. Earlier unselected pairs
remain useful because their teacher-forced outputs condition the selected
suffix targets.

Suffix selection ensures that the retained source context contributes to at
least one selected target, but it biases a single fixed view toward later pair
positions and longer-context predictions. Sampled pair-order augmentations can
move original pairs into the selected suffix across views. Runs that require
complete per-view target coverage should select all candidates instead.

The packed contract therefore represents variable target counts from the
start. Physical FLUX batching and optimizer-batch normalization must use the
actual number of selected target exposures rather than assuming an exact count
per source.

### NVARC execution

Collated source sequences may be bucketed by actual token length. Within an
NVARC microbatch, shorter sequences may be padded only when NVARC receives a
correct attention mask. The mask makes padding semantically safe; whether the
selected kernel also avoids computing padded positions must be measured.

NVARC remains frozen and runs in evaluation mode. The intended reuse path is:

1. run the collated causal source sequence through NVARC once;
2. concatenate the selected hidden layers;
3. retain only the real, unpadded hidden states;
4. create a conditioning view for every recorded target-prefix boundary.

`NvarcBatchRunner` now implements steps 1 through 3 for one physical NVARC
batch. It right-pads complete sources, supplies the attention mask, performs one
batch forward, and returns only real positions in packed source-major order.
Step 4 remains outside the runner so target metadata cannot enter the NVARC
abstraction. It trusts the collator-produced source contract and a separately
verified stable checkpoint-output contract rather than rescanning both on every
forward. Length bucketing and explicit real-checkpoint numerical comparisons
remain later work; they are not triggered or remembered by a first-forward
branch in the runner.

This reuse is valid only if a prefix computed as part of the full causal
sequence agrees with the same prefix computed in isolation. The real
checkpoint validation must compare

```text
full_sequence_hidden_states[:prefix_end]
```

with

```text
truncated_sequence_hidden_states
```

under the same precision and attention backend. If they do not agree within a
predeclared tolerance, separately evaluating each prefix remains the required
correctness fallback.

### FLUX execution

The reference FLUX path processes one target exposure at a time:

```text
one exact conditioning prefix + 900 image tokens -> one flow loss
```

It uses no text padding, no cross-sample attention, and no unverified masking
behavior. Losses are accumulated until the configured optimizer-batch target
weight has been reached. Each microbatch contribution must be scaled by its
actual target weight so that differently sized microbatches are not weighted
equally by accident.

This single-target FLUX path is expected to be inefficient. Its purpose is to
provide a simple numerical reference for every later batching optimization.

## Staged efficiency roadmap

Every stage must preserve the experiment and be compared with the preceding
validated reference. A stage is useful only if it improves an end-to-end
measurement or provides a clearly documented engineering finding.

### Stage 0 - Single-target correctness reference

- Process exact NVARC prefixes or validate full-sequence prefix reuse.
- Execute FLUX separately for every target exposure.
- Use no FLUX padding and no packed cross-sample attention.
- Verify shapes, dtypes, finite values, loss scaling, and gradient boundaries.
- Establish deterministic tiny tests and a real-checkpoint smoke report.

### Stage 1 - NVARC length bucketing

- Group complete source sequences by similar full token length.
- Use a correct NVARC attention mask for any padding.
- Remove padded positions before producing target conditioning.
- Measure padded-token fraction, NVARC throughput, peak memory, and total step
  time rather than assuming bucketing improves the full training step.

### Stage 2 - Bounded target-exposure pool

- Collect detached conditioning prefixes from one or more frozen-NVARC
  microbatches in a bounded staging pool.
- Preserve the mapping from each prefix to its source sequence, target image,
  augmentation identity, and initial unit target weight.
- Regroup target exposures for FLUX independently of the original NVARC
  microbatches.
- Bound the pool so that retained conditioning states cannot cause unmeasured
  host or device memory growth.

### Stage 3 - Exact-length FLUX batching

- Batch multiple target exposures only when their exact conditioning lengths
  are compatible with the unmodified FLUX interface.
- Compare batched losses, gradients, and parameter updates with the sequential
  reference.
- Measure whether exact-length groups occur frequently enough to improve
  end-to-end throughput.

This stage requires no FLUX padding mask, but it may provide limited benefit if
exact lengths are sparse.

### Stage 4 - Variable-length FLUX attention

- Evaluate a true variable-length kernel or block-diagonal attention path.
- Keep each target's text and 900 image tokens isolated from every other
  target.
- Support per-target timesteps and all timestep-dependent modulation without
  broadcasting one target's state into another target.
- Demonstrate numerical agreement with the single-target reference before
  using the path for learning experiments.

A dense block-diagonal mask can be semantically correct while still computing
the full concatenated attention matrix. It must not be described as an
efficiency improvement unless profiling confirms a benefit.

### Stage 5 - Measured cost-based packing

- Replace simple length buckets with hard resource limits or an empirically
  fitted cost estimator.
- Consider FLUX token volume, FLUX attention cost, NVARC token volume, NVARC
  attention cost, target count, and staging-pool memory.
- Keep the optimizer-batch target weight fixed even when physical microbatch
  composition varies.
- Randomize within bounded planning windows so that length-aware scheduling
  does not turn the training curriculum into a global length sort.

### Stage 6 - Distributed workload balancing

- Build complete optimizer-step plans with equal target weight across ranks.
- Balance estimated compute as a separate objective to reduce stragglers.
- Keep rank microstep counts and synchronization points compatible with DDP or
  FSDP2 requirements.
- Compare numerical behavior and learning curves with the validated
  single-process reference.

## Correctness invariants

Every batching implementation must preserve the following invariants:

- A target conditions only on the end-exclusive prefix recorded for it.
- Future assistant outputs cannot influence an earlier target's conditioning.
- Padded NVARC positions are masked and discarded before FLUX conditioning.
- FLUX never treats padding as real conditioning.
- Tokens belonging to different target exposures cannot attend to one another.
- Every target retains its own target image, flow noise, timestep, and source
  metadata.
- Initial loss normalization gives every target exposure one unit of weight
  across the complete global optimizer batch.
- Task-level train/validation separation and augmentation leakage protections
  remain unchanged by length-aware scheduling.
- Any optimized path matches reference losses, gradients, and parameter
  updates within tolerances fixed before the comparison.

## Measurement protocol

Each stage should report at least:

- target exposures per second;
- real NVARC tokens per second;
- real FLUX text tokens per second;
- padded NVARC token count and fraction;
- median and p95 microstep and optimizer-step time;
- peak allocated and reserved device memory;
- staging-pool host and device memory where applicable;
- per-rank workload and straggler measurements for distributed runs;
- loss, gradient, and parameter-update differences from the reference;
- short overfitting-run behavior after a material optimization.

The run record must include the batching strategy, bucket or budget settings,
cost-model revision, seeds, data split and augmentation identities, precision,
attention backend, hardware, and distributed configuration.

Throughput improvements must be reported end to end. A faster isolated
attention kernel is not sufficient if NVARC execution, target staging, VAE
work, synchronization, or optimizer steps erase the gain.

## Deferred decisions and non-goals

This document intentionally does not yet select:

- a permanent maximum targets-per-source value;
- final NVARC bucket boundaries;
- a maximum token or compute budget;
- a variable-length attention library or kernel;
- a fixed analytical cost model;
- non-unit target weighting;
- a final DDP or FSDP2 scheduling policy.

Offline conditioning caches are outside the approved training path. The
remaining decisions require measurements from the correct reference pipeline.
This roadmap also does not authorize changes to the frozen/trainable component
boundaries, the flow-only objective, the 30x30 target representation, the
task-level split, or the primary evaluation protocol.
