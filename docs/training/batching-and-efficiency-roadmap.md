# Historical NVARC and FLUX Batching Roadmap

This is the historical design note for the initial padded pipeline and planned
length-aware schedulers. References below to "current" or "planned" describe
that stage, not the B3 implementation. B3 retains the FIFO selection, ownership,
flow objective and loss-weighting invariants, but replaces both bucket catalogs
with unpadded FA3 segments. The lookahead schedulers below are superseded
proposals, not the next implementation step; that is distributed training.
The original schema-v1/v2 descriptions are also historical; current configs
require schema v8.
For the current path, see the [training overview](README.md),
[NVARC encoder](nvarc-batch-runner.md) and
[FLUX packing guide](flux-sequence-packing.md).

## The problem

One ARC sequence produces several FLUX inputs. Running NVARC, VAE, and FLUX
separately for every target repeats frozen work and wastes compute through padding.

The three models also have different useful batch boundaries:

- NVARC processes complete token sequences and benefits from grouping sources
  with similar sequence lengths.
- The frozen VAE processes fixed-size target images and can use a batch size
  independent of NVARC and FLUX.
- FLUX processes one target at a time logically, but targets with similar text
  prefix lengths can share a physical forward with less padding.

An optimizer update introduces a fourth boundary: its FLUX inputs must be fixed
before physical grouping, or a planner could change the training order or objective.

The design therefore separates source lookahead, frozen NVARC execution,
prepared target buffering, logical optimizer-batch selection, and physical
FLUX microbatch grouping. The current FIFO implementation is the correctness
baseline; the two-buffer, length-aware scheduler is the planned optimization.

## Buffers, batches, and bucketing

A buffer holds inputs that are waiting for work. A batch is the subset removed
from a buffer for one model execution or optimizer update. Bucketing is only
the strategy used to select compatible inputs; it is not another pipeline
object.

| Term | Meaning |
| --- | --- |
| `ArcTokenizedTrainingSample` | One tokenized ARC sequence with all recorded target positions and target grids. |
| `NvarcInputBuffer` | Planned bounded lookahead pool for length-aware NVARC selection. |
| `NvarcBatch` | A physical batch of complete NVARC sequences, normally grouped by similar length. |
| `NvarcBatchOutput` | One independent, unpadded hidden-state tensor for each sequence in an `NvarcBatch`. |
| `NvarcSequenceState` | The owner of one sequence tensor and the prefix ends that still need it. |
| `FluxInput` | One target containing a shared sequence-state reference, relative `prefix_end`, and clean VAE latent. |
| `FluxInputBuffer` | Prepared target-level inputs waiting for optimizer-batch selection; currently represented by a private FIFO queue. |
| `OptimizerBatch` | The complete logical set of FLUX inputs contributing to one local optimizer update. |
| `FluxMicrobatch` | Independently allocated padded text and stacked image tensors for one physical FLUX forward/backward. |
| `GlobalOptimizerBatch` | The union of all per-rank `OptimizerBatch` objects for one distributed update. |

`NvarcBatch` is not called a microbatch because NVARC is frozen and has no
optimizer accumulation boundary. `FluxMicrobatch` is a true microbatch because
several of them accumulate into one `OptimizerBatch`.

Buffer capacity and batch capacity are separate settings. Buffer capacity
limits waiting state and determines how much choice bucketing has. Batch
capacity limits one execution. For example, 32 sequences may wait in the
`NvarcInputBuffer` while only four similar-length sequences form the next
`NvarcBatch`. The current fixed consecutive batching needs no separate NVARC
lookahead buffer; such a buffer becomes necessary only when a planner selects
sources by length or cost. Likewise, FLUX buffer capacity limits staged inputs,
FLUX microbatch capacity limits one forward, and optimizer batch size defines
one parameter update.

## Object and execution model

```text
consecutive ArcTokenizedTrainingSample items
    |
    | prepare_flux_inputs_batch(...)
    |
    +-- input_ids + sequence_offsets
    |       -> NvarcBatchRunner
    |       -> NvarcBatchOutput
    |       -> NvarcSequenceState
    |
    `-- target grids -> target image Tensor
            -> FluxVaeEncoder
            -> clean latent Tensor
                    |
                    v
              tuple[FluxInput, ...]
                    |
                    v
              internal FIFO deque[FluxInput]
                    |
                    v
              build_flux_microbatch(...)
                    |
                    +-- exact prefixes -> FluxTextInputs
                    +-- stacked clean latents
                    `-- generated image position IDs
                    |
                    v
              FluxMicrobatch
                    |
                    v
              flow sampling + FLUX forward/backward
```

Preparation and FLUX training interleave throughout the run. The fixed-batch FIFO
queue is in-memory state owned by `run_single_gpu_training`; it is not an
offline cache and does not persist across runs. A separate `NvarcInputBuffer`
remains deferred until lookahead bucketing is implemented.

## Target two-buffer scheduler

The planned scheduler makes both waiting pools and all three selection
boundaries explicit:

```text
training dataset
    -> NvarcInputBuffer(capacity C_N)
    -> NvarcBatchPlanner selects S similarly sized sources
    -> frozen NVARC executes each selected source once
    -> each source s expands to T_s FluxInput entries sharing H_s
    -> FluxInputBuffer(capacity C_F)
    -> OptimizerBatchPlanner selects B ready targets
    -> fixed OptimizerBatch
    -> MicrobatchPlanner groups similar prefix lengths within B
    -> M_1 ... M_r, each |M_i| <= m
    -> FLUX forward/backward for every M_i
    -> one optimizer.step()
```

The `NvarcBatchPlanner` uses length-aware bucketing to combine compatible
source sequences and fill a configured NVARC batch budget. It does not simply
take the globally longest sequences. Each selected source produces one shared
`NvarcSequenceState`; its `T_s` eligible target entries retain references to
that owner instead of duplicating the large hidden-state tensor.

The `OptimizerBatchPlanner` chooses the complete set of `B` ready targets from
the `FluxInputBuffer`. That choice can change the training order and batch
composition relative to FIFO, so its policy is part of the experiment rather
than a transparent systems optimization. Once those `B` targets are selected,
membership is fixed. The `MicrobatchPlanner` may then reorder and group only
those targets to reduce prefix padding; it cannot pull replacements from the
buffer.

The [historical diagram](diagrams/training-workflow.svg) visualizes this target
design and its proposed components. The next section describes its original
implemented reference.

## Initial single-GPU scheduling policy

The implemented consumer-first baseline uses fixed consecutive source batches
without length-aware selection:

```text
Input source        = one DataLoader or dataset sample stream
NvarcBatch size     = configured consecutive source count S
VAE batch size      = configured physical target count V
FluxMicrobatch size = configured physical target count M
OptimizerBatch size = configured logical target count B
Run length          = exact max_optimizer_steps
```

The trainer always prefers a parameter update over producing more staged
inputs:

```python
for source_batch in consecutive_batches(training_samples, size=S):
    flux_input_buffer.extend(
        prepare_flux_inputs_batch(source_batch, vae_microbatch_size=V)
    )

    while len(flux_input_buffer) >= optimizer_batch_size:
        train_one_optimizer_batch()
        if optimizer_steps == max_optimizer_steps:
            break

    if optimizer_steps == max_optimizer_steps:
        break

if sample_stream_exhausted and flux_input_buffer:
    train_final_optimizer_batch()
```

The sample stream normally comes from a DataLoader over `ArcTrainingDataset`.
The loop collects up to `S` consecutive samples, removes ineligible sources,
runs the retained sources through one frozen NVARC forward, and encodes their
targets through VAE chunks of at most `V`. It then appends all eligible inputs
in source-major order to the internal FIFO queue. The loop immediately consumes
every available complete `OptimizerBatch` before it requests another source
batch.

For example, let `S1`, `S2`, and `S3` produce two, three, and one FLUX inputs,
respectively. With NVARC source batch size two, optimizer batch size four, and
FLUX microbatch size one:

| Action | Remaining sample stream | Internal `FluxInput` queue |
| --- | --- | --- |
| Initial state | `S1, S2, S3` | empty |
| Prepare source batch `S1, S2` | `S3` | `S1-T1, S1-T2, S2-T1, S2-T2, S2-T3` |
| Select update | `S3` | retain `S2-T3` |

The selected `OptimizerBatch` is
`[S1-T1, S1-T2, S2-T1, S2-T2]`. It runs as four size-one
`FluxMicrobatch` objects, each scaled by `1 / 4`, followed by one
`optimizer.step()`. Preparation then resumes because only `S2-T3` remains.

If the sample stream ends before the configured step limit, any remaining
inputs form one smaller final `OptimizerBatch` and use its actual size as the
loss denominator. Reaching `max_optimizer_steps` instead stops immediately:
the loop performs no extra tail update and requests no additional sample.

## Which FLUX inputs are trained

The training configuration defines:

```text
min_context_pairs_before_target = M
```

A target is eligible when at least `M` complete augmented pairs precede it.
The default `M = 1` makes every pair after the first one a FLUX input. Changing
`M` changes the training distribution and is not a batching optimization.

Every eligible FLUX input has weight `1`. ARC tasks or augmentations producing
more eligible targets therefore contribute more total weight. Alternative
task-balanced weighting is outside the initial design and would be a separate
training experiment.

Eligibility is applied before model execution. The run records the number of
accepted ARC sequences, eligible FLUX inputs, and excluded targets.

## Input order and physical reordering

In the current baseline, the seeded sampler and the target positions inside
each `ArcTokenizedTrainingSample` define a stable logical order. The FIFO queue
preserves it when selecting each optimizer batch. `FluxInput` does not
duplicate sampler, task, or augmentation metadata.

The target scheduler intentionally introduces two configured selections:

1. `NvarcBatchPlanner` chooses `S` compatible sources from the source
   lookahead buffer.
2. `OptimizerBatchPlanner` chooses the next `B` targets from all ready entries
   in the `FluxInputBuffer`.

Those choices may alter the order and membership of an update relative to the
FIFO baseline. The policies must therefore be deterministic under the recorded
seed and configuration, define fair progress with no starvation, and expose
enough sidecar identity to audit the resulting schedule. They must not silently
prefer cheap inputs indefinitely.

After the optimizer planner has selected `B`, physical reordering is narrower:
the `MicrobatchPlanner` may sort or bucket only those fixed targets by prefix
length. This reduces padding without changing which targets contribute to that
update or what weight each target receives.

If later checkpointing, exact per-input randomness, or diagnostics require a
stable occurrence key, the scheduler retains it in a separate sidecar. It does
not belong to the tensor-bearing `FluxInput` contract.

## NVARC preparation and prefix reuse

The correctness oracle executes every conditioning prefix separately and
without padding through NVARC. It establishes the expected downstream FLUX
output, loss, gradients, and parameter update.

The implemented fixed-batch path executes each complete retained NVARC sequence
once, stores its unpadded hidden states, and lets all inputs from that sequence
reference their `prefix_end`. Full-sequence reuse is accepted only if
it passes a predefined real-checkpoint comparison against the exact-prefix
oracle under the selected precision and attention backend.

`NvarcBatchRunner` may receive similarly sized complete sequences. It
right-pads them, creates the NVARC attention mask, runs one frozen forward, and
removes padded positions. Its `NvarcBatchOutput.sequence_hidden_states` tuple
contains one independent owning `[L_s, 7680]` tensor per source in input order.
The temporary packed input offsets do not remain part of the output contract.

Target prefixes are not copied eagerly. A `FluxInput` stores a reference to its
`NvarcSequenceState` and its exact relative `prefix_end`. The owner stores the
source tensor once and tracks every pending prefix end. Prefix ends are unique
within one source and therefore need no additional identity. The prefix is
copied only when `build_flux_microbatch` creates the physical batch.

## VAE preparation and the FLUX input buffer

The frozen VAE encodes every eligible fixed-size target image to one clean
`[900, 128]` latent. Its physical batch size is independent of both
`NvarcBatch` and `FluxMicrobatch` sizes.

`FluxInputBuffer` entries retain:

- the NVARC sequence-state reference and `prefix_end`;
- the clean VAE latent.

Flow noise and timesteps are not buffered. They are created when the input
enters FLUX execution.

All buffered inputs from one source share exactly one `NvarcSequenceState`.
Its pending-prefix set provides explicit ownership accounting without storing
references back to the inputs. `build_flux_microbatch` first validates every
requested prefix, copies exact prefixes into the independent padded text
tensor, stacks all clean latents, and generates fixed image position IDs for
the physical batch. Only after every allocation succeeds does it commit
consumption. A failed build consumes nothing; the final successful commit sets
the owner's hidden-state reference to `None`.

The small owner object can remain reachable from an already-consumed
`FluxInput`, but it no longer retains the large tensor. Python later collects
the owner itself when the buffer, inputs, and local variables no longer refer
to it. CUDA may retain released blocks in its caching allocator, so allocator
reservation is not the same as live tensor allocation.

The planned `FluxInputBuffer` will have explicit capacity accounting, with
count, token, or byte limits chosen from measurement. Backpressure must pause
preparation before staged state grows without bound. Configuration must prove
that at least one complete optimizer batch can be formed within those limits.
An unusually large single sequence needs an explicit one-source oversize rule;
it must not deadlock the producer and consumer. The current private deque has
no separately configurable capacity because preparation and FIFO consumption
interleave after every fixed source batch.

## FLUX execution and loss normalization

One `OptimizerBatch` contains total target weight `B`. It is divided into one
or more prefix-length-aware `FluxMicrobatch` objects. For a microbatch with
`K_i` unit-weight inputs and mean loss `L_i`, backward uses:

```python
scaled_loss = L_i * K_i / B
scaled_loss.backward()
```

Summing the microbatch gradients therefore produces the mean over all inputs
in the optimizer batch. Dividing each microbatch loss only by the number of
microbatches is wrong when their sizes differ.

`build_flux_microbatch` selects and right-pads exact NVARC prefixes through the
pure `pad_flux_text_prefixes` operation, builds text position IDs and the text
mask, stacks clean latents, and creates the fixed image position IDs. Joint-mask
construction, flow sampling, and the trainable FLUX forward/backward happen
when that microbatch is executed.

Image length remains fixed at 900 tokens. The variable length relevant to FLUX
grouping is the NVARC conditioning prefix.

## Randomness

The initial single-GPU reference uses a recorded sequential generator together
with the fixed FIFO input order. Physical regrouping can change generator draw
order. If later numerical comparisons require randomness to be independent of
grouping, the scheduler must bind noise and timesteps to external occurrence
keys stored in its sidecar rather than adding metadata to `FluxInput`.

## Later distributed extension

Distributed planning is deferred until the single-GPU policy is validated.
Its primary setting will be `global_optimizer_batch_size`; all ranks must agree
on target weight, synchronization points, tail handling, loss scaling, and the
configured global selection policy. DDP and FSDP2 must reproduce the validated
global schedule without sampler duplicates, starvation, or silently dropped
inputs.

## Implementation order

The implementation follows the authoritative milestones in the root
[`README.md`](../../README.md):

1. exact single-GPU execution with one NVARC prefix and one FLUX input;
2. the consumer-first buffered reference defined above;
3. DDP and then FSDP2 correctness with the same global input order;
4. checkpoint and resume at optimizer-step boundaries;
5. measured NVARC, VAE, and padded FLUX batching optimizations; and
6. optional variable-length FLUX attention only after profiling justifies it.

Each change is accepted only after loss, gradient, parameter-update, memory,
and end-to-end throughput comparisons with the preceding validated path.

## Required invariants

- NVARC and the VAE remain frozen; the complete FLUX transformer is trainable.
- A FLUX input sees only its recorded end-exclusive conditioning prefix.
- Every input selected for an optimizer batch is consumed exactly once.
- Reaching `max_optimizer_steps` never triggers an additional tail update.
- Physical grouping never changes optimizer target membership or weight.
- Padded NVARC and FLUX text positions are masked and excluded from stored
  sequence states.
- Inputs in one FLUX microbatch cannot attend across batch rows.
- NVARC prefixes share source storage instead of holding eager copies.
- A failed FLUX microbatch build does not consume any pending prefix.
- A source tensor is released immediately after its final dependent input has
  been copied into an independent physical microbatch.
- Buffer limits create backpressure, not silent target loss.
- Pool-based planners are deterministic under the recorded seed and
  configuration and cannot starve eligible sources or targets.
- Optimized execution must pass predefined numerical gates before training.
- Task-level train/validation separation remains unchanged.

## Measurements and open performance choices

Scheduler experiments must measure complete-path throughput, real and padded
tokens, buffer occupancy and bytes, component timings, peak memory, loss,
gradient norm, and numerical differences from the FIFO reference. Buffer
limits, physical batch sizes, device placement, attention backend, activation
checkpointing, and selection policy remain measurement-driven settings. The
current runner does not yet persist this complete measurement set.

Dense padded attention scales with physical batch maxima, and planning must
also account for the transient `[batch_size, max_prefix, 7680]` conditioning
tensor.

Token-length eligibility must be checked in an explicit run preflight before
model work. None of these choices may appear as a silent runtime fallback.

## Current implementation boundary

The repository implements independent `NvarcBatchOutput` tensors, fixed
consecutive source preparation through `prepare_flux_inputs_batch`, bounded VAE
microbatches, explicit `NvarcSequenceState` ownership, pure
`pad_flux_text_prefixes`, and transactional `build_flux_microbatch` assembly.
`compute_flux_microbatch_loss` composes flow sampling, joint-mask
construction, one trainable FLUX forward, and the unscaled velocity loss. The
step-limited `run_single_gpu_training` loop consumes an
`ArcTokenizedTrainingSample` stream, maintains its private FIFO queue, forms
optimizer and FLUX microbatches, scales losses, accumulates gradients, performs
optimizer updates, flushes only a true stream tail, and exposes an optional
backend-neutral update callback. A separate `NvarcInputBuffer` remains
unnecessary while fixed consecutive batches are used without lookahead
bucketing. The offline `train_single_gpu.py` entrypoint and strict JSON run
configuration compose these pieces without changing their ownership. The single-GPU path
keeps persistent FLUX parameters, gradients, and AdamW state in FP32 while
executing the FLUX forward under BF16 autocast. Scalar loss and global gradient
non-finites fail before an optimizer update.

The offline entrypoint supports strict schema-v1 configurations as the original
single-source compatibility mode. Schema v2 adds explicit
`nvarc_source_batch_size` and `vae_microbatch_size` settings. This fixed batching
was intentionally pulled forward without changing the remaining milestone
order; its real-checkpoint performance and numerical acceptance still require
the planned measurement gate. Durable run telemetry, checkpoint writing,
resume support, the two-buffer scheduler, and distributed execution remain
later work.
