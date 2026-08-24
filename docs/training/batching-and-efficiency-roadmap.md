# NVARC and FLUX Batching Roadmap

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

The design therefore separates the two input buffers, logical optimizer
batches, and physical FLUX microbatches.

## Buffers, batches, and bucketing

A buffer holds inputs that are waiting for work. A batch is the subset removed
from a buffer for one model execution or optimizer update. Bucketing is only
the strategy used to select compatible inputs; it is not another pipeline
object.

| Term | Meaning |
| --- | --- |
| `ArcTokenizedTrainingSample` | One tokenized ARC sequence with all recorded target positions and target grids. |
| `NvarcInputBuffer` | ARC sequences waiting to be selected for NVARC execution. |
| `NvarcBatch` | A physical batch of complete NVARC sequences, normally grouped by similar length. |
| `NvarcBatchOutput` | One independent, unpadded hidden-state tensor for each sequence in an `NvarcBatch`. |
| `NvarcSequenceState` | The owner of one sequence tensor and the prefix ends that still need it. |
| `FluxInput` | One target containing a shared sequence-state reference, relative `prefix_end`, and clean VAE latent. |
| `FluxInputBuffer` | Prepared FLUX inputs waiting to be selected for training. |
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
`NvarcBatch`. The NVARC buffer is unnecessary while `NvarcBatch` size remains
one and no lookahead selection is used. Likewise, FLUX buffer capacity limits
staged inputs, FLUX microbatch capacity limits one forward, and optimizer batch
size defines one parameter update.

## Object and execution model

```text
ArcTokenizedTrainingSample
    |
    | prepare_flux_inputs(...)
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
              [future FluxInputBuffer]
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

Preparation and FLUX training interleave throughout the run. Neither buffer is
an offline cache or persists across runs. The `NvarcInputBuffer` and
`FluxInputBuffer` may have different capacities and selection strategies.

## Initial single-GPU scheduling policy

The first implementation is consumer-first and uses the simplest physical
execution:

```text
NvarcInputBuffer selection = FIFO
NvarcBatch size            = 1 sequence
FluxMicrobatch size        = 1 input
OptimizerBatch size        = configured target count B
```

The trainer always prefers a parameter update over producing more staged
inputs:

```python
while training_not_finished:
    if flux_input_buffer.has_complete_optimizer_batch():
        train_one_optimizer_batch()
    elif nvarc_work_remains:
        fill_nvarc_input_buffer()
        run_one_nvarc_batch_and_prepare_flux_inputs()
    elif flux_input_buffer:
        train_final_optimizer_batch()
    else:
        break
```

`fill_nvarc_input_buffer` reads ARC sequences up to its configured capacity.
Because the initial `NvarcBatch` size is one, it removes the oldest sequence.
One frozen NVARC forward and the required frozen VAE work then append all of
that sequence's eligible inputs to the `FluxInputBuffer`. The loop immediately
checks again whether a complete `OptimizerBatch` can be trained.

For example, let `S1`, `S2`, and `S3` produce two, three, and one FLUX inputs,
respectively. With optimizer batch size four and FLUX microbatch size one:

| Action | `NvarcInputBuffer` | `FluxInputBuffer` |
| --- | --- | --- |
| Initial fill | `S1, S2, S3` | empty |
| Prepare `S1` | `S2, S3` | `S1-T1, S1-T2` |
| Prepare `S2` | `S3` | `S1-T1, S1-T2, S2-T1, S2-T2, S2-T3` |
| Select update | `S3` | retain `S2-T3` |

The selected `OptimizerBatch` is
`[S1-T1, S1-T2, S2-T1, S2-T2]`. It runs as four size-one
`FluxMicrobatch` objects, each scaled by `1 / 4`, followed by one
`optimizer.step()`. Preparation then resumes because only `S2-T3` remains.

At epoch exhaustion, any remaining inputs form a smaller final
`OptimizerBatch` and use its actual size as the loss denominator. The
correctness reference never drops eligible inputs.

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

The seeded sampler and the target positions inside each
`ArcTokenizedTrainingSample` define the stable logical input order. The initial
FIFO scheduler preserves that order outside the model objects; `FluxInput`
does not duplicate sampler, epoch, task, or augmentation metadata.

The next `OptimizerBatch` is selected from this order before any FLUX backward
call. NVARC and VAE preparation may run in a different physical order. The
selected inputs may also be sorted by prefix length when splitting that already
fixed optimizer batch into `FluxMicrobatch` objects.

If later checkpointing, exact per-input randomness, or diagnostics require a
stable occurrence key, the scheduler must retain it in a separate sidecar. It
does not belong to the tensor-bearing `FluxInput` contract.

This reordering may reduce padding, but it must not change which inputs
contribute to an update. Selecting cheaper inputs from a larger buffer would
change the training schedule and must be treated as a separately configured
experiment rather than a transparent optimization.

## NVARC preparation and prefix reuse

The correctness oracle executes every conditioning prefix separately and
without padding through NVARC. It establishes the expected downstream FLUX
output, loss, gradients, and parameter update.

The intended optimized path executes each complete retained NVARC sequence
once, stores its unpadded hidden states, and lets all inputs from that
sequence reference their `prefix_end`. Full-sequence reuse is accepted only if
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

The `FluxInputBuffer` has explicit count, token, and byte limits. Backpressure
pauses preparation before staged state grows without bound. Configuration must
prove that at least one complete optimizer batch can be formed within these
limits. An unusually large single sequence needs an explicit one-source
oversize rule; it must not deadlock the producer and consumer.

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
on target weight, synchronization points, tail handling, and loss scaling. DDP
and FSDP2 must preserve the validated global FLUX input order without sampler
duplicates or silently dropped inputs.

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
- Every eligible input is consumed exactly once under the configured epoch
  policy.
- Physical grouping never changes optimizer target membership or weight.
- Padded NVARC and FLUX text positions are masked and excluded from stored
  sequence states.
- Inputs in one FLUX microbatch cannot attend across batch rows.
- NVARC prefixes share source storage instead of holding eager copies.
- A failed FLUX microbatch build does not consume any pending prefix.
- A source tensor is released immediately after its final dependent input has
  been copied into an independent physical microbatch.
- Buffer limits create backpressure, not silent target loss.
- Optimized execution must pass predefined numerical gates before training.
- Task-level train/validation separation remains unchanged.

## Measurements and open performance choices

Every stage records complete-path throughput, real and padded tokens, buffer
occupancy and bytes, component timings, peak memory, loss, gradient norm, and
numerical differences from its reference. Buffer limits, physical batch sizes,
device placement, attention backend, activation checkpointing, and later
cost-based selection remain measurement-driven settings.

Dense padded attention scales with physical batch maxima, and planning must
also account for the transient `[batch_size, max_prefix, 7680]` conditioning
tensor.

Token-length eligibility must be checked in an explicit run preflight before
model work. None of these choices may appear as a silent runtime fallback.

## Current implementation boundary

The repository now implements the independent `NvarcBatchOutput` tensors,
single-source `prepare_flux_inputs`, explicit `NvarcSequenceState` ownership,
pure `pad_flux_text_prefixes`, and transactional `build_flux_microbatch`
assembly. `compute_flux_microbatch_loss` now composes flow sampling, joint-mask
construction, one trainable FLUX forward, and the unscaled velocity loss. The
old fused step, loop, and training entrypoint are not part of this path.
`FluxInputBuffer`, optimizer-batch selection, loss scaling, backward and
optimizer execution, and the consumer-first scheduler remain the next
implementation layers. A separate `NvarcInputBuffer` remains unnecessary while
NVARC batch size is one and no lookahead bucketing is used.
