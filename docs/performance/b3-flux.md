# B3 FLUX: FA3 packing for training

The [NVARC results](b3-nvarc.md) raised the same question for FLUX: could a
different attention backend make it faster and remove the need for padded
batches? We tested that idea in FLUX's double-stream and single-stream blocks.
Unlike frozen NVARC, FLUX also needed a working backward pass through the
complete model.

The result is **FA3 sequence packing with FP32 modulation-gradient sums**.
B3 replaces the padded shape catalog with a **12,288 joint-token budget**.

## How we tested it

We separated three questions: whether the reference implementation agreed,
whether packing changed the computation, and whether the new path was faster.

- Reuse the same weights, prepared NVARC/VAE outputs, noise, timesteps, and
  loss weights. Compare independent targets, padded batches, and packed targets.
  Reverse target order and perturb individual targets to check their separation.
- Compare complete forwards and parameter gradients, then replay individual
  attention operations with identical inputs and upstream gradients. Use FP32
  references to distinguish BF16 rounding from layout and backend changes.
  Controls covered native image-prompt conditioning and ARC inputs before and
  after adaptation.
- Measure forward/backward time on fixed B2 groups, then probe larger token
  counts. H200 performance tests kept frozen models, FLUX gradients, and
  initialized optimizer state resident, with BF16 compute, FP32 parameters,
  and activation checkpointing disabled.

## What mattered

**Some numerical differences already existed before packing.** The original
Black Forest Labs and Diffusers forwards agreed closely in FP32, but differed
in BF16. Small controlled tests traced differences to Q/K normalization and
rotary position encoding. We kept the existing Diffusers arithmetic; switching
the attention kernel did not require replacing the transformer architecture.

**Packing affected more than attention.** Isolated attention tests supported
separate target boundaries. Complete-model tests also exposed differences in
projections and modulation, including large gradient deviations within the
same attention backend. Small output or loss differences alone did not explain
the backward behavior.

**FP32 modulation sums fixed a major source of gradient error.** Each target's
timestep modulation is shared across its tokens. In backward, their contributions
must be added back together. The packed implementation's BF16 indexed sums
introduced substantial error. We changed these three sums to FP32 and cast
back once, leaving the forward values unchanged.

| FA3 packed execution | Parameter-gradient relative L2 vs FA3 singletons |
| --- | ---: |
| BF16 modulation sums | 17.59–17.99% |
| FP32 modulation sums | 3.14–3.22% |

These ranges cover two repetitions of the same real target pair. Forward
outputs and losses were unchanged by the intervention. The remaining gradient
difference was not resolved. A separate timing comparison found no meaningful
runtime penalty or increase in peak memory from the FP32 sums.

**FA3 packing was faster on the tested groups.** Across six fixed B2 groups,
assembly plus forward/backward took **9–25% less time than padded cuDNN**.
FA3 and FA4 were within 0.52% of each other; FA2 was slower throughout. We chose
FA3, which already had the full-model gradient comparison for FP32 modulation
sums. Packing did not consistently reduce memory, so capacity needed its own
measurement.

## Token budget and B3 integration

A controlled single-sequence test passed at **13,312 joint tokens** and
repeatedly ran out of memory at **13,568**. This motivated a starting budget
of **12,288**, counting each target's exact text prefix plus 900 image tokens.
The single-sequence limit alone did not guarantee that every mixed pack would fit.

B3 groups whole targets within the selected 48-target optimizer batch, placing
larger targets first into the tightest remaining space. Each target keeps its
own attention segment, timestep, and loss weight. Noise and timesteps are
assigned before regrouping. Both block types use noncausal FA3; no padding
catalog remains.

The [combined B3 benchmark](b3.md#benchmark-result) then exercised changing packs
through real optimizer updates. FLUX averaged **12.02 calls per step**, down
from B2's **18.68**, with no text padding. Full-pipeline throughput increased
by **25.0%**; that includes the NVARC changes as well.

## Archived analyses

- Reference and precision: [BFL comparison](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-bfl-reference.md),
  [normalization and RoPE](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-bfl-precision.md),
  [input conditions](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-backend-conditions.md).
- Packing and diagnosis: [initial comparisons](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-packing.md),
  [test strategy](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/attention-diagnostic-plan.md),
  [isolated operations](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/attention-execution.md),
  [backward layouts](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-backward-layouts.md).
- Selected change: [FP32 modulation backward](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-modulation-backward.md).
- Timing and capacity: [backend performance](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-packing-performance.md),
  [reduction cost and mixed packs](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-packed-capacity.md),
  [single-sequence limit](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-sequence-capacity.md).
