# FLUX batch-shape probe

**Status: shape probe and combined B2 GPU benchmark complete.** The probe
measured candidate FLUX microbatches. The resulting initial catalog schedules
targets within the existing 48-target optimizer batch; see the
[B2 benchmark and profiler results](README.md#b2--length-aware-dynamic-batch-single-gpu-pipeline).

## Existing evidence

The [uncheckpointed memory experiment](../../experiments/training/single-gpu-memory-footprint-modal.ipynb)
and its [August 25 results](../../experiments/results/single-gpu-memory-footprint/20260825T054229658645Z/report.json)
used an H200 with 139.80 GiB, frozen resident NVARC/VAE, FP32 FLUX parameters
and AdamW moments, and BF16 FLUX computation. Each text length adds 900 image
tokens per target.

| Text tokens | Largest passing batch | Largest batch with peak reserved <=90% VRAM | First OOM batch |
| ---: | ---: | ---: | ---: |
| 128 | 16 | 8 | 32 |
| 256 | 8 | 8 | 16 |
| 512 | 8 | 8 | 16 |
| 1,024 | 8 | 4 | 16 |
| 2,048 | 4 | 4 | 8 |
| 4,096 | 2 | 2 | 4 |
| 8,192 | 1 | 1 | 2 |

That experiment cleared gradients with `set_to_none=True` before each case.
Later microbatches in an accumulated optimizer step start with gradients already
resident: the complete FP32 FLUX gradients occupy 14.44 GiB. The prepared
source/target buffer is also resident in the current pipeline. Peak overlap
depends on execution order, so adding 14.44 GiB to every old peak is not an
exact prediction. The old single-call times also include possible shape setup
costs and are not steady-state throughput measurements.

## Candidate lengths

The [B0 input analysis](b1-bucket-policy.md#data-and-histogram) contains 2,880
target prefixes in 60 original optimizer batches. All three repetitions use
identical inputs, counted once. Real text lengths range from 86 to 8,400; the
median is 1,836.5 and nearest-rank p95 is 4,158.

The initial 16 boundaries minimize additional text positions for individual
targets rounded upward to a catalog of that size:

```text
487, 778, 1090, 1348, 1578, 1818, 2126, 2422,
2749, 2990, 3325, 3714, 4210, 4837, 6100, 8400
```

On these inputs, rounding adds 7.96% text positions, versus B1's 7.92% with
19 boundaries. These data-fitted boundaries are candidates, not a GPU runtime
optimum. The selected dataset is a benchmark subset, not the entire training
pool. Actual grouping adds its own padding and depends on the measured feasible
batch sizes and times.

## Probe protocol

The [probe configuration](../../configs/diagnostics/flux_batch_shapes.json)
references the current B2 preparation training config and tests batch sizes
1, 2, 4, and 8 at each candidate length. Each shape executes **two warm-up calls
and five measured calls in one sweep**. The first call is part of warm-up and
is also reported separately; it is not an extra repetition.

- Load the current production models from local cached checkpoints only.
  Keep activation checkpointing disabled and the benchmark precision policy.
- Initialize real AdamW moments and gradients with a production training step
  at learning rate zero. The override applies to every optimizer step.
- Retain a conservative preparation reserve: 16 full 8,400-token BF16 source
  states and 96 FP32 target latents (current and previous optimizer batches),
  totaling 1.964 GiB. This reserve remains resident for every candidate.
- Generate one deterministic synthetic five-pair 30x30 source through the
  current tokenizer, frozen NVARC, and VAE preparation code. Reuse its hidden
  states and one latent inside the reserve. No dataset download is required.
- Build each microbatch through `build_flux_microbatch` with the selected
  bucket boundary. Row `i` uses `max(1, boundary - 1 - i)` real text positions,
  exercising explicit padding and a nontrivial mask even at batch size one.
  These slices are shape probes, not semantically aligned ARC training targets.
- Run `compute_flux_microbatch_loss` and backward with the production `B / 48`
  loss weight. Clear gradient values between calls with `set_to_none=False`,
  outside timing, preserving their storage. This models the memory footprint
  of later accumulation microbatches without asserting learning equivalence.
- Use CUDA events for forward/loss and backward spans and also record wall
  time and assembly time. Event spans can include CPU dispatch gaps. Report
  medians from the five measured calls and time per target. Perform LR-zero
  AdamW separately after each shape, outside the forward/backward timing.
- Peak allocated/reserved includes assembly, warm-up, measured calls, and the
  separate optimizer step; setup peaks are recorded separately. Report raw
  headroom, without automatically labeling a shape safe for training.

Models stay loaded across successful shapes. A candidate OOM is persisted,
larger batches at that same length are marked **skipped**, and that worker
exits. The next length starts in a fresh process. This skip assumes increasing
batch size will not recover feasibility at the same length; skipped shapes
are never reported as measured OOMs. Setup OOMs, other errors, incomplete
worker records, and timeouts stop the sweep instead of producing capacity claims.

The configured total worker budget is **1,200 seconds**, including model reloads.
The supervisor kills an active worker at the deadline and preserves partial
results. The Modal wrapper uses one H200, no automatic retries, and a separate
30-minute function timeout. Each completed shape is persisted to the existing
model volume. Results contain configurations, source/configuration hashes,
package versions, hardware, per-call data, and per-worker logs. Full checkpoint
weights are not hashed or copied.

## Measured results and initial B2 catalog

Probe `20260907T140406512796Z` completed on September 7, 2026, in 619.76 seconds
on one H200 with 139.80 GiB. It recorded 39 successful shapes, 15 OOMs, and
10 larger shapes skipped after an OOM at the same length. The protocol above
includes resident FP32 gradients, AdamW moments, and the preparation reserve.

| Text boundary | Largest passing microbatch |
| --- | ---: |
| 487 | 8 |
| 778, 1090, 1348, 1578, 1818, 2126, 2422 | 4 |
| 2749, 2990, 3325, 3714, 4210, 4837 | 2 |
| 6100, 8400 | 1 |

The [B2 benchmark config](https://github.com/joshuakauling/diffusion-arc/blob/db6d49f/configs/training/benchmarks/single_gpu_b2.json)
uses these **24 explicit shapes**, with the probe's median whole-call
forward/backward milliseconds stored as `forward_backward_ms` for each shape:

| Microbatch size | Text padding boundaries |
| ---: | --- |
| 4 | 487, 778, 1090, 1348, 1578, 1818, 2126 |
| 2 | 487, 1348, 2126, 2422, 2749, 2990, 3325, 3714, 4210, 4837 |
| 1 | 487, 2422, 3325, 4210, 4837, 6100, 8400 |

We exclude `2422 x 4`: its peak allocation was 133.20 GiB, only 6.60 GiB below
device capacity. Removing it adds about 44 ms per optimizer batch in the cost
model. We also exclude `487 x 8`, selected only four times across the 60 replay
windows, saving about 3 ms per step. The retained shapes peaked at at most
127.90 GiB allocated in the probe, 11.90 GiB below capacity. These differences
are observed allocation margins, not guarantees for mixed-shape training.
Reserved-memory peaks include allocator history from earlier shapes.

Starting from those 37 shapes, greedy backward elimination removed combinations
whose absence added the least modeled time over the original 60 windows. This
is a data-fitted starting policy, not a globally optimal catalog.

| CPU replay, all 60 windows | Mean calls | Modeled forward/backward time | Text padding above real positions |
| --- | ---: | ---: | ---: |
| Microbatch 1, all 16 candidate boundaries | 48.00 | 10.678 s | 7.96% |
| 37 shapes after the two exclusions | 19.32 | 9.467 s | 9.27% |
| Selected 24 shapes | 18.68 | 9.492 s | 10.11% |

The selected catalog costs 25 ms per optimizer batch relative to the 37-shape
catalog and reduces modeled FLUX time by 11.1% relative to candidate-boundary
singletons. The 50 measurement windows average 9.473 s. Selecting a 24-shape
catalog on either alternating 30-window subset and checking the other gave
0.30–0.34% extra modeled time relative to the 37-shape catalog. This is an
internal sensitivity check on the benchmark subset, not independent validation.
Times sum warm per-shape measurements; they exclude preparation, assembly,
optimizer work, and cold shape setup. They are not measured B2 step times.

### Scheduling and training semantics

The scheduler stably sorts the already selected optimizer batch by descending
prefix length. Dynamic programming partitions this order into contiguous
groups and minimizes the sum of their configured whole-call times. A short
target may join a larger boundary if doing so lowers the total cost. Ties
prefer fewer calls, then ascending batch size and boundary. There are no dummy
rows, bucket queues, dropped targets, or groups crossing optimizer boundaries.
Catalog singletons cover odd and partial tails without uncatalogued shapes.

Each physical batch pads explicitly to its chosen boundary and masks positions
beyond each real prefix. Source states remain alive until all their targets
have been copied. Each microbatch mean loss receives weight `B / K`, where
`K` is the actual optimizer-batch target count (normally 48); the optimizer
updates once after all groups.

For dynamic batching, flow times and noise are sampled once per target in
logical FIFO order before executing groups, using B1's singleton draw shapes
and order. The resulting noisy latents, velocities, and times follow their
targets through reordering. This retains about 42.2 MiB of flow tensors for
48 targets and prevents the scheduling policy from reassigning randomness.
The legacy fixed-microbatch sampling path remains unchanged. CPU tests verify
per-target RNG assignment, accumulated gradients, updates, tails, ownership,
and tiny BF16 FLUX attention. The completed GPU benchmark checks full-model
step-loss agreement with B1; full-model gradient/update equivalence remains
unmeasured.

Schema version 6 adds `flux_scheduling.shapes`. With scheduling enabled,
`loop.flux_microbatch_size` and `padding.flux_bucket_boundaries` must both be
null; the shape catalog defines both. Versions 1–5 preserve their existing
behavior. The new B2 config includes the existing
[NVARC scheduling and CPU prefetch policy](b2-preparation-policy.md).

Benchmark `flux_batches` records actual execution order, each padding length,
and target indices into the original FIFO optimizer batch. Both
`target_prefix_tokens` and `flux_padded_prefix_tokens` remain in logical order.
The scheduler is not recomputed for telemetry.

## Running the probe

From the repository root with the existing Modal environment and model volume:

```bash
.venv/bin/modal run --env diffusion-arc --detach scripts/cloud/probe_flux_batch_shapes_modal.py --config configs/diagnostics/flux_batch_shapes.json
```

For a machine with locally cached checkpoints and a compatible CUDA GPU:

```bash
.venv/bin/python -B scripts/diagnostics/probe_flux_batch_shapes.py --config configs/diagnostics/flux_batch_shapes.json
```

The launcher prints the result directory. This is a bounded shape diagnostic,
not the formal B2 benchmark.
