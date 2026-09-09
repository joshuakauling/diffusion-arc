# FLUX training with token-budget packing

The [packed training recipe](../../configs/training/benchmarks/single_gpu_packed.json)
replaces B2's FLUX padding/shape catalog with a **12,288 joint-token budget**
and FA3 variable-length attention. The implementation lives in `src/`; the
training path does not import diagnostic scripts.

## Scheduling and execution

Each target costs its valid NVARC prefix length **plus 900 image tokens**.
Within the fixed FIFO optimizer batch, best-fit-decreasing processes larger
targets first and places each in the tightest remaining space. Ties retain
deterministic source order. Targets are neither split nor moved across optimizer
boundaries. This is a fast call-count heuristic, not an exact bin-packing solver.

Replaying the four recorded 48-target windows in
`configs/diagnostics/flux_packing.json` gives:

| Window | B2 calls | Packed calls | Token-sum lower bound |
| --- | ---: | ---: | ---: |
| Short | 16 | 10 | 10 |
| Median | 19 | 12 | 12 |
| Longest source | 23 | 14 | 14 |
| Largest total | 21 | 15 | 14 |

The first three plans attain a proven minimum. The last lower bound does not
prove a feasible 14-call plan. These are reconstructed schedules, not new GPU
throughput measurements. New benchmark rows record actual target indices,
joint tokens and zero text padding.

The microbatch builder copies valid prefixes into one text tensor before
consuming their NVARC owners. Each target retains a separate attention segment:
`[text_A, image_A] | [text_B, image_B] | ...`. CUDA int32 cumulative boundaries
isolate those segments; text positions reset for each target. Both double- and
single-stream FLUX blocks use noncausal FA3 after their existing Q/K
normalization and RoPE. Q/K/V are converted together to the V projection's
BF16 dtype at the kernel boundary.

Each example retains its own timestep modulation. The three modulation gathers
use unchanged forward values and **FP32 per-example sums in backward**, followed
by one conversion to the original modulation dtype. This changes neither all
matrix multiplication reductions nor all activation storage to FP32. FLUX
parameters, parameter gradients and AdamW state remain FP32.

Flow noise and timesteps are assigned in original FIFO order before reordering.
Each microbatch's mean loss is weighted by its target count divided by the
optimizer batch's actual target count, including a partial final batch. One
AdamW update follows all microbatches. Finished copies are released before the
next physical batch is constructed.

## Configuration and runtime

Schema v7 adds `"flux_packing": {"max_tokens": 12288}`. For this mode,
`flux_scheduling`, `loop.flux_microbatch_size` and
`padding.flux_bucket_boundaries` must be null. FA3 and FP32 modulation sums are
mandatory in the packed training path. Older schemas and recorded B0/B1/B2
recipes retain their original execution for reproducible comparisons.

NVARC source scheduling, CPU prefetch, frozen VAE preparation, data selection,
optimizer batches and the LR schedule are inherited from B2. This change
integrates **FLUX training**; NVARC's production encoder and existing generation
and fixed-probe entrypoints keep their existing attention paths. A matching
packed generation adapter and a production NVARC-FA3 migration are separate work.

`load_single_gpu_models` explicitly checks the CUDA/Hopper FA3 build using a
small BF16, head-128, unequal-segment forward/backward before loading the models.
It then checks the FLUX architecture, trainable FP32 parameters and disabled
checkpointing. Missing or incompatible FA3 is an error. Model parameter names
and checkpoint formats remain unchanged. This implementation targets the
current single-process trainer; distributed wrappers and gradient checkpointing
require separate integration.

The working FA3 build is pinned by cached image `im-R5CncDboPfMQryWnq0UZn4`;
see the [build evidence](../performance/flux-packing.md). The budget starts below
the [single-sequence execution boundary](../performance/flux-sequence-capacity.md),
which confirmed 13,312 joint tokens and OOM at 13,568. A single-sequence limit
does not certify every mixed-shape training allocation history.

## Validation and prepared GPU check

95 targeted local tests passed across model, scheduling, construction, loop,
configuration, setup and entrypoints. They include:

- Independent original FLUX singletons versus packed ABC/CBA and split packs:
  outputs, weighted loss, all parameter/input gradients and an SGD update in FP32.
- Example isolation, BF16 Q/K/V conversion, FP32 modulation cancellation against
  an FP64 oracle, and processor restoration after failure.
- Token accounting, stable plans, shared-state release and failed allocations.
- Identical per-target flow draws and RNG state, optimizer membership, loss
  weighting and updates across two full windows plus a partial tail.
- Explicit backend failure, strict config rejection and actual packed benchmark
  metadata through both training entrypoints.

The [three-step smoke recipe](../../configs/training/benchmarks/single_gpu_packed_smoke.json)
uses the existing B0 dataset, 48 targets per update and three real AdamW updates
(one warmup, two measured), with one repetition. The bounded
[cloud launcher](../../scripts/cloud/benchmark_flux_packing_modal.py) uses the
cached FA3 image and limits its H200 lifecycle to 480 seconds; its resource
estimate is at most $0.724 at the recorded rate. It is **prepared but not run**.
An integrated H200 pass with changing packs and initialized optimizer state is
the remaining validation before claiming production GPU capacity or throughput.

Relevant implementation:
[scheduler](../../src/diffusion_arc/training/flux_schedule.py),
[microbatch assembly](../../src/diffusion_arc/training/flux_microbatch.py),
[FA3 execution](../../src/diffusion_arc/model/flux_packing.py),
[modulation backward](../../src/diffusion_arc/model/flux_modulation.py),
[training loop](../../src/diffusion_arc/training/single_gpu_loop.py).
