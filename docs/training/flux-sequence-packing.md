# FLUX training with token-budget packing

The [packed training recipe](../../configs/training/benchmarks/single_gpu_b3.json)
replaces B2's FLUX padding/shape catalog with a **12,288 joint-token budget**
and FA3 variable-length attention. The implementation lives in `src/`; the
training path does not import diagnostic scripts.

## Scheduling and execution

Each target costs its valid NVARC prefix length **plus 900 image tokens**.
Within the fixed FIFO optimizer batch, best-fit-decreasing processes larger
targets first and places each in the tightest remaining space. Ties retain
deterministic source order. Targets are neither split nor moved across optimizer
boundaries. This is a fast call-count heuristic, not an exact bin-packing solver.

Replaying the four recorded 48-target windows in the
[archived fixture](https://github.com/joshuakauling/diffusion-arc/blob/2ff42764b06163fc1df855fc2529ae98a5cf2e2b/configs/diagnostics/flux_packing.json) gives:

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

Schema v8 combines NVARC and FLUX packing. B3 sets
`"flux_packing": {"max_tokens": 12288}`; `flux_scheduling`,
`loop.flux_microbatch_size` and `padding.flux_bucket_boundaries` are null.
FA3 and FP32 modulation sums are mandatory in this packed training path.
Older schemas are rejected. Recorded B0/B1/B2 recipes and diagnostic code are
recoverable from the [archive](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b/).

The same runtime packs each 16-source NVARC window without padding and stops
after block 27, retaining raw hidden states 9, 18 and 27. CPU prefetch, frozen
VAE chunks of 48, FIFO optimizer membership, data selection and the LR schedule
remain unchanged. See the [NVARC encoder guide](nvarc-batch-runner.md).
Existing generation and fixed-probe entrypoints keep their own FLUX attention
path; a matching packed generation adapter remains separate work.

`load_single_gpu_models` explicitly checks the CUDA/Hopper FA3 build using a
small BF16, head-128, unequal-segment forward/backward before loading the models.
It then checks the FLUX architecture, trainable FP32 parameters and disabled
checkpointing. Missing or incompatible FA3 is an error. Model parameter names
and checkpoint formats remain unchanged. This implementation targets the
current single-process trainer; distributed wrappers and gradient checkpointing
require separate integration.

The shared Modal image builds pinned FA3 revision
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` with BF16 forward and backward,
head dimension 128 and Hopper/SM90 support. NVARC and FLUX use separate adapters
to the same extension, with causal and noncausal segments respectively.
The budget starts below the [single-sequence execution boundary](https://github.com/joshuakauling/diffusion-arc/blob/271be103ded3599eafe268606f26357391a8bf78/docs/performance/flux-sequence-capacity.md),
which confirmed 13,312 joint tokens and OOM at 13,568. A single-sequence limit
does not certify every mixed-shape training allocation history.

## Validation

Local CPU and mocked-runtime tests cover the NVARC encoder, FLUX model,
packing, construction, loop, configuration, setup and entrypoints, including:

- Independent original FLUX singletons versus packed ABC/CBA and split packs:
  outputs, weighted loss, all parameter/input gradients and an SGD update in FP32.
- Example isolation, BF16 Q/K/V conversion, FP32 modulation cancellation against
  an FP64 oracle, and processor restoration after failure.
- Token accounting, stable plans, shared-state release and failed allocations.
- Identical per-target flow draws and RNG state, optimizer membership, loss
  weighting and updates across two full windows plus a partial tail.
- Explicit backend failure, strict config rejection and actual packed benchmark
  metadata through both training entrypoints.

The [combined B3 benchmark](../performance/b3.md#benchmark-result) completed
three H200 repetitions with changing packs and real optimizer updates.
It records full-step throughput, peak memory, and a step-loss comparison with B2.
The [FLUX report](../performance/b3-flux.md) summarizes the component tests and
remaining gradient differences; long-term learning equivalence is still open.

For a shorter execution check, the
[three-step smoke recipe](../../configs/training/benchmarks/single_gpu_b3_smoke.json)
and [cloud launcher](../../scripts/cloud/benchmark_flux_packing_modal.py) use
the same packed path with 48 targets per update.

Relevant implementation:
[scheduler](../../src/diffusion_arc/training/flux_schedule.py),
[microbatch assembly](../../src/diffusion_arc/training/flux_microbatch.py),
[FA3 execution](../../src/diffusion_arc/model/flux_packing.py),
[modulation backward](../../src/diffusion_arc/model/flux_modulation.py),
[training loop](../../src/diffusion_arc/training/single_gpu_loop.py).
