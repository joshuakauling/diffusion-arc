# NVARC bucket shape and alignment results

**Status: measured; all 72 cases completed.** The B2 bucket diagnostic found
shape-specific alignment gains, but no universal rounding policy. Its later
replacement is [FA3 Varlen packing without padding](nvarc-packing.md). The
measured bucket costs were never adopted in the production scheduler.

## Measured results

Run `20260907T174749742239Z` completed on H200 in **782.70 seconds**. It measured
20 existing B2 shapes and upward rounding to 8/32/64/128, deduplicated to 72
physical cases. Every case has three rounds with two warm-ups and five measured
calls per round: **1,080 measured calls** in total. Lengths above 8,400 were
excluded by contract, not reported as OOMs.

The table compares each original shape with the lowest pooled median among its
measured candidates, including the original. Times are milliseconds per NVARC
call. Selecting the minimum is optimistic and is not an independently validated
scheduling policy.

| Batch | Original length | Original ms | Best observed length | Best ms | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 864 | 56.359 | 864 | 56.359 | +0.00% |
| 2 | 3022 | 190.672 | 3040 | 182.434 | -4.32% |
| 2 | 4250 | 268.645 | 4250 | 268.645 | +0.00% |
| 2 | 6010 | 397.385 | 6016 | 389.621 | -1.95% |
| 2 | 6870 | 478.296 | 6880 | 471.454 | -1.43% |
| 2 | 8400 | 619.942 | 8400 | 619.942 | +0.00% |
| 4 | 1083 | 135.332 | 1088 | 129.031 | -4.66% |
| 4 | 1431 | 175.553 | 1432 | 164.962 | -6.03% |
| 4 | 1718 | 201.327 | 1728 | 192.419 | -4.42% |
| 4 | 2190 | 246.223 | 2192 | 245.808 | -0.17% |
| 4 | 2824 | 323.655 | 2848 | 311.534 | -3.75% |
| 4 | 3122 | 348.870 | 3128 | 343.511 | -1.54% |
| 4 | 3372 | 372.371 | 3372 | 372.371 | +0.00% |
| 4 | 3780 | 430.198 | 3780 | 430.198 | +0.00% |
| 4 | 4204 | 491.352 | 4208 | 479.542 | -2.40% |
| 4 | 4507 | 631.696 | 4512 | 534.929 | -15.32% |
| 4 | 4836 | 577.299 | 4840 | 574.960 | -0.41% |
| 4 | 5380 | 663.836 | 5384 | 658.006 | -0.88% |
| 8 | 2528 | 517.494 | 2528 | 517.494 | +0.00% |
| 8 | 2702 | 557.060 | 2702 | 557.060 | +0.00% |

The strongest gain, **4 × 4507 → 4 × 4512**, reduces the pooled median from
631.696 to 534.929 ms (15.3%). Its round medians change from
633.272/615.809/635.507 to 537.808/537.672/526.380 ms, improving in all three
rounds. Conversely, 2 × 4250 outperforms all tested aligned neighbors. Small
median differences must be interpreted alongside round-to-round variation.

All compared compact hidden states had **maximum absolute difference zero**.
The highest allocated-memory peak during timing was **72.240 GiB**, including
resident models, optimizer state, gradients and the diagnostic reserve. This
is not a guarantee about complete training allocator histories. All four
profiles for batch 2 and lengths 3022/3024/3040/3072 recorded 36 cuDNN SDPA
operators each. No mask hooks or SDPA replacements were installed.

## Workload and measurement

The probe calls the existing B2 `NvarcBatchRunner`. For a boundary L, row i has
L − 1 − i real tokens, so even the original shape contains masked PAD. Seeded
synthetic five-pair 30×30 ARC templates supply the consumed prefix tokens,
excluding the final target output. Every candidate in a comparison uses the
same token IDs and source lengths. These are shape probes, not complete real
training windows or held-out quality measurements.

NVARC remains BF16 and frozen. The VAE, FLUX parameters and optimizer state
remain resident with the original preparation reserve (16 source states and
96 target latents, about 1.964 GiB). A learning-rate-zero initialization step
allocates AdamW state; there are no subsequent optimizer updates. FLUX compute
uses BF16 and activation checkpointing is disabled.

Synchronized wall time covers CPU input transfer, padding and mask assembly,
the model forward and compact hidden-state extraction. Tokenization, output
checks, logging and output destruction are outside timing. CUDA-event spans
include dispatch gaps. Group order is seeded and candidate order rotates
between rounds. Profiles and numerical comparisons are outside timing.

An earlier run, `20260907T172740665627Z`, failed before timing at 8 × 2752 because
an SDPA mock retained attention tensors. That instrumentation was removed; the
failed run supplies no usable capacity result. The successful corrected run
above supersedes the previously pending status.

## Exploratory scheduling estimates

A separate offline analysis supplied the following cost-model estimates for
14,490 training tasks, one five-pair augmentation per task, seed 167: 43,470
targets in 906 consecutive source windows, with ten sources in the final window.
The first 960 task keys and lengths matched the B0 benchmark. These figures were
reported in the analysis handoff; the full-pool computation was not rerun when
this document was finalized and its script/output was not retained as an
independent artifact. They are supporting estimates, not GPU measurements.

| Cost model | Estimated NVARC seconds | NVARC reduction | Estimated full-training throughput gain |
| --- | ---: | ---: | ---: |
| Existing B2 schedule | 1364.110 | — | — |
| Same groups, best observed alignment | 1328.770 | 2.59% | 0.34% |
| Reschedule with measured costs, original shapes | 1346.796 | 1.27% | 0.17% |
| Reschedule with all 72 shapes | 1299.270 | 4.75% | 0.63% |

The model transfers synthetic shape costs to real padding distributions and
calibrates remaining work against B2 run `20260907T164538783729Z`. Selecting on
two rounds and evaluating on the third yielded estimated NVARC reductions of
2.18–4.44%; this is neither an independent GPU repetition nor a confidence
interval. The optimum is restricted to the catalog and each 16-source window.

A separate conservative packing calculation covered only the 960-source B0
bundle. It moved complete sources into free space in subsets of existing B2
calls, preserving each row's capacity and optimizer-window membership. It
predicted 248 → 216 calls and 3,006,848 → 2,872,944 padded positions (31 of 60
windows improved). Assigning ordinary padded-shape costs to those packed calls
predicted 89.065 → 84.954 seconds, or 82.731 seconds with best alignment.
That assumption was unverified: packed masks can select different kernels.
The later actual Varlen measurements supersede this hypothetical packing model
for choosing the next NVARC implementation.

## Reproduction and provenance

The [configuration](../../configs/diagnostics/nvarc_batch_shapes.json),
[diagnostic](../../scripts/diagnostics/probe_nvarc_batch_shapes.py) and
[Modal launcher](../../scripts/cloud/probe_nvarc_batch_shapes_modal.py) capture
the original protocol. Raw results remain in
`results/nvarc-batch-shapes/20260907T174749742239Z/` in the `diffusion-arc-models`
Volume, environment `diffusion-arc`; traces and raw results are not committed.
The manifest records commands, seeds, model paths, source/config hashes and
runtime provenance. Source hashes matched the archived code when this report
was finalized. Checkpoint weight contents are not hashed.

```bash
.venv/bin/modal run --detach --env diffusion-arc scripts/cloud/probe_nvarc_batch_shapes_modal.py \
  --config configs/diagnostics/nvarc_batch_shapes.json
```

This command starts a paid H200 run using cached checkpoints. No additional
run is needed for the selected FA3 packing approach.

Primary references explaining the original hypothesis:

- [Qwen3 SDPA dispatch](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen3/modeling_qwen3.py).
- [NVIDIA matrix multiplication alignment guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
- [cuDNN attention support](https://docs.nvidia.com/deeplearning/cudnn/v1.25.0/operations/Attention.html#important-notes-on-support-surface).
