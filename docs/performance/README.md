# Performance Benchmarks

This report compares the completed single-GPU training variants and explains
their measured performance and remaining costs.

## Benchmark variants

| Variant | Status | NVARC source batch | VAE microbatch | FLUX microbatch | Optimizer batch | Length policy | Target order |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| B0 | Completed | 1 | 1 | 1 | 48 | Exact prefix length | FIFO |
| B1 | Completed | 16 | 48 | 1 | 48 | Fixed bucket padding | FIFO |

B0 establishes the baseline. B1 combines larger fixed preparation batches
with reusable attention shapes.

## Shared benchmark setup

### Dataset

- Source: `nvarc_train_arc_agi_1_training_only`, containing 14,490 synthetic
  tasks whose parents come only from the official ARC-AGI-1 training split.
- Benchmark: 960 distinct tasks selected by a seeded shuffle of sorted IDs.
- One variant per task, with identity geometry and a random color permutation.
- Five distinct pairs per variant: two demonstrations and three open targets.
- Seed `167` fixes task selection, pair selection/order, and source order.
- Maximum consumed NVARC prefix: 8,400 tokens; each target adds 900 image tokens.

The same variants are reused in every repetition: 160 sources for warm-up and
800 for measurement. Selection is random, without length balancing or
filtering. The final target output is excluded from the NVARC input.
The [dataset preparation script](../../scripts/data/prepare_nvarc_benchmark.py)
verifies parent membership in the training split and records the selected data.

### Hardware and training

Both variants use one NVIDIA H200 with 139.80 GiB device memory, eight physical
CPU cores, and 64 GiB host RAM. The software environment is PyTorch 2.13.0,
CUDA 13.0, Transformers 4.56.1, Diffusers 0.39.0, and Accelerate 1.12.0.

NVARC remains frozen in BF16 and the VAE remains frozen in FP32. The complete
FLUX transformer is trained with FP32 parameters and BF16 compute. Activation
checkpointing is disabled. Each optimizer update contains the same 48 targets
from 16 sources, with unchanged target order and loss weighting in B0 and B1.

AdamW uses a peak learning rate of `1e-5`, linear warm-up over 213 updates,
then cosine decay to zero at update 7,245. This fixed horizon is separate from
benchmark timing warm-up; all benchmark updates remain in LR warm-up.
The linked variant configs specify the remaining optimizer settings.

### Protocol

| Item | Value |
| --- | ---: |
| Independent repetitions | 3 |
| Warm-up steps per repetition | 10 |
| Measured steps per repetition | 50 |
| Total executed steps | 180 |
| Total measured steps | 150 |
| Measured target executions | 7,200 |

Each repetition starts in a fresh process from the same checkpoint, optimizer
initialization, seed, and data order. A measured step spans data loading,
NVARC and VAE preparation, buffering, FLUX forward/backward, gradient handling,
and the optimizer update. CUDA synchronization brackets the measurement;
model and data initialization are excluded.

Results report median and p95 step latency, target throughput, and peak
allocated/reserved GPU memory over the pooled measured steps. Allocated memory
tracks live tensor storage; reserved memory also includes allocator-held
blocks available for reuse. Neither includes every allocation outside PyTorch.

### Profiling

Separate seven-step runs use two wait steps, two profiler warm-up steps, and
three active steps, recording optimizer steps 5–7. They retain each variant's
training configuration and enable CPU/CUDA tracing without shape, memory, or
stack recording. Profiler overhead affects timing, so phase measurements
explain the mechanism while the unprofiled benchmark measures overall speed.

Phase timings distinguish **activity**, the union of recorded GPU kernel,
memory-copy, and memory-set intervals within a GPU range, from **gaps**, the
remaining time inside that range: **total range = activity + gaps**. Activity
includes padding work and is not a measure of useful FLOPs or SM utilization.

Configs:
[B0 profiler](../../configs/training/profiling/single_gpu_b0_torch_profile.json)
and [B1 profiler](../../configs/training/profiling/single_gpu_b1_torch_profile.json).

### Comparison and reproducibility

B0 and B1 keep the dataset, optimizer-batch membership, loss normalization,
precision, optimizer schedule, software, and hardware allocation fixed.
Step records were checked for matching source order, target counts, real
prefix lengths, learning rates, and data hashes. Pooled and per-repetition
summaries were independently recomputed from those records.

Run manifests retain commands, resolved configs, source revisions, and
environment details. Both runs included uncommitted source changes, so the
recorded base commits alone do not identify the executed code. Checkpoint
weight contents were not hashed; exact checkpoint identity remains a
reproducibility limitation.

## B0 — Exact-length FIFO baseline

The [B0 config](../../configs/training/benchmarks/single_gpu_b0.json) processes
one NVARC source, one VAE image, and one FLUX target at a time. FLUX uses exact
text-prefix lengths; the optimizer accumulates 48 targets in FIFO order.

### Benchmark result

| Median step | p95 step | Targets/s | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: | ---: |
| 15.45 s | 17.59 s | 3.094 | 113.91 GiB | 138.12 GiB |

All repetitions completed, including inputs reaching 8,400 tokens.
Benchmark run: `20260903T230458451916Z`.

### Profiler observation: cuDNN attention-plan setup

Three consecutive FLUX microbatches in the first active step illustrate the
bottleneck. Their text prefixes contain 454, 612, and 770 tokens, each
combined with the fixed 900 image tokens.

![B0 cuDNN attention plan-cache trace](assets/b0-cudnn-attention-plan-cache.png)

| Text prefix | FLUX GPU range | Recorded GPU activity | Range without GPU activity |
| ---: | ---: | ---: | ---: |
| 454 | 119.9 ms | 31.2 ms | 88.7 ms |
| 612 | 48.4 ms | 34.5 ms | 13.9 ms |
| 770 | 127.3 ms | 37.0 ms | 90.3 ms |

The 612-token prefix had already occurred in optimizer step 1, microbatch 34;
the other two lengths were new. GPU activity grows slightly across these
examples, while the new lengths have much larger gaps.

Across all 144 active forwards, every one of the 128 previously unseen prefix
lengths had two long `aten::_cudnn_attention_forward` calls; none of the 16
previously seen lengths did. Every forward still launched 25 attention
kernels.

The two setup sites are the first attention call in each block family:
five dual-stream blocks followed by twenty single-stream blocks. Their QKV
layouts differ. PyTorch's cuDNN attention cache includes dimensions and strides,
and a cache miss builds a new graph and execution plan. The red and orange
callouts mark the inferred setup sites. See the
[cuDNN MHA implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cudnn/MHA.cpp#L1341-L1382).

The optimization opportunity is therefore to reuse a limited set of attention
shapes across different real prefix lengths.

## B1 — Bucketed fixed-batch single-GPU pipeline

B1 addresses repeated attention-plan setup and the many small preparation
calls in B0:

- NVARC encodes 16 sources together, followed by one VAE batch of their
  48 targets. NVARC-to-VAE preparation remains coupled.
- NVARC pads each source batch to the smallest configured boundary covering
  its longest real source. FLUX independently pads each target's text prefix
  to its next boundary, retaining a FLUX microbatch size of one.
- Padding is masked. FIFO order, the 48-target optimizer-batch membership,
  and loss weighting remain unchanged.

The [B1 config](../../configs/training/benchmarks/single_gpu_b1.json) fixes these
batch sizes and boundaries. The [bucket policy](b1-bucket-policy.md) documents
the seven NVARC and 19 FLUX boundaries, the input-length histogram, and their
padding/setup-cost trade-off.

### Preliminary NVARC capacity probe

The [capacity diagnostic](../../scripts/diagnostics/probe_nvarc_batch_capacity.py)
tested synthetic sources with 8,400 real tokens while all models, initialized
optimizer state, gradients, and conservative buffer reserves were resident.
With one repetition per candidate, 29 sources succeeded and 30 ran out of
memory. At batch size 16, peak allocated/reserved memory was 98.54/110.23 GiB.

This established a capacity bound for the synthetic workload, rather than a
throughput-optimal batch size. It did not exercise mixed-length padding masks
or complete NVARC-to-VAE preparation. The B1 benchmark below tested the actual
coupled pipeline, including the largest bucket.

### B1 benchmark result

| Metric | B0 | B1 | Change |
| --- | ---: | ---: | ---: |
| Median optimizer step | 15.45 s | 13.91 s | -10.0% |
| p95 optimizer step | 17.59 s | 15.77 s | -10.3% |
| Targets/s | 3.094 | 3.461 | +11.9% |
| Peak allocated | 113.91 GiB | 113.90 GiB | Approximately unchanged |
| Peak reserved | 138.12 GiB | 118.84 GiB | -19.28 GiB (-14.0%) |

B1 improves complete-pipeline throughput by **11.9%**. Batching and bucket
padding changed together, so this result measures their combined effect.
The memory reduction is in reserved allocator memory; peak allocated tensor
memory is essentially unchanged.

| B1 repetition | Measured steps | Median step | p95 step | Targets/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50 | 13.904 s | 15.747 s | 3.465 |
| 2 | 50 | 13.929 s | 15.767 s | 3.460 |
| 3 | 50 | 13.927 s | 15.771 s | 3.460 |

All repetitions completed without OOM, including the 8,400-token NVARC bucket
at step 31. Benchmark run: `20260905T040037141298Z`.

Across 180 matched updates, the mean absolute step-loss difference was
0.000102, with a maximum of 0.000676. Small numerical differences are expected
when changing batch shapes in mixed precision; see
[PyTorch's numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
The loss comparison alone does not establish gradient/update equivalence;
equivalent long-term training behavior and ARC quality were not evaluated.

### B1 profiler findings: activity, gaps, and total ranges

The table reports seconds per optimizer step, averaging the three active
steps. Range durations and activity unions are summed per phase, then divided
by three. Values are rounded independently; **Change total** compares B1's
total GPU range with B0's using unrounded values. Negative changes mean shorter
ranges.

| Phase | Activity B0 | Activity B1 | Gaps B0 | Gaps B1 | Total B0 | Total B1 | Change total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVARC | 0.698 | 2.198 | 0.743 | 0.016 | 1.440 | 2.214 | +53.7% |
| VAE | 0.632 | 0.374 | 0.030 | 0.001 | 0.662 | 0.375 | -43.4% |
| FLUX forward | 2.657 | 2.824 | 3.400 | 0.409 | 6.057 | 3.233 | -46.6% |
| FLUX backward (reconstructed) | 6.013 | 6.380 | 2.910 | 0.323 | 8.923 | 6.702 | -24.9% |

**NVARC's total range increases by 53.7%** even though its gaps nearly
disappear. Fixed FIFO groups of 16 process 4,386,048 padded token positions
for 2,050,665 real positions during benchmark measurement: **2.14x the
real-token count**. The active profiler window has a 2.26x ratio. This extra
padding is consistent with GPU activity rising from 0.70 to 2.20 s per step;
the position ratios are not FLOP or runtime multipliers.

Mixed-length padding also changes NVARC's attention path: Transformers builds
an explicit mask and repeats K/V heads instead of requesting native grouped
query attention. The trace does not isolate these costs from the extra token
positions. See the [mask construction](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/masking_utils.py)
and [SDPA adapter](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/integrations/sdpa_attention.py).
Changing mask values at a fixed shape does not itself require new plans;
the [cuDNN cache key](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cudnn/MHA.cpp)
records mask presence, dimensions, and strides rather than individual values.

**VAE's total range decreases by 43.4%.** Encoding 48 images per call reduces
recorded GPU activity from 0.63 to 0.37 s per step compared with B0's
individual-image calls.

FLUX forward changes from approximately `2.66 + 3.40 = 6.06 s` to
`2.82 + 0.41 = 3.23 s`: **46.6% less total range time**. Backward changes
from `6.01 + 2.91 = 8.92 s` to `6.38 + 0.32 = 6.70 s`: **24.9% less**.
Both perform slightly more recorded GPU work, consistent with added padding,
but the reduction in gaps dominates. Padding's isolated cost was not measured.

Backward lacks a named profiler range. Its bounds were reconstructed for each
microbatch from the first to last GPU activity attributed to the autograd
worker through CUDA correlation IDs. The 144 non-overlapping spans include
loss-gradient propagation and parameter-gradient accumulation, but exclude
time outside those GPU bounds. They do not measure the full Python
`.backward()` call. The phase table also omits CPU preparation and optimizer
work, so it is not a decomposition of benchmark wall time.

The attention backend remains cuDNN. The following counts use only
`aten::_cudnn_attention_forward` and `aten::_cudnn_attention_backward` CPU
events, excluding their enclosing wrappers. A long call exceeds 20 ms.

| Attention calls across all three active steps | B0 total | B0 long | B1 total | B1 long |
| --- | ---: | ---: | ---: | ---: |
| FLUX forward | 3,600 | 256 | 3,600 | 2 |
| FLUX backward | 3,600 | 256 | 3,600 | 2 |
| NVARC forward | 1,728 | 45 | 108 | 0 |

FLUX retains the same number of attention calls while nearly eliminating the
long setup calls in both passes. This supports plan reuse as the explanation
for the reduced gaps. CPU call durations overlap GPU work and cannot be added
as separate savings.

The early trace overrepresents B0's first-use shapes: 88.9% of its active
forwards introduce a new prefix length, compared with 62.0% during the full
measurement window. B1 has already used 16 of 19 buckets before profiling;
only the 256-token bucket is new in the active window. The large early phase
improvement therefore does not imply the same percentage gain over the full
benchmark.

### Data-loading observation

Data loading fetches 16 samples synchronously before NVARC preparation,
taking 0.171 s of CPU range time per step. This is not a measurement of GPU
idle time.
