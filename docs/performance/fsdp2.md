# FSDP2 benchmarks on two H200 GPUs

The first FSDP2 benchmark, **B5**, completed on **6 October 2026**, with **8.710 targets/s**
and **103.90 GiB peak allocated memory per GPU**. Compared with the historical
two-GPU DDP benchmark, throughput was **8.9% lower** and peak allocation was
**20.8% lower**. This establishes the initial FSDP2 baseline before workload-specific
performance tuning.

**B6**, measured later the same day, increased pack capacity from 12,288 to
16,384 tokens. It reached **9.015 targets/s** at **123.46 GiB peak allocated
memory per GPU**: **3.5% more throughput** than B5 for **19.56 GiB more memory**.

## Setup and measurement

Both variants used two H200 GPUs, 16 allocated CPU cores and 128 GiB host RAM on Modal.
The prepared view contains 960 sources; each optimizer batch has 48 targets from
16 sources. Seed `167` and `global_step_target_v1` define the data/flow sequence.
NVARC remains frozen in BF16 and the VAE in FP32. FLUX uses FP32 parameters and
reductions with BF16 compute, FA3 attention, `legacy_bfd_v1` packing with a
12,288-token budget in B5 or 16,384 in B6, and no activation checkpointing. The runtime uses PyTorch
2.13.0, CUDA 13.0 and expandable allocator segments.

FLUX is sharded at transformer-block boundaries. Blocks reshard after forward;
every microbatch backward synchronizes gradients, which accumulate as shards
until one optimizer update. Ranks with fewer packs execute zero-weight dummy
forward/backward calls to match the collective schedule. Frozen models remain
replicated.

Each variant used three fresh runs with the same initialization and seed, each
with **10 warm-up updates and 50 measured updates**. Timing includes data retrieval, NVARC/VAE
preparation and the complete training step, using the slowest rank per step.
Model loading, validation, sampling, W&B logging and checkpointing are excluded.
Memory statistics are maxima per GPU over the measured window, not GPU sums.

## B5 baseline results

| Repetition | Median step | p95 step | Targets/s |
| --- | ---: | ---: | ---: |
| 1 | 5.544 s | 6.442 s | 8.699 |
| 2 | 5.535 s | 6.430 s | 8.714 |
| 3 | 5.528 s | 6.435 s | 8.717 |
| **All 150 measured steps** | **5.537 s** | **6.441 s** | **8.710** |

All 180 updates completed with finite logged losses. The measured window processed
7,200 targets in **826.627 s**. Peak allocated/reserved memory was
**103.90 / 109.66 GiB**; throughput spread between repetitions was **0.20%**.
The stored aggregate was reproduced from all step records.

The [DDP baseline](ddp.md) reached 9.566 targets/s, a 5.073 s median step and
131.13 / 133.23 GiB peak allocated/reserved memory. Logged batch membership,
prefix lengths and learning rates match, but source versions and packing differ:
DDP used the revised balanced packing. These are complete-system comparisons,
not isolated measurements of sharding overhead.

B5 executed **150 dummy calls among 2,046 FLUX calls (7.3%)** in the measured
window; 108 of 150 updates included padding. Additional communication and dummy
work are plausible costs, but their time contributions have not been established
from traces. B6 uses part of this memory headroom for larger packs; its measured
benefit is reported below.

## B5 logged loss comparison

All 60 B5 updates, including warm-up, were compared with the three DDP runs from
17 September and one single-GPU run from 15 September (DDP with `world_size=1`).
The logged source/target order, prefix lengths, endpoint counts and learning rates
match. Manifests also agree on loss, optimizer, schedule and model initialization.
The older B3 recipe is not used for this loss comparison.

For each matching step and every repetition pair, the relative difference is
`abs(loss_A - loss_B) / abs(loss_B)`. This yields 540 comparisons for FSDP2/DDP
and 180 for each comparison with single-GPU execution; curves are not averaged
before computing differences.

| Comparison (A / B) | Mean absolute relative difference | Maximum |
| --- | ---: | ---: |
| FSDP2 / DDP | 0.43% | 3.80% at step 59 |
| FSDP2 / single GPU | 0.52% | 6.35% at step 42 |
| DDP / single GPU | 0.45% | 6.40% at step 42 |

Repeated runs of the same strategy also differ: maximum pairwise differences are
3.70% within FSDP2 and 3.88% within DDP. At step 42, single-GPU loss is 0.363510,
versus repetition means of 0.341374 for DDP and 0.342741 for FSDP2. This larger
single-GPU discrepancy predates FSDP2. The logs show similar loss behavior, but
do not establish gradient/update equivalence or long-run convergence.

## B6 larger packs

[B6's configuration](../../configs/training/benchmarks/b6_fsdp2_16k.json) changes
only the pack limit from B5. Logged batch membership, prefix lengths, endpoint
counts and learning rates match. Manifests also match in dataset fingerprint,
hardware/software environment and training settings apart from that limit;
source-code hashes differ. The separate
[capacity probe](../../artifacts/flux-capacity/20261006T082941914783Z/summary.json)
passed at 18,432 tokens but ran out of memory at 19,456 in its second update.
B6 uses 16,384 to leave headroom for the complete training pipeline.

| Repetition | Median step | p95 step | Targets/s |
| --- | ---: | ---: | ---: |
| 1 | 5.305 s | 6.217 s | 9.010 |
| 2 | 5.301 s | 6.215 s | 9.018 |
| 3 | 5.296 s | 6.221 s | 9.015 |
| **All 150 measured steps** | **5.300 s** | **6.222 s** | **9.015** |

All **180 updates** completed without OOM and with finite logged losses and
gradient norms. The measured window processed 7,200 targets in **798.703 s**;
peak allocated/reserved memory was **123.46 / 128.79 GiB per GPU**. Throughput
spread `(max - min) / mean` across repetitions was **0.09%**. The aggregate and
each repetition summary agree exactly with the step records.

Relative to B5, throughput increased **3.5%**, median step time fell **4.3%**,
and total measured time fell **3.4%**. Peak allocation rose **19.56 GiB per GPU**.
FLUX calls across both ranks fell from **2,046 to 1,566 (23.5%)**, matching the
prior CPU packing calculation: mean calls per rank/update fell from **6.82 to
5.22**, and dummy calls from **150 to 117**. Of 150 measured updates, **99**
included dummy padding, versus 108 for B5.

This converts some of FSDP2's memory savings into throughput, but the gain is
modest relative to the call-count reduction. Larger packs still process the
same real targets; fewer calls do not imply proportionally less computation.
B6 remains **5.8% below B4 DDP throughput**, with **7.67 GiB less peak allocated
memory per GPU**. No B6 profiler capture was collected; these logs do not
separate communication savings from compute time or other overhead.

Using the same loss metric as above over all 60 updates and all nine repetition
pairs gives 540 comparisons for each reference:

| Comparison (A / B) | Mean absolute relative difference | Maximum |
| --- | ---: | ---: |
| B6 / B5 | 0.49% | 6.06% at step 52 |
| B6 / B4 DDP | 0.51% | 5.05% at step 59 |

Within B6, the three unordered repetition pairs have a mean absolute relative
loss difference of **0.47%**, with a **7.03%** maximum at step 59. Average loss
differences are similar across and within configurations, but individual steps
differ more. These logged losses do not establish gradient/update equivalence
or long-run convergence.

## B5 profiler stream mapping

The separate B5 FSDP2 captures for [rank 0](../../rank-0.trace.json.gz) and
[rank 1](../../rank-1.trace.json.gz) record optimizer step 11 after 10 warm-up
updates. Both show the same stream roles. The following counts refer to rank 0
(trace ID `9C065FA113484377919AD51DDB9263A4`), which contains six FLUX
forward/backward calls.

![FSDP2 rank-0 trace excerpt with CPU threads above GPU streams 7, 17, 18, 22 and 26](assets/fsdp2-stream-overview.png)

The screenshot shows an excerpt of optimizer step 11. Time runs from left to
right. The upper `python 41` tracks are CPU activity: thread 41 enters the
forward/backward calls, while thread 110 executes autograd work. The lower
`python 0` tracks belong to GPU 0; each stream is a separate queue of GPU work.
Labels such as `stream 18 18` show the stream name followed by its numeric ID.

The broad `ProfilerStep#10` and `optimizer_step/11` bars mark enclosing ranges,
not continuous kernel execution. The profiler uses a zero-based step label;
the training step is 11. The tooltip's roughly 4.792 s is the duration of that
profiler range, not the benchmark median or the duration of this visible excerpt.
The table below identifies the streams using the full trace; its counts cover
the complete recorded step.

| CUDA stream | Observed work | Role |
| --- | --- | --- |
| 7 | Model/optimizer kernels, copies and parameter copy-out | Main computation stream |
| 18 | 306 `multi_tensor_apply_kernel` copy kernels, launched by `aten::_foreach_copy_` | Copy local parameter shards into the AllGather input buffer |
| 22 | 306 NCCL AllGather kernels | Assemble full block parameters before forward/backward computation |
| 26 | 156 NCCL ReduceScatter kernels and 845 addition kernels | Reduce gradients into local shards and accumulate them across microbatches |
| 17 | One FP32 AllReduce of 169 elements, lasting about 5.5 microseconds | Gradient-norm reduction inside `train/clip_grad` |

Stream 18 carries `FSDP::all_gather (...)` annotations because its copies belong
to that phase; its kernels do not communicate between GPUs. Stream 22 performs
the actual AllGather. The [DDP rank-0 capture](../../rank-0_ddp.trace.json.gz)
instead contains 108 gradient AllReduce kernels on stream 17.

In the screenshot, stream 26 is active during the backward regions, while
streams 18 and 22 prepare and gather parameters. The small mark on stream 17
near the end is the gradient-norm AllReduce. Compare horizontal positions to
see when work overlaps; the colors alone do not identify an operation's role.
This zoom level illustrates the schedule but does not establish how much
communication delays computation.

Stream IDs are specific to a capture, not permanent assignments. Identify work
by the kernel name and collective arguments, then follow its CUDA launch
correlation or `External id` to the CPU operation and enclosing ranges such as `FSDP::all_gather`,
`FSDP::post_backward_reduce`, or `train/clip_grad`.

The block schedule is AllGather → forward → reshard → AllGather → backward →
ReduceScatter. The outer FSDP group retains full parameters after forward.
Separate streams permit overlap across blocks, and FSDP2 already supplies
prefetching and communication overlap by default
([PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)).
Thus, this baseline already includes framework optimizations. Kernel durations
on different streams must not be added as step overhead: they can overlap, and
NCCL kernel time can include waiting for another rank. Quantifying exposed
communication time and rank imbalance remains a separate analysis.

## Evidence

- B5: [benchmark configuration](../../configs/training/benchmarks/fsdp2_training.json),
  [launch record](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/launch.json),
  [combined results](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/summary.json)
  and [runtime/source manifest](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/repetition-1/manifest.json).
- B5 step records: [run 1](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/repetition-1/steps.jsonl),
  [run 2](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/repetition-2/steps.jsonl),
  [run 3](../../artifacts/fsdp2/20261006T073103320421Z-benchmark-2gpu/repetition-3/steps.jsonl).
- B6: [launch record](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/launch.json),
  [combined results](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/summary.json)
  and [runtime/source manifest](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/repetition-1/manifest.json).
- B6 step records: [run 1](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/repetition-1/steps.jsonl),
  [run 2](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/repetition-2/steps.jsonl),
  [run 3](../../artifacts/fsdp2/20261006T085059072341Z-benchmark-2gpu/repetition-3/steps.jsonl).
- [DDP reference results](../../artifacts/retired-worktrees/ddp-7b16/artifacts/ddp/20260917T085250710411Z-benchmark-2gpu/summary.json)
  and [single-GPU step records](../../artifacts/retired-worktrees/ddp-7b16/artifacts/ddp/benchmark-1gpu-20260915T153715-steps.jsonl).

Raw artifacts are retained locally and on Modal, outside Git. Both FSDP2 runs
are stored on Volume `diffusion-arc-models` in environment `diffusion-arc`, under
`results/ddp/`, using run directories `20261006T073103320421Z-benchmark-2gpu`
(B5) and `20261006T085059072341Z-benchmark-2gpu` (B6). The historical `ddp`
directory name does not describe their actual execution strategy.
