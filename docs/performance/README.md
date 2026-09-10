# Performance benchmarks

We used the same training workload to develop a faster single-GPU pipeline:
[B0](b0.md) established the simple end-to-end run, [B1](b1.md) introduced fixed
padded batches, [B2](b2.md) added batch selection and CPU prefetch, and
[B3](b3.md) replaced the padding catalogs with FA3 sequence packing.

**B3 is the single-GPU baseline for implementing distributed training.**
The variant reports contain the implementation changes and profiler findings.
The B3 investigations are summarized separately for [NVARC](b3-nvarc.md) and
[FLUX](b3-flux.md).

## Shared benchmark setup

### Dataset

The benchmark uses 960 distinct tasks selected by a seeded shuffle from
`nvarc_train_arc_agi_1_training_only`. This pool contains 14,490 synthetic tasks
whose parents belong only to the official ARC-AGI-1 training split.

Each task supplies one augmentation with identity geometry, a random color
permutation, and five distinct pairs: two demonstrations and three targets.
Seed `167` fixes task selection, pair selection, and source order. Selection
does not balance or filter by length. NVARC consumes prefixes of up to 8,400
tokens, excluding the final target output; each FLUX target adds 900 image tokens.
The [preparation script](../../scripts/data/prepare_nvarc_benchmark.py) records
the selection and verifies parent membership in the training split.

### Hardware and training

All four variants use one **NVIDIA H200 with 139.80 GiB device memory**,
eight allocated CPU cores, and 64 GiB host RAM. They share PyTorch 2.13.0,
CUDA 13.0, Transformers 4.56.1, Diffusers 0.39.0, and Accelerate 1.12.0.
B3 additionally uses FlashAttention-3.

NVARC stays frozen in BF16 and the VAE stays frozen in FP32. The complete FLUX
transformer is trained with FP32 parameters and BF16 compute. Activation
checkpointing is disabled. Each optimizer update uses the same 48 targets
from 16 sources with the same per-target loss weighting. B2 and B3 may change
physical execution order within that batch; optimizer-batch membership stays fixed.

### Protocol

Each variant runs **three independent repetitions**, each starting in a fresh
process from the same checkpoint, optimizer initialization, seed, and data order.
Every repetition uses **10 warm-up steps followed by 50 measured steps**.
This gives 150 measured steps and 7,200 measured target executions per variant.
The same 160 sources serve as warm-up and the same 800 as measurement in each run.

CUDA synchronization brackets each timed step. Timing covers input retrieval,
NVARC/VAE preparation, buffering, FLUX forward/backward, gradient handling, and
the optimizer update. Model and dataset initialization are excluded.

The comparison below pools all 150 measured steps. Median and p95 describe
step latency; throughput is total targets divided by total measured time.
Memory columns report the maximum PyTorch allocated and reserved memory across
those steps. Individual runs are listed in the variant reports.

### Profiling

Separate Torch Profiler runs execute seven optimizer steps: two wait steps,
two profiler warm-up steps, and three recorded steps (5–7). CPU and CUDA tracing
show model execution and gaps between GPU operations. Shape, memory, and stack
recording are disabled for these traces.

In the phase tables, **activity** is the combined duration of GPU kernels,
copies, and memory sets, counting overlapping intervals once. **Gaps** are the
remaining time within the phase range: **total range = activity + gaps**.
Backward ranges were reconstructed from GPU activity where no explicit range
was available.

These short traces help explain bottlenecks. Their timings include profiler
overhead and cover different steps from the benchmark, so they are not a
breakdown of its full-run latency. Variant-specific observations are documented
in the individual reports.

### Comparison and reproducibility

Source order, target counts, real prefix lengths, learning rates, and dataset
hashes were checked across the compared runs. Reported summaries were checked
against the step records. This is a performance comparison on a fixed workload;
loss and numerical checks are documented in the variant reports.

Those reports link the configs, run manifests, summaries, and original analyses.
Manifests record the software environment and source identity. Some runs included
uncommitted changes, so the base commit alone does not identify their code.
Checkpoint weight contents were not hashed in these benchmarks.

Historical B0–B2 recipes and FLUX diagnostics are preserved at
[2ff4276](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
NVARC diagnostics are preserved at
[db6d49f](https://github.com/joshuakauling/diffusion-arc/tree/db6d49f).
The current runtime uses B3.

## Results

All rows combine 150 measured steps on the same workload. Throughput gains
are relative to B0 and include all changes in each variant.

| Variant | Median step | p95 step | Targets/s | Throughput vs B0 | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [B0 — individual calls](b0.md) | 15.45 s | 17.59 s | 3.094 | — | 113.91 GiB | 138.12 GiB |
| [B1 — fixed buckets](b1.md) | 13.91 s | 15.77 s | 3.461 | +11.9% | 113.90 GiB | 118.84 GiB |
| [B2 — scheduled batches and prefetch](b2.md) | 11.32 s | 13.47 s | 4.201 | +35.8% | 126.48 GiB | 133.06 GiB |
| [**B3 — FA3 packing**](b3.md) | **9.06 s** | **10.63 s** | **5.250** | **+69.7%** | **131.59 GiB** | **137.44 GiB** |
