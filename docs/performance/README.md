# Performance Benchmarks

## Dataset

- Source: `nvarc_train_arc_agi_1_training_only`
- 14,490 synthetic tasks whose parents come only from the official ARC-AGI-1
  training split
- Eight reproducible augmentations per task
- Five distinct pairs per variant: two demonstrations and three open targets
- Deterministic geometry, color permutation, pair order, and sampling seeds
- Maximum consumed NVARC prefix: 8,400 tokens

Worst case: `9 * (900 cells + 29 line breaks) + 39 turn tokens = 8400`.
The final output stays in stored `input_ids` but is excluded from NVARC input;
no length-based resampling is needed for five pairs of at most 30x30 grids.

Formal benchmarks use one fixed representative subset and ordering. Its
manifest must record the dataset revision, split ID, seeds, source IDs, source
and target counts, and token-length distributions. Until this subset has an
immutable identifier, a run is not a formal B0 result.

## Configurations

| Setting | B0 | B1 |
| --- | ---: | ---: |
| NVARC source batch | 1 | Largest validated batch safe at 8,400 tokens |
| VAE microbatch | 1 | Largest separately validated safe batch |
| FLUX microbatch | 1 | 1 |
| Optimizer batch | 48 targets | 48 targets |
| Target selection | FIFO | FIFO |

B1 changes only frozen NVARC and VAE preprocessing throughput. Length-aware
reordering, dynamic batching, and different optimizer-batch membership are
separate later experiments.

With three targets per source, a full optimizer batch contains 16 source
variants.

## Benchmark protocol

| Item | Value |
| --- | ---: |
| Independent repetitions | 3 |
| Warm-up steps per repetition | 10 |
| Measured steps per repetition | 50 |
| Executed steps per repetition | 60 |
| Total executed steps | 180 |
| Total measured steps | 150 |
| Measured target executions | 7,200 |

Each repetition starts from the same checkpoint, optimizer initialization,
seed, data order, precision policy, software environment, and hardware
allocation. Model and data initialization are excluded from step statistics.

One measured step spans the complete pipeline between consecutive optimizer
updates: data loading, NVARC, VAE, buffering, FLUX forward/backward,
gradient handling, and the optimizer update. CUDA is synchronized at both
timing boundaries.

## Reported measurements

- Median and p95 end-to-end optimizer-step latency
- Target and source-variant throughput
- NVARC, VAE, FLUX, backward, and optimizer phase latency when available
- Peak allocated and reserved GPU memory
- Targets and FLUX microbatches per optimizer step
- NVARC source count and padded/unpadded token counts
- Target-prefix token distribution
- Buffer occupancy and retained bytes when available
- Total process duration and infrastructure cost

Raw step values are retained. Statistics are reported per repetition and across
all three repetitions.

## Comparison rules

Compared runs keep the following fixed:

- Dataset subset, order, and seeds
- Model, package, and container revisions
- Optimizer-batch membership and loss normalization
- Precision and activation-checkpointing policy
- GPU, CPU, and memory allocation
- Warm-up, measurement, metrics, and telemetry configuration

Record the exact command and resolved configuration. Keep failed and OOM runs
as negative results. A changed workload or protocol starts a new benchmark
series.

## Profiling

Profiler runs are separate diagnostic experiments, usually around ten steps on
the same data distribution. They locate bottlenecks and provide only rough
runtime estimates; profiler timings are not benchmark results.

Do not commit traces or other generated artifacts. Estimate run cost using:

```text
cost = billed runtime in hours * hourly resource rate recorded at run time
```
