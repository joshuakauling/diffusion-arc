# Performance Benchmarks

## Dataset

- Source: `nvarc_train_arc_agi_1_training_only`
- 14,490 synthetic tasks whose parents come only from the official ARC-AGI-1
  training split
- Benchmark: 960 distinct tasks selected by a seeded shuffle of sorted task IDs
- One variant per selected task; identity geometry and a random color permutation
- Five distinct pairs per variant: two demonstrations and three open targets
- Reproducible pair selection/order and source order; seed `167`
- Maximum consumed NVARC prefix: 8,400 tokens

Worst case: `9 * (900 cells + 29 line breaks) + 39 turn tokens = 8400`.
The final output stays in stored `input_ids` but is excluded from NVARC input;
no length-based resampling is needed for five pairs of at most 30x30 grids.

The same 960 variants are reused in every repetition: 160 for warm-up and 800
for measurement. Selection is random, not balanced or filtered by token length;
no distribution comparison is required for this first version. Multiple
variants per task and pair-coverage policies for full training remain open.

The preparation manifest records input/task checksums, selected task order,
seeds, counts, and the key-file checksum. The benchmark records token lengths
during execution; preparation does not tokenize the full dataset.

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

Prepared B0 config:
[`single_gpu_b0.json`](../../configs/training/benchmarks/single_gpu_b0.json).
It references `data/benchmarks/single_gpu_b0/splits/train`. Prepare this bundle
once from the existing local archive and training-only index:

```bash
.venv/bin/python -B scripts/data/prepare_nvarc_benchmark.py
```

- Writes only the 960 selected raw tasks plus split/key files and
  `splits/train/manifest.json`; refuses to overwrite an existing bundle.
- Verifies both parents against the official ARC-AGI-1 training keys and checks
  selected task bytes against the source index.
- Converts labelled pair-bank lists to `train` lists with an empty `test` list,
  preserving pair indices. This is a training-only performance workload.
- Uses the existing key generator with one sampled variant, `max_pairs=5`, and
  `max_cells=9000` so five valid 30x30 input/output pairs are never size-filtered.
- Local preparation only: no downloads, model loading, or GPU execution.

The bundle uses the existing `splits/` and `raw/` layout and is Git-ignored.

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

Learning rate: peak `1e-5`, linear warm-up over **213 optimizer updates**, then
cosine decay to zero at update **7,245** (a fixed benchmark reference horizon,
not an epoch of this subset or a commitment to the full-training recipe).
This is separate from the ten timing warm-up steps. Each repetition restarts
the schedule, so all 60 updates remain in LR warm-up; the standard scheduler
starts the first update at LR zero. This is a throughput experiment, not a
learning-curve comparison.

## Running and outputs

To launch a GPU benchmark after preparing the data (separate run approval):

```bash
modal run --detach --env diffusion-arc \
  scripts/cloud/benchmark_single_gpu_modal.py \
  --config configs/training/benchmarks/single_gpu_b0.json
```

- One H200, eight physical CPU cores, 64 GiB RAM, two-hour timeout for all
  repetitions combined; three sequential fresh Python processes.
- Existing `diffusion-arc-models` Volume; results in `results/benchmarks/<run-id>/`.
- Missing split/key files fail before the GPU Function is invoked.
- Each repetition writes `manifest.json`, `steps.jsonl`, and `summary.json`;
  the run directory contains the manifest and pooled summary.
- No profiler, W&B, checkpoint saving, resume, or activation checkpointing.
  B0 completed all three repetitions with prefixes up to 8,400 tokens on H200.
- Timing includes the LR scheduler, but excludes metric bookkeeping and result
  writes. CUDA synchronization happens only at optimizer-step boundaries for
  timing; existing training checks remain unchanged.
- A short stream, incomplete optimizer batch, or exception marks the run failed.
  Completed step records survive ordinary Python/CUDA errors; a forced container
  termination may lose the current repetition's in-memory records.

The profiler entrypoint and its existing config remain available separately.
Local CUDA execution uses `python scripts/benchmark_single_gpu.py --config ...`.
Output defaults to `artifacts/benchmarks/<run-id>/`; override with `--output-dir`.

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

Currently emitted: step latency, loss, LR used, target/microbatch counts,
completed-source throughput, prepared source keys and lengths, padded/unpadded
NVARC token counts, target-prefix lengths, and allocated/reserved memory peaks.
Source completion means consuming its final target, including across updates.
Per-phase times, buffer bytes, and provider billing/cost remain separate follow-ups.
Reserved memory can include the CUDA allocator cache populated during warm-up.

## Results

| Run | Median step (s) | p95 step (s) | Targets/s | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 15.45 | 17.59 | 3.094 | 113.91 | 138.12 |

- Run ID: `20260903T230458451916Z`; all three repetitions completed successfully.
- 3 × 60 optimizer steps, including 150 measured steps. Timing and throughput
  are pooled across measured steps; memory values are their maxima.
- Total wall time: **48 min 42 s**, including per-repetition initialization.
- Local results (Git-ignored):
  [summary](../../artifacts/benchmarks/20260903T230458451916Z/summary.json),
  [manifest and repetition files](../../artifacts/benchmarks/20260903T230458451916Z/).
  Remote: `results/benchmarks/20260903T230458451916Z/` on
  `diffusion-arc-models` in the `diffusion-arc` environment.
- Code provenance: `e6f5fd9` plus uncommitted changes recorded in the manifest;
  the commit alone does not identify the executed code.

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

Manifests include resolved settings, split/key file hashes, source revision,
package versions, GPU identity, Modal image/allocation, and checkpoint configs.
Raw-data identity is recorded in `splits/train/manifest.json` in the data bundle.
Checkpoint weights are not hashed; record their immutable revisions before
reporting formal results.

## Profiling

Profiler runs are separate diagnostic experiments, usually around ten steps on
the same data distribution. They locate bottlenecks and provide only rough
runtime estimates; profiler timings are not benchmark results.

Do not commit traces or other generated artifacts. Estimate run cost using:

```text
cost = billed runtime in hours * hourly resource rate recorded at run time
```
