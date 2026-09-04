# Performance Benchmarks

## Benchmark variants

| Variant | Status | NVARC source batch | VAE microbatch | FLUX microbatch | Optimizer batch | Length policy | Target order |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| B0 | Completed | 1 | 1 | 1 | 48 | Exact prefix length | FIFO |
| B1 | Planned | Largest validated safe batch | Largest validated safe batch | 1 | 48 | Exact prefix length | FIFO |

B0 is the single-GPU baseline. B1 changes only frozen NVARC and VAE
preprocessing throughput. Length-aware reordering, dynamic batching, and
different optimizer-batch membership are not part of either variant.

## Shared benchmark setup

### Dataset

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
for measurement. Selection is random, not balanced or filtered by token length.
Multiple variants per task and pair-coverage policies for full training remain
open.

Prepare the benchmark bundle once from the existing local archive and
training-only index:

```bash
.venv/bin/python -B scripts/data/prepare_nvarc_benchmark.py
```

The command writes only the 960 selected raw tasks plus split, key, and manifest
files. It verifies both parents against the official ARC-AGI-1 training keys,
checks selected task bytes against the source index, and refuses to overwrite
an existing bundle. The resulting `data/benchmarks/single_gpu_b0/` bundle is
Git-ignored.

### Protocol

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
updates: data loading, NVARC, VAE, buffering, FLUX forward/backward, gradient
handling, and the optimizer update. CUDA is synchronized at both timing
boundaries. With three targets per source, one optimizer batch contains 16
source variants and 48 FLUX microbatches.

### Reported measurements

- Median and p95 end-to-end optimizer-step latency
- Target and completed-source throughput
- Peak allocated and reserved GPU memory
- Targets, FLUX microbatches, and source variants per optimizer step
- NVARC padded/unpadded tokens and target-prefix token distribution
- Per-phase latency, buffer occupancy, retained bytes, and infrastructure cost
  when available

Raw step values are retained. The current implementation emits step latency,
loss, learning rate, throughput counts, prepared source keys and lengths,
token counts, and memory peaks. Per-phase times, buffer bytes, and provider cost
remain unavailable. Source completion means consuming its final target,
including across optimizer updates.

### Comparison rules

Compared variants keep the following fixed unless a new benchmark series is
declared:

- Dataset subset, order, and seeds
- Model, package, and container revisions
- Optimizer-batch membership and loss normalization
- Precision and activation-checkpointing policy
- GPU, CPU, and memory allocation
- Warm-up, measurement, metrics, and telemetry configuration

Record the exact command and resolved configuration. Keep failed and OOM runs
as negative results. Manifests include the configuration, split/key hashes,
source revision, package versions, GPU identity, Modal allocation, and
checkpoint configs. Checkpoint weights are not hashed; record their immutable
revisions before reporting formal results.

## Running and outputs

Launch the benchmark after preparing the data:

```bash
modal run --detach --env diffusion-arc \
  scripts/cloud/benchmark_single_gpu_modal.py \
  --config configs/training/benchmarks/single_gpu_b0.json
```

- One H200, eight physical CPU cores, 64 GiB RAM, and a two-hour timeout
- Three sequential repetitions in fresh Python processes
- Existing `diffusion-arc-models` Volume
- Results in `results/benchmarks/<run-id>/`
- Missing split/key files fail before the GPU Function is invoked
- No profiler, W&B, checkpoints, resume, or activation checkpointing

Each repetition writes `manifest.json`, `steps.jsonl`, and `summary.json`; the
run directory contains the pooled summary and manifest. Completed step records
survive ordinary Python/CUDA errors, while forced container termination may
lose the current repetition's in-memory records.

Local CUDA execution uses
`python scripts/benchmark_single_gpu.py --config <config>`. Output defaults to
`artifacts/benchmarks/<run-id>/`; override it with `--output-dir`.

Estimate provider cost from the billed duration and the resource price recorded
at run time:

```text
cost = billed runtime in hours * hourly resource rate
```

## B0 — Exact-length FIFO baseline

### Configuration

[`single_gpu_b0.json`](../../configs/training/benchmarks/single_gpu_b0.json)
references `data/benchmarks/single_gpu_b0/splits/train`. NVARC, VAE, and FLUX
all use a microbatch of one; each optimizer update accumulates 48 targets in
FIFO order.

Learning rate: peak `1e-5`, linear warm-up over 213 optimizer updates, then
cosine decay to zero at update 7,245. This is a fixed benchmark horizon, not an
epoch of the subset or a commitment to the full-training recipe. It is separate
from the ten timing warm-up steps. Each repetition restarts the schedule, so all
60 updates remain in LR warm-up.

### Benchmark result

| Median step | p95 step | Targets/s | Peak allocated | Peak reserved |
| ---: | ---: | ---: | ---: | ---: |
| 15.45 s | 17.59 s | 3.094 | 113.91 GiB | 138.12 GiB |

- Run ID: `20260903T230458451916Z`
- All 3 x 60 optimizer steps completed, including 150 measured steps
- Total wall time: 48 min 42 s, including per-repetition initialization
- Maximum consumed prefix: 8,400 tokens
- Local results (Git-ignored):
  [summary](../../artifacts/benchmarks/20260903T230458451916Z/summary.json) and
  [repetition files](../../artifacts/benchmarks/20260903T230458451916Z/)
- Remote results: `results/benchmarks/20260903T230458451916Z/` on
  `diffusion-arc-models` in the `diffusion-arc` environment
- Code provenance: `e6f5fd9` plus uncommitted changes recorded in the manifest;
  the commit alone does not identify the executed code

### Profiler observation: cuDNN attention-plan setup

The separate B0 profiler run kept the same data order, batch configuration, and
LR scheduler. Its seven steps used a 2 wait + 2 profiler warm-up + 3 active
window, so the trace covers optimizer steps 5–7. Profiler timings are not
benchmark results.

Profiler config:
[`single_gpu_b0_torch_profile.json`](../../configs/training/profiling/single_gpu_b0_torch_profile.json).

```bash
modal run --detach --env diffusion-arc \
  scripts/cloud/profile_single_gpu_modal.py \
  --config configs/training/profiling/single_gpu_b0_torch_profile.json
```

Trace on `diffusion-arc-models`:
`results/torch-profiler/<run-id>/single_gpu_b0_torch_profile/trace.json`.

The first active profiler step contains three consecutive FLUX microbatches
with text prefixes of 454, 612, and 770 tokens. Each also contains the fixed
900 image tokens.

![B0 cuDNN attention plan-cache trace](assets/b0-cudnn-attention-plan-cache.png)

| Text prefix | FLUX GPU range | Recorded GPU activity | Range without GPU activity |
| ---: | ---: | ---: | ---: |
| 454 | 119.9 ms | 31.2 ms | 88.7 ms |
| 612 | 48.4 ms | 34.5 ms | 13.9 ms |
| 770 | 127.3 ms | 37.0 ms | 90.3 ms |

Recorded GPU activity is the union of kernels, memory copies, and memory sets
inside the profiler range; it is not an SM-utilization measurement. Useful GPU
work increases with sequence length, while the large latency difference is
primarily host-side waiting with no recorded GPU activity.

The 612-token prefix had already occurred in optimizer step 1, microbatch 34.
The 454- and 770-token prefixes were new in this process. Across all 144 active
FLUX forwards, every one of the 128 previously unseen prefix lengths had two
long `aten::_cudnn_attention_forward` calls; none of the 16 previously seen
lengths had a long call. All forwards still launched 25 attention kernels, so
attention was not skipped for the faster case.

The two long calls are the first attention call in each FLUX block family. The
model has five dual-stream followed by twenty single-stream blocks. Their QKV
layouts differ, and tensor dimensions and strides are part of PyTorch's cuDNN
MHA cache key. On a cache miss, PyTorch builds a new cuDNN graph and execution
plan. The red and orange callouts mark these inferred dual-stream and
single-stream setup sites. See the
[PyTorch 2.13 cuDNN MHA implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cudnn/MHA.cpp#L1341-L1382).

Exact FLUX sequence lengths therefore produced frequent cuDNN attention-plan
cache misses in B0. Repeated sequence lengths reused the cached plans and did
not show the two long host-side attention calls.

Do not commit profiler traces or other large generated artifacts.
