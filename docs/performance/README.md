# Performance Benchmarks

## Benchmark variants

| Variant | Status | NVARC source batch | VAE microbatch | FLUX microbatch | Optimizer batch | Length policy | Target order |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| B0 | Completed | 1 | 1 | 1 | 48 | Exact prefix length | FIFO |
| B1 | Planned | Fixed worst-case-safe size | Fixed worst-case-safe size | 1 | 48 | Fixed bucket padding | FIFO |
| B2 | Concept | Length-dependent | Fixed size | Length-dependent | 48 | Fixed bucket padding | Length-aware |

B0 is the single-GPU baseline. B1 adds fixed batching and stable padded shapes
without changing FIFO order or optimizer-batch membership. Length-aware
reordering and dynamic batch sizes start with B2.

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

## B1 — Bucketed fixed-batch single-GPU pipeline

B1 keeps the NVARC-to-VAE preparation flow and FIFO target order unchanged.
The selected NVARC source batch size is 16; FLUX keeps a microbatch size of one
and the optimizer batch remains 48 targets. The fixed VAE microbatch size is
still open. NVARC sources and FLUX text prefixes use fixed bucket padding.

The [B1 bucket policy](b1-bucket-policy.md) records the selected seven NVARC and
19 FLUX boundaries, their occupancy on the B0 inputs, the length histogram,
and the padding/setup-cost trade-off. The boundaries are selected;
implementation and end-to-end performance measurements are pending.

### Preliminary NVARC capacity probe

Before implementing buckets, use
[`probe_nvarc_batch_capacity.py`](../../scripts/diagnostics/probe_nvarc_batch_capacity.py)
to bound the number of 8,400-token sources that the existing NVARC runner can
encode with training memory resident. The completed Modal run
`20260905T023541561165Z` used one repetition per candidate: 29 sources succeeded
and 30 ran out of memory. At the selected B1 source batch size of 16, peak
allocated memory was 98.54 GiB and peak reserved memory was 110.23 GiB on an
H200 with 139.80 GiB total memory. Raw results are stored at
`results/nvarc-capacity/20260905T023541561165Z/` on `diffusion-arc-models` in the
`diffusion-arc` environment. New GPU execution requires separate approval.

On the approved CUDA host with the existing checkpoints available locally:

```bash
.venv/bin/python -B scripts/diagnostics/probe_nvarc_batch_capacity.py \
  --config configs/training/benchmarks/single_gpu_b0.json \
  --sequence-length 8400 --max-batch-size 64 --repetitions 3
```

Alternatively, after GPU-run approval, launch the complete search on Modal:

```bash
modal run --detach --env diffusion-arc \
  scripts/cloud/probe_nvarc_batch_capacity_modal.py \
  --config configs/training/benchmarks/single_gpu_b0.json \
  --max-batch-size 64 --repetitions 3
```

The [Modal launcher](../../scripts/cloud/probe_nvarc_batch_capacity_modal.py)
uses one H200, eight physical CPU cores, 64 GiB RAM, and a 30-minute Function
timeout. One Function owns the entire sequential search, so `--detach` covers
all candidates. There are no automatic Function retries. Only `src`, `scripts`,
and `configs` are uploaded; datasets, local checkpoint weights, and root-level
profiler traces are excluded. The existing `diffusion-arc-models` Volume supplies
the checkpoints and stores results under `results/nvarc-capacity/<run-id>/`.

Local validation checks the options, config layout, and checkpoint paths without
requiring local weights or a dataset. Missing remote checkpoints fail before
candidate workers start. Local Git provenance, command, Modal version, image ID,
and resource allocation are recorded in the manifest. Uploaded source and config
hashes must match the launch metadata; Git is not required in the container.
The Volume is committed after each completed candidate (including failure
records) and again when the search ends. Failed searches raise a Modal error.
A hard timeout or container termination can still lose the currently running
candidate's uncommitted updates; earlier committed candidates remain available.
The completed run above exercised both the launcher and the GPU probe. It used
the documented Modal command with `--repetitions 1` and `--detach`.

The probe uses B0's checkpoint paths, seed, precision, optimizer, and initial
scheduled learning rate. It does not read the benchmark dataset. Production
prompt rendering and the checkpoint tokenizer construct a deterministic
synthetic five-pair task with 30x30 grids. Its final consumed prefix contains
exactly 8,400 real tokens; the final output is excluded. Each physical batch
repeats that source. The length argument currently accepts only 8,400, avoiding
truncation or artificial valid padding tokens.

Each candidate runs in a fresh, sequential Python process with all three models
loaded. One real single-target optimizer step on a separate synthetic two-pair
1x1 task initializes AdamW moments and FLUX gradients. Its target still uses
the normal 480x480 canvas and 900 image tokens. Initialization is not a benchmark
step; its allocations are reported separately. Every FLUX parameter must have
a gradient and both AdamW moment tensors afterward. Gradients stay resident,
matching preparation before the next optimizer batch in the current loop.

The process also retains an explicit conservative storage reserve:

- 16 independent NVARC source tensors of shape `[8400, 7680]` in NVARC dtype;
- 48 target latents in VAE dtype, covering a full FIFO optimizer batch even
  though the queue holds fewer than 48 targets before preparation;
- another 48 target latents for the previous `optimizer_inputs` tuple, which
  remains referenced while the loop prepares the next source batch.

These zero-filled reserve tensors are never model inputs. Counts, dtypes, and
bytes are written to each candidate's result. They model retained tensor
storage rather than replaying the FIFO allocator history or VAE chunk storage.
No `empty_cache()` calls alter the allocator between probe repetitions.

The search doubles batch size from one, capped by `--max-batch-size`, then
bisects the observed success/OOM bracket. Success requires every requested
runner call to finish, including source-state concatenation and copies.
Synchronization brackets each call; the first call includes cold shape setup,
and subsequent calls reuse that process. Outputs are released before the next
call. Only a typed CUDA OOM during probing supplies an OOM bound. Setup OOM,
other exceptions, killed workers, and incomplete results abort the search.

Results default to the Git-ignored `artifacts/nvarc-capacity/<run-id>/` directory;
`--output-dir` selects a different new directory. The script refuses to
overwrite an existing destination. `manifest.json` records the resolved config,
source hashes, and revision; each `batch-*/result.json` records software, GPU,
checkpoint configs, workload hash, initialization and reserve details, timings,
and allocated/reserved peaks. Completed repetitions are persisted individually,
including before a subsequent failure. Worker logs, `candidates.json`, and
`summary.json` retain the search history. Checkpoint weights are not hashed;
immutable model revisions must still be established for formal comparisons.

`largest_tested_success` and `smallest_tested_oom` describe the observed bracket,
assuming feasibility is monotone across the searched batch sizes. Reaching the
search limit without OOM leaves `boundary_known` false. Allocated headroom is
total device memory minus peak allocated; it is not driver-free memory and
does not include every allocation outside PyTorch. Reserved memory is reported
separately. No safety margin or final B1 batch size is selected automatically.

This is a capacity diagnostic, not a throughput benchmark or numerical
agreement test. Full-length valid sources do not establish memory behavior for
mixed-length padding masks, coupled NVARC-to-VAE preparation, or complete B1
optimizer steps. No additional capacity probe is planned for B1. The script
does not modify training code, checkpoint
weights, or benchmark configs, download models/data, or launch a remote job.

## B2 — Length-aware dynamic-batch single-GPU pipeline

B2 may select NVARC and FLUX batch sizes by length bucket. Its scheduling,
buffering, and remainder policy are intentionally left open until B1 is
measured.
