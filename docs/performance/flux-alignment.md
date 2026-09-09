# FLUX alignment diagnostic

**Status: prepared; H200 measurements pending.** This is the controlled
alignment mode of the existing [FLUX batch-shape probe](flux-batch-shapes.md).
It compares the 24 explicit shapes currently configured in
[`single_gpu_b2.json`](../../configs/training/benchmarks/single_gpu_b2.json).
It does not change the training catalog, batch scheduler, or training code.

## Start

Inspect the complete plan locally, without checkpoint access, GPU allocation,
downloads, or generated files:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/diagnostics/probe_flux_batch_shapes.py \
  --config configs/diagnostics/flux_alignment.json --dry-run
```

Start the GPU diagnostic explicitly through the existing Modal launcher:

```bash
.venv/bin/modal run --env diffusion-arc --detach \
  scripts/cloud/probe_flux_batch_shapes_modal.py \
  --config configs/diagnostics/flux_alignment.json
```

This launches one H200 with 8 CPUs and 64 GiB host memory, using checkpoints
already cached in the `diffusion-arc-models` volume. No dataset is needed.
There are no automatic retries. The worker budget is **1,700 seconds**,
including model loading, warm-ups, timed calls, and profiling. Modal has a
separate 1,800-second function timeout, leaving time to persist partial results.

The previous catalog times imply approximately 23 minutes of forward/backward
work for the default timed and warm-up calls alone. This is a planning estimate,
not a runtime guarantee: setup, assembly, profiler overhead, and filesystem work
are additional. A timeout preserves partial results and is never reported as
a completed experiment.

## Comparisons

For each existing `(batch_size, text_length)` shape, keep its batch size fixed
and compare:

- The original text boundary.
- Text boundary rounded up to each multiple of 8, 32, 64, and 128.
- Joint text-plus-image length rounded up to each of those multiples, keeping
  the image sequence fixed at 900 tokens. The padded text length is
  `ceil((original_text + 900) / multiple) * multiple - 900`.

For example, text length 2,422 becomes 2,432 for text alignment to 32, or 2,428
for joint alignment to 32 (3,328 joint tokens). Identical physical candidates
within a comparison are measured once and retain every applicable alignment
label. Candidates above the existing 8,400-text-token contract are listed as
excluded by the limit, not as OOMs. The default plan has **140 physical cases
across 24 comparisons**.

The probe reuses the existing runtime setup: cached production checkpoints,
frozen NVARC/VAE, FP32 FLUX parameters/gradients/AdamW moments, BF16 FLUX compute,
disabled activation checkpointing, and the conservative resident preparation
reserve. Initializing optimizer state uses learning rate zero. No optimizer
updates occur during the alignment comparisons or profiles.

The existing preparation functions produce one seeded synthetic five-pair
30x30 source. Every row in a comparison uses a fixed prefix of its real NVARC
hidden states: `original_boundary - 1 - row_index`. Each variant uses the same
hidden values and VAE latent. Flow times and noise are sampled once per target
using the production sampler and B2's singleton draw order, then reused across
all variants and rounds. Seeds, prefix lengths, and candidate order are recorded.

`build_flux_microbatch` performs the actual padding. Every row has masked text
positions even at the original boundary. `compute_flux_microbatch_loss` passes
the production joint attention mask, and backward uses the production `B / 48`
loss weight. There are no mask hooks, SDPA replacements, or per-layer tensor
audits. Only padded text length changes within a comparison.

## Timing and profiling

Each candidate runs **three rounds**, each with two warm-up calls followed by
five measured calls (15 measured calls per case). Group order is seeded; each
group starts with a seeded candidate permutation rotated across rounds to
reduce order effects. Candidate blocks are interleaved within each group.

CUDA events record forward/loss, backward (including loss scaling), and their
combined span. Synchronized wall clocks separately record assembly,
forward/backward wall time, and total assembly-plus-forward/backward wall time.
Event spans may include CPU dispatch gaps. Gradient values are zeroed outside
the clocks with `set_to_none=False`, retaining their storage. Scalar loss
readback and its finite check occur after timing. Results retain every warm-up
and measured call, pooled median/min/max/p95, per-round medians, and allocated
and reserved memory peaks. Reported case peaks include its warm-ups; setup
peaks are separate. Profiler calls do not enter any of these case statistics.

After **all** timed comparisons, the configured groups `(1, 2422)`, `(2, 2749)`,
and `(4, 1348)` receive one additional warm-up and one separate CPU/CUDA profiler
call per candidate. Gzipped Chrome traces contain named assembly, forward, and
backward regions. Tensor-shape recording, stack recording, and profiler memory
tracking are disabled to limit interference and retained tensors. Attention
operator names are also saved with each trace; actual selected kernels remain
observable in the CUDA timeline. This records dispatch without forcing a backend.

## Results and interpretation

Modal prints the result directory:
`/mnt/diffusion-arc-models/results/flux-batch-shapes/<run_id>/`.

- `report.md`: all measured original/aligned cases side by side, including
  forward, backward, assembly, total wall time, memory, and relative speedup.
- `summary.json`: per-case timing statistics and round medians. Speedups compare
  against the freshly measured original within the same group, not old timings.
- `manifest.json`: exact configurations, code/config hashes, local Git identity,
  launch metadata, seeds, candidate rules, exclusions, and comparison plan.
- `worker/result.json`: raw calls, environment/package versions, checkpoint
  configs, setup footprint, profiler paths, and the original error if any.
- `worker/traces/*.json.gz`: separate profiler captures for selected groups.
- `worker.log`: worker output.

Worker records are replaced atomically after each candidate block. Completed
groups and profile groups trigger volume commits; the supervisor commits again
on exit. OOM or other worker errors stop the sweep without retrying that CUDA
context or assuming neighboring shapes fail. A setup OOM is reported as a setup
error. Timeouts retain the last persisted block; partial rows are labeled and
do not receive speedup ratios against incomplete baselines.

These are synthetic near-boundary shape comparisons, not complete training
steps or a dataset-weighted speedup estimate. Original catalog costs are retained
for context but used neither as new baselines nor as acceptance thresholds.
Input content, padding distribution, allocator history, clocks, and noise can
affect observations. The best observed candidate is subject to selection bias;
check consistency across rounds and remeasure close decisions before adoption.
Scalar loss differences are diagnostic only and do not prove full-model
gradient equivalence. CPU tests cover fixed masks/inputs and tiny production
FLUX loss/gradient agreement across padding variants; full H200 correctness and
performance remain to be measured. Checkpoint weights themselves are not hashed.

Replaying validated measured costs over the full training dataset and changing
the scheduling catalog are separate follow-ups.
