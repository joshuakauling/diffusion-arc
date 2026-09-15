# First ARC-AGI-1 training run

This is the agreed recipe for the first full epoch of single-GPU training.
It tests whether the lessons from [overfitting](../experiments/overfitting/README.md)
transfer to a broader task distribution. The runner executes the training epoch
with loss validation, fixed-seed sampling, W&B logging and full-state checkpoints.
RunPod is the target platform; volume preparation and final-test checkpoint
selection remain open.

The [run configuration](../../configs/training/arc_agi_1_epoch.json) records
these decisions as `training_run` schema version 1. Its
[loader](../../src/diffusion_arc/training/config.py) validates the complete
recipe and normalizes it into the same `TrainingConfig` used by B3.
The [trainer step](../../src/diffusion_arc/training/trainer.py) executes one
complete optimizer update, including NVARC/VAE encoding. Sampler, loss, optimizer
and scheduler are built outside and supplied to the trainer. The outer loop
owns epoch accounting and checks its expected target/update totals.
The [training entrypoint](../../scripts/train_single_gpu.py) loads this config
before importing the GPU runtime and calls
[`run_training(config)`](../../src/diffusion_arc/training/runner.py). The runner
checks that the prepared view contains the expected number of distinct tasks
with the configured pair count, retains the partial final batch and checks
the observed sequence, target and update totals. Event conditions use the
configured start, periodic and final steps. Validation and sampling share one
W&B record with the current optimizer update, plus an initial record at step zero.
Checkpoints are published after sampling and their metrics join that record.
The separate [profiler](../../scripts/profile_single_gpu.py) and benchmark
entrypoints require baseline configs. Resolve relative paths from the JSON
file's directory. Before the complete run, set the
RunPod output directory and W&B entity/project before execution; credentials
come from `WANDB_API_KEY`. `checkpointing.keep_last: null` retains all three
first-epoch checkpoints. A numeric retention limit is deferred to volume setup,
and final testing stays disabled until its checkpoint policy is agreed.

## Model and execution

Start from the original FLUX.2-klein-base-4B checkpoint with fresh optimizer
state. Train the complete transformer. Keep NVARC/Qwen3-4B and the FLUX VAE
frozen, retaining the established 30×30 canvas, 480×480 rendering and palette.

Target RunPod for the training run. Prepare and populate its persistent volume
after the local training implementation; make data, model and output paths
configurable so the runner is independent of the cloud provider. The existing
Modal jobs remain historical benchmark evidence.

Use the [B3 execution path](../performance/b3.md) on one H200: packed FA3 for
NVARC and FLUX, NVARC early exit after block 27 with raw states from 9/18/27,
16-source preparation windows, VAE chunks of 48 and FLUX packs capped at
12,288 joint tokens. NVARC and FLUX compute use BF16; VAE execution, FLUX
parameters, optimizer state and loss reduction use FP32. Keep activation
checkpointing disabled for this run.

## Data and batching

| Setting | Decision |
| --- | --- |
| Training pool | 14,490 synthetic NVARC tasks whose two parents are in official ARC-AGI-1 training |
| Duration | One complete epoch, one sampled sequence per synthetic task |
| Sequence | Five distinct pairs sampled without replacement, in random order |
| Supervised targets | Outputs of pairs 3, 4 and 5, conditioned on 2, 3 and 4 preceding demonstration pairs |
| Optimizer batch | 16 sequences × 3 targets = 48 targets |
| NVARC call budget | 134,400 real tokens (8,400 × 16); split whole sources for smaller budgets |
| Final batch | 10 sequences = 30 targets; retain this batch |
| Total | 43,470 target exposures and 906 optimizer updates |
| Seed | 167; preserve task order and augmentation/RNG state for resume |

Shuffle tasks reproducibly. For each sequence, sample one permutation of all
10 ARC colors and one of the eight D4 symmetries uniformly, including identity.
Apply the same color mapping and geometry to every input/output pair in that
sequence. Random pair selection already determines pair order. Geometry is
enabled from the first epoch because synthetic tasks can share parent tasks;
it does not multiply the number of training sequences.

Use the `nvarc_train_arc_agi_1_training_only` pool, recorded in
`data/nvarc_data/nvarc_train_arc_agi_1_training_only_tasks.jsonl`.
Validation/test tasks and their derived augmentations must remain outside
training. Do not introduce the VAE-safety filtering used in the tiny overfit
experiment into this full-pool run.

The [epoch preparation command](../../scripts/data/prepare_training_data.py)
reuses the benchmark's archive checks, task conversion and augmentation-key
generation. Run it against the locally available archive and index:

```bash
PYTHONPATH=src .venv/bin/python scripts/data/prepare_training_data.py --config configs/training/arc_agi_1_epoch.json
```

It requires the complete indexed pool, checks both parents against the official
training IDs, verifies source checksums and publishes a new bundle only after
all tasks and keys validate. Existing bundles are never overwritten. The split
manifest records source/config/code hashes, task order, augmentation identities
and expected batch counts. Generated data under `data/training/` is ignored by Git.
Configuration loading and preparation require neither model loading nor W&B
credentials; tokenization still enforces the prefix limit when the dataset is read.

The prepared key order already contains the seeded shuffle, so the shared B3
DataLoader uses `shuffle=False`. Sampled split views explicitly opt into
`geometry_sampling: "uniform"`; historical views without this option retain
their original geometry cycle and random pair/color choices.

## Loss and flow times

For each target, construct `z_t = (1 - t) * z_0 + t * epsilon` with independent
Gaussian noise and velocity target `epsilon - z_0`. Compute squared velocity
error in FP32 and average over latent channels before the spatial reductions:

```text
target_loss = 0.1 * mean_canvas + 0.8 * mean_grid + 0.1 * mean_boundary
```

The canvas term covers all 900 tokens. The grid term covers the true top-left
output grid. The boundary is the one-cell L immediately below/right of that
grid, including its corner and clipped to the canvas. For grid height `h` and
width `w`, it has `min(h + 1, 30) * min(w + 1, 30) - h * w` tokens.
When the boundary is absent at 30×30, omit its term and divide the remaining
sum by 0.9, giving ordinary mean squared error.

Each target receives the same total coefficient mass, regardless of grid size.
Average over the actual optimizer target count, including the final 30-target
batch. Weight physical microbatch contributions by their share of this count.
There is no additional area-based target weight.

In every full optimizer batch, randomly choose **4 of 48 targets for exact
`t=1`** and draw the remaining 44 times independently from `Uniform(0, 1)`.
Assign time/noise in logical target order before packing, independently of grid
size and target position. The final batch uses 3 endpoints and 27 uniform
draws: 3,623 endpoint exposures across the epoch. Endpoint targets receive no
additional loss multiplier.

Training and loss validation share the
[regional reduction](../../src/diffusion_arc/training/step/loss.py), including
the missing-boundary rule. Prepared targets retain grid dimensions only for
this loss; the transformer receives no target-shape hint. The loop assigns
times and noise to the logical optimizer batch before deciding its physical
packs. Each step returns CPU-only target metrics containing regional means,
weighted contributions, times and demonstration counts for later W&B aggregation.
Targets keep their logical order; preparation and packing diagnostics are separate.

## Optimizer and continuation

| Setting | Decision |
| --- | --- |
| Optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01 |
| Learning rate | 1e-5 |
| Warmup | Linear over 50 optimizer updates: 2e-7 on update 1, 1e-5 on update 50 |
| Subsequent schedule | Constant learning rate |
| Gradient clipping | Global norm 1.0 after accumulation, before the optimizer update |

Continuation keeps optimizer, scheduler, RNG, data position and global update
count; it does not restart warmup. A later cosine cooldown can be a separate
experiment, but is not part of this first epoch.

The [warmup scheduler](../../src/diffusion_arc/training/step/lr_schedule.py) advances
after each optimizer update. Step metrics capture the learning rate used before
that advance. A CPU save/reload test verifies exact continuation of random draws,
losses, gradients, learning rates, AdamW moments and weights through HF checkpoints.
GPU numerical agreement remains part of the integration smoke run.

## Validation and final test

Use the existing official ARC-AGI-1 v1.0.2 evaluation tasks in
`data/benchmarks/arc-agi-1/evaluation`. Shuffle the 400 sorted task IDs with
Python `Random(167)`: the first 100 are validation and the remaining 300 are
reserved for the final test. The first 32 validation tasks form the fixed
sampling panel. The frozen identities and payload hashes are in the
[split manifest](../../data/benchmarks/arc_agi_1_validation/manifest.json).
Select recipes and checkpoints without using the reserved test tasks.

| Check | Workload | Update numbers in the first epoch |
| --- | --- | --- |
| Loss | All 100 tasks / 104 queries at t = 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 | 0, 100, 200, …, 800, 906 |
| Direct sampling | Fixed 32 tasks / 33 queries, two independent samples with 40 linear Euler steps | 0, 300, 600, 906 |

The agreed intervals are **100 updates for loss and 300 for sampling**, plus
the initial and final checks. The final check at 906 replaces the scheduled
check at 900. Use all original demonstrations and all query inputs, without
augmentation, voting, target-shape hints or test-time training.

Fix validation noise by query identity and time so checkpoints see the same
probes. Report the weighted loss and its regions separately at every time.
The combined diagnostic is `11/12 * mean_inner_times + 1/12 * loss_at_t1`;
the five inner points approximate the uniform-time expectation. Use separate
fixed seeds for generation. Keep Gaussian initial states and Euler integration
in FP32, then apply VAE decoding and strict PAD-boundary grid decoding.

Exact-grid pass@1 and pass@2 are primary; shape and cell accuracy are diagnostics.
Shape accuracy and invalid-grid rate use all 66 predictions as their denominator.
Cell accuracy is the equally weighted mean of per-prediction cell accuracies
among shape-correct predictions, regardless of grid area. It is omitted when
no predicted shape matches; the shape-correct count remains zero.
For tasks with multiple queries, one shared sample index must solve all queries
for a task-level success. NVARC conditioning never receives the query outputs.
The released NVARC checkpoint has known public-benchmark contamination. This
split isolates FLUX tuning and final comparison; it does not establish unseen
generalization for the entire system. See the
[baseline](../experiments/baselines/nvarc-arc-agi-1-direct-sampling.md).

Cache frozen validation conditioning and VAE target latents once on CPU, staging
one packed group to the GPU at a time. A two-repetition H200 timing check measured
**32.5 s per loss check** and **115.4 s per sampling check**, including transfers,
packing, decoding and metrics. Cache preparation took 7.9 s once and used 2.54 GiB
of CPU memory. At the agreed cadence, validation adds about **13.1 minutes** to
the B3-derived training estimate of 138 minutes: approximately **2 h 31 min**
before model setup, checkpoint writes, image logging and the final test.
These are runtime estimates, not evidence of learning quality. Reproducibility
details remain in the [timing artifact](../../artifacts/benchmarks/validation/20260911T112135652096Z/measurement/report.json).

## Logging and checkpoints

Use Weights & Biases for experiment logging. Associate training metrics,
validation results, generated samples and checkpoints with the global optimizer
update count. Keep individual optimizer-step observations; do not replace them
with 50-step averages in the training process.

### Agreed scalar logging

| Cadence | Measurements |
| --- | --- |
| Every optimizer update | Total weighted loss; unweighted per-target canvas, grid and boundary MSE means; actual weighted contributions of these three terms |
| Every optimizer update | Weighted target-loss means and target counts for exact `t=1`, uniform-time draws, and each of 2/3/4 preceding demonstrations |
| Every optimizer update | Learning rate used for the update, global gradient norm before clipping, clipping indicator |
| Every optimizer update | Actual and cumulative target counts, epoch progress, full training-step duration and targets per second |
| Every optimizer update | NVARC processed tokens, FLUX text/image/joint tokens, and actual FLUX forward-call count; retain per-update and cumulative counts |
| Every loss validation | Weighted loss and regional means at each of the six fixed times, the inner-five-time mean, exact `t=1` loss, the combined diagnostic, and check duration |

Region means average targets equally after reducing channels and the region's
tokens within each target. Exclude targets without a boundary from the raw
boundary mean and record its valid-target count. Their boundary contribution
to the weighted objective is zero; use the actual renormalized coefficients
for the remaining terms. Averaging weighted contributions across all targets
must recover the logged total loss.

Keep subgroup counts so later window averages can be reconstructed as
`sum(step_mean * step_count) / sum(step_count)`, including the smaller final
batch. Dashboard smoothing is a visualization choice, not a substitute for
that target-weighted reduction. The separate subgroup curves do not retain
individual target records or arbitrary intersections of time and context groups.

NVARC tokens count positions actually encoded in each source preparation call.
FLUX text tokens sum the conditioning-prefix lengths of consumed targets,
including overlapping prefixes reused across targets. Image tokens are
`900 * target_count`, and joint tokens are text plus image tokens. These count
forward input positions, not FLOPs or backward work. FLUX calls count executed
packed forwards, not targets or optimizer updates. Keep training, validation
and sampling counters separate; evaluation counts include each repeated time
or Euler-step call. Reuse the existing preparation and executed-pack metadata.

The inner-five-time validation mean is a fixed-probe approximation to the
uniform-time expectation; label it accordingly. It is not a new random-time
validation pass.

### Agreed metric names

The names below are fixed across tasks and runs. Braces abbreviate a finite
set of separate keys; they are not literal characters in a logged name.

| Measurements | Keys |
| --- | --- |
| Training objective and regions | `train/loss`, `train/mse_{canvas,grid,boundary}`, `train/contribution_{canvas,grid,boundary}` |
| Training subgroups | `train/loss_t1`, `train/loss_uniform`, `train/loss_demos_{2,3,4}` |
| Subgroup denominators | `train/targets_t1`, `train/targets_uniform`, `train/targets_demos_{2,3,4}` |
| Optimizer | `train/learning_rate`, `train/grad_norm`, `train/grad_clipped` |
| Training timing | `train/step_seconds`, `train/targets_per_second` |
| Training workload | `train/targets`, `train/nvarc_tokens`, `train/flux_text_tokens`, `train/flux_image_tokens`, `train/flux_joint_tokens`, `train/flux_forward_calls` |
| Progress | `progress/optimizer_step`, `progress/epoch_fraction`, `progress/targets_seen` |
| Validation objective | `val/loss`, `val/loss_uniform_proxy`, `val/loss_t1`, `val/loss_t_{0_1,0_3,0_5,0_7,0_9}`, `val/seconds` |
| Sampling accuracy | `sample/pass_at_1`, `sample/pass_at_2`, `sample/solved_tasks_at_1`, `sample/solved_tasks_at_2`, `sample/shape_accuracy`, `sample/cell_accuracy`, `sample/invalid_grid_rate` |
| Sampling denominators and duration | `sample/task_count`, `sample/shape_correct_count`, `sample/seconds` |
| Checkpoint records | `checkpoint/seconds`, `checkpoint/size_bytes`, `checkpoint/path` |

Append `_total` to cumulative token and forward-call counters. Validation and
sampling reuse the workload names under `val/` and `sample/`. Regional
validation means follow the time-specific names, for example
`val/mse_grid_t_0_5` and `val/mse_grid_t1`. `val/loss` is the combined diagnostic.
Counts and rates retain the reduction rules above; the checkpoint path is a
string record rather than a scalar curve.

### Agreed sample presentation

Log a W&B table under `sample/predictions` at each sampling check: updates
**0, 300, 600 and 906**. Each table contains one row per query, giving 33 rows
for the fixed 32-task validation panel. Associate each table version with the
optimizer update and model state used for sampling.

| Column | Content |
| --- | --- |
| `task_id` | Original ARC task identifier |
| `query_index` | Zero-based query index within the task |
| `comparison` | One labeled comparison image showing input, expected output, sample 1 and sample 2 |
| `sample_1_exact` | Whether sample 1 exactly solves this query |
| `sample_2_exact` | Whether sample 2 exactly solves this query |

Show both classified 30×30 canvases in full using the model's target palette,
so incorrect dimensions and padding remain visible. These are discrete class
predictions from the VAE decode, not its raw RGB output. Do not crop generated
canvases to the expected output shape. Input and reference grids use the same
cell scale and are padded to 30×30 for comparison.
The row flags describe query-level correctness; aggregate task-level pass@1
and pass@2 keep the multi-query rules defined above.

Build the 33 comparison images from the samples already generated for scoring.
This adds image preparation and upload, with no additional FLUX forwards.
Each check creates a fresh immutable table. `sample/seconds` includes generation,
VAE decoding and scoring; it excludes image rendering and W&B serialization/upload.
Sampling reuses the validation conditioning cache, so `sample/nvarc_tokens` is
zero; the one-time encoder work is counted under validation.

### Agreed hardware logging policy

Disable W&B's automatic system-metric collection and collect only the listed
measurements at the agreed intervals below.
The implementation pins W&B 0.30.0 and explicitly sets `x_disable_stats`;
verify the disabled collector in the integrated smoke test. W&B documents it in its
[system-metrics guidance](https://docs.wandb.ai/support/models/articles/how-can-i-disable-logging-of-system-metr).

| Cadence | Measurements | Keys |
| --- | --- | --- |
| Every optimizer update | PyTorch GPU allocated bytes, reserved bytes, and peak allocated bytes during the complete training step | `system/gpu_allocated_bytes`, `system/gpu_reserved_bytes`, `system/gpu_peak_allocated_bytes` |
| Every 30 seconds | Active GPU utilization (%), power draw (W), and temperature (C) | `system/gpu_utilization_percent`, `system/gpu_power_watts`, `system/gpu_temperature_celsius` |
| Every 30 seconds | Training-process CPU utilization and resident RAM bytes | `system/cpu_utilization_percent`, `system/ram_rss_bytes` |

Record system samples with `system/elapsed_seconds` and `system/phase`. Keep
training memory peaks separate from validation, sampling and checkpoint work.
Reset/read the peak counters at explicit phase boundaries. CPU utilization
uses 100% per fully occupied CPU core and can exceed 100%. It describes the
training process since the previous CPU read, which may span phases; GPU
utilization is the device's recent NVML sample, not a 30-second average.

The runner reads training memory immediately after `Trainer.step`, before any
validation or sampling. Other sensors are read at the first training-step
boundary at least 30 seconds after their previous sample; there is no background
hardware thread. Long steps or evaluation blocks can therefore extend the
interval. Due hardware readings join the same W&B call as the update's metrics,
sent after the run-service blocks. Use fixed metric names, without task IDs in keys.
For example, `train/loss` and `train/loss_demos_2` are shared time series; task
identities belong in sample tables or diagnostic records, not separate metric
names such as `train/loss_<task_id>`.
Do not enable parameter/gradient/activation histograms, model graph watching,
profiler traces, or automatic console capture. Keep profiling disabled throughout
this training run. Preserve diagnostic console logs in the run artifacts.
Collect the small loss reductions together
with the existing loss calculation, without additional model forwards or a
separate GPU synchronization per scalar. Verify logging overhead against the
same workload without logging before the full epoch.

### Checkpoint cadence

Save checkpoints every 300 optimizer updates, aligned with direct sampling,
and at the end of the epoch. Following the final-validation convention above,
the first epoch saves at updates **300, 600 and 906**. Sampling results and the
corresponding checkpoint must describe the same model state. Save full-state
checkpoints to the persistent storage prepared for RunPod, with the output
directory supplied by configuration. Retention must be configurable; the
discussed allowance of approximately 100 checkpoints is provisional. The first
epoch produces only three scheduled checkpoints, all of which are retained.
Volume capacity and population are part of the later RunPod setup.

`training/run/checkpointing.py` writes under
`<output_directory>/checkpoints/<wandb_run_id>/step-000300/`:

- `transformer/`: Diffusers `save_pretrained` configuration and FP32 safetensors,
  split into shards of at most 5 GB except any larger indivisible tensor.
- `training_state.pt`: AdamW, scheduler, dedicated flow generator and Python,
  NumPy, Torch CPU and selected-device CUDA RNG states. Loading uses
  `torch.load(..., weights_only=True, map_location="cpu")`.
- `checkpoint.json`: completed update/source/target counts, workload totals,
  resolved recipe, ordered training-content fingerprint, validation identity,
  W&B run identity, package versions, source hash and file-size inventory.

The frozen NVARC/VAE weights and validation cache are not copied. Saves use a
temporary sibling directory, flush files and publish by directory rename. Failed
writes leave earlier checkpoints intact. Existing checkpoint paths are never
overwritten. `keep_last: null` keeps all; a positive count removes only completed
checkpoints belonging to this W&B run. File sizes detect incomplete/truncated
writes; the inventory is not a checksum of every saved weight byte.

To continue, set `models.resume_from` to the completed checkpoint directory in
the same recipe. The loader resolves it relative to the config file. Resume
requires the same training recipe, ordered data content and validation split;
model/data paths and frozen model files must stay unchanged. Deployment output
paths and retention can change. The current implementation resumes within this
finite epoch; changing the recipe or extending a completed epoch is separate work.

The runner loads FLUX from `transformer/`, reconstructs the frozen CPU cache,
then restores optimizer/scheduler and RNGs before the next update. The data
cursor skips consumed sources before tokenization, independent of prefetched
worker samples. The step-zero checks and warmup do not repeat. Rebuilt NVARC
cache tokens are counted once at the next validation check.

W&B uses the saved entity/project/run ID with `resume="must"`. If its history
extends past the checkpoint, training recomputes those updates and suppresses
their duplicate log records until it catches up; existing history is preserved.
Use the latest completed checkpoint for recovery. A checkpoint and a W&B upload
are separate writes: records absent from the server at interruption are not
reconstructed. W&B server resume and GPU/storage overhead still need the remote
integration test; local tests substitute the resumed SDK state.

At roughly 4 billion parameters, FP32 weights plus two FP32 AdamW moments imply
about **48 GB per full checkpoint**, or **144 GB for three**, before small metadata
overhead. This is a capacity estimate, not a measured file size; a save also needs
space for its complete temporary directory before retention can remove older saves.

## Remaining run setup

Choose the checkpoint-selection rule before using the reserved test tasks.
The proposed first-epoch protocol is to evaluate the final checkpoint at update
906 on all 300 reserved tasks, using two direct samples and 40 Euler steps per
query. Keep final testing a separate explicit action after training; this
proposal still needs agreement.

Implementation work:

1. The validated configuration, deterministic data preparation and recipe core
   are implemented. CPU numerical tests cover regional loss and gradients,
   exact endpoint quotas, unchanged random assignments across pack budgets,
   48-target batches plus a 30-target tail, warmup and gradient clipping.
   Step results contain update scalars and individual CPU-only target metrics;
   logging derives grouped losses outside the trainer. Gradients are released
   before control returns to the outer loop. These tests use small substitute models;
   GPU execution and tuning of the NVARC budget remain unverified.
2. The normal training entrypoint and epoch runner are implemented, reusing B3
   model preparation and packing without starting a profiler. CPU orchestration
   tests cover complete and partial batches, incompatible prepared views,
   early exhaustion and propagation of failed steps. Historical profiling
   remains under `scripts/profile_single_gpu.py` with baseline configs.
   Loss validation now uses the same bundle checks, tokenization and CPU-cache
   preparation as `scripts/benchmark_validation.py`. Preparation respects the
   configured NVARC token budget and VAE image batch size. The runner checks the
   expected 100 tasks / 104 queries before loading models and reuses the cache
   for all six time probes at each scheduled check. Numerical tests verify mode,
   gradient and RNG preservation. Token and forward counters include all six
   probes; NVARC preparation tokens appear only at the first validation event.
   `val/seconds` measures the recurring check, excluding one-time preparation.
3. W&B training scalars and the bounded hardware collector are implemented in
   `training/run/logging.py`. Tests cover target/group reductions, absent boundaries,
   partial batches, selected sensors, intervals, GPU identity, one log call per
   update and failure propagation. The SDK receives the resolved recipe without
   credentials; automatic system/code/console capture is disabled. Before a run,
   supply `logging.project`, `deployment.output_directory` and `WANDB_API_KEY`;
   `logging.entity` can select the owner. Validation losses use the recipe's
   regional weights and endpoint/uniform mixture, with raw regional means for
   each time and separate validation workload totals. Initial validation logs
   at step zero; subsequent checks join that update's single metrics record.
   Fixed sampling now reuses the same cache for 32 tasks / 33 queries and calls
   `inference.generate_packed_arc_samples` through the shared scoring wrapper.
   Generation receives conditioning only; expected outputs enter scoring and
   the comparison table afterward. Tests cover full invalid canvases, task-level
   multi-query success, metric denominators, event cadence and failure propagation.
   A local offline check with W&B 0.30.0 wrote exactly two history records and
   no automatic system-stat records. A further offline media check preserved
   two distinct table versions with 33 query rows each and their comparison
   images. All inputs were synthetic. GPU reads were substituted; real GPU
   measurements and logging overhead still need the integrated smoke run.
4. Full-state save/resume is implemented and tested with real small HF
   safetensors. Tests cover numerical continuation, rejected incompatible or
   incomplete saves, failed writes, retention, data cursors and W&B history
   catch-up. No remote run or full-size checkpoint has been created locally.
5. Verify an integrated GPU smoke run, including interrupted continuation and
   the existing W&B run, and measure logging/checkpoint overhead before the
   full epoch.

W&B project/account settings, RunPod volume paths, hardware/runtime verification
and volume population belong to deployment preparation, after implementation.
