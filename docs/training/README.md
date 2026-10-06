# Training System Overview

## Scope

This page explains the overall Diffusion-ARC training flow and links to the
relevant code and detailed documentation. The root
[`README.md`](../../README.md) remains authoritative for project scope,
milestones, evaluation, and non-goals.

The [first ARC-AGI-1 training recipe](arc-agi-1-recipe.md) records the agreed
dataset, augmentation, objective, optimizer and validation setup for the next
full-epoch run, including logging and checkpoint continuation.

## Overall workflow

**Encode each source once. Train its eligible targets from the shared state.**

The flow below describes the [B3 single-GPU baseline](../performance/b3.md).

The ordered CPU loader prefetches tokenized samples. An outer loop groups whole
sequences by their eligible target count using `iter_optimizer_batches`. In the
current recipe, 16 five-pair sequences provide 48 targets; the final ten sources
provide 30. Each group is passed to `Trainer.step(sequences)` for one update.
No prepared targets are carried between steps.

The step trims each source before its final target output and calls frozen
NVARC under a **134,400-token budget** (8,400 × 16). Smaller budgets split whole
sources into consecutive packs while preserving logical order. Each source has
separate causal attention and reset position IDs. NVARC stops after block 27,
retaining raw states after blocks 9, 18 and 27. The VAE then encodes target
canvases in independent image batches (48 in the current recipe).

Flow noise/times are assigned in logical target order. FLUX packs those targets
under its separate **12,288 joint-token budget**, counting each prefix plus 900
image tokens. Backward weights each pack by its share of the actual target
count. One optimizer update and one scheduler advance follow all packs.
Native execution clears gradients with `set_to_none=True` before the step
returns; DDP zeroes its bucket views in place to retain shared storage.
Construction clears gradients once
before the first step and sets FLUX to training mode; NVARC/VAE remain frozen.

`TrainStepResult` contains total loss, gradient/LR metrics, total step time
including encoding and workload counters. Its `target_metrics` tuple holds CPU-only
measurements in source order, then eligible target order: loss, sampled time,
demonstration count, boundary presence, regional MSEs and weighted contributions.
The tuple index also identifies the target in the packing diagnostics. Regional
measurements are `None` when the objective does not provide them; a 30×30 target
has `has_boundary=False`, no boundary MSE and zero regional boundary contribution.
Detailed preparation and packing metadata live separately under `diagnostics`.
Its `flux_microbatch_plans` describe target assignments; the distribution strategy
turns them into a call schedule that can include FSDP padding calls.
No tensors or autograd graphs leave the step.

The outer loop owns global progress, logging and event timing. Logging derives
endpoint/uniform and demonstration groups from individual target measurements,
including their counts and combinations across steps. The
[training entrypoint](../../scripts/train.py) loads a validated
`TrainingConfig` and passes it to
[`run_training`](../../src/diffusion_arc/training/runner.py). The training runner attaches configured services to the shared
[`training_loop.py`](../../src/diffusion_arc/training/training_loop.py), which builds components
and executes the selected update count. The separate
[`benchmark_runner.py`](../../src/diffusion_arc/training/benchmark_runner.py)
uses the same `run_training_loop` function with `_BenchmarkCallbacks` instead of
`_TrainingCallbacks`. These callback methods run synchronously at defined points
in the loop; benchmark callbacks add measurements without training services.
[`logging.py`](../../src/diffusion_arc/training/run/logging.py) derives the configured
training scalars, reads selected hardware counters and sends one combined W&B
record per update. Validation prepares fixed CPU caches once and evaluates them
at the configured steps, including the initial model. Its losses and workload
counters join the same update record. Fixed-seed sampling reuses a subset of
that cache, calls the existing inference pipeline and adds scores plus a W&B
comparison table. Full-state checkpoints follow sampling at the configured
steps. A resume restores the optimizer, scheduler, RNGs and committed data cursor,
then continues the same W&B run without replaying existing history records.
The separate [profiling entrypoint](../../scripts/profile_single_gpu.py) retains
the B3 Torch/Nsight workflow. The auxiliary `TrainingCollator` serves diagnostics
and is not part of the production step.

## Selecting single-GPU or DDP execution

Both strategies use the same training loop. Schema v2 recipes require explicit
execution, RNG and packing choices; missing fields fail validation:

| Recipe field | Native single GPU | DDP |
| --- | --- | --- |
| `execution.distributed_strategy` | `single_process` | `ddp` |
| `deployment.gpu_count` | `1` | Requested rank count |
| `flow.rng_scheme` | `legacy_stream_v1`, or `global_step_target_v1` | `global_step_target_v1` |
| `execution.flux_packing_algorithm` | `legacy_bfd_v1`, or `balanced_release_v1` | `balanced_release_v1`, or `legacy_bfd_v1` |

Use `python scripts/train.py --config <recipe>` for native execution and
`torchrun --standalone --nproc-per-node=N scripts/train.py --config <recipe>`
for a DDP recipe with `deployment.gpu_count=N`. The cloud training launchers
select the corresponding command from the recipe. Native execution creates no
process group. `scripts/train_single_gpu.py` remains a compatibility entrypoint.

For strategy comparisons, select `global_step_target_v1` and the same packing
algorithm on both sides, with identical data, global batch size and seeds.
Endpoint and high-noise quotas apply to the global batch before rank assignment.
Historical single-GPU recipes retain their stateful RNG and original packing.

DDP aggregates training metrics across ranks. Rank zero owns validation,
sampling, W&B and shared checkpoint files; every rank participates in checkpoint
coordination and records its RNG state. Full-state resume requires the same
strategy and world size. A weights-only start can change execution strategy
because it starts a new optimizer and scheduler. Legacy single-GPU checkpoints
remain readable with their original default behavior.

## Model boundaries

| Component | State | Role |
| --- | --- | --- |
| NVARC/Qwen3-4B | Frozen | Encode complete ARC source sequences. |
| FLUX.2 VAE | Frozen | Encode target canvases into clean image latents. |
| FLUX.2 transformer | Fully trainable | Predict flow velocity from noisy image latents and NVARC conditioning. |
| AdamW | FLUX parameters only | Update the complete FLUX transformer. |

NVARC and the VAE run without training autograd graphs. FLUX parameters and
optimizer state remain FP32, the FLUX forward uses BF16 autocast, and the loss
is reduced in FP32.

## Core objects

| Object | Meaning |
| --- | --- |
| `ArcTokenizedTrainingSample` | One complete tokenized ARC source sequence with its recorded target positions and grids. |
| `NvarcSequenceState` | One source's NVARC hidden states, shared by all of its eligible target inputs. |
| `FluxInput` | One target, its exact prefix boundary, its clean latent, and a reference to the shared sequence state. |
| `EncodedTrainingBatch` | Prepared targets plus CPU metadata for their NVARC/VAE execution. |
| `TrainStepResult` | Update scalars, workload counters, target measurements and separate diagnostics returned to the outer loop. |
| `TargetMetrics` | One target's loss, flow time, demonstration count and regional measurements, before logging aggregation. |
| `StepDiagnostics` | CPU-only preparation details, physical FLUX packs and logical target prefix lengths. |
| `FluxMicrobatch` | Exact text prefixes, attention-segment boundaries and image positions for one packed FLUX call; flow tensors are supplied separately. |
| `PackedFluxForward` | A normal module forward for one physical FLUX pack, sharing the original transformer's parameters; optionally wrapped in DDP. |

The shared hidden-state tensor is released after its last target prefix has been
copied into a physical microbatch. Python later collects the small
`NvarcSequenceState` object when no `FluxInput` references it anymore.

Microbatch losses are weighted by target count, so changing the physical
grouping preserves the mathematical objective; finite-precision gradients can
still differ. The [FLUX packing guide](flux-sequence-packing.md) documents
physical grouping and loss normalization, including a smaller final batch.

## Package responsibilities

`arc_data` defines ARC tasks, augmentation, prompts, and tokenization. `model`
defines neural components, forward operations, and their tensor interfaces.
`training` composes those components with targets, objectives, optimizer steps,
and validation. `inference` generates predictions from target-free conditioning.
Neither `model` nor `inference` imports training code; validation passes only
conditioning to generation and scores the returned predictions afterward.

Within `model`, `nvarc/` contains the frozen encoder and explicit attention
binding. `flux/` contains the transformer, text conditioning, velocity prediction,
packing and modulation. The independent VAE encoder/decoder lives in
[`model/vae.py`](../../src/diffusion_arc/model/vae.py); shared tensor interfaces
remain in [`model/contracts.py`](../../src/diffusion_arc/model/contracts.py).
The component packages import no backends on their own; model setup still
explicitly registers NVARC attention and selects the FLUX packing backend.

## Training module map

`runner.py` and `benchmark_runner.py` select training services or measurement
hooks around `training_loop.py`. Both execute `Trainer.step()` for each optimizer update.
`step/` contains update components; `run/` contains validation, logging and
checkpoint services. Shared configuration, setup, data loading and optional
profiling remain at the package root.

`Trainer.distribution_strategy` is selected by `build_distribution_strategy()`.
`DdpStrategy` and `FsdpStrategy` inherit shared rank communication and loss checks
from `DistributedStrategy`, which extends `SingleDeviceStrategy`. These classes
coordinate device/process execution; they do not select the loss or training recipe.
The training loop creates `PackedFluxForward` before applying DDP or FSDP2.
The trainer clips and clears gradients; the strategy only specifies whether
clearing may set gradients to `None`, preserving DDP gradient bucket views.

| Module | Responsibility |
| --- | --- |
| [`config.py`](../../src/diffusion_arc/training/config.py) | Validate one strict schema v2 and construct `TrainingConfig`. |
| [`setup.py`](../../src/diffusion_arc/training/setup.py) | Select the device, seed RNGs, load frozen/trainable models, and construct AdamW. |
| [`data.py`](../../src/diffusion_arc/training/data.py) | Build the dataset, DataLoader and whole-sequence optimizer batches; retain auxiliary CPU collation. |
| [`trainer.py`](../../src/diffusion_arc/training/trainer.py) | Execute one complete update from supplied sequences, clip and clear gradients, and return aggregated step metrics. |
| [`distribution_strategy/`](../../src/diffusion_arc/training/distribution_strategy/) | Provide `SingleDeviceStrategy`, `DdpStrategy` and `FsdpStrategy` for model calls, microbatch coordination, gradient synchronization, loss scaling and gradient cleanup settings. |
| [`runner.py`](../../src/diffusion_arc/training/runner.py) | Attach optional logging, validation and checkpoint services to the shared training loop. |
| [`benchmark_runner.py`](../../src/diffusion_arc/training/benchmark_runner.py) | Attach timing and profiling hooks without changing training settings. |
| [`training_loop.py`](../../src/diffusion_arc/training/training_loop.py) | Build models, data and optimizer; execute the common update loop and resume cursor. |
| [`distributed.py`](../../src/diffusion_arc/training/distributed.py) | Own rank/device context, process-group lifetime and coordinated primary-rank services. |
| [`distributed_data.py`](../../src/diffusion_arc/training/distributed_data.py) | Plan global batches and assign complete sources to ranks, including resume cursors. |
| [`profiling.py`](../../src/diffusion_arc/training/profiling.py) | Record explicit benchmark measurements and optional profiler ranges. |
| [`step/encoding.py`](../../src/diffusion_arc/training/step/encoding.py) | Encode source sequences and target grids through frozen NVARC and VAE models. |
| [`step/flux_microbatch.py`](../../src/diffusion_arc/training/step/flux_microbatch.py) | Build physical microbatches, execute FLUX, and apply the selected loss. |
| [`step/flux_batching.py`](../../src/diffusion_arc/training/step/flux_batching.py) | Group one optimizer window by packed token budget. |
| [`step/flow_sampling.py`](../../src/diffusion_arc/training/step/flow_sampling.py) | Draw noise/times and construct straight-path supervision, including endpoint quotas. |
| [`step/loss.py`](../../src/diffusion_arc/training/step/loss.py) | Define plain MSE and equal-target canvas/grid/boundary reductions. |
| [`step/lr_schedule.py`](../../src/diffusion_arc/training/step/lr_schedule.py) | Construct linear-warmup constant or cosine learning-rate schedules. |
| [`run/logging.py`](../../src/diffusion_arc/training/run/logging.py) | Reduce measurements, render prediction tables, collect selected hardware counters and own the W&B run. |
| [`run/validation.py`](../../src/diffusion_arc/training/run/validation.py) | Verify the held-out bundle, prepare fixed CPU caches, evaluate loss and score generated predictions. |
| [`run/checkpointing.py`](../../src/diffusion_arc/training/run/checkpointing.py) | Save HF transformer weights and training state, verify checkpoint contracts, restore RNG/optimizer state and apply retention. |
| [`run/experiment.py`](../../src/diffusion_arc/training/run/experiment.py) | Attach bounded benchmark, memory and profiler measurements to the shared runner. |

Trainer phases use `training_range(...)` for Torch Profiler and Nsight. There is
no per-microbatch stage-log callback. CUDA memory history still captures allocator
events and saves snapshots on failure or run completion; selected first-backward
snapshots are no longer supported.

Import components from their owning modules. Loading `training.config` requires
neither PyTorch nor model initialization. The outer caller selects and constructs
models, optimizer, sampler, loss and scheduler. `Trainer` accepts those components
and execution limits directly, without a run config or logging callbacks. Its
only public execution method is `step`; it does not own epochs or stop criteria.
Logging and checkpoints serialize the recipe through `TrainingConfig.recipe_dict()`.
Training and validation share `plan_nvarc_batches` from `step.encoding`.

For controlled distributed steps, pass `DDP(PackedFluxForward(...))` to
`Trainer` as `flux_transformer`, with `forward_sync_buffers=False` and
`static_graph=False`. The adapter and trainer must share the same compute dtype
and backend. Keep the original transformer for inference and checkpoint access.
The caller initializes the process group, assigns complete sources and supplies
a flow sampler with globally consistent draws and endpoint quotas. Every rank
must enter the same updates; empty ranks are rejected collectively.

The trainer counts global targets before encoding, then uses `no_sync()` around
all but the final local forward/backward. Pack losses are weighted by
`world_size * pack_targets / global_targets`. After gradient synchronization,
loss-finiteness is checked collectively before clipping and updating. This path
uses standard DDP averaging without custom communication hooks or `join()`.
Returned losses, targets, timings and workload counters describe the local rank;
the outer caller owns aggregation. Gradient norms describe synchronized gradients.

### Bounded training benchmark

[`scripts/benchmark_training.py`](../../scripts/benchmark_training.py) starts a
benchmark with the recipe's configured execution strategy. For a DDP recipe,
run it with `torchrun`; `--max-steps` defaults to three global updates:

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_training.py \
  --config configs/training/arc_agi_1_epoch.json --max-steps 3
```

For a one-GPU DDP recipe, use `--nproc-per-node=1` and
`deployment.gpu_count=1`. `TrainingRuntime` selects `cuda:LOCAL_RANK`, initializes
NCCL and destroys the process group on exit; the DDP execution strategy wraps
`PackedFluxForward`. Existing Hopper/FA3 requirements apply.
`scripts/train_ddp.py` remains a compatibility alias for this benchmark entrypoint.

[`distributed_data.py`](../../src/diffusion_arc/training/distributed_data.py)
plans global target batches from the ordered dataset keys and divides whole
sources between ranks. Each rank loads only its assigned sequences. The global
batch size, source order and final partial batch are preserved; sources are
never repeated or dropped to balance ranks. A batch with fewer sources than
ranks is rejected during preflight, including an undersized epoch tail.
`RankBatchPlan.epoch_batch_index` is zero-based and retains the original batch
identity during diagnostic replay. The one-based `diagnostic_batch_number`
selects that batch; completed optimizer updates are counted separately.

[`step/indexed_flow.py`](../../src/diffusion_arc/training/step/indexed_flow.py)
assigns times and exact endpoint/high-noise quotas globally, then generates only local
noise tensors. Seeds depend on the global step and target position, so target
draws agree across rank layouts on the same device type and package versions.
Native execution can select the same indexed draws; historical recipes retain
their stateful RNG stream. The step limit preserves the recipe's full-epoch
learning-rate schedule and endpoint policy.
`FlowTimeQuotaPolicy` defines the endpoint and high-noise quotas shared by
`QuotaFlowSampler` and `IndexedFlowSampler`; the latter derives draws from explicit
batch and target identities independently of the distribution strategy.

[`run_training_benchmark`](../../src/diffusion_arc/training/benchmark_runner.py)
attaches `BenchmarkInstrumentation`, which emits JSON on rank zero: input recipe,
effective training settings, runtime/source identity, and target-weighted global
loss with per-rank counts, times and peak allocation
and reservation. Rank and CUDA synchronization precede each timed step; CUDA
completion precedes its end timestamp. The maximum rank duration includes data
retrieval and `Trainer.step`, including its communication, but excludes the outer
barrier, diagnostic-record collective and output. CPU prefetch starts before the
loop, as in the single-GPU benchmark.
The benchmark entrypoint attaches bounded measurement callbacks to the common
training loop. It explicitly omits validation, sampling, W&B and checkpoint saves, and
rejects resume requests. Use `scripts/train.py` for the complete DDP lifecycle.

### Prepared smoke and benchmark runs

[`benchmarks/ddp_training.json`](../../configs/training/benchmarks/ddp_training.json)
combines the 960 prepared B3 sources with the current epoch recipe. It inherits
the regional loss, four global endpoint targets per full batch, gradient clipping
and the 50-update learning-rate warmup followed by constant LR. Only the data
view changes; the benchmark does not alter the training schedule. The input
recipe's full-epoch counts remain provenance, while the benchmark config requires
exactly 60 full batches from its selected view.

`--mode smoke` runs the first three updates. `--mode benchmark` runs ten training
updates excluded from statistics, followed by 50 measured updates. Both modes
start fresh from the base checkpoint and share their initial data/noise sequence.
Each command is one run; repetitions are not launched automatically. Start with
the two smoke runs, inspect their counts and finite global losses, then run the
benchmarks. One GPU uses the same DDP path with `world_size=1`.

```bash
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 1 --mode smoke
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 2 --mode smoke
```

After successful smoke runs, on the same GPU type and software environment:

```bash
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 1 --mode benchmark
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 2 --mode benchmark
```

The Modal launcher uploads the prepared local B3 bundle, reuses the pinned FA3
image and mounts `diffusion-arc-models` for cached checkpoints and results. It
requests H200 GPUs with eight CPU cores and 64 GiB host RAM per GPU, sets
`OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8`, and starts `torchrun` inside one remote
function. Each training process explicitly sets eight PyTorch CPU threads. Function
timeouts are ten minutes for smoke and thirty for benchmark, with no retries.
DataLoader workers remain per rank: two on one GPU, four in total on two GPUs.
Each run creates `results/ddp/<run-id>-<mode>-<gpus>gpu/` on the volume, containing
`launch.json` and a `training/` directory. Data and model downloads and profiling
remain separate. Rank zero writes `manifest.json`, flushed `steps.jsonl` and `summary.json` on
success. Python failures retain completed rows and write `failure.json`; abrupt
process termination may leave only the partial files. No summary means the run
did not finish successfully.

The summary excludes warmup from latency, throughput and memory statistics.
It reports median and linearly interpolated p95 of maximum rank step duration,
total measured targets divided by total measured time, and maximum per-GPU
allocated/reserved memory across measured steps. The rows preserve source keys,
prefix lengths, global target positions, learning rates, losses and endpoint
counts for matched comparisons. The manifest records effective settings, data
fingerprint, source-content hash, package versions and the FA3 binding. This is
a new reference under the current recipe, not an exact repeat of historical B3.

### Numerical comparison

`--mode numerics --gpus 2` runs three updates without a DDP wrapper, then three
fresh updates with two DDP ranks, using the same current Trainer and first 48
benchmark sources. Both paths use the real NVARC/VAE preparation and independently
pack their local FLUX targets. Initial model states, actual inputs, noise and
timesteps must match. The checks are fixed before execution in
[`ddp_numerics.json`](../../configs/training/benchmarks/ddp_numerics.json): at most
0.1% relative global-loss difference in each step and 5% full-gradient relative
L2 difference in step one, measured before clipping. These are experiment
criteria, not universal floating-point correctness bounds.

```bash
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 2 --mode numerics
```

The single allocation is limited to two H200s and 30 minutes total, with no
retries. CPU gradient chunks use temporary local scratch, never the mounted
volume. The job also checks identical DDP gradient/parameter fingerprints and
unchanged frozen modules. `training/{reference,ddp}/` retains manifests and step
records; `training/summary.json` reports `passed` or `failed`, including per-block
gradient diagnostics. A completed job can fail its numerical criteria. This
instrumented run is not a throughput benchmark or a long-run convergence test.

## Further reading

| Document | Details |
| --- | --- |
| [`../data/data-pipeline.md`](../data/data-pipeline.md) | ARC loading, task splits, augmentation, prompts, tokenization, and target canvases. |
| [`../data/training-data-assembly.md`](../data/training-data-assembly.md) | The auxiliary `TrainingCollator` and `PackedTrainingBatch` contract. |
| [`nvarc-batch-runner.md`](nvarc-batch-runner.md) | Frozen NVARC packing, causal boundaries, early exit, and checkpoint evidence. |
| [`flux-text-conditioning.md`](flux-text-conditioning.md) | Training prefix ownership and the separate dense inference interface. |
| [`flux-image-encoding.md`](flux-image-encoding.md) | Frozen VAE target encoding, latent packing, and normalization. |
| [`flow-matching.md`](flow-matching.md) | Flow sampling, velocity supervision, loss computation, and validation. |
| [`batching-and-efficiency-roadmap.md`](batching-and-efficiency-roadmap.md) | Superseded design note for the padded pipeline and proposed lookahead schedulers. |
| [`flux-sequence-packing.md`](flux-sequence-packing.md) | Production FA3 packing, FP32 modulation sums, token budgeting, and validation limits. |

## Validation

Run the local unit-test suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B -m unittest discover -s tests
```

The local tests cover tensor contracts, hidden-state ownership, loss weighting,
step limits, and tail handling. Real checkpoints and backend-specific behavior
remain explicit smoke or preflight tests.

[`test_ddp_trainer.py`](../../tests/training/test_ddp_trainer.py) first compares
the single-process trainer with an independent global-loss reference, then runs
two CPU/Gloo ranks. It uses a tiny FP32 FLUX transformer, explicit CPU attention,
frozen encoder doubles and prescribed packs with fixed global flow draws.
Three updates cover unequal target/pack counts with plain and regional losses,
comparing gradients, parameters, clipping, AdamW and scheduler state. Separate
checks exercise coordinated empty-rank and non-finite failures. These tests do
not measure GPU precision, memory, communication cost or throughput.

[`test_ddp_runner.py`](../../tests/training/test_ddp_runner.py) compares a plain
Trainer reference with the new runner on one and two CPU/Gloo ranks. The real
batch planner, indexed flow sampler, packing, gradient averaging and metric
aggregation execute over three updates, including a partial batch with unequal
rank target counts. Model loading and the CUDA runtime are replaced by small
local fixtures. Separate tests cover data ownership, endpoint quotas, rank
configuration, cleanup and CLI validation. Benchmark tests additionally check
the fixed-view override, measured/warmup phase boundaries, slowest-rank timing,
numerical agreement and warmup exclusion from summaries.

[`test_unified_runner.py`](../../tests/training/test_unified_runner.py) exercises
the shared full-run loop with a real tiny FLUX model, AdamW and on-disk HF
checkpoints. It compares native execution with two CPU/Gloo ranks, including
weights-only initialization, a partial batch, high-noise quotas, primary-rank
service ownership and next-epoch continuation. Interrupting after a saved update
and resuming reproduces the uninterrupted DDP model, optimizer and RNG states
exactly. Frozen encoders and external logging/validation services use local test
doubles; full-model GPU behavior is outside this test.

## Current boundaries

- Single-GPU and DDP share training, validation, sampling, W&B, full-state resume
  and epoch continuation. Local CPU/Gloo tests cover the integrated lifecycle;
  a full-model GPU smoke run of that lifecycle remains pending. The separate
  [DDP report](../performance/ddp.md) records prior bounded GPU/NCCL/FA3 and
  profiling experiments. FSDP2 and activation checkpointing remain pending.
- All training, benchmark and profiling workloads use schema v2 with
  `config_type="training_run"`. Loss, timestep sampling, scheduler, execution
  strategy and packing are explicit choices. `validation`, `logging`,
  `checkpointing`, `scheduler` and `benchmark` accept explicit `null` when
  disabled. `loop.epochs: 1` selects one complete prepared augmentation epoch;
  `loop.max_steps` selects a bounded prefix, with the other field set to `null`.
  Further epochs require separately prepared augmentations and continuation.
  Derived target and step counts are not user inputs. Historical saved epoch
  recipes have an explicit checkpoint migration; new input files are never
  silently completed or migrated.
- Training always uses FLUX packing with an explicit token budget. Padded
  training, fixed FLUX microbatch sizes, and measured padding-cost catalogs
  have been removed; retired fields are rejected even when set to null.
- The [B3 benchmark](../../configs/training/benchmarks/single_gpu_b3.json),
  [three-step smoke](../../configs/training/benchmarks/single_gpu_b3_smoke.json),
  and [profiler](../../configs/training/profiling/single_gpu_b3_torch_profile.json)
  all combine NVARC and FLUX packing.
- The shared Modal image requires Hopper/SM90 and BF16 FA3, with head dimension
  128 and both forward and backward support.
- The [B3 benchmark](../performance/b3.md#benchmark-result) measures combined
  throughput, peak memory, and step-loss differences from B2 on H200.
  Gradient/update equivalence and long-term learning quality remain open.
