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

For **B6**, select
[`benchmarks/b6_fsdp2_16k.json`](../../configs/training/benchmarks/b6_fsdp2_16k.json)
with `--config`. It uses two-GPU FSDP2 and sets
`execution.flux_max_joint_tokens=16384`. This optional benchmark field overrides
the recipe's pack capacity; the effective value is validated by the training
config and recorded in the run manifest. When omitted, the recipe's limit is
retained, including 12,288 tokens for the B5 FSDP2 baseline. The global optimizer
batch remains 48 targets.

For **B7**, use the matched 14,336-token pair:
[`no forward resharding`](../../configs/training/benchmarks/b7_fsdp2_14k_no_forward_reshard.json)
and its [`forward resharding control`](../../configs/training/benchmarks/b7_fsdp2_14k_forward_reshard.json).
They differ only in `execution.fsdp_reshard_after_forward` (`false` versus `true`)
and both use three repetitions with ten warmup and 50 measured updates.
With `false`, FLUX blocks retain gathered parameters after forward, avoiding
the backward all-gather. Both variants reshard after backward and synchronize
gradients in the final scheduled microbatch; the packed root always retains parameters
after forward. The optional boolean is accepted in training recipes and
benchmark overrides only for `distributed_strategy=fsdp2`. Omission retains the
recipe setting, or the original block policy (`true`) for older recipes. The
effective override is recorded in the run recipe and used by the numerical
diagnostic. Capacity probes select their policy separately through their CLI.
Run `--mode smoke` for the complete pipeline before the benchmark; the synthetic
capacity result alone does not establish its memory limit.

FSDP2 normally accumulates full local gradients until the final scheduled microbatch
of each optimizer update, including a padded dummy call, then reduces before
clipping and AdamW. Training, benchmarks, numerical diagnostics and the capacity
probe default to this policy. A bounded benchmark can instead set the top-level
`fsdp_gradient_sync="per_microbatch"` to reduce after every scheduled call and
accumulate gradient shards. This is not a training-recipe option. The benchmark
manifest records the policy; numerical and capacity diagnostics reject this
override rather than silently ignoring it. Parameter
all-gathers follow the resharding policy above. Retaining full gradients can raise
peak memory; earlier capacity measurements with reduction after every microbatch
do not establish the current limit.

The optional `execution.fsdp_blocks_per_group` groups consecutive FLUX blocks
into one FSDP collective group in their existing forward order. It is a positive
integer accepted only with `fsdp2`; omission keeps one block per group. A final
partial group is retained, and the packed root remains separate. B14 uses
[`five-block groups`](../../configs/training/benchmarks/b14_fsdp2_14k_grouped_blocks.json)
with B13's 14,336-token budget and token-balanced assignment. Grouping changes
communication granularity without changing parameter names or block execution;
larger gathered buffers can increase peak memory. The effective grouping is
recorded in the recipe for benchmarks, profiling and numerical diagnostics.

The optional `execution.rank_batch_assignment=token_balanced_v1` applies to both
DDP and FSDP2. It retains the ordered global optimizer batches and assigns whole
sources within each batch by descending `sum(eligible_prefix_tokens + 900)` to
the rank with the least assigned load. Local source order and original global
target IDs are retained, so indexed noise and timesteps follow the same targets.
It permits unequal source counts per rank. Omission retains `contiguous_v1`.

This mode requires explicitly prepared `<view_name>.token_lengths.json` in the
selected split directory. Run `scripts/data/prepare_training_token_lengths.py
--config <config>` with the cached NVARC tokenizer, or add `--steps-jsonl <path>`
to reuse a complete matching benchmark. GPU startup only reads this small
metadata file; consumed samples check their actual prefixes against it.
New epoch keys require new metadata. The metadata hash is recorded in benchmark,
numerical and checkpoint identities. B10's
[`token-balanced configuration`](../../configs/training/benchmarks/b10_fsdp2_12k_token_balanced.json)
matches B9's 12,288-token, no-forward-resharding settings with one repetition;
only source assignment changes. CPU planning estimates do not establish GPU throughput.

The synthetic [FLUX capacity launcher](../../scripts/cloud/probe_flux_capacity_modal.py)
accepts `--gpus 2`, `4` or `8`, matching the selected FSDP configuration and
24 training targets per GPU. The
[four-GPU](../../configs/training/benchmarks/fsdp2_4gpu_12k_background_packing.json)
and [eight-GPU](../../configs/training/benchmarks/fsdp2_8gpu_12k_background_packing.json) configurations
retain gathered weights after forward and reduce gradients at batch end.
Use `--no-reshard-after-forward` for that capacity policy. Each candidate runs
in fresh processes for at least two AdamW updates with frozen model weights and
eight maximal source-state buffers resident per rank. This synthetic workload
uses two full stress packs per update; it does not execute the encoders or the
background batch planner. It establishes a tested pack bound to confirm in the
full pipeline, not a universal training capacity guarantee. Candidate reports
include CUDA allocator retries as well as peak memory and hard OOMs.

The capacity launcher defaults to the new `diffusion-arc-training-v2` volume
mounted at `/workspace`, with models under `models/`. Existing
`DIFFUSION_ARC_MODAL_*` overrides remain accepted. `--time-limit-seconds 600`
limits the complete search to ten minutes, with another 90 seconds reserved
for cleanup and persistence. Results remain in `results/flux-capacity/` on
that volume, including completed candidates when the time limit is reached.

Set `execution.rank_batch_assignment=joint_packing_v1` to plan source ownership
and aligned FLUX microbatches during regular training or a benchmark. It uses
the same bounded joint planner as the precomputed experiments. Each training
rank owns one spawned CPU worker, with at most four queued searches ahead of
the next requested batch. All ranks deterministically derive the same global
plan from prepared lengths and the runtime world size; sources remain whole,
and global batches, target IDs, noise and timestep identities are preserved.

The DataLoader requests planned source indices before loading their samples.
Its source prefetch and the training loop share cached plans; completed plans
are released after training consumes them. Resume starts at the original next
batch ID and regenerates its plans without loading completed sources. Worker
failures propagate, and workers close on normal completion or error. The mode
requires indexed flow RNG, `legacy_bfd_v1`, `shuffle=false`, and the same prepared
token metadata as token balancing. Do not combine it with `precompute_packing`.

The background planner supports the actual configured world size, provided each
optimizer batch, including the epoch tail, has at least one source per rank.
The Modal benchmark launcher accepts 1, 2, 4 or 8 GPUs, matching
`execution.gpu_count` in the benchmark config; numerical diagnostics still use
exactly two GPUs and reject background packing. Modal supports up to eight
H200s in one container ([GPU documentation](https://modal.com/docs/guide/gpu)).
Larger optimizer batches must be selected explicitly in the training recipe or
the benchmark wrapper's `loop.optimizer_batch_size`. An optional wrapper
`flow.endpoint_targets_per_full_batch` keeps the endpoint fraction constant.

The [four-GPU](../../configs/training/benchmarks/ddp_4gpu_12k_background_packing.json)
and [eight-GPU](../../configs/training/benchmarks/ddp_8gpu_12k_background_packing.json)
DDP configurations use background joint packing and a 12,288-token FLUX budget.
They select 96 and 192 global targets respectively, retaining 24 targets per GPU
on average and the original endpoint fraction. Each runs one repetition with
ten warmup updates and 25 measured updates. Their separate prepared views contain
1,120 and 2,240 sources from the same seeded training-only selection; existing
benchmark bundles remain unchanged. The launcher uploads these explicit views
when present, including their prepared token lengths.

The selected [four-GPU FSDP2 benchmark](../../configs/training/benchmarks/fsdp2_4gpu_14k_background_packing.json)
uses the same 96-target batch, prepared view and 10/25-update measurement window,
with a 14,336-token pack budget, forward weight retention, batch-end gradient
reduction and one block per FSDP group. Background joint packing remains enabled.
The [capacity result](../performance/fsdp2.md#four-gpu-pack-budget-selection)
selected this budget; the complete benchmark finished all 35 updates and
measured a 4.762 s median, versus DDP's 4.680 s.

The selected [eight-GPU FSDP2 benchmark](../../configs/training/benchmarks/fsdp2_8gpu_16k_background_packing.json)
uses 16,384 tokens with the same FSDP settings and measurement window, 192 global
targets and the prepared eight-GPU view. The capacity probe completed two updates
at this budget without allocator retries on all eight ranks; 18,432 tokens failed
on the second update. The complete benchmark finished all 35 updates and
measured a 4.821 s median, versus DDP's 4.677 s. The probe artifacts are stored
under `artifacts/flux-capacity/20261013T155732028769Z/`.

The [complete four/eight-GPU comparison](../performance/fsdp2.md#four--and-eight-gpu-complete-pipeline-results)
selects DDP with background joint packing for this workload. Larger FSDP packs
reduced FLUX calls but did not improve throughput. Retain FSDP as a capacity
option when an actual memory constraint requires it; the bounded runs do not
establish long-run numerical equivalence.

For the `joshuakauling` profile's `main` environment, select the existing storage
with `DIFFUSION_ARC_MODAL_VOLUME=diffusion-arc-training-v2`,
`DIFFUSION_ARC_MODAL_VOLUME_ROOT=/workspace` and
`DIFFUSION_ARC_MODAL_MODEL_SUBDIRECTORY=models`. The mount preserves the cached
NVARC symlink's absolute target. Before the first GPU run,
[`prepare_ddp_backend_modal.py`](../../scripts/cloud/prepare_ddp_backend_modal.py)
copies only the missing original FLUX transformer from the old workspace using
a separate CPU allocation. It does not substitute a trained checkpoint.
The launch record stores the selected volume, mount and model directory.

The [12k background-packing benchmark](../../configs/training/benchmarks/fsdp2_12k_background_packing.json)
retains the measured FSDP reference's sharding and optimizer policies, with one
repetition. Its manifest records the planner contract and implementation hash;
step records retain the executed packs. Waiting for a future plan during timed
data retrieval is included in throughput. The first plan and initial DataLoader
prefetch occur before the timed loop and are excluded. CPU tests cover 1/2/4/8
rank plans, worker prefetch, resume, errors and real optimizer updates. The
four- and eight-GPU benchmarks above include overlapping background planning.

`--mode smoke` runs the first three updates. The two-GPU reference configs use
`--mode benchmark` for ten training updates excluded from statistics, followed
by 50 measured updates; other configs select their own window. Both modes
start fresh from the base checkpoint and share their initial data/noise sequence.
The Modal launcher executes the configured repetitions automatically in benchmark
mode (three for the historical reference configs, one for the scaling configs),
each in fresh worker processes.
Smoke mode runs once. Inspect smoke counts and finite global losses before the
benchmark. One GPU uses the same DDP path with `world_size=1`; select a benchmark
config with `execution.gpu_count=1` for that run. The checked-in reference below
uses two GPUs.

```bash
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 2 --mode smoke
```

After successful smoke runs, on the same GPU type and software environment:

```bash
python -m modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --gpus 2 --mode benchmark
```

The Modal launcher uploads the prepared local B3 bundle, reuses the pinned FA3
image and mounts `diffusion-arc-models` for cached checkpoints and results. It
requests H200 GPUs with eight CPU cores and 64 GiB host RAM per GPU, sets
`OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8`, and starts `torchrun` inside one remote
function. Each training process explicitly sets eight PyTorch CPU threads. Function
timeouts are ten minutes for smoke and thirty per benchmark repetition, with no
retries (90 minutes for three benchmark repetitions).
DataLoader workers remain per rank: two on one GPU, four in total on two GPUs.
Each run creates `results/ddp/<run-id>-<mode>-<gpus>gpu/` on the volume, containing
`launch.json` and a `training/` directory for smoke/profile or `repetition-N/`
directories and a combined `summary.json` for benchmarks. Data and model downloads and profiling
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

Selecting the FSDP2 configuration adds three fresh FSDP2 updates after the
single-GPU and DDP phases. All three paths use `legacy_bfd_v1` packing and the
same criteria. The summary checks each pair's losses over three updates and full
gradients before clipping in step one. FSDP gradients are reconstructed one
parameter at a time; parameter fingerprints describe local shards and are not
required to match across FSDP ranks. Parameter updates and AdamW moments are not
directly compared between strategies.

```bash
.venv/bin/modal run --detach --env diffusion-arc scripts/cloud/train_ddp_modal.py \
  --config configs/training/benchmarks/fsdp2_training.json \
  --gpus 2 --mode numerics
```

This uses the same two-H200, 30-minute allocation with no retries and cached
models only. Results persist under `results/fsdp2/<run-id>-numerics-2gpu/` in the
`diffusion-arc-models` volume, with `training/{reference,ddp,fsdp2}/` phase records
and `training/summary.json` containing the three pairwise results.

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
