# Attention execution diagnostic plan

This report preserves the diagnostic state and evidence before B3 integration.
Its code, configurations and any commands below belong to the
[archived implementation](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
they are not current entrypoints. Later production choices are summarized in
[B3](README.md#b3-packed-nvarc-and-flux).

**Status: first wave complete, 2026-09-09.** Local structural tests and the
first bounded H200 diagnostic are implemented and validated. The fixed-Q/K/V
tests support segmentation; operation replays isolate shape-sensitive linear
outputs and BF16 modulation-gradient reduction differences. Later training,
native cuDNN ragged and performance waves remain proposals.
See [first-wave execution](attention-execution.md) for evidence and limits.

The [budget-limited backward layout diagnostic](flux-backward-layouts.md) has
executed eight layouts, including AB and BA for both packed paths, each twice.
Its numerical matrix completed; requested CUDA trace evidence remained
incomplete. Its 20-minute lifecycle and $2 planned budget superseded the larger
proposed second-wave ceiling below for that run. Optimizer updates remain deferred.

The decision is which supported attention backend and execution layout provide
the best complete training throughput with validated sample isolation and
acceptable learning behavior. FA3 has no preferred status. Backend selection,
attention packing, and full-transformer packing are separate experimental axes.

## Existing evidence and open questions

- [NVARC packing](nvarc-packing.md) retains the measured conditioning values
  exactly with cuDNN ragged; FA2/FA3/FA4 differ by approximately 0.47–0.57%.
  cuDNN ragged was only 0.9–1.7% slower than FA3 in three complete windows.
  That makes both serious end-to-end candidates for frozen NVARC.
- [FLUX packing](flux-packing.md) reproduced A+B and B+A exactly after restoring
  sample order. Perturbing one sample left the other unchanged. Relative final
  velocity differences from FA3 singletons remained 0.009861% and 0.547451%.
  The first observed differences were outside attention. Full parameter-gradient
  differences of 18–20% from the earlier mixed-layout comparisons remain open.
- [Input-condition comparisons](flux-backend-conditions.md) tested native Qwen,
  base FLUX with ARC, and overfit FLUX with familiar/new ARC examples. They do
  not need another full sweep. Small kernel-output differences did not imply
  equally small dQ differences or full-model training equivalence.
- The [Thinking Machines explanation](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  distinguishes repeatability from batch invariance. It supplies a hypothesis
  for shape-dependent arithmetic, not proof of the cause in this model.

## Common contracts

Prepare frozen NVARC and VAE outputs once through one pinned, recorded path
(initially cuDNN SDPA and FP32 VAE), and reuse exact tensors across FLUX cases.
Record token/source identities, positions, timesteps, noise, targets, checkpoint
hashes, tensor hashes, package versions, backend flags and actual dispatch.
Use only existing cached checkpoints and approved training-source fixtures.

Keep FLUX parameters and optimizer state FP32, BF16 compute, native Q/K
normalization and RoPE, noncausal attention, and zero dropout. Kernel replays
use identical post-RoPE BF16 Q/K/V. A small FP32 Math oracle promotes these
values without regenerating projections or conditioning. FP32 is a diagnostic
reference, not a proposed replacement training precision.

Start with the historical short A/B pair to reproduce the unresolved result.
Add the existing medium and long pairs for kernel/shape checks. Use the
step-4096 checkpoint with seen/unseen ARC views as a training-relevance control.
Use synthetic labelled tensors only for structural and boundary tests; label
them separately from real-data measurements. Do not take a Cartesian product
of every checkpoint, prompt, length, layout and precision.

## Backend and layout matrix

| Execution path | FA2 / FA3 / FA4 | cuDNN |
| --- | --- | --- |
| Original FLUX forward, independent singletons | Same backend through an attention-only adapter | Forced PyTorch SDPA |
| Original batch layout, attention alone packed | Candidate diagnostic adapter | Native padded SDPA baseline; native ragged is a separate preflight |
| Full-transformer packing | Existing diagnostic processors | Existing per-segment SDPA control; native ragged is a separate preflight |

The independent reference must not call `packing_layout`, `packed_velocity`,
or the packed processors under test. It uses the original Diffusers forward
and constructs singleton boundaries directly as `[0, sequence_length]`.

The intermediate attention-only architecture keeps projections, MLPs and
timestep modulation in the original batch layout. It gathers actual valid
token positions inside attention and scatters outputs back. With the layout
`valid text → text padding → image`, slicing the first `valid_len` tokens is
incorrect. Either gather valid Q as well and restore unused positions, or
retain every Q and use separate Q and K/V cumulative boundaries; validate the
chosen contract explicitly, including backward and padding isolation.

The current NVARC cuDNN-ragged adapter is causal, forward-only and has no
autograd binding. It is not yet a FLUX training backend. A separate bounded
preflight must establish noncausal BF16 forward/backward with 24 heads and
head dimension 128, including saved statistics, segment isolation and actual
dispatch. A segmented Python loop is a correctness oracle, not a substitute
for native-ragged performance. Unsupported paths are reported as unsupported;
other candidates continue without silently changing backend.

## Stage 0 — Exact packing contracts

Extend the existing small tests with an independent labelled-token oracle:
every real token appears exactly once; pack/unpack restores its identity;
positions, masks, timestep ownership and output ownership match the source.
Include asymmetric lengths, several segments, reversed/permuted orders,
boundary lengths around 128/256, and the maximum 8,400-text-token contract.
Length limits apply to text separately from the 900 image tokens.

Test the backward of gathers/scatters with known analytical gradients.
Unique token permutations must invert exactly. Repeated ownership indices
used for timestep modulation require a sum, not a permutation; test those
separately with hand-checkable values. Padding must contribute neither valid
outputs nor gradients to real inputs. Malformed metadata must fail preflight.

Deliverable: structural failures are identified before expensive model runs.

## Stage 1 — Fixed-input kernel and operation replays

For each supported backend, compare A, B, A+B, and B+A through the same API,
using identical Q/K/V and sample-associated upstream gradients. Include three
upstream seeds and unchanged repetitions. Perturb A and B independently.
With loss/upstream only on A, gradients of B's independent inputs must be zero.
Do not confuse these input gradients with gradients of shared model weights.

Capture real Q/K/V at `double.0`, `double.4`, `single.0`, and `single.19`.
Compare outputs and dQ/dK/dV both within each backend and directly between
backends, with FP32 Math as an additional numerical reference. Direct pairwise
comparisons avoid interpreting similar distances from Math as mutual equality.
Report per-example/per-block absolute errors, relative L2, reference norms,
cosine, maxima and repeatability. Use deterministic backward diagnostically
where supported; also measure the intended production mode. Determinism flags
do not establish batch invariance.

In the same stage, replay the existing first-difference boundaries with frozen
operands: A's final output projection and B's first-double-block operations
after attention. Narrow B's boundary to gating/residual, normalization and MLP
as needed. Vary execution shape without changing logical inputs or weights.
Replay timestep-modulation expansion/reduction with an identical upstream to
test the suspected repeated-index accumulation independently of attention.
Preserve production arithmetic first; a precision or reduction-strategy change
is a labelled causal diagnostic, not an automatic production correction.

Deliverable: separate kernel segmentation, surrounding shape-sensitive
arithmetic, and modulation-gradient accumulation. A remaining discrepancy is
recorded with its earliest observed boundary, not dismissed as generic GPU noise.

## Stage 2 — Complete backward and one optimizer update

Use the historical short pair first, then the overfit seen/unseen control.
For each backend, compare the independent original-forward reference with
attention-only packing and full-model packing. Preserve sample-specific
timesteps, positions and random draws. Compare A+B and B+A by sample identity.

Accumulate singleton gradients for the same logical loss as the joint batch:
for equal example weights, `L = (L_A + L_B) / 2`. Preserve the production
microbatch/global-batch weighting when using the 48-target optimizer batch.
Never update parameters between A and B. Clip gradients once, after full
accumulation, and advance the optimizer/scheduler once per logical batch.

Every branch starts from identical weights, AdamW moments, optimizer step,
scheduler and RNG state. Prefer an existing compatible optimizer snapshot;
otherwise construct and label a shared nonzero-moment state, with a fresh-state
check separately. A transformer-only overfit checkpoint does not imply that
its optimizer state is available. Avoid judging AdamW solely from its first
zero-moment update, which can obscure gradient-magnitude differences.

Compare velocity, loss, complete gradient coverage, per-component gradients,
pre/post-clipping norms, updated moments, and `delta = theta_after - theta_before`.
Measure update error relative to the reference update, not the full parameter
norm. Compute differences of the actual FP32 parameters in FP64 for reporting.
Keep one model resident and restore branch state explicitly to bound memory.

Establish correctness without checkpointing, then repeat the representative
training comparison with gradient checkpointing enabled. The current packed
diagnostic rejects checkpointing; supporting it requires retaining the selected
processors and immutable metadata through backward recomputation. Removing
the guard alone is not an implementation.

Deliverable: evidence for the same-backend training effect of each layout.
The original 18–20% gradient discrepancy must be reproduced/localized or
explicitly remain unresolved; it is not waived by smaller forward errors.

## Decision rules

- Wrong ownership, cross-example dependence, missing gradients, mutated shared
  inputs, unexpected dispatch or non-finite values invalidate a case. Preserve
  partial evidence and continue only independent safe cases.
- Exact same-shape checks use the recorded repeatability contract. If production
  backward is nondeterministic, repeat it and isolate the case in a supported
  deterministic mode before claiming an exact invariance violation. A
  disconnected sample's gradient must still be zero.
- Report execution completion separately from numerical screening and training
  qualification. Existing full-model screens (2% velocity, 5% gradient, 1% loss)
  remain investigation triggers, not demonstrated acceptance tolerances. The
  training wave also records update errors without inventing a post-hoc pass limit.
- Freeze screening configuration before each run. Repeated-run variation is
  useful context but does not explain away systematic layout bias. Do not raise
  limits merely to accommodate an observed failure.
- A numerical flag can motivate a diagnostic timing measurement; it cannot
  silently qualify that path for training. Quantitative learning-regression and
  update criteria must be set before the subsequent confirmation runs.

## Stage 3 — Performance and learning qualification

Benchmark supported candidates with the same logical optimizer batches and
the existing short/medium/long workload groups. Include assembly, casts,
forward, backward, clipping and optimizer work. Report examples per second,
median/p95 complete-step time, peak allocated/reserved memory, and first-use
cost separately. Use warmups, paired interleaved timing rounds and raw samples;
profile outside timed rounds. Do not rank candidates by a single fastest call
or claim a meaningful 1% win from overlapping small-sample timings.

Measure attention-only packing as a genuine alternative to full-model packing.
If the latter adds little useful throughput, keeping native modulation and
MLP layouts may remove complexity and earlier workarounds.

Take the fastest eligible candidates into short paired learning runs with the
same initialization and sample/noise schedule, initially three seeds. Compare
loss progress, time to a fixed target, and decoded-grid behavior on declared
training/proxy views. Predeclare steps, margins and stopping rules using the
measured step cost; this is not a held-out ARC benchmark or proof of final
training quality.

Finally test the best FLUX paths end to end with both NVARC cuDNN-ragged and
FA3 candidates. First hold FLUX inputs fixed for timing attribution, then use
each encoder's own outputs to measure its actual effect. Do not rerun all
NVARC backends and all FLUX layouts as an exhaustive cross product.

## Implementation boundaries and proposed execution waves

Reuse `_flux_packing.py`, its correctness/forward helpers, the fixed-input
backend replay, existing real-input fixtures, and checkpoint/source manifests.
Add diagnostic adapters and a single staged coordinator/configuration rather
than another copied transformer. Keep new code under `scripts/diagnostics/`,
its launcher under `scripts/cloud/`, configuration under `configs/diagnostics/`,
and focused tests under `tests/diagnostics/`. Do not change `src/`, README
scope, training defaults, model weights, or the selected production backend
while implementing the diagnostic proposal.

Each stage needs an offline `--plan` that lists exact cases, dependencies,
expected comparison counts, supported paths and time limits. Upload code and
configuration hashes, save incremental results, and validate completeness on
return. No silent fallback, automatic retry, extra GPUs, or model downloads.

| Proposed wave | Scope | Worker limit | Function limit |
| --- | --- | ---: | ---: |
| Local | Stage 0 and orchestration/reference tests | Local only | No GPU |
| First H200 run | Stage 1: supported kernel and operation replays | 15 min | 17 min |
| Second H200 run | Stage 2: bounded full-gradient/update matrix | 30 min | 32 min |
| Separate cuDNN preflight | Native-ragged FLUX training support | 10 min | 12 min |

Later-wave budgets remain proposed ceilings, not submitted jobs. The completed
first wave and its actual runtimes are recorded in the linked report. Run one
H200 at a time, with zero automatic retries and incremental artifact commits.
The second wave is selected from first-wave findings; the cuDNN preflight does
not block other paths. Timing and learning budgets are fixed separately after
step cost and eligible layouts are known. A timeout is an incomplete experiment,
never successful qualification. Local checks and the first H200 diagnostic
wave are complete; full-model backward/update qualification remains next.
