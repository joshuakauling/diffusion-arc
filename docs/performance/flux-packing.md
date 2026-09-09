# FLUX attention packing diagnostic

**Status: reversing A+B to B+A produced bitwise identical per-example outputs
at every observed boundary on H200. Both repetitions and all four example-
isolation checks passed exactly. Singleton-versus-packed differences remain:
0.009861% final velocity relative L2 for A and 0.547451% for B, independent of
order. The earlier full-model gradient differences remain unresolved. No
complete throughput comparison is available.**
This standalone diagnostic compares the existing padded FLUX
execution with fully packed FA2, FA3 and FA4 execution. It changes no files under
`src/`, training defaults, model weights, or scheduling policy.

Here, "original" refers to the unchanged **Diffusers** forward. Independent
agreement with Black Forest Labs is covered by the separate
[BFL reference diagnostic](flux-bfl-reference.md), whose real-model CUDA
comparison passed every FP32 threshold on two exposures. Native BF16 already
shows internal activation errors up to 14.37% without FA3 or packing, although
final velocity and loss remain within that diagnostic's thresholds. The
packing results below do not establish BFL parity or explain those differences.

## Start

Inspect the configuration and exact groups without loading checkpoints,
allocating a GPU, contacting Modal, or writing artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B \
  scripts/diagnostics/probe_flux_packing.py \
  --config configs/diagnostics/flux_packing.json --dry-run
```

Start the paid diagnostic explicitly:

```bash
.venv/bin/modal run --detach --env diffusion-arc \
  scripts/cloud/probe_flux_packing_modal.py \
  --config configs/diagnostics/flux_packing.json
```

The launcher requests one H200, 8 CPUs and 64 GiB host memory, with no automatic
retries. The complete worker budget is **2,400 seconds (40 minutes)**; Modal's
function limit is **2,700 seconds (45 minutes)**. These are limits, not runtime
estimates. Model loading, preparation, JIT compilation, numerical comparisons
and profiling all count against the worker budget. CPU image building occurs
before the GPU function and outside that budget.

The image builds pinned FA2 2.8.3.post1 and FA3 revision
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` against PyTorch 2.13.0/CUDA 13.0,
and installs FA4 4.0.0b29 with its pinned CuTe dependencies. FA3 explicitly keeps
BF16 backward, Varlen and head dimension 128 enabled. It is **not** the
forward-only NVARC image. The first remote attempt verified the build and small
Q/K/V forward/backward preflights. A complete FA2/FA3/FA4 throughput comparison
is still unavailable.

Existing checkpoints come from the `diffusion-arc-models` Volume. Only the
existing B0 data bundle is uploaded. No models or datasets are downloaded;
image construction installs the declared attention dependencies.

## Focused correctness mode

### Singleton versus packed forward

Preview the forward-only plan offline by adding `--forward-only --dry-run` to
the diagnostic Python command. Launch this separately authorized diagnostic with:

```bash
.venv/bin/modal run --detach --env diffusion-arc \
  scripts/cloud/probe_flux_packing_modal.py \
  --config configs/diagnostics/flux_packing.json --forward-only
```

The dedicated function requests one H200, 8 CPUs and 64 GiB host memory, with
a **600-second function limit**, at most **480 seconds for the complete worker**,
and **no automatic retries**. It reuses the existing image and checkpoints.
Any CPU image building precedes the GPU function and is outside its time limit.
This mode is mutually exclusive with `--correctness`; neither focused mode runs
the throughput sweep.

[`_flux_packing_forward.py`](../../scripts/diagnostics/_flux_packing_forward.py)
reuses the production model loader, real input preparation, and existing packed
processors. Unlike the training-memory setup used by the earlier diagnostic,
it initializes no optimizer, persistent gradients, or training-memory reserve,
and performs no backward calls. Persistent FLUX weights remain FP32; all model
comparisons use `no_grad` and BF16 autocast in the same model mode.

The original short_b4 inputs and flow draws are prepared once. A and B are its
first two targets (window target indices 15 and 12), with valid text lengths
468 and 406. Each retains all 900 image tokens, giving packed boundaries
`[0, 1368, 2674]`. Singletons slice the prepared tensors without re-encoding or
resampling. Input hashes record exact text, latents, timesteps and positions.

The current diagnostic (`experiment_version=2`) executes twelve full forwards
in a fixed order. The initial seven-call run is retained below as a historical
result with its own source hashes.

Version 2 has passed **406 local repository tests** and the bounded H200 run
reported below. Both versions' measured results retain their own source identity.

1. FA3 singleton A.
2. FA3 singleton B.
3. FA3 packed A+B, compared with those singletons.
4. An unchanged FA3 A+B repetition, checked for exact reproducibility.
5. FA3 A+B with only B's text, noisy latents and timestep changed on new storage.
6. FA3 A+B with only A changed.
7. FA3 B+A, compared by example identity with the singletons and A+B.
8. An unchanged FA3 B+A repetition.
9. FA3 B+A with only A changed.
10. FA3 B+A with only B changed.
11. Original model singleton A with SDPA, compared with its singleton and both packing orders.
12. Original model singleton B with SDPA, compared with its singleton and both packing orders.

Each perturbation preserves all shapes, positions and packing metadata:
text and noisy latents become `-x + 0.125`; its timestep becomes 0.83 if the
original is below 0.5, otherwise 0.17. A-only and B-only perturbations are each
constructed once, then reused in both orders. The untouched example's inputs,
captured outputs and final velocity must be bitwise unchanged; the changed
example's velocity must change. Each perturbation is compared with its own
unchanged order. A failed unchanged repetition makes that order's isolation
test inconclusive; it does not prove examples interacted.

B+A preserves every example's text, latents, timestep and coordinates, with
boundaries `[0, 1306, 2674]` and the same maximum joint length 1368 as A+B.
Hashes check identities before model execution; labels `(B,A)` map outputs
back to the original examples before comparisons. The direct B+A versus A+B
comparison uses the same backend, token total and maximum length. It changes
example order and sequence offsets, so an observed order effect alone would
not identify its underlying operation.

Temporary hooks capture both text and image outputs after all five Double-
and twenty Single-Stream blocks, then the final velocity. Canonicalization
uses explicit example lengths/offsets independently of the packing permutation
indices. Candidates are compared as each block finishes; only required CPU
references remain resident. Hooks and processors are restored even on failure.
Operator counts and CUDA kernel names verify FA3 forward execution and record
the actual SDPA backend of each original singleton.

The passive hooks in
[`_flux_packing_forward_boundaries.py`](../../scripts/diagnostics/_flux_packing_forward_boundaries.py)
also capture the first Double block's text/image inputs, its normalized and
modulated Attention inputs, and its Attention outputs. The final adaptive
normalization's image and timestep-embedding inputs and its output are captured
before the existing velocity observation. These five additional stages reuse
the same forwards and help narrow where differences first become visible;
they do not replace projections, processors, kernels, or normalization.

Every block/example/stream retains maximum absolute error, RMS, relative L2,
and exact equality. The existing **2% final velocity relative-L2 screening
limit** is fixed before execution; intermediate block errors are descriptive.
The original singletons independently check model assembly, but their comparison
also changes SDPA versus FA3. The first differing block identifies an observed
output difference, not its causal operation. Even a successful forward test
does not establish gradient or training agreement.

Artifacts are written to `results/flux-packing-forward/<run_id>/` on the Volume:
`manifest.json`, incremental `result.json`, `summary.json`, `report.md`, and
`worker.log`. Numerical failures remain separate from execution failures, and
the worker preserves partial results on errors or timeout.

#### H200 order and isolation result: 20260908T044830217531Z

The twelve-call run completed in **109.201 seconds**, with
`numerical_passed=true`, return code 0, no timeout, no OOM and no automatic
retries. The manifest records the unchanged 480-second worker and 600-second
H200 function limits. Modal app:
[`ap-mjdlkuwyC6j4GCof50Jqrw`](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-mjdlkuwyC6j4GCof50Jqrw).

All input hashes match between orders and also match the initial seven-call
experiment. The earlier cases reproduced every previously recorded block and
velocity comparison metric. All ten FA3 cases executed exactly 25 FA3 forward
operators each, without SDPA fallback or backward. The two original controls
each executed 25 cuDNN SDPA backend calls.

| Actual versus reference | A final velocity relative L2 | B final velocity relative L2 |
| --- | ---: | ---: |
| A+B versus FA3 singletons | 0.009861% | 0.547451% |
| B+A versus FA3 singletons | 0.009861% | 0.547451% |
| B+A versus A+B, restored by example label | 0% | 0% |

Reversing the order preserved **all 60 captured stream values per example
bitwise**, including block outputs, extra component boundaries and final
velocity. B still differed from its singleton when it was first, and A retained
the same much smaller final difference when it was second. For this pair,
placing an example second does not explain the singleton-versus-packed
discrepancy.

The unchanged A+B and B+A repetitions were both bitwise identical everywhere.
All four isolation checks passed: changing A left B bitwise identical, and
changing B left A bitwise identical, in both orders. The changed example's
velocity changed in every case. These findings establish the tested invariances
for these inputs and perturbations, not for every possible batch.

The additional boundaries narrow the first visible differences:

- **A:** all transformer block outputs, the final adaptive norm's image and
  conditioning inputs, and its output were bitwise identical to the singleton.
  The first observed difference was the final velocity, after the existing
  `proj_out` linear projection. This identifies that operation's boundary;
  it does not establish why its numerics differ between shapes.
- **B:** the first Double block's image/text inputs, normalized/modulated
  Attention inputs, and projected Attention outputs were bitwise identical.
  The first difference appeared at the block's text output (0.012129% relative
  L2), after its remaining gating, residual, normalization and feed-forward
  operations. The observations do not distinguish which of those operations
  introduced it. The block's image output was still identical there.

B's largest intermediate block relative L2 remained **2.809689%** at
`single_transformer_blocks.18/text`. Intermediate values are descriptive;
the predefined 2% gate applies only to final velocity. Original SDPA controls
remained within 0.649–0.912% final relative L2 of their FA3 references,
including direct comparisons against both packed orders.

These results redirect the next useful investigation toward the output
projection and the first Double block's operations after Attention. They do
not justify a correction to sequence offsets, a blanket precision change,
or treating FlashAttention as correct in every later block. A next isolated
comparison would hold the selected operation's inputs fixed while varying
singleton versus concatenated execution shape. That would be a separate
experiment; no such numerical cause was established in this run, and the
previous backward discrepancy is still open.

Complete metrics, hashes, operator evidence and the exact plan remain under
`results/flux-packing-forward/20260908T044830217531Z/` on the
`diffusion-arc-models` Volume. The recorded source and configuration hashes
match the implementation that passed 406 local repository tests.

#### Initial H200 forward result: 20260908T041120657916Z

The single authorized run completed in **164.327 seconds** with
`numerical_passed=true`, no OOM, no timeout and no retries. The manifest records
the 480-second worker / 600-second function limits. All uploaded source hashes
match the locally tested implementation. Modal app:
[`ap-V8IrPf2c9DWswzHX6DVzAF`](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-V8IrPf2c9DWswzHX6DVzAF).

The input timesteps were A=0.37817550 and B=0.91339040. Their singleton input
hashes matched their packed counterparts, including positions. The B
perturbation changed its timestep to 0.17 and preserved A and all packing
metadata. The operator audit found exactly 25 FA3 forward calls in each of
the five FA3 cases. Both original singleton controls used cuDNN SDPA, with
25 backend calls each. No backward was executed and no parameter gradients
were populated.

| FA3 packed versus FA3 singleton | Final velocity relative L2 | Final maximum absolute error | Block outputs |
| --- | ---: | ---: | --- |
| A | 0.009861% | 0.015625 | All 25 text/image block outputs bitwise identical |
| B | 0.547451% | 0.048828125 | First difference in first Double block's text output |

For A the first observed difference occurs at the final velocity output,
after identical transformer-block outputs. For B the first Double text-output
relative L2 is 0.012129%; its image output is still bitwise identical there.
The largest block relative L2 is **2.809689%**, in text at
`single_transformer_blocks.18`. Intermediate values were descriptive; the
predefined 2% screening gate applies to final velocity only. No operation-level
cause is established by these observations.

Both A and B were **bitwise identical at every captured output** in the
unchanged A+B repetition. When only B changed, A remained **bitwise identical
after all 25 blocks and at final velocity**. B's final velocity changed
(76.659503% relative L2), confirming that the perturbation had an effect.
This supports example isolation for this fixed pair and perturbation; it does
not prove all possible inputs are isolated.

The independent original controls also passed final-velocity screening:

| Original SDPA singleton versus reference | A relative L2 | B relative L2 |
| --- | ---: | ---: |
| FA3 singleton reference | 0.911999% | 0.705052% |
| FA3 packed reference | 0.911950% | 0.649355% |

These rows use the FA3 output as the relative-L2 denominator and include the
SDPA/FA3 backend change. They provide an independent model-assembly check
within the stated tolerance, not exact identity or a kernel-error attribution.

The result supports retaining the current packing implementation for further
diagnosis; it provides no basis for a precision or training-pipeline change.
The earlier 18–20% parameter-gradient differences are still unexplained.
A future investigation can use the first differing B block and A's final
output boundary to narrow capture locations before testing backward hypotheses.
Any such follow-up is a separate experiment, not part of this run.

Complete block/stream metrics, input hashes, package versions and source identity
remain in `results/flux-packing-forward/20260908T041120657916Z/` on the
`diffusion-arc-models` Volume. The local implementation passed all **400 repository
tests**; the eight forward-helper tests also passed after the final change to
persist input identities before model execution.

### Three-way forward/backward comparison

```bash
.venv/bin/modal run --detach --env diffusion-arc \
  scripts/cloud/probe_flux_packing_modal.py \
  --config configs/diagnostics/flux_packing.json --correctness
```

This mode uses the same pinned image with a separate **600-second H200 function
limit**, a worker budget of at most **480 seconds**, and no automatic retries.
It selects only the recorded short_b4 group and its original input window. The
effective scope and budget are recorded in the manifest; the original sweep
configuration is unchanged. An offline preview uses the diagnostic Python
command above with both `--correctness` and `--dry-run`.

The diagnostic-only helper
[`_flux_packing_correctness.py`](../../scripts/diagnostics/_flux_packing_correctness.py)
executes three passes with identical weights, inputs, noise, timesteps and
production weighted flow loss:

1. The original masked SDPA model path.
2. The packed model path with one SDPA call per example at each attention layer.
3. The same packed model path with FA3 Varlen.

The segmented oracle explicitly forces the SDPA backend observed in the
original pass. Operator counts verify the backend and number of calls. This
avoids silently changing SDPA backends while testing the packing path; changed
tensor shapes can still affect numerical reduction order within a backend.
The oracle is not native packed SDPA and is not a throughput candidate.

Each later pass is compared with every preceding pass. The report includes
velocity/loss differences and gradient relative L2, cosine, and squared-error
contributions per block/component. The JSON additionally retains every
parameter's statistics. Ranking by error contribution avoids mistaking a large
relative error in a tiny gradient for the dominant source of the total error.

The segmented pass captures real post-RoPE Q/K/V and the real upstream gradient
at the first Double-Stream and first Single-Stream attention. Isolated FA3 and
SDPA calls then use exactly these same inputs and upstream gradients to compare
their outputs and dQ/dK/dV. FA3 extension forward/backward counts and CUDA kernel
names are recorded. These two sampled blocks do not prove every layer agrees.

Numerical tolerance failures are recorded without skipping subsequent
comparisons. `status=complete` means the diagnostic finished;
`numerical_passed` separately reports whether its screening limits passed.
Execution failures, OOM, non-finite comparisons and the deadline still stop the
worker and preserve partial results. No optimizer updates or timing sweep take
place; individual diagnostic-call timings must not be used as speedup results.
Artifacts are stored in `results/flux-packing-correctness/<run_id>/` on the Volume.

### H200 result: 20260908T032214872710Z

The diagnostic completed in **124.067 seconds**, with no worker timeout or
optimizer updates. Modal app: `ap-3ijUHy3RYUH2a5RW5hzx4Y`.
The original and segmented SDPA passes both used `CUDNN_ATTENTION`, verified
from executed operators. The existing 2% velocity, 5% gradient and 1% loss
screening limits were unchanged; all three model comparisons failed the
gradient limit.

| Actual vs reference | Velocity relative L2 | Gradient relative L2 | Gradient cosine | Loss relative |
| --- | ---: | ---: | ---: | ---: |
| Packed segmented SDPA vs original SDPA | 0.6660% | 18.3613% | 0.988939 | 0.2323% |
| Packed FA3 vs original SDPA | 0.6611% | 20.1735% | 0.982301 | 0.1631% |
| Packed FA3 vs packed segmented SDPA | 0.7126% | 10.6575% | 0.994794 | 0.3945% |

The image double-stream modulation accounts for 86.80% of squared gradient
error in packed SDPA vs original, and 86.41% in FA3 vs original. Single-stream
modulation adds 10.10% and 11.82%, respectively. Including text modulation,
the three modulation modules account for **97.42%** and **98.50%** of those
squared errors. These are error contributions, not shares of parameters or
the fraction of individual gradients that differ.

Isolated FA3 vs cuDNN SDPA used identical real BF16 inputs and identical
captured upstream gradients:

| Sampled attention | Output relative L2 | dQ relative L2 | dK relative L2 | dV relative L2 |
| --- | ---: | ---: | ---: | ---: |
| First Double-Stream block | 0.0731% | 0.2291% | 0.2540% | 0.0074% |
| First Single-Stream block | 0.0961% | 0.4099% | 0.2337% | 0.0105% |

Both sampled kernels passed; FA3 forward/backward extension counts and CUDA
kernel names were captured. This supports correct kernel binding at those
interfaces. It does not prove all layers or the complete BF16 backward agree.
In particular, the 10.66% full-model difference between the two packed variants
remains material and cannot be dismissed on the basis of two isolated checks.

The largest squared gradient errors occur in modulation modules, and the
full-model discrepancy is also present without FA3. Those modules already
contain approximately 86% of the squared reference gradient norm, so their
97–98.5% error contribution alone does not identify the cause. BF16 backward
accumulation through the repeated `index_select` indices is an untested
hypothesis. No isolated accumulation comparison or precision correction was
performed. The subsequent singleton-versus-packed forward test is documented
above; it does not resolve these earlier gradient discrepancies.

Full per-parameter/component metrics and reproducibility metadata are in
`results/flux-packing-correctness/20260908T032214872710Z/{summary.json,manifest.json,report.md}`
on the `diffusion-arc-models` Volume. The preceding run
`20260908T024953580110Z` stopped after its first FA3 comparison at 19.5617%
gradient relative L2; it had already confirmed the kernel-boundary BF16 fix.

## Fixed inputs and comparisons

The configuration records original ARC augmentation keys, consumed source
lengths and all 48 target-prefix lengths from B0 repetition 1. The existing B2
scheduler reconstructs each group before execution. The selected target indices
and original padded boundary must match exactly. Tokenization and production
NVARC/VAE preparation verify real contents and lengths outside timing. Selected
frozen inputs are copied to CPU, and only the current group's inputs are moved
back to the GPU for measurement.

| Group | B0 step | Batch size | Real text lengths | Baseline text boundary |
| --- | ---: | ---: | --- | ---: |
| `short_b4` | 35 | 4 | 468, 406, 357, 288 | 487 |
| `medium_b2` | 35 | 2 | 2740, 2526 | 2749 |
| `medium_b4` | 37 | 4 | 2077, 2017, 2009, 1956 | 2126 |
| `long_b1` | 37 | 1 | 4948 | 6100 |
| `maximum_b1` | 31 | 1 | 8400 | 8400 |
| `long_b2` | 34 | 2 | 4745, 4227 | 4837 |

Each target retains all **900 image tokens**, including canvas sentinel cells.
Only masked text padding is removed. Noise and timesteps are drawn once per
target with the production flow sampler and reused across every variant and
round. This uses a recorded diagnostic seed per group; it does not claim to
replay the original training step's RNG state or updated weights.

Every group compares:

1. `sdpa_b2`: production padding, joint attention mask and velocity function.
2. `fa2`: completely packed model execution with official FA2 Varlen attention.
3. `fa3`: the identical packed execution with official FA3 Varlen attention.
4. `fa4`: the identical packed execution with official FA4 Varlen attention.

FA2/FA3/FA4 receive identical contiguous Q/K/V tensors and noncausal segment
boundaries. Their comparison measures backend differences within a common
execution path. The comparison against SDPA includes changes to both attention
and model layout/padding. Profiling records SDPA's actual selected backend;
previous cuDNN observations are not assumed to prove this run's dispatch.

There is **no alignment sweep, homogeneity sweep, regrouping, token-budget
search, or attempt to place all 48 targets into one physical forward**. Those
would change the workload and belong to a later experiment. These six selected
groups do not represent a dataset-weighted throughput estimate.

## Packed implementation

[`_flux_packing.py`](../../scripts/diagnostics/_flux_packing.py) uses the real
FLUX Single- and Double-Stream block modules, normalization, Q/K/V projections,
fused QKV/MLP projections, MLPs, timestep modulation, positional embeddings and
velocity head. No model parameters are duplicated. Explicit attention processors
use the model's existing processor interface and are restored after every case,
including failures. There are no global dispatch patches or interception hooks.

Text and image streams contain only their real tokens. Double-Stream processors
gather Q/K/V into `[text A, image A, text B, image B, ...]`, execute Varlen
attention, and return results to the separate streams. Single-Stream blocks
keep that joint order throughout. Final image tokens are restored to
`[batch, 900, channels]` before the existing output normalization and head.

Original per-example text positions and image coordinates are retained. Token
ownership expands each example's timestep modulation over its own tokens. This
explicit expansion allocates additional tensors; its time and memory, together
with all within-forward permutations, are included in measurements. Boundary
metadata and index tensors are constructed once per group, with no repeated
GPU-to-CPU length reads inside attention layers.

CUDA autocast evaluates Q/K RMSNorm in FP32. After normalization and RoPE,
the diagnostic converts Q/K/V to the V projection's BF16 compute dtype before
calling the official kernel, matching SDPA's input autocast behavior. The
attention output is restored to the original Q dtype, as in the production
processor. Parameters and optimizer states remain FP32. This conversion and
its backward path are included in the measurements.

The initial run `20260908T022241651580Z` failed at
`short_b4/fa3/first_forward_backward` because that conversion was missing.
The tiny preflights supplied BF16 Q/K/V directly and therefore did not exercise
the preceding model normalization. CPU autocast also did not reproduce CUDA's
FP32 RMSNorm policy. A regression test now explicitly reproduces FP32 Q/K with
BF16 V in both real block types, requires uniform BF16 kernel inputs, and checks
outputs and parameter gradients against the padded path. That test failed
before the correction and passes after it. The initial run produced no complete
backend timing comparison.

## Correctness and measurement

Before loading the full training footprint, each backend runs a tiny BF16,
noncausal, variable-length Q/K/V forward/backward comparison against explicitly
selected PyTorch math SDPA. Perturbing one segment must leave the other segment
bitwise unchanged at the same shape and grad mode. There is no backend fallback.

For every real group, a fresh production reference forward/backward supplies
velocity outputs, loss and **all populated FLUX parameter gradients**. Reference
gradients are retained on CPU; comparisons stream through bounded GPU scratch
space outside timing. Each variant's first full forward/backward is recorded
separately and checked against this reference. Frozen input preparation remains
identical across backends.

The configuration fixes diagnostic gates before execution: relative L2 error
at most **2% for velocity**, **5% for gradients**, and **1% relative loss error**.
These are screening limits, not established training-equivalence criteria.
Maximum absolute error, RMS error and relative L2 error are retained. Non-finite
values, missing gradient coverage, or exceeded limits stop the sweep and retain
partial results. CPU tests additionally compare all tiny-model parameter and
input gradients, different timesteps, masking, sample isolation and reuse of
the production blocks.

Each of the **24 cases** runs three rounds of two warm-ups and five measured
calls: **360 measured calls**, plus 144 warm-ups. Variant order is seeded and
rotated between rounds. First calls, reference calls and profiler calls are
additional and excluded from timing statistics. First calls are not independent
cold starts: small preflights and earlier shapes can populate kernel caches.

The diagnostic reuses FLUX shape-probe initialization: frozen BF16 NVARC,
frozen FP32 VAE, FP32 FLUX parameters/gradients/AdamW moments, BF16 compute,
activation checkpointing disabled, and a conservative resident preparation
reserve. Optimizer memory initialization uses learning rate zero. There are no
optimizer updates during comparisons. Backward uses the production `B / 48`
loss weight, and gradient storage remains resident between calls.

Synchronized wall time covers input assembly plus forward/loss/backward.
Assembly uses production text padding for the baseline and concatenates exact
prefixes for packed variants. CUDA events separately measure forward/loss,
weighted backward and their combined span; these include dispatch gaps and
are not kernel-duration sums. Data loading, frozen preparation, flow sampling,
static metadata construction, gradient zeroing, comparisons, logging and
output destruction are outside these clocks. Peak memory covers each call's
resident footprint plus assembly and execution; summary peaks use measured
calls, while raw warm-up/first-call peaks are also retained.

After all timed variants of a group, each receives one separate profiler pass.
Both Single- and Double-Stream blocks are covered. Profiles verify the expected
explicit attention call count, FA2/FA3 forward **and backward** extension calls,
FA4 backward evidence, and CUDA kernel names. Tensor-shape, stack and profiler
memory recording are disabled to avoid retaining activations.

## Artifacts and failure behavior

Results remain under `/mnt/diffusion-arc-models/results/flux-packing/<run_id>/`:

- `manifest.json`: source/config/data hashes, Git identity, exact plan, command,
  image/resource metadata and configurations. Weight contents are not hashed.
- `result.json`: worker progress, raw first/warm-up/measured calls, numerical
  comparisons, runtime versions, setup footprint and backend evidence.
- `summary.json` and `report.md`: medians, p95, round medians, peak memory and
  speedups only when both comparison rows are complete.
- `traces/*.json.gz`: separate CPU/CUDA timelines for all 24 cases.
- `worker.log`: streamed progress and original errors.

The worker atomically saves progress around individual timing calls, and commits
the Volume at phase/round boundaries and on exit. OOMs and errors stop the run;
the same CUDA context is not retried. A global timeout identifies the last
active phase and preserves partial results. It does not establish a defect in
whichever backend happened to be running when the overall budget expired.

Sources: [FA2 Varlen interface](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post1/flash_attn/flash_attn_interface.py),
[pinned FA3 interface](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/hopper/flash_attn_interface.py),
[FA3 build flags](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/hopper/setup.py),
and [FA4 API](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute).
The runtime records installed package versions; source inspection alone is not
evidence that the selected H200 training configuration works.
