# Attention execution diagnostic — first wave

This report preserves the diagnostic state and evidence before B3 integration.
Its code, configurations and any commands below belong to the
[archived implementation](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
they are not current entrypoints. Later production choices are summarized in
[B3](README.md#b3-packed-nvarc-and-flux).

**Status: complete and independently validated, 2026-09-09.**

This implements stages 0 and 1 of the approved
[attention diagnostic plan](attention-diagnostic-plan.md). It measures token
ownership, isolated attention input gradients, frozen post-attention operations
and timestep-modulation output gradients. It does not measure complete model
parameter gradients, optimizer updates, learning behavior or training throughput.

## Fixed experiment

The original unpadded Diffusers singleton forward supplies real post-RoPE BF16
Q/K/V through an attention-only FA3 adapter. The reference neither constructs
nor consumes packing layouts. Its capture backend is an input-generation choice,
not a preference in the backend comparison. Every kernel receives the same
Q/K/V values and sample-associated upstream; FP32 Math promotes those values.

| Source group | Text tokens A / B | Image tokens per sample | Captured blocks |
| --- | --- | ---: | --- |
| Historical short pair | 468 / 406 | 900 | double.0, double.4, single.0, single.19 |
| Medium pair | 2740 / 2526 | 900 | same four blocks |
| Long pair | 4745 / 4227 | 900 | same four blocks |

Existing B0 training-source fixtures and cached checkpoints are reused. Frozen
NVARC conditioning is prepared once with forced cuDNN SDPA and the FP32 VAE
with explicitly forced Math SDPA,
then retained unchanged. Flow matching draws retain seed 167 and sample order.
The manifest records source/configuration/data hashes; results record actual
checkpoint files, inputs, positions, noise/timesteps/targets, package versions
and profiler dispatch evidence. No model or data downloads are requested.

Twelve pair/block cases cover FP32 Math, BF16 Math, cuDNN and FA2/FA3/FA4.
Each backend uses A, B, AB, BA and an AB repeat for three upstream seeds.
The first seed adds independent A/B perturbations and an A-only upstream to
check zero gradients into B. That is 1,296 production forward/VJP evaluations,
plus at most 180 evaluations of explicitly supported deterministic Flash
backward modes. Direct backend pairs provide 540 comparisons, each with two
examples and output/dQ/dK/dV metrics. The short first block is profiled.
Math/cuDNN execute separate segment calls; this is explicitly a correctness
control and makes no native-ragged performance claim.

The short pair additionally replays frozen operands of the final output
projection, all three timestep-modulation modules, and the first Double block's
post-attention gates, residuals, normalization and MLP. Separate primitives
identify local shape effects; a chained replay shows propagation. Modulation
output gradients compare broadcast sums with repeated `index_select` ownership
using the same upstreams in BF16 and an explicitly labelled FP32 diagnostic.

## Structural checks and execution boundaries

An independent labelled-token oracle checks permutations, inverse gathers,
positions, ownership, analytical gradients, repeated-index sums, boundaries
around 128/256 and 8,400 text plus 900 image tokens. A new explicit diagnostic
preflight rejects malformed layouts. A primitive padding gather/scatter test
checks middle padding; an attention-only packed model adapter is a later stage.

The approved single-H200 worker ceiling was 900 seconds including setup, with
120 seconds reserved for artifact finalization. After the preparation failure
described below, the corrected attempt used the remaining budget conservatively:
750 worker seconds and an 870-second Modal function limit. Automatic retries
are disabled. Results
are written incrementally and a timeout is incomplete. Backend errors preserve
partial evidence; independent cases continue. Completion, numerical differences
and qualification for training are separate outcomes.

The entry points are `scripts/diagnostics/probe_attention_execution.py` and
`scripts/cloud/probe_attention_execution_modal.py`, with configuration in
`configs/diagnostics/attention_execution.json`. The diagnostic supports an
offline `--plan` listing exact case counts and limits. Production code, weights,
training defaults and the root README remain unchanged.

## Validation and run

The full repository suite passed 512 tests in 26.268 seconds. After tightening
the operation-result validator, 19 focused operation/coordinator/launcher tests
passed in 0.440 seconds. `git diff --check` passed.

The first attempt, `20260908T231427590447Z`, stopped after 109.346 worker seconds
before any kernel comparisons. The cuDNN selection had incorrectly enclosed
the FP32 VAE as well as BF16 NVARC; cuDNN rejected the VAE's FP32 Q/K/V. The
diagnostic now scopes an explicit Math context to the VAE and restores cuDNN
for NVARC. Twelve focused coordinator/launcher tests passed after this fix,
including a regression for backend scope and restoration. No precision change
or silent fallback was introduced.

The corrected run is `20260908T232008821191Z`, on the existing Modal environment
`diffusion-arc` and volume `diffusion-arc-models`:
[H200 execution](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-ka5lTVNdjYqgGW31eQOhVB).
The worker completed in **225.286 seconds**, with all 1,476 kernel evaluations,
540 direct backend/seed comparisons, 130 operation-forward cases and 72
modulation-reduction cases. Combined worker time across both attempts is
334.632 seconds, below the approved 900-second ceiling. The corrected
supervisor took **636.603 seconds**, including synchronous progress/artifact
commits; the worker time is not the complete GPU allocation time. Both
supervisors together took 753.170 seconds. These are diagnostic runtimes,
not training-throughput measurements.

The final `summary.json`, `result.json`, `manifest.json`, `report.md` and
`worker.log` are stored at
`diffusion-arc-models:results/attention-execution/20260908T232008821191Z`.
The downloaded artifacts passed the full completion validator; all **76 source,
3 configuration and 965 dataset file hashes** matched the local launch inputs.
The worker returned zero, no timeout occurred, and the final summary preserves
every worker result. All nine backend profiles include CUDA kernel evidence.

## Measured kernel packing behavior

Every backend had **zero measured output, dK and dV error** between singletons,
AB, BA and unchanged repetitions. FP32/BF16 Math also had zero dQ error.
The following are maximum relative L2 errors in percent over all three lengths,
four blocks, three seeds and both examples, in the default backward mode:

| Backend | dQ: AB vs singletons | dQ: BA vs singletons | dQ: AB vs BA | dQ: unchanged repeat |
| --- | ---: | ---: | ---: | ---: |
| cuDNN segmented control | 0.005134 | 0.005195 | 0.003638 | 0.002996 |
| FA2 | 0.004507 | 0.002966 | 0.004745 | 0.004690 |
| FA3 | 0.003532 | 0.004843 | 0.004835 | 0.003420 |
| FA4 | 0.004282 | 0.004692 | 0.004822 | 0.003749 |

With deterministic backward, FA3 and FA4 had zero measured error for all four
tensors in every comparison. Deterministic FA2 retained at most **0.002030% dQ**
between singleton and packed execution (`short_b4/double.4/A`, seed 167), while
AB/BA and repetition agreed. This directly separates repeatability from batch
invariance. Zero kernel error means numerical equality in these metrics;
output byte identity, including signed zeros, was not separately recorded.

All **144 perturbation cases** preserved the other sample's output. All **72
A-only backward cases** produced zero B input gradients. No perturbation
gradient variation occurred alongside an exact unchanged backward repetition.
Nine profiles verified both forward/backward operators and CUDA kernels: six
default backend paths and deterministic FA2/FA3/FA4. These results support the
isolated segment contract, not full-transformer training qualification.

## Backend differences remain separate

Direct comparisons on identical singleton Q/K/V and upstream gradients give
these maximum relative L2 percentages against cuDNN:

| Backend | Output | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| FA2 | 0.106317 | 0.555932 | 0.442440 | 0.044904 |
| FA3 | 0.103046 | 0.530796 | 0.432520 | 0.044782 |
| FA4 | 0.104069 | 0.538885 | 0.436302 | 0.042029 |

Against the promoted-input FP32 Math reference, all accelerated paths reach
approximately 0.170% output error and 1.242–1.244% dQ error. BF16 Math reaches
0.166758% output and 0.166451% dQ. The largest accelerated dQ occurs at
`long_b2/single.19/B`, seed 169. FA3 there has error RMS 0.00100924, reference
RMS 0.08115134 and cosine 0.99992268. Similar output errors therefore do not
imply similar gradient errors. None of these distances defines an acceptance
threshold for learning or selects a fastest backend.

## Surrounding operations and modulation reduction

The 36 capture/replay comparisons reproduced the original operations bitwise.
All 84 direct forward-repetition comparisons were also bitwise identical.
Frozen-input projection replays then isolated two shape effects:

- Final `proj_out`, singleton versus `[2,900,3072]`: **0.011106%** error for A,
  **0.009818%** for B; maximum absolute error 0.015625.
- First Double block, text B: attention gate, residual, norm, modulation,
  MLP input projection and SwiGLU remained identical. The first difference was
  **`ff_context.linear_out`**, with **0.019631%** error and maximum absolute
  error 0.25. After the remaining gate/residual, block-output error was
  **0.004364%**, maximum 0.125. Image A/B and text A remained identical in this
  replay. No specific GEMM algorithm or scheduling cause was identified.

These are local frozen-operand measurements; they are not a rerun of the
previous end-to-end 0.547451% velocity discrepancy. BA reproduced the AB
comparison statistics, but a direct BA/AB tensor comparison was not retained
for operation forwards and should not be claimed from equal error norms.

The detached modulation-output experiment shows a larger backward effect.
Ranges below cover A/B and three identical upstream seeds; upstream gradients
are synthetic, while modulation values and token lengths come from real inputs.

| Modulation | BF16 indexed reduction vs broadcast | BF16 unchanged indexed repeat | Maximum absolute error vs broadcast |
| --- | ---: | ---: | ---: |
| Image, 900 / 900 tokens | 3.498–3.545% | 2.289–2.546% | 9.5 |
| Text, 468 / 406 tokens | 2.347–2.595% | 1.878–1.983% | 5.25 |
| Joint, 1368 / 1306 tokens | 4.184–4.393% | 2.692–3.063% | 13.0 |

Using FP32 for reduction of the **same values** reduced the maximum relative
error to **2.95e-8** (0.00000295%), maximum absolute error 1.53e-5. Native BF16
broadcast itself differs from FP32 broadcast by only 0.164–0.168%. The indexed
BF16 path also changes between unchanged repetitions; its AB/BA differences
cannot therefore be attributed solely to deterministic ordering.

## Consequences and remaining validation

Sequence segmentation is supported by these fixed-Q/K/V tests across all
three Flash backends. The largest measured packing effect is instead in
BF16 modulation-gradient reduction. This identifies an isolated numerical
problem; it **does not explain the prior 18–20% full-parameter-gradient error**.
The true loss-derived upstream, all model gradients, clipping and AdamW updates
still need the planned second wave.

Two alternatives remain useful: attention-only packing preserves native MLP
and modulation layouts; full-model packing could accumulate ownership gradients
in FP32 while preserving BF16 forward values. The smallest next validation is
a controlled full-backward comparison of that reduction change, with an
unchanged-forward check and one optimizer update. Checkpointing and complete
step throughput follow separately. NVARC cuDNN-ragged and FA3 remain end-to-end
candidates from the earlier measurements; this study changes neither the
production backend nor their performance ranking.
