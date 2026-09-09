# FLUX modulation backward intervention

This report preserves the diagnostic state and evidence before B3 integration.
Its code, configurations and any commands below belong to the
[archived implementation](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
they are not current entrypoints. Later production choices are summarized in
[B3](README.md#b3-packed-nvarc-and-flux).

**Status: completed successfully, 2026-09-09.** All 47 targeted local tests and
all ten full-model forward/backward calls passed their execution checks.

The preceding [backward-layout experiment](flux-backward-layouts.md) measured
17–19% full-parameter gradient differences between fully packed execution and
unpadded singletons within the same backend. Approximately 99% of squared
gradient error was in the three modulation modules. Isolated attention packing
did not reproduce the large discrepancy. This experiment tests whether the
modulation ownership reduction causes a substantial part of that difference.

## Measured result

Run `20260909T022044974841Z` reduces the full-parameter gradient discrepancy
from **17.59–17.99% to 3.14–3.22%**, while forward fingerprints and per-example
losses remain exactly unchanged. Both repetitions use the same singleton
reference, with all 169 trainable tensors / 3,875,544,576 elements included.

| Execution / unpadded singleton reference | Gradient relative L2, two repetitions | Cosine, first repetition |
| --- | ---: | ---: |
| Native packed AB | 17.9922%, 17.5863% | 0.986149088 |
| FP32 reduction, packed AB | 3.2210%, 3.1416% | 0.999610622 |
| FP32 reduction, packed BA | 3.1885%, 3.1748% | 0.999600807 |

For the first AB executions this is **82.10% less relative L2 error**, or
96.80% less squared error against the same reference. It is a measured error
reduction, not a claim to have uniquely attributed that fraction of every
original gradient difference. The intervention directly establishes that the
native BF16 ownership backward is a major contributor to this fixture's large
full-model discrepancy; FA3's attention arithmetic did not change.

Unchanged singleton repetition differs by **0.4926%**. Native packed repetition
differs by **1.4719%**, compared with **0.5284%** for FP32 AB and **0.5530%** for
FP32 BA. FP32 BA versus AB differs by **0.7619%**. Two repetitions do not establish
a distribution or prove complete order invariance.

Modulation-only gradient error falls from **19.1579% to 2.9594%** in the first
AB pair. Other parameters change from **5.5228% to 4.6177%**; reduced modulation
gradients also feed the shared timestep embedding. The remaining total error
around 3.2% is above the approximately 0.5% repeat scale and is not explained or
declared harmless by this experiment.

The FP32 option is implemented in the diagnostic packing path; production
`src/` code is unchanged. Retain this option for the next packed-model
evaluation. One optimizer update and a complete-step performance comparison
are the next useful validation, followed by the existing small overfitting
workflow. Their outcomes are not inferred from this gradient improvement.

The [Modal app](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-tIOzNynG98Mt7Wu1jduxjp)
is stopped with zero tasks. Its creation-to-stop interval was **271 seconds**,
giving a lifecycle resource-price estimate of **$0.40864**, not a measured
account charge. The worker completed in 210.7 seconds and the supervisor in
227.8 seconds. No timeout or additional GPU submission occurred.

All 84 source, four configuration and 965 dataset hashes match the launch
manifest. Input tensor identities and checkpoint evidence also match the
preceding backward-layout run. All cases used unchanged weights and inputs;
each modified execution recorded exactly three BF16-source forward calls and
three FP32-reduction backward calls. The raw result retains `status=complete`.

## Intervention and controls

Only the backward of the three modulation `index_select` operations changes.
Forward uses the same gather, with unchanged values, dtype, shape and strides.
Backward sums each example's contiguous token contributions in FP32, then casts
once to the original BF16 modulation dtype. Existing metadata preflight checks
the correspondence between ownership indices and segment lengths. The custom
autograd operation supports first-order derivatives only.

Use the same real A/B fixture, base weights, timesteps, positions, noise and
loss as the preceding experiment: text lengths 468/406, 900 image tokens each,
seed 167, and each example's loss weighted by 1/48. Trainable parameters and
persistent gradients remain FP32; the full forward remains under BF16 autocast.

| Execution | Repetitions | Complete forward/backward calls |
| --- | ---: | ---: |
| FA3 unpadded singletons with accumulated gradients | 2 | 4 |
| FA3 packed AB, native ownership backward | 2 | 2 |
| FA3 packed AB, FP32 ownership backward | 2 | 2 |
| FA3 packed BA, FP32 ownership backward | 2 | 2 |

All eight logical batches report complete parameter gradients against the
first singleton reference, except that reference itself. Repeated layouts are
also compared to their first execution. FP32 AB is compared directly to native
AB, and FP32 BA to FP32 AB. At most two complete CPU gradient snapshots coexist.

Exact byte fingerprints of velocity outputs and five captured intermediate
activations, restored to A/B identity, must agree between modified packing and
native AB. Per-example losses must also be identical. These observations verify
the unchanged gather implementation in the full model; they do not assert that
every internal tensor was captured. Each modified backward must record exactly
three intervention calls with BF16 sources and FP32 reduction.

Report relative L2, cosine, norm ratio, modulation-only error, remaining
parameter error and repeatability. A substantial decrease supports a causal
role for ownership reduction; any residual difference remains a separate result.
No optimizer-update, convergence or training-throughput qualification is made.
Ordinary batching and attention-only packing remain alternatives if full-model
packing does not justify its additional complexity. NVARC's frozen forward path
has no corresponding trainable modulation reduction to change.

## Validation and resource limits

All **47 targeted local tests passed**, including independent FP64 summation of
BF16 contributions, noncontiguous tensors, uneven segments and reversal,
complete tiny-model gradients against an independent singleton-loss definition,
unchanged BF16 forward values, intervention coverage, incomplete-evidence
rejection, source/config identity checks, and bounded cancellation/persistence.

The intended FA3 route is checked through CPU forward/backward operator counts
for every block. CUDA kernel-event capture is explicitly not requested in this
experiment; this does not revise the previous run's failed profiler-evidence
status. No isolated attention replay, optimizer update, clipping or activation
checkpointing is included.

The user approved at most **$1 additional GPU spending**. Reuse cached image
`im-R5CncDboPfMQryWnq0UZn4`, existing checkpoints and dataset, with no build,
download, requested retry or premium placement. One H200, eight CPU cores and
64 GiB host memory have explicit request/limit values. The worker is bounded to
420 seconds, function to 480, startup to 90, and local lifecycle to 600 seconds
including a 30-second cancellation reserve. One explicit final volume commit
is bounded to 60 seconds; a function timeout may interrupt that commit.

At the [Modal list prices](https://modal.com/pricing) checked on 2026-09-09,
the configured resources total $0.00150788/second, or **$0.904728 for 600 seconds**.
This is a resource-price estimate, not a measured account charge.

Entry points: `scripts/diagnostics/probe_flux_modulation_backward.py` (offline
`--plan`) and `scripts/cloud/probe_flux_modulation_backward_modal.py`.
Configuration: `configs/diagnostics/flux_modulation_backward.json`.
Results: `diffusion-arc-models:results/flux-modulation-backward/<run-id>/`.
Manifests retain source, configuration, dataset and checkpoint identities.
