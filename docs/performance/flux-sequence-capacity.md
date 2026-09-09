# FA3 single-sequence length limit

This report preserves the diagnostic state and evidence before B3 integration.
Its code, configurations and any commands below belong to the
[archived implementation](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
they are not current entrypoints. Later production choices are summarized in
[B3](README.md#b3-packed-nvarc-and-flux).

**13,312 joint tokens pass full FLUX forward/backward; 13,568 tokens fail
with an allocation OOM.** The lower length passed three additional independent
confirmations, and the upper OOM reproduced. The boundary is resolved to 256
tokens for the recorded H200 runtime.

This corrects the scope of the earlier
[whole-example capacity experiment](flux-packed-capacity.md): that experiment
grew particular packs of real examples rather than isolating sequence length.
Its native-versus-FP32 performance comparison remains valid.

## Method

Every probe has exactly one attention segment, `cu_seqlens=[0,L]`: `L-900`
text tokens followed by 900 image tokens. Text values repeat the existing
8,400 × 7,680 BF16 NVARC hidden-state template, with continuous positions and
no new sequence boundaries. The latent, timestep, noise seed (170), model and
loss weight (1/48) remain fixed. No additional model or dataset is downloaded.
The diagnostic length override does not change NVARC's production 8,400-token
contract or establish generation quality at longer contexts.

The complete FLUX transformer uses FA3, BF16 compute and the three explicit
FP32 modulation-gradient sums. FP32 weights, gradients and initialized AdamW
state, frozen NVARC/VAE and the existing preparation reserve remain resident.
Checkpointing and TF32 are disabled. No optimizer update occurs during the
search; one LR=0 setup step initializes optimizer storage.

The search doubles length from 8,192 until failure, then bisects in steps of
256. Garbage collection and allocator cache cleanup precede every probe,
including each confirmation. Only model-execution allocation OOMs define an
upper bound. Other errors abort with partial evidence saved.

## Measurements

| Joint tokens | Text tokens + image tokens | Result | Peak allocated GiB |
| ---: | --- | --- | ---: |
| 8,192 | 7,292 + 900 | Pass | 113.35 |
| 12,288 | 11,388 + 900 | Pass | 133.02 |
| 13,312 | 12,412 + 900 | Pass, plus 3 confirmations | 137.94 |
| 13,568 | 12,668 + 900 | OOM, independently reproduced | — |

The 13,312-token confirmations have identical loss (0.8467051983), complete
finite FP32 gradients for all 169 parameter tensors / 3,875,544,576 parameters,
and unchanged inputs, weights and persistent gradient storage. Their median
forward/backward time is 998.84 ms; assembly plus forward/backward is 1,016.75
ms. These few capacity calls are not a steady-state throughput benchmark.
The small control's CPU operator profile verifies 25 FA3 forward and 25
backward dispatches across all transformer blocks.

The allocator is `native`; both allocator configuration environment variables
are unset. At 13,312 tokens, peak reserved memory is 138.98 GiB and only
69.19 MiB of device memory remains free after the call. At 13,568 tokens,
both failures request 160 MiB with 125.19 MiB free. This is a reproducible
execution bound under the recorded allocator policy, not an allocator-independent
physical maximum.

For initial microbatch design, **12,288 joint tokens** provides a measured
starting point below the edge. Count every example's text and its 900 image
tokens in a future pack. The single-segment result alone does not certify
every multi-example composition or mixed-shape allocation history. The old B2
shape catalog need not determine the new packing budget; its scheduler can
eventually be replaced by token-based grouping once that integration is checked.

## Reproduction and cost

Run `20260909T064617025018Z` completed on H200 with PyTorch 2.13.0, CUDA 13.0,
Diffusers 0.39.0 and flash-attn-3 3.0.0, using cached image
`im-R5CncDboPfMQryWnq0UZn4`. The
[Modal app](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-7TrohllxUeysIhzk5mOmvx)
is stopped with zero tasks. The worker took 113.61 seconds and app lifecycle
147 seconds. The configured resource-rate estimate is **$0.221658** for this
run, **$0.404112** including the earlier capacity run, leaving approximately
**$0.915888** of the user's reported $1.32. Actual account billing was not queried.

- [Configuration](https://github.com/joshuakauling/diffusion-arc/blob/2ff42764b06163fc1df855fc2529ae98a5cf2e2b/configs/diagnostics/flux_sequence_capacity.json)
- [Coordinator](https://github.com/joshuakauling/diffusion-arc/blob/2ff42764b06163fc1df855fc2529ae98a5cf2e2b/scripts/diagnostics/probe_flux_sequence_capacity.py)
- [Bounded launcher](https://github.com/joshuakauling/diffusion-arc/blob/2ff42764b06163fc1df855fc2529ae98a5cf2e2b/scripts/cloud/probe_flux_sequence_capacity_modal.py)
- Volume artifacts: `results/flux-sequence-capacity/20260909T064617025018Z`

Artifacts contain every probe, confirmation, allocation statistic/error,
checkpoint identity, input fingerprint, package version, seed and source/config/
dataset hash. Retrieved evidence passed complete-result validation and matched
all 85 source, three configuration and 965 dataset hashes. The 52 targeted local
tests passed, including single-segment construction beyond 8,400 text tokens,
unchanged default preflight limits, bracket refinement, failed-graph release,
confirmation failures, partial persistence and cloud budget guards.
