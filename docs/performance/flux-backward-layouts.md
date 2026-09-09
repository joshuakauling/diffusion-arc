# FLUX backward layout diagnostic

This report preserves the diagnostic state and evidence before B3 integration.
Its code, configurations and any commands below belong to the
[archived implementation](https://github.com/joshuakauling/diffusion-arc/tree/2ff42764b06163fc1df855fc2529ae98a5cf2e2b);
they are not current entrypoints. Later production choices are summarized in
[B3](README.md#b3-packed-nvarc-and-flux).

**Status: numerical matrix completed; requested CUDA trace evidence incomplete,
2026-09-09.** All 36 focused local tests passed. The remote result correctly
retains `status=error` because the final profiler-evidence check failed after
all numerical cases had finished. No rerun was submitted.

## Measured results

Run `20260909T013747877768Z` executed all **22 full forward/backward calls** and
**72 fixed-input attention forward/VJPs**. Every full-model case covered all
169 trainable tensors / 3,875,544,576 parameter elements, using identical input
values and unchanged weights. Every padded text-input gradient was zero.

| Gradient comparison | Relative L2 | Cosine | Norm ratio |
| --- | ---: | ---: | ---: |
| cuDNN padded singletons / unpadded singletons | 3.668690% | 0.999391806 | 0.987994 |
| cuDNN padded batch / unpadded singletons | 3.639215% | 0.999394976 | 0.988685 |
| cuDNN padded batch / padded singletons | 0.509417% | 0.999987278 | 1.000699 |
| cuDNN packed AB / unpadded singletons | 19.020895% | 0.984386698 | 1.056474 |
| cuDNN packed BA / unpadded singletons | 18.695175% | 0.985025991 | 1.057325 |
| cuDNN packed AB / padded batch | 18.493075% | 0.986197284 | 1.068565 |
| FA3 packed AB / FA3 singletons | 17.591368% | 0.986702165 | 1.053984 |
| FA3 packed BA / FA3 singletons | 17.101672% | 0.987398812 | 1.052230 |

Unchanged repetitions differ by **0.4723–0.6841%** for the native singleton/
padded layouts and **1.5555–2.0172%** for packed layouts. Packed BA versus AB
differs by **2.2979%** for cuDNN and **1.6641%** for FA3; their outputs and
losses are equal. The two repetitions do not establish a noise distribution
or cleanly separate order effects from nondeterministic backward variation.

For this pair, padding accounts for most of the native singleton-to-batch
gradient difference. Batching with the same padding changes gradients at
approximately the unchanged-repeat scale. Full-model packing retains a much
larger discrepancy **within each backend**, so a cross-backend difference or
batch-size loss factor cannot explain it away.

The three modulation modules contain **98.66–99.06% of squared gradient error**
in packed-versus-singleton comparisons and approximately **87.2% of reference
gradient energy**. The remaining parameters still differ by **5.16–5.52%**.
This reinforces modulation backward as a useful intervention target, but
does not prove its accumulation strategy causes the complete difference.

## Token boundaries

There is a pronounced endpoint effect in **padded singleton A versus unpadded
singleton A**, at the `single.0` output gradient: relative L2 is **14.5378%**,
and the last 32 of 1,368 tokens contain **86.448% of squared error**, compared
with **37.482% of reference gradient energy**. The largest error is image token
position 1,342 in the joint sequence. This happens without a second example.
At the same observation, batching already-padded examples changes gradients
by 0.5992%, and an unchanged unpadded repetition changes them by 0.5668%.

Packing does not show a universal boundary-only effect. Its largest observed
B output-gradient differences occur at the first text MLP output projection:
**8.2544%** for cuDNN AB and **10.5149%** for FA3 AB. The text interior contains
**88.115%** and **66.717%** of their squared error, respectively. FA3's boundary
windows contain 33.283% of error against 29.657% of reference gradient energy.
These local activation derivatives do not quantify the cause of the separate
17–19% aggregate parameter-gradient difference.

Fixed-input Attention packing/reversal/repetition gives zero measured output,
dK and dV differences. Maximum dQ errors are:

| Comparison | cuDNN | FA3 |
| --- | ---: | ---: |
| AB / singletons | 0.0037675% | 0.0018584% |
| BA / singletons | 0.0037648% | 0.0021481% |
| BA / AB | 0.0037651% | 0.0021810% |
| Unchanged repetition | 0.0037667% | 0.0020654% |

Some of these tiny dQ errors are highly localized: cuDNN `double.4/A` places
99.942% of AB-versus-singleton error in edge windows, but an unchanged singleton
repeat places 99.992% there at nearly the same total error. Concentration alone
does not distinguish packing from ordinary backward variability.

The isolated padded cuDNN batch differs from singletons by at most 0.061084%
in output, **0.181625% dQ**, 0.111518% dK and 0.011524% dV. Its largest dQ case
is `single.19/A`, peaking at joint token **571 in the interior**. All padded
Q/K/V input gradients remain zero. These kernel results do not reproduce the
large full-model endpoint discrepancy as a comparably large local kernel spike.

The subsequent authorized [modulation backward intervention](flux-modulation-backward.md)
changed only ownership-gradient reduction to FP32, keeping BF16 forward values
fixed. It reduced the full FA3 gradient difference from approximately 18% to
3.2%. This identifies a major contributor without establishing optimizer-update
or training equivalence. Ordinary batching and isolated attention packing
remain alternatives to reassess against complete-step performance.

## Execution evidence and limitations

The [Modal app](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-0mrH4OLYXyziH8izWxgzHN)
is stopped with zero tasks. Its full creation-to-stop interval was **410 seconds**;
the worker reached its final validation after 353.6 seconds, and the supervisor
took 368.6 seconds before the final volume commit. The configured resource
prices give an upper lifecycle estimate of **$0.61823**, not an actual billing
measurement. No timeout, additional GPU job or automatic rerun occurred.

All 81 source, three configuration and 965 dataset hashes matched the launch
manifest after download. The original artifacts are under the result path
below; a local `local-audit.json` separately records completeness and limits.

All full-model cuDNN/FA3 forward and backward operator counts matched their
requested routes. The three sparse replay profiles also verified the intended
operators (cuDNN padded 1+1, cuDNN segmented AB 2+2, FA3 AB 1+1), but **all three
contained zero CUDA kernel events** despite requesting them. They retain
`evidence_level=backend_operators_only`; the cause of missing CUDA events has
not been isolated. Numerical coverage is complete, while the stronger requested
profiler evidence is not. The original failed status is preserved.

The experiment compares all trainable FLUX parameter gradients for the same
two real examples. Earlier Math-SDPA forward comparisons already measured
0.8098% and 0.5612% differences between padded batching and unpadded singleton
execution; they did not measure accumulated full-model gradients. The present
test separates padding, batching, model packing and packed sample order.

## Fixed comparison

Use cached base FLUX, the historical short A/B examples (468/406 text tokens,
900 image tokens each), seed 167, and padded text length 487. Prepare frozen
NVARC once with cuDNN and the FP32 VAE with Math SDPA. Reuse exact conditioning,
positions, timesteps, noise and target velocity throughout.

| Backend | Layouts |
| --- | --- |
| cuDNN | unpadded singletons; padded singletons; padded batch; packed AB; packed BA |
| FA3 | unpadded original-model singletons; packed AB; packed BA |

Each layout is repeated unchanged, for 16 logical batches and 22 full
forward/backward calls. The cuDNN packed path calls attention separately for
each segment inside the packed model; it is not a native ragged backend.
FA3 singletons change only the attention kernel boundary of the original model.

Every example contributes `L_i / 48`. Singleton gradients accumulate without
clearing between examples; paired mean loss is weighted by `2/48`. Parameters
and persistent gradients remain FP32, forward compute uses BF16 autocast and
backward runs outside autocast. No optimizer update, clipping or checkpointing
is performed. Every native padded call, including singleton calls, applies the
actual-position key mask. Padding input gradients must be zero.

Comparisons report relative L2, cosine, angle, norm ratio and squared-error
contribution for all parameters and components. AB/BA is compared after
restoring sample identity, and each unchanged repeat has its own reference.
At most two full CPU gradient snapshots remain resident simultaneously.

Passive hooks capture outputs and true-loss output gradients at `double.0`,
`double.4`, `single.0`, `single.19` and the first text MLP output projection.
Per-token absolute RMS/max errors and top positions are retained. First/last
32 tokens, both sides of the text/image transition and the sequence interior
are compared. Boundary error energy is normalized by token count, and relative
token errors use the whole-sequence reference RMS to avoid tiny local denominators.

Additional fixed-Q/K/V replays use the real weighted cuDNN singleton loss's
upstream gradient at four blocks. cuDNN singletons, padded batch, segmented AB
and BA, and FA3 singletons/AB/BA each run twice: 72 forward/VJP evaluations.
Padding is inserted between valid text and image, with zero padded upstream.
These local derivatives are distinct from full parameter gradients.

## Resource and validation limits

The user has $2.75 remaining; planned spending is at most $2, preserving $0.75.
Reuse image `im-R5CncDboPfMQryWnq0UZn4` with current source mounted without
image builds, existing checkpoints and the existing training fixture only.
One H200, eight CPU cores and 64 GiB host memory have explicit resource limits.
Worker timeout is 840 seconds, function timeout 960, startup timeout 120, and
the local lifecycle watchdog is 1,200 seconds including cancellation reserve.
No requested retries, premium placement or additional jobs are allowed.

At the checked [Modal resource prices](https://modal.com/pricing), this is
$0.00150788 per second, or $1.809456 for 20 minutes. This estimate is not an
account billing measurement. Progress is written locally on the Volume;
explicit synchronization occurs only once at exit, with a 60-second limit,
to avoid the large synchronous-commit overhead in the previous diagnostic.

Local checks cover analytical loss weighting, real small-model gradients,
sample/order restoration, padding isolation, capture transparency, reference
memory lifetime, deadline cleanup and result completeness. Remote manifests
record source/config/data/checkpoint identities, dispatch and actual timings.
Unexpected dispatch, missing gradients, nonfinite values or identity errors
invalidate a case. Numerical differences are reported without inventing new
acceptance thresholds. This diagnostic does not establish optimizer-update,
learning or training-throughput equivalence.

Entry points: `scripts/diagnostics/probe_flux_backward_layouts.py` (offline
`--plan`) and `scripts/cloud/probe_flux_backward_layouts_modal.py`. Configuration
is `configs/diagnostics/flux_backward_layouts.json`; results are written to
`diffusion-arc-models:results/flux-backward-layouts/<run-id>/`.
