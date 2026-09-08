# NVARC packing and attention backends

**Decision: use frozen BF16 NVARC with FlashAttention-3 Varlen, one packed
16-source forward, and no padding or alignment.** This is the selected NVARC
component for B3. Full-pipeline integration, memory and training agreement still
need validation; the measurements below load NVARC alone.

## Real-input results

Run `20260908T000222783174Z` reached the 600-second worker limit after **48 of
58 cases completed**. It retained 241 of 290 measured calls: 240 belong to
complete cases, and one belongs to the interrupted fourth-window FA3 case.
The first three windows are complete. In the fourth, FA2 and cuDNN completed;
FA3 and FA4 lack complete results for all five layouts. No replacement sweep
was requested: the complete windows suffice for the initial backend choice.

All rows replay original B0 benchmark sources, augmentation keys and consumed
prefixes. Times are medians in **milliseconds per complete 16-source window**,
using no padding:

| Window | Real tokens | FA2 | FA3 | FA4 | cuDNN ragged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 31,335 | 644.054 | 616.068 | 615.159 | 621.327 |
| Median | 41,104 | 858.791 | 812.759 | 814.820 | 820.284 |
| With longest source | 49,437 | 1,081.090 | 1,000.706 | 1,012.790 | 1,017.356 |
| Largest total | 53,367 | 1,145.017 | Incomplete | Incomplete | 1,082.742 |

FA3 takes 4.3–7.4% less time than FA2 across the three complete windows. FA4
is effectively tied on the shorter windows and 1.2% slower on the window with
the longest source. cuDNN ragged is within 0.9–1.7% of FA3. These are small-sample
observations, not a claim that FA3 universally outperforms the other backends.

## Alignment

The tested layouts are unpadded packing, total length aligned to 8/32, and each
source aligned to 8/32. Total alignment adds an isolated tail PAD segment;
source alignment appends causal PAD positions before the next source. Position
IDs reset per segment, and PAD outputs are discarded. An 8,400-token real
source occupies 8,416 physical positions when aligned to 32.

Changes below compare FA3 medians with unpadded FA3; negative values are faster.

| Alignment | Short | Median | With longest source |
| --- | ---: | ---: | ---: |
| Total / 8 | -0.38% | +0.00% | +0.21% |
| Total / 32 | +1.80% | -0.43% | +0.21% |
| Each source / 8 | -0.63% | -0.31% | +1.40% |
| Each source / 32 | +0.68% | +1.34% | -0.10% |

The median window is already divisible by 8. Total / 8 and total / 32 produce
the same layout for the longest-source window. These aliases were measured
once. Every observed alignment benefit is below 0.7%, with overlapping sample
ranges against the corresponding unpadded case. There is no consistent benefit
to justify additional alignment logic.

## Comparison with the previous implementation

The earlier run `20260907T221142782547Z` compared SDPA/cuDNN with B2 groups,
FA2 with the same groups, and FA2 with padding-free packing. It completed all
nine cases and 45 measured calls in 170.75 seconds on H200. Unlike the new run,
it used seeded synthetic ARC token contents at the same source lengths.

| Window | Previous SDPA + B2 | Previous FA2 + B2 | Previous FA2 packed | New FA3 packed | FA3 change vs previous SDPA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 1,258.165 | 925.603 | 646.201 | 616.068 | -51.0% |
| Median | 1,473.154 | 1,084.858 | 852.379 | 812.759 | -44.8% |
| With longest source | 1,812.240 | 1,361.030 | 1,071.664 | 1,000.706 | -44.8% |

This suggests 44.8–51.0% less isolated NVARC time than the old path. It is not a
controlled comparison of identical token contents. FA2 packed times agree
within 1% between runs, supporting the scale of the estimate. Similar gains
with cuDNN ragged show that the old comparison cannot be read as a pure cuDNN
versus FlashAttention kernel comparison: physical layout, unpadding and model
call count also changed. B2 required 3/4/5 model forwards for these windows;
packing requires one. FLUX and VAE time are outside these measurements.

## Memory and numerical checks

For unpadded cases, all tested backends have the same peak allocated memory to
three decimal places: **14.199, 16.503, 18.472 and 19.398 GiB**, respectively.
The fourth value is observed only for FA2 and cuDNN. Model weights are included;
FLUX, VAE, optimizer states and preparation buffers are absent. The old SDPA/B2
peaks were 11.830/11.982/10.893 GiB for the first three windows. One packed
forward retains more intermediate states simultaneously, despite doing less
padding work. No full-pipeline capacity claim follows from these numbers.

The new run compares selected real-position hidden states with an SDPA/B2
reference computed from the same real inputs:

| Window | FA2 relative L2 | FA3 relative L2 | FA4 relative L2 | FA3 max absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.4877% | 0.4961% | 0.4704% | 18.50 |
| Median | 0.5565% | 0.5514% | 0.5690% | 11.00 |
| With longest source | 0.5637% | 0.5462% | 0.5426% | 8.50 |

cuDNN ragged has zero measured difference from the reference in all four
windows. Alignment variants have identical reported difference statistics
within each FlashAttention backend/window; this alone does not prove equality
between their tensors. The previous synthetic-input FA2 comparison yielded
0.3344–0.4062% relative L2 error and 2.0–2.5 maximum absolute error. Those
numerical results must not be substituted for the new real-input checks.

All complete cases passed shape, dtype, finite-value and frozen-gradient checks.
All four backends passed small fixed-shape tests that perturb one source and
PAD values: other-source outputs and real outputs, respectively, remained
bitwise equal. No numerical acceptance threshold was applied to the SDPA
comparison; hidden-state differences do not establish loss/gradient agreement
or training equivalence.

## Protocol and provenance

- Frozen NVARC/Qwen3-4B, BF16, H200, seed 167; full causal-LM forward, including
  the output head. Hidden-state layers 9/18/27 are concatenated and restored to
  logical source order. Sources are trimmed to their final required prefix.
- Original B0 repetition-1 steps 35/37/31/34, with source keys and lengths
  checked on replay. The benchmark bundle contains training data only; this
  diagnostic measures execution and supplies no held-out quality result.
- Each case records its first invocation separately, then two warm-ups and
  five measured calls. Backend/layout order rotates between windows. Exact
  duplicate physical layouts are deduplicated, leaving 58 cases.
- Synchronized wall time includes input assembly, transfer, model execution
  and compact output extraction. Loading, tokenization, first calls, warm-ups,
  backend profiles, comparisons and output destruction are outside medians.
  CUDA spans include dispatch gaps and are not kernel-only times.
- Profiles verify explicit backend bindings and CUDA execution. FA2/FA3 must
  additionally execute their own extension operators. Diagnostic adapters avoid
  Transformers' shared lazy FlashAttention import cache. No silent fallback.
- PyTorch 2.13.0, CUDA 13.0, cuDNN runtime 92000, Transformers 4.56.1, Python
  3.11.5; FA2 2.8.3.post1, FA3 3.0.0 at revision
  `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`, FA4 4.0.0b29, cuDNN Frontend 1.13.0.
  FA3 uses the BF16 forward build for SM90 and head dimension 128.

The worker elapsed time was 601.58 seconds including supervisor shutdown.
Saved measured calls total 206.63 seconds, and saved first calls total 50.66
seconds. The initial FA4 first call was 7.93 seconds, versus a 615 ms warm median;
first calls may include compilation or plan creation and reuse prior caches.
The remaining worker time includes setup, warm-ups and diagnostics. The
600-second limit was too short for the full sweep, not evidence of a kernel
failure. A preceding launch stopped before measurement because a diagnostic
check incorrectly required `model.config.pad_token_id`; the tokenizer's
validated PAD ID is 13 while the checkpoint config leaves it unset.

The [probe config](../../configs/diagnostics/nvarc_packing.json),
[diagnostic](../../scripts/diagnostics/probe_nvarc_packing.py) and
[Modal launcher](../../scripts/cloud/probe_nvarc_packing_modal.py) define the
replay. Results are retained under `results/nvarc-packing/<run_id>/` in the
`diffusion-arc-models` Volume, environment `diffusion-arc`. Manifests record
source/config/data hashes, commands and environment. Raw results and traces
remain outside Git. Source hashes for the new run matched the archived code
when this report was completed; checkpoint weight contents were not hashed.

Reproduction requires the recorded checkpoint/data and package versions:

```bash
.venv/bin/modal run --detach --env diffusion-arc scripts/cloud/probe_nvarc_packing_modal.py \
  --config configs/diagnostics/nvarc_packing.json
```

The launcher builds extensions on CPU before allocating its H200 worker. It
uses 8 CPUs, 64 GiB host RAM, a 600-second worker budget, a 900-second Modal
limit and no retries; partial results persist on timeout. This command would
start a paid run, which is not needed for the current decision.

Primary implementation references:

- [Transformers 4.56.1 attention integration](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/modeling_flash_attention_utils.py).
- [Pinned FA3 implementation](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/hopper/flash_attn_interface.py).
- [Pinned FA4 implementation](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/interface.py).
- [cuDNN Frontend SDPA bindings](https://github.com/NVIDIA/cudnn-frontend/blob/v1.13.0/python/pygraph/sdpa.cpp).
