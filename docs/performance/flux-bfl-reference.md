# Independent BFL transformer reference

**Status: all ten real Klein 4B forwards completed on H200 on 2026-09-08.
FP32 passed every declared threshold. BF16 passed final velocity and loss
thresholds but failed the internal activation screen, with relative L2 errors
up to 14.37%. These differences already occur without FA3 or full packing.**

Local validation on 2026-09-08 passed all **434 repository tests**, including
28 new reference, mapping, evidence-validation and launcher tests. The original
BFL and Diffusers small-model FP32 forwards agree across the captured
boundaries, and CPU profiler tests verify the real Math SDPA route. A separate
meta-device preflight checked the full Klein 4B configuration: all 169
Diffusers tensors map to all 149 BFL tensors, with matching shapes and
3,875,544,576 parameter elements, without allocating full model weights.

The earlier [packing comparisons](flux-packing.md) use the unchanged Diffusers
forward as their baseline. They test our modifications, but do not establish
agreement with the independent Black Forest Labs implementation. This probe
adds that missing reference before further attention or packing changes.

## Reference chain

The same two exposures are evaluated in three forms:

1. Original BFL singletons, without text padding.
2. Unchanged Diffusers singletons with the same weights and input tensors.
3. The existing Diffusers mask path with both examples in a padded batch.

The BFL-to-Diffusers comparison tests the implementation boundary. The
singleton-to-batch comparison tests our padding extension separately. Existing
FA3 and full-packing diagnoses remain subsequent comparisons against Diffusers;
this probe never installs their processors or calls their kernels.

## Source and weights

The BFL reference is pinned to commit
[`50fe5162777813d869182b139e83b10743caef15`](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15).
Its transformer and Apache 2.0 license are copied byte for byte under
[`scripts/diagnostics/_third_party/bfl_flux2`](../../scripts/diagnostics/_third_party/bfl_flux2/UPSTREAM.md).
Independent hash constants verify the source and provenance before import.
Production files under `src/` remain unchanged.

The adapter constructs the BFL model on the meta device and maps the existing
Diffusers checkpoint into it. Every source and target parameter must be covered
exactly, with matching shapes and total element counts. Double-stream Q/K/V
weights are concatenated in their original order. The final modulation changes
from Diffusers' scale/shift order to BFL's shift/scale order. No weight is cast.
Tests also round-trip the mapping through Diffusers' official conversion code.

Identity mappings share storage to avoid duplicating the entire 4B model.
The BFL Parameter objects are separate and frozen; both models remain read-only
throughout this forward-only probe. The report records the complete mapping and
the bytes shared or copied. This tests the model computation and adapter using
one common checkpoint; it does not independently prove that the published
Diffusers checkpoint conversion matches a separately downloaded raw BFL file.

## Identical inputs and precision

Input preparation reuses the existing short B4 benchmark group. The full group
and its flow draws are prepared once, then targets 15 and 12 are sliced as A
and B, preserving the previous diagnostic's seed and exposure selection:

| Example | Valid text tokens | Image tokens |
| --- | ---: | ---: |
| A | 468 | 900 |
| B | 406 | 900 |

The masked pair retains the recorded padded text length of 487. Singletons
have exactly their valid text length. No text encoder, VAE encoding, timestep
or noise draw is repeated for a model comparison. Previously prepared floating
tensors are promoted identically to FP32; this does not restore precision lost
in the original frozen NVARC/VAE preparation. The exact assembled call tensors
are fingerprinted before and after each forward and reused across precisions.

Native Diffusers broadcasts the first sample's position IDs across a batch.
The probe therefore validates the shared position system before batching;
it rejects incompatible per-example coordinates instead of changing BFL's
positions to make a comparison pass.

Both models retain FP32 parameters. The first condition disables autocast;
the second enables BF16 autocast while preserving each implementation's native
normalization and RoPE dtype transitions. TF32 is disabled and Math SDPA is
forced in both conditions. This controls the attention backend while comparing
implementations; the BF16 condition is not the production cuDNN training path.
BFL's original calls for empty reference-token sequences remain intact.

## Measurements and screening

Each precision executes BFL A, BFL B, Diffusers A, Diffusers B, and the masked
Diffusers A+B batch: **ten complete forwards in total**. No optimizer, backward,
gradient allocation or weight update is requested.

Thirty common boundaries are observed: the timestep embedding, image and text
projections, five Double blocks, twenty Single blocks, the final modulated
normalization, and velocity. Valid text and image streams are compared
separately. Padding-query outputs are excluded because they are not outputs of
the unpadded reference. BFL calls its block methods directly, so passive method
wrappers capture their returned tensors; all wrappers and module hooks are
restored even on failure. They never replace BFL arithmetic.

The configuration declares these screening thresholds before execution:

| Condition | Relative L2 at every captured output stream | Relative flow-loss error |
| --- | ---: | ---: |
| FP32 | 0.01% | 0.01% |
| BF16 | 2% | 1% |

These are diagnostic thresholds, not established training-acceptance criteria.
Every comparison records the first nonidentical boundary, first boundary above
tolerance, maximum boundary error, final velocity error and per-example loss.
A small final error cannot hide an intermediate boundary failure. The result
validator checks complete case/example/stream coverage and recalculates the
summary from the recorded metrics. CUDA profiles must show the forced Math
SDPA calls and actual CUDA kernels, with no FA3 or backward execution.

Execution completion and numerical success are separate fields. A completed
experiment may report failed numerical screening; a timeout or execution error
cannot retain a successful overall numerical result. Finished cases and the
current phase are persisted incrementally.

## H200 results

Run `20260908T060840854654Z` completed in **74.57 seconds** with worker exit
code 0 and no timeout. Its separate numerical result is **failed**. All ten
forwards covered all thirty boundaries, producing 660 stream comparisons and
twelve per-example loss comparisons. Every forward executed exactly 25 Math
SDPA operations; no FA3, cuDNN attention or backward kernel appeared.

**FP32 passed all comparisons**, including masked batching. The largest
internal relative L2 error was **0.001046%**, below the 0.01% threshold.
Diffusers singleton velocity errors against BFL were **0.000097% for A** and
**0.000130% for B**. The largest final velocity error across all FP32
comparisons was 0.000251%.

BF16 results below use relative errors in percent. The maximum stream error
includes intermediate text and image states, not just the final prediction.

| Candidate / reference | Example | Final velocity | Maximum stream | Flow loss |
| --- | --- | ---: | ---: | ---: |
| Diffusers singleton / BFL | A | 1.2415% | 14.3740% | 0.1418% |
| Diffusers singleton / BFL | B | 0.9607% | 14.0597% | 0.8942% |
| Diffusers masked pair / BFL | A | 1.2751% | 14.2622% | 0.6328% |
| Diffusers masked pair / BFL | B | 0.9205% | 14.0808% | 0.8143% |
| Diffusers masked pair / singleton | A | 0.8098% | 2.2409% | 0.4904% |
| Diffusers masked pair / singleton | B | 0.5612% | 2.7640% | 0.0791% |

All six BF16 comparisons pass their final velocity and loss thresholds, but
fail at intermediate boundaries. BFL and Diffusers singletons first differ
at `double.0/text`; their first 2% violations are `double.4/text` for A and
`double.2/text` for B. Maximum text errors occur at `single.16` and `single.18`.
Internal image streams also exceed 2%, reaching **3.3331% for A** and
**2.6604% for B**. For masked Diffusers batching versus its own singletons,
only late text streams exceed 2%; every image stream remains below it.

The completed run's artifacts are in
`results/flux-bfl/20260908T060840854654Z/` on the `diffusion-arc-models` Modal
Volume, environment `diffusion-arc`. The
[Modal run](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-rbv4nN5imqW41Oma7Yibwe)
and its manifest record the command, source/config/data hashes, shared input
fingerprints, flow seed 167, and environment: PyTorch 2.13.0, CUDA 13.0,
Diffusers 0.39.0, one H200, FP32 parameters, TF32 disabled, and no activation
checkpointing. The assembled input digest is
`935cc691ec2e8f4b5500a12a991dcd4edcec88a2cb437fbd523cf59402a88d9c`.

Two execution issues were repaired before completion. First, Modal relocates
the launcher file, so container import needs a remote repository root rather
than local parent-directory indexing. The failed startup app was stopped
before model loading. Second, run `20260908T055340081940Z` stopped after
113.44 seconds because its profiler check assumed one Diffusers SDPA API event
per block. BF16 autocast redispatch produced two nested API events while still
executing exactly one Math SDPA operation. The corrected check permits wrapper
redispatch but still requires exactly 25 Math operations and rejects other
attention backends. CPU autocast and relocated-launcher regression tests cover
both repairs. The completed rerun reproduced all seven previously persisted
case reports exactly; this establishes identical saved metrics, not bitwise
identity of unsaved raw tensors. The partial run's artifacts remain under its
own run ID on the same Volume.

## Running

Inspect source identity and the exact plan locally without loading checkpoints,
contacting Modal or allocating a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B \
  scripts/diagnostics/probe_flux_bfl.py --plan
```

After explicit authorization for the new paid run:

```bash
.venv/bin/modal run --detach --env diffusion-arc \
  scripts/cloud/probe_flux_bfl_modal.py
```

The launcher reuses the pinned packing image and existing model Volume, with
one H200, eight CPUs and 64 GiB host memory. The whole worker has at most
**480 seconds**; the Modal function has a **600-second limit**, one container
and no retries. Loading, input preparation, conversion and profiling count
against the worker limit. A CPU image build, if needed, precedes that limit.
The runtime uses offline checkpoint loading and only the existing B0 data
bundle; it does not download models or datasets.

Artifacts are saved to `results/flux-bfl/<run-id>/` on `diffusion-arc-models`:
source/config/data hashes and the plan in `manifest.json`, incremental cases in
`result.json`, final metrics and mapping in `summary.json`, plus `worker.log`
and a compact `report.md`. No tensor snapshots or weights are committed.

## Interpretation and next step

For these two exposures, FP32 agreement supports the weight mapping,
transformer computation and padding extension. Native BF16 implementations
already differ before adding FA3 or packing. This does not identify the cause
of the earlier 18–20% gradient discrepancies: activation errors and gradient
errors measure different quantities, and this probe uses Math SDPA rather than
the earlier cuDNN/FA3 backends. It also screens every internal stream, a stricter
criterion than checking final velocity alone.

The [precision follow-up](flux-bfl-precision.md) now isolates Q/K normalization
and RoPE in small CPU Double blocks. BFL norm plus BFL RoPE coefficients makes
every captured boundary bitwise equal in three seeds; an extra final dtype
cast alone does not reproduce BFL's normalization. CPU and CUDA autocast have
different RMSNorm rules, so the same intervention still needs the real first-
block inputs on H200 before attributing the full-model error. A verified BFL
precision contract could guide both SDPA and attention-only FA3 processors.
The smaller singleton-to-masked-batch differences remain a separate control;
numerical disagreement alone does not prove a mask bug.

Once this reference chain is understood, the attention-only FA3 comparison can
reuse its validated input and model contracts. Full packing is still optional:
its extra changes to modulation, projections and MLP batch shapes should be
justified by measured benefit. Two exposures do not establish general model
equivalence. This probe alone establishes no gradient or
parameter-update equivalence, checkpointing correctness, throughput advantage,
or ARC learning result.
