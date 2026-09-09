# Attention backends across input conditions

**Status: completed on H200 on 2026-09-08.** All 13 conditions, 104 transformer
forwards, and 312 isolated attention forward/backward pairs completed in
**261.65 seconds**, without a retry. Local validation passed all **472 repository tests**. This diagnostic tests
whether attention errors depend on native text conditioning, adaptation to an
ARC task, or unfamiliar ARC examples. It combines full-transformer forward
comparisons with isolated attention forward/backward comparisons.

The measured backend differences depend on the conditioning and checkpoint.
ARC adaptation reduced final-output differences on both the trained and new
examples. It also reduced the largest isolated dQ errors.
Neither discrepancy disappeared, and this small comparison does not establish
training equivalence.

## Measured results

The table reports **relative L2 error of final image velocity**, comparing each
of cuDNN SDPA, FA2, FA3, and FA4 with Diffusers BF16 Math SDPA. Ranges cover all
four backends and, for ARC, both timesteps. They are not image-quality scores.

| Conditioning and FLUX checkpoint | Backend difference |
| --- | ---: |
| Original Qwen image prompt, base FLUX | 0.423–0.432% |
| Real NVARC ARC examples, base FLUX | 0.485–0.901% |
| Trained ARC view, overfit FLUX | 0.303–0.447% |
| Unseen same-task view, overfit FLUX | 0.305–0.455% |
| Unrelated training task, overfit FLUX | 0.301–0.455% |

This reduction is also present in absolute error. On the same unrelated task
at `t=0.5`, FA3 versus BF16 Math falls from **0.01066 to 0.00514 absolute RMS**
after adaptation; reference velocity RMS remains similar, **1.18245 versus
1.15933**. The change is therefore not merely a larger denominator. At `t=1`,
overfit forward differences are approximately 0.30–0.31%; at `t=0.5`, they are
0.44–0.45%. The trained view has no clear advantage over the two new views.
Maximum relative flow-loss differences across these four backends are 0.661%
for base ARC and 0.218% for overfit ARC.

Against the **full FP32 Diffusers forward**, all five BF16 variants remain
0.732–0.844% away on overfit ARC. Backend agreement alone therefore does not
remove the broader BF16 arithmetic difference.

The isolated tests give a different scale. All values below are maximum
relative L2 percentages across the 52 captured Q/K/V triplets, compared with
FP32 Math operating on exactly the same BF16 input values promoted to FP32.

| Backend | Attention output | dQ | dK | dV |
| --- | ---: | ---: | ---: | ---: |
| BF16 Math SDPA | 0.1675% | 0.1670% | 0.1753% | 0.1692% |
| cuDNN SDPA | 0.1725% | 2.2514% | 0.7805% | 0.2421% |
| FA2 | 0.1717% | 2.2502% | 0.7960% | 0.2420% |
| FA3 | 0.1722% | 2.2487% | 0.7862% | 0.2421% |
| FA4 | 0.1722% | 2.2499% | 0.7884% | 0.2421% |

The largest dQ difference occurs at `single.19` for base FLUX on the unseen
same-task example at `t=1`. FA3's absolute error RMS is **0.001726**, reference
RMS **0.076763**, and cosine **0.999747**. The maximum across cuDNN/FA2/FA3/FA4
is **1.428%** for the native prompt and **1.022%** for overfit ARC. Thus valid
conditioning changes the isolated derivative errors too, but does not make
them negligible or establish correctness of full-model training gradients.
The similar cuDNN and FlashAttention error norms do not identify an FA3-specific
defect. They also do not prove direct pairwise gradient agreement: each backend
is compared against Math, not against the other fused backends.
Relative to BF16 Math, their isolated output differences stay below
0.124%, while complete forwards differ by up to 0.901%; accumulated changes
through the transformer matter alongside the local kernel discrepancy.

The improvement is component-dependent. Overfit reduces dQ error in 92 of 96
matched comparisons, but isolated attention-output relative error is slightly
larger in 75 of 96; its scale remains approximately 0.17% against FP32 Math.
dK and dV do not improve consistently either. Smaller final-output errors
therefore do not mean that every kernel calculation became more accurate.

The separate BFL/Diffusers FP32 control agrees closely: maximum observed state
error **0.000386%**, maximum final-velocity error **0.000061%**. With BF16, their
velocity difference is **0.949%** for the native prompt, **0.616–1.074%** for
base ARC, and **0.325–0.495%** for overfit ARC. This implementation comparison
must remain separate from the fixed-Q/K/V backend test. There is no evidence
here requiring the higher-precision native Diffusers normalization/RoPE path
to be replaced with BFL's BF16 rounding pattern.

All six profiled backends passed both operator and CUDA-kernel dispatch checks,
including backward. Source, configuration, fixture and input identities, the
overfit checkpoint manifest, completeness, and numerical screens were checked
again after downloading the artifacts. The screens flag 109 full-model state
comparisons and 24 isolated derivative comparisons; execution completion is
independent of these predeclared numerical flags.

The smallest further kernel check is the late-block dQ case directly between
cuDNN and FA3, with an overfit control and several identical upstream gradients.
A subsequent training check should compare full loss gradients and one
parameter update on fixed real examples under an attention-only backend switch.
Keeping the existing layout would separate that question from the earlier
packing changes. These results do not justify expanding packing or changing
production precision settings yet.

### Run evidence

- [Completed Modal run](https://modal.com/apps/joshua-kauling01/diffusion-arc/ap-821jcX2tjply6pj4GLzcdT).
- Volume `diffusion-arc-models`, environment `diffusion-arc`, directory
  `results/flux-backend-conditions/20260908T210359174247Z/`.
- `summary.json` contains every numerical comparison, fingerprints, profiler
  evidence and checkpoint hashes; `manifest.json` pins code/configuration/input
  hashes and the command. `result.json`, `report.md`, and `worker.log` retain
  incremental evidence and a readable report.
- H200, PyTorch 2.13.0, CUDA 13.0, cuDNN 9.20, Diffusers 0.39.0,
  Transformers 4.56.1; backend versions are listed below and in the artifacts.

## Fixed comparison matrix

Every exposure uses 900 image tokens, corresponding to the normal 480×480 image
and 30×30 packed latent canvas. The ARC exposures use the identical prepared
text states, noisy latents, positions, and timesteps for both checkpoints.

| Conditioning | FLUX checkpoint | Timesteps | Conditions |
| --- | --- | --- | ---: |
| Original Qwen encoder, ordinary image prompt | Base | 1 | 1 |
| NVARC, trained augmentation of `0d3d703e` | Base and step-4096 overfit | 1, 0.5 | 4 |
| NVARC, unseen augmentation of `0d3d703e` | Base and step-4096 overfit | 1, 0.5 | 4 |
| NVARC, original view of unrelated training task `3c9b0459` | Base and step-4096 overfit | 1, 0.5 | 4 |

“Base” means pretrained FLUX before ARC adaptation. The native prompt is:

> A red ceramic mug on a wooden table beside a window, soft natural daylight, realistic photograph.

Each of the **13 conditions** runs eight full-transformer variants: original
BFL with FP32 Math SDPA and BF16 Math SDPA; Diffusers with FP32 Math SDPA, BF16
Math SDPA, cuDNN SDPA, FA2, FA3, and FA4. This gives **104 full forwards**.

Full-model observations include text/image states after `double.0`, `double.4`,
`single.0`, and `single.19`, followed by final image velocity. The BFL and
Diffusers implementations share mapped checkpoint weights through the
[independent BFL reference](flux-bfl-reference.md). Their native projection,
normalization, and RoPE arithmetic remains intact.

## Inputs and provenance

The [input fixture](../../configs/diagnostics/flux_backend_inputs.json) embeds
only the two selected ARC training tasks, their source identities, augmentation
membership, and the successful checkpoint manifest. Its historical source is
the [step-4096 report](../../experiments/results/single-task-augmentation-generalization-overfitting/20260827T011822050154Z/report.json).
Source-file hashes, report SHA256, source commit, and complete train/holdout key
sets make the selection auditable without uploading the full report.

The trained view is `0d3d703e_ID_0458276319_0,1,3,2,4`. Its completed cumulative
exposure audit records **64 training contacts**, including 32 during the final
continuation. The unseen view is `0d3d703e_ID_8106794325_0,1,2,3,4`; it belongs
to the disjoint same-task holdout pool and received no updates. The unrelated
task is explicitly included in the existing ARC training whitelist. No
held-out evaluation tasks enter this diagnostic.

ARC preparation uses the exact final target-free token prefix, NVARC layers
9/18/27, the historical `greedy_rgb_maximin_v1` palette, top-left alignment,
white padding, and the frozen FP32 VAE posterior mode. Runtime checks compare
the seen/unseen token hashes and 138-token prefix lengths with the historical
report. NVARC runs in BF16 and its outputs are retained as FP32 for reuse.

**Historical hidden-state values are not replayed:** the same frozen NVARC
checkpoint is re-encoded with physical batch size one. Historical preparation
used batches of 512 and 129. This preserves token content and preprocessing but
can introduce a different encoder rounding pattern. Both FLUX checkpoints
receive the same newly prepared tensors.

One seeded FP32 Gaussian tensor supplies noise for every condition. For ARC,
`x_t = (1 − t) × clean + t × noise`, with target velocity `noise − clean`.
The native prompt uses pure noise at `t=1` and has no paired photograph or
flow-loss target. It calls the installed FLUX pipeline's original Qwen helper,
retaining **all 512 text positions**, including padding-derived states, as
unmasked FLUX conditioning. ARC prefixes contain only their actual tokens.

## Isolating the attention kernel

The Diffusers BF16 Math forward captures Q/K/V after normalization and RoPE at
the four declared blocks. All three tensors are converted to the BF16 V dtype
once at the attention boundary. Each exposure therefore contributes four
captured inputs: **52 Q/K/V triplets in total**.

Each triplet is replayed through FP32 Math SDPA, BF16 Math SDPA, cuDNN SDPA,
FA2, FA3, and FA4: **312 isolated forward/backward pairs**. The FP32 reference
promotes the captured BF16 values exactly; it does not recompute projections,
normalization, or RoPE. Every backend receives the same numerical Q/K/V values,
noncausal attention, zero dropout, and head-dimension scaling. FlashAttention
uses one varlen segment per singleton, with no cross-example packing.

One seeded BF16 Gaussian upstream gradient is shared by every backend for a
given captured input, and promoted unchanged for FP32. The comparison measures
the attention output and `dQ`, `dK`, `dV` from this vector–Jacobian product.
These are **local attention derivatives**, not gradients of the model loss or
transformer parameters. A representative replay profiles every backend to
verify dispatch and backward execution; artifacts state separately whether
CUDA kernel tracing was available.

Full forwards keep persistent FLUX parameters in FP32 and use either FP32 or
BF16 autocast. Isolated replays disable autocast and set the input dtype
explicitly. TF32 is disabled. Input fingerprints guard reuse and mutation.
Metrics include relative L2, absolute RMS and maximum error, reference and
actual norms, and cosine similarity so small reference norms remain visible.

The [configuration](../../configs/diagnostics/flux_backend_conditions.json)
fixes numerical screens before execution: 0.01% for FP32 model comparisons,
2% for BF16 model comparisons, 1% for isolated attention outputs, and 2% for
isolated attention derivatives. These are diagnostic screens, not demonstrated
training tolerances.

## Execution and interpretation

The [launcher](../../scripts/cloud/probe_flux_backend_conditions_modal.py) uses
one H200, eight CPU cores, and 64 GiB host memory, with a 1,800-second worker
limit, a 1,920-second function timeout, one container, and zero configured
retries. All model components come from the existing `diffusion-arc-models`
volume. Actual checkpoint files are hashed; the overfit checkpoint must match
its saved manifest. Source/configuration/input hashes and backend versions are
retained with incremental results under `results/flux-backend-conditions/`.

The cached image pins FA2 `2.8.3.post1`, FA3 commit
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`, and FA4 `4.0.0b29`.
No optimizer updates, model backward passes, or gradient checkpointing occur.
Elapsed times describe diagnostic execution and are not throughput benchmarks.

These measurements do not explain the earlier **18.36% packed-SDPA and 20.17%
packed-FA3 parameter-gradient errors**
in the [packing diagnostic](flux-packing.md): those changed execution layout
and measured a different derivative through the full model.

The native prompt, two same-task views, and one unrelated task provide bounded
controls, not a representative distribution study. A subsequent training
decision would still require matching loss gradients and parameter updates on
the chosen execution path.
