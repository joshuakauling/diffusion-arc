# BFL and Diffusers precision differences

**Status: source inspection and synthetic CPU controls completed on 2026-09-08.
Q/K normalization is the first BF16 difference in all three small Double-block
cases. Using BFL's normalization and RoPE coefficients makes all 25 observed
boundaries bitwise equal in FP32 and BF16. This has not yet been repeated with
the real 4B weights and exposures on H200.**

The preceding [full-model reference comparison](flux-bfl-reference.md) passed
FP32 screening but found BF16 internal activation errors up to 14.37%. This
follow-up identifies concrete implementation differences and isolates their
effects locally; it does not assign a percentage of the H200 error to each one.

## What differs in the source

Our vendored transformer matches the installed Diffusers 0.39.0 transformer
except for rewritten imports. Its arithmetic has not been replaced by a new
attention mechanism. The [vendoring record](../../src/diffusion_arc/_third_party/diffusers_flux2/UPSTREAM.md)
documents this boundary. Our text-padding mask is supplied through attention
arguments; full packing and FA3 comparisons are separate diagnostic paths.

| Operation | Original BFL | Diffusers 0.39.0 |
| --- | --- | --- |
| Double-block Q/K/V projection | One joint linear projection per stream | Three separate projections by default; fusion is optional |
| Q/K normalization with BF16 inputs and FP32 weights | Normalize in FP32, round to BF16, multiply by learned FP32 scale, round to V's BF16 dtype | `torch.nn.RMSNorm`; CUDA autocast promotes its input to FP32, keeping normalization and scale multiplication in FP32 |
| RoPE frequency construction for our integer positions | FP32 frequency and trigonometric calculation | FP64 frequency and trigonometric calculation, then FP32 coefficients |
| Q/K entering RoPE under CUDA BF16 autocast | BF16 | FP32 |
| RoPE arithmetic | FP32, then cast to the incoming Q/K dtype | FP32, then cast to the incoming Q/K dtype |

The primary implementation sites are BFL's
[`DoubleStreamBlock`, `RMSNorm`, `QKNorm`, and RoPE](https://github.com/black-forest-labs/flux2/blob/50fe5162777813d869182b139e83b10743caef15/src/flux2/model.py)
and Diffusers'
[`Flux2AttnProcessor`, `Flux2Attention`, and `Flux2PosEmbed`](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_flux2.py).
Diffusers uses **PyTorch's** RMSNorm here, not the separate RMSNorm class in
Diffusers' normalization module.

For a normalized FP32 value `u` and learned FP32 scale `w`, BFL computes
`BF16(BF16(u) * w)`. Under CUDA autocast, Diffusers carries `u * w` in FP32
through RoPE until SDPA autocast converts Q/K to BF16. Adding only a final
BF16 cast after native RMSNorm cannot reproduce BFL's earlier rounding before
scale multiplication.

CUDA and CPU autocast differ here. PyTorch's
[FP32 operator list](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/autocast_mode.h)
contains `rms_norm`, and its
[autocast registrations](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/autocast_mode.cpp)
apply this rule on CUDA but not CPU. With our local CPU build, native RMSNorm
instead receives and returns BF16, while normalizing and multiplying by the
FP32 weight before its final cast. This matches the
[native composite implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/layer_norm.cpp).
CPU execution therefore cannot be presented as a measurement of CUDA's dtype
path.

In Single blocks, both implementations already combine QKV and the MLP input
projection. The normalization and RoPE differences still apply. Residual and
MLP formulas match mathematically. The final head's reversed scale/shift weight
order is already corrected by the reference adapter. BFL's helper named
`causal_attn_fn` implements noncausal joint text/image attention when reference
image tokens are absent, as in these experiments. None of these observations
requires replacing the transformer architecture.

## Local intervention results

The [precision probe](../../scripts/diagnostics/probe_flux_bfl_precision.py)
executes the actual first Double blocks with shared random FP32 weights and
once-drawn inputs. It retains head dimension 128, with two heads, 17 image
tokens and 13 text tokens. Learned norm scales are sampled from `[0.5, 1.5)`;
unit scales would hide the rounding before scale multiplication. Seeds are
37, 73 and 109. BF16 cases use common BF16 block inputs and modulation values;
FP32 cases use their common FP32 sources. Math SDPA is forced throughout.

Eight Diffusers variants independently change normalization, QKV fusion or
RoPE coefficients and selected combinations. Original BFL remains the
reference. All hooks and temporary method replacements are restored after
each call, including exceptions. Weight and input fingerprints verify that
the experiment leaves its shared state intact.

The 48 comparisons contain **1,200 observed boundary metrics**:

| Intervention | Measured result across three seeds |
| --- | --- |
| Native FP32 | First difference at rotated Q; maximum block relative L2 `1.81e-8` |
| Native BF16 | First difference at normalized image Q; maximum block error 0.04433% |
| BF16 with only an extra cast after native norm | Same result as native CPU BF16; the cast is a no-op here |
| BF16 with BFL norm | Much smaller errors; remaining RoPE differences reach 0.01198% at the block output |
| BFL norm and BFL RoPE coefficients | All 25 boundaries bitwise equal in both precisions for all three seeds |
| QKV fusion | No observed change in these CPU cases |

An additional **CPU replay of CUDA's RMSNorm dtype rule** starts from identical
BF16 Q/K/V and explicitly promotes only the native RMSNorm inputs to FP32.
Both routes use the same BFL RoPE coefficients and BF16 tensors at the Math
SDPA boundary. Across the three seeds, attention error is **0.4825–0.5329%**
for that FP32 norm rule and **0.4755–0.5059%** after adding a BF16 cast before
RoPE. The cast alone leaves a measurable difference. Reusing BFL norm is an
exact positive control, not a separate implementation validation.

This replay measures the effect of those casts using CPU arithmetic. It does
not emulate CUDA kernels, the full CUDA autocast policy, or actual H200 outputs.
The complete CPU block intervention above provides the independent BFL versus
Diffusers comparison.

## Reproduction and limits

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B \
  scripts/diagnostics/probe_flux_bfl_precision.py \
  --seeds 37 73 109 --output /private/tmp/flux-bfl-precision-20260908.json
```

The JSON artifact records all metrics, actual/reference dtypes, source hashes,
weight and input fingerprints, model configuration, seeds, package versions,
command and platform. The measured environment was macOS ARM64, PyTorch 2.13.0,
Diffusers 0.39.0 and one CPU thread. There are no model/data downloads or remote
jobs. Repository validation passed **438 tests**; four new tests cover passive
observation, intervention isolation, exception cleanup and dtype replay.

The result establishes a concrete rounding difference and localizes it in
these small controls. It does not show that BF16 training is unsuitable or
that BFL's rounding is more accurate. Matching BFL is a reference-contract
decision, distinct from proximity to an FP32 result or training quality.

The next useful H200 experiment is the same isolated intervention on the real
first-block inputs: native, BFL norm only, BFL coefficients only, and both,
with QKV fusion as a separate control. Only then should the matched contract
be checked through all blocks, gradients and a parameter update. No production
precision rule has been changed by this diagnosis.

For the attention-only FA3 adapter, this finding separates kernel compatibility
from numerical parity: casting Q/K/V to one dtype satisfies FA3's interface,
but does not undo earlier rounding choices. A verified normalization/RoPE
contract could be shared between SDPA and FA3 processors while retaining
FP32 master parameters. Full packing should remain a separate experiment,
because it also changes projection, modulation and MLP shapes.
