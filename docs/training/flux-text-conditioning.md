# FLUX Text Conditioning

## Status and scope

`prepare_flux_text_inputs` is the implemented target-aware boundary between
compact frozen-NVARC hidden states and a physical FLUX text microbatch. It
derives only the conditioning prefix recorded for each target exposure, pads
the resulting prefixes to the longest one in that microbatch, and records which
positions are real.

This component does not execute NVARC or FLUX. It does not receive target
images, latents, noise, or flow timesteps. Image preparation and the complete
FLUX forward remain separate stages.

The implementation lives in
[`src/diffusion_arc/model/flux_text.py`](../../src/diffusion_arc/model/flux_text.py),
and the immutable `FluxTextInputs` contract lives in
[`src/diffusion_arc/model/contracts.py`](../../src/diffusion_arc/model/contracts.py).

## Inputs and target boundary

The preparation function consumes trusted internal objects produced by the
validated collation and frozen-NVARC paths:

```python
from diffusion_arc.model import prepare_flux_text_inputs


text_inputs = prepare_flux_text_inputs(
    packed_hidden_states,
    batch.target_sequence_indices,
    batch.target_prefix_ends,
)
```

| Input | Shape and dtype | Meaning |
| --- | --- | --- |
| `packed_hidden_states.hidden_states` | `[T, 7680]`, floating point | Every real NVARC source position in source-major order. |
| `packed_hidden_states.sequence_offsets` | `[S + 1]`, `torch.int64` | Source boundaries inside the compact hidden-state tensor. |
| `target_sequence_indices` | `[K]`, `torch.int64` | Source belonging to each target exposure. |
| `target_prefix_ends` | `[K]`, `torch.int64` | End-exclusive prefix length relative to that source. |

For target `k`, the exact source span is:

```python
source_start = sequence_offsets[target_sequence_indices[k]]
prefix = hidden_states[source_start : source_start + target_prefix_ends[k]]
```

No position after `target_prefix_ends[k]` is copied into that target's FLUX
conditioning. Future assistant outputs therefore cannot enter an earlier
target through the padding operation.

## Output contract

For prefix lengths `(p_0, ..., p_(K-1))`, let `N = max(p_k)`. The result is:

```python
FluxTextInputs(
    encoder_hidden_states=...,  # [K, N, 7680]
    text_position_ids=...,      # [K, N, 4]
    text_attention_mask=...,    # [K, N]
)
```

`encoder_hidden_states` preserves target-exposure order. Every real prefix is
left-aligned and every remaining position is filled with zero. The floating
dtype and device are inherited from the packed NVARC hidden states.

`text_attention_mask` is boolean. `True` marks a real prefix position and
`False` marks right padding.

`text_position_ids` is `torch.int64` on the hidden-state device and uses
FLUX's four-axis text coordinates:

```text
(t, h, w, l) = (0, 0, 0, l)
```

Coordinates are generated through the padded length. Coordinates belonging to
padding cannot affect a valid FLUX output when the corresponding keys remain
masked.

## Joint FLUX attention mask

FLUX attention operates on the combined text and image sequence. The text-only
mask is therefore not passed directly to the transformer. The helper
`build_flux_joint_attention_mask` appends one valid key for every image token:

```python
joint_mask = build_flux_joint_attention_mask(
    text_inputs.text_attention_mask,
    image_token_count=900,
)

prediction = flux(
    encoder_hidden_states=text_inputs.encoder_hidden_states,
    txt_ids=text_inputs.text_position_ids,
    joint_attention_kwargs={"attention_mask": joint_mask},
    # image state, image IDs, and timestep omitted here
)
```

For `M` image tokens, the joint mask has shape `[K, 1, 1, N + M]`:

```text
[real text | masked text padding | all image keys valid]
```

This is a key-validity mask. Padded text queries may still be computed by a
dense backend, but their states remain masked whenever they become keys in the
next block. The image output and valid text states therefore cannot depend on
them. Padding can still consume compute; semantic masking alone is not evidence
of a throughput improvement.

The mask route matches the public `joint_attention_kwargs` path in the pinned
[`diffusers` FLUX.2 implementation](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_flux2.py).
The same attention kwargs reach its double-stream and single-stream blocks.

## Local correctness evidence

The unit test constructs a random CPU FP32 `Flux2Transformer2DModel` with one
double-stream block, one single-stream block, the production conditioning width
of 7,680, and two prefix lengths. It compares:

1. one dynamically padded masked batch;
2. the same target exposures executed separately without padding;
3. the same masked batch after replacing padded context values with large
   values; and
4. FLUX parameter gradients from the batched and sequential losses.

The test tolerance is `3e-6` for outputs, losses, and gradients. Changing
masked padding values must leave the output exactly unchanged, while omitting
the mask must change the shorter sample's output. The test uses three tiny image
tokens to keep CPU execution small; mask construction is independent of the
production image-token count of 900.

Run it with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -B -m unittest tests.model.test_flux_text
```

## Remaining real-model validation

The local test establishes the mask semantics only for the native CPU FP32
attention route in `diffusers==0.39.0`. Before the path is used for a learning
experiment, an explicit Colab smoke or preflight must compare masked batched and
unpadded single-target execution with:

- the pinned FLUX.2-klein-base-4B checkpoint revision;
- BF16 on the selected CUDA device;
- the exact attention backend intended for training;
- realistic prefix lengths and 900 image tokens; and
- predefined output, loss, and gradient tolerances.

Attention backends differ in mask support. An unsupported backend must fail
explicitly; the training path must not silently remove the mask or change
conditioning semantics. Runtime and memory measurements are required before
describing dynamic padding as an efficiency improvement.
