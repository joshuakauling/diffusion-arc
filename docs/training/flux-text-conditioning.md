# FLUX Text Conditioning

For the current end-to-end training path and documentation reading order, see
the [training system overview](README.md).

## Status and scope

This page describes the padded reference interface. B3 training instead copies
exact prefixes into unpadded segments; see the
[FLUX packing guide](flux-sequence-packing.md). Prefix boundaries, per-source
ownership and transactional consumption remain the same, while dense padding
masks are replaced by cumulative attention-segment boundaries in that path.

`pad_flux_text_prefixes` is the implemented pure tensor boundary between exact
frozen-NVARC prefix tensors and the padded text portion of a physical FLUX
microbatch. It pads the supplied prefixes to the longest one in that
microbatch and records which positions are real.

This model-side component does not select prefixes, own NVARC state, execute
NVARC or FLUX, or receive target images, latents, noise, or flow timesteps.
Prefix selection and lifetime management belong to the training-side
`build_flux_microbatch` operation.

The implementation lives in
[`src/diffusion_arc/model/flux_conditioning.py`](../../src/diffusion_arc/model/flux_conditioning.py),
and the immutable `FluxTextInputs` contract lives in
[`src/diffusion_arc/model/contracts.py`](../../src/diffusion_arc/model/contracts.py).

## Exact-prefix input

The padding function consumes a non-empty ordered sequence of exact prefix
tensors:

```python
from diffusion_arc.model import pad_flux_text_prefixes


text_inputs = pad_flux_text_prefixes(
    (
        first_sequence_hidden_states[:first_prefix_end],
        second_sequence_hidden_states[:second_prefix_end],
    )
)
```

Each input has shape `[p_k, 7680]` and uses the same floating dtype and device.
The function preserves input order. It trusts that the training-side caller
already selected the correct end-exclusive prefix:

```python
prefix = sequence_hidden_states[:prefix_end]
```

There is no packed-source offset arithmetic at this boundary. No position after
`prefix_end` is supplied or copied into that target's FLUX conditioning, so
future assistant outputs cannot enter an earlier target through padding.

## Output contract

For prefix lengths `(p_0, ..., p_(K-1))`, let `N = max(p_k)`. The result is:

```python
FluxTextInputs(
    encoder_hidden_states=...,  # [K, N, 7680]
    text_position_ids=...,      # [K, N, 4]
    text_attention_mask=...,    # [K, N]
)
```

`encoder_hidden_states` preserves the supplied prefix order. Every real prefix
is left-aligned and every remaining position is filled with zero. The floating
dtype and device are inherited from the first prefix.

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

## Training-side ownership and microbatch construction

`NvarcBatchRunner` returns one independent owning hidden-state tensor per ARC
source. The initial training preparation wraps each source tensor in one
`NvarcSequenceState`. All `FluxInput` objects produced from that source share
the owner and store a relative `prefix_end`; they do not copy the complete
hidden states.

Prefix ends are strictly increasing within one source, so the owner uses its
pending `prefix_end` values directly for lifetime bookkeeping.
`build_flux_microbatch(flux_inputs)` performs the complete deterministic
physical assembly:

1. validate that every prefix is unique within its source and still pending;
2. obtain each exact `hidden_states[:prefix_end]` view;
3. call `pad_flux_text_prefixes` to copy those prefixes into an independent
   padded `FluxTextInputs` tensor;
4. stack the clean target latents and generate the fixed image position IDs;
   and
5. only after all construction succeeds, commit all consumed prefix ends
   without another fallible operation.

A failed build leaves every source state pending and available for diagnosis or
retry. After the final dependent prefix is committed, the owner replaces its
hidden-state reference with `None`. The completed `FluxMicrobatch` retains only
its independently allocated padded text tensor and stacked image tensors, so it
does not keep the source owner or its original hidden-state storage alive.

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

The model-side unit test constructs a random CPU FP32
`Flux2Transformer2DModel` with one double-stream block, one single-stream block,
the production conditioning width of 7,680, and two prefix lengths. It compares:

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
  .venv/bin/python -B -m unittest tests.model.test_flux_conditioning
```

Training-side tests separately verify successful microbatch construction,
partial and final owner release, automatic owner collection after references
disappear, rejection of duplicate or stale prefixes, and no consumption when
tensor construction fails.

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
