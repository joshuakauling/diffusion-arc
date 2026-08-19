# NVARC Batch Runner

## Status and scope

`NvarcBatchRunner` is the implemented target-blind boundary between packed ARC
source tokens and frozen NVARC hidden states. It executes complete source
sequences only. Target indices, target-prefix boundaries, target images, VAE
state, and FLUX state are deliberately absent from its interface.

The local tests use a deterministic fake NVARC module. They validate packing,
padding, masking, layer selection, and frozen-module behavior without loading a
checkpoint. Real-checkpoint numerical validation remains a required Colab step.

The runner lives in
[`src/diffusion_arc/model/nvarc.py`](../../src/diffusion_arc/model/nvarc.py), and
its public tensor contracts live in
[`src/diffusion_arc/model/contracts.py`](../../src/diffusion_arc/model/contracts.py).

## Public interface

The runner is constructed with an already loaded NVARC-compatible PyTorch
module:

```python
from diffusion_arc.model import NvarcBatchRunner


nvarc_runner = NvarcBatchRunner(nvarc_model)
packed_hidden_states = nvarc_runner(
    input_ids=batch.input_ids,
    sequence_offsets=batch.sequence_offsets,
)
```

Its forward method accepts exactly two tensors:

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `input_ids` | `[T]`, `torch.int64` | `S` complete source sequences concatenated in source-major order. |
| `sequence_offsets` | `[S + 1]`, `torch.int64` | End-exclusive source boundaries, starting at zero and ending at `T`. |

It returns:

```python
PackedNvarcHiddenStates(
    hidden_states=...,      # [T, 7680]
    sequence_offsets=...,   # the original input tensor
)
```

The returned `sequence_offsets` field is the same tensor object passed to the
runner. Hidden states reside on the NVARC output device; offsets retain their
original device. This allows CPU-side boundaries from the collator to remain an
unchanged audit record while the compact hidden states remain on the model
device.

## Fixed NVARC contract

The initial architecture fixes the following values:

| Property | Value |
| --- | ---: |
| Vocabulary size | 16 |
| Padding token ID | 13 |
| Maximum source length | 8192 |
| Hidden width per selected state | 2560 |
| Selected hidden-state indices | `(9, 18, 27)` |
| Concatenated context width | 7680 |

These values are not runtime tuning parameters. Changing them would alter the
verified NVARC-to-FLUX interface and requires a separate checkpoint-contract
decision.

## Execution layout

For source lengths `(L_0, ..., L_(S-1))`, the runner performs:

```text
flat input_ids [T]
  -> split at sequence_offsets
  -> right-pad with token ID 13 to [S, max(L_s)]
  -> build bool attention_mask with true only at real positions
  -> one frozen NVARC forward with batch size S
  -> select hidden_states[9], [18], and [27]
  -> concatenate on the feature axis to [S, max(L_s), 7680]
  -> remove every padded position
  -> compact hidden_states [T, 7680]
```

The NVARC call explicitly requests hidden states, disables the key/value cache,
and uses dictionary-style output. The model executes under `torch.no_grad()`,
not `torch.inference_mode()`, so the resulting ordinary tensors can later be
consumed by the trainable FLUX path.

The packed token IDs are transferred internally to the device of NVARC's input
embeddings. Padding and the attention mask are created directly on that device.

## Frozen-state guarantee

Construction applies `requires_grad_(False)` to the complete NVARC module and
places both runner and model in evaluation mode. Calling `train(True)` on the
runner still leaves both in evaluation mode. The forward method also restores
the contained model to evaluation mode before every call in case it was changed
directly.

The runner rejects output that:

- omits hidden states or does not reach index 27;
- has a selected-state shape other than `[S, max_length, 2560]`;
- mixes selected-state dtypes or devices;
- is not floating point;
- requires gradients after packing; or
- contains non-finite values at real source positions.

## Target boundary

The runner never receives `target_sequence_indices` or `target_prefix_ends`.
Those tensors belong to a later target-aware operation:

```text
PackedNvarcHiddenStates
  + target_sequence_indices
  + target_prefix_ends
  -> exact per-target conditioning prefixes
```

Keeping this operation separate ensures that NVARC always processes each
collated complete source once and cannot acquire dependencies on target images
or FLUX execution policy.

## Remaining real-checkpoint validation

Local fake-model tests cannot establish the numerical behavior of the released
NVARC checkpoint or its CUDA attention backend. A later offline Colab test must
record the checkpoint revision, Transformers version, precision, attention
backend, and GPU, then compare:

1. each sequence run alone versus the same sequence in a right-padded batch;
2. real positions only, excluding every padded output;
3. the exact dtypes and shapes of hidden states 9, 18, and 27; and
4. full-sequence prefix states versus separately executed truncated prefixes.

No prefix reuse should be connected to FLUX until those comparisons meet a
predeclared numerical tolerance.
