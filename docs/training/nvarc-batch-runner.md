# NVARC Batch Runner

## Status and scope

`NvarcBatchRunner` is the implemented target-blind boundary between packed ARC
source tokens and frozen NVARC hidden states. It executes complete source
sequences only. Target indices, target-prefix boundaries, target images, VAE
state, and FLUX state are deliberately absent from its interface.

The local tests use a deterministic fake NVARC module. They validate packing,
padding, masking, layer selection, and frozen-module behavior without loading a
checkpoint. Stable real-checkpoint shapes, dtypes, hidden-state availability,
and numerical behavior remain a separate required smoke or preflight
experiment.

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

Its forward method accepts exactly two trusted tensors produced by the
collation path:

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
NVARC-to-FLUX interface and requires a separate checkpoint-contract decision
and smoke experiment. The normal forward path assumes that this fixed contract
has already been established for the selected checkpoint and environment.

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
If the module does not expose a tensor-backed input-embedding weight, runner
execution fails with a clear interface error; it does not guess from another
parameter or buffer and does not silently fall back to CPU.

## Frozen-state guarantee

Construction applies `requires_grad_(False)` to the complete NVARC module and
places both runner and model in evaluation mode. Calling `train(True)` on the
runner still leaves both in evaluation mode. Callers must not mutate the
contained model directly between forwards.

The forward hot path trusts both the packed-source contract and the stable
checkpoint-output contract. It therefore does not repeat complete type, shape,
dtype, device, gradient, or finite-value scans on every call. Failures that
make the required layer selection or concatenation impossible still surface
naturally. Full contract checks belong to the explicit checkpoint smoke or
preflight experiment described below, while tensor validators remain available
for diagnostics and tests.

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

That target-aware operation is now implemented by `prepare_flux_text_inputs`.
It derives the exact recorded prefixes, dynamically right-pads a physical FLUX
text microbatch, and returns its text mask and position IDs. Its contract and
remaining real-model checks are documented in
[`flux-text-conditioning.md`](flux-text-conditioning.md).

## Remaining real-checkpoint validation

Local fake-model tests cannot establish the numerical behavior of the released
NVARC checkpoint or its CUDA attention backend. A separate offline Colab smoke
or preflight experiment must record the checkpoint revision, Transformers
version, precision, attention backend, and GPU, then compare:

1. each sequence run alone versus the same sequence in a right-padded batch;
2. real positions only, excluding every padded output;
3. the exact dtypes and shapes of hidden states 9, 18, and 27; and
4. full-sequence prefix states versus separately executed truncated prefixes.

No prefix reuse should be connected to FLUX until those comparisons meet a
predeclared numerical tolerance. This experiment is run explicitly for the
checkpoint and execution environment; the runner neither detects a "first"
forward nor caches a preflight result inside the training path.

[`experiments/nvarc_batch_preflight_colab.ipynb`](../../experiments/nvarc_batch_preflight_colab.ipynb)
implements that real-checkpoint preflight with two differently sized synthetic
ARC sources and multiple target prefixes per source. It loads only the NVARC
checkpoint already present in Google Drive, runs the selected states both
individually and concatenated, and writes a timestamped JSON report before its
final pass/fail assertion. The optional 8192-token capacity probe is disabled by
default and is not part of the causal-equivalence verdict.

This Colab result complements rather than replaces the local unit tests. Local
tests establish deterministic packing and interface behavior with a fake model;
the Colab report establishes checkpoint-, backend-, precision-, and GPU-specific
numerical evidence. A new report is required whenever any of those execution
variables changes.
