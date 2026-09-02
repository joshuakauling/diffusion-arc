# NVARC Batch Runner

For the current end-to-end training path and documentation reading order, see
the [training system overview](README.md).

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
[`src/diffusion_arc/model/nvarc_encoder.py`](../../src/diffusion_arc/model/nvarc_encoder.py), and
its public tensor contracts live in
[`src/diffusion_arc/model/contracts.py`](../../src/diffusion_arc/model/contracts.py).

## Public interface

The runner is constructed with an already loaded NVARC-compatible PyTorch
module:

```python
from diffusion_arc.model import NvarcBatchRunner


nvarc_runner = NvarcBatchRunner(nvarc_model)
nvarc_output = nvarc_runner(
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
NvarcBatchOutput(
    sequence_hidden_states=(
        ...,  # [L_0, 7680]
        ...,  # [L_1, 7680]
    ),
)
```

`sequence_hidden_states` follows input source order and contains one unpadded
tensor per complete source. Every tensor owns independent storage; it is not a
view into one larger packed result. Sequence boundaries are therefore resolved
once inside the runner rather than carried into the target-aware training
path. Hidden states reside on the NVARC output device.

Independent ownership is intentional. All targets belonging to one ARC source
can share that source's tensor, while the tensor can be released as soon as its
last dependent target has entered a physical FLUX microbatch. Releasing one
source never keeps unrelated sources from the same NVARC batch alive.

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
  -> for each source, concatenate its real positions to [L_s, 7680]
  -> return S independent owning tensors in source order
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

The runner never receives target identities, prefix boundaries, target images,
or FLUX state. Target-aware ownership and prefix selection happen later:

```text
NvarcBatchOutput.sequence_hidden_states[source_index]
  -> one NvarcSequenceState per source
  + one relative prefix_end per FluxInput
  -> exact prefix hidden_states[:prefix_end]
```

Keeping this operation separate ensures that NVARC always processes each
collated complete source once and cannot acquire dependencies on target images
or FLUX execution policy.

The initial single-source preparation path wraps the returned source tensor in
`NvarcSequenceState`. Each `FluxInput` stores only a reference to that owner and
its relative `prefix_end`. `build_flux_microbatch` obtains exact prefix views,
and the pure `pad_flux_text_prefixes` operation copies them into the independent
padded FLUX text tensor. Their contracts and remaining real-model checks are
documented in
[`flux-text-conditioning.md`](flux-text-conditioning.md).

## Measured real-checkpoint preflight and interpretation

Local fake-model tests cannot establish the numerical behavior of the released
NVARC checkpoint or its CUDA attention backend. The explicit real-checkpoint
preflight therefore covered two separate questions:

1. whether real-token hidden states from a right-padded NVARC batch agree with
   the same sequences executed individually without padding; and
2. whether a prefix sliced from a complete causal source forward agrees with a
   separately executed truncated-prefix forward.

The first comparison was a broad, environment-specific numerical check rather
than evidence of an unresolved expectation that NVARC lacked padding-mask
support. NVARC already exposes the standard `attention_mask` path used by the
runner. The project's separate attention-mask concern arose at the FLUX
boundary: the original inference path did not expose the padding-mask route
needed for variable-length text batching, while the pinned Diffusers
implementation does. That FLUX-specific contract and its tests are documented
in [`flux-text-conditioning.md`](flux-text-conditioning.md).

[`experiments/preflight/nvarc_batch_preflight_colab.ipynb`](../../experiments/preflight/nvarc_batch_preflight_colab.ipynb)
ran both NVARC comparisons with two differently sized synthetic ARC sources and
multiple target prefixes per source. It used the released checkpoint in BF16,
the SDPA attention implementation, Transformers 4.56.1, and an NVIDIA A100. The
runner's structural checks all passed: shapes, dtypes, selected layers, packed
offsets, finite values, frozen parameters, and evaluation mode matched the
then-current contract. The retained notebook and report use the earlier packed
output surface; their numerical comparisons remain relevant, while their
structural output fields do not validate the current tuple contract.

The predeclared elementwise `torch.allclose` gate used `atol=0.05` and
`rtol=0.02`. It returned `False` for the nontrivial padded and truncated-prefix
comparisons, so the historical report correctly retains the formal status
`FAIL`. That status must not be interpreted as padding, masking, or causal
prefix reuse being functionally broken. One out-of-tolerance element is enough
to fail a complete `[T,2560]` tensor, and the concatenated `7680` records repeat
the same three selected-layer differences rather than representing independent
failures.

The diagnostic measurements show near-equivalent representations despite the
hard gate: every tensor was finite, cosine similarities ranged from
`0.99999648` to `1.0`, and mean absolute errors ranged from approximately
`0.0026` to `0.0368`. The much larger isolated maximum absolute errors therefore
dominate the binary result without describing the typical error. The accepted
engineering interpretation is that right-padded NVARC batching and complete-
source prefix reuse are operationally valid in the tested BF16/SDPA environment,
but they are not elementwise identical under the original tolerance.

The retained
[`report.json`](../../experiments/results/nvarc-batch-preflight/20260819T160051Z/report.json)
records the exact environment, aggregate errors, and formal verdict. This
post-run interpretation does not retroactively change the predeclared
acceptance criterion. Before treating the optimization as fully validated for a
learning experiment, a downstream comparison should measure FLUX outputs,
losses, gradients, and parameter updates against exact unpadded and separately
executed-prefix references. A new NVARC report is also required whenever the
checkpoint, Transformers version, precision, attention backend, or GPU family
changes. The optional 8192-token capacity probe remained disabled and was not
part of the result.
