# NVARC Batch Runner

The training encoder uses **FlashAttention-3 Varlen without padding**. One
forward processes all eligible sources in the current preparation window
(normally 16). Each source retains independent causal attention and position
IDs starting at zero. Sources are not sorted or scheduled into length buckets.

## Interface and data flow

`NvarcBatchRunner` accepts validated CPU tensors: flat `input_ids` of shape
`[T]` and `sequence_offsets` of shape `[S + 1]`, both int64. Offsets start at
zero and end at the total token count. The interface contains no target metadata.

```text
flat IDs + source offsets
  -> [1, total_tokens] IDs and reset position IDs
  -> int32 GPU cumulative lengths + longest individual source length
  -> frozen FA3 Varlen blocks 1 through 27; no dense attention mask
  -> hidden states 9, 18, 27 concatenated per source
  -> S independently stored [source_length, 7680] tensors in input order
```

The cumulative lengths prevent attention between sources. `causal=True`
prevents attention to later tokens within a source. Reset positions preserve
per-source RoPE semantics. There are no padding tokens between sources or at
the end of the packed stream, and no fixed total-token budget.

The encoder calls the original Qwen embedding, rotary-position module and first
27 decoder blocks directly. It retains only the states after blocks 9, 18 and
27, and skips blocks 28–36, the final normalization and the language-model head.
The last selected state is the original intermediate output: shortening the
ordinary Hugging Face forward would instead normalize its last returned state.
The KV cache is disabled and `torch.no_grad()` keeps the output usable by the
trainable FLUX transformer. The full checkpoint is still loaded and retained;
this change skips execution and avoids retaining unused intermediate states,
but does not remove the unused parameter tensors from GPU memory.
Construction freezes the model and keeps it in evaluation mode even if
`train(True)` is called on the runner. Callers must not change the underlying
model or its backend after construction.

Each returned source tensor owns compact storage. Targets share an
`NvarcSequenceState` and their exact prefix ends; after its last prefix is
copied into a FLUX microbatch, the full source state can be released independently.
VAE encoding follows the packed forward and spans the window's targets in
source-major order. FIFO optimizer membership and per-target loss weighting
are unchanged.

## Runtime contract

Model setup explicitly registers `nvarc_fa3` with Transformers and loads only
local checkpoints. The binding calls the official FA3 Varlen function directly,
so it does not share Transformers' lazy FlashAttention backend cache with FLUX.
There is no alternate backend or fallback in the training encoder.

The pinned Modal image compiles FA3 during its CPU build. It requires PyTorch
2.13.0 / CUDA 13.0 and uses revision
`ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`. The build keeps BF16 forward,
head dimension 128, GQA and Varlen for Hopper/SM90; other dimensions, dtypes
and backward are omitted. CUDA-device and checkpoint preflights enforce these
contracts before training. The extension is not installed in the CPU unit-test
environment. Local CUDA installations need the same extension and build contract.

NVARC retains vocabulary size 16, 36 layers, hidden width 2560, and the selected
states `(9, 18, 27)`. The maximum consumed source is 8,400 tokens; the tokenizer
allows 9,330 before the final target output is trimmed. No checkpoint pad-token
setting is needed by the packed forward; tokenizer compatibility is still checked.

## Validation and evidence

CPU tests use a tiny real Qwen model and an independent per-sequence SDPA
reference at the FA3 call boundary. They verify reset positions, causal prefix
behavior, source isolation, selected layers, compact ownership and freezing.
Pipeline tests cover source/target order, incomplete windows, the same two
48-target optimizer batches, loss weights, random draws and parameter updates.
A tiny 36-block Qwen comparison also requires bitwise-equal selected states
against the complete forward and checks that later blocks, final normalization
and the language-model head are never called. These CPU tests verify integration
semantics, not execution of the CUDA extension.

The [real-input packing diagnostic](../performance/nvarc-packing.md) measured
FA3 on short, median and long 16-source windows. It supports the no-alignment
choice but does not establish end-to-end B3 speed, peak memory or downstream
FLUX loss/gradient agreement. Those remain to be measured after combining the
NVARC and FLUX changes. The older SDPA/bucket paths and their diagnostic scripts
are recoverable at [db6d49f](https://github.com/joshuakauling/diffusion-arc/tree/db6d49f).

## Historical SDPA preflight (not a validation of FA3)

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
