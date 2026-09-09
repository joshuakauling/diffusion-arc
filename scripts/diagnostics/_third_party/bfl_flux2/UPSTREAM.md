# BFL FLUX.2 diagnostic reference

`model.py` and `LICENSE.md` are byte-for-byte copies from
[Black Forest Labs FLUX.2](https://github.com/black-forest-labs/flux2/tree/50fe5162777813d869182b139e83b10743caef15),
commit `50fe5162777813d869182b139e83b10743caef15`. `source.json` records
their upstream paths and SHA-256 hashes. The diagnostic verifies these files
before importing or building the reference model. There are no source edits.

This copy is restricted to diagnostics. Production continues to use the pinned
Diffusers implementation. The adapter maps the existing Diffusers checkpoint
into the original BFL parameter layout without downloading another checkpoint.
Both models receive the same externally prepared tensors.

BFL calls block `forward_kv_extract` methods directly. Diagnostic observers wrap
these calls only to capture their returned tensors, then restore the methods;
they do not replace the computation. Ordinary module hooks observe projections
and the final normalization boundary. The source's empty reference-token
attention calls also remain intact.
