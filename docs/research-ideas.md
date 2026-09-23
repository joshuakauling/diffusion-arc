# Research Ideas

Compact notes for possible future experiments. These are untested ideas;
the root [README](../README.md) remains the source of truth for project scope.

## 2026-09-07 — Later NVARC hidden states

Try conditioning FLUX on NVARC hidden states `18/27/36` instead of `9/18/27`.
Later layers may contain more useful information for solving ARC tasks;
the original FLUX layer selection may not be optimal for ARC-specialized NVARC.
Using layer 36 would require running all NVARC blocks again, giving up B3's
early exit after block 27. Compare any quality gain with that additional cost.

For Distributed Training we can select the sequences for each gpu by the amount of the flux calls to stay balanced
