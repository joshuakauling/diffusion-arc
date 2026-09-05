# Experiment Documentation

This directory separates durable findings from technical validation and
historical experiment protocols. Executable notebooks are indexed under
[`experiments/`](../../experiments/README.md), and durable local result
snapshots live under [`experiments/results/`](../../experiments/results/).

## Main findings

- [Overfitting investigation](overfitting/README.md): the path from the first
  eight-task pilots to successful same-task augmentation generalization.
- [VAE palette and round-trip analysis](vae/palette-and-roundtrip.md): the
  measured palette decision and the known padding-sensitive VAE limits.
- [NVARC direct-sampling baseline](baselines/nvarc-arc-agi-1-direct-sampling.md):
  direct ARC-AGI-1 sampling results for the frozen NVARC checkpoint.

## Technical validation

- [FLUX.2 VAE smoke test](vae/smoke-test.md): reproducible interface,
  resolution, and exact-cell validation for the frozen VAE.

## Supporting overfitting protocols

These documents preserve the preregistered questions, controls, stopping
rules, and interpretation gates behind the summarized overfitting result. They
are supporting provenance rather than entry points for the current conclusion.
The completed experiment notebooks, runners, configurations, and dedicated tests
are retained in [Git history](overfitting/README.md#historical-source); the findings
and local result snapshots remain in the current tree.

- [Eight-task pilot v1](overfitting/protocols/eight-task-pilot-v1.md)
- [Eight-task pilot v2](overfitting/protocols/eight-task-pilot-v2.md)
- [Single-task spatial ablation](overfitting/protocols/spatial-ablation.md)
- [Single-task conditioning ladder](overfitting/protocols/conditioning-ladder.md)
