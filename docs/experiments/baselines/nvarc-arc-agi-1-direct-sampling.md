# NVARC ARC-AGI-1 Direct-Sampling Baseline

On 2026-08-27, the frozen NVARC/Qwen3-4B checkpoint was evaluated by direct
stochastic sampling on the public ARC-AGI-1 v1.0.2 evaluation split. The run
used 32 samples per output at temperature 1.0 without test-time training,
test-time augmentation, voting, rescoring, retries, or target-shape hints.
Exact-grid task scores use strict sample-index alignment for tasks with multiple
test outputs.

| Metric | Solved tasks | Accuracy |
| --- | ---: | ---: |
| pass@1 | 38/396 | 9.60% |
| pass@2 | 67/396 | 16.92% |
| pass@32 | 170/396 | 42.93% |

Direct sampling therefore produced a large gain over pass@1, although the
pass@32 result required 32 generations per output. This is released-checkpoint
behavior on a known-contaminated public benchmark, not a generalization
estimate. Four tasks (`47996f11`, `981571dc`, `af22c60d`, and `f9d67f8b`) were
excluded because their required context exceeded the fixed 8,192-token
protocol, so the result is not a full 400-task score.

The complete protocol and aggregate measurements are recorded in the
[run configuration](../../../experiments/results/nvarc-arc-agi-1-pass32/arc-agi-1-v1.0.2-subset396-s32-temp1-seed42-v1/run_config.json)
and [metrics](../../../experiments/results/nvarc-arc-agi-1-pass32/arc-agi-1-v1.0.2-subset396-s32-temp1-seed42-v1/metrics.json).
