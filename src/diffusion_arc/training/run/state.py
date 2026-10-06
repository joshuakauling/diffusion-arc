"""Progress and metric aggregation shared by training and benchmark runs."""

from dataclasses import dataclass, field, replace
from ..trainer import StepDiagnostics
from ..step.encoding import PreparationMetrics


@dataclass
class RunProgress:
    """Global counts, including the update at which this optimizer was created."""

    optimizer_step: int = 0
    sources_seen: int = 0
    targets_seen: int = 0
    workload_totals: dict[str, int] = field(default_factory=dict)
    epoch: int = 1
    optimizer_start_step: int = 0


def aggregate_results(results):
    """Combine CPU metrics with global target weighting and rank-local timing."""
    if len(results) == 1:
        return results[0]
    target_count = sum(result.target_count for result in results)
    counters = ("source_count", "target_count", "nvarc_tokens", "nvarc_calls", "vae_calls",
                "flux_text_tokens", "flux_image_tokens", "flux_joint_tokens", "flux_forward_calls",
                "padding_forward_calls")
    source_offset = target_offset = 0
    sources, lengths, nvarc_batches, microbatch_plans, prefixes = [], [], [], [], []
    for result in results:
        diagnostics = result.diagnostics
        preparation = diagnostics.preparation
        sources.extend(preparation.sources)
        lengths.extend(preparation.source_lengths)
        nvarc_batches.extend(tuple(i + source_offset for i in batch) for batch in preparation.nvarc_batches)
        microbatch_plans.extend(replace(plan, target_indices=tuple(i + target_offset for i in plan.target_indices))
                               for plan in diagnostics.flux_microbatch_plans)
        prefixes.extend(diagnostics.target_prefix_tokens)
        source_offset += result.source_count
        target_offset += result.target_count
    return replace(results[0], **{name: sum(getattr(r, name) for r in results) for name in counters},
        loss=sum(r.loss * r.target_count for r in results) / target_count,
        step_seconds=max(r.step_seconds for r in results),
        gradient_norm=max(r.gradient_norm for r in results),
        gradient_clipped=any(r.gradient_clipped for r in results),
        target_metrics=tuple(target for result in results for target in result.target_metrics),
        diagnostics=StepDiagnostics(PreparationMetrics(tuple(sources), tuple(lengths), tuple(nvarc_batches),
                                    sum(r.vae_calls for r in results)), tuple(microbatch_plans), tuple(prefixes)))
