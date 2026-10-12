"""Benchmark bounded CPU batch planning from prepared token-length metadata.

This standalone experiment does not change training, load tasks/models, or start
GPUs. It compares the existing whole-source LPT/BFD assignment, local packing
repair, and bounded joint assignment/packing repair. Costs model token work, not
measured CUDA time. Sources stay on one rank; targets remain indivisible.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from diffusion_arc.training import batch_packing
from diffusion_arc.training.batch_packing import (
    Pack,
    PackingPlan,
    SearchBudget,
    Source,
    _fit_bins,
    _gpu_pairs,
    _profiles,
    _target_layout,
    plan_batch,
    plan_score,
    validate_plan,
)


def plan_metrics(sources, plan, capacity):
    profiles, ranks = _profiles(plan), len(plan.rank_packs)
    count, critical, variance = plan_score(plan)
    tokens = sum(sum(s.target_tokens) for s in sources)
    spreads, relative_spreads = [], []
    real_spreads, real_relative = [], []
    for slot in range(count):
        loads = [p[slot] if slot < len(p) else 0 for p in profiles]
        spread, average = max(loads) - min(loads), sum(loads) / ranks
        spreads.append(spread)
        relative_spreads.append(100 * spread / average)
        if all(loads):
            real_spreads.append(spread)
            real_relative.append(100 * spread / average)
    nvarc = [sum(max(0, max(sources[s].target_tokens) - 900) for s in owner) for owner in plan.rank_sources]
    attention_critical = 0
    costs, _ = _target_layout(sources)
    for slot in range(count):
        attention_critical += max(sum(costs[i] ** 2 for i in packs[slot].target_ids)
                                  if slot < len(packs) else 0 for packs in plan.rank_packs)
    return {
        "collective_slots": count,
        "real_calls": sum(map(len, plan.rank_packs)),
        "dummy_calls": ranks * count - sum(map(len, plan.rank_packs)),
        "slot_max_tokens": critical,
        "critical_token_excess_pct": 100 * (ranks * critical / tokens - 1),
        "slot_variance": variance,
        "slot_token_spreads": spreads,
        "slot_relative_spreads_pct": relative_spreads,
        "real_slot_token_spreads": real_spreads,
        "real_slot_relative_spreads_pct": real_relative,
        "unused_budget_pct": 100 * (1 - tokens / (ranks * count * capacity)),
        "source_counts": list(map(len, plan.rank_sources)),
        "nvarc_max_prefix_tokens": max(nvarc),
        "attention_squared_slot_max": attention_critical,
        "rank_packs": [[asdict(p) for p in packs] for packs in plan.rank_packs],
        "rank_sources": plan.rank_sources,
    }


def _distribution(values):
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": percentile(.95), "max": max(values)}


def load_metadata(path):
    """Read prepared lengths only; never scan or tokenize source task files."""
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected source-token-length metadata schema_version=1")
    keys, lengths = payload.get("source_keys"), payload.get("source_prefix_token_lengths")
    if not isinstance(keys, list) or not isinstance(lengths, list) or not keys or len(keys) != len(lengths):
        raise ValueError("metadata must contain matching source keys and prefix lists")
    if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("metadata source keys must be nonempty and unique")
    sources = []
    for key, prefix in zip(keys, lengths, strict=True):
        if (not isinstance(prefix, list) or not prefix or
                any(type(p) is not int or p <= 0 for p in prefix) or
                any(a >= b for a, b in zip(prefix, prefix[1:]))):
            raise ValueError("metadata prefixes must be positive, strictly increasing integers")
        sources.append(Source(key, tuple(p + 900 for p in prefix)))
    return sources, {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(),
                     "data_fingerprint": payload.get("data_fingerprint"),
                     "view_name": payload.get("view_name"), "provenance": payload.get("provenance")}


def _optimizer_batches(sources, target_count):
    batches, pending, count = [], [], 0
    for source in sources:
        if count + len(source.target_tokens) > target_count:
            raise ValueError("whole sources do not fit the requested global target batch")
        pending.append(source)
        count += len(source.target_tokens)
        if count == target_count:
            batches.append(tuple(pending))
            pending, count = [], 0
    return batches, len(pending)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-lengths", type=Path, default=Path("data/benchmarks/single_gpu_b0/splits/train/train.token_lengths.json"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--targets-per-gpu", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=14_336)
    parser.add_argument("--skip-sources", type=int, default=160)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--source-pool-size", type=int, default=16)
    parser.add_argument("--source-candidates", type=int, default=16)
    parser.add_argument("--source-passes", type=int, default=2)
    parser.add_argument("--pack-passes", type=int, default=3)
    parser.add_argument("--completion-nodes", type=int, default=2_000)
    parser.add_argument("--completion-finalists", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.targets_per_gpu <= 0 or args.max_batches <= 0 or args.repetitions <= 0 or
            args.skip_sources < 0 or args.max_tokens <= 0 or any(g <= 0 for g in args.gpus)):
        parser.error("counts/capacity must be positive and skip-sources nonnegative")
    budget = SearchBudget(args.source_pool_size, args.source_candidates, args.source_passes,
                          args.pack_passes, args.completion_nodes, args.completion_finalists)
    all_sources, metadata = load_metadata(args.token_lengths)
    selected = all_sources[args.skip_sources:]
    artifact = {"schema_version": 1, "metadata": metadata,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "planner_implementation_sha256": hashlib.sha256(Path(batch_packing.__file__).read_bytes()).hexdigest(),
                "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                "command": sys.argv if argv is None else [__file__, *argv],
                "search_budget": asdict(budget), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runs": []}
    print("GPUs targets strategy batches mean_ms p95_ms slots dummy critical_excess_pct real_spread_pct", flush=True)
    for gpu_count in args.gpus:
        batches, partial_sources = _optimizer_batches(selected, gpu_count * args.targets_per_gpu)
        batches = batches[:args.max_batches]
        if not batches:
            parser.error(f"no complete optimizer batches for {gpu_count} GPUs")
        for strategy in ("baseline", "local", "joint"):
            observations, records = [], []
            for batch_index, batch in enumerate(batches):
                reference = None
                for _ in range(args.repetitions):
                    started = time.perf_counter()
                    plan = plan_batch(batch, gpu_count, args.max_tokens, strategy, budget)
                    observations.append(1000 * (time.perf_counter() - started))
                    validate_plan(batch, plan, args.max_tokens)
                    if reference is not None and plan != reference:
                        raise RuntimeError("planner must be deterministic across repetitions")
                    reference = plan
                records.append({"batch_index": batch_index, "source_keys": [s.key for s in batch],
                                "source_target_tokens": [s.target_tokens for s in batch],
                                "metrics": plan_metrics(batch, plan, args.max_tokens)})
            metrics = [r["metrics"] for r in records]
            spreads = [x for m in metrics for x in m["real_slot_relative_spreads_pct"]]
            token_spreads = [x for m in metrics for x in m["real_slot_token_spreads"]]
            summary = {"gpu_count": gpu_count, "global_target_count": gpu_count * args.targets_per_gpu,
                       "strategy": strategy, "batches": len(batches), "partial_sources_omitted": partial_sources,
                       "planning_ms": _distribution(observations),
                       "mean_collective_slots": statistics.mean(m["collective_slots"] for m in metrics),
                       "dummy_calls": sum(m["dummy_calls"] for m in metrics),
                       "mean_critical_token_excess_pct": statistics.mean(m["critical_token_excess_pct"] for m in metrics),
                       "real_slot_token_spread": _distribution(token_spreads),
                       "real_slot_relative_spread_pct": _distribution(spreads),
                       "mean_nvarc_max_prefix_tokens": statistics.mean(m["nvarc_max_prefix_tokens"] for m in metrics),
                       "mean_attention_squared_slot_max": statistics.mean(m["attention_squared_slot_max"] for m in metrics)}
            artifact["runs"].append({"summary": summary, "records": records})
            timing = summary["planning_ms"]
            print(f"{gpu_count} {summary['global_target_count']} {strategy} {len(batches)} "
                  f"{timing['mean']:.2f} {timing['p95']:.2f} {summary['mean_collective_slots']:.3f} "
                  f"{summary['dummy_calls']} {summary['mean_critical_token_excess_pct']:.3f} "
                  f"{summary['real_slot_relative_spread_pct']['mean']:.3f}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        print(f"Results: {args.output.resolve()}")
    return artifact


if __name__ == "__main__":
    main()
