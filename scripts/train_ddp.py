#!/usr/bin/env python3
"""Run a bounded torchrun experiment; no validation, W&B or checkpoint saving."""

import argparse
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diffusion_arc.training.config import load_training_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="training_run recipe or ddp_benchmark config")
    parser.add_argument("--max-steps", type=int, help="global updates for a plain recipe (default: 3)")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "diagnostic", "memory", "profile"),
                        help="prepared benchmark data; profile captures one step after the configured warmup")
    parser.add_argument("--diagnostic-batch", type=int, help="one-based batch to repeat three times from fresh state")
    parser.add_argument("--memory-snapshot-step", type=int,
                        help="memory mode: run through this step and snapshot after its first backward on each rank")
    parser.add_argument("--output-dir", type=Path, help="new artifact directory; required with --mode")
    args = parser.parse_args(argv)
    options = {}
    experiment = None
    if (args.mode == "diagnostic") != (args.diagnostic_batch is not None):
        parser.error("--mode diagnostic and --diagnostic-batch must be used together")
    if args.diagnostic_batch is not None and args.diagnostic_batch <= 0:
        parser.error("--diagnostic-batch must be positive")
    if args.memory_snapshot_step is not None and (args.mode != "memory" or args.memory_snapshot_step < 1):
        parser.error("--memory-snapshot-step requires --mode memory and a positive global step")
    if args.mode:
        if args.max_steps is not None or args.output_dir is None:
            parser.error("--mode requires --output-dir and takes step counts from the config, not --max-steps")
        from diffusion_arc.training.benchmark_config import load_ddp_benchmark_config
        from diffusion_arc.training.run.benchmarking import BenchmarkWindow
        experiment = load_ddp_benchmark_config(args.config)
        config = experiment.training
        max_steps = (experiment.warmup_steps + 1 if args.mode == "profile" else
                     3 if args.mode == "diagnostic" else
                     experiment.smoke_steps if args.mode == "smoke" else experiment.benchmark_steps)
        options["expected_source_count"] = experiment.expected_source_count
        if args.mode == "benchmark":
            options["measurement"] = BenchmarkWindow(experiment.warmup_steps, experiment.measured_steps)
        elif args.mode == "diagnostic":
            options.update(diagnostic_batch=args.diagnostic_batch, trace_directory=args.output_dir)
        elif args.mode == "memory":
            options.update(memory_history=True, trace_directory=args.output_dir)
            if args.memory_snapshot_step is not None:
                if args.memory_snapshot_step > experiment.benchmark_steps:
                    parser.error("--memory-snapshot-step exceeds the prepared benchmark window")
                max_steps = args.memory_snapshot_step
                options["memory_snapshot_step"] = args.memory_snapshot_step
        elif args.mode == "profile":
            if experiment.warmup_steps < 1:
                parser.error("--mode profile requires at least one configured warmup step")
            options.update(profile_warmup_steps=experiment.warmup_steps, trace_directory=args.output_dir)
    else:
        config = load_training_config(args.config, expected_type="training_run")
        max_steps = 3 if args.max_steps is None else args.max_steps
    if not 1 <= max_steps <= config.loop.max_optimizer_steps:
        parser.error("--max-steps must fit the recipe's optimizer-step count")
    from diffusion_arc.training.ddp_runner import run_training_ddp

    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    # Include uncommitted source contents; a base commit and dirty flag alone
    # cannot identify the code used in comparisons across separate launches.
    digest = hashlib.sha256()
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "configs/training"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    identity = dict(
        git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
        source_sha256=digest.hexdigest(),
        command=sys.argv if argv is None else [str(Path(__file__)), *argv],
    )
    if experiment is not None:
        identity["benchmark_config"] = dict(path=str(experiment.path), mode=args.mode,
                                             expected_source_count=experiment.expected_source_count,
                                             smoke_steps=experiment.smoke_steps,
                                             warmup_steps=experiment.warmup_steps,
                                             measured_steps=experiment.measured_steps)
    from diffusion_arc.training.run.benchmarking import RunArtifacts
    output = (RunArtifacts(args.output_dir) if args.output_dir is not None and os.environ.get("RANK") == "0"
              else nullcontext(None))
    with output as artifacts:
        if artifacts is not None:
            options["emit"] = artifacts
        run_training_ddp(config, max_steps=max_steps, source_identity=identity, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
