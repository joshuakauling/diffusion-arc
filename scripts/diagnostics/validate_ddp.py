#!/usr/bin/env python3
"""Three actual training updates: plain reference, two DDP ranks, then CPU comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diffusion_arc.training.benchmark_config import load_training_benchmark_config
from diffusion_arc.training.numerical_runner import compare_numerical_phases, run_numerical_phase
from diffusion_arc.training.run.numerics import NumericalCriteria, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("reference", "ddp", "compare"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    criteria = NumericalCriteria.load(args.criteria)
    experiment = load_training_benchmark_config(args.config)
    if args.phase == "reference" and any(key in os.environ for key in ("RANK", "WORLD_SIZE")):
        parser.error("reference must run without torchrun")
    digest = hashlib.sha256()
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "configs/training"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    identity = dict(source_sha256=digest.hexdigest(), command=sys.argv if argv is None else argv,
                    criteria_file=str(args.criteria), benchmark_config=str(args.config),
                    benchmark_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    try:
        if args.phase == "compare":
            result = compare_numerical_phases(args.output_dir, criteria)
            print(json.dumps(dict(event=result["event"], status=result["status"],
                                  failures=result["failures"], steps=result["steps"]), allow_nan=False), flush=True)
        else:
            run_numerical_phase(experiment, criteria, phase=args.phase,
                                output_directory=args.output_dir, scratch_directory=args.scratch_dir,
                                source_identity=identity)
    except BaseException as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / f"{args.phase}-rank-{os.environ.get('RANK', '0')}-failure.json",
                   dict(error_type=type(error).__name__, message=str(error)))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
