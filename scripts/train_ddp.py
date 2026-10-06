#!/usr/bin/env python3
"""Compatibility entrypoint for the configurable training benchmark command."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.benchmark_training import main


if __name__ == "__main__":
    raise SystemExit(main())
