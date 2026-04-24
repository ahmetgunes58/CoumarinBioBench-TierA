"""Run P0 project audit.

This runner keeps executable scripts short and places reusable logic inside
the coumarinbiobench package.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coumarinbiobench.audit import run_project_audit


if __name__ == "__main__":
    run_project_audit()
