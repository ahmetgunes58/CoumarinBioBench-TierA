"""Generate Figure 2 target landscape.

This runner imports Figure 2 plotting logic from visualization_figure2.py so
the locked Figure 1 code in visualization.py remains untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coumarinbiobench.visualization_figure2 import generate_figure2_target_landscape


if __name__ == "__main__":
    generate_figure2_target_landscape()
