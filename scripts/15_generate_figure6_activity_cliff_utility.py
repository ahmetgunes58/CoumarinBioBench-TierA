"""P6c runner: generate Figure 6 activity-cliff utility demonstration."""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from coumarinbiobench.visualization_figure6 import (  # noqa: E402
    create_figure6,
    figure6_summary,
    save_figure6,
    validate_figure6_inputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"15_generate_figure6_activity_cliff_utility_{timestamp}.log"

    logger = logging.getLogger("15_generate_figure6_activity_cliff_utility")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.info("Logger initialized.")
    logger.info("Log file: %s", log_path)
    return logger


def sha256_file(path: Path) -> str:
    """Return SHA256 checksum of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(
    report_path: Path,
    *,
    input_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    summary_df: pd.DataFrame,
) -> None:
    """Write Figure 6 generation report."""

    lines = [
        "P6c — Figure 6 activity-cliff utility report",
        "=" * 64,
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Inputs",
        "------",
    ]

    for label, path in input_paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        lines.append(f"{label}: {path} [{status}]")
        if path.exists():
            lines.append(f"{label} sha256: {sha256_file(path)}")

    lines.extend(["", "Summary", "-------"])

    for _, row in summary_df.iterrows():
        lines.append(f"{row['metric']}: {row['value']}")

    lines.extend(["", "Outputs", "-------"])

    for label, path in output_paths.items():
        lines.append(f"{label}: {path}")
        if path.exists():
            lines.append(f"{label} sha256: {sha256_file(path)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logger = setup_logger(ROOT)

    processed_dir = ROOT / "data" / "processed"
    metadata_dir = ROOT / "data" / "metadata"
    figures_dir = ROOT / "outputs" / "figures"
    source_data_dir = figures_dir / "source_data"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)

    cliff_summary_path = processed_dir / "p6_activity_cliff_summary.csv"
    cliff_pairs_path = processed_dir / "p6_activity_cliff_pairs.csv"
    representative_path = processed_dir / "p6_representative_cliff_pairs.csv"

    figure6_summary_path = processed_dir / "figure6_summary.csv"
    report_path = metadata_dir / "p6c_figure6_report.txt"

    fig6_summary_source_path = source_data_dir / "Figure_6_source_data_activity_cliff_summary.csv"
    fig6_pairs_source_path = source_data_dir / "Figure_6_source_data_activity_cliff_pairs.csv"
    fig6_representative_source_path = source_data_dir / "Figure_6_source_data_representative_pairs.csv"

    for path in [cliff_summary_path, cliff_pairs_path, representative_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required P6b table not found: {path}. "
                "Run scripts/09_analyze_activity_cliffs.py first."
            )

    logger.info("Loading P6 activity-cliff summary: %s", cliff_summary_path)
    cliff_summary = pd.read_csv(cliff_summary_path, low_memory=False)
    logger.info("Summary rows loaded: %s", f"{len(cliff_summary):,}")

    logger.info("Loading P6 activity-cliff pair table: %s", cliff_pairs_path)
    cliff_pairs = pd.read_csv(cliff_pairs_path, low_memory=False)
    logger.info("Pair rows loaded: %s", f"{len(cliff_pairs):,}")

    logger.info("Loading P6 representative cliff pairs: %s", representative_path)
    representative = pd.read_csv(representative_path, low_memory=False)
    logger.info("Representative rows loaded: %s", f"{len(representative):,}")

    validate_figure6_inputs(cliff_summary, cliff_pairs, representative)

    logger.info("Creating Figure 6.")
    fig = create_figure6(cliff_summary, cliff_pairs, representative)

    logger.info("Saving Figure 6 PNG/PDF/SVG.")
    figure_paths = save_figure6(fig, figures_dir)

    summary_df = figure6_summary(cliff_summary, cliff_pairs)

    logger.info("Writing Figure 6 summary: %s", figure6_summary_path)
    summary_df.to_csv(figure6_summary_path, index=False)

    logger.info("Writing Figure 6 source-data files.")
    cliff_summary.to_csv(fig6_summary_source_path, index=False)
    cliff_pairs.to_csv(fig6_pairs_source_path, index=False)
    representative.to_csv(fig6_representative_source_path, index=False)

    output_paths = {
        "Figure_6_png": figure_paths["png"],
        "Figure_6_pdf": figure_paths["pdf"],
        "Figure_6_svg": figure_paths["svg"],
        "figure6_summary": figure6_summary_path,
        "Figure_6_source_data_activity_cliff_summary": fig6_summary_source_path,
        "Figure_6_source_data_activity_cliff_pairs": fig6_pairs_source_path,
        "Figure_6_source_data_representative_pairs": fig6_representative_source_path,
    }

    write_report(
        report_path,
        input_paths={
            "p6_activity_cliff_summary": cliff_summary_path,
            "p6_activity_cliff_pairs": cliff_pairs_path,
            "p6_representative_cliff_pairs": representative_path,
        },
        output_paths=output_paths,
        summary_df=summary_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("Figure 6 generation completed successfully.")


if __name__ == "__main__":
    main()
