"""P3c runner: generate Figure 3 chemical-space visualization."""

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


from coumarinbiobench.visualization_figure3 import (  # noqa: E402
    create_figure3,
    figure3_summary,
    save_figure3,
    validate_figure3_inputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"12_generate_figure3_chemical_space_{timestamp}.log"

    logger = logging.getLogger("12_generate_figure3_chemical_space")
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
    """Write Figure 3 generation report."""

    lines = [
        "P3c — Figure 3 chemical-space visualization report",
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

    lines.extend(
        [
            "",
            "Summary",
            "-------",
        ]
    )

    for _, row in summary_df.iterrows():
        lines.append(f"{row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "Outputs",
            "-------",
        ]
    )

    for label, path in output_paths.items():
        lines.append(f"{label}: {path}")
        if path.exists():
            lines.append(f"{label} sha256: {sha256_file(path)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logger = setup_logger(ROOT)

    source_data_path = ROOT / "outputs" / "figures" / "source_data" / "Figure_3_source_data.csv"
    comparison_path = ROOT / "data" / "processed" / "degree_descriptor_comparison.csv"

    figures_dir = ROOT / "outputs" / "figures"
    metadata_dir = ROOT / "data" / "metadata"
    processed_dir = ROOT / "data" / "processed"

    figure3_summary_path = processed_dir / "figure3_summary.csv"
    report_path = metadata_dir / "p3c_figure3_report.txt"

    if not source_data_path.exists():
        raise FileNotFoundError(
            f"Figure 3 source data not found: {source_data_path}. "
            "Run scripts/05_generate_chemical_space.py first."
        )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Descriptor comparison table not found: {comparison_path}. "
            "Run scripts/04_compute_descriptors.py first."
        )

    logger.info("Loading Figure 3 source data: %s", source_data_path)
    source_df = pd.read_csv(source_data_path, low_memory=False)
    logger.info("Figure 3 source rows loaded: %s", f"{len(source_df):,}")

    logger.info("Loading descriptor comparison table: %s", comparison_path)
    comparison_df = pd.read_csv(comparison_path, low_memory=False)
    logger.info("Descriptor comparison rows loaded: %s", f"{len(comparison_df):,}")

    validate_figure3_inputs(source_df, comparison_df)

    logger.info("Creating Figure 3.")
    fig = create_figure3(source_df, comparison_df)

    logger.info("Saving Figure 3 PNG/PDF/SVG.")
    figure_paths = save_figure3(fig, figures_dir)

    summary_df = figure3_summary(source_df, comparison_df)
    logger.info("Writing Figure 3 summary: %s", figure3_summary_path)
    summary_df.to_csv(figure3_summary_path, index=False)

    output_paths = {
        "Figure_3_png": figure_paths["png"],
        "Figure_3_pdf": figure_paths["pdf"],
        "Figure_3_svg": figure_paths["svg"],
        "figure3_summary": figure3_summary_path,
    }

    write_report(
        report_path,
        input_paths={
            "Figure_3_source_data": source_data_path,
            "degree_descriptor_comparison": comparison_path,
        },
        output_paths=output_paths,
        summary_df=summary_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("Figure 3 generation completed successfully.")


if __name__ == "__main__":
    main()
