"""P5b runner: generate Figure 5 QSAR-readiness visualization."""

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


from coumarinbiobench.visualization_figure5 import (  # noqa: E402
    create_figure5,
    figure5_summary,
    save_figure5,
    validate_figure5_inputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"14_generate_figure5_qsar_readiness_{timestamp}.log"

    logger = logging.getLogger("14_generate_figure5_qsar_readiness")
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
    """Write Figure 5 generation report."""

    lines = [
        "P5b — Figure 5 QSAR-readiness visualization report",
        "=" * 72,
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

    readiness_path = processed_dir / "target_qsar_readiness.csv"
    tier_summary_path = processed_dir / "readiness_tier_summary.csv"
    family_summary_path = processed_dir / "readiness_family_summary.csv"
    ready_targets_path = processed_dir / "ready_target_table.csv"

    figure5_summary_path = processed_dir / "figure5_summary.csv"
    report_path = metadata_dir / "p5b_figure5_report.txt"

    fig5_readiness_source_path = source_data_dir / "Figure_5_source_data_target_qsar_readiness.csv"
    fig5_tier_source_path = source_data_dir / "Figure_5_source_data_tier_summary.csv"
    fig5_family_source_path = source_data_dir / "Figure_5_source_data_family_summary.csv"
    fig5_ready_source_path = source_data_dir / "Figure_5_source_data_ready_targets.csv"

    for path in [
        readiness_path,
        tier_summary_path,
        family_summary_path,
        ready_targets_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required P5a table not found: {path}. "
                "Run scripts/07_assess_qsar_readiness.py first."
            )

    logger.info("Loading target QSAR-readiness table: %s", readiness_path)
    readiness_df = pd.read_csv(readiness_path, low_memory=False)
    logger.info("Readiness rows loaded: %s", f"{len(readiness_df):,}")

    logger.info("Loading readiness tier summary: %s", tier_summary_path)
    tier_summary = pd.read_csv(tier_summary_path, low_memory=False)
    logger.info("Tier-summary rows loaded: %s", f"{len(tier_summary):,}")

    logger.info("Loading readiness family summary: %s", family_summary_path)
    family_summary = pd.read_csv(family_summary_path, low_memory=False)
    logger.info("Family-summary rows loaded: %s", f"{len(family_summary):,}")

    logger.info("Loading Ready-target table: %s", ready_targets_path)
    ready_targets = pd.read_csv(ready_targets_path, low_memory=False)
    logger.info("Ready-target rows loaded: %s", f"{len(ready_targets):,}")

    validate_figure5_inputs(
        readiness_df,
        tier_summary,
        family_summary,
        ready_targets,
    )

    logger.info("Creating Figure 5.")
    fig = create_figure5(
        readiness_df,
        tier_summary,
        family_summary,
        ready_targets,
    )

    logger.info("Saving Figure 5 PNG/PDF/SVG.")
    figure_paths = save_figure5(fig, figures_dir)

    summary_df = figure5_summary(readiness_df, tier_summary, ready_targets)
    logger.info("Writing Figure 5 summary: %s", figure5_summary_path)
    summary_df.to_csv(figure5_summary_path, index=False)

    logger.info("Writing Figure 5 source-data files.")
    readiness_df.to_csv(fig5_readiness_source_path, index=False)
    tier_summary.to_csv(fig5_tier_source_path, index=False)
    family_summary.to_csv(fig5_family_source_path, index=False)
    ready_targets.to_csv(fig5_ready_source_path, index=False)

    output_paths = {
        "Figure_5_png": figure_paths["png"],
        "Figure_5_pdf": figure_paths["pdf"],
        "Figure_5_svg": figure_paths["svg"],
        "figure5_summary": figure5_summary_path,
        "Figure_5_source_data_target_qsar_readiness": fig5_readiness_source_path,
        "Figure_5_source_data_tier_summary": fig5_tier_source_path,
        "Figure_5_source_data_family_summary": fig5_family_source_path,
        "Figure_5_source_data_ready_targets": fig5_ready_source_path,
    }

    write_report(
        report_path,
        input_paths={
            "target_qsar_readiness": readiness_path,
            "readiness_tier_summary": tier_summary_path,
            "readiness_family_summary": family_summary_path,
            "ready_target_table": ready_targets_path,
        },
        output_paths=output_paths,
        summary_df=summary_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("Figure 5 generation completed successfully.")


if __name__ == "__main__":
    main()
