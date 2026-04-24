"""P4b runner: generate Figure 4 target-family landscape."""

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


from coumarinbiobench.visualization_figure4 import (  # noqa: E402
    create_figure4,
    figure4_summary,
    save_figure4,
    validate_figure4_inputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"13_generate_figure4_target_family_landscape_{timestamp}.log"

    logger = logging.getLogger("13_generate_figure4_target_family_landscape")
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
    """Write Figure 4 generation report."""

    lines = [
        "P4b — Figure 4 target-family landscape report",
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

    source_data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    family_summary_path = processed_dir / "target_family_summary.csv"
    endpoint_composition_path = processed_dir / "family_endpoint_composition.csv"
    organism_summary_path = processed_dir / "family_organism_summary.csv"

    figure4_summary_path = processed_dir / "figure4_summary.csv"
    report_path = metadata_dir / "p4b_figure4_report.txt"

    fig4_family_source_path = source_data_dir / "Figure_4_source_data_family_summary.csv"
    fig4_endpoint_source_path = source_data_dir / "Figure_4_source_data_endpoint_composition.csv"
    fig4_organism_source_path = source_data_dir / "Figure_4_source_data_organism_summary.csv"

    for path in [
        family_summary_path,
        endpoint_composition_path,
        organism_summary_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required P4a table not found: {path}. "
                "Run scripts/06_annotate_target_families.py first."
            )

    logger.info("Loading target-family summary: %s", family_summary_path)
    family_summary = pd.read_csv(family_summary_path, low_memory=False)
    logger.info("Family summary rows loaded: %s", f"{len(family_summary):,}")

    logger.info("Loading endpoint composition: %s", endpoint_composition_path)
    endpoint_composition = pd.read_csv(endpoint_composition_path, low_memory=False)
    logger.info("Endpoint composition rows loaded: %s", f"{len(endpoint_composition):,}")

    logger.info("Loading organism summary: %s", organism_summary_path)
    organism_summary = pd.read_csv(organism_summary_path, low_memory=False)
    logger.info("Organism summary rows loaded: %s", f"{len(organism_summary):,}")

    validate_figure4_inputs(
        family_summary,
        endpoint_composition,
        organism_summary,
    )

    logger.info("Creating Figure 4.")
    fig = create_figure4(
        family_summary,
        endpoint_composition,
        organism_summary,
    )

    logger.info("Saving Figure 4 PNG/PDF/SVG.")
    figure_paths = save_figure4(fig, figures_dir)

    summary_df = figure4_summary(family_summary, organism_summary)
    logger.info("Writing Figure 4 summary: %s", figure4_summary_path)
    summary_df.to_csv(figure4_summary_path, index=False)

    logger.info("Writing Figure 4 source-data files.")
    family_summary.to_csv(fig4_family_source_path, index=False)
    endpoint_composition.to_csv(fig4_endpoint_source_path, index=False)
    organism_summary.to_csv(fig4_organism_source_path, index=False)

    output_paths = {
        "Figure_4_png": figure_paths["png"],
        "Figure_4_pdf": figure_paths["pdf"],
        "Figure_4_svg": figure_paths["svg"],
        "figure4_summary": figure4_summary_path,
        "Figure_4_source_data_family_summary": fig4_family_source_path,
        "Figure_4_source_data_endpoint_composition": fig4_endpoint_source_path,
        "Figure_4_source_data_organism_summary": fig4_organism_source_path,
    }

    write_report(
        report_path,
        input_paths={
            "target_family_summary": family_summary_path,
            "family_endpoint_composition": endpoint_composition_path,
            "family_organism_summary": organism_summary_path,
        },
        output_paths=output_paths,
        summary_df=summary_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("Figure 4 generation completed successfully.")


if __name__ == "__main__":
    main()
