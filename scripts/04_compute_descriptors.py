"""P3a runner: compute RDKit descriptor matrix for CoumarinBioBench Tier-A Core."""

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


from coumarinbiobench.chemistry import (  # noqa: E402
    BASIC_DESCRIPTOR_COLUMNS,
    COMPOUND_ID_CANDIDATES,
    build_descriptor_matrix,
    compare_single_multi_descriptors,
    merge_summary_and_comparison,
    resolve_column,
    summarise_descriptor_panel,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging for this runner."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"04_compute_descriptors_{timestamp}.log"

    logger = logging.getLogger("04_compute_descriptors")
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
    """Return the SHA256 checksum of a file."""

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
    core_records: int,
    unique_input_compounds: int,
    descriptor_rows: int,
    failed_molecules: int,
    descriptor_count: int,
    fingerprint_bits: int,
    degree_coverage: int,
) -> None:
    """Write a compact P3a descriptor-generation report."""

    lines = [
        "P3a — RDKit descriptor matrix generation report",
        "=" * 56,
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
            f"Core records loaded: {core_records:,}",
            f"Unique input compounds: {unique_input_compounds:,}",
            f"Descriptor-matrix rows: {descriptor_rows:,}",
            f"Failed molecules: {failed_molecules:,}",
            f"Basic RDKit descriptor columns: {descriptor_count}",
            f"ECFP4 fingerprint bits: {fingerprint_bits}",
            f"Compounds with target-degree annotation: {degree_coverage:,}",
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

    processed_dir = ROOT / "data" / "processed"
    metadata_dir = ROOT / "data" / "metadata"
    tables_dir = ROOT / "outputs" / "tables"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    core_path = processed_dir / "CoumarinBioBench_TierA_core.csv"
    degree_path = processed_dir / "compound_degree_table.csv"

    descriptor_matrix_path = processed_dir / "descriptor_matrix.csv"
    descriptor_summary_path = processed_dir / "descriptor_summary_table.csv"
    degree_comparison_path = processed_dir / "degree_descriptor_comparison.csv"
    descriptor_failures_path = processed_dir / "descriptor_generation_failures.csv"

    table_p3_path = tables_dir / "Table_P3_descriptor_summary.csv"
    supplementary_stats_path = tables_dir / "Supplementary_Table_S3_descriptor_statistics.csv"
    report_path = metadata_dir / "p3_descriptor_report.txt"

    if not core_path.exists():
        raise FileNotFoundError(
            f"Tier-A Core dataset not found: {core_path}. "
            "Run scripts/02_build_tierA_core.py first."
        )

    logger.info("Loading Tier-A Core dataset: %s", core_path)
    core_df = pd.read_csv(core_path, low_memory=False)
    logger.info("Tier-A Core rows loaded: %s", f"{len(core_df):,}")

    if degree_path.exists():
        logger.info("Loading compound degree table: %s", degree_path)
        degree_df = pd.read_csv(degree_path, low_memory=False)
        logger.info("Compound degree rows loaded: %s", f"{len(degree_df):,}")
    else:
        logger.warning(
            "Compound degree table not found: %s. Degree will be inferred from core data.",
            degree_path,
        )
        degree_df = None

    logger.info("Computing RDKit descriptors and ECFP4 fingerprints.")
    descriptor_df, failures_df = build_descriptor_matrix(
        core_df,
        degree_df=degree_df,
        radius=2,
        n_bits=2048,
    )

    logger.info("Descriptor matrix rows: %s", f"{len(descriptor_df):,}")
    logger.info("Descriptor failures: %s", f"{len(failures_df):,}")

    summary_df = summarise_descriptor_panel(descriptor_df)
    comparison_df = compare_single_multi_descriptors(descriptor_df)
    supplementary_df = merge_summary_and_comparison(summary_df, comparison_df)

    logger.info("Writing descriptor matrix: %s", descriptor_matrix_path)
    descriptor_df.to_csv(descriptor_matrix_path, index=False)

    logger.info("Writing descriptor summary table: %s", descriptor_summary_path)
    summary_df.to_csv(descriptor_summary_path, index=False)

    logger.info("Writing degree descriptor comparison: %s", degree_comparison_path)
    comparison_df.to_csv(degree_comparison_path, index=False)

    logger.info("Writing descriptor failures: %s", descriptor_failures_path)
    failures_df.to_csv(descriptor_failures_path, index=False)

    logger.info("Writing manuscript table: %s", table_p3_path)
    summary_df.to_csv(table_p3_path, index=False)

    logger.info("Writing supplementary descriptor statistics: %s", supplementary_stats_path)
    supplementary_df.to_csv(supplementary_stats_path, index=False)

    degree_coverage = int(descriptor_df["chembl_target_degree"].notna().sum())

    compound_col = resolve_column(
        core_df,
        COMPOUND_ID_CANDIDATES,
        required=False,
        label="compound identifier",
    )
    unique_input_compounds = (
        core_df[compound_col].nunique()
        if compound_col is not None
        else len(descriptor_df)
    )

    write_report(
        report_path,
        input_paths={
            "Tier-A Core": core_path,
            "compound_degree_table": degree_path,
        },
        output_paths={
            "descriptor_matrix": descriptor_matrix_path,
            "descriptor_summary_table": descriptor_summary_path,
            "degree_descriptor_comparison": degree_comparison_path,
            "descriptor_generation_failures": descriptor_failures_path,
            "Table_P3_descriptor_summary": table_p3_path,
            "Supplementary_Table_S3_descriptor_statistics": supplementary_stats_path,
        },
        core_records=len(core_df),
        unique_input_compounds=int(unique_input_compounds),
        descriptor_rows=len(descriptor_df),
        failed_molecules=len(failures_df),
        descriptor_count=len(BASIC_DESCRIPTOR_COLUMNS),
        fingerprint_bits=2048,
        degree_coverage=degree_coverage,
    )
    logger.info("Report written: %s", report_path)

    logger.info("P3a descriptor generation completed successfully.")


if __name__ == "__main__":
    main()
