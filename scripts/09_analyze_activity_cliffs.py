"""P6b runner: fingerprint-based SALI/activity-cliff analysis."""

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


from coumarinbiobench.activity_cliffs import (  # noqa: E402
    build_activity_cliff_pairs,
    build_activity_cliff_summary,
    build_representative_cliff_pairs,
    validate_activity_cliff_outputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"09_analyze_activity_cliffs_{timestamp}.log"

    logger = logging.getLogger("09_analyze_activity_cliffs")
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
    summary_table: pd.DataFrame,
) -> None:
    """Write compact P6b activity-cliff report."""

    lines = [
        "P6b — Fingerprint-based SALI/activity-cliff analysis report",
        "=" * 76,
        f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Parameters",
        "----------",
        "Fingerprint: ECFP4/Morgan radius 2, 2048 bits",
        "Near-neighbour threshold: Tanimoto >= 0.70",
        "Activity-cliff threshold: |delta pActivity| >= 2.00",
        "SALI: |delta pActivity| / (1 - Tanimoto)",
        "",
        "Inputs",
        "------",
    ]

    for label, path in input_paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        lines.append(f"{label}: {path} [{status}]")
        if path.exists():
            lines.append(f"{label} sha256: {sha256_file(path)}")

    lines.extend(["", "Case-study summary", "------------------"])

    for _, row in summary_table.iterrows():
        lines.append(
            f"{row['case_id']} | {row['target_label']} | "
            f"{row['endpoint']} | compounds={int(row['compound_count'])} | "
            f"near_pairs={int(row['near_neighbor_pair_count_tanimoto_ge_0_70'])} | "
            f"cliffs={int(row['activity_cliff_pair_count'])} | "
            f"cliff_density={float(row['cliff_density_per_100_compounds']):.2f}/100 compounds | "
            f"max_delta={float(row['max_delta_pactivity_cliffs']):.2f} | "
            f"max_SALI={float(row['max_sali_cliffs']):.2f}"
        )

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
    tables_dir = ROOT / "outputs" / "tables"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    compounds_path = processed_dir / "p6_case_study_compounds.csv"

    pair_path = processed_dir / "p6_activity_cliff_pairs.csv"
    summary_path = processed_dir / "p6_activity_cliff_summary.csv"
    representative_path = processed_dir / "p6_representative_cliff_pairs.csv"
    failure_path = processed_dir / "p6_fingerprint_failures.csv"

    table_p6_path = tables_dir / "Table_P6_activity_cliff_summary.csv"
    supplementary_s6_path = tables_dir / "Supplementary_Table_S6_activity_cliff_pairs.csv"

    report_path = metadata_dir / "p6b_activity_cliff_report.txt"

    if not compounds_path.exists():
        raise FileNotFoundError(
            f"P6a compound-level input not found: {compounds_path}. "
            "Run scripts/08_prepare_activity_cliff_case_studies.py first."
        )

    logger.info("Loading P6 case-study compound table: %s", compounds_path)
    compound_table = pd.read_csv(compounds_path, low_memory=False)
    logger.info("Compound-level rows loaded: %s", f"{len(compound_table):,}")

    logger.info("Computing fingerprint near-neighbour and activity-cliff pairs.")
    pair_table, fingerprint_failures = build_activity_cliff_pairs(
        compound_table,
        tanimoto_threshold=0.70,
        delta_pactivity_threshold=2.00,
        radius=2,
        n_bits=2048,
    )
    logger.info("Near-neighbour pairs retained: %s", f"{len(pair_table):,}")
    logger.info(
        "Activity-cliff pairs identified: %s",
        f"{int(pair_table['activity_cliff_flag'].sum()) if not pair_table.empty else 0:,}",
    )
    logger.info("Fingerprint failures: %s", f"{len(fingerprint_failures):,}")

    logger.info("Building activity-cliff summary.")
    summary_table = build_activity_cliff_summary(compound_table, pair_table)

    logger.info("Selecting representative cliff pairs.")
    representative_table = build_representative_cliff_pairs(
        pair_table,
        top_n_per_case=10,
    )

    validate_activity_cliff_outputs(
        compound_table,
        pair_table,
        summary_table,
        fingerprint_failures,
    )

    logger.info("Writing activity-cliff pair table: %s", pair_path)
    pair_table.to_csv(pair_path, index=False)

    logger.info("Writing activity-cliff summary: %s", summary_path)
    summary_table.to_csv(summary_path, index=False)

    logger.info("Writing representative cliff pairs: %s", representative_path)
    representative_table.to_csv(representative_path, index=False)

    logger.info("Writing fingerprint failures: %s", failure_path)
    fingerprint_failures.to_csv(failure_path, index=False)

    logger.info("Writing main P6 summary table: %s", table_p6_path)
    summary_table.to_csv(table_p6_path, index=False)

    logger.info("Writing Supplementary Table S6: %s", supplementary_s6_path)
    pair_table.to_csv(supplementary_s6_path, index=False)

    output_paths = {
        "p6_activity_cliff_pairs": pair_path,
        "p6_activity_cliff_summary": summary_path,
        "p6_representative_cliff_pairs": representative_path,
        "p6_fingerprint_failures": failure_path,
        "Table_P6_activity_cliff_summary": table_p6_path,
        "Supplementary_Table_S6_activity_cliff_pairs": supplementary_s6_path,
    }

    write_report(
        report_path,
        input_paths={"p6_case_study_compounds": compounds_path},
        output_paths=output_paths,
        summary_table=summary_table,
    )

    logger.info("Report written: %s", report_path)
    logger.info("P6b activity-cliff analysis completed successfully.")


if __name__ == "__main__":
    main()
