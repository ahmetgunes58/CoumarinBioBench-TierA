"""P6a runner: prepare activity-cliff case-study inputs."""

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
    build_case_study_target_summary,
    build_duplicate_resolution_table,
    collapse_case_study_compounds,
    get_case_study_definitions,
    prepare_case_study_activity_records,
    validate_case_study_inputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"08_prepare_activity_cliff_case_studies_{timestamp}.log"

    logger = logging.getLogger("08_prepare_activity_cliff_case_studies")
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
    target_summary: pd.DataFrame,
) -> None:
    """Write P6a input-preparation report."""

    lines = [
        "P6a — Activity-cliff case-study input report",
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

    lines.extend(["", "Selected case-study subsets", "---------------------------"])

    for _, row in target_summary.iterrows():
        lines.append(
            f"{row['case_id']} | {row['target_label']} | "
            f"{row['target_chembl_id']} | {row['endpoint']} | "
            f"records={int(row['raw_record_count'])} | "
            f"unique_compounds={int(row['unique_compounds_raw'])} | "
            f"collapsed_compounds={int(row['compound_level_rows_after_median_collapse'])} | "
            f"pActivity_range={float(row['median_pactivity_range']):.2f} | "
            f"DOIs={int(row['doi_count'])} | "
            f"duplicate_compounds={int(row['compounds_with_duplicate_activity_records'])}"
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

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    core_path = processed_dir / "CoumarinBioBench_TierA_core.csv"
    readiness_path = processed_dir / "target_qsar_readiness.csv"

    case_targets_path = processed_dir / "p6_case_study_targets.csv"
    activity_records_path = processed_dir / "p6_case_study_activity_records.csv"
    compounds_path = processed_dir / "p6_case_study_compounds.csv"
    duplicate_resolution_path = processed_dir / "p6_case_study_duplicate_resolution.csv"

    report_path = metadata_dir / "p6a_case_study_input_report.txt"

    if not core_path.exists():
        raise FileNotFoundError(
            f"Tier-A Core dataset not found: {core_path}."
        )

    logger.info("Loading Tier-A Core dataset: %s", core_path)
    core_df = pd.read_csv(core_path, low_memory=False)
    logger.info("Tier-A Core rows loaded: %s", f"{len(core_df):,}")

    if readiness_path.exists():
        logger.info("Loading QSAR-readiness table: %s", readiness_path)
        readiness_df = pd.read_csv(readiness_path, low_memory=False)
        logger.info("QSAR-readiness rows loaded: %s", f"{len(readiness_df):,}")
    else:
        readiness_df = None
        logger.warning("QSAR-readiness table not found; continuing without it.")

    logger.info("Preparing selected case-study definitions.")
    case_definitions = get_case_study_definitions()

    logger.info("Filtering dominant-endpoint activity records.")
    activity_records = prepare_case_study_activity_records(core_df)

    logger.info("Collapsing duplicate compound-target-endpoint records by median pActivity.")
    compound_table = collapse_case_study_compounds(activity_records)

    logger.info("Building case-study target summary.")
    target_summary = build_case_study_target_summary(activity_records, compound_table)

    logger.info("Building duplicate-resolution table.")
    duplicate_resolution = build_duplicate_resolution_table(compound_table)

    validate_case_study_inputs(target_summary, activity_records, compound_table)

    logger.info("Writing case-study target summary: %s", case_targets_path)
    target_summary.to_csv(case_targets_path, index=False)

    logger.info("Writing case-study activity records: %s", activity_records_path)
    activity_records.to_csv(activity_records_path, index=False)

    logger.info("Writing case-study compound table: %s", compounds_path)
    compound_table.to_csv(compounds_path, index=False)

    logger.info("Writing duplicate-resolution table: %s", duplicate_resolution_path)
    duplicate_resolution.to_csv(duplicate_resolution_path, index=False)

    output_paths = {
        "p6_case_study_targets": case_targets_path,
        "p6_case_study_activity_records": activity_records_path,
        "p6_case_study_compounds": compounds_path,
        "p6_case_study_duplicate_resolution": duplicate_resolution_path,
    }

    write_report(
        report_path,
        input_paths={
            "Tier-A Core": core_path,
            "target_qsar_readiness": readiness_path,
        },
        output_paths=output_paths,
        target_summary=target_summary,
    )

    logger.info("Report written: %s", report_path)
    logger.info("P6a activity-cliff case-study input preparation completed successfully.")


if __name__ == "__main__":
    main()
