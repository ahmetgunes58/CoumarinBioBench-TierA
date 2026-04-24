"""P4a runner: annotate target families and organism-aware target landscape."""

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


from coumarinbiobench.targets import (  # noqa: E402
    add_target_family_to_records,
    build_compound_family_degree_table,
    build_family_endpoint_composition,
    build_family_organism_summary,
    build_target_family_annotation,
    build_target_family_summary,
    validate_target_family_outputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"06_annotate_target_families_{timestamp}.log"

    logger = logging.getLogger("06_annotate_target_families")
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
    annotation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    compound_family_degree_df: pd.DataFrame,
) -> None:
    """Write compact P4a report."""

    lines = [
        "P4a — Target-family annotation report",
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
            f"Annotated protein targets: {len(annotation_df):,}",
            f"Unique target families: {annotation_df['target_family'].nunique():,}",
            f"Human targets: {(annotation_df['organism_group'] == 'Human').sum():,}",
            f"Non-human targets: {(annotation_df['organism_group'] == 'Non-human').sum():,}",
            f"Compound family-degree rows: {len(compound_family_degree_df):,}",
            "",
            "Target-family summary",
            "---------------------",
        ]
    )

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['target_family']}: "
            f"{int(row['record_count']):,} records; "
            f"{int(row['unique_targets']):,} targets; "
            f"{int(row['unique_compounds']):,} compounds; "
            f"dominant endpoint={row['dominant_endpoint']}; "
            f"endpoint homogeneity={row['endpoint_homogeneity']:.4f}"
        )

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

    processed_dir = ROOT / "data" / "processed"
    metadata_dir = ROOT / "data" / "metadata"
    tables_dir = ROOT / "outputs" / "tables"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    core_path = processed_dir / "CoumarinBioBench_TierA_core.csv"

    target_annotation_path = processed_dir / "target_family_annotation.csv"
    target_summary_path = processed_dir / "target_family_summary.csv"
    endpoint_composition_path = processed_dir / "family_endpoint_composition.csv"
    organism_summary_path = processed_dir / "family_organism_summary.csv"
    compound_family_degree_path = processed_dir / "compound_family_degree_table.csv"

    table_p4_path = tables_dir / "Table_P4_target_family_overview.csv"
    supplementary_s2_path = tables_dir / "Supplementary_Table_S2_target_annotation.csv"

    report_path = metadata_dir / "p4_target_family_report.txt"

    if not core_path.exists():
        raise FileNotFoundError(
            f"Tier-A Core dataset not found: {core_path}. "
            "Run scripts/02_build_tierA_core.py first."
        )

    logger.info("Loading Tier-A Core dataset: %s", core_path)
    core_df = pd.read_csv(core_path, low_memory=False)
    logger.info("Tier-A Core rows loaded: %s", f"{len(core_df):,}")

    logger.info("Building target-family annotation table.")
    annotation_df = build_target_family_annotation(core_df)
    logger.info("Annotated targets: %s", f"{len(annotation_df):,}")

    logger.info("Merging target-family annotations onto core records.")
    core_annotated_df = add_target_family_to_records(core_df, annotation_df)

    logger.info("Building target-family summary.")
    summary_df = build_target_family_summary(core_annotated_df)

    logger.info("Building family endpoint-composition table.")
    endpoint_composition_df = build_family_endpoint_composition(core_annotated_df)

    logger.info("Building family organism-summary table.")
    organism_summary_df = build_family_organism_summary(core_annotated_df)

    logger.info("Building compound family-degree table.")
    compound_family_degree_df = build_compound_family_degree_table(core_annotated_df)

    validate_target_family_outputs(annotation_df, summary_df, core_df)

    logger.info("Writing target-family annotation: %s", target_annotation_path)
    annotation_df.to_csv(target_annotation_path, index=False)

    logger.info("Writing target-family summary: %s", target_summary_path)
    summary_df.to_csv(target_summary_path, index=False)

    logger.info("Writing family endpoint composition: %s", endpoint_composition_path)
    endpoint_composition_df.to_csv(endpoint_composition_path, index=False)

    logger.info("Writing family organism summary: %s", organism_summary_path)
    organism_summary_df.to_csv(organism_summary_path, index=False)

    logger.info("Writing compound family-degree table: %s", compound_family_degree_path)
    compound_family_degree_df.to_csv(compound_family_degree_path, index=False)

    logger.info("Writing main P4 table: %s", table_p4_path)
    summary_df.to_csv(table_p4_path, index=False)

    logger.info("Writing supplementary target annotation table: %s", supplementary_s2_path)
    annotation_df.to_csv(supplementary_s2_path, index=False)

    output_paths = {
        "target_family_annotation": target_annotation_path,
        "target_family_summary": target_summary_path,
        "family_endpoint_composition": endpoint_composition_path,
        "family_organism_summary": organism_summary_path,
        "compound_family_degree_table": compound_family_degree_path,
        "Table_P4_target_family_overview": table_p4_path,
        "Supplementary_Table_S2_target_annotation": supplementary_s2_path,
    }

    write_report(
        report_path,
        input_paths={"Tier-A Core": core_path},
        output_paths=output_paths,
        annotation_df=annotation_df,
        summary_df=summary_df,
        compound_family_degree_df=compound_family_degree_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("P4a target-family annotation completed successfully.")


if __name__ == "__main__":
    main()
