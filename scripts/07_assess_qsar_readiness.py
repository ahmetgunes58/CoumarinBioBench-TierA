"""P5a runner: assess target-specific QSAR readiness."""

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


from coumarinbiobench.readiness import (  # noqa: E402
    build_readiness_family_summary,
    build_readiness_tier_summary,
    build_ready_target_table,
    build_target_qsar_readiness,
    validate_readiness_outputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"07_assess_qsar_readiness_{timestamp}.log"

    logger = logging.getLogger("07_assess_qsar_readiness")
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
    tier_summary: pd.DataFrame,
    ready_targets: pd.DataFrame,
) -> None:
    """Write compact P5a report."""

    lines = [
        "P5a — QSAR-readiness assessment report",
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

    lines.extend(["", "Readiness tier summary", "----------------------"])

    for _, row in tier_summary.iterrows():
        lines.append(
            f"{row['readiness_tier']}: "
            f"{int(row['target_count'])} targets "
            f"({float(row['percent_of_targets']):.2f}%)"
        )

    lines.extend(["", "Ready targets", "-------------"])
    if ready_targets.empty:
        lines.append("No Ready-tier targets identified.")
    else:
        for _, row in ready_targets.iterrows():
            lines.append(
                f"{row['target_chembl_id']} | {row['target_name']} | "
                f"{row['target_organism']} | {row.get('target_family', 'NA')} | "
                f"n={int(row['unique_compounds'])} | "
                f"{row['dominant_endpoint']} | "
                f"homogeneity={float(row['endpoint_homogeneity']):.4f} | "
                f"range={float(row['dominant_endpoint_pactivity_range']):.2f} | "
                f"DOIs={int(row['doi_count'])}"
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

    core_path = processed_dir / "CoumarinBioBench_TierA_core.csv"
    target_annotation_path = processed_dir / "target_family_annotation.csv"

    readiness_path = processed_dir / "target_qsar_readiness.csv"
    tier_summary_path = processed_dir / "readiness_tier_summary.csv"
    family_summary_path = processed_dir / "readiness_family_summary.csv"
    ready_target_path = processed_dir / "ready_target_table.csv"

    table_p5_path = tables_dir / "Table_P5_qsar_readiness_summary.csv"
    supplementary_s4_path = tables_dir / "Supplementary_Table_S4_target_qsar_readiness.csv"

    report_path = metadata_dir / "p5_qsar_readiness_report.txt"

    if not core_path.exists():
        raise FileNotFoundError(
            f"Tier-A Core dataset not found: {core_path}."
        )

    logger.info("Loading Tier-A Core dataset: %s", core_path)
    core_df = pd.read_csv(core_path, low_memory=False)
    logger.info("Tier-A Core rows loaded: %s", f"{len(core_df):,}")

    target_annotation_df = None
    if target_annotation_path.exists():
        logger.info("Loading target-family annotation: %s", target_annotation_path)
        target_annotation_df = pd.read_csv(target_annotation_path, low_memory=False)
        logger.info("Target-family annotation rows loaded: %s", f"{len(target_annotation_df):,}")
    else:
        logger.warning(
            "Target-family annotation not found. Readiness table will be generated "
            "without target_family/organism_group annotations."
        )

    logger.info("Building target-specific QSAR-readiness table.")
    readiness_df = build_target_qsar_readiness(core_df, target_annotation_df)

    validate_readiness_outputs(readiness_df, core_df)

    logger.info("Building readiness tier summary.")
    tier_summary = build_readiness_tier_summary(readiness_df)

    logger.info("Building readiness family summary.")
    family_summary = build_readiness_family_summary(readiness_df)

    logger.info("Building Ready-target table.")
    ready_targets = build_ready_target_table(readiness_df)

    logger.info("Writing target QSAR-readiness table: %s", readiness_path)
    readiness_df.to_csv(readiness_path, index=False)

    logger.info("Writing readiness tier summary: %s", tier_summary_path)
    tier_summary.to_csv(tier_summary_path, index=False)

    logger.info("Writing readiness family summary: %s", family_summary_path)
    family_summary.to_csv(family_summary_path, index=False)

    logger.info("Writing Ready-target table: %s", ready_target_path)
    ready_targets.to_csv(ready_target_path, index=False)

    logger.info("Writing main P5 summary table: %s", table_p5_path)
    tier_summary.to_csv(table_p5_path, index=False)

    logger.info("Writing supplementary QSAR-readiness table: %s", supplementary_s4_path)
    readiness_df.to_csv(supplementary_s4_path, index=False)

    output_paths = {
        "target_qsar_readiness": readiness_path,
        "readiness_tier_summary": tier_summary_path,
        "readiness_family_summary": family_summary_path,
        "ready_target_table": ready_target_path,
        "Table_P5_qsar_readiness_summary": table_p5_path,
        "Supplementary_Table_S4_target_qsar_readiness": supplementary_s4_path,
    }

    write_report(
        report_path,
        input_paths={
            "Tier-A Core": core_path,
            "target_family_annotation": target_annotation_path,
        },
        output_paths=output_paths,
        tier_summary=tier_summary,
        ready_targets=ready_targets,
    )

    logger.info("Report written: %s", report_path)
    logger.info("P5a QSAR-readiness assessment completed successfully.")


if __name__ == "__main__":
    main()
