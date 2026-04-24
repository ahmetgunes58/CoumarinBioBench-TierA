"""P3b runner: generate ECFP4-based UMAP chemical-space embedding."""

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


from coumarinbiobench.chemical_space import (  # noqa: E402
    build_figure3_source_data,
    compute_umap_embedding,
    summarize_embedding,
    validate_umap_outputs,
)


def setup_logger(root: Path) -> logging.Logger:
    """Configure terminal and file logging."""

    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"05_generate_chemical_space_{timestamp}.log"

    logger = logging.getLogger("05_generate_chemical_space")
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
    """Write P3b UMAP report."""

    lines = [
        "P3b — ECFP4 UMAP chemical-space embedding report",
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

    processed_dir = ROOT / "data" / "processed"
    metadata_dir = ROOT / "data" / "metadata"
    source_data_dir = ROOT / "outputs" / "figures" / "source_data"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)

    descriptor_matrix_path = processed_dir / "descriptor_matrix.csv"
    umap_embedding_path = processed_dir / "umap_embedding.csv"
    figure3_source_path = source_data_dir / "Figure_3_source_data.csv"
    umap_summary_path = processed_dir / "umap_embedding_summary.csv"
    report_path = metadata_dir / "p3b_umap_report.txt"

    if not descriptor_matrix_path.exists():
        raise FileNotFoundError(
            f"Descriptor matrix not found: {descriptor_matrix_path}. "
            "Run scripts/04_compute_descriptors.py first."
        )

    logger.info("Loading descriptor matrix: %s", descriptor_matrix_path)
    descriptor_df = pd.read_csv(descriptor_matrix_path, low_memory=False)
    logger.info("Descriptor matrix rows loaded: %s", f"{len(descriptor_df):,}")
    logger.info("Descriptor matrix columns loaded: %s", f"{len(descriptor_df.columns):,}")

    logger.info(
        "Computing UMAP embedding using ECFP4 bits "
        "(metric=jaccard, n_neighbors=15, min_dist=0.1, random_state=42)."
    )

    result = compute_umap_embedding(
        descriptor_df,
        n_neighbors=15,
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
        expected_bits=2048,
    )

    embedding_df = result.embedding
    source_df = build_figure3_source_data(descriptor_df, embedding_df)
    validate_umap_outputs(descriptor_df, embedding_df, source_df)

    summary_df = summarize_embedding(
        embedding_df,
        source_df,
        result.parameters,
    )

    logger.info("UMAP embedding rows: %s", f"{len(embedding_df):,}")
    logger.info("Figure 3 source-data rows: %s", f"{len(source_df):,}")

    logger.info("Writing UMAP embedding: %s", umap_embedding_path)
    embedding_df.to_csv(umap_embedding_path, index=False)

    logger.info("Writing Figure 3 source data: %s", figure3_source_path)
    source_df.to_csv(figure3_source_path, index=False)

    logger.info("Writing UMAP summary: %s", umap_summary_path)
    summary_df.to_csv(umap_summary_path, index=False)

    write_report(
        report_path,
        input_paths={
            "descriptor_matrix": descriptor_matrix_path,
        },
        output_paths={
            "umap_embedding": umap_embedding_path,
            "Figure_3_source_data": figure3_source_path,
            "umap_embedding_summary": umap_summary_path,
        },
        summary_df=summary_df,
    )

    logger.info("Report written: %s", report_path)
    logger.info("P3b chemical-space embedding completed successfully.")


if __name__ == "__main__":
    main()
