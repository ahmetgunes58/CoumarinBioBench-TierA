"""P0 curation audit pipeline for CoumarinBioBench-TierA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from coumarinbiobench.config import ProjectConfig, load_config
from coumarinbiobench.io_utils import (
    compute_sha256,
    read_csv_safely,
    write_dataframe,
    write_text,
)
from coumarinbiobench.logging_utils import setup_logger
from coumarinbiobench.validation import (
    count_unique_non_null,
    non_empty_mask,
    resolve_column,
    resolve_optional_column,
)


@dataclass
class ResolvedColumns:
    """Resolved column names used by the audit pipeline."""

    molecule_id: str
    target_id: str
    target_type: str
    standard_type: str
    standard_relation: str
    standard_value: str
    standard_units: str

    smiles: str | None = None
    target_name: str | None = None
    organism: str | None = None
    uniprot: str | None = None
    pchembl: str | None = None
    confidence_score: str | None = None
    data_validity_comment: str | None = None
    source_id: str | None = None
    doi: str | None = None
    journal: str | None = None
    year: str | None = None


class ProjectAuditor:
    """Run P0 curation audit and dataset freeze checks."""

    def __init__(self, project_config: ProjectConfig) -> None:
        """Initialize auditor.

        Parameters
        ----------
        project_config : ProjectConfig
            Loaded project configuration.
        """
        self.project_config = project_config
        self.config = project_config.config
        self.logger = setup_logger(
            name="coumarinbiobench.audit",
            log_dir=project_config.logs_dir,
            prefix="00_project_audit",
        )
        self.audit_rows: list[dict[str, Any]] = []

    def run(self) -> None:
        """Execute complete P0 audit pipeline."""
        self.logger.info("Starting P0 project audit.")
        df_raw = read_csv_safely(self.project_config.raw_data)
        self.logger.info("Raw dataset loaded: %s records", len(df_raw))
        self.logger.info("Raw columns detected: %s", len(df_raw.columns))

        columns = self._resolve_columns(df_raw)
        self._log_column_mapping(columns)

        self._add_audit_row(
            step="P0_raw_dataset",
            before_df=None,
            after_df=df_raw,
            columns=columns,
            notes="Original raw dataset loaded from data/raw.",
        )

        df_protein = self._filter_equals(
            df=df_raw,
            column=columns.target_type,
            value=self.config["curation"]["accepted_target_type"],
            step="P0_01_single_protein_targets",
            columns=columns,
        )

        self._write_confidence_summary(df_protein, columns)

        df_relation = self._filter_equals(
            df=df_protein,
            column=columns.standard_relation,
            value=self.config["curation"]["accepted_relation"],
            step='P0_02_exact_relation_equals',
            columns=columns,
        )

        self._write_potency_summary(df_relation, columns)

        accepted_endpoints = self.config["curation"]["accepted_endpoints"]
        df_endpoints = self._filter_in(
            df=df_relation,
            column=columns.standard_type,
            accepted_values=accepted_endpoints,
            step="P0_03_accepted_endpoints",
            columns=columns,
        )

        df_units = self._filter_equals(
            df=df_endpoints,
            column=columns.standard_units,
            value=self.config["curation"]["accepted_unit"],
            step="P0_04_nM_standardized_units",
            columns=columns,
        )

        df_positive = self._filter_positive_numeric(
            df=df_units,
            column=columns.standard_value,
            step="P0_05_positive_numeric_values",
            columns=columns,
        )

        df_screened = df_positive.copy()
        self.logger.info("Tier-A Screened records: %s", len(df_screened))

        screened_path = self.project_config.resolve("outputs.tierA_screened")
        write_dataframe(df_screened, screened_path)

        validity_mask = self._validity_warning_mask(df_screened, columns)
        df_validity_warning = df_screened.loc[validity_mask].copy()
        df_core = df_screened.loc[~validity_mask].copy()

        self.logger.info("Validity-warning records: %s", len(df_validity_warning))
        self.logger.info("Tier-A Core records: %s", len(df_core))

        self._add_audit_row(
            step="P0_06_remove_data_validity_warnings",
            before_df=df_screened,
            after_df=df_core,
            columns=columns,
            notes="Tier-A Core after excluding non-null ChEMBL data_validity_comment records.",
        )

        write_dataframe(
            df_validity_warning,
            self.project_config.resolve("outputs.validity_warning_records"),
        )
        write_dataframe(
            df_core,
            self.project_config.resolve("outputs.tierA_core"),
        )
        write_dataframe(
            df_core,
            self.project_config.resolve("outputs.tierA_core_processed"),
        )

        self._write_audit_table()
        self._write_dataset_summary(
            df_raw=df_raw,
            df_screened=df_screened,
            df_core=df_core,
            df_validity_warning=df_validity_warning,
            columns=columns,
        )
        self._write_uniprot_coverage_report(df_core, columns)
        self._write_target_organism_summary(df_core, columns)
        self._write_dataset_freeze_report(
            df_raw=df_raw,
            df_screened=df_screened,
            df_core=df_core,
            df_validity_warning=df_validity_warning,
            columns=columns,
        )
        self._write_checksums()

        self.logger.info("P0 project audit completed successfully.")

    def _resolve_columns(self, df: pd.DataFrame) -> ResolvedColumns:
        """Resolve all required and optional columns."""
        cfg = self.config["columns"]

        return ResolvedColumns(
            molecule_id=resolve_column(
                df, cfg["molecule_id_candidates"], "molecule_id"
            ),
            target_id=resolve_column(
                df, cfg["target_id_candidates"], "target_id"
            ),
            target_type=resolve_column(
                df, cfg["target_type_candidates"], "target_type"
            ),
            standard_type=resolve_column(
                df, cfg["standard_type_candidates"], "standard_type"
            ),
            standard_relation=resolve_column(
                df, cfg["standard_relation_candidates"], "standard_relation"
            ),
            standard_value=resolve_column(
                df, cfg["standard_value_candidates"], "standard_value"
            ),
            standard_units=resolve_column(
                df, cfg["standard_units_candidates"], "standard_units"
            ),
            smiles=resolve_optional_column(
                df, cfg["smiles_candidates"], "smiles"
            ),
            target_name=resolve_optional_column(
                df, cfg["target_name_candidates"], "target_name"
            ),
            organism=resolve_optional_column(
                df, cfg["organism_candidates"], "organism"
            ),
            uniprot=resolve_optional_column(
                df, cfg["uniprot_candidates"], "uniprot"
            ),
            pchembl=resolve_optional_column(
                df, cfg["pchembl_candidates"], "pchembl"
            ),
            confidence_score=resolve_optional_column(
                df, cfg["confidence_score_candidates"], "confidence_score"
            ),
            data_validity_comment=resolve_optional_column(
                df,
                cfg["data_validity_comment_candidates"],
                "data_validity_comment",
            ),
            source_id=resolve_optional_column(
                df, cfg["source_id_candidates"], "source_id"
            ),
            doi=resolve_optional_column(
                df, cfg["doi_candidates"], "doi"
            ),
            journal=resolve_optional_column(
                df, cfg["journal_candidates"], "journal"
            ),
            year=resolve_optional_column(
                df, cfg["year_candidates"], "year"
            ),
        )

    def _log_column_mapping(self, columns: ResolvedColumns) -> None:
        """Log resolved column mapping."""
        self.logger.info("Resolved column mapping:")
        for field_name, value in columns.__dict__.items():
            self.logger.info("  %s -> %s", field_name, value)

    def _add_audit_row(
        self,
        step: str,
        before_df: pd.DataFrame | None,
        after_df: pd.DataFrame,
        columns: ResolvedColumns,
        notes: str,
    ) -> None:
        """Add one audit-table row."""
        before_count = len(before_df) if before_df is not None else 0
        after_count = len(after_df)
        removed = before_count - after_count if before_df is not None else 0

        percent_removed = (
            round((removed / before_count) * 100, 4)
            if before_df is not None and before_count > 0
            else 0.0
        )

        self.audit_rows.append(
            {
                "step": step,
                "records_before": before_count,
                "records_after": after_count,
                "records_removed": removed,
                "percent_removed": percent_removed,
                "unique_molecules": count_unique_non_null(
                    after_df, columns.molecule_id
                ),
                "unique_targets": count_unique_non_null(
                    after_df, columns.target_id
                ),
                "notes": notes,
            }
        )

    def _filter_equals(
        self,
        df: pd.DataFrame,
        column: str,
        value: str,
        step: str,
        columns: ResolvedColumns,
    ) -> pd.DataFrame:
        """Filter dataframe by case-insensitive equality."""
        before = df
        mask = (
            before[column]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(str(value).strip().upper())
        )
        after = before.loc[mask].copy()

        self.logger.info(
            "%s: %s -> %s records",
            step,
            len(before),
            len(after),
        )

        self._add_audit_row(
            step=step,
            before_df=before,
            after_df=after,
            columns=columns,
            notes=f"Retained rows where {column} == {value}.",
        )
        return after

    def _filter_in(
        self,
        df: pd.DataFrame,
        column: str,
        accepted_values: list[str],
        step: str,
        columns: ResolvedColumns,
    ) -> pd.DataFrame:
        """Filter dataframe by accepted case-insensitive values."""
        before = df
        accepted = {str(value).strip().upper() for value in accepted_values}
        mask = before[column].astype(str).str.strip().str.upper().isin(accepted)
        after = before.loc[mask].copy()

        self.logger.info(
            "%s: %s -> %s records",
            step,
            len(before),
            len(after),
        )

        self._add_audit_row(
            step=step,
            before_df=before,
            after_df=after,
            columns=columns,
            notes=f"Accepted endpoint types: {', '.join(accepted_values)}.",
        )
        return after

    def _filter_positive_numeric(
        self,
        df: pd.DataFrame,
        column: str,
        step: str,
        columns: ResolvedColumns,
    ) -> pd.DataFrame:
        """Filter dataframe to positive numeric values."""
        before = df.copy()
        numeric_values = pd.to_numeric(before[column], errors="coerce")
        mask = numeric_values.notna() & (numeric_values > 0)
        after = before.loc[mask].copy()

        self.logger.info(
            "%s: %s -> %s records",
            step,
            len(before),
            len(after),
        )

        self._add_audit_row(
            step=step,
            before_df=before,
            after_df=after,
            columns=columns,
            notes=f"Retained rows where {column} is numeric and > 0.",
        )
        return after

    def _validity_warning_mask(
        self,
        df: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> pd.Series:
        """Return mask for rows carrying ChEMBL data-validity warnings."""
        if columns.data_validity_comment is None:
            self.logger.warning(
                "data_validity_comment column not found; no validity warnings removed."
            )
            return pd.Series(False, index=df.index)

        return non_empty_mask(df[columns.data_validity_comment])

    def _write_confidence_summary(
        self,
        df: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Log confidence score distribution."""
        if columns.confidence_score is None:
            self.logger.warning("confidence_score column not found.")
            return

        distribution = (
            df[columns.confidence_score]
            .value_counts(dropna=False)
            .sort_index()
        )
        self.logger.info("Confidence score distribution after protein filter:")
        for score, count in distribution.items():
            self.logger.info("  confidence_score=%s: %s records", score, count)

    def _write_potency_summary(
        self,
        df: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Write summary of excluded Potency endpoint records."""
        excluded_endpoint = str(self.config["curation"]["excluded_endpoint"]).upper()
        mask = (
            df[columns.standard_type]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(excluded_endpoint)
        )
        potency = df.loc[mask].copy()

        doi_count = 0
        doi_coverage = 0.0
        if columns.doi is not None and len(potency) > 0:
            doi_mask = non_empty_mask(potency[columns.doi])
            doi_count = int(doi_mask.sum())
            doi_coverage = round(100 * doi_count / len(potency), 4)

        journal_count = 0
        if columns.journal is not None and len(potency) > 0:
            journal_count = int(non_empty_mask(potency[columns.journal]).sum())

        source_distribution = {}
        if columns.source_id is not None and len(potency) > 0:
            source_distribution = (
                potency[columns.source_id]
                .value_counts(dropna=False)
                .to_dict()
            )

        summary_rows = [
            {"metric": "potency_records", "value": len(potency), "notes": ""},
            {"metric": "potency_doi_count", "value": doi_count, "notes": ""},
            {"metric": "potency_doi_coverage_percent", "value": doi_coverage, "notes": ""},
            {"metric": "potency_journal_non_null_count", "value": journal_count, "notes": ""},
            {
                "metric": "potency_source_id_distribution",
                "value": str(source_distribution),
                "notes": "Distribution of src_id/source_id among Potency records.",
            },
        ]

        summary = pd.DataFrame(summary_rows)
        write_dataframe(
            summary,
            self.project_config.resolve("outputs.potency_exclusion_summary"),
        )
        write_dataframe(
            summary,
            self.project_config.resolve("outputs.potency_exclusion_table"),
        )

        self.logger.info("Potency endpoint records at endpoint-selection stage: %s", len(potency))
        self.logger.info("Potency DOI coverage: %s%%", doi_coverage)

    def _write_audit_table(self) -> None:
        """Write Supplementary Table S0 curation audit."""
        audit_df = pd.DataFrame(self.audit_rows)
        write_dataframe(
            audit_df,
            self.project_config.resolve("outputs.curation_audit_table"),
        )
        self.logger.info("Audit table written: %s rows", len(audit_df))

    def _write_dataset_summary(
        self,
        df_raw: pd.DataFrame,
        df_screened: pd.DataFrame,
        df_core: pd.DataFrame,
        df_validity_warning: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Write manuscript Table 1 dataset summary."""
        rows = [
            self._summary_row("Raw", df_raw, columns),
            self._summary_row("Tier-A Screened", df_screened, columns),
            self._summary_row("Validity-warning records", df_validity_warning, columns),
            self._summary_row("Tier-A Core", df_core, columns),
        ]
        summary = pd.DataFrame(rows)
        write_dataframe(
            summary,
            self.project_config.resolve("outputs.table_1_dataset_summary"),
        )

    def _summary_row(
        self,
        label: str,
        df: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> dict[str, Any]:
        """Build one dataset summary row."""
        doi_coverage = None
        if columns.doi is not None and len(df) > 0:
            doi_coverage = round(100 * non_empty_mask(df[columns.doi]).sum() / len(df), 4)

        return {
            "dataset_layer": label,
            "records": len(df),
            "unique_molecules": count_unique_non_null(df, columns.molecule_id),
            "unique_targets": count_unique_non_null(df, columns.target_id),
            "doi_coverage_percent": doi_coverage,
        }

    def _write_uniprot_coverage_report(
        self,
        df_core: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Write UniProt coverage report for core targets."""
        target_cols = [
            col for col in [
                columns.target_id,
                columns.target_name,
                columns.organism,
                columns.uniprot,
            ]
            if col is not None
        ]

        target_table = df_core[target_cols].drop_duplicates().copy()

        if columns.uniprot is not None:
            target_table["has_uniprot"] = non_empty_mask(target_table[columns.uniprot])
        else:
            target_table["has_uniprot"] = False

        write_dataframe(
            target_table,
            self.project_config.resolve("outputs.uniprot_coverage_report"),
        )

        total_targets = count_unique_non_null(df_core, columns.target_id)
        targets_with_uniprot = int(
            target_table.loc[target_table["has_uniprot"], columns.target_id].nunique()
        )

        self.logger.info(
            "UniProt coverage: %s/%s targets",
            targets_with_uniprot,
            total_targets,
        )

    def _write_target_organism_summary(
        self,
        df_core: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Write target organism summary."""
        if columns.organism is None:
            summary = pd.DataFrame(
                [{"target_organism": "NOT_AVAILABLE", "record_count": len(df_core)}]
            )
        else:
            summary = (
                df_core[columns.organism]
                .fillna("MISSING")
                .astype(str)
                .str.strip()
                .value_counts()
                .rename_axis("target_organism")
                .reset_index(name="record_count")
            )

        write_dataframe(
            summary,
            self.project_config.resolve("outputs.target_organism_summary"),
        )

    def _write_dataset_freeze_report(
        self,
        df_raw: pd.DataFrame,
        df_screened: pd.DataFrame,
        df_core: pd.DataFrame,
        df_validity_warning: pd.DataFrame,
        columns: ResolvedColumns,
    ) -> None:
        """Write text dataset freeze report."""
        dataset_version = self.config["project"]["dataset_version"]

        report = f"""CoumarinBioBench-TierA Dataset Freeze Report

Status: P0 AUDIT COMPLETED

Dataset version: {dataset_version}

Raw records: {len(df_raw)}
Raw unique molecules: {count_unique_non_null(df_raw, columns.molecule_id)}
Raw unique targets: {count_unique_non_null(df_raw, columns.target_id)}

Tier-A Screened records: {len(df_screened)}
Tier-A Screened unique coumarins: {count_unique_non_null(df_screened, columns.molecule_id)}
Tier-A Screened protein targets: {count_unique_non_null(df_screened, columns.target_id)}

Validity-warning records removed: {len(df_validity_warning)}

Tier-A Core records: {len(df_core)}
Tier-A Core unique coumarins: {count_unique_non_null(df_core, columns.molecule_id)}
Tier-A Core protein targets: {count_unique_non_null(df_core, columns.target_id)}

Raw data file:
{self.project_config.raw_data}

Notes:
- Confidence score is treated as a quality characteristic, not as a record-removing filter.
- Potency endpoint exclusion is documented in Supplementary Table S1.
- Tier-A Core excludes non-null ChEMBL data_validity_comment records.
"""

        write_text(
            report,
            self.project_config.resolve("outputs.dataset_freeze_report"),
        )

    def _write_checksums(self) -> None:
        """Write SHA-256 checksums for important files."""
        files = [
            self.project_config.raw_data,
            self.project_config.resolve("outputs.tierA_screened"),
            self.project_config.resolve("outputs.tierA_core"),
            self.project_config.resolve("outputs.tierA_core_processed"),
            self.project_config.resolve("outputs.curation_audit_table"),
            self.project_config.resolve("outputs.table_1_dataset_summary"),
        ]

        rows = ["filename,sha256"]
        for path in files:
            if path.exists():
                rows.append(f"{path.relative_to(self.project_config.root)},{compute_sha256(path)}")

        write_text(
            "\n".join(rows) + "\n",
            self.project_config.resolve("outputs.checksum_file"),
        )


def run_project_audit() -> None:
    """Run P0 project audit from config.yaml."""
    project_config = load_config()
    auditor = ProjectAuditor(project_config)
    auditor.run()
