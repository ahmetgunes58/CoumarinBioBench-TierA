"""Compound-target network utilities for CoumarinBioBench-TierA.

This module builds reusable network and provenance tables from the Tier-A Core
benchmark. The outputs are used for Figure 2, Results Section 3.3, and
Supplementary Table S5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from coumarinbiobench.config import ProjectConfig, load_config
from coumarinbiobench.io_utils import read_csv_safely, write_dataframe, write_text
from coumarinbiobench.logging_utils import setup_logger
from coumarinbiobench.validation import (
    non_empty_mask,
    resolve_column,
    resolve_optional_column,
)


@dataclass
class NetworkPaths:
    """Input and output paths for P2 network table generation."""

    tier_a_core: Path
    compound_target_edges: Path
    compound_degree_table: Path
    target_richness_table: Path
    degree_distribution_table: Path
    endpoint_composition_table: Path
    network_overview_table: Path
    high_degree_pains_table: Path
    network_report: Path


@dataclass
class NetworkColumns:
    """Resolved column names for the Tier-A Core network analysis."""

    molecule_id: str
    target_id: str
    standard_type: str

    smiles: str | None = None
    target_name: str | None = None
    organism: str | None = None
    uniprot: str | None = None
    doi: str | None = None
    year: str | None = None
    pchembl: str | None = None
    confidence_score: str | None = None


def _resolve_network_paths(project_config: ProjectConfig) -> NetworkPaths:
    """Resolve input and output paths for P2.

    Parameters
    ----------
    project_config : ProjectConfig
        Loaded project configuration.

    Returns
    -------
    NetworkPaths
        Resolved paths.
    """
    root = project_config.root
    processed_dir = root / project_config.config["paths"]["processed_dir"]
    metadata_dir = root / project_config.config["paths"]["metadata_dir"]
    tables_dir = root / project_config.config["paths"]["tables_dir"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return NetworkPaths(
        tier_a_core=project_config.resolve("outputs.tierA_core_processed"),
        compound_target_edges=processed_dir / "compound_target_edges.csv",
        compound_degree_table=processed_dir / "compound_degree_table.csv",
        target_richness_table=processed_dir / "target_richness_table.csv",
        degree_distribution_table=processed_dir / "degree_distribution_table.csv",
        endpoint_composition_table=processed_dir / "endpoint_composition_table.csv",
        network_overview_table=tables_dir / "Table_P2_network_overview.csv",
        high_degree_pains_table=(
            tables_dir / "Supplementary_Table_S5_high_degree_pains_provenance.csv"
        ),
        network_report=metadata_dir / "p2_network_report.txt",
    )


def _resolve_network_columns(
    df: pd.DataFrame,
    project_config: ProjectConfig,
) -> NetworkColumns:
    """Resolve required and optional columns for network analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Tier-A Core dataframe.
    project_config : ProjectConfig
        Loaded project configuration.

    Returns
    -------
    NetworkColumns
        Resolved column names.
    """
    cfg = project_config.config["columns"]

    return NetworkColumns(
        molecule_id=resolve_column(
            df,
            cfg["molecule_id_candidates"],
            "molecule_id",
        ),
        target_id=resolve_column(
            df,
            cfg["target_id_candidates"],
            "target_id",
        ),
        standard_type=resolve_column(
            df,
            cfg["standard_type_candidates"],
            "standard_type",
        ),
        smiles=resolve_optional_column(
            df,
            cfg["smiles_candidates"],
            "smiles",
        ),
        target_name=resolve_optional_column(
            df,
            cfg["target_name_candidates"],
            "target_name",
        ),
        organism=resolve_optional_column(
            df,
            cfg["organism_candidates"],
            "organism",
        ),
        uniprot=resolve_optional_column(
            df,
            cfg["uniprot_candidates"],
            "uniprot",
        ),
        doi=resolve_optional_column(
            df,
            cfg["doi_candidates"],
            "doi",
        ),
        year=resolve_optional_column(
            df,
            cfg["year_candidates"],
            "year",
        ),
        pchembl=resolve_optional_column(
            df,
            cfg["pchembl_candidates"],
            "pchembl",
        ),
        confidence_score=resolve_optional_column(
            df,
            cfg["confidence_score_candidates"],
            "confidence_score",
        ),
    )


def _clean_string(value: Any) -> str:
    """Return a cleaned string representation.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    str
        Cleaned string. Empty string for null-like values.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _join_unique(values: pd.Series, max_items: int = 12) -> str:
    """Join unique non-empty values in a stable order.

    Parameters
    ----------
    values : pandas.Series
        Input series.
    max_items : int
        Maximum number of values to include before truncation.

    Returns
    -------
    str
        Semicolon-separated unique values.
    """
    unique_values = []
    seen = set()

    for value in values:
        cleaned = _clean_string(value)
        if cleaned and cleaned not in seen:
            unique_values.append(cleaned)
            seen.add(cleaned)

    if len(unique_values) > max_items:
        shown = unique_values[:max_items]
        shown.append(f"...(+{len(unique_values) - max_items} more)")
        return "; ".join(shown)

    return "; ".join(unique_values)


def _first_non_empty(values: pd.Series) -> str:
    """Return the first non-empty value from a series."""
    for value in values:
        cleaned = _clean_string(value)
        if cleaned:
            return cleaned
    return ""


def _dominant_value(values: pd.Series) -> str:
    """Return the most frequent non-empty value."""
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin(["", "nan", "none", "null"])]
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().idxmax())


def _non_empty_nunique(values: pd.Series) -> int:
    """Count unique non-empty values in a series."""
    if values.empty:
        return 0
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin(["", "nan", "none", "null"])]
    return int(cleaned.nunique())


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    """Convert a series to numeric values.

    Parameters
    ----------
    series : pandas.Series or None
        Input series.

    Returns
    -------
    pandas.Series
        Numeric series. Empty numeric series if input is None.
    """
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _year_min(values: pd.Series) -> Any:
    """Return minimum publication year if available."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return int(numeric.min())


def _year_max(values: pd.Series) -> Any:
    """Return maximum publication year if available."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return int(numeric.max())


def _pchembl_min(values: pd.Series) -> Any:
    """Return minimum pChEMBL value if available."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return round(float(numeric.min()), 4)


def _pchembl_max(values: pd.Series) -> Any:
    """Return maximum pChEMBL value if available."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return round(float(numeric.max()), 4)


def _pchembl_range(values: pd.Series) -> Any:
    """Return pChEMBL activity range if available."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return round(float(numeric.max() - numeric.min()), 4)


def _endpoint_homogeneity(values: pd.Series) -> float:
    """Compute dominant-endpoint fraction.

    Parameters
    ----------
    values : pandas.Series
        Endpoint labels.

    Returns
    -------
    float
        Dominant endpoint fraction between 0 and 1.
    """
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin(["", "nan", "none", "null"])]
    if cleaned.empty:
        return 0.0
    return round(float(cleaned.value_counts().max() / len(cleaned)), 4)


def _prepare_network_dataframe(
    df: pd.DataFrame,
    columns: NetworkColumns,
) -> pd.DataFrame:
    """Prepare standardised working columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Tier-A Core dataframe.
    columns : NetworkColumns
        Resolved column names.

    Returns
    -------
    pandas.DataFrame
        Working dataframe with internal underscore-prefixed columns.
    """
    work_df = df.copy()

    work_df["_molecule_id"] = work_df[columns.molecule_id].astype(str).str.strip()
    work_df["_target_id"] = work_df[columns.target_id].astype(str).str.strip()
    work_df["_standard_type"] = work_df[columns.standard_type].astype(str).str.strip()

    work_df["_smiles"] = (
        work_df[columns.smiles].astype(str).str.strip()
        if columns.smiles is not None
        else ""
    )
    work_df["_target_name"] = (
        work_df[columns.target_name].astype(str).str.strip()
        if columns.target_name is not None
        else ""
    )
    work_df["_organism"] = (
        work_df[columns.organism].astype(str).str.strip()
        if columns.organism is not None
        else ""
    )
    work_df["_uniprot"] = (
        work_df[columns.uniprot].astype(str).str.strip()
        if columns.uniprot is not None
        else ""
    )
    work_df["_doi"] = (
        work_df[columns.doi].astype(str).str.strip()
        if columns.doi is not None
        else ""
    )
    work_df["_year"] = (
        pd.to_numeric(work_df[columns.year], errors="coerce")
        if columns.year is not None
        else pd.NA
    )
    work_df["_pchembl"] = (
        pd.to_numeric(work_df[columns.pchembl], errors="coerce")
        if columns.pchembl is not None
        else pd.NA
    )
    work_df["_confidence_score"] = (
        work_df[columns.confidence_score]
        if columns.confidence_score is not None
        else pd.NA
    )

    required_mask = (
        non_empty_mask(work_df["_molecule_id"])
        & non_empty_mask(work_df["_target_id"])
        & non_empty_mask(work_df["_standard_type"])
    )

    return work_df.loc[required_mask].copy()


def _build_compound_target_edges(work_df: pd.DataFrame) -> pd.DataFrame:
    """Build unique compound-target edge table."""
    grouped = work_df.groupby(["_molecule_id", "_target_id"], dropna=False)

    edges = grouped.agg(
        record_count=("_target_id", "size"),
        canonical_smiles=("_smiles", _first_non_empty),
        target_name=("_target_name", _first_non_empty),
        target_organism=("_organism", _first_non_empty),
        uniprot_accession=("_uniprot", _first_non_empty),
        endpoint_types=("_standard_type", _join_unique),
        dominant_endpoint=("_standard_type", _dominant_value),
        doi_count=("_doi", _non_empty_nunique),
        year_min=("_year", _year_min),
        year_max=("_year", _year_max),
        pchembl_min=("_pchembl", _pchembl_min),
        pchembl_max=("_pchembl", _pchembl_max),
        pchembl_range=("_pchembl", _pchembl_range),
    ).reset_index()

    edges = edges.rename(
        columns={
            "_molecule_id": "molecule_chembl_id",
            "_target_id": "target_chembl_id",
        }
    )

    edges = edges.sort_values(
        ["record_count", "molecule_chembl_id", "target_chembl_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return edges


def _build_compound_degree_table(work_df: pd.DataFrame) -> pd.DataFrame:
    """Build compound-level target-degree table."""
    grouped = work_df.groupby("_molecule_id", dropna=False)

    degree_table = grouped.agg(
        record_count=("_molecule_id", "size"),
        chembl_target_degree=("_target_id", "nunique"),
        canonical_smiles=("_smiles", _first_non_empty),
        endpoint_types=("_standard_type", _join_unique),
        dominant_endpoint=("_standard_type", _dominant_value),
        doi_count=("_doi", _non_empty_nunique),
        year_min=("_year", _year_min),
        year_max=("_year", _year_max),
        pchembl_min=("_pchembl", _pchembl_min),
        pchembl_max=("_pchembl", _pchembl_max),
        pchembl_range=("_pchembl", _pchembl_range),
        target_ids=("_target_id", _join_unique),
        target_names=("_target_name", _join_unique),
        target_organisms=("_organism", _join_unique),
    ).reset_index()

    degree_table = degree_table.rename(
        columns={"_molecule_id": "molecule_chembl_id"}
    )

    degree_table = degree_table.sort_values(
        ["chembl_target_degree", "record_count", "molecule_chembl_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return degree_table


def _build_target_richness_table(work_df: pd.DataFrame) -> pd.DataFrame:
    """Build target-level richness and endpoint table."""
    grouped = work_df.groupby("_target_id", dropna=False)

    target_table = grouped.agg(
        record_count=("_target_id", "size"),
        unique_compounds=("_molecule_id", "nunique"),
        target_name=("_target_name", _first_non_empty),
        target_organism=("_organism", _first_non_empty),
        uniprot_accession=("_uniprot", _first_non_empty),
        endpoint_types=("_standard_type", _join_unique),
        dominant_endpoint=("_standard_type", _dominant_value),
        endpoint_homogeneity=("_standard_type", _endpoint_homogeneity),
        doi_count=("_doi", _non_empty_nunique),
        year_min=("_year", _year_min),
        year_max=("_year", _year_max),
        pchembl_min=("_pchembl", _pchembl_min),
        pchembl_max=("_pchembl", _pchembl_max),
        pchembl_range=("_pchembl", _pchembl_range),
    ).reset_index()

    target_table = target_table.rename(
        columns={"_target_id": "target_chembl_id"}
    )

    target_table = target_table.sort_values(
        ["unique_compounds", "record_count", "target_chembl_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return target_table


def _build_degree_distribution_table(
    degree_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build binned target-degree distribution table."""
    degree = degree_table["chembl_target_degree"]

    bins = [
        ("1", int((degree == 1).sum())),
        ("2", int((degree == 2).sum())),
        ("3-4", int(((degree >= 3) & (degree <= 4)).sum())),
        ("5-9", int(((degree >= 5) & (degree <= 9)).sum())),
        (">=10", int((degree >= 10).sum())),
    ]

    total = int(len(degree_table))
    rows = []
    for label, count in bins:
        rows.append(
            {
                "degree_bin": label,
                "compound_count": count,
                "percent_of_compounds": round(100 * count / total, 4)
                if total > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _build_endpoint_composition_table(work_df: pd.DataFrame) -> pd.DataFrame:
    """Build endpoint composition table."""
    endpoint_counts = (
        work_df["_standard_type"]
        .value_counts()
        .rename_axis("endpoint")
        .reset_index(name="record_count")
    )
    endpoint_counts["percent_of_core_records"] = (
        endpoint_counts["record_count"] / len(work_df) * 100
    ).round(4)
    return endpoint_counts


def _try_compute_pains_alerts(smiles: str) -> str:
    """Compute PAINS alerts for one SMILES string.

    Parameters
    ----------
    smiles : str
        Canonical SMILES.

    Returns
    -------
    str
        Semicolon-separated PAINS alerts, ``none`` if no alert, or an explicit
        non-evaluated flag if RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    except ImportError:
        return "NOT_EVALUATED_RDKit_UNAVAILABLE"

    cleaned = _clean_string(smiles)
    if not cleaned:
        return "NO_SMILES"

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return "INVALID_SMILES"

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog(params)

    matches = catalog.GetMatches(mol)
    if not matches:
        return "none"

    descriptions = sorted({match.GetDescription() for match in matches})
    return "; ".join(descriptions)


def _build_high_degree_table(
    compound_degree_table: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Build high-degree compound provenance table.

    Parameters
    ----------
    compound_degree_table : pandas.DataFrame
        Compound-level target-degree table.
    top_n : int
        Number of highest-degree compounds to retain.

    Returns
    -------
    pandas.DataFrame
        High-degree provenance table.
    """
    high_degree = compound_degree_table.head(top_n).copy()
    high_degree["pains_alerts"] = high_degree["canonical_smiles"].apply(
        _try_compute_pains_alerts
    )

    columns = [
        "molecule_chembl_id",
        "chembl_target_degree",
        "record_count",
        "doi_count",
        "year_min",
        "year_max",
        "pchembl_min",
        "pchembl_max",
        "pchembl_range",
        "dominant_endpoint",
        "endpoint_types",
        "pains_alerts",
        "target_ids",
        "target_names",
        "target_organisms",
        "canonical_smiles",
    ]

    return high_degree[columns]


def _build_network_overview_table(
    work_df: pd.DataFrame,
    edges: pd.DataFrame,
    compound_degree_table: pd.DataFrame,
    target_richness_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row network overview table."""
    total_compounds = int(compound_degree_table["molecule_chembl_id"].nunique())
    single_target = int(
        (compound_degree_table["chembl_target_degree"] == 1).sum()
    )
    multi_target = int(
        (compound_degree_table["chembl_target_degree"] >= 2).sum()
    )
    five_or_more = int(
        (compound_degree_table["chembl_target_degree"] >= 5).sum()
    )
    ten_or_more = int(
        (compound_degree_table["chembl_target_degree"] >= 10).sum()
    )

    max_degree = int(compound_degree_table["chembl_target_degree"].max())
    max_degree_molecule = str(
        compound_degree_table.iloc[0]["molecule_chembl_id"]
    )

    top_target = target_richness_table.iloc[0]

    rows = [
        {
            "metric": "core_records",
            "value": int(len(work_df)),
            "notes": "Tier-A Core bioactivity records.",
        },
        {
            "metric": "unique_compounds",
            "value": total_compounds,
            "notes": "Unique molecule ChEMBL IDs in Tier-A Core.",
        },
        {
            "metric": "unique_targets",
            "value": int(target_richness_table["target_chembl_id"].nunique()),
            "notes": "Unique ChEMBL protein targets.",
        },
        {
            "metric": "unique_compound_target_edges",
            "value": int(len(edges)),
            "notes": "Unique molecule-target pairs.",
        },
        {
            "metric": "single_target_compounds",
            "value": single_target,
            "notes": "Compounds annotated to one ChEMBL target.",
        },
        {
            "metric": "single_target_percent",
            "value": round(100 * single_target / total_compounds, 4),
            "notes": "Percentage of compounds annotated to one ChEMBL target.",
        },
        {
            "metric": "multi_target_compounds",
            "value": multi_target,
            "notes": "Compounds annotated to at least two ChEMBL targets.",
        },
        {
            "metric": "multi_target_percent",
            "value": round(100 * multi_target / total_compounds, 4),
            "notes": "Percentage of compounds annotated to at least two targets.",
        },
        {
            "metric": "compounds_with_degree_ge_5",
            "value": five_or_more,
            "notes": "Compounds annotated to five or more targets.",
        },
        {
            "metric": "compounds_with_degree_ge_10",
            "value": ten_or_more,
            "notes": "Compounds annotated to ten or more targets.",
        },
        {
            "metric": "max_chembl_target_degree",
            "value": max_degree,
            "notes": f"Highest degree molecule: {max_degree_molecule}.",
        },
        {
            "metric": "top_target_by_unique_compounds",
            "value": str(top_target["target_chembl_id"]),
            "notes": (
                f"{top_target['target_name']} | "
                f"{top_target['unique_compounds']} compounds | "
                f"{top_target['record_count']} records."
            ),
        },
    ]

    return pd.DataFrame(rows)


def _write_network_report(
    paths: NetworkPaths,
    overview_table: pd.DataFrame,
    degree_distribution_table: pd.DataFrame,
    target_richness_table: pd.DataFrame,
) -> None:
    """Write plain-text P2 network report."""
    overview_lookup = dict(
        zip(overview_table["metric"], overview_table["value"])
    )

    report = f"""CoumarinBioBench-TierA P2 Network Report

Status: P2 NETWORK TABLES COMPLETED

Core records: {overview_lookup.get("core_records")}
Unique compounds: {overview_lookup.get("unique_compounds")}
Unique protein targets: {overview_lookup.get("unique_targets")}
Unique compound-target edges: {overview_lookup.get("unique_compound_target_edges")}

Single-target compounds: {overview_lookup.get("single_target_compounds")}
Single-target percent: {overview_lookup.get("single_target_percent")}%

Multi-target compounds: {overview_lookup.get("multi_target_compounds")}
Multi-target percent: {overview_lookup.get("multi_target_percent")}%

Compounds with degree >= 5: {overview_lookup.get("compounds_with_degree_ge_5")}
Compounds with degree >= 10: {overview_lookup.get("compounds_with_degree_ge_10")}

Maximum ChEMBL-target degree: {overview_lookup.get("max_chembl_target_degree")}

Top target by unique compound count:
{target_richness_table.iloc[0].to_dict()}

Degree distribution:
{degree_distribution_table.to_string(index=False)}
"""

    write_text(report, paths.network_report)


def build_compound_target_network_tables(
    project_config: ProjectConfig | None = None,
    top_n_high_degree: int = 20,
) -> None:
    """Build P2 compound-target network and provenance tables.

    Parameters
    ----------
    project_config : ProjectConfig, optional
        Loaded project configuration. If None, ``config.yaml`` is loaded.
    top_n_high_degree : int
        Number of high-degree compounds to include in Supplementary Table S5.
    """
    project_config = project_config or load_config()
    logger = setup_logger(
        name="coumarinbiobench.network",
        log_dir=project_config.logs_dir,
        prefix="03_build_compound_target_network",
    )

    logger.info("Starting P2 compound-target network table generation.")
    paths = _resolve_network_paths(project_config)

    core_df = read_csv_safely(paths.tier_a_core)
    logger.info("Tier-A Core loaded: %s records", len(core_df))

    columns = _resolve_network_columns(core_df, project_config)
    logger.info("Resolved network columns:")
    for field_name, value in columns.__dict__.items():
        logger.info("  %s -> %s", field_name, value)

    work_df = _prepare_network_dataframe(core_df, columns)
    logger.info("Working network dataframe: %s valid records", len(work_df))

    edges = _build_compound_target_edges(work_df)
    compound_degree_table = _build_compound_degree_table(work_df)
    target_richness_table = _build_target_richness_table(work_df)
    degree_distribution_table = _build_degree_distribution_table(
        compound_degree_table
    )
    endpoint_composition_table = _build_endpoint_composition_table(work_df)
    high_degree_table = _build_high_degree_table(
        compound_degree_table,
        top_n=top_n_high_degree,
    )
    overview_table = _build_network_overview_table(
        work_df=work_df,
        edges=edges,
        compound_degree_table=compound_degree_table,
        target_richness_table=target_richness_table,
    )

    write_dataframe(edges, paths.compound_target_edges)
    write_dataframe(compound_degree_table, paths.compound_degree_table)
    write_dataframe(target_richness_table, paths.target_richness_table)
    write_dataframe(degree_distribution_table, paths.degree_distribution_table)
    write_dataframe(endpoint_composition_table, paths.endpoint_composition_table)
    write_dataframe(high_degree_table, paths.high_degree_pains_table)
    write_dataframe(overview_table, paths.network_overview_table)

    _write_network_report(
        paths=paths,
        overview_table=overview_table,
        degree_distribution_table=degree_distribution_table,
        target_richness_table=target_richness_table,
    )

    logger.info("compound_target_edges written: %s rows", len(edges))
    logger.info(
        "compound_degree_table written: %s rows",
        len(compound_degree_table),
    )
    logger.info(
        "target_richness_table written: %s rows",
        len(target_richness_table),
    )
    logger.info(
        "high_degree_pains_table written: %s rows",
        len(high_degree_table),
    )

    overview_lookup = dict(zip(overview_table["metric"], overview_table["value"]))
    logger.info(
        "Unique compound-target edges: %s",
        overview_lookup["unique_compound_target_edges"],
    )
    logger.info(
        "Single-target compounds: %s (%s%%)",
        overview_lookup["single_target_compounds"],
        overview_lookup["single_target_percent"],
    )
    logger.info(
        "Multi-target compounds: %s (%s%%)",
        overview_lookup["multi_target_compounds"],
        overview_lookup["multi_target_percent"],
    )
    logger.info(
        "Maximum ChEMBL-target degree: %s",
        overview_lookup["max_chembl_target_degree"],
    )
    logger.info("P2 compound-target network table generation completed.")
