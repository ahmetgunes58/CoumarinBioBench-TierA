"""QSAR-readiness assessment utilities for CoumarinBioBench.

P5a assesses target-specific suitability for endpoint-specific QSAR modelling
using compound coverage, endpoint homogeneity, dominant-endpoint activity range
and DOI diversity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COMPOUND_ID_CANDIDATES: Tuple[str, ...] = (
    "molecule_chembl_id",
    "compound_chembl_id",
    "compound_id",
    "chembl_id",
)

TARGET_ID_CANDIDATES: Tuple[str, ...] = (
    "target_chembl_id",
    "target_id",
)

TARGET_NAME_CANDIDATES: Tuple[str, ...] = (
    "target_pref_name_full",
    "target_pref_name",
    "target_name",
)

TARGET_ORGANISM_CANDIDATES: Tuple[str, ...] = (
    "target_organism_full",
    "target_organism",
    "organism",
)

ENDPOINT_CANDIDATES: Tuple[str, ...] = (
    "standard_type",
    "endpoint",
    "endpoint_type",
)

PCHEMBL_CANDIDATES: Tuple[str, ...] = (
    "pchembl_value",
    "pActivity",
    "pactivity",
)

DOI_CANDIDATES: Tuple[str, ...] = (
    "doi",
    "document_doi",
)

YEAR_CANDIDATES: Tuple[str, ...] = (
    "year",
    "publication_year",
)

TARGET_FAMILY_CANDIDATES: Tuple[str, ...] = (
    "target_family",
    "family",
)

ORGANISM_GROUP_CANDIDATES: Tuple[str, ...] = (
    "organism_group",
    "human_nonhuman",
)


READINESS_TIER_ORDER: Tuple[str, ...] = (
    "Ready",
    "Curatable",
    "Exploratory",
    "Not Suitable",
)


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    """Resolve a column name using exact and punctuation-tolerant matching."""

    exact_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact_lookup:
            return str(exact_lookup[candidate.lower()])

    normalised_lookup = {_normalise_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalise_name(candidate)
        if key in normalised_lookup:
            return str(normalised_lookup[key])

    if required:
        expected = ", ".join(candidates)
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Could not resolve {label}. Expected one of [{expected}]. "
            f"Available columns: {available}"
        )
    return None


def _mode_or_missing(series: pd.Series) -> object:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        return pd.NA
    modes = clean.mode()
    if modes.empty:
        return clean.iloc[0]
    return modes.iloc[0]


def _sorted_unique_join(values: Iterable[object], sep: str = "; ") -> str:
    clean = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return sep.join(clean)


def _dominant_endpoint_and_homogeneity(series: pd.Series) -> Tuple[object, float, int]:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        return pd.NA, np.nan, 0

    counts = clean.value_counts()
    dominant = counts.index[0]
    dominant_count = int(counts.iloc[0])
    homogeneity = float(dominant_count / counts.sum())
    return dominant, homogeneity, dominant_count


def _numeric_range(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max() - values.min())


def _numeric_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.min())


def _numeric_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


def classify_readiness_tier(
    *,
    unique_compounds: int,
    endpoint_homogeneity: float,
    dominant_endpoint_pactivity_range: float,
    doi_count: int,
) -> str:
    """Classify target-specific QSAR readiness.

    Ready requires all high-confidence modelling criteria.
    Curatable requires moderate compound coverage and data consistency.
    Exploratory captures small but non-trivial target-specific subsets.
    Not Suitable captures targets with insufficient compound coverage.
    """

    homogeneity = 0.0 if pd.isna(endpoint_homogeneity) else float(endpoint_homogeneity)
    activity_range = (
        0.0
        if pd.isna(dominant_endpoint_pactivity_range)
        else float(dominant_endpoint_pactivity_range)
    )

    if (
        unique_compounds >= 100
        and homogeneity >= 0.80
        and activity_range >= 2.50
        and doi_count >= 5
    ):
        return "Ready"

    if (
        unique_compounds >= 40
        and homogeneity >= 0.60
        and activity_range >= 1.50
        and doi_count >= 3
    ):
        return "Curatable"

    if unique_compounds >= 20:
        return "Exploratory"

    return "Not Suitable"


def _criteria_flags(
    *,
    unique_compounds: int,
    endpoint_homogeneity: float,
    dominant_endpoint_pactivity_range: float,
    doi_count: int,
) -> Dict[str, int]:
    homogeneity = 0.0 if pd.isna(endpoint_homogeneity) else float(endpoint_homogeneity)
    activity_range = (
        0.0
        if pd.isna(dominant_endpoint_pactivity_range)
        else float(dominant_endpoint_pactivity_range)
    )

    return {
        "ready_compound_count_pass": int(unique_compounds >= 100),
        "ready_endpoint_homogeneity_pass": int(homogeneity >= 0.80),
        "ready_activity_range_pass": int(activity_range >= 2.50),
        "ready_doi_diversity_pass": int(doi_count >= 5),
        "curatable_compound_count_pass": int(unique_compounds >= 40),
        "curatable_endpoint_homogeneity_pass": int(homogeneity >= 0.60),
        "curatable_activity_range_pass": int(activity_range >= 1.50),
        "curatable_doi_diversity_pass": int(doi_count >= 3),
    }


def build_target_qsar_readiness(
    core_df: pd.DataFrame,
    target_annotation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build one-row-per-target QSAR-readiness table."""

    compound_col = resolve_column(
        core_df,
        COMPOUND_ID_CANDIDATES,
        label="compound identifier",
    )
    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )
    target_name_col = resolve_column(
        core_df,
        TARGET_NAME_CANDIDATES,
        label="target preferred name",
    )
    target_organism_col = resolve_column(
        core_df,
        TARGET_ORGANISM_CANDIDATES,
        label="target organism",
    )
    endpoint_col = resolve_column(
        core_df,
        ENDPOINT_CANDIDATES,
        label="endpoint",
    )
    pchembl_col = resolve_column(
        core_df,
        PCHEMBL_CANDIDATES,
        label="pChEMBL value",
    )
    doi_col = resolve_column(
        core_df,
        DOI_CANDIDATES,
        required=False,
        label="DOI",
    )
    year_col = resolve_column(
        core_df,
        YEAR_CANDIDATES,
        required=False,
        label="publication year",
    )

    annotation = None
    if target_annotation_df is not None and not target_annotation_df.empty:
        annotation = target_annotation_df.copy()

    rows: List[Dict[str, object]] = []

    for target_id, group in core_df.groupby(target_col, dropna=False):
        target_name = _mode_or_missing(group[target_name_col])
        target_organism = _mode_or_missing(group[target_organism_col])

        dominant_endpoint, endpoint_homogeneity, dominant_endpoint_record_count = (
            _dominant_endpoint_and_homogeneity(group[endpoint_col])
        )

        dominant_group = group[group[endpoint_col].astype(str) == str(dominant_endpoint)].copy()

        doi_values = (
            group[doi_col].dropna().astype(str).str.strip()
            if doi_col
            else pd.Series(dtype=str)
        )
        doi_values = doi_values[doi_values != ""]

        dominant_doi_values = (
            dominant_group[doi_col].dropna().astype(str).str.strip()
            if doi_col
            else pd.Series(dtype=str)
        )
        dominant_doi_values = dominant_doi_values[dominant_doi_values != ""]

        year_values = (
            pd.to_numeric(group[year_col], errors="coerce").dropna()
            if year_col
            else pd.Series(dtype=float)
        )

        unique_compounds = int(group[compound_col].nunique())
        dominant_unique_compounds = int(dominant_group[compound_col].nunique())

        pactivity_min = _numeric_min(group[pchembl_col])
        pactivity_max = _numeric_max(group[pchembl_col])
        pactivity_range = _numeric_range(group[pchembl_col])

        dominant_pactivity_min = _numeric_min(dominant_group[pchembl_col])
        dominant_pactivity_max = _numeric_max(dominant_group[pchembl_col])
        dominant_pactivity_range = _numeric_range(dominant_group[pchembl_col])

        doi_count = int(doi_values.nunique())
        dominant_doi_count = int(dominant_doi_values.nunique())

        tier = classify_readiness_tier(
            unique_compounds=unique_compounds,
            endpoint_homogeneity=endpoint_homogeneity,
            dominant_endpoint_pactivity_range=dominant_pactivity_range,
            doi_count=doi_count,
        )

        flags = _criteria_flags(
            unique_compounds=unique_compounds,
            endpoint_homogeneity=endpoint_homogeneity,
            dominant_endpoint_pactivity_range=dominant_pactivity_range,
            doi_count=doi_count,
        )

        row = {
            "target_chembl_id": target_id,
            "target_name": target_name,
            "target_organism": target_organism,
            "record_count": int(len(group)),
            "unique_compounds": unique_compounds,
            "endpoint_types": _sorted_unique_join(group[endpoint_col]),
            "dominant_endpoint": dominant_endpoint,
            "dominant_endpoint_record_count": int(dominant_endpoint_record_count),
            "dominant_endpoint_unique_compounds": dominant_unique_compounds,
            "endpoint_homogeneity": endpoint_homogeneity,
            "doi_count": doi_count,
            "dominant_endpoint_doi_count": dominant_doi_count,
            "year_min": int(year_values.min()) if not year_values.empty else pd.NA,
            "year_max": int(year_values.max()) if not year_values.empty else pd.NA,
            "pactivity_min": pactivity_min,
            "pactivity_max": pactivity_max,
            "pactivity_range": pactivity_range,
            "dominant_endpoint_pactivity_min": dominant_pactivity_min,
            "dominant_endpoint_pactivity_max": dominant_pactivity_max,
            "dominant_endpoint_pactivity_range": dominant_pactivity_range,
            "readiness_tier": tier,
            **flags,
        }

        rows.append(row)

    readiness = pd.DataFrame(rows)

    if annotation is not None:
        annotation_cols = [
            col
            for col in [
                "target_chembl_id",
                "target_family",
                "organism_group",
                "uniprot_accession",
            ]
            if col in annotation.columns
        ]
        readiness = readiness.merge(
            annotation[annotation_cols].drop_duplicates("target_chembl_id"),
            on="target_chembl_id",
            how="left",
            validate="one_to_one",
        )

    if "target_family" not in readiness.columns:
        readiness["target_family"] = pd.NA
    if "organism_group" not in readiness.columns:
        readiness["organism_group"] = pd.NA
    if "uniprot_accession" not in readiness.columns:
        readiness["uniprot_accession"] = pd.NA

    tier_order_map = {tier: idx for idx, tier in enumerate(READINESS_TIER_ORDER)}
    readiness["tier_order"] = readiness["readiness_tier"].map(tier_order_map)

    readiness["ready_criteria_pass_count"] = readiness[
        [
            "ready_compound_count_pass",
            "ready_endpoint_homogeneity_pass",
            "ready_activity_range_pass",
            "ready_doi_diversity_pass",
        ]
    ].sum(axis=1)

    readiness["curatable_criteria_pass_count"] = readiness[
        [
            "curatable_compound_count_pass",
            "curatable_endpoint_homogeneity_pass",
            "curatable_activity_range_pass",
            "curatable_doi_diversity_pass",
        ]
    ].sum(axis=1)

    readiness = readiness.sort_values(
        [
            "tier_order",
            "unique_compounds",
            "endpoint_homogeneity",
            "dominant_endpoint_pactivity_range",
            "doi_count",
            "target_chembl_id",
        ],
        ascending=[True, False, False, False, False, True],
    ).drop(columns=["tier_order"])

    return readiness.reset_index(drop=True)


def build_readiness_tier_summary(readiness_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise targets across readiness tiers."""

    rows = []
    total_targets = int(len(readiness_df))

    for tier in READINESS_TIER_ORDER:
        subset = readiness_df[readiness_df["readiness_tier"] == tier]
        rows.append(
            {
                "readiness_tier": tier,
                "target_count": int(len(subset)),
                "percent_of_targets": (
                    float(len(subset) / total_targets * 100.0)
                    if total_targets
                    else np.nan
                ),
                "median_unique_compounds": (
                    float(subset["unique_compounds"].median())
                    if len(subset)
                    else np.nan
                ),
                "median_endpoint_homogeneity": (
                    float(subset["endpoint_homogeneity"].median())
                    if len(subset)
                    else np.nan
                ),
                "median_dominant_endpoint_pactivity_range": (
                    float(subset["dominant_endpoint_pactivity_range"].median())
                    if len(subset)
                    else np.nan
                ),
                "median_doi_count": (
                    float(subset["doi_count"].median())
                    if len(subset)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_readiness_family_summary(readiness_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise readiness tiers by target family."""

    if "target_family" not in readiness_df.columns:
        raise KeyError("readiness_df must contain target_family.")

    grouped = (
        readiness_df
        .groupby(["target_family", "readiness_tier"], dropna=False)
        .size()
        .reset_index(name="target_count")
    )

    totals = (
        readiness_df
        .groupby("target_family", dropna=False)
        .size()
        .rename("family_target_count")
        .reset_index()
    )

    grouped = grouped.merge(totals, on="target_family", how="left")
    grouped["percent_of_family_targets"] = (
        grouped["target_count"] / grouped["family_target_count"] * 100.0
    )

    tier_order_map = {tier: idx for idx, tier in enumerate(READINESS_TIER_ORDER)}
    grouped["tier_order"] = grouped["readiness_tier"].map(tier_order_map)

    return (
        grouped
        .sort_values(["target_family", "tier_order"])
        .drop(columns=["tier_order"])
        .reset_index(drop=True)
    )


def build_ready_target_table(readiness_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Ready-tier targets for main-text inspection."""

    columns = [
        "target_chembl_id",
        "target_name",
        "target_organism",
        "target_family",
        "organism_group",
        "unique_compounds",
        "record_count",
        "dominant_endpoint",
        "endpoint_homogeneity",
        "dominant_endpoint_pactivity_range",
        "doi_count",
        "year_min",
        "year_max",
    ]

    available = [col for col in columns if col in readiness_df.columns]

    ready = readiness_df[readiness_df["readiness_tier"] == "Ready"].copy()

    return (
        ready[available]
        .sort_values(
            [
                "unique_compounds",
                "endpoint_homogeneity",
                "dominant_endpoint_pactivity_range",
                "doi_count",
            ],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def validate_readiness_outputs(
    readiness_df: pd.DataFrame,
    core_df: pd.DataFrame,
) -> None:
    """Validate readiness outputs."""

    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )

    expected_targets = int(core_df[target_col].nunique())
    observed_targets = int(readiness_df["target_chembl_id"].nunique())

    if expected_targets != observed_targets:
        raise ValueError(
            f"Target count mismatch: expected {expected_targets}, "
            f"observed {observed_targets}."
        )

    if readiness_df["readiness_tier"].isna().any():
        raise ValueError("Some targets are missing readiness_tier.")

    unexpected_tiers = sorted(set(readiness_df["readiness_tier"]) - set(READINESS_TIER_ORDER))
    if unexpected_tiers:
        raise ValueError("Unexpected readiness tiers: " + "; ".join(unexpected_tiers))

    if readiness_df["dominant_endpoint"].isna().any():
        raise ValueError("Some targets are missing dominant_endpoint.")

    if readiness_df["unique_compounds"].isna().any():
        raise ValueError("Some targets are missing unique_compounds.")
