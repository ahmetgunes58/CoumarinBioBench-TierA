"""Activity-cliff utility preparation for CoumarinBioBench.

P6a prepares clean dominant-endpoint compound-level tables for a limited
activity-cliff demonstration using three Ready-tier targets:
human MAO-B, human CA9 and human AChE.

No MMP or SALI analysis is performed in P6a.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CASE_STUDY_TARGETS: Tuple[Dict[str, str], ...] = (
    {
        "case_id": "MAOB_HUMAN_IC50",
        "target_chembl_id": "CHEMBL2039",
        "target_label": "Human MAO-B",
        "required_endpoint": "IC50",
        "priority": "primary",
    },
    {
        "case_id": "CA9_HUMAN_KI",
        "target_chembl_id": "CHEMBL3594",
        "target_label": "Human CA9",
        "required_endpoint": "Ki",
        "priority": "primary",
    },
    {
        "case_id": "ACHE_HUMAN_IC50",
        "target_chembl_id": "CHEMBL220",
        "target_label": "Human AChE",
        "required_endpoint": "IC50",
        "priority": "primary",
    },
)


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

SMILES_CANDIDATES: Tuple[str, ...] = (
    "canonical_smiles",
    "smiles",
    "canonical_SMILES",
)

INCHIKEY_CANDIDATES: Tuple[str, ...] = (
    "standard_inchi_key",
    "RDKit_InchiKey",
    "inchi_key",
    "inchikey",
)

DOI_CANDIDATES: Tuple[str, ...] = (
    "doi",
    "document_doi",
)

YEAR_CANDIDATES: Tuple[str, ...] = (
    "year",
    "publication_year",
)

RECORD_ID_CANDIDATES: Tuple[str, ...] = (
    "activity_id",
    "record_id",
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


def _mode_or_missing(series: pd.Series | pd.DataFrame) -> object:
    """Return the most frequent non-empty value.

    Handles duplicate column selections defensively: if a duplicate column name
    returns a DataFrame instead of a Series, the first non-null value across
    duplicate columns is used row-wise.
    """

    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.NA
        series = series.bfill(axis=1).iloc[:, 0]

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


def _numeric_min(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def _numeric_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def _numeric_range(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max() - values.min()) if not values.empty else np.nan


def _numeric_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _numeric_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _safe_int(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    try:
        return int(float(value))
    except Exception:
        return pd.NA


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate column names by taking the first non-null value row-wise.

    This protects against source tables that contain both compact and *_full
    versions of the same metadata field after harmonised renaming.
    """

    if not df.columns.duplicated().any():
        return df

    out = pd.DataFrame(index=df.index)

    for col in dict.fromkeys(df.columns):
        same = df.loc[:, df.columns == col]

        if isinstance(same, pd.Series):
            out[col] = same
        elif same.shape[1] == 1:
            out[col] = same.iloc[:, 0]
        else:
            out[col] = same.bfill(axis=1).iloc[:, 0]

    return out


def get_case_study_definitions() -> pd.DataFrame:
    """Return selected P6 case-study targets as a dataframe."""

    return pd.DataFrame(list(CASE_STUDY_TARGETS))


def prepare_case_study_activity_records(
    core_df: pd.DataFrame,
    *,
    case_targets: Sequence[Dict[str, str]] = CASE_STUDY_TARGETS,
) -> pd.DataFrame:
    """Filter Tier-A Core records to selected target + dominant endpoint subsets."""

    target_col = resolve_column(core_df, TARGET_ID_CANDIDATES, label="target identifier")
    endpoint_col = resolve_column(core_df, ENDPOINT_CANDIDATES, label="endpoint")
    compound_col = resolve_column(core_df, COMPOUND_ID_CANDIDATES, label="compound identifier")
    pchembl_col = resolve_column(core_df, PCHEMBL_CANDIDATES, label="pChEMBL value")
    smiles_col = resolve_column(core_df, SMILES_CANDIDATES, label="canonical SMILES")

    target_name_col = resolve_column(
        core_df,
        TARGET_NAME_CANDIDATES,
        required=False,
        label="target preferred name",
    )
    target_organism_col = resolve_column(
        core_df,
        TARGET_ORGANISM_CANDIDATES,
        required=False,
        label="target organism",
    )
    inchikey_col = resolve_column(
        core_df,
        INCHIKEY_CANDIDATES,
        required=False,
        label="InChIKey",
    )
    doi_col = resolve_column(core_df, DOI_CANDIDATES, required=False, label="DOI")
    year_col = resolve_column(core_df, YEAR_CANDIDATES, required=False, label="year")
    record_id_col = resolve_column(
        core_df,
        RECORD_ID_CANDIDATES,
        required=False,
        label="record identifier",
    )

    selected_rows: List[pd.DataFrame] = []

    for case in case_targets:
        target_id = case["target_chembl_id"]
        endpoint = case["required_endpoint"]

        subset = core_df[
            (core_df[target_col].astype(str) == target_id)
            & (core_df[endpoint_col].astype(str) == endpoint)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No records found for case {case['case_id']} "
                f"({target_id}, {endpoint})."
            )

        subset["case_id"] = case["case_id"]
        subset["target_label"] = case["target_label"]
        subset["case_priority"] = case.get("priority", "primary")
        subset["required_endpoint"] = endpoint

        rename_map = {
            compound_col: "compound_id",
            target_col: "target_chembl_id",
            endpoint_col: "endpoint",
            pchembl_col: "pactivity",
            smiles_col: "canonical_smiles",
        }

        if target_name_col:
            rename_map[target_name_col] = "target_name"
        if target_organism_col:
            rename_map[target_organism_col] = "target_organism"
        if inchikey_col:
            rename_map[inchikey_col] = "inchi_key"
        if doi_col:
            rename_map[doi_col] = "doi"
        if year_col:
            rename_map[year_col] = "year"
        if record_id_col:
            rename_map[record_id_col] = "record_id"

        subset = subset.rename(columns=rename_map)
        subset = _coalesce_duplicate_columns(subset)

        keep_cols = [
            "case_id",
            "target_label",
            "case_priority",
            "target_chembl_id",
            "target_name",
            "target_organism",
            "compound_id",
            "canonical_smiles",
            "inchi_key",
            "endpoint",
            "required_endpoint",
            "pactivity",
            "doi",
            "year",
            "record_id",
        ]

        for col in keep_cols:
            if col not in subset.columns:
                subset[col] = pd.NA

        selected_rows.append(subset[keep_cols])

    out = pd.concat(selected_rows, ignore_index=True)

    out["pactivity"] = pd.to_numeric(out["pactivity"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").map(_safe_int)

    return out


def collapse_case_study_compounds(activity_records: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate records to one row per compound-target-endpoint.

    Duplicate activity measurements for the same compound, target and endpoint
    are collapsed using the median pActivity value.
    """

    required = [
        "case_id",
        "target_label",
        "target_chembl_id",
        "compound_id",
        "canonical_smiles",
        "endpoint",
        "pactivity",
    ]
    missing = [col for col in required if col not in activity_records.columns]
    if missing:
        raise KeyError(
            "activity_records is missing required columns: " + ", ".join(missing)
        )

    rows: List[Dict[str, object]] = []

    group_cols = ["case_id", "target_chembl_id", "endpoint", "compound_id"]

    for keys, group in activity_records.groupby(group_cols, dropna=False):
        case_id, target_id, endpoint, compound_id = keys

        pvalues = pd.to_numeric(group["pactivity"], errors="coerce").dropna()

        rows.append(
            {
                "case_id": case_id,
                "target_label": _mode_or_missing(group["target_label"]),
                "case_priority": _mode_or_missing(group["case_priority"]),
                "target_chembl_id": target_id,
                "target_name": _mode_or_missing(group["target_name"]),
                "target_organism": _mode_or_missing(group["target_organism"]),
                "endpoint": endpoint,
                "compound_id": compound_id,
                "canonical_smiles": _mode_or_missing(group["canonical_smiles"]),
                "inchi_key": _mode_or_missing(group["inchi_key"]),
                "median_pactivity": _numeric_median(group["pactivity"]),
                "mean_pactivity": _numeric_mean(group["pactivity"]),
                "std_pactivity": _numeric_std(group["pactivity"]),
                "min_pactivity": _numeric_min(group["pactivity"]),
                "max_pactivity": _numeric_max(group["pactivity"]),
                "pactivity_range_within_duplicate_records": _numeric_range(group["pactivity"]),
                "record_count_for_compound": int(len(group)),
                "doi_count_for_compound": int(
                    group["doi"].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique()
                ),
                "doi_list": _sorted_unique_join(group["doi"]),
                "year_min": _safe_int(_numeric_min(group["year"])),
                "year_max": _safe_int(_numeric_max(group["year"])),
                "record_id_list": _sorted_unique_join(group["record_id"]),
            }
        )

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["case_id", "median_pactivity", "compound_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    out["smiles_missing"] = out["canonical_smiles"].isna() | (
        out["canonical_smiles"].astype(str).str.strip() == ""
    )

    out["duplicate_activity_flag"] = (out["record_count_for_compound"] > 1).astype(int)

    return out


def build_case_study_target_summary(
    activity_records: pd.DataFrame,
    compound_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise case-study target subsets before MMP/SALI analysis."""

    rows: List[Dict[str, object]] = []

    for case_id, record_group in activity_records.groupby("case_id", dropna=False):
        compound_group = compound_table[compound_table["case_id"] == case_id].copy()

        pvalues = pd.to_numeric(compound_group["median_pactivity"], errors="coerce")

        doi_values = (
            record_group["doi"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
        )

        duplicate_compounds = int(
            (compound_group["record_count_for_compound"] > 1).sum()
        )

        rows.append(
            {
                "case_id": case_id,
                "target_label": _mode_or_missing(record_group["target_label"]),
                "target_chembl_id": _mode_or_missing(record_group["target_chembl_id"]),
                "target_name": _mode_or_missing(record_group["target_name"]),
                "target_organism": _mode_or_missing(record_group["target_organism"]),
                "endpoint": _mode_or_missing(record_group["endpoint"]),
                "raw_record_count": int(len(record_group)),
                "unique_compounds_raw": int(record_group["compound_id"].nunique()),
                "compound_level_rows_after_median_collapse": int(len(compound_group)),
                "compounds_with_duplicate_activity_records": duplicate_compounds,
                "duplicate_compound_percent": (
                    float(duplicate_compounds / len(compound_group) * 100.0)
                    if len(compound_group)
                    else np.nan
                ),
                "missing_smiles_compounds": int(compound_group["smiles_missing"].sum()),
                "doi_count": int(doi_values.nunique()),
                "year_min": _safe_int(_numeric_min(record_group["year"])),
                "year_max": _safe_int(_numeric_max(record_group["year"])),
                "median_pactivity_min": float(pvalues.min()) if not pvalues.dropna().empty else np.nan,
                "median_pactivity_max": float(pvalues.max()) if not pvalues.dropna().empty else np.nan,
                "median_pactivity_range": float(pvalues.max() - pvalues.min()) if not pvalues.dropna().empty else np.nan,
            }
        )

    out = pd.DataFrame(rows)

    preferred_order = [case["case_id"] for case in CASE_STUDY_TARGETS]
    order_map = {case_id: idx for idx, case_id in enumerate(preferred_order)}
    out["case_order"] = out["case_id"].map(order_map)

    return (
        out.sort_values("case_order")
        .drop(columns=["case_order"])
        .reset_index(drop=True)
    )


def build_duplicate_resolution_table(compound_table: pd.DataFrame) -> pd.DataFrame:
    """Return compounds with multiple source activity records."""

    duplicate = compound_table[
        compound_table["record_count_for_compound"] > 1
    ].copy()

    return duplicate.sort_values(
        ["case_id", "record_count_for_compound", "pactivity_range_within_duplicate_records"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def validate_case_study_inputs(
    target_summary: pd.DataFrame,
    activity_records: pd.DataFrame,
    compound_table: pd.DataFrame,
) -> None:
    """Validate P6a case-study input tables."""

    expected_cases = {case["case_id"] for case in CASE_STUDY_TARGETS}
    observed_cases = set(target_summary["case_id"].astype(str))

    if expected_cases != observed_cases:
        raise ValueError(
            "Case-study target mismatch. Expected "
            + "; ".join(sorted(expected_cases))
            + ", observed "
            + "; ".join(sorted(observed_cases))
        )

    if activity_records["pactivity"].isna().any():
        missing = int(activity_records["pactivity"].isna().sum())
        raise ValueError(f"Selected activity records contain {missing} missing pActivity values.")

    if compound_table["median_pactivity"].isna().any():
        missing = int(compound_table["median_pactivity"].isna().sum())
        raise ValueError(f"Compound table contains {missing} missing median_pactivity values.")

    if compound_table["smiles_missing"].any():
        missing = int(compound_table["smiles_missing"].sum())
        raise ValueError(f"Compound table contains {missing} compounds with missing SMILES.")

    for case in CASE_STUDY_TARGETS:
        case_id = case["case_id"]
        target_id = case["target_chembl_id"]
        endpoint = case["required_endpoint"]

        subset = activity_records[activity_records["case_id"] == case_id]

        if subset.empty:
            raise ValueError(f"No activity records for case {case_id}.")

        if set(subset["target_chembl_id"].astype(str)) != {target_id}:
            raise ValueError(f"Unexpected target ID in case {case_id}.")

        if set(subset["endpoint"].astype(str)) != {endpoint}:
            raise ValueError(f"Unexpected endpoint in case {case_id}.")

        compound_subset = compound_table[compound_table["case_id"] == case_id]
        if compound_subset.empty:
            raise ValueError(f"No compound-level rows for case {case_id}.")

# ---------------------------------------------------------------------------
# P6b — fingerprint-based SALI/activity-cliff analysis
# ---------------------------------------------------------------------------

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _mol_from_smiles(smiles: object):
    """Parse SMILES defensively."""

    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def _ecfp4_fingerprint(mol, *, radius: int = 2, n_bits: int = 2048):
    """Generate ECFP4/Morgan fingerprint."""

    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def build_activity_cliff_pairs(
    compound_table: pd.DataFrame,
    *,
    tanimoto_threshold: float = 0.70,
    delta_pactivity_threshold: float = 2.00,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build near-neighbour and activity-cliff pairs for P6 case-study targets.

    The analysis is intentionally fingerprint-based rather than MMPA-based for
    the main benchmark-utility demonstration. This keeps the P6 layer compact,
    reproducible and robust across the three selected Ready-tier target subsets.

    Parameters
    ----------
    compound_table:
        One row per compound-target-endpoint after median pActivity collapse.
    tanimoto_threshold:
        Minimum ECFP4 Tanimoto similarity for near-neighbour pair retention.
    delta_pactivity_threshold:
        Minimum absolute pActivity difference for cliff classification.
    radius:
        Morgan fingerprint radius.
    n_bits:
        Morgan fingerprint bit length.

    Returns
    -------
    pair_table, fingerprint_failures
    """

    required = [
        "case_id",
        "target_label",
        "target_chembl_id",
        "target_name",
        "target_organism",
        "endpoint",
        "compound_id",
        "canonical_smiles",
        "median_pactivity",
    ]
    missing = [col for col in required if col not in compound_table.columns]
    if missing:
        raise KeyError(
            "compound_table is missing required columns: " + ", ".join(missing)
        )

    all_pairs: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for case_id, case_df in compound_table.groupby("case_id", dropna=False):
        working = case_df.copy().reset_index(drop=True)

        mols = []
        fps = []

        for _, row in working.iterrows():
            mol = _mol_from_smiles(row["canonical_smiles"])
            fp = _ecfp4_fingerprint(mol, radius=radius, n_bits=n_bits)

            mols.append(mol)
            fps.append(fp)

            if mol is None or fp is None:
                failures.append(
                    {
                        "case_id": case_id,
                        "compound_id": row["compound_id"],
                        "canonical_smiles": row["canonical_smiles"],
                        "failure_reason": "RDKit MolFromSmiles or fingerprint failed",
                    }
                )

        valid_indices = [idx for idx, fp in enumerate(fps) if fp is not None]

        for pos, i in enumerate(valid_indices):
            fp_i = fps[i]
            comparison_indices = valid_indices[pos + 1 :]
            if not comparison_indices:
                continue

            comparison_fps = [fps[j] for j in comparison_indices]
            similarities = DataStructs.BulkTanimotoSimilarity(fp_i, comparison_fps)

            row_i = working.iloc[i]
            p_i = float(row_i["median_pactivity"])

            for j, sim in zip(comparison_indices, similarities):
                if sim < tanimoto_threshold:
                    continue

                row_j = working.iloc[j]
                p_j = float(row_j["median_pactivity"])

                delta = abs(p_i - p_j)
                distance = 1.0 - float(sim)

                duplicate_structure_flag = int(distance <= 1e-9)

                sali = np.nan
                if distance > 1e-9:
                    sali = float(delta / distance)

                cliff_flag = int(
                    (delta >= delta_pactivity_threshold)
                    and (sim >= tanimoto_threshold)
                    and (duplicate_structure_flag == 0)
                )

                if p_i >= p_j:
                    more_active_compound = row_i["compound_id"]
                    less_active_compound = row_j["compound_id"]
                    more_active_pactivity = p_i
                    less_active_pactivity = p_j
                else:
                    more_active_compound = row_j["compound_id"]
                    less_active_compound = row_i["compound_id"]
                    more_active_pactivity = p_j
                    less_active_pactivity = p_i

                all_pairs.append(
                    {
                        "case_id": case_id,
                        "target_label": row_i["target_label"],
                        "target_chembl_id": row_i["target_chembl_id"],
                        "target_name": row_i["target_name"],
                        "target_organism": row_i["target_organism"],
                        "endpoint": row_i["endpoint"],
                        "compound_id_a": row_i["compound_id"],
                        "compound_id_b": row_j["compound_id"],
                        "canonical_smiles_a": row_i["canonical_smiles"],
                        "canonical_smiles_b": row_j["canonical_smiles"],
                        "median_pactivity_a": p_i,
                        "median_pactivity_b": p_j,
                        "delta_pactivity": float(delta),
                        "tanimoto_ecfp4": float(sim),
                        "fingerprint_distance": float(distance),
                        "sali": sali,
                        "activity_cliff_flag": cliff_flag,
                        "duplicate_structure_flag": duplicate_structure_flag,
                        "more_active_compound_id": more_active_compound,
                        "less_active_compound_id": less_active_compound,
                        "more_active_pactivity": float(more_active_pactivity),
                        "less_active_pactivity": float(less_active_pactivity),
                        "record_count_a": int(row_i.get("record_count_for_compound", 1)),
                        "record_count_b": int(row_j.get("record_count_for_compound", 1)),
                        "doi_count_a": int(row_i.get("doi_count_for_compound", 0)),
                        "doi_count_b": int(row_j.get("doi_count_for_compound", 0)),
                    }
                )

    pair_table = pd.DataFrame(all_pairs)
    failure_table = pd.DataFrame(failures)

    if not pair_table.empty:
        pair_table = pair_table.sort_values(
            [
                "case_id",
                "activity_cliff_flag",
                "sali",
                "delta_pactivity",
                "tanimoto_ecfp4",
            ],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)

    return pair_table, failure_table


def build_activity_cliff_summary(
    compound_table: pd.DataFrame,
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise fingerprint-based activity-cliff results by case-study target."""

    rows: list[dict[str, object]] = []

    case_order = [case["case_id"] for case in CASE_STUDY_TARGETS]
    order_map = {case_id: idx for idx, case_id in enumerate(case_order)}

    for case_id, compound_group in compound_table.groupby("case_id", dropna=False):
        pairs = pair_table[pair_table["case_id"] == case_id].copy()
        cliffs = pairs[pairs["activity_cliff_flag"] == 1].copy()

        pvalues = pd.to_numeric(
            compound_group["median_pactivity"],
            errors="coerce",
        ).dropna()

        cliff_compounds = set()
        if not cliffs.empty:
            cliff_compounds.update(cliffs["compound_id_a"].astype(str))
            cliff_compounds.update(cliffs["compound_id_b"].astype(str))

        rows.append(
            {
                "case_id": case_id,
                "case_order": order_map.get(str(case_id), 999),
                "target_label": _mode_or_missing(compound_group["target_label"]),
                "target_chembl_id": _mode_or_missing(compound_group["target_chembl_id"]),
                "target_name": _mode_or_missing(compound_group["target_name"]),
                "target_organism": _mode_or_missing(compound_group["target_organism"]),
                "endpoint": _mode_or_missing(compound_group["endpoint"]),
                "compound_count": int(len(compound_group)),
                "pactivity_min": float(pvalues.min()) if not pvalues.empty else np.nan,
                "pactivity_max": float(pvalues.max()) if not pvalues.empty else np.nan,
                "pactivity_range": float(pvalues.max() - pvalues.min()) if not pvalues.empty else np.nan,
                "near_neighbor_pair_count_tanimoto_ge_0_70": int(len(pairs)),
                "activity_cliff_pair_count": int(len(cliffs)),
                "cliff_compound_count": int(len(cliff_compounds)),
                "cliff_compound_percent": (
                    float(len(cliff_compounds) / len(compound_group) * 100.0)
                    if len(compound_group)
                    else np.nan
                ),
                "cliff_density_per_100_compounds": (
                    float(len(cliffs) / len(compound_group) * 100.0)
                    if len(compound_group)
                    else np.nan
                ),
                "max_delta_pactivity_all_pairs": (
                    float(pairs["delta_pactivity"].max()) if not pairs.empty else np.nan
                ),
                "max_delta_pactivity_cliffs": (
                    float(cliffs["delta_pactivity"].max()) if not cliffs.empty else np.nan
                ),
                "median_delta_pactivity_cliffs": (
                    float(cliffs["delta_pactivity"].median()) if not cliffs.empty else np.nan
                ),
                "max_sali_cliffs": (
                    float(cliffs["sali"].replace([np.inf, -np.inf], np.nan).max())
                    if not cliffs.empty
                    else np.nan
                ),
                "median_sali_cliffs": (
                    float(cliffs["sali"].replace([np.inf, -np.inf], np.nan).median())
                    if not cliffs.empty
                    else np.nan
                ),
                "max_tanimoto_cliffs": (
                    float(cliffs["tanimoto_ecfp4"].max()) if not cliffs.empty else np.nan
                ),
                "median_tanimoto_cliffs": (
                    float(cliffs["tanimoto_ecfp4"].median()) if not cliffs.empty else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)

    return (
        out.sort_values("case_order")
        .drop(columns=["case_order"])
        .reset_index(drop=True)
    )


def build_representative_cliff_pairs(
    pair_table: pd.DataFrame,
    *,
    top_n_per_case: int = 10,
) -> pd.DataFrame:
    """Select representative high-SALI cliff pairs for inspection."""

    if pair_table.empty:
        return pd.DataFrame()

    cliffs = pair_table[pair_table["activity_cliff_flag"] == 1].copy()

    if cliffs.empty:
        return pd.DataFrame(columns=list(pair_table.columns) + ["representative_rank"])

    selected: list[pd.DataFrame] = []

    for case_id, group in cliffs.groupby("case_id", dropna=False):
        ranked = group.sort_values(
            ["sali", "delta_pactivity", "tanimoto_ecfp4"],
            ascending=[False, False, False],
        ).head(top_n_per_case).copy()

        ranked["representative_rank"] = range(1, len(ranked) + 1)
        selected.append(ranked)

    return pd.concat(selected, ignore_index=True)


def validate_activity_cliff_outputs(
    compound_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    fingerprint_failures: pd.DataFrame,
) -> None:
    """Validate P6b activity-cliff outputs."""

    expected_cases = {case["case_id"] for case in CASE_STUDY_TARGETS}

    observed_cases = set(compound_table["case_id"].astype(str))
    if observed_cases != expected_cases:
        raise ValueError(
            "Compound table cases do not match expected cases: "
            + "; ".join(sorted(observed_cases))
        )

    summary_cases = set(summary_table["case_id"].astype(str))
    if summary_cases != expected_cases:
        raise ValueError(
            "Summary table cases do not match expected cases: "
            + "; ".join(sorted(summary_cases))
        )

    if not fingerprint_failures.empty:
        raise ValueError(
            f"Fingerprint failures detected: {len(fingerprint_failures)}"
        )

    if pair_table.empty:
        raise ValueError("No near-neighbour pairs were generated.")

    if pair_table["tanimoto_ecfp4"].lt(0.70).any():
        raise ValueError("Pair table contains pairs below the Tanimoto threshold.")

    if pair_table["delta_pactivity"].isna().any():
        raise ValueError("Pair table contains missing delta_pactivity values.")
