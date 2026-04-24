"""RDKit chemistry utilities for CoumarinBioBench.

This module contains reusable functions for P3a descriptor matrix generation.
It intentionally does not touch figure-generation modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hashlib
import math

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:  # pragma: no cover - depends on RDKit build
        rdMolStandardize = None
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "RDKit is required for descriptor generation. Install it with: "
        "conda install -c conda-forge rdkit"
    ) from exc


SMILES_CANDIDATES: Tuple[str, ...] = (
    "canonical_smiles",
    "standard_smiles",
    "smiles",
    "canonical_smiles_x",
    "molecule_structures.canonical_smiles",
)

COMPOUND_ID_CANDIDATES: Tuple[str, ...] = (
    "molecule_chembl_id",
    "compound_chembl_id",
    "chembl_id",
    "molecule_id",
    "compound_id",
)

TARGET_ID_CANDIDATES: Tuple[str, ...] = (
    "target_chembl_id",
    "target_id",
    "target_pref_name",
    "uniprot_accession",
)

ENDPOINT_CANDIDATES: Tuple[str, ...] = (
    "standard_type",
    "endpoint",
    "endpoint_type",
    "activity_type",
)

DEGREE_CANDIDATES: Tuple[str, ...] = (
    "chembl_target_degree",
    "target_degree",
    "degree",
    "n_targets",
    "target_count",
    "unique_targets",
)

BASIC_DESCRIPTOR_COLUMNS: Tuple[str, ...] = (
    "MolWt",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "AromaticRingCount",
    "HeavyAtomCount",
    "FractionCSP3",
    "LipinskiRo5Violations",
    "LipinskiRo5Compliant",
)


@dataclass(frozen=True)
class StandardizedMolecule:
    """Container for a sanitized and canonicalized molecule."""

    original_smiles: str
    canonical_smiles: str
    mol: Chem.Mol


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    """Resolve a column name using case-insensitive and punctuation-tolerant matching."""

    if df is None or df.empty:
        if required:
            raise ValueError(f"Cannot resolve {label}: dataframe is empty.")
        return None

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
        available = ", ".join(map(str, df.columns))
        expected = ", ".join(candidates)
        raise KeyError(
            f"Could not resolve {label}. Expected one of [{expected}]. "
            f"Available columns: {available}"
        )
    return None


def _stable_smiles_id(canonical_smiles: str) -> str:
    digest = hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:12]
    return f"SMI_{digest}"


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("SMILES produced no valid molecular fragments.")
    return max(
        fragments,
        key=lambda frag: (frag.GetNumHeavyAtoms(), Descriptors.MolWt(frag)),
    )


def standardize_smiles(smiles: object) -> StandardizedMolecule:
    """Sanitize, salt-strip by largest fragment, uncharge when available, and canonicalize."""

    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        raise ValueError("Missing SMILES value.")

    original = str(smiles).strip()
    if not original:
        raise ValueError("Empty SMILES string.")

    mol = Chem.MolFromSmiles(original, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {original}")

    try:
        if rdMolStandardize is not None:
            mol = rdMolStandardize.Cleanup(mol)
            chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
            mol = chooser.choose(mol)
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        else:
            mol = _largest_fragment(mol)

        Chem.SanitizeMol(mol)
    except Exception:
        mol = _largest_fragment(mol)
        Chem.SanitizeMol(mol)

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not canonical:
        raise ValueError(f"Canonicalization failed for SMILES: {original}")

    return StandardizedMolecule(
        original_smiles=original,
        canonical_smiles=canonical,
        mol=mol,
    )


def compute_basic_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute the core RDKit physicochemical descriptor panel."""

    mol_wt = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))

    ro5_violations = int(mol_wt > 500.0)
    ro5_violations += int(clogp > 5.0)
    ro5_violations += int(hbd > 5)
    ro5_violations += int(hba > 10)

    return {
        "MolWt": mol_wt,
        "cLogP": clogp,
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": hbd,
        "HBA": hba,
        "RotB": int(Lipinski.NumRotatableBonds(mol)),
        "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRingCount": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "LipinskiRo5Violations": int(ro5_violations),
        "LipinskiRo5Compliant": int(ro5_violations == 0),
    }


def compute_ecfp4_bits(
    mol: Chem.Mol,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute binary ECFP4/Morgan fingerprint bits."""

    arr = np.zeros((n_bits,), dtype=np.uint8)

    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )
        fp = generator.GetFingerprint(mol)
    except Exception:  # pragma: no cover - compatibility fallback
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
        )

    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _mode_or_missing(series: pd.Series) -> object:
    clean = series.dropna().astype(str)
    if clean.empty:
        return pd.NA
    modes = clean.mode()
    if modes.empty:
        return clean.iloc[0]
    return modes.iloc[0]


def _prepare_degree_table(
    core_df: pd.DataFrame,
    degree_df: Optional[pd.DataFrame],
    *,
    compound_col: str,
) -> pd.DataFrame:
    """Load or infer per-compound ChEMBL target degree."""

    if degree_df is not None and not degree_df.empty:
        degree_compound_col = resolve_column(
            degree_df,
            COMPOUND_ID_CANDIDATES,
            label="compound identifier in degree table",
        )
        degree_col = resolve_column(
            degree_df,
            DEGREE_CANDIDATES,
            label="target-degree column",
        )

        prepared = degree_df[[degree_compound_col, degree_col]].copy()
        prepared.columns = ["compound_id", "chembl_target_degree"]
        prepared["compound_id"] = prepared["compound_id"].astype(str)
        prepared["chembl_target_degree"] = pd.to_numeric(
            prepared["chembl_target_degree"],
            errors="coerce",
        )
        prepared = prepared.dropna(subset=["compound_id"])
        return prepared.drop_duplicates(subset=["compound_id"])

    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier in core dataset",
    )
    prepared = (
        core_df[[compound_col, target_col]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .groupby(compound_col, as_index=False)[target_col]
        .nunique()
    )
    prepared.columns = ["compound_id", "chembl_target_degree"]
    return prepared


def build_descriptor_matrix(
    core_df: pd.DataFrame,
    degree_df: Optional[pd.DataFrame] = None,
    *,
    smiles_col: Optional[str] = None,
    compound_col: Optional[str] = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-row-per-compound RDKit descriptor and ECFP4 matrix."""

    smiles_col = smiles_col or resolve_column(
        core_df,
        SMILES_CANDIDATES,
        label="SMILES column",
    )
    compound_col = compound_col or resolve_column(
        core_df,
        COMPOUND_ID_CANDIDATES,
        required=False,
        label="compound identifier",
    )
    endpoint_col = resolve_column(
        core_df,
        ENDPOINT_CANDIDATES,
        required=False,
        label="endpoint column",
    )

    if compound_col is None:
        working = core_df[[smiles_col]].copy()
        working["compound_id"] = pd.NA
    else:
        working = core_df[[compound_col, smiles_col]].copy()
        working = working.rename(columns={compound_col: "compound_id"})

    working = working.dropna(subset=[smiles_col]).copy()
    working["compound_id"] = working["compound_id"].astype("string")
    working[smiles_col] = working[smiles_col].astype(str)

    if endpoint_col is not None and compound_col is not None:
        endpoint_mode = (
            core_df[[compound_col, endpoint_col]]
            .dropna(subset=[compound_col])
            .groupby(compound_col)[endpoint_col]
            .agg(_mode_or_missing)
            .reset_index()
            .rename(
                columns={
                    compound_col: "compound_id",
                    endpoint_col: "dominant_endpoint",
                }
            )
        )
        endpoint_mode["compound_id"] = endpoint_mode["compound_id"].astype(str)
    else:
        endpoint_mode = pd.DataFrame(columns=["compound_id", "dominant_endpoint"])

    rows = []
    bit_rows = []
    failures = []
    seen_compounds = set()

    for index, record in working.iterrows():
        raw_compound_id = record.get("compound_id", pd.NA)
        raw_smiles = record[smiles_col]

        try:
            standardized = standardize_smiles(raw_smiles)
            compound_id = (
                str(raw_compound_id)
                if pd.notna(raw_compound_id) and str(raw_compound_id).strip()
                else _stable_smiles_id(standardized.canonical_smiles)
            )

            if compound_id in seen_compounds:
                continue

            seen_compounds.add(compound_id)
            descriptors = compute_basic_descriptors(standardized.mol)
            bits = compute_ecfp4_bits(
                standardized.mol,
                radius=radius,
                n_bits=n_bits,
            )

            row = {
                "compound_id": compound_id,
                "original_smiles": standardized.original_smiles,
                "canonical_smiles": standardized.canonical_smiles,
                "rdkit_version": rdBase.rdkitVersion,
            }
            row.update(descriptors)
            rows.append(row)
            bit_rows.append(bits)

        except Exception as exc:
            failures.append(
                {
                    "row_index": index,
                    "compound_id": (
                        str(raw_compound_id)
                        if pd.notna(raw_compound_id)
                        else pd.NA
                    ),
                    "smiles": raw_smiles,
                    "error": str(exc),
                }
            )

    if not rows:
        raise RuntimeError("No valid molecules remained after RDKit standardization.")

    descriptor_df = pd.DataFrame(rows)

    bit_columns = [f"ECFP4_{i:04d}" for i in range(n_bits)]
    fp_df = pd.DataFrame(np.vstack(bit_rows), columns=bit_columns, dtype=np.uint8)

    descriptor_df = pd.concat(
        [descriptor_df.reset_index(drop=True), fp_df.reset_index(drop=True)],
        axis=1,
    )

    if not endpoint_mode.empty:
        descriptor_df = descriptor_df.merge(endpoint_mode, on="compound_id", how="left")
    else:
        descriptor_df["dominant_endpoint"] = pd.NA

    if compound_col is not None:
        degree_table = _prepare_degree_table(
            core_df,
            degree_df,
            compound_col=compound_col,
        )
        descriptor_df = descriptor_df.merge(degree_table, on="compound_id", how="left")
    else:
        descriptor_df["chembl_target_degree"] = pd.NA

    descriptor_df["chembl_target_degree"] = pd.to_numeric(
        descriptor_df["chembl_target_degree"],
        errors="coerce",
    )
    descriptor_df["single_vs_multi"] = np.select(
        [
            descriptor_df["chembl_target_degree"] == 1,
            descriptor_df["chembl_target_degree"] >= 2,
        ],
        ["single-target", "multi-target"],
        default="unknown",
    )
    descriptor_df["high_degree_flag"] = (
        descriptor_df["chembl_target_degree"].fillna(0) >= 5
    ).astype(int)

    ordered_meta = [
        "compound_id",
        "original_smiles",
        "canonical_smiles",
        "dominant_endpoint",
        "chembl_target_degree",
        "single_vs_multi",
        "high_degree_flag",
        "rdkit_version",
    ]
    ordered_descriptors = list(BASIC_DESCRIPTOR_COLUMNS)
    ordered_bits = bit_columns

    descriptor_df = descriptor_df[
        ordered_meta + ordered_descriptors + ordered_bits
    ]

    return descriptor_df, pd.DataFrame(failures)


def summarise_descriptor_panel(
    descriptor_df: pd.DataFrame,
    descriptor_columns: Sequence[str] = BASIC_DESCRIPTOR_COLUMNS,
) -> pd.DataFrame:
    """Generate overall summary statistics for the physicochemical descriptors."""

    rows = []
    for descriptor in descriptor_columns:
        if descriptor not in descriptor_df.columns:
            continue

        values = pd.to_numeric(descriptor_df[descriptor], errors="coerce").dropna()
        if values.empty:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        rows.append(
            {
                "descriptor": descriptor,
                "n": int(values.shape[0]),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
                "median": float(values.median()),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction that preserves NaNs."""

    p_array = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p_array, np.nan, dtype=float)

    valid_mask = ~np.isnan(p_array)
    valid = p_array[valid_mask]
    if valid.size == 0:
        return adjusted.tolist()

    order = np.argsort(valid)
    ranked = valid[order]
    n = ranked.size

    corrected = np.empty(n, dtype=float)
    cumulative = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        cumulative = min(cumulative, ranked[idx] * n / rank)
        corrected[idx] = cumulative

    restored = np.empty(n, dtype=float)
    restored[order] = np.minimum(corrected, 1.0)
    adjusted[valid_mask] = restored
    return adjusted.tolist()


def _cliffs_delta_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    """Fallback Cliff's delta. Positive values indicate x > y."""

    if x.size == 0 or y.size == 0:
        return float("nan")

    greater = 0
    less = 0
    for value in x:
        greater += int(np.sum(value > y))
        less += int(np.sum(value < y))
    return float((greater - less) / (x.size * y.size))


def compare_single_multi_descriptors(
    descriptor_df: pd.DataFrame,
    descriptor_columns: Sequence[str] = BASIC_DESCRIPTOR_COLUMNS,
) -> pd.DataFrame:
    """Compare descriptor distributions between multi-target and single-target compounds."""

    try:
        from scipy.stats import mannwhitneyu
    except Exception:  # pragma: no cover - scipy optional
        mannwhitneyu = None

    rows = []
    for descriptor in descriptor_columns:
        if descriptor not in descriptor_df.columns:
            continue

        single = pd.to_numeric(
            descriptor_df.loc[
                descriptor_df["single_vs_multi"] == "single-target",
                descriptor,
            ],
            errors="coerce",
        ).dropna()
        multi = pd.to_numeric(
            descriptor_df.loc[
                descriptor_df["single_vs_multi"] == "multi-target",
                descriptor,
            ],
            errors="coerce",
        ).dropna()

        if single.empty or multi.empty:
            p_value = float("nan")
            cliff_delta = float("nan")
            u_stat = float("nan")
        elif mannwhitneyu is not None:
            test = mannwhitneyu(
                multi.to_numpy(),
                single.to_numpy(),
                alternative="two-sided",
                method="auto",
            )
            u_stat = float(test.statistic)
            p_value = float(test.pvalue)
            cliff_delta = float(
                (2.0 * u_stat / (len(multi) * len(single))) - 1.0
            )
        else:
            u_stat = float("nan")
            p_value = float("nan")
            cliff_delta = _cliffs_delta_from_arrays(
                multi.to_numpy(),
                single.to_numpy(),
            )

        rows.append(
            {
                "descriptor": descriptor,
                "single_n": int(single.shape[0]),
                "multi_n": int(multi.shape[0]),
                "single_median": (
                    float(single.median()) if not single.empty else float("nan")
                ),
                "multi_median": (
                    float(multi.median()) if not multi.empty else float("nan")
                ),
                "median_difference_multi_minus_single": (
                    float(multi.median() - single.median())
                    if not single.empty and not multi.empty
                    else float("nan")
                ),
                "mannwhitney_u": u_stat,
                "p_value": p_value,
                "cliffs_delta_multi_vs_single": cliff_delta,
            }
        )

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison["p_adjusted_bh"] = benjamini_hochberg(
            comparison["p_value"].to_list()
        )
        comparison["significant_bh_0_05"] = comparison["p_adjusted_bh"] < 0.05

    return comparison


def merge_summary_and_comparison(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact descriptor statistics table for supplementary output."""

    if comparison_df.empty:
        return summary_df.copy()

    keep_cols = [
        "descriptor",
        "single_n",
        "multi_n",
        "single_median",
        "multi_median",
        "median_difference_multi_minus_single",
        "cliffs_delta_multi_vs_single",
        "p_value",
        "p_adjusted_bh",
        "significant_bh_0_05",
    ]
    return summary_df.merge(
        comparison_df[keep_cols],
        on="descriptor",
        how="left",
    )
