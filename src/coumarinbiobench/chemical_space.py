"""Chemical-space embedding utilities for CoumarinBioBench.

P3b uses the ECFP4 bit matrix generated in P3a to compute a reproducible
UMAP embedding. This module does not create figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


FINGERPRINT_PREFIX = "ECFP4_"

DEFAULT_UMAP_PARAMS: Dict[str, object] = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "jaccard",
    "random_state": 42,
}


CORE_METADATA_COLUMNS: Tuple[str, ...] = (
    "compound_id",
    "canonical_smiles",
    "dominant_endpoint",
    "chembl_target_degree",
    "single_vs_multi",
    "high_degree_flag",
)

FIGURE3_DESCRIPTOR_COLUMNS: Tuple[str, ...] = (
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
class UMAPResult:
    """Container for UMAP embedding and parameter metadata."""

    embedding: pd.DataFrame
    parameters: Dict[str, object]
    fingerprint_columns: List[str]


def _fingerprint_sort_key(column: str) -> int:
    suffix = column.replace(FINGERPRINT_PREFIX, "")
    try:
        return int(suffix)
    except ValueError:
        return 10**9


def resolve_fingerprint_columns(
    descriptor_df: pd.DataFrame,
    *,
    expected_bits: int = 2048,
) -> List[str]:
    """Resolve and validate ECFP4 bit columns."""

    fp_cols = [
        col for col in descriptor_df.columns
        if str(col).startswith(FINGERPRINT_PREFIX)
    ]
    fp_cols = sorted(fp_cols, key=_fingerprint_sort_key)

    if not fp_cols:
        raise KeyError(
            "No ECFP4 fingerprint columns found. Expected columns like "
            "ECFP4_0000 ... ECFP4_2047 in descriptor_matrix.csv."
        )

    if len(fp_cols) != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} ECFP4 bit columns, found {len(fp_cols)}."
        )

    expected_names = [f"{FINGERPRINT_PREFIX}{i:04d}" for i in range(expected_bits)]
    missing = sorted(set(expected_names) - set(fp_cols))
    if missing:
        raise ValueError(
            "Fingerprint columns are incomplete. First missing columns: "
            + ", ".join(missing[:10])
        )

    return fp_cols


def prepare_binary_fingerprint_matrix(
    descriptor_df: pd.DataFrame,
    fingerprint_columns: Sequence[str],
) -> np.ndarray:
    """Prepare a validated binary fingerprint matrix for UMAP."""

    matrix = descriptor_df.loc[:, list(fingerprint_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if matrix.isna().any().any():
        n_missing = int(matrix.isna().sum().sum())
        raise ValueError(f"Fingerprint matrix contains {n_missing} missing values.")

    values = matrix.to_numpy(dtype=np.uint8, copy=True)

    unique_values = np.unique(values)
    invalid = [int(v) for v in unique_values if v not in (0, 1)]
    if invalid:
        raise ValueError(
            "Fingerprint matrix must be binary. Invalid values found: "
            + ", ".join(map(str, invalid[:10]))
        )

    if values.shape[0] == 0:
        raise ValueError("Fingerprint matrix has zero rows.")

    if values.shape[1] == 0:
        raise ValueError("Fingerprint matrix has zero columns.")

    return values.astype(bool)


def compute_umap_embedding(
    descriptor_df: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "jaccard",
    random_state: int = 42,
    expected_bits: int = 2048,
) -> UMAPResult:
    """Compute a two-dimensional UMAP embedding from ECFP4 bits."""

    try:
        import umap
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "The 'umap-learn' package is required for P3b. Install it with:\n"
            "conda install -c conda-forge umap-learn\n"
            "or\n"
            "pip install umap-learn"
        ) from exc

    if "compound_id" not in descriptor_df.columns:
        raise KeyError("descriptor_matrix.csv must contain a compound_id column.")

    fp_cols = resolve_fingerprint_columns(
        descriptor_df,
        expected_bits=expected_bits,
    )
    x = prepare_binary_fingerprint_matrix(descriptor_df, fp_cols)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=random_state,
    )

    coordinates = reducer.fit_transform(x)

    embedding = descriptor_df[["compound_id"]].copy()
    embedding["UMAP1"] = coordinates[:, 0]
    embedding["UMAP2"] = coordinates[:, 1]

    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": 2,
        "metric": metric,
        "random_state": random_state,
        "n_compounds": int(x.shape[0]),
        "n_fingerprint_bits": int(x.shape[1]),
    }

    return UMAPResult(
        embedding=embedding,
        parameters=params,
        fingerprint_columns=list(fp_cols),
    )


def build_figure3_source_data(
    descriptor_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge UMAP coordinates with metadata and selected descriptors for Figure 3."""

    if "compound_id" not in descriptor_df.columns:
        raise KeyError("descriptor_df must contain compound_id.")

    if "compound_id" not in embedding_df.columns:
        raise KeyError("embedding_df must contain compound_id.")

    metadata_cols = [
        col for col in CORE_METADATA_COLUMNS
        if col in descriptor_df.columns
    ]
    descriptor_cols = [
        col for col in FIGURE3_DESCRIPTOR_COLUMNS
        if col in descriptor_df.columns
    ]

    source = embedding_df.merge(
        descriptor_df[metadata_cols + descriptor_cols],
        on="compound_id",
        how="left",
        validate="one_to_one",
    )

    if source[["UMAP1", "UMAP2"]].isna().any().any():
        raise ValueError("UMAP source data contains missing coordinates.")

    if source["compound_id"].nunique() != len(source):
        raise ValueError("Figure 3 source data must contain unique compound_id rows.")

    return source


def summarize_embedding(
    embedding_df: pd.DataFrame,
    source_df: pd.DataFrame,
    parameters: Dict[str, object],
) -> pd.DataFrame:
    """Create a compact summary table for the UMAP embedding."""

    rows = [
        {
            "metric": "n_compounds",
            "value": int(len(embedding_df)),
        },
        {
            "metric": "unique_compounds",
            "value": int(embedding_df["compound_id"].nunique()),
        },
        {
            "metric": "umap1_min",
            "value": float(embedding_df["UMAP1"].min()),
        },
        {
            "metric": "umap1_max",
            "value": float(embedding_df["UMAP1"].max()),
        },
        {
            "metric": "umap2_min",
            "value": float(embedding_df["UMAP2"].min()),
        },
        {
            "metric": "umap2_max",
            "value": float(embedding_df["UMAP2"].max()),
        },
    ]

    if "single_vs_multi" in source_df.columns:
        for label, count in source_df["single_vs_multi"].value_counts().items():
            rows.append(
                {
                    "metric": f"degree_class_{label}",
                    "value": int(count),
                }
            )

    if "high_degree_flag" in source_df.columns:
        rows.append(
            {
                "metric": "high_degree_compounds_degree_ge_5",
                "value": int(source_df["high_degree_flag"].sum()),
            }
        )

    for key, value in parameters.items():
        rows.append(
            {
                "metric": f"parameter_{key}",
                "value": value,
            }
        )

    return pd.DataFrame(rows)


def validate_umap_outputs(
    descriptor_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> None:
    """Validate consistency between descriptor matrix, embedding and source data."""

    n_descriptor = len(descriptor_df)
    n_embedding = len(embedding_df)
    n_source = len(source_df)

    if n_descriptor != n_embedding:
        raise ValueError(
            f"Descriptor rows ({n_descriptor}) and embedding rows ({n_embedding}) differ."
        )

    if n_descriptor != n_source:
        raise ValueError(
            f"Descriptor rows ({n_descriptor}) and Figure 3 source rows ({n_source}) differ."
        )

    if embedding_df["compound_id"].nunique() != n_embedding:
        raise ValueError("UMAP embedding contains duplicate compound IDs.")

    if source_df["compound_id"].nunique() != n_source:
        raise ValueError("Figure 3 source data contains duplicate compound IDs.")

    if embedding_df[["UMAP1", "UMAP2"]].isna().any().any():
        raise ValueError("UMAP embedding contains missing coordinates.")

    if not np.isfinite(embedding_df[["UMAP1", "UMAP2"]].to_numpy()).all():
        raise ValueError("UMAP embedding contains non-finite coordinates.")
