"""Validation and column-resolution utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_column_name(name: str) -> str:
    """Normalize a column name for case-insensitive matching.

    Parameters
    ----------
    name : str
        Raw column name.

    Returns
    -------
    str
        Normalized column name.
    """
    return str(name).strip().lower()


def non_empty_mask(series: pd.Series) -> pd.Series:
    """Return mask for non-empty, non-null values.

    Parameters
    ----------
    series : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Boolean mask.
    """
    return (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & series.astype(str).str.strip().str.lower().ne("nan")
        & series.astype(str).str.strip().str.lower().ne("none")
    )


def count_non_empty(df: pd.DataFrame, column: str) -> int:
    """Count non-empty values in a dataframe column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str
        Column name.

    Returns
    -------
    int
        Number of non-empty values.
    """
    if column not in df.columns:
        return 0
    return int(non_empty_mask(df[column]).sum())


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    """Resolve a required column from candidate names.

    The resolver is data-aware: if multiple candidate columns exist, it selects
    the candidate with the highest non-empty value count. This prevents empty
    placeholder columns, such as an all-NaN ``target_type`` column, from being
    selected over a populated alternative such as ``target_type_full``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    candidates : Iterable[str]
        Candidate column names.
    label : str
        Human-readable field label.

    Returns
    -------
    str
        Actual column name found in dataframe.

    Raises
    ------
    KeyError
        If no candidate column is found.
    ValueError
        If candidate columns exist but all are empty.
    """
    normalized_map = {
        normalize_column_name(column): column for column in df.columns
    }

    found_columns: list[str] = []
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_map:
            found_columns.append(normalized_map[key])

    if not found_columns:
        available = ", ".join(map(str, df.columns))
        checked = ", ".join(candidates)
        raise KeyError(
            f"Required column for '{label}' not found. "
            f"Checked candidates: [{checked}]. "
            f"Available columns: [{available}]"
        )

    coverage = {column: count_non_empty(df, column) for column in found_columns}
    best_column = max(coverage, key=coverage.get)

    if coverage[best_column] == 0:
        available = ", ".join(map(str, df.columns))
        checked = ", ".join(candidates)
        coverage_text = ", ".join(
            f"{column}={count}" for column, count in coverage.items()
        )
        raise ValueError(
            f"Candidate columns for '{label}' exist but are all empty. "
            f"Checked candidates: [{checked}]. "
            f"Coverage: [{coverage_text}]. "
            f"Available columns: [{available}]"
        )

    return best_column


def resolve_optional_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    label: str,
) -> str | None:
    """Resolve an optional column from candidate names.

    If candidate columns exist but are all empty, the function returns None
    rather than raising an exception.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    candidates : Iterable[str]
        Candidate column names.
    label : str
        Human-readable field label.

    Returns
    -------
    str or None
        Actual column name if found and non-empty, otherwise None.
    """
    try:
        return resolve_column(df, candidates, label)
    except (KeyError, ValueError):
        return None


def count_unique_non_null(df: pd.DataFrame, column: str | None) -> int:
    """Count unique non-null and non-empty values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    column : str or None
        Column name.

    Returns
    -------
    int
        Unique non-empty count. Returns 0 if column is None.
    """
    if column is None or column not in df.columns:
        return 0

    cleaned = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    cleaned = cleaned[
        ~cleaned.str.lower().isin(["nan", "none", "null"])
    ]

    return int(cleaned.nunique())
