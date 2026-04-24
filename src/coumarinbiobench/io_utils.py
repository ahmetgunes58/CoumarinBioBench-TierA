"""Input/output utilities for CoumarinBioBench-TierA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file with robust defaults.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the CSV is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    return df


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write dataframe as CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to write.
    path : Path
        Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_text(content: str, path: Path) -> None:
    """Write text content to file.

    Parameters
    ----------
    content : str
        Text content.
    path : Path
        Output text path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def compute_sha256(path: Path, chunk_size: int = 1_048_576) -> str:
    """Compute SHA-256 checksum for a file.

    Parameters
    ----------
    path : Path
        File path.
    chunk_size : int
        Number of bytes read at once.

    Returns
    -------
    str
        SHA-256 hex digest.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot checksum missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
