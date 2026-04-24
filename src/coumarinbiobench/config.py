"""Configuration utilities for CoumarinBioBench-TierA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration.

    Attributes
    ----------
    root : Path
        Project root directory.
    raw_data : Path
        Raw input CSV path.
    tables_dir : Path
        Output table directory.
    logs_dir : Path
        Output log directory.
    config : dict[str, Any]
        Parsed YAML configuration.
    """

    root: Path
    raw_data: Path
    tables_dir: Path
    logs_dir: Path
    config: dict[str, Any]

    def resolve(self, key_path: str) -> Path:
        """Resolve a path stored in the config.

        Parameters
        ----------
        key_path : str
            Dot-separated path in the YAML config, e.g.
            ``outputs.tierA_core``.

        Returns
        -------
        Path
            Absolute path resolved against project root.

        Raises
        ------
        KeyError
            If the key path is not found.
        """
        value: Any = self.config
        for key in key_path.split("."):
            if key not in value:
                raise KeyError(f"Config key not found: {key_path}")
            value = value[key]
        return self.root / str(value)


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by locating config.yaml.

    Parameters
    ----------
    start : Path, optional
        Starting path. Defaults to current file location.

    Returns
    -------
    Path
        Project root directory.

    Raises
    ------
    FileNotFoundError
        If config.yaml cannot be found.
    """
    current = start or Path.cwd()
    current = current.resolve()

    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent

    raise FileNotFoundError("Could not locate config.yaml in current path or parents.")


def load_config(project_root: Path | None = None) -> ProjectConfig:
    """Load YAML configuration and resolve key paths.

    Parameters
    ----------
    project_root : Path, optional
        Project root path. If None, root is auto-detected.

    Returns
    -------
    ProjectConfig
        Resolved project configuration.
    """
    root = project_root or find_project_root()
    config_path = root / "config.yaml"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    raw_data = root / config["paths"]["raw_data"]
    tables_dir = root / config["paths"]["tables_dir"]
    logs_dir = root / config["paths"]["logs_dir"]

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return ProjectConfig(
        root=root,
        raw_data=raw_data,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        config=config,
    )
