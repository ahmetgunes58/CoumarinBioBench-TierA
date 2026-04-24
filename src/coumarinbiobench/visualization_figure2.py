"""Figure 2 visualization for CoumarinBioBench-TierA.

This module is intentionally separated from visualization.py to protect the
locked Figure 1 code. It generates the compound-target network / target
landscape figure from P2 network tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from coumarinbiobench.config import ProjectConfig, load_config
from coumarinbiobench.io_utils import read_csv_safely, write_dataframe
from coumarinbiobench.logging_utils import setup_logger


COLOR_NAVY = "#183A59"
COLOR_BLUE = "#2F80ED"
COLOR_TEAL = "#2A9D8F"
COLOR_AMBER = "#E9A23B"
COLOR_RED = "#C94C4C"
COLOR_GRAY = "#6B7280"
COLOR_TEXT = "#111827"


@dataclass
class Figure2Paths:
    """Input and output paths for Figure 2 generation."""

    target_richness_table: Path
    degree_distribution_table: Path
    high_degree_table: Path
    endpoint_composition_table: Path
    network_overview_table: Path
    figure_png: Path
    figure_pdf: Path
    figure_svg: Path
    source_data: Path


def _set_style() -> None:
    """Apply publication-oriented plotting defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _format_int(value: Any) -> str:
    """Format integer-like values with commas."""
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any, digits: int = 1) -> str:
    """Format a numeric value as percentage."""
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _resolve_paths(project_config: ProjectConfig) -> Figure2Paths:
    """Resolve input and output paths."""
    root = project_config.root
    config = project_config.config

    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    source_dir = figures_dir / "source_data"

    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    return Figure2Paths(
        target_richness_table=processed_dir / "target_richness_table.csv",
        degree_distribution_table=processed_dir / "degree_distribution_table.csv",
        high_degree_table=(
            tables_dir / "Supplementary_Table_S5_high_degree_pains_provenance.csv"
        ),
        endpoint_composition_table=processed_dir / "endpoint_composition_table.csv",
        network_overview_table=tables_dir / "Table_P2_network_overview.csv",
        figure_png=figures_dir / "Figure_2_target_landscape.png",
        figure_pdf=figures_dir / "Figure_2_target_landscape.pdf",
        figure_svg=figures_dir / "Figure_2_target_landscape.svg",
        source_data=source_dir / "Figure_2_source_data.csv",
    )


def _get_metric(overview_df: pd.DataFrame, metric: str, default: Any = "") -> Any:
    """Get one metric from Table_P2_network_overview."""
    matched = overview_df.loc[overview_df["metric"].eq(metric), "value"]
    if matched.empty:
        return default
    return matched.iloc[0]


def _short_target_name(name: Any) -> str:
    """Create compact target display names."""
    text = str(name)

    replacements = {
        "Amine oxidase [flavin-containing] B": "MAO-B",
        "Amine oxidase [flavin-containing] A": "MAO-A",
        "Acetylcholinesterase": "AChE",
        "Carbonic anhydrase 9": "CA IX",
        "Carbonic anhydrase 12": "CA XII",
        "Carbonic anhydrase 1": "CA I",
        "Carbonic anhydrase 2": "CA II",
    }

    if text in replacements:
        return replacements[text]

    return text.replace("Carbonic anhydrase", "CA")


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Draw panel title."""
    ax.set_title(
        f"{label}  {title}",
        loc="left",
        fontweight="bold",
        fontsize=11.2,
        pad=9,
        color=COLOR_TEXT,
    )


def _prepare_top_targets(target_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Prepare top targets by unique compound count."""
    required = {
        "target_chembl_id",
        "target_name",
        "unique_compounds",
        "record_count",
        "dominant_endpoint",
        "endpoint_homogeneity",
    }
    missing = required.difference(target_df.columns)
    if missing:
        raise ValueError(
            "target_richness_table.csv missing required columns: "
            + ", ".join(sorted(missing))
        )

    top_targets = (
        target_df.sort_values(
            ["unique_compounds", "record_count", "target_chembl_id"],
            ascending=[False, False, True],
        )
        .head(n)
        .copy()
    )

    top_targets["display_label"] = top_targets.apply(
        lambda row: (
            f"{_short_target_name(row['target_name'])}\n"
            f"{row['target_chembl_id']}"
        ),
        axis=1,
    )

    return top_targets


def _prepare_degree_distribution(degree_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ordered target-degree distribution."""
    required = {"degree_bin", "compound_count", "percent_of_compounds"}
    missing = required.difference(degree_df.columns)
    if missing:
        raise ValueError(
            "degree_distribution_table.csv missing required columns: "
            + ", ".join(sorted(missing))
        )

    order = ["1", "2", "3-4", "5-9", ">=10"]
    degree_df = degree_df.copy()
    degree_df["degree_bin"] = pd.Categorical(
        degree_df["degree_bin"],
        categories=order,
        ordered=True,
    )
    return degree_df.sort_values("degree_bin").reset_index(drop=True)


def _prepare_high_degree(high_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Prepare highest-degree coumarins."""
    required = {
        "molecule_chembl_id",
        "chembl_target_degree",
        "record_count",
        "pains_alerts",
    }
    missing = required.difference(high_df.columns)
    if missing:
        raise ValueError(
            "Supplementary_Table_S5_high_degree_pains_provenance.csv "
            "missing required columns: "
            + ", ".join(sorted(missing))
        )

    high_degree = (
        high_df.sort_values(
            ["chembl_target_degree", "record_count", "molecule_chembl_id"],
            ascending=[False, False, True],
        )
        .head(n)
        .copy()
    )

    no_alert_values = {
        "none",
        "",
        "nan",
        "not_evaluated_rdkit_unavailable",
        "no_smiles",
        "invalid_smiles",
    }

    high_degree["pains_flag"] = high_degree["pains_alerts"].apply(
        lambda value: "PAINS alert"
        if str(value).strip().lower() not in no_alert_values
        else "No PAINS alert"
    )

    return high_degree


def _prepare_endpoint_composition(endpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare endpoint composition."""
    required = {"endpoint", "record_count", "percent_of_core_records"}
    missing = required.difference(endpoint_df.columns)
    if missing:
        raise ValueError(
            "endpoint_composition_table.csv missing required columns: "
            + ", ".join(sorted(missing))
        )

    return endpoint_df.sort_values("record_count", ascending=False).reset_index(drop=True)


def _write_source_data(
    top_targets: pd.DataFrame,
    degree_dist: pd.DataFrame,
    high_degree: pd.DataFrame,
    endpoint_comp: pd.DataFrame,
    overview_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write consolidated Figure 2 source data."""
    frames = []

    for panel, frame in [
        ("A_top_targets", top_targets),
        ("B_degree_distribution", degree_dist),
        ("C_high_degree_coumarins", high_degree),
        ("D_endpoint_composition", endpoint_comp),
        ("network_overview", overview_df),
    ]:
        temp = frame.copy()
        temp.insert(0, "panel", panel)
        frames.append(temp)

    source_data = pd.concat(frames, ignore_index=True, sort=False)
    write_dataframe(source_data, path)


def _draw_panel_a(ax: plt.Axes, top_targets: pd.DataFrame) -> None:
    """Draw top protein targets by compound coverage."""
    plot_df = top_targets.sort_values("unique_compounds", ascending=True)
    y_positions = range(len(plot_df))

    ax.barh(
        y_positions,
        plot_df["unique_compounds"],
        height=0.68,
        color=COLOR_NAVY,
        alpha=0.92,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["display_label"], fontsize=7.9)
    ax.set_xlabel("Unique coumarins")
    _panel_title(ax, "A", "Top protein targets by compound coverage")

    max_value = int(plot_df["unique_compounds"].max())
    ax.set_xlim(0, max_value * 1.35)

    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows()):
        endpoint = str(row["dominant_endpoint"])
        homogeneity = float(row["endpoint_homogeneity"]) * 100
        text = (
            f"{_format_int(row['unique_compounds'])} "
            f"({endpoint}, {homogeneity:.1f}%)"
        )
        ax.text(
            row["unique_compounds"] + max_value * 0.025,
            y_pos,
            text,
            va="center",
            fontsize=7.7,
            color=COLOR_TEXT,
            clip_on=False,
        )

    ax.text(
        0.99,
        0.035,
        "Parentheses: dominant endpoint and endpoint homogeneity.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)


def _draw_panel_b(
    ax: plt.Axes,
    degree_dist: pd.DataFrame,
    overview_df: pd.DataFrame,
) -> None:
    """Draw compound target-degree distribution."""
    x_positions = list(range(len(degree_dist)))
    colors = [COLOR_TEAL, COLOR_BLUE, COLOR_NAVY, COLOR_AMBER, COLOR_RED]

    ax.bar(
        x_positions,
        degree_dist["compound_count"],
        color=colors[: len(degree_dist)],
        alpha=0.92,
        width=0.64,
    )

    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(degree_dist["degree_bin"])
    ax.set_xlabel("Distinct targets per compound")
    ax.set_ylabel("Compound count (log scale)")
    _panel_title(ax, "B", "Compound target-degree distribution")

    ymax = float(degree_dist["compound_count"].max()) * 4.0
    ax.set_ylim(1, ymax)

    for x_pos, (_, row) in zip(x_positions, degree_dist.iterrows()):
        label = (
            f"{_format_int(row['compound_count'])}\n"
            f"{_format_percent(row['percent_of_compounds'], 1)}"
        )
        ax.text(
            x_pos,
            float(row["compound_count"]) * 1.30,
            label,
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=COLOR_TEXT,
            clip_on=False,
        )

    single_pct = float(_get_metric(overview_df, "single_target_percent", 0))
    multi_pct = float(_get_metric(overview_df, "multi_target_percent", 0))
    ax.text(
        0.98,
        0.93,
        f"Single-target: {single_pct:.1f}%\nMulti-target: {multi_pct:.1f}%",
        ha="right",
        va="top",
        fontsize=8.0,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)


def _draw_panel_c(ax: plt.Axes, high_degree: pd.DataFrame) -> None:
    """Draw highest-degree coumarins."""
    plot_df = high_degree.sort_values("chembl_target_degree", ascending=True)
    y_positions = range(len(plot_df))

    colors = [
        COLOR_RED if row["pains_flag"] == "PAINS alert" else COLOR_BLUE
        for _, row in plot_df.iterrows()
    ]

    ax.barh(
        y_positions,
        plot_df["chembl_target_degree"],
        height=0.68,
        color=colors,
        alpha=0.90,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["molecule_chembl_id"], fontsize=8.0)
    ax.set_xlabel("ChEMBL-target degree")
    _panel_title(ax, "C", "Highest-degree coumarins")

    max_degree = int(plot_df["chembl_target_degree"].max())
    ax.set_xlim(0, max_degree * 1.45)

    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows()):
        text = (
            f"{int(row['chembl_target_degree'])} targets; "
            f"{_format_int(row['record_count'])} records"
        )
        ax.text(
            row["chembl_target_degree"] + max_degree * 0.030,
            y_pos,
            text,
            va="center",
            fontsize=7.7,
            color=COLOR_TEXT,
            clip_on=False,
        )

    ax.text(
        0.99,
        0.035,
        "No PAINS alert = blue; PAINS alert = red.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)


def _draw_panel_d(ax: plt.Axes, endpoint_comp: pd.DataFrame) -> None:
    """Draw endpoint composition."""
    plot_df = endpoint_comp.sort_values("record_count", ascending=True).copy()
    y_positions = range(len(plot_df))

    color_map = {
        "IC50": COLOR_NAVY,
        "Ki": COLOR_TEAL,
        "EC50": COLOR_BLUE,
        "AC50": COLOR_AMBER,
        "Kd": COLOR_RED,
    }
    colors = [color_map.get(str(endpoint), COLOR_GRAY) for endpoint in plot_df["endpoint"]]

    ax.barh(
        y_positions,
        plot_df["record_count"],
        height=0.68,
        color=colors,
        alpha=0.92,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["endpoint"])
    ax.set_xlabel("Tier-A Core records")
    _panel_title(ax, "D", "Endpoint composition of Tier-A Core")

    max_records = int(plot_df["record_count"].max())
    ax.set_xlim(0, max_records * 1.35)

    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows()):
        label = (
            f"{_format_int(row['record_count'])} "
            f"({_format_percent(row['percent_of_core_records'], 1)})"
        )
        ax.text(
            row["record_count"] + max_records * 0.025,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=7.8,
            color=COLOR_TEXT,
            clip_on=False,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)


def generate_figure2_target_landscape(
    project_config: ProjectConfig | None = None,
) -> None:
    """Generate Figure 2 target landscape."""
    _set_style()

    project_config = project_config or load_config()
    logger = setup_logger(
        name="coumarinbiobench.figure2",
        log_dir=project_config.logs_dir,
        prefix="11_generate_figure2_target_landscape",
    )

    logger.info("Starting Figure 2 generation.")
    paths = _resolve_paths(project_config)

    target_df = read_csv_safely(paths.target_richness_table)
    degree_df = read_csv_safely(paths.degree_distribution_table)
    high_df = read_csv_safely(paths.high_degree_table)
    endpoint_df = read_csv_safely(paths.endpoint_composition_table)
    overview_df = read_csv_safely(paths.network_overview_table)

    logger.info("target_richness_table loaded: %s rows", len(target_df))
    logger.info("degree_distribution_table loaded: %s rows", len(degree_df))
    logger.info("high_degree_table loaded: %s rows", len(high_df))
    logger.info("endpoint_composition_table loaded: %s rows", len(endpoint_df))

    top_targets = _prepare_top_targets(target_df, n=10)
    degree_dist = _prepare_degree_distribution(degree_df)
    high_degree = _prepare_high_degree(high_df, n=10)
    endpoint_comp = _prepare_endpoint_composition(endpoint_df)

    _write_source_data(
        top_targets=top_targets,
        degree_dist=degree_dist,
        high_degree=high_degree,
        endpoint_comp=endpoint_comp,
        overview_df=overview_df,
        path=paths.source_data,
    )
    logger.info("Figure 2 source data written: %s", paths.source_data)

    fig = plt.figure(figsize=(15.2, 10.2))
    grid_spec = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.08, 0.92],
        hspace=0.40,
        wspace=0.36,
    )

    ax_a = fig.add_subplot(grid_spec[0, 0])
    ax_b = fig.add_subplot(grid_spec[0, 1])
    ax_c = fig.add_subplot(grid_spec[1, 0])
    ax_d = fig.add_subplot(grid_spec[1, 1])

    _draw_panel_a(ax_a, top_targets)
    _draw_panel_b(ax_b, degree_dist, overview_df)
    _draw_panel_c(ax_c, high_degree)
    _draw_panel_d(ax_d, endpoint_comp)

    core_records = _get_metric(overview_df, "core_records", "11579")
    unique_compounds = _get_metric(overview_df, "unique_compounds", "5390")
    unique_targets = _get_metric(overview_df, "unique_targets", "632")
    unique_edges = _get_metric(overview_df, "unique_compound_target_edges", "9926")

    fig.suptitle(
        "Compound-target network of the Tier-A Core benchmark",
        fontsize=14.5,
        fontweight="bold",
        y=0.986,
        color=COLOR_TEXT,
    )

    fig.text(
        0.5,
        0.952,
        (
            f"{_format_int(core_records)} records; "
            f"{_format_int(unique_compounds)} coumarins; "
            f"{_format_int(unique_targets)} protein targets; "
            f"{_format_int(unique_edges)} unique compound-target edges"
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLOR_GRAY,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.925])

    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=600, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    fig.savefig(paths.figure_svg, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 2 PNG written: %s", paths.figure_png)
    logger.info("Figure 2 PDF written: %s", paths.figure_pdf)
    logger.info("Figure 2 SVG written: %s", paths.figure_svg)
    logger.info("Figure 2 generation completed successfully.")
