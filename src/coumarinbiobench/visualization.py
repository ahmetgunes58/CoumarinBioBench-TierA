"""Publication-grade visualization utilities for CoumarinBioBench-TierA.

This module contains reusable figure-generation functions. Executable files in
``scripts/`` remain thin runners and delegate plotting logic to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from coumarinbiobench.config import ProjectConfig, load_config
from coumarinbiobench.io_utils import read_csv_safely, write_dataframe
from coumarinbiobench.logging_utils import setup_logger
from coumarinbiobench.validation import non_empty_mask


COLOR_NAVY = "#183A59"
COLOR_BLUE = "#2F80ED"
COLOR_TEAL = "#2A9D8F"
COLOR_GREEN = "#238B45"
COLOR_AMBER = "#E9A23B"
COLOR_RED = "#C94C4C"
COLOR_GRAY = "#6B7280"
COLOR_LIGHT_BLUE = "#F2F7FF"
COLOR_LIGHT_TEAL = "#EEF8F6"
COLOR_LIGHT_GRAY = "#F7F8FA"
COLOR_BORDER = "#111827"
COLOR_TEXT = "#111827"


@dataclass
class Figure1Paths:
    """Input and output paths for Figure 1 generation."""

    audit_table: Path
    dataset_summary: Path
    validity_warning_records: Path
    figure_png: Path
    figure_pdf: Path
    figure_svg: Path
    source_data: Path


def _set_publication_style() -> None:
    """Apply publication-grade Matplotlib style defaults."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.4,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _format_int(value: Any) -> str:
    """Format a value as a comma-separated integer."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: Any, digits: int = 2) -> str:
    """Format a number as percentage text."""
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def _resolve_figure1_paths(project_config: ProjectConfig) -> Figure1Paths:
    """Resolve input and output paths for Figure 1."""
    root = project_config.root
    config = project_config.config

    tables_dir = root / config["paths"]["tables_dir"]
    figures_dir = root / config["paths"]["figures_dir"]
    source_dir = figures_dir / "source_data"

    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    return Figure1Paths(
        audit_table=tables_dir / "Supplementary_Table_S0_curation_audit.csv",
        dataset_summary=tables_dir / "Table_1_dataset_summary.csv",
        validity_warning_records=(
            root / "data" / "interim" / "validity_warning_records.csv"
        ),
        figure_png=figures_dir / "Figure_1_curation_workflow.png",
        figure_pdf=figures_dir / "Figure_1_curation_workflow.pdf",
        figure_svg=figures_dir / "Figure_1_curation_workflow.svg",
        source_data=source_dir / "Figure_1_source_data.csv",
    )


def _load_figure1_inputs(
    paths: Figure1Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Figure 1 input tables."""
    audit_df = read_csv_safely(paths.audit_table)
    summary_df = read_csv_safely(paths.dataset_summary)
    validity_df = read_csv_safely(paths.validity_warning_records)
    return audit_df, summary_df, validity_df


def _prepare_stepwise_data(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ordered curation data from Supplementary Table S0."""
    required_columns = {
        "step",
        "records_after",
        "records_removed",
        "percent_removed",
        "unique_molecules",
        "unique_targets",
    }
    missing = required_columns.difference(audit_df.columns)
    if missing:
        raise ValueError(
            "Audit table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    label_map = {
        "P0_raw_dataset": "Raw dataset",
        "P0_01_single_protein_targets": "Single-protein targets",
        "P0_02_exact_relation_equals": "Exact relation '='",
        "P0_03_accepted_endpoints": "Accepted endpoints",
        "P0_04_nM_standardized_units": "nM-standardized values",
        "P0_05_positive_numeric_values": "Tier-A Screened",
        "P0_06_remove_data_validity_warnings": "Tier-A Core",
    }

    rows = []
    for step, label in label_map.items():
        matched = audit_df.loc[audit_df["step"].eq(step)]
        if matched.empty:
            raise ValueError(f"Required audit step not found: {step}")

        row = matched.iloc[0].to_dict()
        rows.append(
            {
                "step": step,
                "label": label,
                "records": int(row["records_after"]),
                "records_removed": int(row["records_removed"]),
                "percent_removed": float(row["percent_removed"]),
                "unique_molecules": int(row["unique_molecules"]),
                "unique_targets": int(row["unique_targets"]),
            }
        )

    stepwise_df = pd.DataFrame(rows)
    raw_records = int(stepwise_df.loc[0, "records"])
    stepwise_df["retained_percent_of_raw"] = (
        stepwise_df["records"] / raw_records * 100
    )
    return stepwise_df


def _prepare_layer_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset layer summary for Figure 1 panel C."""
    required_columns = {
        "dataset_layer",
        "records",
        "unique_molecules",
        "unique_targets",
        "doi_coverage_percent",
    }
    missing = required_columns.difference(summary_df.columns)
    if missing:
        raise ValueError(
            "Dataset summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    wanted = ["Raw", "Tier-A Screened", "Tier-A Core"]
    summary = summary_df.loc[summary_df["dataset_layer"].isin(wanted)].copy()
    summary["dataset_layer"] = pd.Categorical(
        summary["dataset_layer"],
        categories=wanted,
        ordered=True,
    )
    summary = summary.sort_values("dataset_layer").reset_index(drop=True)

    if len(summary) != 3:
        raise ValueError(
            "Dataset summary must contain Raw, Tier-A Screened, and Tier-A Core rows."
        )

    return summary


def _prepare_validity_summary(validity_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare validity-warning summary for Figure 1 panel D."""
    column = "data_validity_comment"
    if column not in validity_df.columns:
        raise ValueError(
            "validity_warning_records.csv must contain data_validity_comment column."
        )

    filtered = validity_df.loc[non_empty_mask(validity_df[column])].copy()
    summary = (
        filtered[column]
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("validity_comment")
        .reset_index(name="record_count")
    )

    order = [
        "Outside typical range",
        "Potential transcription error",
        "Manually validated",
    ]
    summary["validity_comment"] = pd.Categorical(
        summary["validity_comment"],
        categories=order,
        ordered=True,
    )
    summary = summary.sort_values("validity_comment").reset_index(drop=True)
    summary["validity_comment"] = summary["validity_comment"].astype(str)

    return summary


def _write_figure1_source_data(
    stepwise_df: pd.DataFrame,
    layer_summary_df: pd.DataFrame,
    validity_summary_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write consolidated source data for Figure 1."""
    panel_a = stepwise_df.copy()
    panel_a.insert(0, "panel", "A_workflow")

    panel_b = stepwise_df.copy()
    panel_b.insert(0, "panel", "B_attrition_curve")

    panel_c = layer_summary_df.copy()
    panel_c.insert(0, "panel", "C_benchmark_tiers")

    panel_d = validity_summary_df.copy()
    panel_d.insert(0, "panel", "D_validity_warnings")

    source_data = pd.concat(
        [panel_a, panel_b, panel_c, panel_d],
        ignore_index=True,
        sort=False,
    )
    write_dataframe(source_data, path)


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Draw a consistent panel title."""
    ax.set_title(
        f"{label}  {title}",
        loc="left",
        fontweight="bold",
        fontsize=11.2,
        pad=9,
        color=COLOR_TEXT,
    )


def _workflow_label(label: str, records: int) -> str:
    """Create a compact workflow label."""
    replacements = {
        "Raw dataset": "Raw\ndataset",
        "Single-protein targets": "Single-protein\ntargets",
        "Exact relation '='": "Exact\nrelation '='",
        "Accepted endpoints": "Accepted\nendpoints",
        "nM-standardized values": "nM-standardized\nvalues",
        "Tier-A Screened": "Tier-A\nScreened",
        "Tier-A Core": "Tier-A\nCore",
    }
    return f"{replacements.get(label, label)}\n{_format_int(records)}"


def _draw_panel_a(ax: plt.Axes, stepwise_df: pd.DataFrame) -> None:
    """Draw Figure 1A: clean workflow."""
    ax.axis("off")
    _panel_title(ax, "A", "Curation workflow from raw records to Tier-A Core")

    n_steps = len(stepwise_df)
    y_center = 0.56
    box_width = 0.116
    box_height = 0.30
    x_positions = [
        0.055 + index * (0.89 / (n_steps - 1))
        for index in range(n_steps)
    ]

    for index, row in stepwise_df.iterrows():
        x_center = x_positions[index]
        label = str(row["label"])

        if label == "Tier-A Core":
            facecolor = COLOR_LIGHT_TEAL
            edgecolor = COLOR_TEAL
            linewidth = 1.25
        elif label == "Tier-A Screened":
            facecolor = COLOR_LIGHT_BLUE
            edgecolor = COLOR_BLUE
            linewidth = 1.15
        else:
            facecolor = "white"
            edgecolor = COLOR_BORDER
            linewidth = 0.95

        box = FancyBboxPatch(
            (x_center - box_width / 2, y_center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(box)

        ax.text(
            x_center,
            y_center,
            _workflow_label(label, int(row["records"])),
            ha="center",
            va="center",
            fontsize=8.0,
            linespacing=1.12,
            color=COLOR_TEXT,
            transform=ax.transAxes,
        )

        if index < n_steps - 1:
            next_x = x_positions[index + 1]
            arrow = FancyArrowPatch(
                (x_center + box_width / 2 + 0.010, y_center),
                (next_x - box_width / 2 - 0.010, y_center),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color=COLOR_GRAY,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.11,
        "Tier-A Screened = positive nM records before data-validity exclusion; "
        "Tier-A Core = analysis-grade benchmark used for all downstream analyses.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )


def _draw_panel_b(ax: plt.Axes, stepwise_df: pd.DataFrame) -> None:
    """Draw Figure 1B: attrition curve."""
    plot_df = stepwise_df.copy()
    x_values = list(range(len(plot_df)))

    ax.plot(
        x_values,
        plot_df["records"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        color=COLOR_NAVY,
        zorder=3,
    )

    ax.fill_between(
        x_values,
        plot_df["records"],
        [plot_df["records"].min() * 0.55] * len(plot_df),
        color=COLOR_NAVY,
        alpha=0.06,
        zorder=1,
    )

    ax.set_yscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [
            "Raw",
            "Single\nprotein",
            "Exact\nrelation",
            "Accepted\nendpoints",
            "nM",
            "Screened",
            "Core",
        ]
    )

    ax.set_ylabel("Records retained (log scale)")
    _panel_title(ax, "B", "Record attrition across filters")

    ymax = plot_df["records"].max() * 2.0
    ymin = max(plot_df["records"].min() * 0.50, 1)
    ax.set_ylim(ymin, ymax)

    label_offsets = {
        0: (0, 8),
        1: (0, 8),
        2: (0, 8),
        3: (0, 8),
        4: (0, 22),
        5: (0, -18),
        6: (0, 8),
    }

    for index, row in plot_df.iterrows():
        offset = label_offsets.get(index, (0, 8))
        va = "bottom" if offset[1] >= 0 else "top"
        ax.annotate(
            _format_int(row["records"]),
            xy=(index, row["records"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7.8,
            color=COLOR_TEXT,
            clip_on=False,
        )

    core_percent = float(plot_df.iloc[-1]["retained_percent_of_raw"])
    ax.text(
        0.98,
        0.88,
        f"Tier-A Core retains {core_percent:.2f}%\nof raw records",
        ha="right",
        va="top",
        fontsize=8.2,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.65)


def _draw_tier_row(
    ax: plt.Axes,
    y_bottom: float,
    title: str,
    records: int,
    molecules: int,
    targets: int,
    doi_coverage: float,
    edgecolor: str,
    facecolor: str,
) -> None:
    """Draw one spacious tier-summary row."""
    x_left = 0.04
    width = 0.92
    height = 0.23

    card = FancyBboxPatch(
        (x_left, y_bottom),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(card)

    ax.text(
        x_left + 0.035,
        y_bottom + height * 0.67,
        title,
        ha="left",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=COLOR_TEXT,
        transform=ax.transAxes,
    )

    ax.text(
        x_left + 0.035,
        y_bottom + height * 0.33,
        f"{_format_int(records)} records",
        ha="left",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=edgecolor,
        transform=ax.transAxes,
    )

    metrics = [
        ("Molecules", _format_int(molecules)),
        ("Targets", _format_int(targets)),
        ("DOI", _format_percent(doi_coverage)),
    ]
    metric_x = [x_left + 0.46, x_left + 0.66, x_left + 0.83]

    for x_pos, (label, value) in zip(metric_x, metrics):
        ax.text(
            x_pos,
            y_bottom + height * 0.63,
            label,
            ha="center",
            va="center",
            fontsize=7.4,
            color=COLOR_GRAY,
            transform=ax.transAxes,
        )
        ax.text(
            x_pos,
            y_bottom + height * 0.36,
            value,
            ha="center",
            va="center",
            fontsize=8.5,
            color=COLOR_TEXT,
            transform=ax.transAxes,
        )


def _draw_panel_c(ax: plt.Axes, layer_summary_df: pd.DataFrame) -> None:
    """Draw Figure 1C: benchmark tier summary."""
    ax.axis("off")
    _panel_title(ax, "C", "Benchmark tiers and provenance")

    style_map = {
        "Raw": (COLOR_GRAY, COLOR_LIGHT_GRAY),
        "Tier-A Screened": (COLOR_BLUE, COLOR_LIGHT_BLUE),
        "Tier-A Core": (COLOR_TEAL, COLOR_LIGHT_TEAL),
    }

    y_positions = [0.64, 0.37, 0.10]

    for y_bottom, (_, row) in zip(y_positions, layer_summary_df.iterrows()):
        edgecolor, facecolor = style_map[str(row["dataset_layer"])]
        _draw_tier_row(
            ax=ax,
            y_bottom=y_bottom,
            title=str(row["dataset_layer"]),
            records=int(row["records"]),
            molecules=int(row["unique_molecules"]),
            targets=int(row["unique_targets"]),
            doi_coverage=float(row["doi_coverage_percent"]),
            edgecolor=edgecolor,
            facecolor=facecolor,
        )


def _draw_panel_d(ax: plt.Axes, validity_summary_df: pd.DataFrame) -> None:
    """Draw Figure 1D: data-validity warnings removed."""
    plot_df = validity_summary_df.copy()
    y_positions = list(range(len(plot_df)))

    color_map = {
        "Outside typical range": COLOR_RED,
        "Potential transcription error": COLOR_AMBER,
        "Manually validated": COLOR_BLUE,
    }

    max_count = int(plot_df["record_count"].max())
    ax.set_xscale("log")
    ax.set_xlim(0.75, max_count * 2.2)

    for y_position, (_, row) in zip(y_positions, plot_df.iterrows()):
        label = str(row["validity_comment"])
        count = int(row["record_count"])
        color = color_map.get(label, COLOR_NAVY)

        ax.hlines(
            y=y_position,
            xmin=1,
            xmax=count,
            linewidth=1.7,
            color=color,
            alpha=0.90,
        )
        ax.plot(count, y_position, marker="o", markersize=6.8, color=color)
        ax.text(
            count * 1.17,
            y_position,
            _format_int(count),
            va="center",
            fontsize=8.6,
            color=COLOR_TEXT,
            clip_on=False,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["validity_comment"], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel("Removed records (log scale)")
    _panel_title(ax, "D", "Data-validity warnings removed from the Screened set")

    total_removed = int(plot_df["record_count"].sum())
    ax.text(
        0.985,
        0.10,
        f"Total removed: {_format_int(total_removed)} records",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLOR_GRAY,
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.65)


def generate_figure1_curation_workflow(
    project_config: ProjectConfig | None = None,
) -> None:
    """Generate Figure 1 curation workflow."""
    _set_publication_style()

    project_config = project_config or load_config()
    logger = setup_logger(
        name="coumarinbiobench.figure1",
        log_dir=project_config.logs_dir,
        prefix="10_generate_figure1_curation_workflow",
    )

    logger.info("Starting Figure 1 generation.")
    paths = _resolve_figure1_paths(project_config)

    audit_df, summary_df, validity_df = _load_figure1_inputs(paths)
    logger.info("Audit table loaded: %s rows", len(audit_df))
    logger.info("Dataset summary loaded: %s rows", len(summary_df))
    logger.info("Validity-warning records loaded: %s rows", len(validity_df))

    stepwise_df = _prepare_stepwise_data(audit_df)
    layer_summary_df = _prepare_layer_summary(summary_df)
    validity_summary_df = _prepare_validity_summary(validity_df)

    _write_figure1_source_data(
        stepwise_df=stepwise_df,
        layer_summary_df=layer_summary_df,
        validity_summary_df=validity_summary_df,
        path=paths.source_data,
    )
    logger.info("Figure 1 source data written: %s", paths.source_data)

    fig = plt.figure(figsize=(14.8, 9.4))
    grid_spec = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.02, 1.25, 0.95],
        width_ratios=[1.03, 0.97],
        hspace=0.45,
        wspace=0.30,
    )

    ax_a = fig.add_subplot(grid_spec[0, :])
    ax_b = fig.add_subplot(grid_spec[1, 0])
    ax_c = fig.add_subplot(grid_spec[1, 1])
    ax_d = fig.add_subplot(grid_spec[2, :])

    _draw_panel_a(ax_a, stepwise_df)
    _draw_panel_b(ax_b, stepwise_df)
    _draw_panel_c(ax_c, layer_summary_df)
    _draw_panel_d(ax_d, validity_summary_df)

    fig.suptitle(
        "Tier-A curation workflow for the CoumarinBioBench benchmark",
        fontsize=14.5,
        fontweight="bold",
        y=0.986,
        color=COLOR_TEXT,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.955])

    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=600, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    fig.savefig(paths.figure_svg, bbox_inches="tight")
    plt.close(fig)

    logger.info("Figure 1 PNG written: %s", paths.figure_png)
    logger.info("Figure 1 PDF written: %s", paths.figure_pdf)
    logger.info("Figure 1 SVG written: %s", paths.figure_svg)
    logger.info("Figure 1 generation completed successfully.")

