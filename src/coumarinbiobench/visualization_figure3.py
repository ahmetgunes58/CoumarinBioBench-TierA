"""Figure 3 visualization utilities for CoumarinBioBench.

Figure 3 integrates:
A. UMAP chemical space coloured by single-target vs multi-target class.
B. UMAP chemical space highlighting high-degree compounds.
C. Robust-scaled descriptor distributions for selected physicochemical descriptors.
D. Cliff's delta effect-size panel for single-target vs multi-target comparison.

This module is intentionally separate from Figure 1 and Figure 2 visualization code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


FIGURE3_REQUIRED_SOURCE_COLUMNS: Tuple[str, ...] = (
    "compound_id",
    "UMAP1",
    "UMAP2",
    "single_vs_multi",
    "high_degree_flag",
    "chembl_target_degree",
)

FIGURE3_SELECTED_DESCRIPTORS: Tuple[str, ...] = (
    "MolWt",
    "cLogP",
    "TPSA",
    "RotB",
)

EFFECT_SIZE_DESCRIPTOR_ORDER: Tuple[str, ...] = (
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

DESCRIPTOR_LABELS: Dict[str, str] = {
    "MolWt": "MW",
    "cLogP": "cLogP",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RotB": "RotB",
    "RingCount": "Ring count",
    "AromaticRingCount": "Aromatic rings",
    "HeavyAtomCount": "Heavy atoms",
    "FractionCSP3": "Fraction Csp3",
    "LipinskiRo5Violations": "Ro5 violations",
    "LipinskiRo5Compliant": "Ro5 compliant",
}

CLASS_LABELS: Dict[str, str] = {
    "single-target": "Single-target",
    "multi-target": "Multi-target",
}

CLASS_COLORS: Dict[str, str] = {
    "single-target": "#4C78A8",
    "multi-target": "#F58518",
}

NEUTRAL_GREY = "#B8B8B8"
HIGH_DEGREE_COLOR = "#C44E52"
POSITIVE_COLOR = "#4C78A8"
NEGATIVE_COLOR = "#C44E52"


def validate_figure3_inputs(
    source_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    """Validate Figure 3 input tables."""

    missing_source = [
        col for col in FIGURE3_REQUIRED_SOURCE_COLUMNS
        if col not in source_df.columns
    ]
    if missing_source:
        raise KeyError(
            "Figure 3 source data is missing required columns: "
            + ", ".join(missing_source)
        )

    required_comparison = [
        "descriptor",
        "cliffs_delta_multi_vs_single",
        "p_adjusted_bh",
    ]
    missing_comparison = [
        col for col in required_comparison
        if col not in comparison_df.columns
    ]
    if missing_comparison:
        raise KeyError(
            "Descriptor comparison table is missing required columns: "
            + ", ".join(missing_comparison)
        )

    if source_df["compound_id"].nunique() != len(source_df):
        raise ValueError("Figure 3 source data contains duplicate compound IDs.")

    if source_df[["UMAP1", "UMAP2"]].isna().any().any():
        raise ValueError("Figure 3 source data contains missing UMAP coordinates.")

    coords = source_df[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Figure 3 source data contains non-finite UMAP coordinates.")

    missing_descriptors = [
        col for col in FIGURE3_SELECTED_DESCRIPTORS
        if col not in source_df.columns
    ]
    if missing_descriptors:
        raise KeyError(
            "Figure 3 source data is missing selected descriptor columns: "
            + ", ".join(missing_descriptors)
        )


def _format_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _clean_umap_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_umap_by_degree_class(ax: plt.Axes, source_df: pd.DataFrame) -> None:
    """Panel A: UMAP coloured by single-target vs multi-target class."""

    for cls in ("single-target", "multi-target"):
        subset = source_df[source_df["single_vs_multi"] == cls]
        ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            s=9,
            alpha=0.72,
            linewidths=0,
            c=CLASS_COLORS.get(cls, NEUTRAL_GREY),
            label=f"{CLASS_LABELS.get(cls, cls)} (n={len(subset):,})",
            rasterized=True,
        )

    _clean_umap_axis(ax)
    ax.set_title("Chemical space by target-degree class", fontsize=11)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        markerscale=1.6,
    )
    _format_panel_label(ax, "A")


def plot_umap_high_degree(ax: plt.Axes, source_df: pd.DataFrame) -> None:
    """Panel B: UMAP highlighting compounds with degree >= 5."""

    degree = pd.to_numeric(
        source_df["chembl_target_degree"],
        errors="coerce",
    ).fillna(0)

    high = source_df[pd.to_numeric(source_df["high_degree_flag"], errors="coerce").fillna(0) == 1]
    low = source_df[pd.to_numeric(source_df["high_degree_flag"], errors="coerce").fillna(0) != 1]

    ax.scatter(
        low["UMAP1"],
        low["UMAP2"],
        s=7,
        alpha=0.30,
        linewidths=0,
        c=NEUTRAL_GREY,
        label=f"Degree < 5 (n={len(low):,})",
        rasterized=True,
    )

    high_degree_values = pd.to_numeric(
        high["chembl_target_degree"],
        errors="coerce",
    ).fillna(5)

    high_sizes = 14 + 5 * np.sqrt(high_degree_values.to_numpy(dtype=float))

    ax.scatter(
        high["UMAP1"],
        high["UMAP2"],
        s=high_sizes,
        alpha=0.86,
        linewidths=0.25,
        edgecolors="white",
        c=HIGH_DEGREE_COLOR,
        label=f"Degree >= 5 (n={len(high):,})",
        rasterized=True,
    )

    _clean_umap_axis(ax)
    ax.set_title("High-degree coumarins within chemical space", fontsize=11)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        markerscale=1.2,
    )
    _format_panel_label(ax, "B")


def _robust_scale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        std = numeric.std(ddof=1)
        if not np.isfinite(std) or std == 0:
            return numeric * 0.0
        return (numeric - median) / std
    return (numeric - median) / iqr


def plot_descriptor_distributions(ax: plt.Axes, source_df: pd.DataFrame) -> None:
    """Panel C: robust-scaled descriptor distributions."""

    positions = []
    box_data = []
    box_colors = []
    ytick_positions = []
    ytick_labels = []

    base_positions = np.arange(len(FIGURE3_SELECTED_DESCRIPTORS)) * 3.0

    for idx, descriptor in enumerate(FIGURE3_SELECTED_DESCRIPTORS):
        scaled = _robust_scale(source_df[descriptor])
        working = source_df[["single_vs_multi"]].copy()
        working["scaled_value"] = scaled

        single = working.loc[
            working["single_vs_multi"] == "single-target",
            "scaled_value",
        ].dropna().to_numpy(dtype=float)
        multi = working.loc[
            working["single_vs_multi"] == "multi-target",
            "scaled_value",
        ].dropna().to_numpy(dtype=float)

        pos_single = base_positions[idx] + 0.45
        pos_multi = base_positions[idx] - 0.45

        positions.extend([pos_single, pos_multi])
        box_data.extend([single, multi])
        box_colors.extend([
            CLASS_COLORS["single-target"],
            CLASS_COLORS["multi-target"],
        ])

        ytick_positions.append(base_positions[idx])
        ytick_labels.append(DESCRIPTOR_LABELS.get(descriptor, descriptor))

    box = ax.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.60,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )

    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.65)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Robust-scaled descriptor value")
    ax.set_title("Descriptor distributions by target-degree class", fontsize=11)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=CLASS_COLORS["single-target"],
            markeredgecolor="none",
            markersize=8,
            label="Single-target",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=CLASS_COLORS["multi-target"],
            markeredgecolor="none",
            markersize=8,
            label="Multi-target",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=8,
    )
    _format_panel_label(ax, "C")


def _prepare_effect_size_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    working = comparison_df.copy()
    working = working[
        working["descriptor"].isin(EFFECT_SIZE_DESCRIPTOR_ORDER)
    ].copy()

    order_map = {
        descriptor: idx
        for idx, descriptor in enumerate(EFFECT_SIZE_DESCRIPTOR_ORDER)
    }
    working["order"] = working["descriptor"].map(order_map)
    working = working.sort_values("order")

    working["descriptor_label"] = working["descriptor"].map(
        lambda value: DESCRIPTOR_LABELS.get(value, value)
    )
    working["delta"] = pd.to_numeric(
        working["cliffs_delta_multi_vs_single"],
        errors="coerce",
    )
    working["p_adjusted_bh"] = pd.to_numeric(
        working["p_adjusted_bh"],
        errors="coerce",
    )
    working["significant"] = working["p_adjusted_bh"] < 0.05

    return working.dropna(subset=["delta"])


def plot_effect_sizes(ax: plt.Axes, comparison_df: pd.DataFrame) -> None:
    """Panel D: Cliff's delta effect-size panel."""

    working = _prepare_effect_size_table(comparison_df)

    y = np.arange(len(working))
    colors = [
        POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
        for value in working["delta"]
    ]

    ax.barh(
        y,
        working["delta"],
        color=colors,
        alpha=0.82,
        edgecolor="none",
    )

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(working["descriptor_label"])
    ax.invert_yaxis()

    max_abs = float(np.nanmax(np.abs(working["delta"]))) if len(working) else 0.1
    xlim = max(0.16, max_abs * 1.35)
    ax.set_xlim(-xlim, xlim)

    for ypos, (_, row) in zip(y, working.iterrows()):
        delta = float(row["delta"])
        label_x = delta + (0.012 if delta >= 0 else -0.012)
        ha = "left" if delta >= 0 else "right"
        text = f"{delta:.2f}"
        if bool(row["significant"]):
            text += "*"
        ax.text(
            label_x,
            ypos,
            text,
            va="center",
            ha=ha,
            fontsize=8,
        )

    ax.set_xlabel("Cliff's delta: multi-target vs single-target")
    ax.set_title("Effect sizes for descriptor shifts", fontsize=11)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.99,
        0.02,
        "* BH-adjusted p < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    _format_panel_label(ax, "D")


def create_figure3(
    source_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    title: str = "Chemical-space organization of Tier-A Core coumarins",
) -> plt.Figure:
    """Create the complete Figure 3 matplotlib figure."""

    validate_figure3_inputs(source_df, comparison_df)

    fig = plt.figure(figsize=(13.5, 10.5), constrained_layout=False)
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.30,
        wspace=0.25,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_umap_by_degree_class(ax_a, source_df)
    plot_umap_high_degree(ax_b, source_df)
    plot_descriptor_distributions(ax_c, source_df)
    plot_effect_sizes(ax_d, comparison_df)

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(
        top=0.93,
        left=0.08,
        right=0.985,
        bottom=0.075,
    )

    return fig


def save_figure3(
    fig: plt.Figure,
    output_dir: Path,
    *,
    basename: str = "Figure_3_chemical_space",
    dpi: int = 450,
) -> Dict[str, Path]:
    """Save Figure 3 as PNG, PDF and SVG."""

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "png": output_dir / f"{basename}.png",
        "pdf": output_dir / f"{basename}.pdf",
        "svg": output_dir / f"{basename}.svg",
    }

    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")

    return paths


def figure3_summary(source_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact Figure 3 summary table for reporting."""

    effect_df = _prepare_effect_size_table(comparison_df)
    significant_count = int(effect_df["significant"].sum()) if len(effect_df) else 0

    rows = [
        {
            "metric": "figure3_source_rows",
            "value": int(len(source_df)),
        },
        {
            "metric": "unique_compounds",
            "value": int(source_df["compound_id"].nunique()),
        },
        {
            "metric": "single_target_compounds",
            "value": int((source_df["single_vs_multi"] == "single-target").sum()),
        },
        {
            "metric": "multi_target_compounds",
            "value": int((source_df["single_vs_multi"] == "multi-target").sum()),
        },
        {
            "metric": "high_degree_compounds_degree_ge_5",
            "value": int(pd.to_numeric(source_df["high_degree_flag"], errors="coerce").fillna(0).sum()),
        },
        {
            "metric": "effect_size_descriptors_plotted",
            "value": int(len(effect_df)),
        },
        {
            "metric": "bh_significant_descriptor_shifts",
            "value": significant_count,
        },
        {
            "metric": "selected_distribution_descriptors",
            "value": "; ".join(FIGURE3_SELECTED_DESCRIPTORS),
        },
    ]

    return pd.DataFrame(rows)
