"""Figure 4 visualization utilities for CoumarinBioBench.

Figure 4 integrates the organism-aware target-family landscape:
A. Target-family representation by record count.
B. Target-family representation by unique compound count.
C. Human versus non-human record contribution by target family.
D. Endpoint composition heatmap by target family.

This module is intentionally separate from previous figure modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


FAMILY_LABELS: Dict[str, str] = {
    "Carbonic anhydrases": "Carbonic\nanhydrases",
    "Monoamine oxidases": "Monoamine\noxidases",
    "Cholinesterases": "Cholinesterases",
    "Kinases": "Kinases",
    "Nuclear receptors": "Nuclear\nreceptors",
    "G protein-coupled receptors": "GPCRs",
    "Proteases": "Proteases",
    "Cytochrome P450s": "Cytochrome\nP450s",
    "Ion channels and transporters": "Ion channels /\ntransporters",
    "Other enzymes": "Other\nenzymes",
    "Other/unclassified": "Other /\nunclassified",
}

ENDPOINT_ORDER: Tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50", "AC50")

REQUIRED_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "target_family",
    "record_count",
    "percent_of_records",
    "unique_targets",
    "unique_compounds",
    "percent_of_unique_compounds",
    "dominant_endpoint",
    "endpoint_homogeneity",
)

REQUIRED_ENDPOINT_COLUMNS: Tuple[str, ...] = (
    "target_family",
    "endpoint",
    "record_count",
    "family_record_count",
    "percent_of_family_records",
)

REQUIRED_ORGANISM_COLUMNS: Tuple[str, ...] = (
    "target_family",
    "organism_group",
    "record_count",
)


def validate_figure4_inputs(
    family_summary: pd.DataFrame,
    endpoint_composition: pd.DataFrame,
    organism_summary: pd.DataFrame,
) -> None:
    """Validate required Figure 4 input tables."""

    missing_summary = [
        col for col in REQUIRED_SUMMARY_COLUMNS
        if col not in family_summary.columns
    ]
    if missing_summary:
        raise KeyError(
            "target_family_summary.csv is missing columns: "
            + ", ".join(missing_summary)
        )

    missing_endpoint = [
        col for col in REQUIRED_ENDPOINT_COLUMNS
        if col not in endpoint_composition.columns
    ]
    if missing_endpoint:
        raise KeyError(
            "family_endpoint_composition.csv is missing columns: "
            + ", ".join(missing_endpoint)
        )

    missing_organism = [
        col for col in REQUIRED_ORGANISM_COLUMNS
        if col not in organism_summary.columns
    ]
    if missing_organism:
        raise KeyError(
            "family_organism_summary.csv is missing columns: "
            + ", ".join(missing_organism)
        )

    if family_summary["target_family"].nunique() != len(family_summary):
        raise ValueError("Family summary must contain one row per target family.")

    if family_summary["record_count"].isna().any():
        raise ValueError("Family summary contains missing record_count values.")

    if int(family_summary["record_count"].sum()) <= 0:
        raise ValueError("Family summary has non-positive total record count.")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.25)


def _family_display_labels(families: Sequence[str]) -> List[str]:
    return [FAMILY_LABELS.get(fam, fam) for fam in families]


def _ordered_family_summary(family_summary: pd.DataFrame) -> pd.DataFrame:
    """Order target families by descending record count."""

    return (
        family_summary.copy()
        .sort_values(["record_count", "unique_compounds"], ascending=[False, False])
        .reset_index(drop=True)
    )


def plot_family_records(ax: plt.Axes, ordered_summary: pd.DataFrame) -> None:
    """Panel A: target-family representation by record count."""

    y = np.arange(len(ordered_summary))
    records = pd.to_numeric(ordered_summary["record_count"], errors="coerce")
    percents = pd.to_numeric(ordered_summary["percent_of_records"], errors="coerce")

    ax.barh(y, records, color="#4C78A8", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(_family_display_labels(ordered_summary["target_family"]))
    ax.invert_yaxis()
    ax.set_xlabel("Tier-A Core records")
    ax.set_title("Target-family representation by records", fontsize=11)

    max_x = float(records.max()) if len(records) else 1.0
    ax.set_xlim(0, max_x * 1.22)

    for ypos, value, pct in zip(y, records, percents):
        ax.text(
            value + max_x * 0.015,
            ypos,
            f"{int(value):,} ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    _clean_axis(ax)
    _panel_label(ax, "A")


def plot_family_compounds(ax: plt.Axes, ordered_summary: pd.DataFrame) -> None:
    """Panel B: target-family representation by unique compounds."""

    y = np.arange(len(ordered_summary))
    compounds = pd.to_numeric(ordered_summary["unique_compounds"], errors="coerce")
    percents = pd.to_numeric(
        ordered_summary["percent_of_unique_compounds"],
        errors="coerce",
    )

    ax.barh(y, compounds, color="#F58518", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(_family_display_labels(ordered_summary["target_family"]))
    ax.invert_yaxis()
    ax.set_xlabel("Unique compounds")
    ax.set_title("Target-family representation by compound coverage", fontsize=11)

    max_x = float(compounds.max()) if len(compounds) else 1.0
    ax.set_xlim(0, max_x * 1.24)

    for ypos, value, pct in zip(y, compounds, percents):
        ax.text(
            value + max_x * 0.015,
            ypos,
            f"{int(value):,} ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )

    _clean_axis(ax)
    _panel_label(ax, "B")


def _prepare_organism_percentages(
    family_summary: pd.DataFrame,
    organism_summary: pd.DataFrame,
    family_order: Sequence[str],
) -> pd.DataFrame:
    grouped = (
        organism_summary
        .groupby(["target_family", "organism_group"], dropna=False)["record_count"]
        .sum()
        .reset_index()
    )

    pivot = (
        grouped
        .pivot(index="target_family", columns="organism_group", values="record_count")
        .fillna(0)
        .reset_index()
    )

    for col in ("Human", "Non-human", "Unknown"):
        if col not in pivot.columns:
            pivot[col] = 0

    totals = family_summary[["target_family", "record_count"]].copy()
    totals = totals.rename(columns={"record_count": "family_record_count"})

    merged = totals.merge(pivot, on="target_family", how="left")
    for col in ("Human", "Non-human", "Unknown"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    denom = pd.to_numeric(merged["family_record_count"], errors="coerce").replace(0, np.nan)

    merged["Human_percent"] = merged["Human"] / denom * 100.0
    merged["Non_human_percent"] = merged["Non-human"] / denom * 100.0
    merged["Unknown_percent"] = merged["Unknown"] / denom * 100.0

    order_map = {family: idx for idx, family in enumerate(family_order)}
    merged["order"] = merged["target_family"].map(order_map)

    return merged.sort_values("order").reset_index(drop=True)


def plot_organism_contribution(
    ax: plt.Axes,
    family_summary: pd.DataFrame,
    organism_summary: pd.DataFrame,
    family_order: Sequence[str],
) -> None:
    """Panel C: human versus non-human record contribution by family."""

    organism_df = _prepare_organism_percentages(
        family_summary,
        organism_summary,
        family_order,
    )

    y = np.arange(len(organism_df))
    human = organism_df["Human_percent"].fillna(0).to_numpy(dtype=float)
    nonhuman = organism_df["Non_human_percent"].fillna(0).to_numpy(dtype=float)
    unknown = organism_df["Unknown_percent"].fillna(0).to_numpy(dtype=float)

    ax.barh(y, human, color="#4C78A8", alpha=0.88, label="Human")
    ax.barh(y, nonhuman, left=human, color="#72B7B2", alpha=0.88, label="Non-human")

    if np.nanmax(unknown) > 0:
        ax.barh(
            y,
            unknown,
            left=human + nonhuman,
            color="#B8B8B8",
            alpha=0.88,
            label="Unknown",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(_family_display_labels(organism_df["target_family"]))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Records within family (%)")
    ax.set_title("Human versus non-human record contribution", fontsize=11)

    for ypos, hpct, npct in zip(y, human, nonhuman):
        if hpct >= 12:
            ax.text(
                hpct / 2,
                ypos,
                f"{hpct:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=7.5,
            )
        if npct >= 12:
            ax.text(
                hpct + npct / 2,
                ypos,
                f"{npct:.0f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=7.5,
            )

    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8)
    _clean_axis(ax)
    _panel_label(ax, "C")


def _endpoint_matrix(
    endpoint_composition: pd.DataFrame,
    family_order: Sequence[str],
) -> pd.DataFrame:
    matrix = (
        endpoint_composition
        .pivot_table(
            index="target_family",
            columns="endpoint",
            values="percent_of_family_records",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=family_order)
    )

    for endpoint in ENDPOINT_ORDER:
        if endpoint not in matrix.columns:
            matrix[endpoint] = 0.0

    return matrix.loc[:, list(ENDPOINT_ORDER)].fillna(0.0)


def plot_endpoint_heatmap(
    ax: plt.Axes,
    endpoint_composition: pd.DataFrame,
    family_order: Sequence[str],
) -> None:
    """Panel D: endpoint composition heatmap by target family."""

    matrix = _endpoint_matrix(endpoint_composition, family_order)

    values = matrix.to_numpy(dtype=float)

    im = ax.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=100,
        cmap="Blues",
    )

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(_family_display_labels(matrix.index))
    ax.set_title("Endpoint composition by target family", fontsize=11)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value >= 5:
                color = "white" if value >= 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=color,
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Family records (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.tick_params(axis="x", labelrotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _panel_label(ax, "D")


def create_figure4(
    family_summary: pd.DataFrame,
    endpoint_composition: pd.DataFrame,
    organism_summary: pd.DataFrame,
    *,
    title: str = "Organism-aware target-family landscape of Tier-A Core coumarins",
) -> plt.Figure:
    """Create complete Figure 4."""

    validate_figure4_inputs(
        family_summary,
        endpoint_composition,
        organism_summary,
    )

    ordered = _ordered_family_summary(family_summary)
    family_order = ordered["target_family"].tolist()

    fig = plt.figure(figsize=(14.0, 11.2), constrained_layout=False)
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.08],
        width_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.33,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_family_records(ax_a, ordered)
    plot_family_compounds(ax_b, ordered)
    plot_organism_contribution(ax_c, family_summary, organism_summary, family_order)
    plot_endpoint_heatmap(ax_d, endpoint_composition, family_order)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

    fig.subplots_adjust(
        top=0.93,
        left=0.12,
        right=0.985,
        bottom=0.075,
    )

    return fig


def save_figure4(
    fig: plt.Figure,
    output_dir: Path,
    *,
    basename: str = "Figure_4_target_family_landscape",
    dpi: int = 450,
) -> Dict[str, Path]:
    """Save Figure 4 in PNG, PDF and SVG formats."""

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


def figure4_summary(
    family_summary: pd.DataFrame,
    organism_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact Figure 4 summary table."""

    ordered = _ordered_family_summary(family_summary)

    human_records = int(
        organism_summary.loc[
            organism_summary["organism_group"] == "Human",
            "record_count",
        ].sum()
    )
    nonhuman_records = int(
        organism_summary.loc[
            organism_summary["organism_group"] == "Non-human",
            "record_count",
        ].sum()
    )
    total_records = int(family_summary["record_count"].sum())

    rows = [
        {"metric": "total_records", "value": total_records},
        {"metric": "target_families", "value": int(family_summary["target_family"].nunique())},
        {"metric": "total_targets", "value": int(family_summary["unique_targets"].sum())},
        {"metric": "human_records", "value": human_records},
        {"metric": "nonhuman_records", "value": nonhuman_records},
        {
            "metric": "human_record_percent",
            "value": float(human_records / total_records * 100.0),
        },
        {
            "metric": "nonhuman_record_percent",
            "value": float(nonhuman_records / total_records * 100.0),
        },
        {
            "metric": "top_family_by_records",
            "value": str(ordered.iloc[0]["target_family"]),
        },
        {
            "metric": "top_family_record_count",
            "value": int(ordered.iloc[0]["record_count"]),
        },
        {
            "metric": "top_family_record_percent",
            "value": float(ordered.iloc[0]["percent_of_records"]),
        },
        {
            "metric": "top_family_by_unique_compounds",
            "value": str(
                family_summary.sort_values(
                    "unique_compounds",
                    ascending=False,
                ).iloc[0]["target_family"]
            ),
        },
        {
            "metric": "top_unique_compound_count",
            "value": int(
                family_summary.sort_values(
                    "unique_compounds",
                    ascending=False,
                ).iloc[0]["unique_compounds"]
            ),
        },
    ]

    return pd.DataFrame(rows)
