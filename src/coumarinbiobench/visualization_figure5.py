"""Figure 5 visualization utilities for CoumarinBioBench.

Figure 5 summarizes target-specific QSAR-readiness:
A. Readiness tier distribution across all protein targets.
B. Ready-tier targets ranked by unique compound count.
C. Readiness criteria space: unique compounds vs dominant-endpoint pActivity range.
D. Readiness tier composition by target family.
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


READINESS_TIER_ORDER: Tuple[str, ...] = (
    "Ready",
    "Curatable",
    "Exploratory",
    "Not Suitable",
)

TIER_COLORS: Dict[str, str] = {
    "Ready": "#2CA02C",
    "Curatable": "#1F77B4",
    "Exploratory": "#FF7F0E",
    "Not Suitable": "#B8B8B8",
}

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


REQUIRED_READINESS_COLUMNS: Tuple[str, ...] = (
    "target_chembl_id",
    "target_name",
    "target_organism",
    "target_family",
    "organism_group",
    "readiness_tier",
    "unique_compounds",
    "record_count",
    "dominant_endpoint",
    "endpoint_homogeneity",
    "dominant_endpoint_pactivity_range",
    "doi_count",
)

REQUIRED_TIER_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "readiness_tier",
    "target_count",
    "percent_of_targets",
)

REQUIRED_FAMILY_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "target_family",
    "readiness_tier",
    "target_count",
    "family_target_count",
    "percent_of_family_targets",
)


def validate_figure5_inputs(
    readiness_df: pd.DataFrame,
    tier_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    ready_targets: pd.DataFrame,
) -> None:
    """Validate Figure 5 input tables."""

    missing_readiness = [
        col for col in REQUIRED_READINESS_COLUMNS
        if col not in readiness_df.columns
    ]
    if missing_readiness:
        raise KeyError(
            "target_qsar_readiness.csv is missing columns: "
            + ", ".join(missing_readiness)
        )

    missing_tier = [
        col for col in REQUIRED_TIER_SUMMARY_COLUMNS
        if col not in tier_summary.columns
    ]
    if missing_tier:
        raise KeyError(
            "readiness_tier_summary.csv is missing columns: "
            + ", ".join(missing_tier)
        )

    missing_family = [
        col for col in REQUIRED_FAMILY_SUMMARY_COLUMNS
        if col not in family_summary.columns
    ]
    if missing_family:
        raise KeyError(
            "readiness_family_summary.csv is missing columns: "
            + ", ".join(missing_family)
        )

    if readiness_df["target_chembl_id"].nunique() != len(readiness_df):
        raise ValueError("Readiness table must contain one row per target.")

    unexpected_tiers = sorted(
        set(readiness_df["readiness_tier"].dropna()) - set(READINESS_TIER_ORDER)
    )
    if unexpected_tiers:
        raise ValueError(
            "Unexpected readiness tiers found: " + "; ".join(unexpected_tiers)
        )

    if int(tier_summary["target_count"].sum()) != len(readiness_df):
        raise ValueError(
            "Tier-summary target counts do not sum to readiness table rows."
        )

    if len(ready_targets) != int((readiness_df["readiness_tier"] == "Ready").sum()):
        raise ValueError(
            "Ready-target table row count does not match readiness table."
        )


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


def _clean_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linewidth=0.4, alpha=0.25)


def _tier_ordered_summary(tier_summary: pd.DataFrame) -> pd.DataFrame:
    order = {tier: idx for idx, tier in enumerate(READINESS_TIER_ORDER)}
    out = tier_summary.copy()
    out["tier_order"] = out["readiness_tier"].map(order)
    return out.sort_values("tier_order").drop(columns=["tier_order"]).reset_index(drop=True)


def plot_tier_distribution(ax: plt.Axes, tier_summary: pd.DataFrame) -> None:
    """Panel A: target counts by QSAR-readiness tier."""

    ordered = _tier_ordered_summary(tier_summary)

    tiers = ordered["readiness_tier"].tolist()
    counts = pd.to_numeric(ordered["target_count"], errors="coerce").to_numpy(dtype=float)
    percents = pd.to_numeric(ordered["percent_of_targets"], errors="coerce").to_numpy(dtype=float)

    x = np.arange(len(tiers))
    colors = [TIER_COLORS.get(tier, "#B8B8B8") for tier in tiers]

    ax.bar(x, counts, color=colors, alpha=0.88, edgecolor="none")

    for xpos, count, pct in zip(x, counts, percents):
        ax.text(
            xpos,
            count + max(counts) * 0.025,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=20, ha="right")
    ax.set_ylabel("Protein targets")
    ax.set_title("QSAR-readiness tier distribution", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.18)
    _clean_axis(ax, grid_axis="y")
    _panel_label(ax, "A")


def _short_target_label(row: pd.Series) -> str:
    name = str(row["target_name"])
    organism = str(row["target_organism"])

    replacements = {
        "Amine oxidase [flavin-containing] B": "MAO-B",
        "Amine oxidase [flavin-containing] A": "MAO-A",
        "Carbonic anhydrase 9": "CA9",
        "Carbonic anhydrase 12": "CA12",
        "Carbonic anhydrase 1": "CA1",
        "Carbonic anhydrase 2": "CA2",
        "Carbonic anhydrase 7": "CA7",
        "Acetylcholinesterase": "AChE",
        "Cholinesterase": "ChE",
        "Estrogen receptor": "ER",
    }

    label = replacements.get(name, name)

    if organism == "Homo sapiens":
        suffix = "human"
    elif organism == "Rattus norvegicus":
        suffix = "rat"
    elif organism == "Electrophorus electricus":
        suffix = "E. electricus"
    elif organism == "Equus caballus":
        suffix = "horse"
    else:
        suffix = organism

    return f"{label}\n({suffix})"


def plot_ready_targets(ax: plt.Axes, ready_targets: pd.DataFrame) -> None:
    """Panel B: Ready-tier targets by unique compound count."""

    if ready_targets.empty:
        ax.text(
            0.5,
            0.5,
            "No Ready-tier targets",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        _panel_label(ax, "B")
        return

    ordered = ready_targets.sort_values(
        ["unique_compounds", "endpoint_homogeneity"],
        ascending=[True, True],
    ).reset_index(drop=True)

    y = np.arange(len(ordered))
    values = pd.to_numeric(ordered["unique_compounds"], errors="coerce").to_numpy(dtype=float)
    labels = [_short_target_label(row) for _, row in ordered.iterrows()]

    colors = [
        "#2CA02C" if str(org) == "Human" else "#72B7B2"
        for org in ordered["organism_group"]
    ]

    ax.barh(y, values, color=colors, alpha=0.88, edgecolor="none")

    max_x = float(np.nanmax(values)) if len(values) else 1.0
    ax.set_xlim(0, max_x * 1.18)

    for ypos, value, endpoint, homogeneity in zip(
        y,
        values,
        ordered["dominant_endpoint"],
        ordered["endpoint_homogeneity"],
    ):
        ax.text(
            value + max_x * 0.015,
            ypos,
            f"{int(value)} | {endpoint} | H={float(homogeneity):.2f}",
            va="center",
            fontsize=7.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Unique compounds")
    ax.set_title("Ready-tier targets ranked by compound coverage", fontsize=11)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="#2CA02C",
            markeredgecolor="none",
            markersize=8,
            label="Human",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="#72B7B2",
            markeredgecolor="none",
            markersize=8,
            label="Non-human",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    _clean_axis(ax, grid_axis="x")
    _panel_label(ax, "B")


def _tier_marker_size(doi_count: pd.Series) -> np.ndarray:
    doi = pd.to_numeric(doi_count, errors="coerce").fillna(0).to_numpy(dtype=float)
    return 24 + 11 * np.sqrt(np.clip(doi, 0, None))


def plot_readiness_criteria_space(ax: plt.Axes, readiness_df: pd.DataFrame) -> None:
    """Panel C: criteria space for target-specific QSAR readiness."""

    working = readiness_df.copy()
    working["unique_compounds"] = pd.to_numeric(
        working["unique_compounds"],
        errors="coerce",
    )
    working["dominant_endpoint_pactivity_range"] = pd.to_numeric(
        working["dominant_endpoint_pactivity_range"],
        errors="coerce",
    )

    for tier in reversed(READINESS_TIER_ORDER):
        subset = working[working["readiness_tier"] == tier].copy()
        if subset.empty:
            continue

        ax.scatter(
            subset["unique_compounds"],
            subset["dominant_endpoint_pactivity_range"],
            s=_tier_marker_size(subset["doi_count"]),
            alpha=0.72 if tier != "Not Suitable" else 0.42,
            linewidths=0.35,
            edgecolors="white",
            c=TIER_COLORS.get(tier, "#B8B8B8"),
            label=f"{tier} (n={len(subset)})",
        )

    ax.axvline(20, color="black", linewidth=0.7, linestyle=":", alpha=0.65)
    ax.axvline(40, color="black", linewidth=0.7, linestyle="--", alpha=0.65)
    ax.axvline(100, color="black", linewidth=0.9, linestyle="-", alpha=0.75)
    ax.axhline(1.5, color="black", linewidth=0.7, linestyle="--", alpha=0.65)
    ax.axhline(2.5, color="black", linewidth=0.9, linestyle="-", alpha=0.75)

    ax.set_xscale("log")
    ax.set_xlabel("Unique compounds per target (log scale)")
    ax.set_ylabel("Dominant-endpoint pActivity range")
    ax.set_title("Readiness criteria space", fontsize=11)

    ax.text(
        102,
        ax.get_ylim()[1] * 0.92,
        "Ready\ncompound\nthreshold",
        fontsize=7.2,
        ha="left",
        va="top",
    )

    ax.legend(
        loc="lower right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.92,
        markerscale=0.75,
    )

    _clean_axis(ax, grid_axis="both")
    _panel_label(ax, "C")


def _family_order_from_summary(family_summary: pd.DataFrame) -> List[str]:
    totals = (
        family_summary[["target_family", "family_target_count"]]
        .drop_duplicates()
        .sort_values(["family_target_count", "target_family"], ascending=[False, True])
    )
    return totals["target_family"].tolist()


def _family_display_labels(families: Sequence[str]) -> List[str]:
    return [FAMILY_LABELS.get(fam, fam) for fam in families]


def plot_family_tier_composition(ax: plt.Axes, family_summary: pd.DataFrame) -> None:
    """Panel D: readiness tier composition by target family."""

    family_order = _family_order_from_summary(family_summary)
    y = np.arange(len(family_order))

    pivot = (
        family_summary
        .pivot_table(
            index="target_family",
            columns="readiness_tier",
            values="target_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(family_order)
        .fillna(0)
    )

    for tier in READINESS_TIER_ORDER:
        if tier not in pivot.columns:
            pivot[tier] = 0

    pivot = pivot.loc[:, list(READINESS_TIER_ORDER)]

    left = np.zeros(len(pivot), dtype=float)

    for tier in READINESS_TIER_ORDER:
        values = pivot[tier].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            color=TIER_COLORS.get(tier, "#B8B8B8"),
            alpha=0.88,
            edgecolor="none",
            label=tier,
        )

        for ypos, val, lft in zip(y, values, left):
            if val >= 3:
                ax.text(
                    lft + val / 2,
                    ypos,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if tier in {"Ready", "Curatable"} else "black",
                )

        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(_family_display_labels(family_order), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Protein targets")
    ax.set_title("Readiness-tier composition by target family", fontsize=11)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=7.5,
    )

    _clean_axis(ax, grid_axis="x")
    _panel_label(ax, "D")


def create_figure5(
    readiness_df: pd.DataFrame,
    tier_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    ready_targets: pd.DataFrame,
    *,
    title: str = "QSAR-readiness landscape of Tier-A Core protein targets",
) -> plt.Figure:
    """Create the complete Figure 5."""

    validate_figure5_inputs(
        readiness_df,
        tier_summary,
        family_summary,
        ready_targets,
    )

    fig = plt.figure(figsize=(14.4, 11.4), constrained_layout=False)
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.95, 1.10],
        width_ratios=[0.92, 1.08],
        hspace=0.36,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_tier_distribution(ax_a, tier_summary)
    plot_ready_targets(ax_b, ready_targets)
    plot_readiness_criteria_space(ax_c, readiness_df)
    plot_family_tier_composition(ax_d, family_summary)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

    fig.subplots_adjust(
        top=0.93,
        left=0.12,
        right=0.985,
        bottom=0.075,
    )

    return fig


def save_figure5(
    fig: plt.Figure,
    output_dir: Path,
    *,
    basename: str = "Figure_5_qsar_readiness",
    dpi: int = 450,
) -> Dict[str, Path]:
    """Save Figure 5 as PNG, PDF and SVG."""

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


def figure5_summary(
    readiness_df: pd.DataFrame,
    tier_summary: pd.DataFrame,
    ready_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Create compact Figure 5 summary table."""

    tier_counts = {
        row["readiness_tier"]: int(row["target_count"])
        for _, row in tier_summary.iterrows()
    }

    rows = [
        {"metric": "total_targets", "value": int(len(readiness_df))},
        {"metric": "ready_targets", "value": tier_counts.get("Ready", 0)},
        {"metric": "curatable_targets", "value": tier_counts.get("Curatable", 0)},
        {"metric": "exploratory_targets", "value": tier_counts.get("Exploratory", 0)},
        {"metric": "not_suitable_targets", "value": tier_counts.get("Not Suitable", 0)},
        {
            "metric": "ready_target_percent",
            "value": float(tier_summary.loc[tier_summary["readiness_tier"] == "Ready", "percent_of_targets"].iloc[0])
            if (tier_summary["readiness_tier"] == "Ready").any()
            else 0.0,
        },
        {
            "metric": "top_ready_target",
            "value": str(ready_targets.iloc[0]["target_chembl_id"]) if len(ready_targets) else "NA",
        },
        {
            "metric": "top_ready_target_name",
            "value": str(ready_targets.iloc[0]["target_name"]) if len(ready_targets) else "NA",
        },
        {
            "metric": "top_ready_target_unique_compounds",
            "value": int(ready_targets.iloc[0]["unique_compounds"]) if len(ready_targets) else 0,
        },
    ]

    return pd.DataFrame(rows)
