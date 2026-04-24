"""Figure 6 visualization utilities for CoumarinBioBench.

Figure 6 provides a compact benchmark-utility demonstration showing that
Ready-tier target subsets contain local activity-landscape signal suitable
for downstream SAR/activity-cliff analysis.

No mechanistic binding interpretation is encoded in this figure.
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


CASE_ORDER: Tuple[str, ...] = (
    "MAOB_HUMAN_IC50",
    "CA9_HUMAN_KI",
    "ACHE_HUMAN_IC50",
)

CASE_LABELS: Dict[str, str] = {
    "MAOB_HUMAN_IC50": "Human MAO-B\nIC50",
    "CA9_HUMAN_KI": "Human CA9\nKi",
    "ACHE_HUMAN_IC50": "Human AChE\nIC50",
}

CASE_SHORT_LABELS: Dict[str, str] = {
    "MAOB_HUMAN_IC50": "MAO-B",
    "CA9_HUMAN_KI": "CA9",
    "ACHE_HUMAN_IC50": "AChE",
}

REQUIRED_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "case_id",
    "target_label",
    "target_chembl_id",
    "endpoint",
    "compound_count",
    "near_neighbor_pair_count_tanimoto_ge_0_70",
    "activity_cliff_pair_count",
    "cliff_density_per_100_compounds",
    "max_delta_pactivity_cliffs",
    "max_sali_cliffs",
)

REQUIRED_PAIR_COLUMNS: Tuple[str, ...] = (
    "case_id",
    "compound_id_a",
    "compound_id_b",
    "delta_pactivity",
    "tanimoto_ecfp4",
    "sali",
    "activity_cliff_flag",
)

REQUIRED_REP_COLUMNS: Tuple[str, ...] = (
    "case_id",
    "representative_rank",
    "compound_id_a",
    "compound_id_b",
    "delta_pactivity",
    "tanimoto_ecfp4",
    "sali",
)


def validate_figure6_inputs(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    representative_df: pd.DataFrame,
) -> None:
    """Validate required Figure 6 inputs."""

    missing_summary = [
        col for col in REQUIRED_SUMMARY_COLUMNS
        if col not in summary_df.columns
    ]
    if missing_summary:
        raise KeyError(
            "p6_activity_cliff_summary.csv is missing columns: "
            + ", ".join(missing_summary)
        )

    missing_pairs = [
        col for col in REQUIRED_PAIR_COLUMNS
        if col not in pair_df.columns
    ]
    if missing_pairs:
        raise KeyError(
            "p6_activity_cliff_pairs.csv is missing columns: "
            + ", ".join(missing_pairs)
        )

    missing_rep = [
        col for col in REQUIRED_REP_COLUMNS
        if col not in representative_df.columns
    ]
    if missing_rep:
        raise KeyError(
            "p6_representative_cliff_pairs.csv is missing columns: "
            + ", ".join(missing_rep)
        )

    if set(summary_df["case_id"].astype(str)) != set(CASE_ORDER):
        raise ValueError("Summary table does not contain the expected three P6 cases.")

    if int(pair_df["activity_cliff_flag"].sum()) <= 0:
        raise ValueError("No activity-cliff pairs are available for Figure 6.")


def _order_cases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    order_map = {case: idx for idx, case in enumerate(CASE_ORDER)}
    out["case_order"] = out["case_id"].map(order_map)
    return out.sort_values("case_order").drop(columns=["case_order"]).reset_index(drop=True)


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


def plot_pair_count_summary(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    """Panel A: compounds, near-neighbour pairs and cliff pairs."""

    ordered = _order_cases(summary_df)

    labels = [CASE_LABELS.get(case, case) for case in ordered["case_id"]]
    y = np.arange(len(ordered))

    compound_count = pd.to_numeric(ordered["compound_count"], errors="coerce")
    near_pairs = pd.to_numeric(
        ordered["near_neighbor_pair_count_tanimoto_ge_0_70"],
        errors="coerce",
    )
    cliff_pairs = pd.to_numeric(
        ordered["activity_cliff_pair_count"],
        errors="coerce",
    )

    height = 0.24

    ax.barh(y + height, compound_count, height=height, label="Compounds", alpha=0.88)
    ax.barh(y, near_pairs, height=height, label="Near-neighbour pairs", alpha=0.88)
    ax.barh(y - height, cliff_pairs, height=height, label="Cliff pairs", alpha=0.88)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count (log scale)")
    ax.set_title("Case-study subset size and cliff signal", fontsize=11)

    for ypos, value in zip(y + height, compound_count):
        ax.text(value * 1.05, ypos, f"{int(value):,}", va="center", fontsize=8)
    for ypos, value in zip(y, near_pairs):
        ax.text(value * 1.05, ypos, f"{int(value):,}", va="center", fontsize=8)
    for ypos, value in zip(y - height, cliff_pairs):
        ax.text(value * 1.08, ypos, f"{int(value):,}", va="center", fontsize=8)

    ax.legend(loc="lower right", fontsize=8, frameon=True)
    _clean_axis(ax, grid_axis="x")
    _panel_label(ax, "A")


def plot_cliff_density(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    """Panel B: cliff density versus max delta pActivity."""

    ordered = _order_cases(summary_df)

    x = pd.to_numeric(
        ordered["cliff_density_per_100_compounds"],
        errors="coerce",
    ).to_numpy(dtype=float)
    y = pd.to_numeric(
        ordered["max_delta_pactivity_cliffs"],
        errors="coerce",
    ).to_numpy(dtype=float)
    sizes = 40 + pd.to_numeric(
        ordered["compound_count"],
        errors="coerce",
    ).to_numpy(dtype=float) * 0.22

    ax.scatter(x, y, s=sizes, alpha=0.82, edgecolors="white", linewidths=0.8)

    for xpos, ypos, case in zip(x, y, ordered["case_id"]):
        ax.text(
            xpos + max(x) * 0.035,
            ypos,
            CASE_SHORT_LABELS.get(case, case),
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Cliff density per 100 compounds")
    ax.set_ylabel("Maximum ΔpActivity among cliff pairs")
    ax.set_title("Target-wise cliff density and cliff amplitude", fontsize=11)

    ax.set_xlim(0, max(x) * 1.35)
    ax.set_ylim(0, max(y) * 1.20)

    _clean_axis(ax, grid_axis="both")
    _panel_label(ax, "B")


def plot_delta_distribution(ax: plt.Axes, pair_df: pd.DataFrame) -> None:
    """Panel C: delta pActivity distribution among cliff pairs."""

    cliffs = pair_df[pair_df["activity_cliff_flag"] == 1].copy()

    data = []
    labels = []

    for case in CASE_ORDER:
        values = pd.to_numeric(
            cliffs.loc[cliffs["case_id"] == case, "delta_pactivity"],
            errors="coerce",
        ).dropna()
        data.append(values.to_numpy(dtype=float))
        labels.append(CASE_LABELS.get(case, case))

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )

    for patch in box["boxes"]:
        patch.set_alpha(0.72)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=18,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.35,
        )

    ax.axhline(2.0, linestyle="--", linewidth=0.8, color="black", alpha=0.65)
    ax.text(
        0.55,
        2.03,
        "cliff threshold",
        fontsize=7.5,
        va="bottom",
        ha="left",
    )

    ax.set_ylabel("ΔpActivity among cliff pairs")
    ax.set_title("Activity-cliff amplitude distribution", fontsize=11)

    _clean_axis(ax, grid_axis="y")
    _panel_label(ax, "C")


def _format_pair_id(row: pd.Series) -> str:
    return f"{row['compound_id_a']} / {row['compound_id_b']}"


def plot_representative_table(ax: plt.Axes, representative_df: pd.DataFrame) -> None:
    """Panel D: compact table of top representative cliff pairs."""

    ax.axis("off")

    rows = []

    for case in CASE_ORDER:
        subset = representative_df[
            (representative_df["case_id"] == case)
            & (representative_df["representative_rank"] <= 3)
        ].copy()

        subset = subset.sort_values("representative_rank")

        for _, row in subset.iterrows():
            rows.append(
                [
                    CASE_SHORT_LABELS.get(case, case),
                    _format_pair_id(row),
                    f"{float(row['delta_pactivity']):.2f}",
                    f"{float(row['tanimoto_ecfp4']):.3f}",
                    f"{float(row['sali']):.1f}",
                ]
            )

    col_labels = ["Target", "Compound pair", "ΔpAct", "Tc", "SALI"]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.13, 0.45, 0.12, 0.10, 0.10],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EFEFEF")
        else:
            cell.set_facecolor("white")

    ax.set_title("Representative high-SALI cliff pairs", fontsize=11, pad=12)
    _panel_label(ax, "D")


def create_figure6(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    *,
    title: str = "Activity-cliff utility demonstration in Ready-tier target subsets",
) -> plt.Figure:
    """Create complete Figure 6."""

    validate_figure6_inputs(summary_df, pair_df, representative_df)

    fig = plt.figure(figsize=(14.6, 10.8), constrained_layout=False)

    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 1.05],
        hspace=0.38,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_pair_count_summary(ax_a, summary_df)
    plot_cliff_density(ax_b, summary_df)
    plot_delta_distribution(ax_c, pair_df)
    plot_representative_table(ax_d, representative_df)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)

    fig.subplots_adjust(
        top=0.93,
        left=0.10,
        right=0.985,
        bottom=0.075,
    )

    return fig


def save_figure6(
    fig: plt.Figure,
    output_dir: Path,
    *,
    basename: str = "Figure_6_activity_cliff_utility",
    dpi: int = 450,
) -> Dict[str, Path]:
    """Save Figure 6 as PNG, PDF and SVG."""

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


def figure6_summary(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create compact Figure 6 summary table."""

    cliffs = pair_df[pair_df["activity_cliff_flag"] == 1].copy()

    rows = [
        {"metric": "case_study_targets", "value": int(summary_df["case_id"].nunique())},
        {
            "metric": "total_compounds_across_case_tables",
            "value": int(summary_df["compound_count"].sum()),
        },
        {
            "metric": "near_neighbor_pairs_tanimoto_ge_0_70",
            "value": int(len(pair_df)),
        },
        {
            "metric": "activity_cliff_pairs_delta_ge_2_00",
            "value": int(len(cliffs)),
        },
        {
            "metric": "max_delta_pactivity",
            "value": float(cliffs["delta_pactivity"].max()) if len(cliffs) else np.nan,
        },
        {
            "metric": "max_sali",
            "value": float(cliffs["sali"].max()) if len(cliffs) else np.nan,
        },
        {
            "metric": "target_with_most_cliff_pairs",
            "value": str(
                summary_df.sort_values(
                    "activity_cliff_pair_count",
                    ascending=False,
                ).iloc[0]["case_id"]
            ),
        },
    ]

    return pd.DataFrame(rows)
