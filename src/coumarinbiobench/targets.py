"""Target-family annotation utilities for CoumarinBioBench.

P4a assigns organism-aware functional target-family labels to the
632 Tier-A Core protein targets and generates target-family summary tables.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


COMPOUND_ID_CANDIDATES: Tuple[str, ...] = (
    "molecule_chembl_id",
    "compound_chembl_id",
    "chembl_id",
    "molecule_id",
    "compound_id",
)

TARGET_ID_CANDIDATES: Tuple[str, ...] = (
    "target_chembl_id",
    "target_id",
)

TARGET_NAME_CANDIDATES: Tuple[str, ...] = (
    "target_pref_name_full",
    "target_pref_name",
    "target_name",
    "pref_name",
)

TARGET_ORGANISM_CANDIDATES: Tuple[str, ...] = (
    "target_organism_full",
    "target_organism",
    "organism",
)

UNIPROT_CANDIDATES: Tuple[str, ...] = (
    "uniprot_accession",
    "accession",
    "uniprot_id",
)

ENDPOINT_CANDIDATES: Tuple[str, ...] = (
    "standard_type",
    "endpoint",
    "endpoint_type",
    "activity_type",
)

DOI_CANDIDATES: Tuple[str, ...] = (
    "doi",
    "document_doi",
)

YEAR_CANDIDATES: Tuple[str, ...] = (
    "year",
    "publication_year",
)

PCHEMBL_CANDIDATES: Tuple[str, ...] = (
    "pchembl_value",
    "pActivity",
    "pactivity",
)

TARGET_FAMILIES: Tuple[str, ...] = (
    "Carbonic anhydrases",
    "Monoamine oxidases",
    "Cholinesterases",
    "Kinases",
    "Nuclear receptors",
    "G protein-coupled receptors",
    "Proteases",
    "Cytochrome P450s",
    "Ion channels and transporters",
    "Other enzymes",
    "Other/unclassified",
)


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
    label: str = "column",
) -> Optional[str]:
    """Resolve a column name using exact and punctuation-tolerant matching."""

    if df is None or df.empty:
        if required:
            raise ValueError(f"Cannot resolve {label}: dataframe is empty.")
        return None

    exact_lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact_lookup:
            return str(exact_lookup[candidate.lower()])

    normalised_lookup = {_normalise_name(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalise_name(candidate)
        if key in normalised_lookup:
            return str(normalised_lookup[key])

    if required:
        available = ", ".join(map(str, df.columns))
        expected = ", ".join(candidates)
        raise KeyError(
            f"Could not resolve {label}. Expected one of [{expected}]. "
            f"Available columns: {available}"
        )
    return None


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _mode_or_missing(series: pd.Series) -> object:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        return pd.NA
    modes = clean.mode()
    if modes.empty:
        return clean.iloc[0]
    return modes.iloc[0]


def _sorted_unique_join(values: Iterable[object], sep: str = "; ") -> str:
    clean = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return sep.join(clean)


def _dominant_and_homogeneity(series: pd.Series) -> Tuple[object, float]:
    clean = series.dropna().astype(str)
    clean = clean[clean.str.strip() != ""]
    if clean.empty:
        return pd.NA, np.nan

    counts = clean.value_counts()
    dominant = counts.index[0]
    homogeneity = float(counts.iloc[0] / counts.sum())
    return dominant, homogeneity


def classify_organism(organism: object) -> str:
    """Classify target organism into human/non-human/unknown."""

    text = _clean_text(organism).lower()
    if not text:
        return "Unknown"
    if text == "homo sapiens" or "homo sapiens" in text:
        return "Human"
    return "Non-human"


def assign_target_family(target_name: object, target_organism: object = None) -> str:
    """Assign a broad functional target family from target name patterns.

    The rules are deliberately conservative and transparent. They are intended
    for benchmark-level landscape analysis, not for replacing expert curation
    of individual protein annotations.
    """

    name = _clean_text(target_name).lower()
    if not name:
        return "Other/unclassified"

    # Highly scaffold-relevant enzyme families first.
    if "carbonic anhydrase" in name:
        return "Carbonic anhydrases"

    if (
        "monoamine oxidase" in name
        or "amine oxidase [flavin-containing] a" in name
        or "amine oxidase [flavin-containing] b" in name
        or "flavin-containing] a" in name
        or "flavin-containing] b" in name
        or name in {"mao-a", "mao-b"}
    ):
        return "Monoamine oxidases"

    if (
        "acetylcholinesterase" in name
        or "butyrylcholinesterase" in name
        or "cholinesterase" in name
    ):
        return "Cholinesterases"

    # Cytochrome P450 enzymes are separated because they form a distinct
    # ADME/toxicity-relevant family in medicinal chemistry datasets.
    if "cytochrome p450" in name or "cyp" in name:
        return "Cytochrome P450s"

    # Kinases. Some ChEMBL target names do not explicitly contain "kinase"
    # even though the target is a receptor tyrosine kinase, e.g. EGFR/VEGFR.
    kinase_terms = (
        "kinase",
        "protein kinase",
        "receptor tyrosine-protein kinase",
        "serine/threonine-protein kinase",
        "phosphatidylinositol",
        "mammalian target of rapamycin",
        "mtor",
        "epidermal growth factor receptor",
        "vascular endothelial growth factor receptor",
        "platelet-derived growth factor receptor",
        "hepatocyte growth factor receptor",
        "alk tyrosine kinase receptor",
        "proto-oncogene tyrosine-protein kinase",
        "tyrosine-protein kinase",
        "dual specificity mitogen-activated protein kinase kinase",
        "mitogen-activated protein kinase",
    )
    if any(term in name for term in kinase_terms):
        return "Kinases"

    # Nuclear receptors before GPCRs.
    nuclear_terms = (
        "estrogen receptor",
        "androgen receptor",
        "progesterone receptor",
        "glucocorticoid receptor",
        "mineralocorticoid receptor",
        "peroxisome proliferator-activated receptor",
        "retinoic acid receptor",
        "retinoid x receptor",
        "oxysterols receptor",
        "liver x receptor",
        "aryl hydrocarbon receptor",
        "nuclear receptor",
        "photoreceptor-specific nuclear receptor",
        "ror-gamma",
        "receptor rxr",
    )
    if any(term in name for term in nuclear_terms):
        return "Nuclear receptors"

    gpcr_terms = (
        "5-hydroxytryptamine receptor",
        "serotonin receptor",
        "dopamine receptor",
        "adrenergic receptor",
        "adenosine receptor",
        "cannabinoid receptor",
        "opioid receptor",
        "melatonin receptor",
        "histamine h1 receptor",
        "angiotensin ii receptor",
        "neuropeptide y receptor",
        "galanin receptor",
        "sphingosine 1-phosphate receptor",
        "free fatty acid receptor",
        "metabotropic glutamate receptor",
        "hydroxycarboxylic acid receptor",
        "chemokine receptor",
        "g-protein coupled receptor",
        "g protein-coupled receptor",
        "trace amine-associated receptor",
        "sigma non-opioid intracellular receptor",
        "neurotensin receptor",
        "melanin-concentrating hormone receptor",
        "prostaglandin d2 receptor",
        "kappa-type opioid receptor",
    )
    if any(term in name for term in gpcr_terms):
        return "G protein-coupled receptors"

    protease_terms = (
        "protease",
        "proteinase",
        "peptidase",
        "caspase",
        "cathepsin",
        "trypsin",
        "chymotrypsin",
        "elastase",
        "beta-secretase",
        "cruzipain",
        "calpain",
        "metalloproteinase",
        "collagenase",
        "aminopeptidase",
        "proteasome",
        "prenyl protease",
        "prothrombin",
        "coagulation factor",
        "kallikrein",
    )
    if any(term in name for term in protease_terms):
        return "Proteases"

    ion_transport_terms = (
        "ion channel",
        "channel",
        "transporter",
        "transport",
        "solute carrier",
        "monocarboxylate transporter",
        "anion exchanger",
        "potassium voltage-gated",
        "voltage-gated",
        "atp-dependent translocase",
        "abc",
        "bile acid cotransporter",
        "glutamate receptor ionotropic",
        "acetylcholine receptor subunit",
        "multidrug resistance-associated protein",
        "atp-binding cassette",
        "chloride anion exchanger",
    )
    if any(term in name for term in ion_transport_terms):
        return "Ion channels and transporters"

    other_enzyme_terms = (
        "dehydrogenase",
        "oxidase",
        "monooxygenase",
        "reductase",
        "synthase",
        "synthetase",
        "transferase",
        "hydrolase",
        "isomerase",
        "lyase",
        "ligase",
        "phosphatase",
        "phosphodiesterase",
        "polymerase",
        "topoisomerase",
        "telomerase",
        "aromatase",
        "tyrosinase",
        "urease",
        "luciferase",
        "aldolase",
        "lipoxygenase",
        "cyclooxygenase",
        "prostaglandin g/h synthase",
        "glucosylceramidase",
        "beta-lactamase",
        "beta lactamase",
        "dna gyrase",
        "helicase",
        "sulfatase",
        "paraoxonase",
        "arylesterase",
        "glutathione s-transferase",
        "transglutaminase",
        "carbonyl reductase",
        "reverse transcriptase",
        "integrase",
        "deacetylase",
        "demethylase",
        "methyltransferase",
        "glucoamylase",
        "maltase",
        "amylase",
        "cyclase",
        "epimerase",
        "dioxygenase",
        "peroxidase",
        "exonuclease",
        "endonuclease",
        "dna repair protein",
        "apobec",
        "protein disulfide-isomerase",
        "squalene--hopene cyclase",
    )
    if any(term in name for term in other_enzyme_terms):
        return "Other enzymes"

    return "Other/unclassified"


def build_target_family_annotation(core_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-target annotation table."""

    compound_col = resolve_column(
        core_df,
        COMPOUND_ID_CANDIDATES,
        label="compound identifier",
    )
    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )
    name_col = resolve_column(
        core_df,
        TARGET_NAME_CANDIDATES,
        label="target preferred name",
    )
    organism_col = resolve_column(
        core_df,
        TARGET_ORGANISM_CANDIDATES,
        label="target organism",
    )
    uniprot_col = resolve_column(
        core_df,
        UNIPROT_CANDIDATES,
        required=False,
        label="UniProt accession",
    )
    endpoint_col = resolve_column(
        core_df,
        ENDPOINT_CANDIDATES,
        label="endpoint",
    )
    doi_col = resolve_column(
        core_df,
        DOI_CANDIDATES,
        required=False,
        label="DOI",
    )
    year_col = resolve_column(
        core_df,
        YEAR_CANDIDATES,
        required=False,
        label="publication year",
    )
    pchembl_col = resolve_column(
        core_df,
        PCHEMBL_CANDIDATES,
        required=False,
        label="pChEMBL value",
    )

    rows = []

    for target_id, group in core_df.groupby(target_col, dropna=False):
        dominant_endpoint, endpoint_homogeneity = _dominant_and_homogeneity(
            group[endpoint_col]
        )

        target_name = _mode_or_missing(group[name_col])
        organism = _mode_or_missing(group[organism_col])
        uniprot = _mode_or_missing(group[uniprot_col]) if uniprot_col else pd.NA

        pchembl_values = (
            pd.to_numeric(group[pchembl_col], errors="coerce")
            if pchembl_col
            else pd.Series(dtype=float)
        )

        year_values = (
            pd.to_numeric(group[year_col], errors="coerce").dropna()
            if year_col
            else pd.Series(dtype=float)
        )

        doi_values = (
            group[doi_col].dropna().astype(str).str.strip()
            if doi_col
            else pd.Series(dtype=str)
        )
        doi_values = doi_values[doi_values != ""]

        target_family = assign_target_family(target_name, organism)

        rows.append(
            {
                "target_chembl_id": target_id,
                "target_name": target_name,
                "target_organism": organism,
                "organism_group": classify_organism(organism),
                "uniprot_accession": uniprot,
                "target_family": target_family,
                "record_count": int(len(group)),
                "unique_compounds": int(group[compound_col].nunique()),
                "endpoint_types": _sorted_unique_join(group[endpoint_col]),
                "dominant_endpoint": dominant_endpoint,
                "endpoint_homogeneity": endpoint_homogeneity,
                "doi_count": int(doi_values.nunique()),
                "year_min": int(year_values.min()) if not year_values.empty else pd.NA,
                "year_max": int(year_values.max()) if not year_values.empty else pd.NA,
                "pchembl_min": float(pchembl_values.min()) if not pchembl_values.dropna().empty else np.nan,
                "pchembl_max": float(pchembl_values.max()) if not pchembl_values.dropna().empty else np.nan,
                "pchembl_range": (
                    float(pchembl_values.max() - pchembl_values.min())
                    if not pchembl_values.dropna().empty
                    else np.nan
                ),
            }
        )

    annotation = pd.DataFrame(rows)

    family_order = {family: idx for idx, family in enumerate(TARGET_FAMILIES)}
    annotation["family_order"] = annotation["target_family"].map(family_order)
    annotation = annotation.sort_values(
        ["family_order", "unique_compounds", "record_count", "target_chembl_id"],
        ascending=[True, False, False, True],
    ).drop(columns=["family_order"])

    return annotation.reset_index(drop=True)


def add_target_family_to_records(
    core_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge target-family labels back onto Tier-A Core records."""

    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )

    merge_cols = [
        "target_chembl_id",
        "target_family",
        "organism_group",
        "target_name",
        "target_organism",
        "uniprot_accession",
    ]

    return core_df.merge(
        annotation_df[merge_cols],
        left_on=target_col,
        right_on="target_chembl_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_annotated"),
    )


def build_target_family_summary(
    core_annotated_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise target-family representation by records, targets and compounds."""

    compound_col = resolve_column(
        core_annotated_df,
        COMPOUND_ID_CANDIDATES,
        label="compound identifier",
    )
    target_col = resolve_column(
        core_annotated_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )
    endpoint_col = resolve_column(
        core_annotated_df,
        ENDPOINT_CANDIDATES,
        label="endpoint",
    )

    total_records = len(core_annotated_df)
    total_compounds = core_annotated_df[compound_col].nunique()
    total_targets = core_annotated_df[target_col].nunique()

    rows = []

    for family, group in core_annotated_df.groupby("target_family", dropna=False):
        dominant_endpoint, endpoint_homogeneity = _dominant_and_homogeneity(
            group[endpoint_col]
        )

        human_records = int((group["organism_group"] == "Human").sum())
        nonhuman_records = int((group["organism_group"] == "Non-human").sum())

        unique_edges = (
            group[[compound_col, target_col]]
            .drop_duplicates()
            .shape[0]
        )

        rows.append(
            {
                "target_family": family,
                "record_count": int(len(group)),
                "percent_of_records": float(len(group) / total_records * 100.0),
                "unique_targets": int(group[target_col].nunique()),
                "percent_of_targets": float(group[target_col].nunique() / total_targets * 100.0),
                "unique_compounds": int(group[compound_col].nunique()),
                "percent_of_unique_compounds": float(group[compound_col].nunique() / total_compounds * 100.0),
                "unique_compound_target_edges": int(unique_edges),
                "dominant_endpoint": dominant_endpoint,
                "endpoint_homogeneity": endpoint_homogeneity,
                "endpoint_types": _sorted_unique_join(group[endpoint_col]),
                "human_record_count": human_records,
                "nonhuman_record_count": nonhuman_records,
                "human_record_percent_within_family": (
                    float(human_records / len(group) * 100.0) if len(group) else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)

    family_order = {family: idx for idx, family in enumerate(TARGET_FAMILIES)}
    summary["family_order"] = summary["target_family"].map(family_order)
    summary = summary.sort_values(
        ["family_order", "record_count"],
        ascending=[True, False],
    ).drop(columns=["family_order"])

    return summary.reset_index(drop=True)


def build_family_endpoint_composition(core_annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Build endpoint composition by target family."""

    endpoint_col = resolve_column(
        core_annotated_df,
        ENDPOINT_CANDIDATES,
        label="endpoint",
    )

    grouped = (
        core_annotated_df
        .groupby(["target_family", endpoint_col], dropna=False)
        .size()
        .reset_index(name="record_count")
        .rename(columns={endpoint_col: "endpoint"})
    )

    family_totals = (
        grouped.groupby("target_family")["record_count"]
        .sum()
        .rename("family_record_count")
        .reset_index()
    )

    grouped = grouped.merge(family_totals, on="target_family", how="left")
    grouped["percent_of_family_records"] = (
        grouped["record_count"] / grouped["family_record_count"] * 100.0
    )

    return grouped.sort_values(
        ["target_family", "record_count", "endpoint"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_family_organism_summary(core_annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Build organism contribution summary by target family."""

    compound_col = resolve_column(
        core_annotated_df,
        COMPOUND_ID_CANDIDATES,
        label="compound identifier",
    )
    target_col = resolve_column(
        core_annotated_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )

    grouped = (
        core_annotated_df
        .groupby(["target_family", "organism_group", "target_organism"], dropna=False)
        .agg(
            record_count=(target_col, "size"),
            unique_targets=(target_col, "nunique"),
            unique_compounds=(compound_col, "nunique"),
        )
        .reset_index()
    )

    family_totals = (
        grouped.groupby("target_family")["record_count"]
        .sum()
        .rename("family_record_count")
        .reset_index()
    )
    grouped = grouped.merge(family_totals, on="target_family", how="left")
    grouped["percent_of_family_records"] = (
        grouped["record_count"] / grouped["family_record_count"] * 100.0
    )

    return grouped.sort_values(
        ["target_family", "record_count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_compound_family_degree_table(core_annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Compute compound-level family-aware degree alongside ChEMBL-target degree."""

    compound_col = resolve_column(
        core_annotated_df,
        COMPOUND_ID_CANDIDATES,
        label="compound identifier",
    )
    target_col = resolve_column(
        core_annotated_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )

    rows = []

    for compound_id, group in core_annotated_df.groupby(compound_col, dropna=False):
        target_ids = sorted(group[target_col].dropna().astype(str).unique())
        families = sorted(group["target_family"].dropna().astype(str).unique())

        rows.append(
            {
                "compound_id": compound_id,
                "chembl_target_degree": int(len(target_ids)),
                "family_aware_degree": int(len(families)),
                "target_family_list": "; ".join(families),
                "target_chembl_id_list": "; ".join(target_ids),
                "high_family_degree_flag": int(len(families) >= 3),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["family_aware_degree", "chembl_target_degree", "compound_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def validate_target_family_outputs(
    annotation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    core_df: pd.DataFrame,
) -> None:
    """Validate target annotation consistency."""

    target_col = resolve_column(
        core_df,
        TARGET_ID_CANDIDATES,
        label="target identifier",
    )

    expected_targets = int(core_df[target_col].nunique())
    observed_targets = int(annotation_df["target_chembl_id"].nunique())

    if expected_targets != observed_targets:
        raise ValueError(
            f"Target annotation count mismatch: expected {expected_targets}, "
            f"observed {observed_targets}."
        )

    if annotation_df["target_family"].isna().any():
        raise ValueError("Some targets are missing target_family annotations.")

    if annotation_df["organism_group"].isna().any():
        raise ValueError("Some targets are missing organism_group annotations.")

    if int(summary_df["record_count"].sum()) != int(len(core_df)):
        raise ValueError(
            "Family-summary record counts do not sum to Tier-A Core record count."
        )

    unexpected = sorted(set(annotation_df["target_family"]) - set(TARGET_FAMILIES))
    if unexpected:
        raise ValueError(
            "Unexpected target family labels found: " + "; ".join(unexpected)
        )
