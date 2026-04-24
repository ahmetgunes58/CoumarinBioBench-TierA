# CoumarinBioBench-TierA

**CoumarinBioBench-TierA** is an auditable, ChEMBL-derived benchmark and public reproducibility package for **organism-aware target-family mapping**, **target-level QSAR-readiness assessment**, and **fingerprint-based activity-landscape analysis** in **coumarin-containing bioactivity data**.

This repository accompanies the manuscript:

**CoumarinBioBench-TierA: An Auditable ChEMBL-Derived Benchmark for Organism-Aware Target-Family Mapping and QSAR-Readiness Assessment**

It was built to address a recurring scaffold-focused cheminformatics problem: although public bioactivity repositories contain large volumes of useful medicinal chemistry data, raw database extractions are not automatically suitable for benchmark construction or endpoint-specific modelling without explicit curation, provenance control, and transparent target-level interpretation. This release therefore provides not only processed outputs, but also the retrieval logic, benchmark-construction workflow, canonical supplementary tables, figure source-data files, and reproducibility metadata required to inspect and reuse the study in a structured way.

---

## Repository highlights

This public release contains the **canonical reproducibility package** for the CoumarinBioBench-TierA study. It includes:

- a documented **retrieval package** for reconstructing the coumarin-associated ChEMBL activity export,
- the **canonical benchmark-construction and analysis scripts**,
- the reusable Python source package in `src/coumarinbiobench/`,
- the **processed Tier-A benchmark outputs** used in the manuscript,
- the **supplementary tables** cited in the Supporting Information,
- **figure source-data files** for the manuscript figures,
- benchmark metadata, release manifest, and checksum files,
- the software environment description needed for computational reproducibility.

This release is intentionally **curated rather than exhaustive**. Draft manuscripts, notebooks, logs, patch-stage helper scripts, raw frozen export files, interim tables, caches, and other development-only artifacts are excluded so that the repository remains interpretable, lightweight, and suitable for external review.

---

## Benchmark at a glance

Starting from a ChEMBL-derived coumarin retrieval, the workflow defined a conservative, protein-centric Tier-A benchmark through multi-stage curation. The final analytical benchmark reported in the manuscript contains:

- **180,653** raw ChEMBL-associated activity records in the frozen retrieval layer,
- **11,579** Tier-A Core records,
- **5,390** unique coumarin-containing compounds,
- **632** protein targets,
- **14** Ready-tier target-endpoint subsets under the pre-specified QSAR-readiness framework,
- **67** activity-cliff pairs across three selected human Ready-tier case-study subsets.

The repository is organized so that a reader can reconstruct the benchmark logic step by step: retrieval, curation, descriptor generation, chemical-space analysis, organism-aware target annotation, readiness assessment, and activity-cliff analysis.

---

## Scientific scope

This repository supports three linked forms of reproducibility.

### 1. Retrieval-level reproducibility
The release documents how the ChEMBL-derived coumarin activity export was assembled. The retrieval package preserves the ChEMBL molecule-ID list and the molecule-level retrieval/metadata-assembly script used to reconstruct the raw export logic. A repeat retrieval check reproduced the same main raw counts as the frozen manuscript export.

### 2. Computational reproducibility
The release includes the benchmark-construction scripts, processed benchmark tables, supplementary outputs, figure source-data files, and software environment specification. These materials are intended to allow readers to follow and reproduce the benchmark workflow without relying on hidden local steps.

### 3. Interpretive reproducibility
The benchmark is not only a collection of processed CSV files. It is also an interpretive framework built around:
- organism-aware target annotation,
- broad target-family harmonization,
- target-level QSAR-readiness classification,
- and benchmark-level activity-landscape outputs.

The repository therefore includes the metadata and output tables needed to reproduce not only the numerical results, but also the logic behind the manuscript’s biological and modelling interpretations.

---

## Repository structure

```text
CoumarinBioBench-TierA/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ environment.yml
├─ requirements.txt
│
├─ checksums/
│  └─ checksums_sha256.txt
│
├─ docs/
│  ├─ workflow_overview.md
│  ├─ reproducibility_guide.md
│  ├─ analysis_decisions.md
│  ├─ changelog.md
│  ├─ release_contents.md
│  └─ release_manifest_generated.md
│
├─ scripts/
│  ├─ retrieval/
│  │  ├─ 01_fetch_chembl_coumarin_master_dataset.py
│  │  ├─ ChEMBL_ID.txt
│  │  ├─ README.md
│  │  └─ retrieval_metadata.json
│  └─ ...
│
├─ src/
│  └─ coumarinbiobench/
│
├─ data/
│  ├─ metadata/
│  └─ processed/
│
└─ outputs/
   ├─ tables/
   └─ figures/
      └─ source_data/
```

The public release tree has been checked to ensure that required files are present and non-canonical development artifacts are excluded.

---

## Canonical workflow decisions

To keep the release unambiguous, the public package follows explicit canonical-file decisions.

### Canonical script selections
The release retains:
- `scripts/07_assess_qsar_readiness.py`
- `scripts/09_analyze_activity_cliffs.py`

and excludes the corresponding stub files:
- `scripts/07_compute_qsar_readiness.py`
- `scripts/09_activity_cliff_analysis.py`

because the retained files are the actual working runners that generate the manuscript-facing outputs.

### Canonical Supplementary Table S4
The release retains:
- `outputs/tables/Supplementary_Table_S4_target_qsar_readiness.csv`

and excludes:
- `outputs/tables/Supplementary_Table_S4_qsar_readiness_full.csv`

because the retained file is the canonical manuscript-facing readiness export.

### Canonical Figure 6 assets
The release retains:
- `Figure_6_activity_cliff_utility.pdf`
- `Figure_6_activity_cliff_utility.png`
- `Figure_6_activity_cliff_utility.svg`

and excludes the redundant alternative naming variant so that the figure set remains consistent across Figures 1-6.

---

## Included data layers

### `data/processed/`
This directory contains the processed benchmark outputs used in the manuscript, including:
- the Tier-A Core benchmark table,
- compound-target edges,
- target-degree and family-degree tables,
- descriptor matrix and descriptor summaries,
- endpoint composition summaries,
- target-family annotation,
- readiness outputs,
- chemical-space outputs,
- and activity-cliff case-study outputs.

### `outputs/tables/`
This directory contains the canonical supplementary tables and manuscript-facing summary tables, including:
- Supplementary Tables S0-S6,
- benchmark summary tables,
- readiness summary tables,
- and activity-cliff summary tables.

### `outputs/figures/source_data/`
This directory contains the plot-ready source-data files corresponding to the main manuscript figures.

### `data/metadata/`
This directory contains the benchmark metadata needed for provenance and interpretation, including:
- ChEMBL version/query note,
- dataset freeze report,
- data dictionary,
- organism summary,
- and UniProt coverage report.

---

## Retrieval package

The retrieval package is located in:

```text
scripts/retrieval/
```

It documents the two-stage retrieval logic used for the benchmark:

1. **Stage 1:** identify coumarin-containing ChEMBL molecule entries through the coumarin core query and preserve the resulting ChEMBL ID list,
2. **Stage 2:** perform molecule-level activity and metadata retrieval using the preserved ID list.

The public release includes:
- the ChEMBL molecule-ID list,
- the retrieval script,
- the retrieval README,
- and structured retrieval metadata.

This package should be interpreted as part of the benchmark audit trail rather than as a stand-alone modelling workflow.

---

## What is intentionally not included

The public release does **not** include:
- draft manuscripts and historical manuscript versions,
- local notebooks,
- execution logs,
- patch-stage helper scripts,
- Python cache artifacts,
- the frozen raw export file,
- interim curation outputs not needed for the canonical public workflow.

This exclusion policy is intentional. The aim is to provide a **clean, canonical reproducibility package**, not a dump of every file created during project development.

---

## Reproducibility files

### `environment.yml`
The conda environment specification used for the released workflow.

### `requirements.txt`
A lightweight dependency reference for users who prefer pip-based environment reconstruction.

### `checksums/checksums_sha256.txt`
The SHA-256 checksum manifest for the release package. This file supports file-integrity verification at the archive level.

### `docs/release_contents.md`
Human-readable description of what is included and excluded from the release.

### `docs/release_manifest_generated.md`
Machine-generated manifest showing the files that were copied into the release package and the explicit exclusion list used during release construction.

---

## Quick start

### A. Inspect the benchmark outputs
A practical entry point is:

- `data/processed/CoumarinBioBench_TierA_core.csv`
- `outputs/tables/Supplementary_Table_S0_curation_audit.csv`
- `outputs/tables/Supplementary_Table_S4_target_qsar_readiness.csv`
- `outputs/tables/Supplementary_Table_S6_activity_cliff_pairs.csv`

### B. Inspect figure source data
For figure-level review, begin with:
- `outputs/figures/source_data/README_figure_source_data.md`
- the corresponding `Figure_*_source_data.csv` files.

### C. Reconstruct the workflow
For end-to-end reproducibility, consult:
- `docs/workflow_overview.md`
- `docs/reproducibility_guide.md`
- `scripts/retrieval/README.md`
- the canonical pipeline scripts under `scripts/`
- and the source package under `src/coumarinbiobench/`.

---

## Suggested reading order

For a first-time user, the most useful order is:

1. `README.md`
2. `docs/workflow_overview.md`
3. `docs/reproducibility_guide.md`
4. `scripts/retrieval/README.md`
5. `outputs/tables/Supplementary_Table_S0_curation_audit.csv`
6. `outputs/tables/Supplementary_Table_S4_target_qsar_readiness.csv`
7. `outputs/figures/source_data/README_figure_source_data.md`

This sequence gives a compact route from project overview to retrieval, curation, readiness interpretation, and figure-level outputs.

---

## Citation

If you use this repository, please cite the accompanying manuscript and consult `CITATION.cff` for the repository citation metadata.

---

## Repository status

This repository represents the **canonical public release package** for the CoumarinBioBench-TierA benchmark as prepared for manuscript support, supplementary-data release, and external reproducibility review.

The release audit confirms:
- no missing required files,
- no forbidden manuscript-stage artifacts inside the canonical public package,
- and all key supplementary tables and figure files present in the release tree.
