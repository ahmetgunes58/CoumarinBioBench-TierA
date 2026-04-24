# Retrieval Package

This directory documents the retrieval stage used to assemble the raw ChEMBL-derived coumarin activity export for the CoumarinBioBench-TierA benchmark.

## Scope of this package

The retrieval workflow was designed as a two-stage, provenance-preserving process.

### Stage 1. Coumarin-core identification
Coumarin-containing molecular entries were first identified from the public ChEMBL web interface using a substructure query corresponding to the 2H-chromen-2-one / coumarin core.

The resulting ChEMBL molecule identifiers were preserved locally as:

- `ChEMBL_ID.txt`

This file serves as the fixed molecule-level input for the retrieval step documented here.

### Stage 2. Molecule-level API retrieval and metadata assembly
The script:

- `01_fetch_chembl_coumarin_master_dataset.py`

processes the preserved ChEMBL molecule-ID list and performs molecule-by-molecule retrieval through the ChEMBL API.

For each molecule identifier, the script retrieves activity-level records and assembles associated metadata, including:

- activity identifiers
- assay identifiers
- target identifiers
- document identifiers
- endpoint type
- activity relation
- standard value
- standard unit
- pChEMBL value
- source identifier
- record identifier
- ChEMBL data-validity comments

The activity table is then enriched with molecule, assay, target, and document metadata, including:

- canonical SMILES
- InChIKey
- assay confidence score
- target type
- target organism
- UniProt accession
- DOI
- journal
- publication year

## Important clarification

This script documents the molecule-level retrieval and metadata-assembly step.

It does **not** perform the initial coumarin-core substructure search itself.  
The coumarin-core identification step was completed first through the public ChEMBL web interface, and the resulting molecule-ID list was frozen as `ChEMBL_ID.txt`.

## Frozen manuscript raw export

The frozen raw export used for all manuscript analyses was generated on:

- **10 February 2026**

and preserved as:

- `coumarin_master_20260210_004558.csv`

This frozen export contained:

- **180,653 raw activity records**
- **21,632 unique molecular entries**
- **2,082 ChEMBL target identifiers**

The raw file also retained batch-level `fetch_time` timestamps ranging from:

- **2026-02-10 00:26:24**
- **2026-02-10 00:33:02**

This frozen export was treated as the fixed input to the Tier-A curation pipeline and was not replaced during later manuscript-development steps.

## Repeat retrieval check

To verify retrieval reproducibility, the preserved molecule-ID list (`ChEMBL_ID.txt`) was used again in a repeat retrieval performed on:

- **22 April 2026**

This repeat run generated:

- `coumarin_master_20260422_141057.csv`
- `summary_20260422_141057.json`

and reproduced the same key counts as the frozen manuscript export:

- **180,653 raw activity records**
- **21,632 successful molecule entries**

Additional repeat-retrieval summary values:

- **100.0% SMILES coverage**
- **47.5% raw DOI coverage**

This repeat run was used only as a reproducibility confirmation for the retrieval and metadata-assembly step.  
All manuscript analyses remained anchored to the frozen 10 February 2026 export.

## Files in this directory

- `01_fetch_chembl_coumarin_master_dataset.py` — molecule-level ChEMBL API retrieval and metadata assembly script
- `ChEMBL_ID.txt` — preserved molecule-ID list obtained from the coumarin-core substructure query
- `retrieval_metadata.json` — structured metadata describing the frozen export and repeat retrieval
- `README.md` — this document

## Reproducibility note

This retrieval package is intended to document how the raw ChEMBL-derived coumarin export was assembled and to enable equivalent reconstruction of the Tier-A workflow from the same scaffold-query logic.

It should be interpreted as part of the benchmark audit trail rather than as a separate analytical pipeline.