# Release contents

This release contains the canonical public reproducibility package for CoumarinBioBench-TierA.

## Included
- root reproducibility files (`README.md`, `LICENSE`, `CITATION.cff`, `environment.yml`, `requirements.txt`)
- retrieval package in `scripts/retrieval/`
- canonical pipeline scripts in `scripts/`
- reusable Python package in `src/coumarinbiobench/`
- processed data in `data/processed/`
- selected metadata in `data/metadata/`
- final supplementary and manuscript-facing tables in `outputs/tables/`
- final figures and figure source-data in `outputs/figures/`
- generated checksum manifest in `checksums/checksums_sha256.txt`

## Excluded
- draft manuscripts and old manuscript versions
- notebooks
- logs
- caches
- QC/bootstrap/fix/repair scripts
- raw frozen export
- interim data
- redundant / empty files
- non-canonical stubs

## Canonical decisions
- `scripts/07_assess_qsar_readiness.py` retained; `scripts/07_compute_qsar_readiness.py` excluded
- `scripts/09_analyze_activity_cliffs.py` retained; `scripts/09_activity_cliff_analysis.py` excluded
- `outputs/tables/Supplementary_Table_S4_target_qsar_readiness.csv` retained; empty `..._full.csv` excluded
- `outputs/figures/Figure_6_activity_cliff_utility.*` retained; `Figure_6_activity_cliff.png` excluded
- `config.yaml` excluded from the public release
- raw and interim data excluded from the public release
