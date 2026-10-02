# SQLForge Data Management Directory

This directory houses raw, interim, and processed data assets for SQLForge experiments.

## Directory Structure
* `raw/`: Unmodified source benchmark archives and assets (e.g., Spider train/dev, BIRD subsets, database schemas). Never modified in place.
* `interim/`: Intermediate representations, partitioned tables, serialized schema graphs, or pre-tokenized chunks.
* `processed/`: Canonical, normalized `TextToSQLExample` records formatted as Parquet / JSONL, strictly partitioned by split (train, dev, test, held_out) with leakage verification signatures.

## Data Governance Rules
1. **Never Commit Raw Data or DB Files:** All files in `data/raw`, `data/interim`, and `data/processed` are gitignored except for `.gitkeep`.
2. **Provenance & Integrity:** Every processed dataset artifact must contain a provenance manifest linking to the exact source release and checksum.
3. **No Contamination:** Benchmark evaluation splits (Spider dev, BIRD subset, held-out schemas) must never leak into training partitions.
4. **See Documentation:** Consult `docs/data_governance.md` for license, privacy, and synthetic data guidelines.
