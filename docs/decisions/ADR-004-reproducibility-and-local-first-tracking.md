# ADR-004: Reproducibility Standards and Local-First Experiment Tracking

## Status
Accepted

## Context
Research reproducibility requires recording the exact state of code, data, hyperparameters, and execution environment for every result published. Relying solely on cloud-based tracking services (e.g. Weights & Biases, MLflow) introduces external dependencies, potential API outages, authentication friction, and risk of losing data if accounts change.

## Decision
1. Implement a **Local-First Experiment Tracker** (`ExperimentTracker`) that persists all run metadata, configuration snapshots, metrics, and line-by-line model generations to `artifacts/runs/<run_id>/` as plain JSON and YAML files.
2. Generate structured run identifiers embedding timestamps and random hashes.
3. Automatically capture environment fingerprints on every run:
   - Git commit hash, branch name, and dirty working tree status.
   - Operating system, Python version, and CPU/RAM/GPU hardware details.
   - Random seed initialized across Python, NumPy, and PyTorch.
   - SHA-256 hash of the resolved configuration file.
4. Support optional, asynchronous synchronization to Weights & Biases without making it a hard runtime dependency.

## Consequences
* **Positive:** Complete offline functionality; zero external credentials required to run or reproduce experiments; human-readable and version-controllable experiment archives.
* **Negative:** Local disk space in `artifacts/runs/` must be monitored and pruned periodically.
