# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-02

### Added
- **Step 0 Foundation:** Established production-grade repository architecture and developer tooling for SQLForge.
- **Packaging:** Standard `src/` layout with `pyproject.toml` supporting modular extras (`dev`, `train`, `eval`, `serving`).
- **Core CLI:** `sqlforge` CLI entry point with `env` system inspection, `config validate`, and `experiment init` commands.
- **Typed Schemas:** Robust Pydantic v2 domain models for `SchemaMetadata`, `TextToSQLExample`, `ModelConfig`, `ExperimentConfig`, `ExecutionResult`, `EvaluationMetrics`, and `RunMetadata`.
- **Reproducibility Framework:** Seed management across Python, NumPy, and PyTorch, hardware/software fingerprinting, and run ID generation.
- **Local-First Experiment Tracking:** Zero-credential run registry writing structured JSON/YAML snapshots to `artifacts/runs/`.
- **Modular Configuration System:** Layered YAML configs for datasets, models, experiments, defaults, and evaluation.
- **Documentation Suite:** Architectural specifications, research plan, experiment protocol, data governance, development setup, and research glossary.
- **Architecture Decision Records (ADRs):** ADR-001 through ADR-004 covering layout, typing, SQL safety, and tracking.
- **CI & Quality Checks:** GitHub Actions CI workflow, Ruff linting/formatting rules, MyPy type checking, and Pytest test suite.
