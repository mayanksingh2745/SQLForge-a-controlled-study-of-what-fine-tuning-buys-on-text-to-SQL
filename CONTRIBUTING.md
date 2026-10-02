# Contributing to SQLForge

Thank you for your interest in contributing to SQLForge! This document outlines our development standards, code style, testing requirements, and experiment workflow.

---

## 1. Development Principles

1. **Empirical Rigor:** Every experimental claim must be backed by reproducible execution results, documented seeds, and statistical confidence intervals.
2. **Modular Architecture:** Research code, model interfaces, dataset adapters, execution sandboxes, and serving layers must remain strictly decoupled.
3. **No Unsafe Execution:** Generated SQL must never be executed against production databases. Read-only permissions, timeouts, and sandboxes are mandatory.
4. **Offline First:** Local experimentation and unit testing must never require external API keys, W&B logins, or cloud infrastructure to run.

---

## 2. Setting Up Your Environment

### Prerequisites
* Python 3.11+ (Python 3.11 - 3.13 supported)
* Git

### Installation
Clone the repository and install the development dependencies:

```bash
# Clone repository
git clone https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL.git
cd SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install in editable mode with development extras
pip install -e ".[dev]"
```

Verify your environment with the CLI:
```bash
sqlforge env
```

---

## 3. Code Quality and Testing Standards

We enforce strict linting, formatting, and typing standards via **Ruff**, **Mypy**, and **Pytest**.

### Running Checks

#### On Windows (PowerShell):
```powershell
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/sqlforge

# Run test suite
pytest -v
```
Alternatively, run the automated script:
```powershell
.\scripts\run_checks.ps1
```

#### On Linux / macOS (or Git Bash):
```bash
# Using Makefile
make lint
make typecheck
make test
make all
```

---

## 4. Branching and Commit Guidelines

* Create feature branches off `main`: `feature/your-feature-name` or `experiment/exp-id-topic`.
* Use Conventional Commits formatting:
  * `feat: add schema serializer for BIRD format`
  * `fix: handle NULL column values in execution comparator`
  * `docs: update research plan with sample size rationale`
  * `refactor: isolate database adapters from query generator`
  * `test: add unit tests for bootstrap confidence intervals`

---

## 5. Adding New Experiments Safely

When introducing a new experiment:
1. Define your experiment configuration in a new or extended YAML file in `configs/`.
2. Do not hardcode model IDs, prompt templates, or dataset splits in python code.
3. Ensure the experiment config validates against `sqlforge.schemas.experiments.ExperimentConfig`.
4. Run `sqlforge config validate --config configs/your_config.yaml` to ensure schema compliance before scheduling runs.
5. Record an Architecture Decision Record (ADR) in `docs/decisions/` if adding new architectural abstractions.
