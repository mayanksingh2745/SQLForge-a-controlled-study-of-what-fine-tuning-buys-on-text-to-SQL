# SQLForge: A Controlled Study of What Fine-Tuning Buys on Text-to-SQL

[![CI](https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL/actions/workflows/ci.yml/badge.svg)](https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-informational.svg)](https://mypy-lang.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source, research-grade empirical framework investigating whether parameter-efficient fine-tuning (LoRA / QLoRA) on small open-weight language models (1.5B–8B parameters) can match or exceed commercial frontier APIs on text-to-SQL execution accuracy while slashing inference latency, VRAM footprint, and operational costs.

---

## 1. Research Objectives

Commercial frontier models (e.g., GPT-4o, Claude 3.5 Sonnet) achieve strong zero-shot and few-shot execution accuracy on benchmarks like Spider and BIRD. However, their opacity, high tail latencies, cost per million tokens, and cloud-data privacy concerns prevent deployment in privacy-sensitive or air-gapped database environments.

**SQLForge** investigates the following core scientific questions:

1. **In-Domain Execution Accuracy:** Can a fine-tuned 7B open-weight model (e.g., Qwen2.5-Coder-7B) achieve parity with zero-shot GPT-4o mini on Spider?
2. **LoRA vs QLoRA Efficiency:** Does 4-bit NormalFloat (NF4) quantization degrade text-to-SQL execution accuracy relative to 16-bit LoRA, and how much training VRAM does it save?
3. **LoRA Rank Scaling:** Where does execution accuracy saturate as LoRA rank $r$ scales ($r \in \{8, 16, 32, 64\}$)?
4. **Data Scaling Laws:** How does accuracy scale across sample fractions ($10\%, 25\%, 50\%, 100\%$), and does execution-filtered synthetic training data match human sample efficiency?
5. **Out-of-Domain Generalization:** How sharply does performance drop when evaluating fine-tuned models on a completely unseen, custom database schema?
6. **Inference Pareto Frontier:** What is the optimal tradeoff between execution accuracy (EX), p95 latency, and tokens per second (TPS)?

---

## 2. Current Status: Step 0 Foundation Complete

> [!NOTE]
> **Step 0 Implementation Status:** Production-quality repository foundation, typed data contracts, layered YAML configuration, CLI developer tooling, local-first experiment tracking, defensive SQL execution boundaries, and test suites are complete. Model training experiments and dataset ingestion pipelines begin in Step 1.

---

## 3. Repository Architecture

```text
SQLForge/
├── .github/workflows/ci.yml       # GitHub Actions automated CI matrix
├── .editorconfig                  # Consistent cross-editor whitespace & indentation
├── .env.example                   # Template environment variables (no secrets)
├── .gitattributes                 # Line-ending normalization and binary declarations
├── .gitignore                     # Rigorous exclusion of caches, checkpoints, datasets
├── .pre-commit-config.yaml        # Automated git hook quality enforcement
├── CHANGELOG.md                   # Semantic versioning release log
├── CONTRIBUTING.md                # Development standards and commit conventions
├── LICENSE                        # Apache 2.0 open-source license
├── Makefile                       # Developer shortcuts for Linux/macOS
├── pyproject.toml                 # Standard packaging config with modular extras
├── README.md                      # Project overview and quickstart guide
├── SECURITY.md                    # Untrusted SQL execution policies & vulnerability reporting
├── configs/                       # Layered, typed YAML experiment configs
│   ├── defaults.yaml              # System settings, logging, sandbox boundaries
│   ├── datasets.yaml              # Spider, BIRD, custom schema specifications
│   ├── models.yaml                # Model architectures, LoRA target modules
│   ├── experiments.yaml           # Hypothesis test matrix and parameter sweeps
│   └── evaluation.yaml            # Sandbox limits, metric toggles, bootstrap specs
├── docs/                          # Comprehensive research documentation
│   ├── architecture.md            # System architecture with Mermaid diagram
│   ├── research_plan.md           # Formal hypotheses, experimental factors, metrics
│   ├── experiment_protocol.md     # Run naming, config snapshots, seed protocols
│   ├── data_governance.md         # Benchmark licenses, split integrity, privacy
│   ├── development_setup.md       # Setup instructions and troubleshooting guide
│   ├── glossary.md                # Domain dictionary defining 20+ core concepts
│   └── decisions/                 # Architecture Decision Records (ADRs)
│       ├── ADR-001-repository-layout-and-packaging.md
│       ├── ADR-002-typed-configuration-system.md
│       ├── ADR-003-sql-execution-safety-boundaries.md
│       └── ADR-004-reproducibility-and-local-first-tracking.md
├── src/sqlforge/                  # Core Python package
│   ├── __init__.py                # Package version definition (v0.1.0)
│   ├── cli.py                     # Rich CLI with env, config, and experiment commands
│   ├── settings.py                # Pydantic v2 settings loading and validation
│   ├── logging_config.py          # Structured console & file logging
│   ├── reproducibility.py         # Seed utilities, environment & git auditing
│   ├── schemas/                   # Foundational typed Pydantic contracts
│   │   ├── metadata.py            # ColumnMetadata, SchemaMetadata, TableMetadata
│   │   ├── examples.py            # TextToSQLExample, SyntheticProvenance, Splits
│   │   ├── models.py              # ModelConfig, LoRAHyperparameters, QLoRA
│   │   ├── execution.py           # ExecutionResult, ExecutionStatus
│   │   ├── evaluation.py          # EvaluationMetrics, ConfidenceInterval
│   │   └── experiments.py         # ExperimentConfig, RunMetadata, HardwareMetadata
│   ├── experiments/               # Local-first experiment tracking
│   │   └── tracker.py             # ExperimentTracker creating artifacts/runs/
│   ├── data/                      # Planned: Ingestion & leakage checkers
│   ├── models/                    # Planned: Local & API model runners
│   ├── prompting/                 # Planned: Schema serializers & few-shot retrievers
│   ├── training/                  # Planned: LoRA/QLoRA fine-tuning engines
│   ├── execution/                 # Planned: Hardened read-only execution sandboxes
│   ├── evaluation/                # Planned: Execution accuracy & error taxonomy
│   ├── optimization/              # Planned: AWQ/GGUF quantization & latency profiler
│   ├── serving/                   # Planned: Lightweight vLLM/FastAPI server
│   └── utils/                     # Environment diagnostics and system helpers
├── tests/                         # Comprehensive test suite (unit & integration)
│   ├── conftest.py                # Reusable fixtures for schemas and CLI
│   ├── unit/                      # Fast, isolated unit tests
│   └── integration/               # End-to-end CLI workflow tests
├── scripts/                       # Setup and automated quality check scripts
│   ├── run_checks.ps1             # PowerShell quality checks (Windows)
│   ├── setup_dev.ps1              # PowerShell virtual environment setup (Windows)
│   └── run_checks.sh              # Bash quality checks (Linux/macOS)
├── data/                          # Data directory (raw, interim, processed)
├── artifacts/                     # Local experiment runs and adapters
├── reports/                       # Generated result tables and figures
└── notebooks/                     # Exploratory analysis notebooks
```

---

## 4. Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL.git
cd SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### System Diagnostics

Inspect host resources, CPU cores, RAM, GPU/CUDA status, and Git metadata:
```bash
sqlforge env
```

### Configuration Validation

Validate all YAML experiment definitions against typed Pydantic schemas:
```bash
sqlforge config validate
```

### Initialize a Local Experiment Run

Create a traceable experiment directory with frozen configuration and hardware metadata:
```bash
sqlforge experiment init --name pilot_run --seed 42
```

---

## 5. Development Quality Checks

SQLForge adheres to strict code quality and typing standards:

```powershell
# Windows (PowerShell automated script)
.\scripts\run_checks.ps1

# Linux / macOS (Makefile)
make check
```

Or run individual tools directly:
```bash
ruff check .           # Linting
ruff format --check .  # Code formatting
mypy src/sqlforge      # Static type checking
pytest -v tests/       # Unit & integration tests
```

---

## 6. Defensive SQL Execution Architecture

Generated SQL is untrusted code. SQLForge enforces strict multi-layer boundaries:
* **Connection-Level Read-Only Mode:** SQLite connections use `file:...?mode=ro` preventing any schema or row mutations.
* **Single-Statement Rule:** Rejects multi-statement stacked queries before execution.
* **Wall-Clock Timeouts:** Enforces a 10.0-second query execution deadline.
* **Resource Ceiling:** Memory capped at 1024 MB and returned result sets limited to 5000 rows.
* **Zero Production Access:** Benchmarks execute exclusively against isolated, ephemeral sandbox databases.

---

## 7. Research Roadmap

- [x] **Step 0: Project Foundation, Repository & Architecture**
- [x] **Step 1: Research Specification & Experimental Design** ([docs/research/](docs/research/))
- [ ] **Step 2: Repository Implementation Review & Core Pipeline Harness**
- [ ] **Step 3: Dataset Ingestion & Contamination Audit (Spider, BIRD, Custom Held-Out)**
- [ ] **Step 4: Schema Representation & Few-Shot RAG Pipeline**
- [ ] **Step 5: Frontier API Reference & Zero-Shot Baselines**
- [ ] **Step 6: Supervised Fine-Tuning Setup (LoRA vs QLoRA)**
- [ ] **Step 7: LoRA Hyperparameter & Rank Scaling Sweeps**
- [ ] **Step 8: Training Data Scaling & Synthetic vs. Human Data**
- [ ] **Step 9: Hardened Database Execution Engine & Normalizer**
- [ ] **Step 10: Quantitative Evaluation & Statistical Bootstrap CIs**
- [ ] **Step 11: Quantization & Serving Latency Profiling (vLLM vs HF)**
- [ ] **Step 12: Error Taxonomy Analysis & Final Synthesis**

---

## 8. License

SQLForge is licensed under the [Apache License 2.0](LICENSE).
