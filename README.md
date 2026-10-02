# SQLForge: A Controlled Study of What Fine-Tuning Buys on Text-to-SQL

[![CI](https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL/actions/workflows/ci.yml/badge.svg)](https://github.com/mayanksingh2745/SQLForge-a-controlled-study-of-what-fine-tuning-buys-on-text-to-SQL/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-informational.svg)](https://mypy-lang.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source empirical research study investigating whether parameter-efficient fine-tuning (LoRA / QLoRA) on small open-weight language models (1.5B–8B parameters) can approach or match commercial frontier reference APIs on text-to-SQL execution accuracy while reducing inference latency, VRAM footprint, and operational costs.

---

## 1. Research Objectives

Commercial frontier models (e.g., GPT-4o, Claude 3.5 Sonnet) achieve strong zero-shot and few-shot execution accuracy on benchmarks like Spider and BIRD. However, their opacity, high tail latencies, cost per million tokens, and cloud-data privacy concerns prevent deployment in privacy-sensitive or air-gapped database environments.

**SQLForge** investigates the following core scientific questions:

1. **In-Domain Execution Accuracy:** Can a fine-tuned 7B open-weight model (e.g., Qwen2.5-Coder-7B) demonstrate non-inferior execution accuracy to zero-shot GPT-4o mini on Spider?
2. **LoRA vs QLoRA Efficiency:** Does 4-bit NormalFloat (NF4) quantization degrade text-to-SQL execution accuracy relative to 16-bit LoRA, and how much training VRAM does it save?
3. **LoRA Rank Scaling:** Where does execution accuracy saturate as LoRA rank $r$ scales ($r \in \{8, 16, 32, 64\}$)?
4. **Data Scaling Laws:** How does accuracy scale across nested sample fractions ($10\%, 25\%, 50\%, 100\%$), and does execution-filtered synthetic training data match human sample efficiency?
5. **Out-of-Domain Generalization:** How sharply does performance drop when evaluating fine-tuned models on a completely unseen, custom database schema?
6. **Inference Pareto Frontier:** What is the optimal tradeoff between execution accuracy (EX), p95 latency, and throughput (TPS / QPS)?

---

## 2. Current Implementation Status & Functional Boundaries

The project has completed **Step 3: Dataset Ingestion & Contamination Audit** (Steps 0, 1, 2, and 3 completed; Step 4 schema representation and few-shot RAG pipeline is next). Functionality is strictly categorized as follows:

### Implemented and Tested
* **Architecture & Packaging:** Clean layout (`src/sqlforge/`, `configs/`, `docs/`, `tests/`), `pyproject.toml` packaging, and 6-job GitHub Actions CI testing Python 3.11, 3.12, 3.13 on Ubuntu and Windows.
* **Typed Data Contracts:** Pydantic models in `src/sqlforge/schemas/` for datasets, models, training parameters, evaluation metrics (SVR, ESR, EX, EM with alias reconciliation and CI bounds), execution sandboxes, and run manifests.
* **Layered YAML Configuration:** Centralized config hierarchy (`configs/defaults`, `datasets`, `models`, `experiments`, `mock_pipeline`) validated via `sqlforge config validate`.
* **Hardened Local Experiment Tracking:** `ExperimentTracker` (`src/sqlforge/experiments/tracker.py`) featuring conservative run ID validation (regex, length, path containment), atomic directory reservation (`mkdir` collision safety), non-destructive anomaly logging with error handling, strict cryptographic run verification (`manifest.json` self-exclusion, untracked file detection, record parsing), and frozen config snapshots.
* **Deterministic Core Pipeline Harness (Mock Vertical Slice):**
  - Typed fixture dataset loader with duplicate detection (`src/sqlforge/data/fixtures.py`, `tests/fixtures/dataset/mock_spider.json`).
  - Independent prompt builder (`src/sqlforge/prompting/builder.py`).
  - Offline mock model runner with explicit synthetic disclaimers (`src/sqlforge/models/mock.py`).
  - Mock evaluator verifying contract serialization without claiming benchmark accuracy (`src/sqlforge/evaluation/mock_eval.py`).
  - End-to-end pipeline harness orchestrator (`src/sqlforge/pipeline/harness.py`).
  - Executable CLI command: `sqlforge pipeline mock` (with `--dry-run`, `--config`, `--fixtures`, `--artifact-dir`, `--fail-mode`).
* **Dataset Ingestion Adapters & Provenance Tracking:**
  - Typed provenance contracts (`DatasetProvenance`, `DatasetSplitManifest`, `DatasetManifest`, `ContaminationReport`) recording upstream source, version tag, revision, licensing terms (`CC-BY-SA-4.0`, `CC-BY-NC-SA-4.0`, `Apache-2.0`), normalization version, and SHA-256 file checksums.
  - Spider 1.0 schema extractor and partition loader (`tables.json`, `train_spider.json`, `dev.json`) validating required fields, foreign key relationships, and foreign database references.
  - BIRD Mini-Dev ingestion adapter resolving upstream ambiguity between the 500 SELECT-only subset (`mini_dev_500`, pinned canonical) and 780 Mini-Dev V2 release (`mini_dev_780`), preserving evidence, questions, SQL, and CC BY-NC-SA 4.0 license restrictions.
  - Custom Held-Out Benchmark foundation (`subscription_analytics_db` 6-table relational schema: `customers`, `plans`, `subscriptions`, `invoices`, `transactions`, `support_tickets`) strictly isolated under `DatasetSplit.HELD_OUT` with Apache-2.0 provenance.
* **Contamination Auditing & Runtime Isolation Guards:**
  - `ContaminationAuditor` detecting normalized exact question collisions, exact SQL collisions, configurable word $n$-gram Jaccard lexical overlap, schema disjointness violations ($D_{\text{train}} \cap D_{\text{eval}} = \emptyset$), and split leakage without silently altering official benchmark dev/test splits.
  - `IsolationGuard` asserting that training datasets and retrieval demonstration indexes contain strictly training instances with zero evaluation data or quarantined evaluation schemas.
* **Schema Representation & Few-Shot RAG Pipeline:**
  - **DDL Serializer:** Generates standard SQLite `CREATE TABLE` statements with primary keys, single and composite foreign keys, identifier escaping, deterministic ordering, and optional sample rows and inline comments.
  - **Compact Pipe Serializer:** Formats schemas into token-efficient pipe-delimited strings (`table : col (TYPE, PK) | col (TYPE, FK)`).
  - **JSON Schema Serializer:** Formats structured, machine-readable JSON schema definitions.
  - **Training-Only BM25 Retriever:** In-memory Okapi BM25 retriever indexed strictly over training-partition questions, enforcing `IsolationGuard.assert_retrieval_isolation()` to reject evaluation and held-out instances, with deterministic tie-breaking and traceable `DemonstrationRecord` audit logs.
  - **Prompt Assembly Engine:** `PromptEngine` supporting zero-shot and few-shot configurations ($k \in \{0, 1, 3, 5\}$), BIRD external domain evidence integration, and graceful context budget enforcement (pruning demonstrations and DDL comments before raising `PromptBudgetExceededError`).
* **Developer Tooling & CLI:** `sqlforge env` diagnostics, `sqlforge config validate`, `sqlforge experiment init`, `sqlforge pipeline mock`, `sqlforge data validate`, `sqlforge data audit`, `sqlforge data manifest`, `sqlforge prompt serialize`, and `sqlforge prompt assemble`.
* **Reproducibility Foundation:** Seed management (`set_seed`), platform auditing, Git working tree dirty-status verification, `requirements-constraints.txt`, and 100% offline test fixtures (`tests/fixtures/dataset/`).
* **Automated Test Suite:** 117 automated unit and integration tests passing 100% locally and in CI across Ubuntu and Windows.

### Specified but Not Yet Implemented
* **Research Specifications:** Hypotheses with TOST equivalence margins, baseline tiers B0–B3/T1–T3, dataset governance protocols, and 10-experiment staged matrix in [`docs/research/`](docs/research/).
* **Frontier API Reference & Zero-Shot Baselines:** Standardized reference model evaluation and zero-shot baseline execution (scheduled for Step 5).
* **Database Execution Engine:** Read-only SQLite connection sandboxing (`mode=ro`), 10.0s query timeout watchdogs, and memory limits (scheduled for Step 9).
* **Model Training & Evaluation Harness:** Real LoRA/QLoRA trainer (Step 6) and multiset execution comparator (Step 9/10).
* **Quantization & Serving Benchmarks:** AWQ/GGUF exports and vLLM high-concurrency benchmarks (Step 11).

### Planned
* Execution of experiments EXP-01 through EXP-10, empirical scaling law parameterization, error taxonomy failure mode audit, and publication synthesis.

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
│   ├── data/                      # Ingestion adapters, contamination auditor, isolation guards, manifests

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

### Dataset Ingestion, Validation & Contamination Audit

Validate schemas, ingest benchmark partitions, and audit cross-partition leakage:
```bash
# Validate Spider, BIRD Mini-Dev, and Custom Held-Out contracts
sqlforge data validate

# Audit cross-partition duplicates, fuzzy n-gram lexical overlap, and schema disjointness
sqlforge data audit --threshold 0.85 --output-report reports/contamination_audit.json

# Verify cryptographic SHA-256 dataset manifests
sqlforge data manifest --verify data/processed/dataset_manifest.json
```

### Run the End-to-End Mock Pipeline Harness

Execute the deterministic, offline mock pipeline vertical slice to verify configuration, data contracts, prompt construction, mock inference, mock evaluation, and cryptographic run verification:

```bash
# Dry-run validation (checks config and fixtures without creating run artifacts)
sqlforge pipeline mock --dry-run

# Full mock pipeline execution and manifest verification
sqlforge pipeline mock

# Custom config, fixtures, and artifact directory
sqlforge pipeline mock --config configs/mock_pipeline.yaml --artifact-dir artifacts/runs
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

> [!NOTE]
> **Design Specification:** The defensive execution boundaries below are codified as policies in [`SECURITY.md`](SECURITY.md), [`ADR-001`](docs/decisions/ADR-001-sqlite-sandboxing.md), and [`configs/evaluation.yaml`](configs/evaluation.yaml). Their software implementation in Python will be developed and tested in **Step 9: Hardened Database Execution Engine & Normalizer**.

Generated SQL is untrusted code. SQLForge establishes strict multi-layer boundaries:
* **Connection-Level Read-Only Mode:** SQLite connections use `file:...?mode=ro` preventing any schema or row mutations.
* **Single-Statement Rule:** Rejects multi-statement stacked queries before execution.
* **Wall-Clock Timeouts:** Enforces a 10.0-second query execution deadline.
* **Resource Ceiling:** Memory capped at 1024 MB and returned result sets limited to 5000 rows.
* **Zero Production Access:** Benchmarks execute exclusively against isolated, ephemeral sandbox databases.


---

## 7. Research Roadmap

- [x] **Step 0: Project Foundation, Repository & Architecture**
- [x] **Step 1: Research Specification & Experimental Design** ([docs/research/](docs/research/))
- [x] **Step 2: Repository Implementation Review & Core Pipeline Harness**
- [x] **Step 3: Dataset Ingestion & Contamination Audit (Spider, BIRD, Custom Held-Out)**
- [x] **Step 4: Schema Representation & Few-Shot RAG Pipeline**
- [ ] **Step 5: Frontier API Reference & Zero-Shot Baselines (Next Step)**
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
