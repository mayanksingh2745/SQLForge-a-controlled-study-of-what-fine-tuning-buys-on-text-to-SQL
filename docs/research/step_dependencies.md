# SQLForge Step Dependencies & Execution Roadmap

This document formalizes the dependency graph, phase progression, exit criteria, and research risk mitigations across the 12-step SQLForge roadmap.

---

## 1. Roadmap Alignment & Canonical Progression

To harmonize the initial Step 0 conceptual outline with the rigorous research specifications established in Step 1, the canonical 12-step sequence is mapped as follows:

```mermaid
graph TD
    S0["Step 0: Project Foundation & Architecture"] --> S1["Step 1: Research Specification & Experimental Design"]
    S1 --> S2["Step 2: Repository Implementation Review & Core Pipeline Harness"]
    S2 --> S3["Step 3: Dataset Ingestion & Contamination Audit"]
    S3 --> S4["Step 4: Schema Representation & Few-Shot RAG Baseline"]
    S4 --> S5["Step 5: Frontier API Reference & Zero-Shot Baselines"]
    S5 --> S6["Step 6: Supervised Fine-Tuning Setup (LoRA vs QLoRA)"]
    S6 --> S7["Step 7: LoRA Hyperparameter & Rank Scaling Sweeps"]
    S7 --> S8["Step 8: Training Data Scaling & Synthetic vs. Human Data"]
    S8 --> S9["Step 9: Hardened Database Execution Engine & Normalizer"]
    S9 --> S10["Step 10: Quantitative Evaluation & Statistical Bootstrap CIs"]
    S10 --> S11["Step 11: Quantization & Serving Latency Profiling"]
    S11 --> S12["Step 12: Error Taxonomy Analysis & Final Synthesis"]

    style S0 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S1 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S2 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S3 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S4 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S5 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S6 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S7 fill:#2d6a4f,stroke:#1b4332,color:#ffffff
    style S8 fill:#1d3557,stroke:#457b9d,color:#ffffff
```

---

## 2. Step-by-Step Dependencies, Deliverables & Exit Criteria

### Step 0: Project Foundation & Architecture
* **Status:** Complete (Merged via PR #1).
* **Deliverables:** Package layout (`src/sqlforge/`), typed schemas, YAML configs, CLI tooling, PR/issue templates, ADRs 001–004.
* **Exit Criteria:** Package installable, CLI functioning, 26 unit/integration tests passing in CI.

### Step 1: Research Specification & Experimental Design
* **Status:** Complete (Merged via PR #2; remediated via PR #3).
* **Prerequisites:** Step 0 repository foundation.
* **Deliverables:** `docs/research/` specification suite (hypotheses with TOST equivalence bounds, baseline protocol, dataset governance protocol, complete 10-experiment matrix, metrics & statistics, error taxonomy, compute plan, experiment record spec, step dependencies).
* **Exit Criteria:** All 9 research documents complete, internally consistent, peer-reviewed, and merged via PR.

### Step 2: Repository Implementation Review & Core Pipeline Harness
* **Status:** Complete (Merged via PR #4; hardened via Step 2.1).
* **Prerequisites:** Step 1 research specification.
* **Deliverables:** Hardened `ExperimentTracker` with path traversal defense, atomic directory reservation, non-destructive anomaly logging, and strict cryptographic verification (`manifest.json`); deterministic fixture dataset (`mock_spider.json`); independent prompt builder; mock model runner; mock evaluator; pipeline orchestrator harness; and CLI subcommand `sqlforge pipeline mock`.
* **Exit Criteria:** Dry-run and complete offline end-to-end execution of mock pipeline vertical slice; 64 unit/integration tests passing in CI across Ubuntu/Windows.

### Step 3: Dataset Ingestion & Contamination Audit
* **Status:** Complete (PR #6).
* **Prerequisites:** Step 2 pipeline harness.
* **Deliverables:** Spider 1.0 adapter, BIRD Mini-Dev adapter with ambiguity resolution (500 SELECT-only vs. 780 Mini-Dev V2), custom held-out schema foundation (`subscription_analytics_db`), cross-partition contamination auditor (`ContaminationAuditor`), runtime isolation guards (`IsolationGuard`), deterministic JSONL serialization and manifest generator, and CLI tooling (`sqlforge data validate`, `sqlforge data audit`, `sqlforge data manifest`).
* **Exit Criteria:** 90 unit/integration tests passing in CI; verified zero leakage on clean partitions; machine-readable contamination audit reports.

### Step 4: Schema Representation & Few-Shot RAG Pipeline
* **Status:** Complete (PR #8).
* **Prerequisites:** Step 3 ingested datasets.
* **Deliverables:** DDL, compact pipe, and JSON schema serializers; training-only BM25 retriever (`BM25Retriever`) with `IsolationGuard.assert_retrieval_isolation()` quarantine enforcement; prompt formatting engine (`PromptEngine`) with $k \in \{0, 1, 3, 5\}$, BIRD evidence integration, and graceful token budget degradation; CLI subcommands (`sqlforge prompt serialize`, `sqlforge prompt assemble`).
* **Exit Criteria:** 117 unit/integration tests passing in CI; formatted prompts strictly adhere to context limits; retrieval index strictly isolated to training partition.

### Step 5: Frontier API Reference & Zero-Shot Baselines
* **Status:** Complete.
* **Prerequisites:** Step 4 prompting engine.
* **Deliverables:** Typed model-runner interfaces for local causal LMs (`LocalHFModelRunner`) with lazy-loaded dependencies and commercial frontier models (`OpenAIRunner`) with rate-limiting, bounded exponential backoff, hard cumulative budget ceiling (<= $50.00 USD), secret redaction, and token-based cost accounting; SQL sandboxing with read-only SQLite connections, query timeout watchdogs, multiset row comparator, and ORDER BY sequence sensitivity (`ExecutionComparator`); statistical bootstrap engine (`compute_bootstrap_ci`, `compute_paired_difference_ci`, `mcnemar_test`); baseline orchestration pipeline harness (`BaselinePipelineHarness`) supporting EXP-01 zero-shot and three-shot BM25 baselines; CLI workflow (`sqlforge baseline run`, `sqlforge baseline verify`, `sqlforge baseline stats`).
* **Exit Criteria:** 166 unit/integration tests passing in CI; verified zero leakage on retrieval index; dry-run and offline evaluation pipelines verified with cryptographic manifests.

### Step 6: Supervised Fine-Tuning Setup (LoRA vs. QLoRA)
* **Status:** Complete.
* **Prerequisites:** Step 5 baselines.
* **Deliverables:** Modular SFT fine-tuning pipeline (`src/sqlforge/training/`) supporting 16-bit LoRA and 4-bit NF4 QLoRA for text-to-SQL tasks (`EXP-03`); hardware-aware preflight safety gates (`PreflightChecker`) auditing CUDA VRAM, host RAM, free disk space ($\ge 3.0$ GB), and dependency availability; split-isolated dataset formatter (`SFTDatasetFormatter`) with strict `DatasetSplit.TRAIN` quarantine; completion-only loss masking engine (`CompletionLossMasker`) with `-100` label masking and severe target truncation protection; PEFT and 4-bit quantization config factory (`PEFTConfigFactory`) with architecture-aware target module defaults and non-fallback quantization error guards; atomic checkpoint manager (`CheckpointManager`) with JSON metadata tracking; evaluation handoff bridge to `LocalHFModelRunner`; and CLI suite (`sqlforge train preflight`, `validate`, `run`, `inspect`).
* **Exit Criteria:** 209 unit/integration tests passing in CI across Ubuntu/Windows; loss curves, step progression, and VRAM monitored; checkpoints saved cleanly with JSON metadata and without base weight duplication.

### Step 7: LoRA Hyperparameter & Rank Scaling Sweeps
* **Status:** Infrastructure Complete (Empirical Runs Pending Dedicated GPU & Dataset Availability).
* **Prerequisites:** Step 6 fine-tuning setup.
* **Deliverables:** Typed sweep configuration system (`RankSweepConfig`), deterministic condition generator (`exp04_r{r}_a{alpha}_{tag}_s{seed}`), sweep orchestrator (`LoRARankSweepOrchestrator`) with interrupted-run recovery and collision defense, rank saturation Pareto analysis engine (`RankSaturationPlotter`), anti-fabrication plotting policy (`NoEmpiricalDataError`), publication-quality SVG generator and CSV exporter, and CLI suite (`sqlforge sweep plan`, `validate`, `run`, `status`, `plot`).
* **Exit Criteria (Infrastructure):** 230 unit/integration tests passing in CI across Ubuntu/Windows; dry-run and mock sweep orchestration verified; Pareto curve generation validated on genuine evaluation fixtures; strict refusal to plot from unevaluated or fabricated numbers.
* **Exit Criteria (Empirical Experiment):** Requires dedicated GPU instance ($\ge 16\text{ GB}$ VRAM, ~8 GPU hours) and full Spider train/dev dataset; rank saturation Pareto curve plotted with 95% bootstrap CIs; optimal rank identified.

### Step 8: Training Data Scaling & Synthetic vs. Human Data (Next Step)
* **Prerequisites:** Step 7 sweep infrastructure and optimal rank configuration.
* **Deliverables:** Sample scaling runs ($N \in \{500, 1000, 2000, 5000, 7000\}$) across 3 seeds (`EXP-05`); synthetic data comparison (`EXP-06`).
* **Exit Criteria:** Empirical scaling laws fitted; sample efficiency comparison documented.

### Step 9: Hardened Database Execution Engine & Normalizer
* **Prerequisites:** Step 3 databases.
* **Deliverables:** Read-only SQLite sandboxing engine, query timeout monitors, multiset result set normalizer, and cryptographic result fingerprinter.
* **Exit Criteria:** Safe execution of arbitrary generated SQL with zero data mutation and 100% timeout enforcement.

### Step 10: Quantitative Evaluation & Statistical Bootstrap CIs
* **Prerequisites:** Steps 6–9.
* **Deliverables:** Full execution accuracy evaluation across all model checkpoints, McNemar paired tests, and 95% bootstrap confidence intervals.
* **Exit Criteria:** Comprehensive evaluation tables partitioned by difficulty and schema domain.

### Step 11: Quantization & Serving Latency Profiling
* **Prerequisites:** Step 10 evaluated models.
* **Deliverables:** 4-bit post-training quantization (AWQ, GGUF); vLLM serving benchmark across concurrency levels 1, 8, 32 (`EXP-08`, `EXP-10`).
* **Exit Criteria:** Throughput (TPS/QPS) vs. p95 latency curves generated under realistic client load.

### Step 12: Error Taxonomy Analysis & Final Synthesis
* **Prerequisites:** Steps 10–11.
* **Deliverables:** 10-category error classification breakdown, qualitative analysis of failure modes, final comparative research paper/report, and public model card.
* **Exit Criteria:** Complete research narrative finalized; reproducible open-source release bundle ready.

---

## 3. Major Research Risks & Mitigations

| Risk Description | Probability | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **GPU Memory Saturation (CUDA OOM)** | Medium | High | Pre-flight VRAM feasibility gate; fallback to 4-bit QLoRA and gradient checkpointing; pilot on 1.5B model. |
| **Silent Benchmark Leakage** | Low | Critical | Automated n-gram overlap check, strict retrieval-index isolation, and verified schema disjointness. |
| **API Cost Overruns** | Medium | Medium | Automated $50 hard spending ceiling with token accounting on all commercial API calls. |
| **False-Positive Result Equivalence** | Medium | Medium | Multiset row matching, float tolerance ($\epsilon = 10^{-4}$), and empty result set validation guards. |
| **Windows Platform Dependency Issues** | Low | Medium | Strict separation of OS-agnostic evaluation contracts from Linux-native vLLM serving benchmarks. |
