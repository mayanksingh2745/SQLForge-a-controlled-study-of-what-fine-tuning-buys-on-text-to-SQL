# SQLForge Independent Review Log

## 1. Overview & Purpose
This log documents the independent reviews, structural integrity audits, and gate evaluations conducted across the SQLForge research lifecycle.

---

## 2. Chronological Review Entries

### Entry 001: Pre-Step-2 Structural Integrity Gate Audit
* **Date:** 2026-10-02
* **Auditor:** Senior ML Research Engineer & Repository Maintainer
* **Base Commit:** `ade7795` (Merge of Step 1 PR #2)
* **Branch:** `fix/structural-integrity-pre-step-2`
* **Objective:** Audit repository structure, research specifications, configuration contracts, and metric definitions prior to beginning Step 2 implementation.
* **Key Findings:**
  1. *Implementation vs. Specification Conflation:* README and documentation previously claimed "research-grade empirical framework" and implied SQL sandboxing and evaluation pipelines were active, when they were only specified.
  2. *Artifact Retention Conflict:* Permanent retention requirement conflicted with gitignoring `artifacts/runs/*`. Resolved via explicit storage tiering (local/external archival for raw runs, Git for code/configs/fixtures/reports).
  3. *Metric Ambiguities:* Valid-SQL Rate (VSR) ambiguously mixed AST syntax parsing with SQLite database execution. Resolved by defining Syntax Validity Rate (SVR), Execution Success Rate (ESR), and Execution Accuracy (EX) with strict denominator $N$.
  4. *Equivalence & Parity Ambiguity:* Hypotheses previously conflated failing to reject the null hypothesis of difference ($p \ge 0.05$) with demonstrating equivalence. Resolved by formalizing the Two One-Sided Tests (TOST) framework with practical equivalence bounds.
  5. *Matrix & Config Mismatch:* `configs/experiments.yaml` omitted experiments `exp07` through `exp10` and lacked explicit BM25 vs. dense retrieval arms in `exp02`.
* **Verdict:** Remediations applied and verified. Ready for Step 2 pipeline harness.

### Entry 002: Step 2 Foundation Hardening & Core Pipeline Harness
* **Date:** 2026-10-02
* **Auditor:** Senior ML Research Engineer & Skeptical Reviewer
* **Base Commit:** `a3938cd` (Merge of Structural Remediation PR #3)
* **Branch:** `step-2/core-pipeline-harness`
* **Objective:** Audit and harden foundation tracker and metric contracts; implement deterministic end-to-end mock pipeline harness.
* **Key Findings & Verified Remediations:**
  1. *Run ID & Containment Hardening:* Tracker now strictly validates run IDs (`^[a-zA-Z0-9_\-]+$`, length $\le 128$), prevents path traversal (`..`, slashes, absolute paths), and enforces containment within the configured artifact directory.
  2. *Atomic Reservation:* Directory reservation now uses atomic `mkdir(parents=False, exist_ok=False)` to prevent silent reuse or overwriting of both populated and empty directories.
  3. *Non-Destructive Anomaly Logging:* `log_anomaly` parses existing logs and preserves corrupted or structural mismatches without silent truncation or overwrites, raising actionable `ValueError`.
  4. *Strict Manifest & Record Syntax Verification:* `verify_run` enforces `manifest.json` presence in strict mode, checks SHA-256 self-exclusion, flags untracked files, and validates syntax of `run_metadata.json`, `config.yaml`, `metrics.json`, and `generations.jsonl`.
  5. *Metric Alias & Defaults Consistency:* `valid_sql_rate` and `execution_success_rate` are synchronized bidirectionally with conflict rejection. Uncomputed metrics default to `None` instead of misleading `0.0` or `1.0`. `ConfidenceInterval` enforces `lower <= upper`.
  6. *Deterministic Mock Pipeline Vertical Slice:* Implemented validated configuration loading, deterministic fixtures (`mock_spider.json`), prompt builder, mock model (with clear synthetic tagging), mock evaluator, experiment tracking, and CLI command `sqlforge pipeline mock`.
  7. *Strict Scope Discipline:* Zero real model downloads, zero frontier API requests, zero real dataset ingestion, and zero empirical claims made. All mock metrics visibly labeled synthetic.
* **Verdict:** PASS. All 60 automated unit and integration tests passing. Clean linting and type checking.

