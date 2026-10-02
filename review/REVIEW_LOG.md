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
