# SQLForge Step 2.1 Readiness Scorecard

## Audit Metadata
* **Gate:** Step 2.1 — Manifest Verification & Path Containment Corrective Review Gate
* **Date:** 2026-10-02
* **Evaluator:** Senior ML Research Engineer & Skeptical Reviewer
* **Base Commit:** `4b9912b` (Merge of PR #4)
* **Branch:** `fix/manifest-verification-boundaries`

---

## Category Evaluations

| Evaluation Category | Status | Confidence Score | Notes & Evidence |
| :--- | :---: | :---: | :--- |
| **1. Repository & Package Architecture** | **VERIFIED** | 10/10 | Clean `src/sqlforge/` layout, `pyproject.toml`, 6-job CI matrix on Ubuntu/Windows. |
| **2. Configuration Contracts & Validation** | **VERIFIED** | 10/10 | Complete 10-experiment matrix in `configs/experiments.yaml`, validated via `sqlforge config validate`. |
| **3. Documentation Honesty & Boundaries** | **VERIFIED** | 10/10 | Strict separation of implemented foundation vs. future roadmap steps. All mock outputs tagged synthetic. |
| **4. Artifact Governance & Tracker Hardening** | **VERIFIED** | 10/10 | Manifest pre-resolution path validation (no traversal/absolute/drive paths), symlink safety, 64-hex SHA-256 regex, atomic directory reservation, non-destructive anomaly logging. |
| **5. Evaluation Metric Rigor & Schemas** | **VERIFIED** | 10/10 | Metric alias reconciliation (`valid_sql_rate` $\equiv$ `execution_success_rate`), `ConfidenceInterval` bounds, `None` for uncomputed metrics, full Pydantic `EvaluationMetrics` verification in tracker. |
| **6. Experimental & Statistical Design** | **VERIFIED (Spec)** | 9.5/10 | TOST equivalence framework, nested scaling subsets, synthetic provenance controls, sample limit warnings. |
| **7. Reproducibility & Dependency Policy** | **VERIFIED** | 9/10 | Pinned constraints file, Python 3.11–3.13 documented, GPU determinism limits acknowledged. |
| **8. Automated Test Suite** | **VERIFIED** | 10/10 | 64 automated unit and integration tests passing in pytest (0 linter or type errors). |
| **9. Core Pipeline Harness (Mock Slice)** | **VERIFIED** | 10/10 | Validated config → deterministic fixtures → prompt builder → mock model → mock evaluator → tracker → strict verification. `sqlforge pipeline mock` tested offline. |
| **10. SQL Sandboxing & Execution Engine** | **UNVERIFIED** | N/A | *Not implemented in this step (Scheduled for Step 9).* |
| **11. Fine-Tuning & Model Training** | **UNVERIFIED** | N/A | *Not implemented in this step (Scheduled for Step 6).* |
| **12. Quantization & Serving Benchmarks** | **UNVERIFIED** | N/A | *Not implemented in this step (Scheduled for Step 11).* |

---

## Gate Summary & Verdict

### Step 2.1 Scope Completed & Verified:
- [x] Manifest pre-resolution path containment rejecting `..`, absolute paths, drive roots, and UNC paths before filesystem access.
- [x] SHA-256 hash regex validation (`^[a-fA-F0-9]{64}$`) rejecting malformed or non-hex checksums.
- [x] Symlink safety check preventing links resolving outside the run directory.
- [x] Full artifact schema verification validating `metrics.json` via Pydantic `EvaluationMetrics` and `eval_anomalies.json` via list-of-dicts structure.
- [x] Lifecycle requirements defined: `metrics.json` required for completed runs, optional for failed runs with failure diagnostics preserved.
- [x] Documented precise verification guarantees vs. what cryptographic hashes do not guarantee.
- [x] Updated `docs/research/step_dependencies.md` reflecting completed Steps 0, 1, and 2.
- [x] 64 automated unit and integration tests passing (0 failures).

### Scope Boundary Confirmation:
**Confirmed:** Zero real model weights downloaded, zero frontier API requests made, zero real benchmark datasets (Spider/BIRD) ingested, zero SQLite execution sandbox logic implemented, and zero empirical performance claims made.

### STEP 2.1 GATE VERDICT: PASS
Manifest path containment and complete artifact verification are hardened, fully tested with regression coverage, and verified. Ready to proceed to Step 3.


