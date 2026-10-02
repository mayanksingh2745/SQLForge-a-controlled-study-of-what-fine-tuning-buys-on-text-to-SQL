# SQLForge Pre-Step-2 Audit Issues Registry

This registry tracks the status of all structural and research-design issues identified during the Pre-Step-2 Gate Review.
Per repository rules: **Structural documentation fixes are strictly classified as documentation remedies and not counted as implementation fixes.** Unverified components remain explicitly marked **UNVERIFIED**.

---

## Issue Catalog

### Issue A: Conflation of Implemented vs. Planned Functionality
* **Category:** Repository Transparency / Documentation
* **Status:** **FIXED** (Documentation Remedy)
* **Locations:**
  - `README.md` (Lines 9, 28–70, 196–205)
* **Problem:** Present-tense claims such as "enforces strict multi-layer boundaries" and "research-grade empirical framework" implied that SQL execution sandboxing, data pipelines, model training, and evaluation evaluators were operational in code.
* **Remedy:** Reframed `README.md` Section 2 into strict categories: *Implemented & Tested*, *Specified but Not Implemented*, and *Planned*. Qualified Section 6 with an explicit alert identifying defensive execution boundaries as design specifications to be built in Step 9.
* **Verification:** Verified by inspecting `README.md` diff and running test suite. Implementation status remains appropriately scoped to Step 0/1 foundation.

---

### Issue B: Experiment-Artifact Retention Conflict
* **Category:** Artifact Governance & Reproducibility
* **Status:** **FIXED** (Specification & Schema Enhancement)
* **Locations:**
  - `docs/research/experiment_record_spec.md` (Lines 10–55)
  - `docs/experiment_protocol.md` (Section 5)
  - `src/sqlforge/experiments/tracker.py` (Collision safety, `write_manifest`, `verify_run`)
  - `tests/unit/test_tracker.py` (`test_tracker_collision_safety`, `test_tracker_manifest_and_run_verification`)
* **Problem:** Permanent retention of raw execution logs contradicted `.gitignore` rules excluding `artifacts/runs/*`. Unclear commit boundaries.
* **Remedy:** Formalized a 3-tier retention architecture: Tier 1 (Git tracks code, configs, schemas, tests, and small fixtures in `tests/fixtures/runs/`), Tier 2 (Local/external permanent archive for raw runs, checkpoints, and multi-GB datasets with SHA-256 manifests), and Tier 3 (Never committed credentials/weights). Implemented collision-safety (`FileExistsError` on existing populated directories) and cryptographic run verification in `ExperimentTracker`.
* **Verification:** Verified via `test_tracker_collision_safety` and `test_tracker_manifest_and_run_verification`.

---

### Issue C: Evaluation Metric Terminology & Equivalence Contracts
* **Category:** Evaluation Methodology
* **Status:** **FIXED** (Contract & Specification Remedy; Engine Implementation Pending Step 9)
* **Locations:**
  - `docs/research/metrics_and_statistics.md` (Sections 1–3)
  - `src/sqlforge/schemas/evaluation.py` (`EvaluationMetrics` class)
  - `tests/unit/test_research_spec.py` (`test_schemas_support_research_matrix`)
* **Problem:** VSR was used ambiguously to mean both syntax parsing and database executability. Denominators for invalid SQL, timeouts, and empty results were not formalized. Result equivalence semantics (column alignment, float tolerances, multiset matching) lacked explicit rules.
* **Remedy:** Defined 5 distinct metrics: Syntax Validity Rate (SVR), Execution Success Rate (ESR), Valid-SQL Rate (VSR $\equiv$ ESR alias), Exact Match (EM), and Execution Accuracy (EX). Established strict denominator $N$ across all metrics. Formalized positional column matching, multiset bag equality, strict ORDER BY sensitivity, float tolerance $\epsilon = 10^{-4}$, and `AMBIGUOUS_EMPTY_SET` handling. Added benchmark compatibility notice.
* **Implementation Gap Note:** *The execution comparator itself is not yet implemented (scheduled for Step 9/10). Current fix establishes unambiguous contracts.*

---

### Issue D: Experiment-Design Ambiguities & Equivalence Bounds
* **Category:** Experimental Design & Statistics
* **Status:** **FIXED** (Specification & Config Remedy)
* **Locations:**
  - `configs/experiments.yaml` (Complete 10-experiment sequence `exp01`–`exp10`)
  - `docs/research/hypotheses.md` (Confirmatory Hypotheses 1–4, Section 4 Anti-Leakage Safeguards)
  - `tests/unit/test_research_spec.py` (`test_config_experiments_yaml_contains_all_ten_matrix_experiments`)
* **Problem:** Hypotheses previously conflated non-significant difference ($p \ge 0.05$) with demonstrated equivalence. `configs/experiments.yaml` omitted experiments `exp07` through `exp10`. Sample size limitations for the ~100 custom held-out queries were unstated.
* **Remedy:** Formalized Two One-Sided Tests (TOST) equivalence framework with non-inferiority margins ($\delta = 2.5\%$ for H1, $\delta = 2.0\%$ for H2). Documented that CI must fall strictly within $[-\delta, +\delta]$ to claim equivalence. Added explicit sample limitation alert for custom held-out schema. Formalized nested subset scaling and synthetic data anti-confounding controls. Added all 10 experiments to `configs/experiments.yaml`.
* **Verification:** Validated via `sqlforge config validate` and `test_config_experiments_yaml_contains_all_ten_matrix_experiments`.

---

### Issue E: Retrieval Arms & Serving Stack Characterization
* **Category:** Retrieval & Serving Specifications
* **Status:** **FIXED** (Specification & Config Remedy)
* **Locations:**
  - `configs/experiments.yaml` (`exp02_fewshot_retrieval`, `exp08_quantization`, `exp10_serving_bench`)
  - `docs/research/hypotheses.md` (Inquiries 3 & 4)
  - `tests/unit/test_research_spec.py` (`test_exp02_fewshot_retrieval_specification`)
* **Problem:** `exp02_fewshot_retrieval` claimed to compare BM25 with dense retrieval, but only specified `retriever: bm25`. AWQ and GGUF were loosely grouped without clarifying differing deployment targets.
* **Remedy:** Updated `configs/experiments.yaml` to explicitly specify both `bm25` and `dense_embedding` retrieval arms, dense embedding model (`BAAI/bge-small-en-v1.5`), and prompt token budget (4096). Clarified AWQ (GPU vLLM serving) vs. GGUF (CPU/edge local runtime) as distinct deployment architectures. Separated token throughput (TPS) from query throughput (QPS).
* **Verification:** Verified via `test_exp02_fewshot_retrieval_specification`.

---

### Issue F: Dependency Pinning & Reproducibility Reality
* **Category:** Environment & Build Engineering
* **Status:** **FIXED** (Policy & Infrastructure Documentation)
* **Locations:**
  - `docs/development_setup.md` (Section 7)
  - `requirements-constraints.txt`
  - `src/sqlforge/reproducibility.py`
* **Problem:** Dependency policy in `pyproject.toml` used unpinned `>=` bounds without a pinned constraints mechanism for reproducible execution. Claims of reproducibility needed clear qualification regarding GPU kernel nondeterminism.
* **Remedy:** Created `requirements-constraints.txt` capturing exact package versions for development. Documented supported Python runtimes (3.11, 3.12, 3.13) and added an explicit warning that seed setting provides software repeatability but cannot guarantee bitwise identical outputs across differing GPU architectures or CUDA kernels.
* **Verification:** Constraints file created and tested with local pip environment.

---

### Issue G: Test Strategy Alignment (Structural vs. Behavioral)
* **Category:** Testing & CI
* **Status:** **FIXED** (Test Suite Expansion)
* **Locations:**
  - `tests/unit/test_research_spec.py`
  - `tests/unit/test_tracker.py`
* **Problem:** Initial research spec tests only validated file existence and superficial string matches, failing to verify configuration completeness, retrieval arms, collision safety, or manifest verification.
* **Remedy:** Expanded test suite with 5 new/upgraded behavioral and structural tests:
  - Verifying all 10 experiments in `configs/experiments.yaml` with required fields.
  - Verifying both retrieval arms and embedding model in `exp02`.
  - Verifying collision prevention in `ExperimentTracker` (`FileExistsError`).
  - Verifying SHA-256 manifest generation and tampering detection in `verify_run`.
  - Verifying TOST equivalence clauses in `hypotheses.md`.
* **Implementation Gap Reminder:** *Pipeline execution tests cannot run because Step 2 has not been implemented yet. All missing pipeline components remain marked UNVERIFIED.*
* **Verification:** 34 tests passing cleanly in pytest.

---

### Issue H: Run ID Validation and Path Containment Deficiencies
* **Category:** Foundation Hardening & Security
* **Status:** **FIXED** (Implementation & Regression Tests)
* **Locations:**
  - `src/sqlforge/experiments/tracker.py` (`validate_run_id`, `_resolve_run_dir`, `init_run`)
  - `tests/unit/test_tracker.py` (`test_tracker_invalid_run_ids_rejected`, `test_tracker_empty_directory_collision`)
* **Problem:** Run IDs were not strictly validated, allowing path traversal sequences (`..`), slashes, absolute paths, or special characters. Pre-existing empty directories were not rejected atomically upon initialization.
* **Remedy:** Added conservative validation regex `^[a-zA-Z0-9_\-]+$` with length $\le 128$. Enforced strict containment within the configured artifact directory root (`run_dir.parent == self.base_dir`). Replaced lenient directory checks with atomic directory reservation (`run_dir.mkdir(parents=False, exist_ok=False)`), preventing silent collisions on both populated and empty directories.
* **Verification:** Verified via `test_tracker_invalid_run_ids_rejected` and `test_tracker_empty_directory_collision`.

---

### Issue I: Destructive Anomaly Log Overwrites
* **Category:** Foundation Hardening & Data Integrity
* **Status:** **FIXED** (Implementation & Regression Tests)
* **Locations:**
  - `src/sqlforge/experiments/tracker.py` (`log_anomaly`)
  - `tests/unit/test_tracker.py` (`test_tracker_corrupted_anomaly_log_preserves_content`)
* **Problem:** If `eval_anomalies.json` was corrupted or contained malformed JSON, subsequent writes could silently wipe or reset the log to an empty list, destroying diagnostic evidence.
* **Remedy:** Implemented non-destructive anomaly parsing. If the file is malformed JSON or contains a non-list root structure, `log_anomaly` preserves the existing file completely untouched and raises an actionable `ValueError`.
* **Verification:** Verified via `test_tracker_corrupted_anomaly_log_preserves_content`.

---

### Issue J: Lenient Manifest and Record Syntax Verification
* **Category:** Foundation Hardening & Cryptographic Audit
* **Status:** **FIXED** (Implementation & Regression Tests)
* **Locations:**
  - `src/sqlforge/experiments/tracker.py` (`write_manifest`, `verify_run`, `finish_run`)
  - `tests/unit/test_tracker.py` (`test_tracker_verify_run_strict_vs_lenient`)
* **Problem:** `verify_run` did not mandate `manifest.json` presence by default, did not detect untracked files, did not validate line-by-line syntax in `generations.jsonl`, and relied on ambiguous hash lifecycle timing.
* **Remedy:** Formalized strict verification (`strict=True` default) requiring `manifest.json`. Added manifest self-exclusion in `write_manifest` to avoid hash circularity. Added checks for untracked files, JSONL line corruption, and YAML syntax. Synchronized lifecycle so `finish_run` updates metadata first and records final SHA-256 hashes immediately after. Clarified in documentation that SHA-256 verifies file integrity, not scientific validity.
* **Verification:** Verified via `test_tracker_verify_run_strict_vs_lenient`.

---

### Issue K: Metric Schema Consistency and Uncomputed Defaults
* **Category:** Data Contracts & Schema Validation
* **Status:** **FIXED** (Implementation & Unit Tests)
* **Locations:**
  - `src/sqlforge/schemas/evaluation.py` (`ConfidenceInterval`, `EvaluationMetrics`)
  - `tests/unit/test_schemas.py` (`test_confidence_interval_invalid_bounds_raises`, `test_evaluation_metrics_alias_reconciliation`)
* **Problem:** `valid_sql_rate` and `execution_success_rate` were documented aliases but could silently disagree. Uncomputed metrics defaulted to `0.0` or `1.0`, misleading consumers. `ConfidenceInterval` did not enforce `lower <= upper`.
* **Remedy:** Added `@model_validator(mode="after")` to `ConfidenceInterval` ensuring `lower <= upper`. Updated uncomputed rate fields in `EvaluationMetrics` to default to `None`. Added bidirectional alias reconciliation between `execution_success_rate` and `valid_sql_rate` that raises `ValueError` if explicit values conflict.
* **Verification:** Verified via `test_confidence_interval_invalid_bounds_raises` and `test_evaluation_metrics_alias_reconciliation`.

---

### Issue L: Absence of End-to-End Core Pipeline Harness
* **Category:** Core Pipeline Integration
* **Status:** **FIXED** (Implementation & Comprehensive Integration Suite)
* **Locations:**
  - `src/sqlforge/data/fixtures.py` & `tests/fixtures/dataset/mock_spider.json`
  - `src/sqlforge/prompting/builder.py`
  - `src/sqlforge/models/mock.py`
  - `src/sqlforge/evaluation/mock_eval.py`
  - `src/sqlforge/pipeline/harness.py`
  - `src/sqlforge/cli.py` (`sqlforge pipeline mock`)
  - `configs/mock_pipeline.yaml`
  - `tests/unit/test_fixtures.py`, `tests/unit/test_prompt_builder.py`, `tests/unit/test_mock_pipeline.py`, `tests/integration/test_pipeline_cli.py`
* **Problem:** Repository had configuration schemas and an experiment tracker, but no complete executable pipeline harness connecting configuration, data, prompts, models, evaluation, tracking, and verification.
* **Remedy:** Built a minimal, deterministic mock pipeline vertical slice:
  - Validated config loader with early failure on invalid configs.
  - Deterministic fixture dataset with duplicate ID checks.
  - Independent prompt builder.
  - Deterministic mock model with explicit `/* MOCK_SYNTHETIC */` disclaimers.
  - Mock evaluator checking pipeline contracts and logging anomalies.
  - End-to-end `MockPipelineHarness` with atomic run tracking and strict verification.
  - Dedicated CLI command `sqlforge pipeline mock` with `--dry-run`, `--config`, `--fixtures`, `--artifact-dir`, and `--fail-mode`.
  - Zero external network requests, zero model weights, zero real dataset ingestion, and zero empirical performance claims.
* **Verification:** 60 automated tests passing across unit and CLI integration suites.

---

### Issue M: Manifest Path Containment & Symlink Traversal Deficiencies
* **Category:** Foundation Hardening & Security
* **Status:** **FIXED** (Implementation & Regression Tests)
* **Locations:**
  - `src/sqlforge/experiments/tracker.py` (`verify_run`, `write_manifest`)
  - `tests/unit/test_tracker.py` (`test_tracker_verify_manifest_path_containment_and_traversal`, `test_tracker_verify_manifest_malformed_hashes`, `test_tracker_verify_nested_legitimate_artifacts`)
* **Problem:** In `verify_run`, manifest artifact path keys were not validated before accessing the filesystem (`target = run_dir / rel_path`), allowing path traversal (`../`), absolute paths, or escaping drive paths to be targeted. Malformed non-hex or invalid SHA-256 strings were not rejected. Symlinks resolving outside the run directory could escape containment.
* **Remedy:** Added pre-resolution path validation rejecting absolute paths, drive prefixes, and traversal paths (`..` in parts). Added regex verification for 64-hex SHA-256 hashes (`SHA256_HEX_REGEX`). Enforced strict resolution containment (`target.resolve().relative_to(run_dir.resolve())`) and symlink safety in both `verify_run` and `write_manifest`. Supported legitimate nested directories.
* **Verification:** Verified via `test_tracker_verify_manifest_path_containment_and_traversal`, `test_tracker_verify_manifest_malformed_hashes`, and `test_tracker_verify_nested_legitimate_artifacts`.

---

### Issue N: Incomplete Artifact Contract & Status Lifecycle Verification
* **Category:** Data Contracts & Verification
* **Status:** **FIXED** (Implementation & Regression Tests)
* **Locations:**
  - `src/sqlforge/experiments/tracker.py` (`verify_run`)
  - `tests/unit/test_tracker.py` (`test_tracker_verify_artifact_schemas_and_failed_runs`)
* **Problem:** `verify_run` only verified file existence and checksums for `metrics.json` and `eval_anomalies.json`, but did not validate that `metrics.json` satisfies the Pydantic `EvaluationMetrics` contract or that `eval_anomalies.json` is a valid JSON list of dictionaries. Lifecycle rules for failed runs vs. completed runs were not explicitly differentiated.
* **Remedy:** Added Pydantic schema validation for `metrics.json` via `EvaluationMetrics` and structure validation for `eval_anomalies.json` (JSON list of dicts). Explicitly defined that completed runs require `metrics.json`, while failed runs do not require `metrics.json` but require valid metadata, config, manifest, and intact anomalies if logged. Documented precisely what verification guarantees vs. what cryptographic hashes do not guarantee.
* **Verification:** Verified via `test_tracker_verify_artifact_schemas_and_failed_runs`.


