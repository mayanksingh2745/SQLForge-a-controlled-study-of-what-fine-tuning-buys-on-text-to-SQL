## Summary & Architectural Scope

This pull request implements the **Step 2.1 Corrective Review** for SQLForge, hardening manifest path containment boundaries, malformed hash validation, symlink traversal prevention, and complete artifact schema verification across run lifecycles.

### Key Fixes Implemented

1. **Manifest Path Containment (`src/sqlforge/experiments/tracker.py`):**
   - Validates all manifest artifact path keys before filesystem access.
   - Strictly rejects absolute paths, drive roots (`C:\`), UNC prefixes (`\\`), `..` path traversals, empty keys, and whitespace.
   - Enforces post-resolution containment (`target.resolve().relative_to(run_dir.resolve())`) and symlink defenses so links pointing outside the run directory are rejected as security violations.
   - Validates that legitimate nested artifact directories (e.g. `checkpoints/step_100/adapter_config.json`) are verified cleanly.

2. **Cryptographic Hash Validation (`src/sqlforge/experiments/tracker.py`):**
   - Strictly validates every SHA-256 hash against `^[a-fA-F0-9]{64}$`.
   - Rejects non-hex characters, numbers, and malformed length strings.

3. **Complete Artifact Schema Verification (`src/sqlforge/experiments/tracker.py`):**
   - Validates `metrics.json` against the typed Pydantic `EvaluationMetrics` contract (ensuring total_examples, execution_accuracy, exact_match_accuracy, and bounds).
   - Validates `eval_anomalies.json` root structure as a JSON list of dictionaries.
   - Differentiates required files for completed runs (`metrics.json` required) vs. failed runs (`metrics.json` optional; failure diagnostics preserved).
   - Preserves valid failure states without destructive overwrite.

4. **Documentation & Specification Transparency:**
   - Documented precise verification guarantees (presence, containment, schema validity, bitwise match, completeness) vs. non-guarantees (scientific validity, SQL safety, absence of data leakage) in `verify_run` docstrings.
   - Updated `docs/research/step_dependencies.md` so Steps 0, 1, and 2 accurately reflect their merged and completed status.
   - Updated `review/REVIEW_LOG.md`, `review/ISSUES.md` (Issues M and N), and `review/SCORECARD.md`.

---

## Verification & Test Evidence

All checks executed locally:
* `ruff format --check .` (77 files already formatted)
* `ruff check .` (All checks passed)
* `mypy src/sqlforge` (Success: no issues found in 30 source files)
* `pytest -v tests/` (64 passed in 6.64s)
* `sqlforge --version` (SQLForge 0.1.0)
* `sqlforge --help` (CLI commands displayed)
* `sqlforge config validate` (All 5 configs PASSED)
* `sqlforge pipeline mock --dry-run` (Dry run validated cleanly)
* `sqlforge pipeline mock` (Completed and strictly verified)
* `sqlforge pipeline mock --fail-mode` (Preserved failure diagnostics and exited non-zero)

---

## Strict Scope Boundaries

* Zero real model weights downloaded.
* Zero frontier API network requests.
* Zero benchmark dataset ingestion (scheduled for Step 3).
* Zero empirical claims made.
