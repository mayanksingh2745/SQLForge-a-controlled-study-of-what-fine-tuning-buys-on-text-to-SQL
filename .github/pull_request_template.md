## Summary & Motivation
<!-- Brief summary of what this PR does, why it is needed, and which project step or issue it addresses. -->

## Key Changes
<!-- Bulleted list of architectural, functional, configuration, or documentation changes. -->
- 

## Research & Reproducibility Impact
<!-- How does this change affect experimental reproducibility, random seeds, config hashing, or research metrics? -->
- **Configuration Changes:** 
- **Determinism / Seed Sensitivity:** 
- **Dependencies Added:** 

## Security & SQL Sandbox Safety Review
<!-- Does this change introduce or touch query execution, database connections, timeouts, or credentials? -->
- [ ] Confirmed untrusted SQL execution remains bounded by read-only mode and timeouts.
- [ ] Confirmed no secrets, private tokens, or credentials are hardcoded or committed.
- [ ] Confirmed benchmark databases are isolated and non-production.

## Testing & Verification Evidence
<!-- Provide the exact commands executed locally and their actual outputs. -->
```bash
# Formatter check
ruff format --check .

# Linter check
ruff check .

# Static type check
mypy src/sqlforge

# Test suite
pytest -v tests/
```

- **Unit/Integration Test Results:**
- **Local Environment Verified:** 

## Risks, Tradeoffs & Known Limitations
<!-- Explicitly describe any edge cases, platform-specific limitations (e.g. Windows vs Linux), or unresolved items. -->
- 

## Pre-Merge Checklist
- [ ] Base branch is up-to-date with `main`.
- [ ] Branch follows naming convention (`feat/*`, `exp/*`, `fix/*`, `chore/*`).
- [ ] All new code is covered by automated unit/integration tests.
- [ ] All public functions, classes, and modules have accurate docstrings and type hints.
- [ ] All CI checks pass on GitHub Actions.
- [ ] Documentation and ADRs updated where applicable.
