# SQLForge Mandatory GitHub Workflow & Quality Policy

This document establishes the binding development lifecycle and quality policy for SQLForge. Every new project step, coherent feature, bug fix, or substantial refactor must follow this exact 12-step process.

---

## 1. The 12-Step Development Lifecycle

```text
[1. Inspect Git & Base] ──► [2. Update Base (main)] ──► [3. Create Feature Branch]
                                                                  │
                                                                  ▼
[6. Commit Cleanly]     ◄── [5. Run Local Checks]   ◄── [4. Implement & Test]
        │
        ▼
[7. Push & Open PR]     ──► [8. Fill PR Description] ──► [9. Pass CI Checks]
                                                                  │
                                                                  ▼
[12. Post-Merge & Clean] ◄── [11. Merge to main]    ◄── [10. Review & Audit]
```

### Step 1: Inspect Current Status
Before initiating any work:
```bash
git status
git branch -a
git log -n 3
```
Ensure there are no uncommitted, unverified changes or detached states.

### Step 2: Update Base Branch
Always sync your local base branch with upstream `origin/main` before branching:
```bash
git checkout main
git pull origin main
```

### Step 3: Create a Descriptive Feature Branch
**Never develop directly on `main`.** Create a task-specific branch off `main`:
* Features / Project Steps: `feat/step-<N>-<description>` (e.g. `feat/step-1-dataset-ingestion`)
* Research Experiments: `exp/<exp-id>-<description>` (e.g. `exp/exp03-lora-rank-sweep`)
* Bug Fixes: `fix/<issue-id>-<description>` (e.g. `fix/sqlite-timeout-interrupt`)
* Chores / Tooling: `chore/<description>` (e.g. `chore/update-pre-commit-hooks`)

```bash
git checkout -b feat/step-0-foundation
```

### Step 4: Implement with Tests and Documentation
* Keep research code modular and decoupled from serving and data layers.
* Update or add relevant YAML configurations in `configs/`.
* Write corresponding unit and integration tests in `tests/`.
* Update documentation in `docs/` and add Architecture Decision Records (ADRs) in `docs/decisions/` when introducing structural changes.

### Step 5: Run All Local Checks
Execute the automated local quality suite before committing:
```powershell
# Windows (PowerShell)
.\scripts\run_checks.ps1

# Linux / macOS
make check
```
Inspect the working tree diff:
```bash
git diff
```

### Step 6: Commit Changes
Commit with clear, Conventional Commit messages:
```bash
git add <files>
git commit -m "feat(step-0): establish repository foundation, typed contracts, and tooling"
```
* Never commit secrets, credentials, model weights, or private datasets.

### Step 7: Push Branch and Open PR
Push the feature branch to `origin`:
```bash
git push -u origin feat/step-0-foundation
```
Open a Pull Request against `main`:
```bash
gh pr create --base main --head feat/step-0-foundation
```

### Step 8: Complete Pull Request Description
Populate all required sections of `.github/pull_request_template.md`:
* **Summary & Motivation**
* **Key Changes**
* **Research & Reproducibility Impact** (config hashes, seeds)
* **Security & SQL Sandbox Review**
* **Testing & Verification Evidence** (actual output pasted)
* **Risks & Known Limitations**

### Step 9: Await and Verify Automated CI
Wait for all GitHub Actions CI checks to complete on the PR:
```bash
gh pr checks
```
* **Never bypass failing checks.** If CI fails on any OS or Python version in the matrix, reproduce the failure locally, fix the underlying defect, and push the update.

### Step 10: Technical and Security Review
Verify:
* Architecture and dependency direction (ADR compliance).
* SQL sandboxing and read-only execution boundaries.
* Reproducibility guarantees (fixed seeds, config snapshots).
* Absence of accidental large artifacts or secrets.

### Step 11: Merge into `main`
Merge only after all status checks pass and the review is complete:
```bash
gh pr merge --squash --delete-branch
```
* Use **Squash and Merge** (or Rebase) to maintain a linear, clean commit history on `main`.

### Step 12: Post-Merge Verification & Roadmap Update
Update local `main`, confirm merged code presence, and update roadmap status:
```bash
git checkout main
git pull origin main
git log -n 1
```
Verify that the roadmap in `README.md` is updated.

---

## 2. Hard Quality Invariants

1. **No Direct Commits to `main`:** All code enters `main` exclusively through Pull Requests with passing CI checks.
2. **No Secret Ingestion:** Tokens, private keys, and API secrets must never enter version control.
3. **No Fabricated Confirmations:** Never fabricate PR URLs, CI checks, or test outcomes.
4. **Reproducibility Guarantee:** Every experiment merged into the repository must have an accompanying validated configuration and seed protocol.
5. **Step Completion Standard:** A project step is marked complete only after all required PRs are merged into `main` and verified.
