# SQLForge Experiment Protocol & Execution Standards

## 1. Purpose

This document establishes the binding scientific and operational protocol for all experimental runs within the SQLForge research program. Adherence to this protocol ensures that every experimental result is verifiable, reproducible, and audit-ready.

---

## 2. Run Naming & Identifier Scheme

Every experiment invocation must generate a deterministic, unambiguous run identifier using the format:

```text
<paradigm>_<model_slug>_<dataset_split>_<YYYYMMDD_HHMMSS>_<short_uuid>
```

### Examples
* `zeroshot_qwen25_7b_spiderdev_20261002_143000_a8f1b2`
* `lora_r16_qwen25_7b_spidertrain_20261002_150000_c3e4f5`
* `fewshot_bm25_gpt4omini_spiderdev_20261002_160000_91d8e0`

---

## 3. Configuration Snapshotting & Hash Audit

Before executing model training or evaluation:
1. **Full Config Resolution:** All default and override parameters must be resolved into an immutable configuration instance (`ExperimentConfig`).
2. **Deterministic Checksum:** A SHA-256 hash of the canonical JSON representation of the resolved config is computed (`config_hash`).
3. **Artifact Persistence:** The resolved YAML configuration is saved as `artifacts/runs/<run_id>/config.yaml`.
4. **Environment Audit:** The system records runtime metadata in `run_metadata.json`:
   - Git HEAD commit hash and working tree dirty flag.
   - Exact Python, PyTorch, CUDA, and OS platform versions.
   - Machine hardware resources (CPU cores, RAM GB, GPU device names).
   - Execution timestamp in UTC ISO-8601.

---

## 4. Multi-Seed Randomization Protocol

To prevent reporting anomalous lucky runs:
* **Fine-Tuning Experiments:** Supervised fine-tuning runs must be repeated across at least three distinct random seeds: `[42, 43, 44]`.
* **Reported Metrics:** Tables must report the mean metric alongside the empirical 95% bootstrap confidence interval derived across pooled runs.
* **Deterministic Inference:** Temperature for primary benchmark reporting must be set to `0.0` (greedy decoding) to eliminate sampling variance during evaluation.

---

## 5. Artifact Retention & Quarantine Policy

To maintain repository hygiene, avoid GitHub storage saturation, and ensure permanent scientific reproducibility:

```text
artifacts/runs/<run_id>/
├── config.yaml          # Frozen resolved configuration snapshot (< 10 KB)
├── run_metadata.json    # Provenance, git hash, hardware, seeds (< 5 KB)
├── manifest.json        # Cryptographic SHA-256 manifest of all files (< 5 KB)
├── metrics.json         # Aggregated quality, efficiency, and bootstrap metrics (< 20 KB)
├── generations.jsonl    # Line-by-line model predictions and execution outcomes (< 5 MB)
├── eval_anomalies.json   # Log of timeouts, syntax errors, and ambiguous matches (< 50 KB)
└── checkpoints/         # Ephemeral (LoRA adapters only; base weights NEVER saved)
```

1. **Storage Tiering vs. Git Tracking:**
   - **Tracked in Git:** Code, YAML configurations, Pydantic schemas, unit tests, and lightweight test fixtures (`tests/fixtures/runs/`).
   - **Local / External Persistent Storage (Gitignored):** Raw run outputs (`artifacts/runs/*`) and model checkpoints (`artifacts/checkpoints/*`) are stored locally or published to dedicated external open repositories (e.g. Zenodo or Hugging Face Hub dataset). They are strictly gitignored to keep the repository lightweight and portable.
2. **Adapter Only:** Never save full 7B base model weights. Save only the PEFT LoRA adapter checkpoint (`adapter_model.bin` or `adapter_model.safetensors`, typically < 50 MB).
3. **Collision Safety:** Run directories use timestamped, UUID-tagged identifiers. If an existing populated directory is targeted, `ExperimentTracker` raises `FileExistsError` rather than silently overwriting past records.
4. **Provenance & Manifest Verification:** All completed runs generate a `manifest.json` with SHA-256 digests. Run integrity is validated via `tracker.verify_run(run_id)`.
5. **Checkpoint Pruning:** During hyperparameter sweeps, retain only the checkpoint with the highest validation execution accuracy on the dev set. Delete intermediate optimizer states (`optimizer.pt`) after training concludes.


---

## 6. Evaluation Isolation & Anti-Leakage Rules

1. **Read-Only Database Connections:** All query executions during evaluation must use read-only SQLite flags (`mode=ro`) or dedicated unprivileged test user accounts.
2. **No Dev Feedback to Training:** The training loop must never evaluate on test sets or calculate loss on evaluation databases.
3. **Quarantine Unparseable Queries:** Queries causing catastrophic timeout or system errors must be recorded with their specific error taxonomy code, never silently dropped or replaced.
