# SQLForge Experiment Record & Run Manifest Specification

This document defines the schema, serialization rules, and lifecycle management for machine-readable experiment records in SQLForge.

---

## 1. Directory Structure per Run

Every experiment execution creates an isolated, self-contained directory in `artifacts/runs/<run_id>/`:

```text
artifacts/runs/<run_id>/
├── config.yaml              # Frozen resolved configuration snapshot
├── run_metadata.json        # Machine environment, git commit, seed, status
├── manifest.json            # SHA-256 cryptographic checksums of all run artifacts
├── metrics.json             # Aggregated quality, efficiency, and bootstrap metrics
├── generations.jsonl        # Line-by-line model predictions and execution outcomes
├── eval_anomalies.json       # Record of timeouts, syntax errors, and ambiguous comparisons
├── events.log               # Structured chronological log with error traces
└── checkpoints/             # PEFT LoRA adapter checkpoints (ephemeral / external storage)
```

---

## 2. Artifact Retention Tiers & Git Tracking Policy

To reconcile permanent reproducibility requirements with Git repository cleanliness and GitHub storage limits:

### Tier 1: Tracked in Git
- System source code, schemas, and CLI tooling (`src/sqlforge/`).
- Layered YAML configuration definitions (`configs/`).
- Unit and integration tests (`tests/`).
- Small, representative test fixtures (`tests/fixtures/runs/`) for pipeline testing.
- Aggregated derived summary tables, figures, and research write-ups (`reports/`, `docs/`).

### Tier 2: Local & External Persistent Storage (Gitignored)
- Raw experiment execution directories (`artifacts/runs/*`).
- Intermediate and processed datasets (`data/raw/*`, `data/processed/*`).
- Model checkpoints and adapter weights (`artifacts/checkpoints/*`, `models/weights/*`).
- **Archival Procedure:** For permanent archival, run directories are bundled into an immutable tarball with an overall SHA-256 manifest and published to a public scientific repository (e.g. Zenodo, Hugging Face Hub dataset).

### Tier 3: Never Committed Under Any Circumstances
- Commercial API secret keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
- Machine-specific credentials or sensitive configuration files (`.env`).
- Multi-gigabyte raw model weights.

### Collision-Safe Run Execution
- All run IDs are generated with deterministic timestamps and random UUID suffixes: `<prefix>_<YYYYMMDD_HHMMSS>_<short_uuid>`.
- The `ExperimentTracker` actively checks whether target run directories exist. If an existing directory contains files, execution halts immediately with a `FileExistsError` to prevent silent overwriting of historical data.

### Provenance & Completeness Verification
- At run completion, `tracker.finish_run()` automatically invokes `write_manifest()` to compute SHA-256 digests for all generated files and write `manifest.json`.
- The `tracker.verify_run(run_id)` diagnostic checks:
  1. Presence of required files (`run_metadata.json`, `config.yaml`, and `metrics.json` if status is `completed`).
  2. Bitwise hash agreement of every file listed in `manifest.json`.
  3. Non-corruption of the JSON/JSONL format.


---

## 2. Machine-Readable Schema Specification

### `run_metadata.json` Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SQLForgeRunMetadata",
  "type": "object",
  "required": [
    "run_id",
    "experiment_id",
    "status",
    "timestamp_utc_start",
    "git",
    "environment",
    "seed",
    "config_hash"
  ],
  "properties": {
    "run_id": {
      "type": "string",
      "description": "Unique run identifier: <prefix>_<YYYYMMDD_HHMMSS>_<uuid6>"
    },
    "experiment_id": {
      "type": "string",
      "description": "Correlating experiment matrix ID (e.g. EXP-03-LORA-VS-QLORA)"
    },
    "status": {
      "type": "string",
      "enum": ["initialized", "running", "completed", "failed", "aborted"]
    },
    "timestamp_utc_start": { "type": "string", "format": "date-time" },
    "timestamp_utc_end": { "type": ["string", "null"], "format": "date-time" },
    "elapsed_seconds": { "type": ["number", "null"] },
    "git": {
      "type": "object",
      "properties": {
        "commit": { "type": "string" },
        "branch": { "type": "string" },
        "is_dirty": { "type": "boolean" }
      },
      "required": ["commit", "branch", "is_dirty"]
    },
    "environment": {
      "type": "object",
      "properties": {
        "os": { "type": "string" },
        "python_version": { "type": "string" },
        "torch_version": { "type": ["string", "null"] },
        "cuda_version": { "type": ["string", "null"] },
        "hardware": {
          "type": "object",
          "properties": {
            "cpu_count": { "type": "integer" },
            "total_ram_gb": { "type": "number" },
            "gpu_count": { "type": "integer" },
            "gpu_devices": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "seed": { "type": "integer" },
    "config_hash": { "type": "string", "description": "SHA-256 hash of config.yaml" },
    "artifacts": {
      "type": "object",
      "properties": {
        "config_path": { "type": "string" },
        "metrics_path": { "type": "string" },
        "generations_path": { "type": "string" },
        "checkpoint_path": { "type": ["string", "null"] }
      }
    },
    "failure_reason": { "type": ["string", "null"] }
  }
}
```

---

## 3. Handling Failed, Incomplete & Aborted Runs

To preserve full scientific auditability and avoid survival bias:
1. **Never Silently Discard:** Failed runs (e.g. CUDA OOM, syntax parse failure, loss divergence) must **never be deleted**.
2. **Explicit Failure Logging:** When a failure occurs:
   - `status` is updated to `"failed"`.
   - `failure_reason` records the exact Python exception, traceback, or early-stopping gate trigger.
   - `timestamp_utc_end` and `elapsed_seconds` are populated.
3. **Artifact Quarantine:** Incomplete checkpoints or corrupted generation JSONL files are retained with a `.incomplete` suffix for post-mortem debugging.

---

## 4. Separation of Raw Outputs from Derived Publications

* **Raw Layer (`artifacts/runs/`):** Contains raw JSON, YAML, and JSONL generation traces. Never modified once written; gitignored.
* **Derived Reports (`reports/`):** Markdown summary tables, aggregated comparison charts, and publication figures (`reports/figures/`) generated reproducibly from raw artifacts via explicit reporting scripts.
