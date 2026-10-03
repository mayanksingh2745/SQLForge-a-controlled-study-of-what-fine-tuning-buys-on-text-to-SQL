# SQLForge Supervised Fine-Tuning (SFT) Protocol: LoRA vs. QLoRA (EXP-03)

This document formalizes the architecture, configuration parameters, safety gates, dataset formatting, completion-only loss masking, checkpoint management, and evaluation handoff for the Supervised Fine-Tuning (SFT) infrastructure implemented in Step 6.

---

## 1. Architectural Overview & Separation of Concerns

The fine-tuning infrastructure in `src/sqlforge/training/` is strictly modular, ensuring clear separation of concerns across configuration, preflight verification, dataset adaptation, loss masking, PEFT/quantization setup, checkpoint persistence, and evaluation handoff:

```mermaid
graph TD
    Config["SFTTrainingConfig (YAML/CLI)"] --> Preflight["PreflightChecker (CUDA, RAM, Disk, Deps)"]
    Preflight -->|Safety Check Passed| Engine["SFTFineTuningPipeline"]
    
    DataRaw["Raw Dataset Split (DatasetSplit.TRAIN)"] --> Formatter["SFTDatasetFormatter"]
    Formatter -->|Prompt + SQL Completion| Masker["CompletionLossMasker"]
    Masker -->|Token IDs + Target Labels (-100)| Batch["Collated DataLoader"]
    
    Config --> Adapters["PEFTConfigFactory (LoRA / QLoRA NF4)"]
    Adapters --> Model["Base Model + PEFT Adapter"]
    
    Batch --> Engine
    Model --> Engine
    
    Engine --> Checkpoints["CheckpointManager (Metadata + Adapter Weights)"]
    Engine --> Tracker["ExperimentTracker (Metrics, Manifest, Logs)"]
    Checkpoints --> Handoff["Evaluation Handoff (LocalHFModelRunner + Adapter)"]
```

### Module Responsibilities

1. **`src/sqlforge/training/config.py`**:
   - `SFTTrainingConfig`: Immutable Pydantic model validating base model, LoRA/QLoRA hyperparameters, target modules, optimization hyperparameters, dataset splits, and output paths.
   - `resolve_target_modules()`: Architecture-aware resolution of target modules (e.g., all-linear vs attention-only for Qwen/Llama architectures).
2. **`src/sqlforge/training/preflight.py`**:
   - `PreflightChecker`: System feasibility and safety gate auditing GPU availability, VRAM sufficiency, host RAM, free disk space ($\ge 3.0$ GB), and library dependencies (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`).
   - Distinguishes blocking errors (`FAIL`) from non-blocking resource alerts (`WARNING`), preventing accidental execution crashes or host memory thrashing.
3. **`src/sqlforge/training/data.py`**:
   - `SFTDatasetFormatter`: Converts `TextToSQLExample` records into `SFTExample` structures consisting of serialized schema context, natural-language question, and gold target SQL.
   - Strictly enforces split isolation: examples from `DatasetSplit.DEV`, `TEST`, or `HELD_OUT` raise `DisallowedSplitError`.
4. **`src/sqlforge/training/tokenization.py`**:
   - `CompletionLossMasker`: Tokenizes prompt and target sequences, appending EOS tokens and masking all prompt token positions with `IGNORE_INDEX = -100` so that training loss is computed exclusively on the target SQL completion tokens.
   - Detects severe target truncation (`TargetTruncationError`) where context length limits truncate the assistant completion.
5. **`src/sqlforge/training/adapters.py`**:
   - `PEFTConfigFactory`: Builds Hugging Face `LoraConfig` and `BitsAndBytesConfig` (4-bit NF4 with double quantization and 16-bit compute dtype).
   - Rejects QLoRA initialization on unsupported platforms (`UnsupportedQuantizationPlatformError`) without silent fallback to full-precision or unquantized LoRA.
6. **`src/sqlforge/training/checkpoints.py`**:
   - `CheckpointManager`: Atomically saves and loads adapter checkpoints and metadata (`checkpoint_metadata.json`), recording step, epoch, base model, method, and git state while preventing accidental overwrites.
7. **`src/sqlforge/training/evaluation_handoff.py`**:
   - Bridges fine-tuned adapter checkpoints to `LocalHFModelRunner`, validating adapter files (`adapter_config.json`, weights) and generating `EvaluationConfig` snapshots targeting the evaluation split (`spider:dev`).
8. **`src/sqlforge/training/trainer.py`**:
   - `SFTFineTuningPipeline`: Orchestrates preflight gates, data loading, mock/dry-run and live training loops, logging metrics into `ExperimentTracker` (`artifacts/runs/<run_id>/`).

---

## 2. LoRA vs. QLoRA Controlled Configuration

To fairly isolate the effect of 4-bit NormalFloat (NF4) quantization in `EXP-03`, LoRA and QLoRA configurations share identical adapter architectures and optimization settings:

| Parameter | 16-bit LoRA | 4-bit QLoRA | Research Rationale |
| :--- | :--- | :--- | :--- |
| **Base Model Precision** | 16-bit (bf16/fp16) | 4-bit NormalFloat (`nf4`) | Tests weight quantization impact on SQL synthesis |
| **Double Quantization** | N/A | `True` (`nested_quant=True`) | Saves ~0.37 bits/param in base model quantization |
| **Quantization Compute Dtype**| N/A | `bfloat16` or `float16` | Ensures dynamic range stability during backward pass |
| **LoRA Rank ($r$)** | 16 | 16 | Kept identical for controlled parameter comparison |
| **LoRA Alpha ($\alpha$)** | 32 | 32 | Scaling factor $\alpha / r = 2.0$ held constant |
| **LoRA Dropout** | 0.05 | 0.05 | Regularization held constant |
| **Target Modules** | All linear projections | All linear projections | Standardized across attention and MLP projections |
| **Optimizer** | `adamw_torch` (or paged 8-bit) | `paged_adamw_8bit` (or AdamW) | Mitigates memory spikes during gradient accumulation |
| **Learning Rate** | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ | Standard LoRA learning rate with cosine decay |
| **Gradient Checkpointing** | Enabled | Enabled | Required for memory efficiency under large context |

### Target Module Defaults

For modern transformer architectures (such as `Qwen2.5-Coder` and `Llama-3`), `PEFTConfigFactory` supports:
- **`all-linear` (default):** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **`attention-only`:** `q_proj`, `k_proj`, `v_proj`, `o_proj`.

Explicit module target lists can be provided in configuration or via CLI.

---

## 3. Dataset Preparation & Contamination Safeguards

### Split Isolation Safeguards

Fine-tuning data must never ingest evaluation or held-out records. In `SFTDatasetFormatter`:
1. Every input `TextToSQLExample` has its `split` verified.
2. Any example belonging to `DatasetSplit.DEV`, `DatasetSplit.TEST`, or `DatasetSplit.HELD_OUT` immediately raises a `DisallowedSplitError`.
3. Validation examples are prepared strictly from authorized validation partitions (e.g. synthetic fixtures or isolated validation splits), never dev/test benchmarks.

### Example Construction

Each training example is formatted into a prompt and target pair:
- **Prompt:** Database DDL schema serialized via `DDLSerializer` (or compact-pipe serializer), optional external domain evidence (for BIRD), and the natural language user query:
  ```text
  You are an expert SQL assistant. Given the database schema and question, write a valid SQLite SQL query that answers the question.
  
  ### Database Schema:
  CREATE TABLE customers (
      customer_id INTEGER PRIMARY KEY,
      name TEXT NOT NULL
  );
  
  ### Question:
  List all customer names.
  
  ### SQL:
  ```
- **Target Completion:**
  ```text
   SELECT name FROM customers;
  ```

---

## 4. Completion-Only Loss Masking

Standard causal language modeling computes cross-entropy loss over all tokens in the sequence ($L = -\sum \log P(x_i | x_{<i})$). Accidental training on prompt tokens causes the model to optimize for predicting database DDL and question tokens, degrading sample efficiency and instruction adherence.

### Masking Protocol

`CompletionLossMasker` implements completion-only loss masking:
1. **Prompt Tokenization:** Tokenizes prompt without adding an EOS token.
2. **Target Tokenization:** Tokenizes target SQL completion and appends `tokenizer.eos_token`.
3. **Concatenation:** Concatenates prompt and target token IDs:
   $$\text{input\_ids} = [\text{prompt\_tokens}, \text{target\_tokens}]$$
4. **Label Masking:**
   $$\text{labels}[i] = \begin{cases} -100 & \text{if } i < \text{len(prompt\_tokens)} \\ \text{input\_ids}[i] & \text{if } i \ge \text{len(prompt\_tokens)} \end{cases}$$
5. **Attention Mask:** Set to `1` for all active tokens and `0` for right-padding tokens.

### Target Truncation Safeguard

When the combined length exceeds `max_seq_length`:
- If the remaining budget accommodates $\ge 4$ target tokens, the target sequence is safely truncated with a warning.
- If the prompt alone consumes all available tokens, leaving $< 4$ target tokens, a `TargetTruncationError` is raised to prevent training on empty or uninformative targets.

---

## 5. Checkpoint & Artifact Architecture

Checkpoints and run records are saved under `artifacts/checkpoints/<run_id>/`:

```text
artifacts/checkpoints/<run_id>/
├── checkpoint-100/
│   ├── adapter_config.json        # PEFT adapter configuration
│   ├── adapter_model.safetensors  # Trainable adapter weights only
│   ├── checkpoint_metadata.json   # Step, epoch, loss, git hash, base model
│   └── training_args.bin          # Framework training arguments
├── checkpoint-200/
│   └── ...
└── final_adapter/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── checkpoint_metadata.json
    └── tokenizer/                 # Pinned tokenizer files
```

### Metadata Serialization (`checkpoint_metadata.json`)

Every checkpoint contains a verifiable JSON metadata record:
```json
{
  "run_id": "sft-lora-qwen-1.5b-exp03",
  "step": 200,
  "epoch": 2.0,
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "method": "lora",
  "training_loss": 0.412,
  "validation_loss": 0.448,
  "lora_r": 16,
  "lora_alpha": 32,
  "learning_rate": 0.0002,
  "git_commit": "a1b2c3d4e5f6",
  "git_dirty": false,
  "created_at": "2026-10-03T16:30:00Z"
}
```

---

## 6. Preflight Feasibility & Safety Gates

Before any weights are downloaded or training memory is allocated, `PreflightChecker` evaluates:
- **CUDA & GPU:** Checks `torch.cuda.is_available()`, device count, and device name.
- **GPU VRAM:** Requires $\ge 8.0$ GB for 16-bit LoRA and $\ge 6.0$ GB for 4-bit QLoRA.
- **Host RAM:** Requires $\ge 8.0$ GB available host memory (warns if $< 2.0$ GB).
- **Free Disk Space:** Requires $\ge 3.0$ GB free disk space on the target artifact volume.
- **Dependencies:** Audits `torch`, `transformers`, `peft`, `accelerate`, and `bitsandbytes`.
- **Quantization Support:** For QLoRA, validates that `bitsandbytes` and a compatible CUDA runtime are available. If running on CPU or unsupported platform, raises `UnsupportedQuantizationPlatformError`.

---

## 7. CLI Reference & Developer Experience

The fine-tuning infrastructure is exposed via the `sqlforge train` CLI command group.

### Preflight Diagnostics
Audit hardware, dependencies, and platform feasibility without allocating resources:
```bash
sqlforge train preflight --method lora
sqlforge train preflight --method qlora
```

### Training Configuration Validation
Validate configuration file integrity and target module resolution:
```bash
sqlforge train validate --config configs/experiments.yaml
```

### Dry-Run Training
Execute end-to-end data preparation, tokenization, loss masking, and mock training without GPU compute or downloading large weights:
```bash
sqlforge train run --method lora --dry-run
```

### Live Training Execution
Explicitly initiate supervised fine-tuning (guarded by required `--execute` flag):
```bash
sqlforge train run --method lora --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct --execute
sqlforge train run --method qlora --base-model Qwen/Qwen2.5-Coder-7B-Instruct --execute
```

### Checkpoint Inspection
Inspect saved adapter checkpoints and metadata:
```bash
sqlforge train inspect --run-id sft-lora-qwen-1.5b-exp03
```

---

## 8. Evaluation Handoff

Once fine-tuning completes, the adapter is evaluated using the existing baseline evaluation pipeline (`Step 5`):

```python
from sqlforge.training.evaluation_handoff import prepare_evaluation_runner, create_evaluation_config
from sqlforge.pipeline.baseline_harness import BaselinePipelineHarness

# Load fine-tuned adapter into LocalHFModelRunner
runner = prepare_evaluation_runner(
    base_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    adapter_path="artifacts/checkpoints/sft-lora-qwen-1.5b-exp03/final_adapter",
)

# Create isolated evaluation configuration targeting spider:dev
eval_config = create_evaluation_config(
    experiment_name="EXP-03-lora-eval",
    base_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    adapter_path="artifacts/checkpoints/sft-lora-qwen-1.5b-exp03/final_adapter",
    eval_split="dev",
)
```

This guarantees that baseline and fine-tuned models are evaluated under identical execution sandbox, prompt serialization, and statistical scoring conditions.
