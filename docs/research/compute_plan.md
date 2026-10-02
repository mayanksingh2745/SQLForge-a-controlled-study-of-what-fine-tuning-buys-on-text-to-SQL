# SQLForge Compute Plan, Hardware Profiling & Feasibility Gates

This document defines the hardware infrastructure requirements, memory budgeting, pre-flight safety gates, and compute contingency plans for SQLForge.

---

## 1. Verified Local Environment vs. Future Research Infrastructure

Based on the environment inspection performed in Step 0:
* **Host Operating System:** Windows 10 Pro (64-bit, build 10.0.19045).
* **Python Runtime:** Python 3.13.2.
* **Host CPU / Memory:** 4 logical cores, 7.9 GB total RAM (~1.0 GB free memory).
* **Host Disk Space:** Approximately **5.61 GB free storage** on drive `C:`.
* **Host Acceleration:** **No discrete NVIDIA GPU detected** (`nvidia-smi` not found, CPU-only locally).

### Target Execution Environments by Experiment Type

| Experiment ID | Task Description | Minimum Compute Tier | Recommended Environment | Estimated Duration |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline Prompting (API & Small Model) | CPU / Cloud API | Local CPU (1.5B) + OpenAI API | ~1.5 hours |
| **EXP-02** | SFT Pipeline Smoke Test | Entry GPU / WSL2 | Local CPU (dry-run) or remote GPU | ~30 minutes |
| **EXP-03** | LoRA (16-bit) vs QLoRA (4-bit) | Dedicated GPU ($\ge 16\text{ GB}$ VRAM) | Cloud GPU instance (A10G or L4) | ~6 hours |
| **EXP-04** | LoRA Rank Sweep ($r=8, 16, 32, 64$) | Dedicated GPU ($\ge 16\text{ GB}$ VRAM) | Cloud GPU instance (A10G or L4) | ~8 hours |
| **EXP-05** | Training Data Scaling Law ($N \le 7k$) | Dedicated GPU ($\ge 16\text{ GB}$ VRAM) | Cloud GPU instance (A10G or L4) | ~12 hours |
| **EXP-06** | Human vs Synthetic Data ($N = 3k$) | Dedicated GPU ($\ge 16\text{ GB}$ VRAM) | Cloud GPU instance (A10G or L4) | ~6 hours |
| **EXP-07** | Schema Representation Evaluation | CPU / Inference GPU | Local CPU or remote GPU | ~2 hours |
| **EXP-08** | Post-Training Quantization (AWQ/GGUF) | Hybrid (GPU for AWQ; CPU for GGUF) | Cloud GPU + Local CPU | ~3 hours |
| **EXP-09** | Execution Self-Consistency ($5\times$ samples) | Inference GPU | Cloud GPU instance | ~4 hours |
| **EXP-10** | vLLM Serving & Concurrency Benchmarking | Linux GPU ($\ge 16\text{ GB}$ VRAM) | Cloud Linux GPU instance (Ubuntu 22.04) | ~2.5 hours |

---

## 2. Pre-Flight Hardware & Memory Feasibility Gates

Before launching any GPU training or evaluation job:

### Gate 1: VRAM Safety Check
* A pre-flight script queries `torch.cuda.get_device_properties()`.
* If available free VRAM is less than the model requirement (e.g. $< 14\text{ GB}$ for 7B QLoRA, or $< 22\text{ GB}$ for 7B 16-bit LoRA), **the job immediately aborts** with a clear diagnostic message rather than failing mid-run with a CUDA Out-of-Memory (OOM) error.

### Gate 2: Disk Space Ceiling Check
* Free disk space is evaluated before downloading checkpoints or generating synthetic data.
* If free storage on the target partition is $< 3.0\text{ GB}$, the execution is halted.
* Full 7B model base weights are **never saved locally in the repository**. Only PEFT adapter weights (`adapter_model.safetensors`, $< 50\text{ MB}$) are persisted.

### Gate 3: API Budget Guardian
* Total cumulative spending on commercial APIs (OpenAI / Anthropic) is tracked in `artifacts/runs/api_accounting.json`.
* Hard cap: **$50.00 USD**. Once cumulative costs reach $50.00, further API calls raise an explicit `BudgetExceededError`.

---

## 3. Early Stopping Criteria for Technically Invalid Runs

To prevent wasting compute credits on degraded or broken runs:
1. **Loss Divergence Gate:** If training cross-entropy loss returns `NaN` or `Inf`, or exceeds $10.0$ after step 50, training terminates immediately.
2. **Degenerate Generation Gate:** If during validation on the first 50 dev examples, the Valid-SQL Rate (VSR) is $< 10\%$, generation aborts to diagnose prompt formatting or tokenization bugs.
3. **Runaway Timeout Gate:** If more than $5\%$ of queries in a batch trigger the 10.0-second execution timeout, the evaluation halts to inspect database index health.

---

## 4. Benchmark Timing & Latency Profiling Procedures

To obtain reliable latency and throughput numbers:
1. **Warm-Up Phase:** Every inference benchmark runs 10 un-timed warm-up queries to populate the KV cache and warm up GPU clock frequencies.
2. **Repeated Timing:** Measurements are averaged across 3 distinct randomized trials.
3. **Synchronized Clocks:** On CUDA runs, timing must surround `torch.cuda.synchronize()` barriers before and after generation to avoid asynchronous CUDA kernel execution artifacts.

---

## 5. Contingency Plans for Constrained Hardware

If high-end GPU resources are delayed or constrained:
* **Contingency 1 (Scale to 1.5B):** Perform the complete hyperparameter and scaling law matrix on `Qwen2.5-Coder-1.5B-Instruct` (which fits within $8\text{ GB}$ VRAM) to establish comparative trends before validating top configurations on the 7B model.
* **Contingency 2 (Selective Sweep):** Reduce the rank sweep to $r \in \{8, 32\}$ and the data scaling fractions to $\{1000, 5000, \text{max}\}$.
* **Contingency 3 (CPU GGUF Emulation):** Run inference and evaluation benchmarks on CPU using 4-bit GGUF quantization (`llama-cpp-python`).
