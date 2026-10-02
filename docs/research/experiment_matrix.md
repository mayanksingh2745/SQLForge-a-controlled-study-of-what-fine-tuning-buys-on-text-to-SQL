# SQLForge Experiment Matrix & Execution Specifications

This document defines the structured, staged experimental matrix for SQLForge. To manage compute budgets and avoid factorial explosion, experiments progress through three sequential stages: **Stage I (Pilot Runs)**, **Stage II (Core Controlled Experiments)**, and **Stage III (Ablations & Serving Optimization)**.

---

## 1. Staged Execution Plan

```text
[Stage I: Pilots] ──► [Stage II: Core Controlled] ──► [Stage III: Optimization & Serving]
├── Exp 01: Pilot Baselines  ├── Exp 03: LoRA vs QLoRA    ├── Exp 07: Schema Representations
└── Exp 02: Pipeline Sanity  ├── Exp 04: LoRA Rank Sweep  ├── Exp 08: Post-Training Quantization
                              ├── Exp 05: Data Scaling Law ├── Exp 09: Self-Consistency Decoding
                              └── Exp 06: Human vs Synth   └── Exp 10: vLLM vs HF Concurrency
```

---

## 2. Detailed Experiment Specifications

### EXP-01: Baseline Prompting Benchmark (Pilot & Reference)
* **Experiment ID:** `EXP-01-BASELINES`
* **Research Question:** What is the zero-shot and few-shot text-to-SQL accuracy baseline of open-weight models compared to a frontier reference?
* **Control Group:** `Qwen2.5-Coder-1.5B-Instruct` (Zero-shot DDL).
* **Treatment Groups:**
  1. `Qwen2.5-Coder-7B-Instruct` (Zero-shot DDL).
  2. `Qwen2.5-Coder-7B-Instruct` (3-shot BM25 RAG).
  3. `gpt-4o-mini-2024-07-18` (Zero-shot & 3-shot BM25 RAG).
* **Primary Metric:** Execution Accuracy (EX).
* **Secondary Metrics:** Valid-SQL Rate (VSR), Exact Match (EM), p95 latency.
* **Evaluation Split:** `spider:dev` ($N = 1,034$) and `held_out_custom:test` ($N \approx 100$).
* **Parameters:** $T = 0.0$, max tokens = 512, DDL schema format.
* **Hardware & Dependencies:** CPU or single GPU with $\ge 8\text{ GB}$ VRAM; `openai` SDK for API reference.
* **Compute Cost Category:** Low (< $5.00 API cost; ~1 GPU hour).
* **Dependencies:** Step 0 repository foundation.
* **Repetition Strategy:** Single deterministic greedy pass ($T = 0.0$).
* **Completion Criteria:** Complete `generations.jsonl` and `metrics.json` recorded for all 4 model/prompt conditions.

---

### EXP-02: Pipeline End-to-End Sanity & Smoke Test
* **Experiment ID:** `EXP-02-PILOT-SFT`
* **Research Question:** Does the training, checkpointing, and evaluation harness execute end-to-end without memory leaks or crashes on a small model?
* **Control Group:** `Qwen2.5-Coder-1.5B-Instruct` un-adapted base.
* **Treatment Group:** `Qwen2.5-Coder-1.5B-Instruct` fine-tuned on a 500-example Spider train subset for 1 epoch.
* **Primary Metric:** Pipeline completion without error; training loss monotonically decreasing.
* **Secondary Metrics:** Initial EX delta on 100 dev examples.
* **Evaluation Split:** 100-sample slice of `spider:dev`.
* **Parameters:** $r=8, \alpha=16$, lr $= 2 \times 10^{-4}$, batch size $= 8$, epochs $= 1$.
* **Hardware & Dependencies:** Single GPU ($\ge 8\text{ GB}$ VRAM) or WSL2.
* **Compute Cost Category:** Low (< 0.5 GPU hours).
* **Dependencies:** `EXP-01-BASELINES`.
* **Repetition Strategy:** Single run (seed 42).
* **Completion Criteria:** Checkpoint saved, loaded, and evaluated successfully.

---

### EXP-03: LoRA vs. QLoRA Precision & Memory Comparison
* **Experiment ID:** `EXP-03-LORA-VS-QLORA`
* **Research Question:** How do 16-bit LoRA and 4-bit NF4 QLoRA compare in accuracy, peak training VRAM, and training speed?
* **Control Group:** 16-bit LoRA SFT on `Qwen2.5-Coder-7B-Instruct`.
* **Treatment Group:** 4-bit NormalFloat QLoRA SFT on `Qwen2.5-Coder-7B-Instruct`.
* **Primary Metric:** Execution Accuracy (EX) on Spider dev.
* **Secondary Metrics:** Peak allocated VRAM (GB), step time (ms), Valid-SQL Rate.
* **Evaluation Split:** `spider:dev` ($N = 1,034$) and `held_out_custom:test` ($N \approx 100$).
* **Parameters:** 3 epochs, $r=16, \alpha=32$, lr $= 2 \times 10^{-4}$, target modules: all linear.
* **Hardware & Dependencies:** Single NVIDIA GPU with $\ge 16\text{ GB}$ VRAM (or $24\text{ GB}$ for 16-bit LoRA); Linux/WSL2; `peft`, `bitsandbytes`.
* **Compute Cost Category:** Medium (~6 GPU hours total).
* **Dependencies:** `EXP-02-PILOT-SFT`.
* **Repetition Strategy:** 3 random seeds ($S \in \{42, 43, 44\}$).
* **Completion Criteria:** Paired bootstrap CI computed for $\Delta \text{EX} = \text{EX}_{\text{LoRA}} - \text{EX}_{\text{QLoRA}}$.

---

### EXP-04: LoRA Intrinsic Rank Sweep ($r \in \{8, 16, 32, 64\}$)
* **Experiment ID:** `EXP-04-RANK-SWEEP`
* **Research Question:** At what intrinsic rank does execution accuracy saturate on text-to-SQL?
* **Control Group:** Rank $r=16$ (from EXP-03).
* **Treatment Groups:** Ranks $r=8, r=32, r=64$ (with $\alpha = 2 \times r$).
* **Primary Metric:** Execution Accuracy (EX).
* **Secondary Metrics:** Adapter checkpoint size (MB), inference latency.
* **Evaluation Split:** `spider:dev` ($N = 1,034$).
* **Parameters:** QLoRA 4-bit, base model Qwen2.5-Coder-7B, 3 epochs, target modules: all linear.
* **Hardware & Dependencies:** Single GPU ($\ge 16\text{ GB}$ VRAM); Linux/WSL2.
* **Compute Cost Category:** Medium (~8 GPU hours total).
* **Dependencies:** `EXP-03-LORA-VS-QLORA`.
* **Repetition Strategy:** Seed 42 for all ranks; top-2 ranks replicated on seeds 43 and 44.
* **Completion Criteria:** Accuracy vs. Rank Pareto curve plotted with 95% bootstrap CIs.

---

### EXP-05: Training Data Scaling Law ($N \in \{500, 1000, 2000, 5000, 7000\}$)
* **Experiment ID:** `EXP-05-DATA-SCALING`
* **Research Question:** What is the empirical scaling curve between training sample count and text-to-SQL execution accuracy?
* **Control Group:** $N = 7,000$ (100% eligible Spider train).
* **Treatment Groups:** Subsets $N \in \{500, 1000, 2000, 5000\}$ sampled with fixed stratified seeds.
* **Primary Metric:** Execution Accuracy (EX).
* **Secondary Metrics:** Valid-SQL Rate (VSR), Exact Match (EM).
* **Evaluation Split:** `spider:dev` ($N = 1,034$).
* **Parameters:** Optimal rank from EXP-04, 3 epochs, QLoRA 4-bit.
* **Hardware & Dependencies:** Single GPU ($\ge 16\text{ GB}$ VRAM).
* **Compute Cost Category:** High (~12 GPU hours total).
* **Dependencies:** `EXP-04-RANK-SWEEP`.
* **Repetition Strategy:** 3 random seeds per data scale ($5 \times 3 = 15$ runs).
* **Completion Criteria:** Log-linear and power-law scaling curve fits reported with $R^2$ goodness-of-fit.

---

### EXP-06: Human-Annotated vs. Execution-Filtered Synthetic Data
* **Experiment ID:** `EXP-06-SYNTHETIC-DATA`
* **Research Question:** Does execution-validated synthetic data match human sample efficiency, and does a 50/50 blend outperform pure human data?
* **Control Group:** 3,000 human-annotated Spider train examples.
* **Treatment Groups:**
  1. 3,000 execution-validated synthetic examples.
  2. 1,500 human + 1,500 synthetic examples (50/50 blend).
* **Primary Metric:** Execution Accuracy (EX) on Spider dev and BIRD mini-dev.
* **Secondary Metrics:** Error category taxonomy shift.
* **Evaluation Split:** `spider:dev` ($N = 1,034$) and `bird_mini:dev` ($N = 500$).
* **Parameters:** Standardized $N = 3,000$ training examples, 3 epochs, optimal rank from EXP-04.
* **Hardware & Dependencies:** Single GPU ($\ge 16\text{ GB}$ VRAM).
* **Compute Cost Category:** Medium (~6 GPU hours).
* **Dependencies:** `EXP-05-DATA-SCALING`.
* **Repetition Strategy:** 3 random seeds ($S \in \{42, 43, 44\}$).
* **Completion Criteria:** Comparative error distribution and accuracy table completed.

---

### EXP-07: Schema Representation & Prompt Format Ablation
* **Experiment ID:** `EXP-07-SCHEMA-FORMAT`
* **Research Question:** How sensitive is the fine-tuned model vs. the prompted frontier model to schema format variations?
* **Control Group:** Standard DDL representation (table DDL with types, PK/FK).
* **Treatment Groups:**
  1. Compact pipe-delimited schema (`Table: col1 (type) | col2 (type)`).
  2. Structured JSON schema.
* **Primary Metric:** Execution Accuracy (EX).
* **Secondary Metrics:** Total prompt tokens consumed, p95 generation latency.
* **Evaluation Split:** `spider:dev` ($N = 1,034$) and `held_out_custom:test` ($N \approx 100$).
* **Parameters:** Greedy decoding, evaluated on best fine-tuned 7B model and GPT-4o mini.
* **Hardware & Dependencies:** Inference-only (CPU or GPU).
* **Compute Cost Category:** Low (~2 GPU hours or < $5.00 API).
* **Dependencies:** `EXP-03-LORA-VS-QLORA`.
* **Repetition Strategy:** Deterministic greedy evaluation ($T = 0.0$).
* **Completion Criteria:** Token efficiency vs. accuracy trade-off table recorded.

---

### EXP-08: Post-Training Quantization (AWQ vs. GGUF)
* **Experiment ID:** `EXP-08-QUANTIZATION`
* **Research Question:** How do AWQ (GPU) and GGUF (CPU/edge) 4-bit quantizations compare with unquantized BF16 weights in accuracy and latency?
* **Control Group:** Merged fine-tuned model in full 16-bit BF16 precision.
* **Treatment Groups:**
  1. AWQ 4-bit quantization (GPU).
  2. GGUF Q4_K_M quantization (CPU/GPU).
* **Primary Metric:** Execution Accuracy degradation ($\Delta \text{EX} = \text{EX}_{\text{BF16}} - \text{EX}_{\text{quant}}$).
* **Secondary Metrics:** p50/p95 latency, model disk size (GB), active memory footprint (GB).
* **Evaluation Split:** `spider:dev` ($N = 1,034$).
* **Parameters:** Pinned calibration dataset (128 Spider train samples).
* **Hardware & Dependencies:** GPU for AWQ (`autoawq`); CPU/GPU for GGUF (`llama-cpp-python`).
* **Compute Cost Category:** Low (~2 hours).
* **Dependencies:** `EXP-03-LORA-VS-QLORA`.
* **Repetition Strategy:** Single deterministic pass.
* **Completion Criteria:** Pareto plot (Accuracy vs. Memory vs. Latency) generated.

---

### EXP-09: Decoding Strategy (Greedy vs. Execution-Based Self-Consistency)
* **Experiment ID:** `EXP-09-SELF-CONSISTENCY`
* **Research Question:** Does execution-based self-consistency ($k=5$ samples, clustered by executed result set) yield significant accuracy improvements over greedy decoding?
* **Control Group:** Greedy decoding ($T = 0.0, k = 1$).
* **Treatment Group:** Sampled decoding ($T = 0.7, k = 5$) with majority-vote result clustering.
* **Primary Metric:** Execution Accuracy (EX).
* **Secondary Metrics:** Total generation cost (5x token multiplier), compute time.
* **Evaluation Split:** `spider:dev` ($N = 1,034$).
* **Parameters:** Best fine-tuned 7B model.
* **Hardware & Dependencies:** Inference GPU.
* **Compute Cost Category:** Low (~3 GPU hours).
* **Dependencies:** `EXP-03-LORA-VS-QLORA`.
* **Repetition Strategy:** Fixed random seed for sampling.
* **Completion Criteria:** Accuracy gain relative to 5x compute budget documented.

---

### EXP-10: Serving Engine & Concurrency Benchmarking (vLLM vs. HF)
* **Experiment ID:** `EXP-10-SERVING-BENCH`
* **Research Question:** How does vLLM compare with standard Hugging Face pipelines across concurrent request loads (1, 8, 32 concurrent clients)?
* **Control Group:** Hugging Face `pipeline("text-generation")` with standard sequential batching.
* **Treatment Group:** `vLLM` server with continuous batching and PagedAttention.
* **Primary Metric:** Throughput in Tokens Per Second (TPS) and Queries Per Second (QPS).
* **Secondary Metrics:** p50 and p95 request latency (ms).
* **Evaluation Split:** Synthetic client workload using 200 random prompts from `spider:dev`.
* **Parameters:** Concurrency levels $c \in \{1, 8, 32\}$.
* **Hardware & Dependencies:** Linux GPU environment with $\ge 16\text{ GB}$ VRAM; `vllm`.
* **Compute Cost Category:** Low (~2 GPU hours).
* **Dependencies:** `EXP-08-QUANTIZATION`.
* **Repetition Strategy:** 3 repeated benchmark trials per concurrency level with 10-second warm-up.
* **Completion Criteria:** Concurrency vs. Throughput/Latency benchmark curves recorded.
