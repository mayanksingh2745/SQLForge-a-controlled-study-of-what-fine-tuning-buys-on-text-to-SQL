# SQLForge Empirical Research Plan

## 1. Research Objectives & Motivation

Text-to-SQL promises intuitive, natural language access to structured relational databases. While large commercial frontier models (e.g., GPT-4o, Claude 3.5 Sonnet) achieve high execution accuracy on academic benchmarks, their operational costs, high tail latency, non-deterministic API shifts, and potential data privacy risks make them difficult to deploy in air-gapped enterprise environments.

**SQLForge** conducts a controlled, empirical investigation into:
> *What exact capabilities does parameter-efficient fine-tuning (LoRA / QLoRA) purchase on small open-weight language models (1.5B–8B parameters) compared to prompt-engineered frontier APIs?*

Specifically, does fine-tuning:
1. Improve schema grounding and column resolution?
2. Eliminate common syntax and dialect errors?
3. Generalize to completely unseen schemas, or merely overfit to training database structures?
4. Deliver Pareto-superior cost, VRAM, and latency profiles?

---

## 2. Core Research Hypotheses

* **Hypothesis 1 (In-Domain Equivalence):** A 7B parameter open-weight model fine-tuned via LoRA/QLoRA on Spider train will achieve demonstrated non-inferior execution accuracy (within a $\pm 2.5\%$ TOST equivalence bound) to zero-shot GPT-4o mini on the Spider development set, while reducing inference costs by $>80\%$.
* **Hypothesis 2 (QLoRA Efficiency without Degradation):** 4-bit NormalFloat QLoRA fine-tuning will demonstrate non-inferior execution accuracy (within $2.0\%$ margin) compared to 16-bit LoRA while reducing peak training VRAM by $\ge 40\%$.
* **Hypothesis 3 (LoRA Rank Saturation):** Increasing LoRA rank $r$ beyond $r=32$ across all linear transformer projection layers will yield diminishing execution accuracy gains ($<0.5\%$ increase) while substantially increasing adapter size and checkpoint overhead.
* **Hypothesis 4 (Data Scaling Law):** Execution accuracy will scale logarithmically with respect to training set size ($10\% \to 25\% \to 50\% \to 100\%$), with execution-validated synthetic examples showing comparable marginal efficiency to human-annotated examples.
* **Hypothesis 5 (Generalization Penalty):** Fine-tuned models will exhibit a significantly larger performance drop ($\ge 10\%$ drop) when evaluated on an independently designed, unseen schema (SQLForge Custom Held-Out, $N \approx 100$) than frontier reference models.

---

## 3. Experimental Matrix & Independent Variables

The detailed 10-experiment staged matrix is codified in [`docs/research/experiment_matrix.md`](research/experiment_matrix.md) (`EXP-01` through `EXP-10`).

| Dimension | Factor Levels / Conditions | Rationale |
| :--- | :--- | :--- |
| **Model Architectures** | Qwen2.5-Coder-1.5B-Instruct<br>Qwen2.5-Coder-7B-Instruct<br>GPT-4o mini (Frontier Reference) | Covers small edge-tier (1.5B), standard workstation tier (7B), and commercial reference. |
| **Adaptation Strategies** | 1. Zero-shot Prompting<br>2. Few-shot Prompting (k=1, 3, 5 with BM25 vs Dense)<br>3. 16-bit LoRA SFT<br>4. 4-bit QLoRA SFT | Directly isolates the marginal contribution of fine-tuning over in-context learning. |
| **LoRA Hyperparameters** | Rank $r \in \{8, 16, 32, 64\}$<br>Scaling $\alpha \in \{2 \times r\}$<br>Target modules: All Linear | Disentangles parameter capacity from adaptation efficiency. |
| **Training Data Scale** | $10\%, 25\%, 50\%, 100\%$ of Spider Train (~700 to 7,000 examples, nested subsets) | Establishes empirical scaling curves for text-to-SQL sample efficiency. |
| **Data Provenance** | 1. Human-annotated only (Spider)<br>2. Execution-verified synthetic only<br>3. 50/50 Human + Synthetic blend | Measures the empirical utility and failure modes of LLM-generated training data. |
| **Schema Representation** | 1. Detailed DDL (types, PK/FK)<br>2. Compact pipe-delimited schema<br>3. JSON structured schema | Tests sensitivity of fine-tuned vs prompted models to input prompt formatting. |

---

## 4. Evaluation Benchmarks & Data Partitions

1. **Spider Benchmark (Yu et al., 2018):**
   - Train split: 7,000 examples across 140 databases.
   - Dev split: 1,034 examples across 20 distinct, unseen databases.
2. **BIRD Benchmark Subset (Li et al., 2023):**
   - Mini-dev split: 500 examples featuring dirty values and external evidence.
3. **SQLForge Held-Out Custom Schema:**
   - Approximately 100 carefully authored queries on an independent schema with multi-join, temporal, and aggregation structures to test schema memorization vs generalization.

---

## 5. Metrics & Statistical Protocol

### Primary Quality Metrics
* **Execution Accuracy (EX):** Unordered multiset row matching with floating-point tolerance ($\epsilon = 10^{-4}$) and strict denominator $N$.
* **Syntax Validity Rate (SVR):** Proportion of candidate queries that are syntactically valid SQL (AST parse success).
* **Execution Success Rate (ESR):** Proportion of queries executing on SQLite without runtime error or timeout (`Status == SUCCESS`).
* **Valid-SQL Rate (VSR):** Historical benchmark alias for Execution Success Rate (ESR).
* **Exact Match (EM):** AST-level exact equivalence after whitespace and keyword normalization.


### Operational & Efficiency Metrics
* **Latency Profile:** p50 and p95 inference generation latency (milliseconds).
* **Throughput:** Tokens per second (TPS).
* **Peak VRAM:** Max allocated GPU memory in gigabytes during training and inference.
* **Amortized Cost:** Estimated dollar cost per 1,000 evaluated queries.

### Statistical Significance Rigor
* Every reported metric is accompanied by a **$95\%$ Non-Parametric Bootstrap Confidence Interval** computed over $B = 1,000$ bootstrap iterations.
* Comparative claims between models are evaluated using **McNemar's test** for paired binary classification outcomes, with significance threshold $\alpha = 0.05$.

---

## 6. Contamination & Leakage Prevention

To ensure strict scientific integrity:
1. **N-gram Overlap Audit:** Calculate 8-gram query overlap between Spider train and test/held-out sets. Any exact matches are flagged and quarantined.
2. **Database Isolation:** Training schemas and evaluation schemas are strictly disjoint.
3. **Time-stamped Synthetic Manifests:** Synthetic examples are generated exclusively from training schemas, never from evaluation databases.

---

## 7. Threats to Validity & Limitations

* **SQLite Benchmark Bias:** Both Spider and BIRD primarily utilize SQLite engines; findings may not directly transfer to high-concurrency analytical warehouses (Snowflake, BigQuery).
* **Execution Equivalence Collisions:** An erroneous query can coincidentally return the same result set as the gold query on small databases (false positive EX).
* **Hardware Sensitivity:** Tail latency and throughput depend on kernel implementations (e.g. FlashAttention-2, vLLM optimizations) and GPU memory bandwidth.
