# SQLForge EXP-05: Training Data Scaling Protocol

This document formalizes the empirical protocol, sampling methodology, rank dependency, mathematical fitting, and anti-fabrication constraints for **EXP-05-DATA-SCALING**.

---

## 1. Research Question & Hypotheses

### Primary Research Question
What is the functional form and rate of return of text-to-SQL execution accuracy as a function of fine-tuning training dataset size $N$, under controlled base model, QLoRA rank, and schema representations?

### Hypotheses
- **Hypothesis $H_1$ (Sub-linear scaling):** Marginal execution accuracy gains $\frac{d\text{EX}}{dN}$ diminish monotonically as $N$ increases from 500 to 7,000 examples.
- **Hypothesis $H_2$ (Power-law fit quality):** Text-to-SQL performance adheres more closely to a bounded power-law saturation model $\text{EX}(N) = a - b \cdot N^{-c}$ ($b \ge 0, c > 0$) than an unconstrained log-linear model over the observed domain $N \in [500, 7000]$.
- **Hypothesis $H_3$ (Sample threshold for complex syntax):** Subsets with $N \le 1,000$ demonstrate severe degradation on nested subqueries and multi-table joins due to insufficient exposure to rare syntactic primitives.

---

## 2. Experimental Design & Variables

### Controlled Constants
To isolate training data volume as the sole causal variable, all other dimensions are strictly pinned:
- **Base Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (pinned revision)
- **Training Method:** QLoRA (4-bit NormalFloat4 quantization, double quantization enabled)
- **Hyperparameters:**
  - Optimizer: `paged_adamw_32bit`
  - Learning Rate: $2 \times 10^{-4}$ with cosine decay schedule
  - Warmup Ratio: 0.03
  - Effective Batch Size: 16 (per-device 8, gradient accumulation 2)
  - Sequence Length: 2,048 tokens
  - Epochs: 3
  - Schema Format: DDL
  - Loss Masking: Completion-only (prompt tokens masked with `-100`)
- **Evaluation Dataset:** `spider:dev` ($N = 1,034$ examples)
- **Primary Metric:** Execution Accuracy ($\text{EX}$)
- **Secondary Metrics:** Valid-SQL Rate ($\text{VSR}$), Exact Match ($\text{EM}$)

### Manipulated Variables
1. **Training Dataset Size ($N$):** $N \in \{500, 1000, 2000, 5000, 7000\}$
2. **Random Seeds ($S$):** $S \in \{42, 43, 44\}$

Total planned conditions: $5 \text{ sizes} \times 3 \text{ seeds} = 15 \text{ runs}$.

---

## 3. LoRA Rank Dependency & Gating

The LoRA rank $r$ and scaling factor $\alpha = 2r$ must not be hardcoded or arbitrarily assumed. 

### RankSelectionReference Contract
Under SQLForge governance:
1. `source_experiment_id` must be `EXP-04-RANK-SWEEP`.
2. Referenced run manifests must exist, be cryptographically verified, and reflect genuine empirical evaluation evidence.
3. The selected rank must belong to $\{8, 16, 32, 64\}$ and satisfy the documented EXP-04 selection rule (highest seed-42 Spider dev execution accuracy, with parameter count breaking ties).
4. Unresolved or unverified rank references raise `UnresolvedRankSelectionError`.

### Provisional Dry-Run Policy
For CPU-only planning, configuration verification, and dry runs, an explicitly provisional rank of $r = 16$ ($\alpha = 32$) is permitted with:
- `rank_selection_status: provisional_non_empirical`
- `empirical_execution_allowed: false`

Live GPU training (`--execute`) strictly rejects provisional rank configurations.

---

## 4. Deterministic Stratified Data Sampling

Implemented via `NestedStratifiedSpiderSampler` in `src/sqlforge/training/scaling_sampler.py`.

### Dataset Eligibility & Isolation
- Training examples are sampled strictly from `spider:train`.
- Zero examples may be drawn from `spider:dev`, `spider:test`, `bird_mini:dev`, or `held_out_custom:test`.
- Records with missing SQL targets, invalid schemas, duplicate IDs, or unverified provenance are rejected.
- At least 7,000 eligible records are required to construct the $N = 7,000$ condition; fewer records cause an immediate error rather than silent truncation.

### Fair-Share Stratification Policy
1. Stratification groups are partitioned strictly by `database_id` (not inferred SQL difficulty heuristics).
2. For each random seed:
   - Partition records by `database_id`. Sort database IDs lexicographically.
   - Generate a deterministic per-database permutation using cryptographic SHA-256 ranking keys:
     $$\text{key} = \text{SHA256}(\text{seed} \parallel \text{database\_id} \parallel \text{example\_id})$$
   - Interleave database queues using proportional fair-share scheduling: at each step, select from the nonempty database group with the smallest ratio of chosen examples to total group size:
     $$\arg\min_g \left( \frac{\text{selected}_g}{\text{total}_g} \right)$$
     Ties are broken lexicographically by `database_id`.
3. **Nested-Subset Guarantee:** The identical global order is used for every training size within a given seed:
   $$D_{500}^{(s)} \subset D_{1000}^{(s)} \subset D_{2000}^{(s)} \subset D_{5000}^{(s)} \subset D_{7000}^{(s)}$$
   This guarantees that performance differences between $N_i$ and $N_j$ reflect additional data volume, not random sampling variance.

### Sampling Manifests
Every subset generates a cryptographic manifest in `artifacts/data_scaling/exp05/` containing:
- `experiment_id`, `seed`, `sample_size`, `source_dataset_manifest_hash`
- `sampling_algorithm_version`, `selected_example_ids_sha256`, `database_distribution`
- `parent_subset_hash` (null for $N=500$; chained for $N > 500$)

---

## 5. Planned Experiment Matrix

The 15 canonical conditions:

| Run ID | Training Size ($N$) | Seed | Base Model | Method | Epochs | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exp05_n500_s42` | 500 | 42 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n500_s43` | 500 | 43 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n500_s44` | 500 | 44 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n1000_s42` | 1,000 | 42 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n1000_s43` | 1,000 | 43 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n1000_s44` | 1,000 | 44 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n2000_s42` | 2,000 | 42 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n2000_s43` | 2,000 | 43 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n2000_s44` | 2,000 | 44 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n5000_s42` | 5,000 | 42 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n5000_s43` | 5,000 | 43 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n5000_s44` | 5,000 | 44 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n7000_s42` | 7,000 | 42 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n7000_s43` | 7,000 | 43 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |
| `exp05_n7000_s44` | 7,000 | 44 | Qwen2.5-Coder-7B-Instruct | QLoRA | 3 | Planned |

---

## 6. Statistical Scaling Analysis

Implemented in `DataScalingAnalyzer` (`src/sqlforge/training/scaling_analysis.py`).

### Mathematical Formulations
1. **Log-Linear Formulation:**
   $$\text{EX}(N) = a + b \cdot \ln(N)$$
   Fitted via Ordinary Least Squares (OLS) across the five mean size points $(\bar{N}_i, \bar{\text{EX}}_i)$.
2. **Power-Law Saturation Formulation:**
   $$\text{EX}(N) = a - b \cdot N^{-c}$$
   Subject to constraints: $b \ge 0$, $c > 0$, and predicted $\text{EX} \in [0, 1]$.
   Fitted via bounded nonlinear least squares (grid-initialized Levenberg-Marquardt equivalent). If parameters cannot be uniquely identified or yield degenerate curves, the analyzer reports `fit_unavailable` with the root cause.
3. **Goodness-of-Fit:**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
4. **Bootstrap Confidence Intervals:**
   Resampling is conducted at the seed level ($B = 1,000$), preserving within-seed correlation across dataset sizes. Given $S = 3$, seed-level confidence intervals are explicitly designated as exploratory.
5. **No Extrapolation:** Fits are evaluated strictly over $N \in [500, 7000]$. Extrapolation beyond observed scales is prohibited.

---

## 7. Anti-Fabrication Constraints

1. If empirical evaluation results are absent, calling `analyze_exp05` raises `NoEmpiricalScalingDataError`.
2. Dry runs, mock runs, and provisional rank runs generate execution logs, plans, and sampling manifests, but **never** numerical accuracy scores or scaling law curves.
3. Dry-run summaries explicitly record:
   - `status: dry_run`
   - `final_loss: null`
   - `metrics: {}`
   - `empirical_execution_allowed: false`

---

## 8. Current Implementation Status

- **Infrastructure:** **Complete** (All configuration schemas, fair-share samplers, orchestrators, analyzer math, CLI commands, and unit/integration tests verified).
- **Empirical GPU Training:** **Pending** (Requires dedicated $\ge 16$ GB CUDA GPU, ~12 GPU-hours, and verified EXP-04 empirical rank selection artifact).
