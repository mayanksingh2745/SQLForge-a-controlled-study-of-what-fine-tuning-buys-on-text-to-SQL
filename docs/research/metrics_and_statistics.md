# SQLForge Metrics & Statistical Analysis Specifications

This document defines the mathematical formulations, aggregation methodologies, edge-case protocols, and statistical testing standards for SQLForge.

---

## 1. Primary Quality Metric: Execution Accuracy (EX)

Execution Accuracy is the primary benchmark of text-to-SQL system quality. It measures whether executing the model-generated SQL query against the target database returns the identical result set as the human-verified Gold SQL query.

### Mathematical Formulation

$$\text{EX} = \frac{1}{N} \sum_{i=1}^N \mathcal{E}\left( \text{Exec}(Q_{\text{pred}}^{(i)}, \mathcal{D}_i), \text{Exec}(Q_{\text{gold}}^{(i)}, \mathcal{D}_i) \right)$$

Where:
* $N$: Total number of distinct evaluation examples in the split.
* $Q_{\text{pred}}^{(i)}$: The predicted candidate SQL query for instance $i$.
* $Q_{\text{gold}}^{(i)}$: The reference ground-truth SQL query for instance $i$.
* $\mathcal{D}_i$: The isolated benchmark database instance for example $i$.
* $\text{Exec}(Q, \mathcal{D})$: The tuple $(S, \mathbf{R})$, where $S \in \{\text{SUCCESS}, \text{ERROR}, \text{TIMEOUT}\}$ and $\mathbf{R}$ is the ordered/unordered tabular result set.
* $\mathcal{E}(\mathbf{R}_{\text{pred}}, \mathbf{R}_{\text{gold}}) \in \{0, 1\}$: The result set equivalence indicator function.

### Result Equivalence Rules ($\mathcal{E}$)
Two result sets are evaluated as equivalent ($\mathcal{E} = 1$) if and only if:
1. **Status Match:** Both executions return `SUCCESS`. If $Q_{\text{pred}}$ fails with an error or timeout, $\mathcal{E} = 0$.
2. **Cardinality Match:** Both result sets contain the exact same number of rows and columns.
3. **Multiset Row Equality:** By default, rows are compared as **multisets** (unordered sets with multiplicity), unless the Gold SQL query explicitly includes an `ORDER BY` clause. If `ORDER BY` is present, the row ordering must match strictly.
4. **Floating-Point Tolerance:** Numerical values are compared with an absolute tolerance of $\epsilon = 10^{-4}$ ($|v_{\text{pred}} - v_{\text{gold}}| \le 10^{-4}$).
5. **NULL Handling:** `NULL` matches `NULL` strictly; `NULL` does not match an empty string `""` or `0`.
6. **Column Order Independence:** If column headers match via permutation, column ordering is considered equivalent unless prohibited by the evaluation harness.

### Empty Result Set Protocol
If both $Q_{\text{pred}}$ and $Q_{\text{gold}}$ return an empty result set ($0$ rows):
* An empty result match is flagged as **ambiguous**.
* To prevent degenerate queries (e.g. `SELECT * FROM t WHERE 1=0`) from gaming accuracy, empty matches are credited *only* if $Q_{\text{pred}}$ correctly references the target tables and attributes required by $Q_{\text{gold}}$.

---

## 2. Secondary Quality Metrics

### Valid-SQL Rate (VSR)
Proportion of queries that parse without syntax errors and execute successfully on SQLite, regardless of result correctness:
$$\text{VSR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{Status}(Q_{\text{pred}}^{(i)}) = \text{SUCCESS} \right)$$

### Exact Match Accuracy (EM)
Strict syntactic equivalence between normalized SQL query strings or Abstract Syntax Trees (ASTs):
$$\text{EM} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{AST}(Q_{\text{pred}}^{(i)}) \equiv \text{AST}(Q_{\text{gold}}^{(i)}) \right)$$
*Note:* EM is tracked for historical benchmark continuity but is strictly secondary to EX, as valid SQL queries can achieve identical results through alternative valid syntactic constructions (e.g. JOIN ordering, subquery vs CTE).

### Execution Accuracy by Difficulty Tier
Accuracy broken down across the 4 canonical complexity tiers defined in Spider:
$$\text{EX}_d = \frac{\sum_{i \in \mathcal{I}_d} \mathcal{E}_i}{|\mathcal{I}_d|}, \quad d \in \{\text{Easy}, \text{Medium}, \text{Hard}, \text{Extra}\}$$

### Generalization Delta ($\Delta_{\text{OOD}}$)
The performance drop experienced when moving from in-domain benchmark schemas to the unseen custom held-out schema:
$$\Delta_{\text{OOD}} = \text{EX}_{\text{spider:dev}} - \text{EX}_{\text{held\_out:test}}$$

---

## 3. Computational & Operational Efficiency Metrics

| Metric | Definition | Unit | Measurement Protocol |
| :--- | :--- | :--- | :--- |
| **Peak Allocated VRAM** | Maximum GPU memory explicitly allocated to tensor weights and activations | Gigabytes (GB) | `torch.cuda.max_memory_allocated()` |
| **Peak Reserved VRAM** | Maximum GPU memory held by PyTorch caching allocator | Gigabytes (GB) | `torch.cuda.max_memory_reserved()` |
| **Total GPU Hours** | Cumulative wall-clock GPU processing time across training and evaluation runs | Hours | Hardware execution timer |
| **Token Throughput (TPS)** | Total completion tokens generated divided by inference duration | Tokens/sec | $\frac{\sum \text{completion\_tokens}}{\sum \text{latency\_sec}}$ |
| **Query Throughput (QPS)** | Total completed queries divided by elapsed wall-clock serving time under load | Queries/sec | $\frac{N_{\text{queries}}}{\text{WallClockSeconds}}$ |
| **Median Latency (p50)** | 50th percentile of per-example generation time | Milliseconds (ms) | Sorted empirical distribution |
| **Tail Latency (p95)** | 95th percentile of per-example generation time | Milliseconds (ms) | Sorted empirical distribution |
| **Cost per 1k Queries** | Amortized inference cost based on token counts (API) or cloud GPU rental costs ($/hr) | USD ($) | Documented pricing assumptions |

---

## 4. Statistical Rigor & Hypothesis Testing Protocol

### Paired Comparison Protocol
Because all comparison models are evaluated on the exact same questions and databases, **unpaired tests are strictly prohibited**. All comparative analyses utilize paired observations:
$$d_i = \mathcal{E}_i(\text{Model A}) - \mathcal{E}_i(\text{Model B}) \in \{-1, 0, +1\}$$

### Non-Parametric Bootstrap Confidence Intervals
1. For every reported metric, draw $B = 1,000$ bootstrap replicates with replacement from the $N$ evaluation instances.
2. For comparative claims ($\Delta \text{EX} = \text{EX}_A - \text{EX}_B$), compute the paired difference on each bootstrap sample $b \in \{1, \dots, B\}$.
3. Report the **95% Bootstrap Percentile Interval** $[\theta_{0.025}^*, \theta_{0.975}^*]$.
4. A performance difference is declared statistically significant at the $95\%$ confidence level if and only if the $95\%$ bootstrap confidence interval for $\Delta \text{EX}$ **excludes zero**.

### Hypothesis Testing: McNemar’s Test
For binary paired outcomes (correct vs. incorrect execution), McNemar’s test is applied with continuity correction:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
Where:
* $b$: Number of instances where Model A is correct and Model B is incorrect.
* $c$: Number of instances where Model B is correct and Model A is incorrect.
* Significance threshold: $\alpha = 0.05$.

### Multiple Comparisons Correction
When testing multiple hyperparameter ranks ($r \in \{8, 16, 32, 64\}$) or data scale fractions, adjust $p$-values using the **Benjamini-Hochberg (FDR)** procedure to control false discovery rates.

### Statistical vs. Practical Significance
* **Heuristic Context:** While a 3.0 percentage-point gain is often considered a useful engineering planning heuristic, **it is not a universal statistical law**.
* On a small sample ($N = 100$ custom held-out), a 3% difference (3 examples) has wide uncertainty ($95\%\text{ CI} \approx [\pm 4.5\%]$) and is statistically indistinguishable from noise.
* Conversely, on Spider dev ($N = 1,034$), a 2.5% gain represents 26 additional correct database queries and can be highly statistically significant ($p < 0.01$).
* Every reported gain must state both the empirical difference $\Delta \text{EX}$ and its exact 95% confidence bounds.

### Independence Rule
> [!CRITICAL]
> In self-consistency decoding experiments (where $k=5$ candidate SQL queries are generated per question), **repeated generation samples on the same prompt are not independent evaluation instances**. The evaluation unit $N$ remains the number of unique questions ($N = 1,034$), never $5 \times 1,034$.
