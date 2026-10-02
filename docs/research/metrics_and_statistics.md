# SQLForge Metrics & Statistical Analysis Specifications

This document defines the mathematical formulations, aggregation methodologies, edge-case protocols, and statistical testing standards for SQLForge.

---

## 1. Metric Hierarchy, Terminology & Explicit Denominators

To ensure scientific rigor and avoid conflating syntactic parsing with database executability or semantic correctness, SQLForge strictly distinguishes between five core evaluation metrics.

### Denominator Rule
**The denominator for all evaluation metrics is strictly $N$, the total number of evaluation instances in the benchmark split.**
Any model prediction that fails to parse, crashes during execution, exceeds the wall-clock deadline, produces an empty or ambiguous result, or generates an empty prediction receives a score of $0$ in the numerator and remains in the denominator $N$. Resampling for confidence intervals is performed at the individual question/example level $i \in \{1, \dots, N\}$.

| Metric Name | Symbol | Mathematical Formulation | Definition & Measurement Scope |
| :--- | :---: | :--- | :--- |
| **Syntax Validity Rate** | $\text{SVR}$ | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{AST\_Parse}(Q_{\text{pred}}^{(i)}) = \text{SUCCESS})$ | Proportion of candidate queries that are syntactically valid SQL (parseable into an AST), regardless of whether referenced tables/columns exist in the database. |
| **Execution Success Rate** | $\text{ESR}$ | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Status}(Q_{\text{pred}}^{(i)}, \mathcal{D}_i) = \text{SUCCESS})$ | Proportion of candidate queries that execute against the target SQLite database without any runtime error, syntax error, or timeout. |
| **Valid-SQL Rate** | $\text{VSR}$ | $\text{VSR} \equiv \text{ESR}$ | Historical benchmark alias for Execution Success Rate (ESR). Measures database executability, not merely syntactic validity. |
| **Exact Match Accuracy** | $\text{EM}$ | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Norm\_AST}(Q_{\text{pred}}^{(i)}) \equiv \text{Norm\_AST}(Q_{\text{gold}}^{(i)}))$ | Strict syntactic equivalence between normalized ASTs (case, whitespace, and alias invariant). Tracked for continuity but secondary to execution. |
| **Execution Accuracy** | $\text{EX}$ | $\frac{1}{N} \sum_{i=1}^N \mathcal{E}\left( \text{Exec}(Q_{\text{pred}}^{(i)}, \mathcal{D}_i), \text{Exec}(Q_{\text{gold}}^{(i)}, \mathcal{D}_i) \right)$ | **Primary benchmark metric.** Proportion of queries where predicted SQL execution returns the accepted tabular result set under equivalence rules $\mathcal{E}$. |

---

## 2. Primary Quality Metric: Execution Accuracy (EX)

Execution Accuracy measures whether executing the candidate SQL query against the target database returns the identical result set as the human-verified Gold SQL query.

### Result Equivalence Rules ($\mathcal{E}$)
Two execution results are evaluated as equivalent ($\mathcal{E} = 1$) if and only if:
1. **Status Match:** Both executions return `SUCCESS`. If $Q_{\text{pred}}$ fails with syntax error, table not found, divide-by-zero, or timeout, $\mathcal{E} = 0$.
2. **Cardinality Match:** Both result sets contain the exact same number of rows and columns.
3. **Column Alignment:** Columns are matched positionally by projection list index ($j \in \{1, \dots, C\}$), following standard Spider/BIRD benchmark conventions.
4. **Duplicate Rows (Multiset Semantics):** Result sets are compared as **multisets** (bags with element multiplicity). A query returning duplicate rows must match the exact row count and multiplicity of the gold result set.
5. **Unordered Matching:** Rows are matched independently of row order *unless* the Gold SQL query contains an explicit `ORDER BY` clause.
6. **ORDER BY Sensitivity:** If the Gold SQL query includes an `ORDER BY` clause, row sequence order is evaluated strictly ($r_k^{\text{pred}} == r_k^{\text{gold}}$ for all $k$).
7. **Floating-Point Tolerance:** Numerical values are compared with an absolute tolerance of $\epsilon = 10^{-4}$ ($|v_{\text{pred}} - v_{\text{gold}}| \le 10^{-4}$). Integer and real numbers with identical numerical value (e.g. `1` and `1.0`) are evaluated as equal.
8. **NULL Semantics:** `NULL` matches `NULL` strictly; `NULL` does not match an empty string `""` or numeric `0`.
9. **Supported Data Types:** INTEGER, REAL, TEXT, BLOB, and NULL are supported.

### Handling Ambiguous & Unsupported Comparisons
* **Empty Result Set Protocol:** If both $Q_{\text{pred}}$ and $Q_{\text{gold}}$ return an empty result set ($0$ rows), the comparison is flagged as `AMBIGUOUS_EMPTY_SET`. To prevent degenerate queries (e.g. `SELECT * FROM t WHERE 1=0`) from gaming accuracy, empty matches are credited *only* if $Q_{\text{pred}}$ correctly references the target tables and attributes required by $Q_{\text{gold}}$.
* **Unsupported Comparisons:** If a query invokes unsupported SQLite extensions or unparseable BLOB literals, it is flagged as `UNSUPPORTED_COMPARISON`, assigned $\mathcal{E} = 0$, and logged in `eval_anomalies.json`. Unsupported comparisons are never silently counted as correct.

### Benchmark Compatibility Notice
> [!NOTE]
> The SQLForge execution comparator is an internal research harness designed for controlled empirical experiments. It is not currently certified as 100% bitwise equivalent to the official Spider `test-suite-evaluator` (which uses database-specific test-suite augmentation) or the official BIRD benchmark harness. Full benchmark harness calibration and parity verification will be performed and documented during Step 9 and Step 10.

---

## 3. Sub-Category & Generalization Metrics

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
