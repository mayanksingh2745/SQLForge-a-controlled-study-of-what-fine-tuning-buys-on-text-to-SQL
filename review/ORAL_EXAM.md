# SQLForge Research Methodology Oral Exam & Defense

This document records the defense of the core experimental methodology, statistical choices, leakage controls, and operational trade-offs established in the pre-Step-2 remediation.

---

## Question 1: Statistical Equivalence vs. Null Hypothesis Significance Testing
**Examiner Question:**
> *In Hypotheses 1 and 2, why is demonstrating "no statistically significant difference" ($p \ge 0.05$) between LoRA and QLoRA (or fine-tuning vs. GPT-4o mini) insufficient to claim parity or equivalence?*

**Candidate Defense:**
In classical Null Hypothesis Significance Testing (NHST), the null hypothesis posits zero difference ($H_0: \theta_1 - \theta_2 = 0$). Obtaining $p \ge 0.05$ simply indicates that the data do not provide sufficient statistical evidence to reject $H_0$. This failure to reject can easily occur due to low statistical power, high sample variance, or insufficient sample size—not because the models are genuinely equivalent. Concluding parity from $p \ge 0.05$ is the fallacy of *affirming the null*.

To rigorously claim equivalence, SQLForge adopts the **Two One-Sided Tests (TOST)** equivalence framework:
1. We pre-specify an explicit practical equivalence bound $[-\delta, +\delta]$ ($\delta = 2.5\%$ for frontier parity, $\delta = 2.0\%$ for LoRA vs. QLoRA).
2. We construct the two-sided $90\%$ bootstrap confidence interval for the paired difference $\Delta \text{EX} = \text{EX}_{\text{treatment}} - \text{EX}_{\text{control}}$.
3. Equivalence / non-inferiority is demonstrated **if and only if** the entire confidence interval lies completely within the equivalence region. If the confidence interval crosses $-\delta$, the outcome is classified as *inconclusive*, preventing unwarranted claims of parity.

---

## Question 2: Denominator Integrity in Evaluation Metrics
**Examiner Question:**
> *How does SQLForge define the denominator for Execution Accuracy (EX), Execution Success Rate (ESR), and Syntax Validity Rate (SVR), and why does this matter for benchmark integrity?*

**Candidate Defense:**
A known pitfall in NLP research is conditioning evaluation metrics on execution success, which silently excludes failed or malformed queries and artificially inflates reported accuracy.

In SQLForge:
* The denominator for all quality metrics is strictly $N$, the total count of evaluation instances in the partition.
* If a model outputs invalid SQL, causes a syntax error, triggers a database timeout (10.0s deadline), produces an unparseable result, or outputs an empty string, that instance is assigned a score of $0$ in the numerator and remains in the denominator $N$.
* Resampling for bootstrap confidence intervals is conducted at the individual question level $i \in \{1, \dots, N\}$, ensuring that error-prone queries contribute directly to uncertainty intervals.

---

## Question 3: Contamination & Leakage Controls
**Examiner Question:**
> *Public academic benchmarks like Spider and BIRD have likely appeared in the pre-training corpora of modern open-weight and proprietary LLMs. How does SQLForge prevent benchmark contamination from confounding experimental claims?*

**Candidate Defense:**
Pre-training corpus contamination is an acknowledged epistemic limitation of open-source LLM research that cannot be completely eliminated retroactively. SQLForge mitigates this through a three-layer defense:
1. **Retrieval-Index Partitioning:** For few-shot retrieval (`EXP-02`), candidate demonstrations are retrieved exclusively from `spider:train`. Evaluation databases, questions, and gold queries are strictly isolated and cryptographically audited to ensure zero overlap.
2. **Untouched Custom Held-Out Schema:** We construct an independent database schema and ~100 novel query pairs never released publicly. While this small sample size cannot prove universal domain invariance, it directly measures *schema-specific transfer resistance* against completely novel table and column names.
3. **Difference-in-Differences Design:** By evaluating both fine-tuned and frontier reference models across the exact same in-domain and out-of-domain partitions, we observe the *relative degradation delta* ($\Delta_{\text{OOD}}$) rather than relying on absolute scores.

---

## Question 4: Disentangling Quantization from Serving Architecture
**Examiner Question:**
> *Why are AWQ and GGUF evaluated separately from serving concurrency benchmarks (vLLM vs. Hugging Face)?*

**Candidate Defense:**
Quantization format and serving engine address orthogonal dimensions of inference efficiency:
* **Quantization Quality Trade-off (`EXP-08`):** Measures accuracy degradation ($\Delta \text{EX}$) and memory footprint reduction resulting from weight quantization (4-bit AWQ on GPU vs. 4-bit GGUF on CPU/edge vs. 16-bit unquantized BF16).
* **Serving Concurrency Trade-off (`EXP-10`):** Measures scheduling efficiency, PagedAttention, and continuous batching under multi-client load ($C \in \{1, 8, 32\}$).
Conflating these factors (e.g. running GGUF on a vLLM server or testing AWQ without concurrency) would obscure whether latency differences arise from weight compression or kernel batching dynamics. We isolate each variable independently.
