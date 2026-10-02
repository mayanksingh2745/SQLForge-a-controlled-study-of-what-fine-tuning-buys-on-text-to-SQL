# SQLForge Research Hypotheses & Experimental Inquiries

This document formalizes the scientific hypotheses, operational inquiries, and falsification criteria for the SQLForge controlled study of text-to-SQL fine-tuning.

---

## 1. Scientific Overview & Inquiry Classification

SQLForge distinguishes between **Confirmatory Hypotheses** (pre-registered directional claims with formal null and alternative hypotheses) and **Exploratory Inquiries** (empirical characterizations of trade-offs, scaling behaviors, and taxonomy distributions without rigid directional commitments).

Negative and mixed results are considered equally valuable to positive results. The objective is to produce measured evidence of the capabilities, limits, and failure modes of fine-tuned language models on structured relational data.

---

## 2. Confirmatory Hypotheses

### Hypothesis 1: Fine-Tuning vs. In-Context Learning (In-Domain Parity)
* **Research Question:** Does supervised fine-tuning of a 7B open-weight model achieve comparable or superior execution accuracy on in-domain benchmark schemas compared to few-shot in-context learning with a frontier commercial API?
* **Hypotheses:**
  - $H_0$: There is no significant difference in Execution Accuracy on Spider dev between a 7B model fine-tuned on Spider train and GPT-4o mini evaluated with 3-shot BM25 retrieved demonstrations ($|\text{EX}_{\text{ft}} - \text{EX}_{\text{api}}| = 0$).
  - $H_1$: The fine-tuned 7B model achieves Execution Accuracy within a $\pm 2.5$ percentage-point equivalence margin of GPT-4o mini ($\text{EX}_{\text{ft}} \ge \text{EX}_{\text{api}} - 2.5\%$).
* **Independent Variable:** Adaptation paradigm (Supervised Fine-Tuning via LoRA vs. In-Context Learning with BM25 retrieval).
* **Dependent Variables:** Execution Accuracy (EX), Valid-SQL Rate (VSR), Exact Match (EM).
* **Controlled Variables:** Base schema representation (DDL format), target evaluation examples ($N = 1,034$), execution sandbox limits (10.0s timeout, single statement).
* **Comparison Groups:**
  - *Control:* GPT-4o mini + 3-shot BM25 in-context learning.
  - *Treatment:* Qwen2.5-Coder-7B-Instruct fine-tuned on Spider train ($r=16, \alpha=32$).
* **Evaluation Dataset & Split:** Spider Development Set ($N = 1,034$).
* **Planned Statistical Analysis:** Paired bootstrap resampling ($B = 1,000$ iterations) for the difference $\Delta \text{EX} = \text{EX}_{\text{ft}} - \text{EX}_{\text{api}}$, with a 95% confidence interval and McNemar’s test on paired binary execution outcomes ($\alpha = 0.05$).
* **Potential Confounders:** Commercial API weight updates during testing; differential pre-training corpus exposure to Spider; prompt template sensitivity.
* **Falsification / Rejection Condition:** The hypothesis is unsupported if the lower bound of the 95% bootstrap CI for $\Delta \text{EX}$ falls below $-2.5\%$, or if the fine-tuned model suffers significantly higher syntax error rates ($p < 0.05$).

---

### Hypothesis 2: LoRA vs. QLoRA Precision & Memory Frontier
* **Research Question:** Does 4-bit NormalFloat (NF4) quantized fine-tuning (QLoRA) match the execution accuracy of 16-bit LoRA while significantly cutting peak training VRAM?
* **Hypotheses:**
  - $H_0$: 4-bit QLoRA suffers an execution accuracy loss exceeding 2.0 percentage points compared to 16-bit LoRA ($\text{EX}_{\text{LoRA}} - \text{EX}_{\text{QLoRA}} > 2.0\%$).
  - $H_1$: 4-bit QLoRA retains execution accuracy within 2.0 percentage points of 16-bit LoRA ($\text{EX}_{\text{LoRA}} - \text{EX}_{\text{QLoRA}} \le 2.0\%$) while reducing peak training VRAM by $\ge 40\%$.
* **Independent Variable:** Base weight quantization bit-width during adaptation (16-bit BF16 vs. 4-bit NF4 with double quantization).
* **Dependent Variables:** Execution Accuracy (EX), Peak Training Allocated VRAM (GB), Step Training Latency (ms/step).
* **Controlled Variables:** Base model (Qwen2.5-Coder-7B), LoRA rank ($r=16$), LoRA alpha ($\alpha=32$), target modules (all linear projections), batch size, optimizer (paged AdamW 8-bit for QLoRA vs standard AdamW for LoRA).
* **Comparison Groups:**
  - *Control:* 16-bit LoRA SFT on Spider train.
  - *Treatment:* 4-bit NF4 QLoRA SFT on Spider train.
* **Evaluation Dataset & Split:** Spider Development Set ($N = 1,034$).
* **Planned Statistical Analysis:** Paired differences test; 95% bootstrap confidence interval on $\Delta \text{EX}$; relative VRAM reduction percentage $\frac{\text{VRAM}_{\text{LoRA}} - \text{VRAM}_{\text{QLoRA}}}{\text{VRAM}_{\text{LoRA}}} \times 100\%$.
* **Potential Confounders:** Differences in gradient accumulation dynamics; optimizer memory footprints (paged vs unpaged).
* **Falsification Condition:** Unsupported if $\text{EX}_{\text{LoRA}} - \text{EX}_{\text{QLoRA}} > 2.0\%$ with $p < 0.05$, or if VRAM reduction is less than $40\%$.

---

### Hypothesis 3: LoRA Rank Saturation ($r$)
* **Research Question:** Does increasing LoRA rank $r$ beyond $r=32$ yield statistically significant improvements in execution accuracy on text-to-SQL?
* **Hypotheses:**
  - $H_0$: Model execution accuracy with rank $r=64$ is indistinguishable from or worse than rank $r=32$ ($\text{EX}_{r=64} - \text{EX}_{r=32} \le 0.5\%$).
  - $H_1$: Model execution accuracy with rank $r=64$ improves over rank $r=32$ by $> 0.5$ percentage points ($\text{EX}_{r=64} - \text{EX}_{r=32} > 0.5\%$).
* **Independent Variable:** LoRA intrinsic rank $r \in \{8, 16, 32, 64\}$ (with scaling factor $\alpha = 2 \times r$).
* **Dependent Variables:** Execution Accuracy (EX), Adapter parameter size (MB), Checkpoint saving time.
* **Controlled Variables:** Base model, training data partition (100% Spider train), epochs (3), learning rate ($2 \times 10^{-4}$).
* **Comparison Groups:** Pairwise steps across $r=8 \to r=16 \to r=32 \to r=64$.
* **Evaluation Dataset & Split:** Spider Development Set ($N = 1,034$).
* **Planned Statistical Analysis:** ANOVA across ranks followed by pairwise McNemar tests with Benjamini-Hochberg False Discovery Rate (FDR) correction; empirical bootstrap CIs for each rank.
* **Potential Confounders:** Overfitting at higher ranks without increased weight decay; learning rate optimality differing by rank.
* **Falsification Condition:** If $r=64$ fails to improve over $r=32$ by at least 0.5% (or if accuracy degrades due to overfitting), the hypothesis of continued scaling is rejected, establishing $r \le 32$ as the practical saturation ceiling.

---

### Hypothesis 4: Out-of-Domain Generalization Penalty (Unseen Schemas)
* **Research Question:** Does supervised fine-tuning induce schema-specific memorization that causes a larger degradation on completely unseen database schemas than prompt-based inference with frontier models?
* **Hypotheses:**
  - $H_0$: The relative drop in Execution Accuracy from in-domain (Spider dev) to out-of-domain (Custom Held-Out) is equal for fine-tuned models and frontier prompted models ($\Delta \text{EX}_{\text{drop, ft}} = \Delta \text{EX}_{\text{drop, api}}$).
  - $H_1$: The fine-tuned model suffers a significantly steeper relative degradation on the unseen schema than the frontier prompted model ($\Delta \text{EX}_{\text{drop, ft}} - \Delta \text{EX}_{\text{drop, api}} \ge 10.0\%$).
* **Independent Variable:** Model adaptation paradigm (Fine-tuned open-weight vs. Prompted frontier reference).
* **Dependent Variables:** Generalization Delta ($\Delta_{\text{OOD}} = \text{EX}_{\text{Spider dev}} - \text{EX}_{\text{Custom held-out}}$), Column resolution error rate, Table hallucination rate.
* **Controlled Variables:** Identical DDL schema representation, greedy decoding ($T=0.0$).
* **Comparison Groups:**
  - *Group A:* Fine-tuned Qwen2.5-Coder-7B evaluated on Spider dev vs. Custom Held-Out ($N \approx 100$).
  - *Group B:* Prompted GPT-4o mini evaluated on Spider dev vs. Custom Held-Out ($N \approx 100$).
* **Evaluation Dataset & Split:** Spider Development Set ($N = 1,034$) vs. SQLForge Custom Held-Out Schema ($N \approx 100$).
* **Planned Statistical Analysis:** Difference-in-differences test on paired binary performance drops; bootstrap confidence intervals for the generalization delta.
* **Potential Confounders:** Intrinsic difficulty variance of custom held-out queries compared to Spider dev.
* **Falsification Condition:** Unsupported if the fine-tuned model demonstrates equal or superior robustness to the unseen schema ($\Delta \text{EX}_{\text{drop, ft}} \le \Delta \text{EX}_{\text{drop, api}} + 5\%$).

---

## 3. Exploratory Inquiries

### Inquiry 1: Training Data Scaling Trajectory
* **Research Question:** What is the functional form of execution accuracy gains as training data scales across 500, 1,000, 2,000, 5,000, and full eligible Spider training examples (~7,000)?
* **Independent Variable:** Training dataset fraction $N \in \{500, 1000, 2000, 5000, 7000\}$.
* **Dependent Variables:** Execution Accuracy (EX), Valid-SQL Rate (VSR), Exact Match (EM).
* **Analysis Plan:** Fit empirical power-law and logarithmic curves ($\text{EX} = a \cdot \ln(N) + b$ vs $\text{EX} = c \cdot N^{-\alpha} + d$); test across 3 random seeds ($S=3$) per data fraction to compute variance bars.

### Inquiry 2: Human vs. Execution-Validated Synthetic Examples
* **Research Question:** Does execution-validated synthetic training data achieve equivalent sample efficiency to human-annotated examples, and does a 50/50 blend outperform pure human data?
* **Independent Variable:** Data origin composition (100% human Spider, 100% execution-filtered synthetic, 50/50 blend), normalized to identical total sample count ($N = 3,000$).
* **Dependent Variables:** Execution Accuracy on Spider dev and BIRD mini-dev; error taxonomy distribution.
* **Analysis Plan:** Pairwise McNemar tests across data conditions; qualitative inspection of synthetic error shifts.

### Inquiry 3: Post-Training Quantization Trade-offs (AWQ vs. GGUF)
* **Research Question:** How do 4-bit post-training quantization methods affect execution accuracy, KV cache VRAM footprint, and inference speed on CPU vs. GPU?
* **Independent Variable:** Quantization format (BF16 unquantized, AWQ 4-bit, GGUF Q4_K_M).
* **Dependent Variables:** Execution accuracy degradation ($\Delta \text{EX}$), p95 latency (ms), model memory footprint (GB).
* **Analysis Plan:** Measure Pareto trade-off curves (Accuracy vs. p95 Latency vs. RAM/VRAM).

### Inquiry 4: Serving Architecture Throughput & Concurrency (vLLM vs. HF)
* **Research Question:** How does continuous batching and PagedAttention in vLLM compare with standard Hugging Face pipelines across concurrent request loads (concurrency = 1, 8, 32)?
* **Independent Variable:** Serving framework (vLLM vs. HF) and concurrency load ($c \in \{1, 8, 32\}$).
* **Dependent Variables:** Tokens Per Second (TPS), Queries Per Second (QPS), p50 and p95 latency.
* **Analysis Plan:** Benchmark load harness simulating concurrent client SQL generation requests.

### Inquiry 5: Error Taxonomy Transition Under Fine-Tuning
* **Research Question:** Which specific categories of SQL generation errors (e.g. JOIN topology, aggregation mismatch, column hallucination) are eliminated by fine-tuning, and which persist?
* **Independent Variable:** Model type (Zero-shot base, Few-shot prompted, Fine-tuned LoRA, Fine-tuned QLoRA).
* **Dependent Variables:** Proportional error category counts across the 10-category SQLForge Error Taxonomy.
* **Analysis Plan:** Chi-square test of homogeneity across error category contingency tables.
