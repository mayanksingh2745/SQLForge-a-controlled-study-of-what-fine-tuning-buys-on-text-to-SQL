# SQLForge Domain Glossary & Project Dictionary

This document defines the core concepts, statistical terms, machine learning techniques, and database mechanisms employed throughout the SQLForge research program.

---

### Text-to-SQL
The task of translating a natural language query (e.g., *"What is the average tenure of software engineers in California?"*) into an executable Structured Query Language (SQL) statement against a specified database schema.

### Database Schema
The formal definition of a relational database structure, specifying tables, columns, data types, primary keys, foreign key relationships, constraints, and table descriptions.

### SQL Dialect
The specific variant or specification of the SQL language implemented by a target database engine (e.g., SQLite, PostgreSQL, MySQL, DuckDB, Snowflake). Dialects vary in function names, date/time formatting, string concatenation operators, and typing behavior.

### Gold SQL
The reference ground-truth SQL query authored or verified by human domain experts that accurately answers a natural language question against a given schema.

### Execution Accuracy (EX)
The primary evaluation metric for text-to-SQL systems. It measures whether executing the model-generated SQL query against the target database returns the exact same result set (rows and columns, modulo ordering unless ordered) as executing the Gold SQL query. Unlike syntax-only metrics, execution accuracy rewards semantically equivalent SQL queries even if their AST or textual structure differs from the gold query.

### Exact Match (EM)
A strict syntactic metric that measures whether the generated SQL query exactly matches the Gold SQL query token-for-token or Abstract Syntax Tree (AST)-for-AST after normalization (stripping whitespaces, standardizing uppercase keywords, and removing comments). EM often penalizes valid, semantically equivalent queries (e.g., swapping `JOIN` order or using subqueries instead of window functions).

### Valid-SQL Rate (VSR)
The proportion of model-generated SQL queries that parse without syntax errors and execute successfully on the database engine, regardless of whether the returned result matches the Gold SQL. A high VSR indicates strong syntactic competence and schema awareness.

### Spider
A widely used, cross-domain, complex text-to-SQL benchmark created by Yale University (Yu et al., 2018). It comprises 10,181 natural language questions and 5,693 unique complex SQL queries across 200 databases spanning 138 domains. Spider splits databases across train and development sets to test zero-shot schema generalization.

### BIRD (Benchmark for Large-scale Database Grounded Text-to-SQL)
A challenging benchmark (Li et al., 2023) focusing on large-scale, dirty real-world databases (over 33 GB of data across 95 databases). BIRD emphasizes database-grounded reasoning, external domain knowledge ("evidence"), dirty data handling, and query execution efficiency (VES).

### Data Contamination
An experimental flaw occurring when benchmark evaluation examples (questions, schemas, or gold queries) inadvertently leak into a model's pre-training or fine-tuning dataset, artificially inflating benchmark scores and obscuring true generalization ability.

### Train/Dev/Test Split
A partition of dataset examples into mutually exclusive sets:
* **Train:** Used to compute gradient updates during fine-tuning.
* **Dev (Validation):** Used for hyperparameter tuning, model checkpoint selection, and prompt refinement.
* **Test / Held-Out:** Reserved strictly for final unbiased evaluation on previously unseen schemas and query patterns.

### LoRA (Low-Rank Adaptation)
A parameter-efficient fine-tuning (PEFT) technique (Hu et al., 2021) that freezes pre-trained model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices $\Delta W = B \cdot A$, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$. This drastically reduces trainable parameters and VRAM requirements while matching full fine-tuning performance.

### QLoRA (Quantized Low-Rank Adaptation)
An extension of LoRA (Dettmers et al., 2023) that quantizes the base model weights to a 4-bit NormalFloat (NF4) data type, uses double quantization to compress quantization constants, and employs paged optimizers to prevent VRAM spikes during gradient updates. QLoRA enables fine-tuning 7B–13B parameter models on consumer-grade GPUs with 8GB–16GB VRAM.

### Rank ($r$) and Target Modules
* **Rank ($r$):** The intrinsic dimension of the low-rank adaptation matrices. Controls adaptation capacity and parameter footprint.
* **Target Modules:** The specific weight matrices in the transformer architecture to which LoRA adapters are attached (e.g., query, key, value, and output projection matrices `q_proj`, `k_proj`, `v_proj`, `o_proj`, and MLP layers `gate_proj`, `up_proj`, `down_proj`).

### Quantization
The process of reducing the precision of model weights and activations (e.g., from 16-bit floating-point `FP16`/`BF16` to 8-bit `INT8` or 4-bit `NF4`/`AWQ`/`GGUF`) to decrease memory footprint and increase inference throughput with minimal loss in execution accuracy.

### Tokens Per Second (TPS)
The rate at which an inference engine decodes and outputs tokens, calculated as total generated tokens divided by generation latency in seconds. Measures inference throughput.

### p95 Latency
The 95th percentile response time for inference generation across an evaluation set. Indicates tail latency behavior, meaning 95% of queries execute faster than this threshold.

### VRAM (Video Random Access Memory)
The high-bandwidth memory dedicated to GPU execution. Constrains maximum model parameter size, context length, batch size, and KV cache allocation during fine-tuning and inference.

### Bootstrap Confidence Interval
A non-parametric statistical method that repeatedly resamples (with replacement) the evaluation results $B$ times (e.g., $B = 1000$) to construct empirical confidence intervals (e.g., 95% CI) around primary metrics like execution accuracy. Essential for proving that observed differences between fine-tuned and prompted models are statistically significant rather than noise.

### Synthetic Training Data
Text-to-SQL training examples generated automatically by large language models, program synthesis, or schema graph perturbation rather than human annotation.

### Execution-Based Self-Consistency
A decoding and validation filter for synthetic data generation or inference where multiple candidate SQL queries are sampled for a single question, executed against the database, and clustered by result set. The result cluster with highest consensus is selected, and queries that fail execution are rejected.

### Experiment Run and Random Seed
* **Experiment Run:** A unique, auditable execution instance characterized by an immutable configuration snapshot, environment fingerprint, and output artifact directory.
* **Random Seed:** An integer initializing pseudorandom number generators (PRNGs) across Python, NumPy, and PyTorch, ensuring that dataset shuffling, weight initialization, and stochastic sampling are reproducible.
