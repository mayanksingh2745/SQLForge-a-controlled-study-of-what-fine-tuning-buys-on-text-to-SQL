# SQLForge Data Governance & Provenance Policy

## 1. Data Provenance & Licensing Manifest

SQLForge relies strictly on ethically sourced, openly licensed benchmarks and synthetic data assets.

| Asset Name | Version | Primary Authors | Original License | Permitted Research Use |
| :--- | :--- | :--- | :--- | :--- |
| **Spider Benchmark** | 1.0 | Yale LILY Lab (Yu et al., 2018) | CC BY-SA 4.0 | Non-commercial & commercial academic research with attribution. |
| **BIRD Benchmark Subset** | 1.0 (mini-dev) | HKU / Alibaba (Li et al., 2023) | CC BY-NC-SA 4.0 | Non-commercial academic research only. |
| **SQLForge Custom Held-Out** | 1.0 | SQLForge Maintainers (2026) | Apache 2.0 | Full open-source research and commercial benchmarking. |

---

## 2. Gated Model Weights & Access Tokens

* **Llama-3.1-8B-Instruct:** Subject to the Meta Llama 3.1 Community License Agreement. Users must accept terms on Hugging Face Hub and provide `HF_TOKEN` in `.env` if downloading base weights.
* **Qwen2.5-Coder Series:** Apache 2.0 permissive license. Weights are freely accessible without gated restrictions.
* **Frontier API Endpoints:** Optional; requires user-supplied `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Keys must never be stored in repository configs.

---

## 3. Privacy & Personally Identifiable Information (PII)

1. **Benchmark Databases:** The Spider and BIRD benchmarks consist of synthetic or anonymized public datasets (e.g., academic records, public transit tables, sports statistics). No real personally identifiable customer records are present.
2. **Custom Held-Out Schema:** Built around an e-commerce mock schema with synthetically generated user names and synthetic transactions.
3. **No Secret Ingestion:** The data ingestion pipeline actively checks and rejects datasets containing credentials, API keys, email addresses, or tax IDs.

---

## 4. Split Integrity & Leakage Verification

To prevent benchmark degradation and data contamination:
1. **Schema-Disjoint Splitting:** Database schemas in the evaluation partitions (`spider:dev`, `bird_mini:dev`, `held_out_custom:test`) must have zero overlap with database schemas present in training sets.
2. **Deterministic Hashes:** Every preprocessed dataset stored in `data/processed/*.jsonl` must be indexed with a SHA-256 integrity hash recorded in `data/manifest.json`.
3. **Automated Leakage Audit:** Pre-training validation scripts verify that question embeddings and normalized SQL strings do not exceed an 8-gram Jaccard similarity threshold of $0.85$ between train and dev sets.

---

## 5. Synthetic Training Data Governance

When evaluating synthetic training data:
1. **Provenance Tagging:** Every synthetic example must populate `SyntheticProvenance`:
   - `generator_model`: The exact LLM version utilized for synthesis.
   - `prompt_template_id`: Versioned prompt configuration used for generation.
   - `date_generated`: Timestamp in UTC.
2. **Execution Verification Gate:** Synthetic queries are admitted to training partitions *only* if:
   - The SQL statement parses with zero syntax errors.
   - The query executes against the target schema without throwing errors.
   - The returned result set is non-empty.
   - Consensus is confirmed via execution-based self-consistency ($k \ge 3$ samplings).
