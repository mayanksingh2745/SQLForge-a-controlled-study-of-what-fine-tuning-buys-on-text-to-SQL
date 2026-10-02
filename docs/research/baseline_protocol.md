# SQLForge Baseline Protocol & Fair Comparison Standards

This document establishes the baseline tiers, prompting conventions, retrieval boundaries, and fair comparison rules for SQLForge.

---

## 1. Baseline Groups & Model Tiers

To isolate what fine-tuning contributes beyond prompting and architecture, SQLForge benchmarks five distinct operational groups across identical evaluation examples:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Baseline Hierarchy                              │
├────────────────────────┬───────────────────────────────────────────────┤
│ Tier 1: Zero-Shot Base │ Base un-adapted open-weight model + DDL       │
├────────────────────────┼───────────────────────────────────────────────┤
│ Tier 2: Few-Shot Fixed │ Base model + DDL + static representative demo │
├────────────────────────┼───────────────────────────────────────────────┤
│ Tier 3: Few-Shot RAG   │ Base model + DDL + BM25 retrieved training ex │
├────────────────────────┼───────────────────────────────────────────────┤
│ Tier 4: Frontier API   │ GPT-4o mini reference + matched few-shot RAG  │
├────────────────────────┼───────────────────────────────────────────────┤
│ Tier 5: Fine-Tuned     │ SFT (LoRA/QLoRA) open-weight model + DDL only │
└────────────────────────┴───────────────────────────────────────────────┘
```

### Detailed Group Specifications

1. **Group B0 (Zero-Shot Open-Weight Base):**
   - *Models:* Qwen2.5-Coder-1.5B-Instruct, Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct.
   - *Prompt Formulation:* System role prompt + DDL schema serialization + user question.
   - *Decoding:* Greedy ($T = 0.0$), max tokens = 512.
2. **Group B1 (Static Few-Shot In-Context Learning):**
   - *Models:* Qwen2.5-Coder-7B-Instruct.
   - *Demonstrations:* $k = 3$ static, manually curated text-to-SQL demonstration pairs demonstrating basic SELECT, JOIN, and aggregation syntax on a neutral schema.
3. **Group B2 (Dynamic Retrieval-Augmented Few-Shot):**
   - *Models:* Qwen2.5-Coder-7B-Instruct.
   - *Demonstrations:* $k \in \{1, 3, 5\}$ dynamically retrieved from the training partition using BM25 query similarity against training question text.
4. **Group B3 (Frontier Commercial Reference):**
   - *Model:* `gpt-4o-mini-2024-07-18` (or current pinned snapshot).
   - *Protocol:* Evaluated under identical zero-shot and $k=3$ BM25 retrieval configurations to establish a strong external reference point.
5. **Group T1–T3 (Fine-Tuned Variants):**
   - SFT variants fine-tuned on Spider train with zero-shot prompting at test time (evaluating whether fine-tuning eliminates the need for test-time demonstration tokens).

---

## 2. Fair Comparison & Anti-Confounding Rules

To ensure scientific validity when comparing fine-tuned open models against prompted frontier models:

### Rule 1: Dataset Partition Alignment
All comparison groups are evaluated against the **exact same example instances** (`spider:dev`, $N = 1,034$; `bird_mini:dev`, $N = 500$; `held_out_custom:test`, $N \approx 100$). No partial subsets or selective dropping of difficult queries is permitted.

### Rule 2: Strict Retrieval Index Isolation
When indexing examples for dynamic few-shot retrieval (Group B2 & B3):
* The retrieval index must be populated **exclusively from the designated training split** (`spider:train`).
* Development and test split examples, questions, schemas, and gold queries are strictly forbidden from entering the BM25 or embedding indexes.

### Rule 3: Schema Representation Parity
All models receive the exact same DDL schema representation, including table names, column names, column data types, primary keys, and foreign keys. Variations in schema format (e.g. compact pipe-delimited vs. DDL) are evaluated only within dedicated schema representation experiments (ADR-002).

### Rule 4: Context Length & Prompt Truncation Accounting
* Open-weight 7B models (8k token context) and GPT-4o mini (128k token context) have disparate context windows.
* Prompts are formatted to remain strictly within the conservative 8,192 token limit to ensure that no open-weight model suffers truncation that the frontier model escapes. Total prompt and completion tokens are logged for every example.

### Rule 5: Deterministic Decoding
All primary benchmark results are generated with **greedy decoding** (temperature $T = 0.0$). Sampling ($T > 0$) is restricted strictly to self-consistency experiments where multiple passes ($k \ge 3$) are explicitly budgeted and noted.

---

## 3. Frontier Models as Reference Baselines, Not Theoretical Ceilings

SQLForge explicitly documents that:
1. **Frontier Models are Reference Baselines:** Models such as GPT-4o mini are not an immutable "ceiling" of text-to-SQL performance; they represent the current state-of-the-art for off-the-shelf general-purpose APIs.
2. **Pre-Training Opacity:** Proprietary commercial models have undisclosed pre-training corpora that likely include academic benchmarks (Spider, BIRD) and GitHub repositories containing benchmark solutions. Consequently, high zero-shot performance in commercial models may reflect benchmark exposure rather than superior relational reasoning.
3. **Reproducibility Differences:** Commercial API snapshots can experience undocumented backend adjustments or service deprecations. Open-weight checkpoints with fixed weights (`safetensors`) and pinned Git commit hashes represent the gold standard for scientific permanence.

---

## 4. API Budget, Rate Limits & Tracking

For all frontier API runs:
* **Rate Limiting:** Managed via exponential backoff (initial delay 1.0s, max retries 5) with a concurrency throttle of 5 requests/sec.
* **Cost Tracking:** Automated logging of prompt tokens, completion tokens, and dollar expenditures calculated per run.
* **Safety Budget Ceiling:** Hard dollar spending cap of **$50.00 USD** enforced across all API baseline runs.
