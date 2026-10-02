# SQLForge Dataset Protocol, Leakage Controls & Data Governance

This document establishes the dataset provenance, partition integrity checks, leakage prevention mechanisms, and custom benchmark design protocol for SQLForge.

---

## 1. Benchmark Datasets & Partitioning

SQLForge relies on three dataset tiers to evaluate distinct aspects of text-to-SQL capability:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Dataset Responsibilities                       │
├─────────────────────┬──────────────────┬───────────────────────────────┤
│ Dataset Tier        │ Split / Target   │ Scientific Purpose            │
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ Spider 1.0          │ Train (~7,000)   │ Fine-tuning training base     │
│                     │ Dev (1,034)      │ In-domain cross-schema eval   │
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ BIRD (Mini Subset)  │ Dev (500)        │ Dirty data & external evidence│
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ Custom Held-Out     │ Test (~100)      │ Out-of-distribution OOD eval  │
└─────────────────────┴──────────────────┴───────────────────────────────┘
```

### Dataset Specifications

1. **Spider Benchmark (Yu et al., 2018):**
   - *Provenance:* Yale LILY Lab.
   - *License:* Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
   - *Dialect:* SQLite 3.
   - *Structure:* 10,181 natural language question/SQL pairs across 200 databases spanning 138 domains.
   - *Role:* Primary training corpus (`train`) and canonical cross-schema evaluation set (`dev`).
2. **BIRD Benchmark Subset (Li et al., 2023):**
   - *Provenance:* University of Hong Kong & Alibaba DAMO Academy.
   - *License:* Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
   - *Dialect:* SQLite 3.
   - *Structure:* Large-scale databases (over 33 GB data) with complex schema links, dirty values, and external domain knowledge ("evidence").
   - *Role:* Evaluates robust execution and knowledge grounding on a 500-example dev subset.
3. **SQLForge Custom Held-Out Benchmark:**
   - *Provenance:* Authored natively by SQLForge research maintainers.
   - *License:* Apache 2.0.
   - *Dialect:* SQLite 3.
   - *Structure:* Approximately 100 questions over an independently designed relational schema.
   - *Role:* Strict out-of-distribution generalization testing on a schema guaranteed absent from open benchmarks.

---

## 2. Custom Held-Out Schema & Query Construction Protocol

### Schema Design Principles
The custom held-out schema is designed around an enterprise multi-channel e-commerce and subscription billing database (`subscription_analytics_db`).
* **Independence:** Not cloned, adapted, or translated from Spider, BIRD, Kaggle, or WikiSQL databases.
* **Relational Complexity:** 6 normalized tables (`customers`, `subscriptions`, `plans`, `invoices`, `transactions`, `support_tickets`).
* **Complex Patterns:** Contains explicit primary/foreign keys, date/time timestamps, nullable discount fields, and self-referencing hierarchy keys.

### Query Authoring & Peer Review Protocol
1. **Diverse Intent Distribution:**
   - 25% Simple (single table filter, count, ordering).
   - 35% Moderate (two-table JOIN, date parsing with `strftime`, grouped aggregations).
   - 25% Challenging (multi-table JOINs, subqueries with `HAVING`, window functions or self-joins).
   - 15% Edge cases (handling `NULL` discounts, zero-billing periods, division by zero prevention).
2. **Dual-Expert Review:** Every question and Gold SQL query must be independently reviewed by two engineers. Queries are executed against the populated database to verify non-empty, unambiguous result sets.
3. **Isolation Guarantee:** The custom held-out JSONL file and database file will reside in a separate restricted directory (`data/processed/custom_held_out/`) and are strictly excluded from all training scripts and retrieval indexes.

### Epistemic Limitation of Custom Schemas
> [!IMPORTANT]
> While an independently authored custom schema completely eliminates direct verbatim dataset leakage from Spider and BIRD, **it does not prove that a modern pre-trained foundation model has never encountered similar semantic business logic or relational patterns**. It establishes robust out-of-distribution evaluation relative to public academic benchmarks, not absolute immunization from general pre-training knowledge.

---

## 3. Data Integrity & Leakage Controls

### Split Integrity & Duplicate Audits
Before training begins (in Step 3):
1. **Exact Deduplication:** Compute SHA-256 hashes of normalized question strings ($\text{strip} \to \text{lowercase} \to \text{remove punctuation}$) across all splits. Any question appearing in both train and dev/test sets is flagged and removed.
2. **Fuzzy N-gram Overlap Check:** Calculate 8-gram Jaccard similarity across questions. Question pairs with similarity $> 0.85$ are manually audited for leakage.
3. **Database Schema Disjointness:** Verify that the set of database identifiers in `spider:train` ($\mathcal{D}_{\text{train}}$) and `spider:dev` ($\mathcal{D}_{\text{dev}}$) satisfy:
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{dev}} = \emptyset$$
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{custom}} = \emptyset$$

### Retrieval Index Quarantine
* When building BM25 or semantic vector indexes for few-shot demonstration retrieval, the indexer must be passed only the training split file.
* Automated assertion checks verify that no example with `split == "dev"` or `split == "test"` exists in the index before query serving starts.

---

## 4. Synthetic Training Data Governance

When generating synthetic training pairs for data scaling ablations:
1. **Training Schema Boundary:** Synthetic queries are generated exclusively against schemas present in `spider:train`. Never generate synthetic queries against `spider:dev` or `custom_held_out`.
2. **Provenance Record:** Every synthetic record must include a `SyntheticProvenance` payload:
   - `generator_model`: Specific LLM model identifier used for generation.
   - `prompt_template_id`: Versioned prompt configuration.
   - `execution_verified`: Boolean confirming successful execution.
   - `date_generated`: UTC timestamp.
3. **Execution Filter Threshold:** Synthetic queries must parse without errors, execute without exceptions on the sandbox engine, and return at least 1 row of data.
