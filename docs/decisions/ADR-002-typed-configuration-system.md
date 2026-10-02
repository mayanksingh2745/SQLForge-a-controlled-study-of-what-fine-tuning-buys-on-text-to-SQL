# ADR-002: Typed Configuration System via YAML and Pydantic v2

## Status
Accepted

## Context
Scientific experimentation in NLP and LLMs requires configuring hundreds of hyperparameter permutations (learning rates, LoRA ranks, quantization bit-widths, prompt templates, dataset splits). Hardcoding parameters in Python scripts causes irreproducible research, while unvalidated JSON/YAML configs allow silent runtime failures midway through multi-hour GPU training jobs.

## Decision
1. Maintain version-controlled, human-readable YAML configurations in `configs/` partitioned into domain concerns:
   - `defaults.yaml`: Global runtime settings, logging, tracking.
   - `datasets.yaml`: Dataset catalog and split metadata.
   - `models.yaml`: Model architectures, LoRA target modules, generation defaults.
   - `experiments.yaml`: Experimental conditions and sweep matrices.
   - `evaluation.yaml`: Execution sandbox limits and metric toggles.
2. Validate all configurations at load time against typed Pydantic v2 data models (`sqlforge.schemas.experiments.ExperimentConfig`).
3. Freeze and snapshot the fully resolved configuration alongside every run as `artifacts/runs/<run_id>/config.yaml`.

## Consequences
* **Positive:** Fail-fast validation catches typos, invalid ranges, and missing fields before allocating compute resources; full traceability for every completed run.
* **Negative:** Adding new hyperparameter options requires updating both the YAML configs and their Pydantic schema counterparts.
