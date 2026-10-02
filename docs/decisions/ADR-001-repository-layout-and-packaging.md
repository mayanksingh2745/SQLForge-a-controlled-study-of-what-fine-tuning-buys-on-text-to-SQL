# ADR-001: Standard src/ Layout and Modular Packaging

## Status
Accepted

## Context
Text-to-SQL research codebases often suffer from monolithic file structures, tangled imports, and ad-hoc script execution. As SQLForge expands across 12 planned phases—encompassing data ingestion, LoRA fine-tuning, sandboxed execution, quantization, and serving—the packaging structure must prevent circular dependencies and enforce strict isolation between research experiments and operational tooling.

## Decision
1. Adopt the standard `src/sqlforge/` layout using standard packaging standards (`pyproject.toml` with `setuptools.build_meta`).
2. Separate dependencies into modular extras:
   - `dev`: Testing, linting, type-checking (Ruff, MyPy, Pytest).
   - `train`: Heavyweight ML libraries (PyTorch, Transformers, PEFT, Accelerate, BitsAndBytes).
   - `eval`: SQL parsing and validation (SQLGlot, SQLParse).
   - `serving`: Production serving dependencies.
   - `all`: Meta-extra installing all components.
3. Expose the CLI entry point directly via `[project.scripts] sqlforge = "sqlforge.cli:main"`.

## Consequences
* **Positive:** Clear dependency boundaries prevent local dev environments from needing heavy CUDA or deep learning libraries during Step 0; clean imports avoid circular dependency bugs; package is installable via `pip install -e .`.
* **Negative:** Requires running `pip install -e .` or setting `PYTHONPATH=src` to run scripts directly.
