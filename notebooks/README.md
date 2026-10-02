# SQLForge Exploratory Notebooks

This directory is reserved for exploratory data analysis, interactive schema visualizations, and qualitative error inspection.

## Guidelines
1. **Research Reproducibility:** Core pipelines must reside in `src/sqlforge/`, not in notebooks.
2. **Clean Notebooks:** Strip all notebook execution outputs before committing to Git to avoid repository bloat and diff pollution.
3. **No Direct Model Training:** Training jobs should be launched via the CLI or experiment scripts to ensure proper config hashing, seed logging, and artifact generation.
