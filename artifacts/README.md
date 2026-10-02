# SQLForge Experiment Artifacts

This directory stores outputs generated during research experiments, including:
* `runs/`: Individual run logs, resolved configuration snapshots, execution trace JSONs, and metric summary files.
* `checkpoints/`: LoRA adapters and fine-tuned model checkpoint directories (gitignored).

## Local-First Philosophy
SQLForge defaults to local structured tracking:
* Every run generates an isolated directory: `artifacts/runs/<run_id>/`
* Artifacts inside this folder include:
  - `config.yaml`: Fully resolved frozen configuration snapshot.
  - `run_metadata.json`: Machine environment, git commit, seed, and timestamps.
  - `metrics.json`: Final execution metrics, confidence intervals, and latency stats.
  - `generations.jsonl`: Line-by-line model predictions and execution outcomes.

Never commit checkpoint weights, tokenized caches, or huge execution logs to Git.
