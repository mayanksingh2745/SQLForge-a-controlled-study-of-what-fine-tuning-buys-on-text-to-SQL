"""Local-first experiment tracker and run record manager."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from sqlforge.reproducibility import (
    generate_run_id,
    get_environment_metadata,
    hash_dict,
)
from sqlforge.schemas.evaluation import EvaluationMetrics
from sqlforge.schemas.experiments import HardwareMetadata, RunMetadata


class ExperimentTracker:
    """Manages experiment runs, configuration snapshots, and metric persistence locally."""

    def __init__(self, base_artifact_dir: Path | str = "artifacts/runs"):
        self.base_dir = Path(base_artifact_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def init_run(
        self,
        experiment_name: str,
        config: dict[str, Any] | BaseModel,
        seed: int = 42,
    ) -> RunMetadata:
        """Initialize a new local experiment run and snapshot its environment and configuration.

        Args:
            experiment_name: Name of experiment configuration or paradigm.
            config: Full experiment configuration dict or Pydantic model.
            seed: Random seed used for run.

        Returns:
            Instantiated RunMetadata.
        """
        run_id = generate_run_id(prefix=experiment_name)
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to dict
        config_dict: dict[str, Any]
        if isinstance(config, BaseModel):
            config_dict = config.model_dump(mode="json")
        else:
            config_dict = dict(config)

        config_hash = hash_dict(config_dict)

        # Save configuration snapshot
        config_file = run_dir / "config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)

        # Capture environment
        env_meta = get_environment_metadata()
        git_meta = env_meta.get("git", {})
        gpu_meta = env_meta.get("gpu", {})

        hardware = HardwareMetadata(
            cpu_count=env_meta.get("cpu_count", 1),
            total_ram_gb=env_meta.get("total_ram_gb", 0.0),
            gpu_count=gpu_meta.get("device_count", 0),
            gpu_names=gpu_meta.get("device_names", []),
            cuda_available=gpu_meta.get("cuda_available", False),
        )

        metadata = RunMetadata(
            run_id=run_id,
            experiment_name=experiment_name,
            timestamp_utc=env_meta.get("timestamp_utc", ""),
            git_commit=git_meta.get("commit"),
            git_branch=git_meta.get("branch"),
            git_dirty=git_meta.get("is_dirty", False),
            python_version=env_meta.get("python_version", ""),
            platform=env_meta.get("platform", ""),
            hardware=hardware,
            seed=seed,
            status="running",
            config_hash=config_hash,
        )

        self._save_metadata(run_dir, metadata)
        return metadata

    def log_metrics(
        self,
        run_id: str,
        metrics: EvaluationMetrics | dict[str, Any],
    ) -> Path:
        """Persist evaluation metrics for a run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        metrics_dict: dict[str, Any]
        if isinstance(metrics, BaseModel):
            metrics_dict = metrics.model_dump(mode="json")
        else:
            metrics_dict = dict(metrics)

        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, sort_keys=True)

        return metrics_file

    def finish_run(self, run_id: str, status: str = "completed") -> RunMetadata:
        """Mark a run as completed or failed and update metadata."""
        run_dir = self.base_dir / run_id
        metadata = self.get_run(run_id)
        if metadata is None:
            raise FileNotFoundError(f"Run not found: {run_id}")

        updated_dict = metadata.model_dump()
        updated_dict["status"] = status
        updated = RunMetadata(**updated_dict)
        self._save_metadata(run_dir, updated)
        return updated

    def get_run(self, run_id: str) -> RunMetadata | None:
        """Read and validate RunMetadata for an existing run."""
        meta_file = self.base_dir / run_id / "run_metadata.json"
        if not meta_file.exists():
            return None

        with open(meta_file, encoding="utf-8") as f:
            data = json.load(f)

        return RunMetadata(**data)

    def _save_metadata(self, run_dir: Path, metadata: RunMetadata) -> None:
        """Save RunMetadata to run_metadata.json."""
        meta_file = run_dir / "run_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2)
