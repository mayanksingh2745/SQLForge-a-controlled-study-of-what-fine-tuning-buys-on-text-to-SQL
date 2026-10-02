"""Local-first experiment tracker and run record manager."""

import hashlib
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
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
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
        run_id: str | None = None,
    ) -> RunMetadata:
        """Initialize a new local experiment run and snapshot its environment and configuration.

        Args:
            experiment_name: Name of experiment configuration or paradigm.
            config: Full experiment configuration dict or Pydantic model.
            seed: Random seed used for run.
            run_id: Optional explicit run identifier. If None, generated automatically.

        Returns:
            Instantiated RunMetadata.

        Raises:
            FileExistsError: If a run directory with the given run_id already exists and is non-empty.
        """
        resolved_run_id = run_id or generate_run_id(prefix=experiment_name)
        run_dir = self.base_dir / resolved_run_id

        # Collision-safe check: Never silently overwrite an existing populated run
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Collision detected: Run directory already exists and is non-empty: {run_dir}"
            )

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
            run_id=resolved_run_id,
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

    def log_generation(
        self,
        run_id: str,
        generation: GenerationResult | dict[str, Any],
    ) -> Path:
        """Append an individual query generation record to generations.jsonl."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        gen_dict: dict[str, Any]
        if isinstance(generation, BaseModel):
            gen_dict = generation.model_dump(mode="json")
        else:
            gen_dict = dict(generation)

        gen_file = run_dir / "generations.jsonl"
        with open(gen_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(gen_dict, sort_keys=True) + "\n")

        return gen_file

    def log_anomaly(
        self,
        run_id: str,
        anomaly: dict[str, Any],
    ) -> Path:
        """Record an execution timeout, syntax error, or ambiguous result anomaly."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        anomalies_file = run_dir / "eval_anomalies.json"
        existing: list[dict[str, Any]] = []
        if anomalies_file.exists():
            with open(anomalies_file, encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []

        existing.append(anomaly)
        with open(anomalies_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)

        return anomalies_file

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

    def write_manifest(self, run_id: str) -> Path:
        """Compute SHA-256 hashes of all run artifacts and persist manifest.json."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        manifest: dict[str, str] = {}
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = file_path.relative_to(run_dir).as_posix()
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                manifest[rel_path] = hasher.hexdigest()

        manifest_file = run_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        return manifest_file

    def verify_run(self, run_id: str) -> dict[str, Any]:
        """Verify the structural integrity, completeness, and hash provenance of a run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            return {
                "verified": False,
                "run_id": run_id,
                "error": f"Run directory does not exist: {run_dir}",
            }

        missing_required = []
        for req in ["run_metadata.json", "config.yaml"]:
            if not (run_dir / req).exists():
                missing_required.append(req)

        metadata = self.get_run(run_id)
        if metadata and metadata.status == "completed" and not (run_dir / "metrics.json").exists():
            missing_required.append("metrics.json")

        hash_mismatches = []
        manifest_file = run_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                expected_hashes = json.load(f)

            for rel_path, expected_hash in expected_hashes.items():
                target = run_dir / rel_path
                if not target.exists():
                    missing_required.append(rel_path)
                    continue
                hasher = hashlib.sha256()
                with open(target, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != expected_hash:
                    hash_mismatches.append(rel_path)

        verified = (len(missing_required) == 0) and (len(hash_mismatches) == 0)
        return {
            "verified": verified,
            "run_id": run_id,
            "status": metadata.status if metadata else "unknown",
            "missing_files": sorted(list(set(missing_required))),
            "hash_mismatches": hash_mismatches,
        }

    def finish_run(self, run_id: str, status: str = "completed") -> RunMetadata:
        """Mark a run as completed or failed, generate manifest, and update metadata."""
        run_dir = self.base_dir / run_id
        metadata = self.get_run(run_id)
        if metadata is None:
            raise FileNotFoundError(f"Run not found: {run_id}")

        updated_dict = metadata.model_dump()
        updated_dict["status"] = status
        updated = RunMetadata(**updated_dict)
        self._save_metadata(run_dir, updated)
        self.write_manifest(run_id)
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
