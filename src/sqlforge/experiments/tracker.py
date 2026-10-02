"""Local-first experiment tracker and run record manager."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from sqlforge.reproducibility import (
    generate_run_id,
    get_environment_metadata,
    hash_dict,
)
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
from sqlforge.schemas.experiments import HardwareMetadata, RunMetadata

# Conservative run ID format: alphanumeric, underscore, and hyphen only (max 128 chars)
RUN_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")
MAX_RUN_ID_LENGTH = 128

# Standard SHA-256 checksum format: 64 hexadecimal characters
SHA256_HEX_REGEX = re.compile(r"^[a-fA-F0-9]{64}$")


def validate_run_id(run_id: str) -> str:
    """Validate that a run identifier satisfies conservative naming rules.

    Args:
        run_id: Candidate run identifier string.

    Returns:
        The validated run identifier.

    Raises:
        ValueError: If the identifier is empty, exceeds maximum length, contains
            path separators, or contains invalid characters.
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("Run ID must be a non-empty string.")
    if len(run_id) > MAX_RUN_ID_LENGTH:
        raise ValueError(f"Run ID exceeds maximum length of {MAX_RUN_ID_LENGTH} characters.")
    if not RUN_ID_REGEX.match(run_id):
        raise ValueError(
            f"Invalid run ID '{run_id}'. Run IDs must contain only alphanumeric characters, "
            f"underscores, and hyphens, and cannot contain path separators or traversal characters."
        )
    return run_id


class ExperimentTracker:
    """Manages experiment runs, configuration snapshots, and metric persistence locally."""

    def __init__(self, base_artifact_dir: Path | str = "artifacts/runs"):
        self.base_dir = Path(base_artifact_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_run_dir(self, run_id: str) -> Path:
        """Safely resolve a run directory beneath the base artifact directory.

        Guarantees path containment and rejects any path traversal attempts.

        Args:
            run_id: Run identifier to resolve.

        Returns:
            Resolved absolute Path to the run directory.

        Raises:
            ValueError: If run_id is invalid or attempts to escape base_dir.
        """
        validated_id = validate_run_id(run_id)
        run_dir = (self.base_dir / validated_id).resolve()

        # Strict boundary check: run_dir must be an immediate child of base_dir
        if run_dir.parent != self.base_dir:
            raise ValueError(
                f"Path containment violation: run ID '{run_id}' resolves outside artifact root."
            )
        return run_dir

    def init_run(
        self,
        experiment_name: str,
        config: dict[str, Any] | BaseModel,
        seed: int = 42,
        run_id: str | None = None,
    ) -> RunMetadata:
        """Initialize a new local experiment run and snapshot its environment and configuration.

        Uses atomic directory reservation to reject any pre-existing directory (empty or populated).

        Args:
            experiment_name: Name of experiment configuration or paradigm.
            config: Full experiment configuration dict or Pydantic model.
            seed: Random seed used for run.
            run_id: Optional explicit run identifier. If None, generated automatically.

        Returns:
            Instantiated RunMetadata.

        Raises:
            ValueError: If run_id is invalid or violates path boundaries.
            FileExistsError: If a run directory with the given run_id already exists.
        """
        resolved_run_id = generate_run_id(prefix=experiment_name) if run_id is None else run_id
        run_dir = self._resolve_run_dir(resolved_run_id)

        # Atomic directory reservation: fail if directory exists (even if empty)
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Collision detected: Run directory already exists: '{run_dir}'. "
                f"Cannot initialize an existing run."
            ) from exc

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
        """Append an individual query generation record to generations.jsonl.

        Args:
            run_id: Target experiment run ID.
            generation: Generation record (GenerationResult or dict).

        Returns:
            Path to updated generations.jsonl.
        """
        run_dir = self._resolve_run_dir(run_id)
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
        """Record an execution timeout, syntax error, or ambiguous result anomaly non-destructively.

        Preserves existing data and raises clear errors if the log is corrupted.

        Args:
            run_id: Target experiment run ID.
            anomaly: Anomaly payload to append.

        Returns:
            Path to eval_anomalies.json.

        Raises:
            ValueError: If eval_anomalies.json exists but is malformed JSON or has wrong structure.
        """
        run_dir = self._resolve_run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        anomalies_file = run_dir / "eval_anomalies.json"
        existing: list[dict[str, Any]] = []

        if anomalies_file.exists():
            try:
                with open(anomalies_file, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupted anomaly log at '{anomalies_file}': failed to parse JSON. "
                    f"Original contents preserved; cannot append anomaly."
                ) from exc

            if not isinstance(data, list):
                raise ValueError(
                    f"Invalid anomaly log structure at '{anomalies_file}': expected top-level list, "
                    f"got {type(data).__name__}. Original contents preserved."
                )
            existing = data

        existing.append(dict(anomaly))
        with open(anomalies_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, sort_keys=True)

        return anomalies_file

    def log_metrics(
        self,
        run_id: str,
        metrics: EvaluationMetrics | dict[str, Any],
    ) -> Path:
        """Persist evaluation metrics for a run.

        Args:
            run_id: Target experiment run ID.
            metrics: EvaluationMetrics instance or dict.

        Returns:
            Path to metrics.json.
        """
        run_dir = self._resolve_run_dir(run_id)
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
        """Compute SHA-256 hashes of all run artifacts and persist manifest.json.

        Manifest self-excludes itself from hashing to avoid circularity.
        Symlinks resolving outside the run directory are strictly ignored.

        Args:
            run_id: Target experiment run ID.

        Returns:
            Path to manifest.json.
        """
        run_dir = self._resolve_run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        run_dir_resolved = run_dir.resolve()
        manifest: dict[str, str] = {}
        for file_path in sorted(run_dir.rglob("*")):
            # Self-exclusion: manifest.json is explicitly excluded from the manifest itself
            if file_path.is_file() and file_path.name != "manifest.json":
                # Do not follow symlinks pointing outside run_dir
                try:
                    resolved = file_path.resolve()
                    resolved.relative_to(run_dir_resolved)
                except (ValueError, RuntimeError):
                    continue
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

    def verify_run(self, run_id: str, strict: bool = True) -> dict[str, Any]:
        """Verify the structural integrity, completeness, artifact schemas, and hash provenance of a run.

        What verification GUARANTEES:
            - Structural presence of mandatory files per run lifecycle status (run_metadata.json,
              config.yaml, manifest.json, and metrics.json for completed runs).
            - Strict path containment: run artifacts and manifest paths must not escape the run directory
              (no path traversal '../', absolute paths, Windows drive paths, or symlinks escaping the directory root).
            - Syntax and schema conformance for all structured artifacts: valid Pydantic RunMetadata,
              valid YAML configuration, valid Pydantic EvaluationMetrics, line-by-line valid JSONL
              generation records, and valid JSON array anomaly logs.
            - Bitwise integrity: all tracked files exactly match the recorded SHA-256 hashes in manifest.json.
            - Completeness: no untracked/unhashed rogue files exist in the run directory.

        What cryptographic hashes and verification DO NOT guarantee:
            - Scientific or methodological validity of experimental results.
            - Correctness, safety, or semantic validity of generated SQL queries.
            - Absence of train/test data contamination, schema memorization, or prompt leakage.
            - Reproducibility across differing GPU hardware, driver versions, or nondeterministic kernels.
            - Authenticity against malicious out-of-band replacement of both files and the manifest together.

        Args:
            run_id: Target experiment run ID.
            strict: If True, requires manifest.json to declare run verified.

        Returns:
            Dict containing verification status and detailed error diagnostics.
        """
        try:
            run_dir = self._resolve_run_dir(run_id)
        except ValueError as exc:
            return {
                "verified": False,
                "run_id": run_id,
                "status": "error",
                "errors": [str(exc)],
            }

        if not run_dir.exists():
            return {
                "verified": False,
                "run_id": run_id,
                "status": "missing_directory",
                "errors": [f"Run directory does not exist: {run_dir}"],
            }

        run_dir_resolved = run_dir.resolve()
        missing_required: list[str] = []
        errors: list[str] = []
        hash_mismatches: list[str] = []
        untracked_files: list[str] = []

        # 1. Essential file presence
        for req in ["run_metadata.json", "config.yaml"]:
            if not (run_dir / req).exists():
                missing_required.append(req)

        # 2. Metadata validation
        metadata: RunMetadata | None = None
        if (run_dir / "run_metadata.json").exists():
            try:
                with open(run_dir / "run_metadata.json", encoding="utf-8") as f:
                    meta_raw = json.load(f)
                metadata = RunMetadata(**meta_raw)
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(f"Invalid run_metadata.json: {exc}")

        # Check status-specific required files
        if (
            metadata
            and metadata.status == "completed"
            and not (run_dir / "metrics.json").exists()
        ):
            missing_required.append("metrics.json")
        # For failed or aborted runs, run_metadata.json and config.yaml are the primary requirements.

        # 3. Structure validation for metrics.json
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    metrics_raw = json.load(f)
                if not isinstance(metrics_raw, dict):
                    errors.append("Invalid metrics.json: root structure must be a JSON object")
                else:
                    EvaluationMetrics(**metrics_raw)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                errors.append(f"Invalid metrics.json: {exc}")

        # 4. Structure validation for eval_anomalies.json
        anomalies_file = run_dir / "eval_anomalies.json"
        if anomalies_file.exists():
            try:
                with open(anomalies_file, encoding="utf-8") as f:
                    anomalies_raw = json.load(f)
                if not isinstance(anomalies_raw, list):
                    errors.append("Invalid eval_anomalies.json: root structure must be a JSON list")
                else:
                    for a_idx, anomaly in enumerate(anomalies_raw):
                        if not isinstance(anomaly, dict):
                            errors.append(
                                f"Invalid anomaly record at index {a_idx} in eval_anomalies.json: item must be a dictionary"
                            )
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid eval_anomalies.json: {exc}")

        # 5. YAML config validity
        if (run_dir / "config.yaml").exists():
            try:
                with open(run_dir / "config.yaml", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except Exception as exc:
                errors.append(f"Invalid config.yaml: {exc}")

        # 6. JSONL generation records validity
        if (run_dir / "generations.jsonl").exists():
            try:
                with open(run_dir / "generations.jsonl", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, start=1):
                        if line.strip():
                            try:
                                json.loads(line)
                            except json.JSONDecodeError as exc:
                                errors.append(
                                    f"Corrupted generations.jsonl at line {line_idx}: {exc}"
                                )
            except Exception as exc:
                errors.append(f"Failed to read generations.jsonl: {exc}")

        # 7. Manifest presence, path containment, and hash integrity
        manifest_file = run_dir / "manifest.json"
        if not manifest_file.exists():
            if strict:
                missing_required.append("manifest.json")
        else:
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    expected_hashes = json.load(f)

                if not isinstance(expected_hashes, dict):
                    errors.append("Malformed manifest.json: root structure must be a JSON object")
                else:
                    # Validate and check each entry in the manifest
                    for rel_path, expected_hash in expected_hashes.items():
                        # Validate path string
                        if not isinstance(rel_path, str) or not rel_path.strip():
                            errors.append(
                                f"Malformed manifest entry: path key must be a non-empty string, got {rel_path!r}"
                            )
                            continue

                        # Check for traversal, absolute paths, or drive prefixes before filesystem access
                        p = Path(rel_path)
                        if (
                            p.is_absolute()
                            or p.drive != ""
                            or ".." in p.parts
                            or rel_path.startswith(("/", "\\"))
                        ):
                            errors.append(
                                f"Security violation: manifest entry '{rel_path}' is an invalid or traversal path."
                            )
                            continue

                        # Validate SHA-256 hash string format
                        if not isinstance(expected_hash, str) or not SHA256_HEX_REGEX.match(
                            expected_hash
                        ):
                            errors.append(
                                f"Malformed SHA-256 hash for manifest entry '{rel_path}': "
                                f"expected 64 hexadecimal characters, got {expected_hash!r}"
                            )
                            continue

                        target = run_dir / rel_path

                        # Strict path containment and symlink safety check
                        try:
                            resolved_target = target.resolve()
                            resolved_target.relative_to(run_dir_resolved)
                            if target.is_symlink():
                                # Target symlink must not point outside run_dir
                                target.readlink()
                        except (ValueError, RuntimeError, OSError):
                            errors.append(
                                f"Security violation: manifest entry '{rel_path}' resolves outside run directory."
                            )
                            continue

                        if not target.exists():
                            missing_required.append(rel_path)
                            continue

                        if not target.is_file():
                            errors.append(f"Manifest target '{rel_path}' is not a regular file.")
                            continue

                        hasher = hashlib.sha256()
                        with open(target, "rb") as f:
                            for chunk in iter(lambda: f.read(65536), b""):
                                hasher.update(chunk)
                        if hasher.hexdigest() != expected_hash:
                            hash_mismatches.append(rel_path)

                    # Check for unexpected/untracked files in run_dir
                    for file_path in run_dir.rglob("*"):
                        if file_path.is_file() and file_path.name != "manifest.json":
                            try:
                                file_path.resolve().relative_to(run_dir_resolved)
                            except (ValueError, RuntimeError):
                                errors.append(
                                    f"Security violation: file '{file_path.name}' resolves outside run directory."
                                )
                                continue
                            rel_posix = file_path.relative_to(run_dir).as_posix()
                            if rel_posix not in expected_hashes:
                                untracked_files.append(rel_posix)

            except json.JSONDecodeError as exc:
                errors.append(f"Malformed manifest.json: {exc}")

        is_verified = (
            len(missing_required) == 0
            and len(hash_mismatches) == 0
            and len(errors) == 0
            and len(untracked_files) == 0
        )

        return {
            "verified": is_verified,
            "run_id": run_id,
            "status": metadata.status if metadata else "unknown",
            "missing_manifest": not manifest_file.exists(),
            "missing_files": sorted(list(set(missing_required))),
            "hash_mismatches": sorted(hash_mismatches),
            "untracked_files": sorted(untracked_files),
            "errors": errors,
            "verification_level": "strict" if strict else "lenient",
        }

    def finish_run(self, run_id: str, status: str = "completed") -> RunMetadata:
        """Mark a run as completed or failed, update metadata, and generate manifest.

        Args:
            run_id: Target experiment run ID.
            status: Final run status ('completed', 'failed', 'aborted').

        Returns:
            Updated RunMetadata.
        """
        run_dir = self._resolve_run_dir(run_id)
        metadata = self.get_run(run_id)
        if metadata is None:
            raise FileNotFoundError(f"Run not found: {run_id}")

        updated_dict = metadata.model_dump()
        updated_dict["status"] = status
        updated = RunMetadata(**updated_dict)

        # 1. Update and persist final metadata
        self._save_metadata(run_dir, updated)

        # 2. Compute and write cryptographic manifest covering the final metadata
        self.write_manifest(run_id)
        return updated

    def get_run(self, run_id: str) -> RunMetadata | None:
        """Read and validate RunMetadata for an existing run.

        Args:
            run_id: Target experiment run ID.

        Returns:
            RunMetadata if run exists and is valid, None otherwise.
        """
        run_dir = self._resolve_run_dir(run_id)
        meta_file = run_dir / "run_metadata.json"
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
