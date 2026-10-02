"""Configuration management and typed validation for SQLForge."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LoggingSettings(BaseModel):
    """Logging configuration settings."""

    model_config = ConfigDict(frozen=True)

    level: str = Field(default="INFO")
    format: str = Field(default="rich")
    log_to_file: bool = Field(default=True)
    log_dir: str = Field(default="artifacts/runs/logs")


class TrackingSettings(BaseModel):
    """Experiment tracking provider and destination settings."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="local")
    artifact_dir: str = Field(default="artifacts/runs")
    wandb_project: str = Field(default="sqlforge-controlled-study")
    wandb_entity: str | None = Field(default=None)
    save_generations: bool = Field(default=True)


class ExecutionSafetySettings(BaseModel):
    """Safety bounds for query execution sandbox."""

    model_config = ConfigDict(frozen=True)

    enforce_read_only: bool = Field(default=True)
    single_statement_only: bool = Field(default=True)
    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_memory_mb: int = Field(default=1024, ge=128)
    max_row_limit: int = Field(default=5000, ge=1)
    allowed_dialects: list[str] = Field(default_factory=lambda: ["sqlite", "postgresql"])


class ReproducibilitySettings(BaseModel):
    """Reproducibility flags."""

    model_config = ConfigDict(frozen=True)

    deterministic_torch: bool = Field(default=False)
    record_environment: bool = Field(default=True)
    hash_configs: bool = Field(default=True)


class SystemDefaults(BaseModel):
    """Top-level defaults model corresponding to configs/defaults.yaml."""

    model_config = ConfigDict(frozen=True)

    project_name: str = Field(default="SQLForge")
    version: str = Field(default="0.1.0")
    seed: int = Field(default=42)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    execution_safety: ExecutionSafetySettings = Field(default_factory=ExecutionSafetySettings)
    reproducibility: ReproducibilitySettings = Field(default_factory=ReproducibilitySettings)


def get_default_config_dir() -> Path:
    """Resolve the default configs/ directory path relative to project root."""
    current = Path(__file__).resolve().parent
    # Check parent dirs for configs/
    for parent in [current, current.parent, current.parent.parent]:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    return Path("configs")


def load_yaml_config(file_path: Path | str) -> dict[str, Any]:
    """Load and parse a YAML file safely.

    Args:
        file_path: Absolute or relative path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the YAML formatting is invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration in {path} must parse into a mapping/dict, got {type(data)}"
        )

    return data


def load_system_defaults(config_dir: Path | str | None = None) -> SystemDefaults:
    """Load and validate system defaults from configs/defaults.yaml."""
    base_dir = Path(config_dir) if config_dir else get_default_config_dir()
    defaults_file = base_dir / "defaults.yaml"
    raw_data = load_yaml_config(defaults_file)
    return SystemDefaults(**raw_data)


def validate_all_configs(config_dir: Path | str | None = None) -> dict[str, Any]:
    """Validate all standard configuration files in configs/ directory.

    Returns:
        Dictionary mapping config filename to validation status or error message.
    """
    base_dir = Path(config_dir) if config_dir else get_default_config_dir()
    expected_files = [
        "defaults.yaml",
        "datasets.yaml",
        "models.yaml",
        "experiments.yaml",
        "evaluation.yaml",
    ]

    results: dict[str, Any] = {}
    for filename in expected_files:
        filepath = base_dir / filename
        if not filepath.exists():
            results[filename] = {"valid": False, "error": "File does not exist"}
            continue

        try:
            raw_data = load_yaml_config(filepath)
            # Perform targeted validation
            if filename == "defaults.yaml":
                SystemDefaults(**raw_data)
            results[filename] = {"valid": True, "path": str(filepath.resolve())}
        except (ValidationError, Exception) as exc:
            results[filename] = {"valid": False, "error": str(exc)}

    return results
