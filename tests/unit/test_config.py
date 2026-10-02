"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sqlforge.settings import (
    SystemDefaults,
    load_system_defaults,
    load_yaml_config,
    validate_all_configs,
)


def test_load_system_defaults() -> None:
    """Ensure system defaults load and pass Pydantic validation."""
    defaults = load_system_defaults()
    assert isinstance(defaults, SystemDefaults)
    assert defaults.project_name == "SQLForge"
    assert defaults.seed == 42
    assert defaults.execution_safety.enforce_read_only is True
    assert defaults.execution_safety.timeout_seconds > 0


def test_load_yaml_config_missing_file(tmp_path: Path) -> None:
    """Ensure FileNotFoundError is raised for non-existent configs."""
    missing = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_config(missing)


def test_load_yaml_config_invalid_format(tmp_path: Path) -> None:
    """Ensure ValueError is raised if YAML does not parse to dict."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must parse into a mapping"):
        load_yaml_config(bad_yaml)


def test_system_defaults_validation_error() -> None:
    """Ensure validation fails when types or constraints are violated."""
    with pytest.raises(ValidationError):
        SystemDefaults(
            execution_safety={"timeout_seconds": -5.0},  # Must be gt=0.0
        )


def test_validate_all_configs() -> None:
    """Ensure all stock configuration files pass validation."""
    results = validate_all_configs()
    assert "defaults.yaml" in results
    assert "datasets.yaml" in results
    assert "models.yaml" in results
    assert "experiments.yaml" in results
    assert "evaluation.yaml" in results
    for filename, info in results.items():
        assert info["valid"] is True, f"{filename} failed validation: {info.get('error')}"
