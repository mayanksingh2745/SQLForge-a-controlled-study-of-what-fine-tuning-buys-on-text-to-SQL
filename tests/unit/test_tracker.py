"""Unit tests for the local-first experiment tracker."""

from pathlib import Path

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.evaluation import EvaluationMetrics
from sqlforge.schemas.experiments import ExperimentConfig


def test_tracker_init_and_finish_run(temp_artifact_dir: Path) -> None:
    """Test full lifecycle of a run: init, log metrics, and finish."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)

    exp_config = ExperimentConfig(
        experiment_id="unit_test_run",
        description="Testing tracker lifecycle",
        seed=100,
        model_id="qwen25_coder_1_5b",
    )

    metadata = tracker.init_run(
        experiment_name="unit_test_run",
        config=exp_config,
        seed=100,
    )

    assert metadata.status == "running"
    assert metadata.seed == 100
    assert metadata.config_hash is not None

    run_dir = temp_artifact_dir / metadata.run_id
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run_metadata.json").is_file()

    # Log metrics
    metrics = EvaluationMetrics(
        total_examples=10,
        valid_sql_rate=1.0,
        exact_match_accuracy=0.8,
        execution_accuracy=0.9,
    )
    metrics_path = tracker.log_metrics(metadata.run_id, metrics)
    assert metrics_path.is_file()

    # Finish run
    finished_meta = tracker.finish_run(metadata.run_id, status="completed")
    assert finished_meta.status == "completed"

    # Read back
    retrieved = tracker.get_run(metadata.run_id)
    assert retrieved is not None
    assert retrieved.status == "completed"


def test_tracker_non_existent_run_raises(temp_artifact_dir: Path) -> None:
    """Test error raised when logging to non-existent run."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    with pytest.raises(FileNotFoundError):
        tracker.finish_run("non_existent_run_id")


def test_tracker_collision_safety(temp_artifact_dir: Path) -> None:
    """Verify that ExperimentTracker refuses to silently overwrite existing populated runs."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    fixed_id = "test_fixed_collision_run"

    # Initialize first run
    tracker.init_run(
        experiment_name="first_run",
        config={"key": "val1"},
        seed=42,
        run_id=fixed_id,
    )

    # Attempting to re-init with the same run_id must raise FileExistsError
    with pytest.raises(FileExistsError, match="Collision detected"):
        tracker.init_run(
            experiment_name="colliding_run",
            config={"key": "val2"},
            seed=99,
            run_id=fixed_id,
        )


def test_tracker_manifest_and_run_verification(temp_artifact_dir: Path) -> None:
    """Verify cryptographic manifest generation and run verification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(
        experiment_name="manifest_test",
        config={"model": "test"},
        seed=42,
    )

    # Log metrics and finish
    tracker.log_metrics(
        meta.run_id,
        {
            "total_examples": 1,
            "valid_sql_rate": 1.0,
            "exact_match_accuracy": 1.0,
            "execution_accuracy": 1.0,
        },
    )
    tracker.finish_run(meta.run_id, status="completed")

    # Run verification must succeed
    res = tracker.verify_run(meta.run_id)
    assert res["verified"] is True
    assert res["status"] == "completed"
    assert len(res["missing_files"]) == 0
    assert len(res["hash_mismatches"]) == 0

    # Tamper with config file to verify hash mismatch detection
    run_dir = temp_artifact_dir / meta.run_id
    config_file = run_dir / "config.yaml"
    config_file.write_text("tampered: true\n", encoding="utf-8")

    tampered_res = tracker.verify_run(meta.run_id)
    assert tampered_res["verified"] is False
    assert "config.yaml" in tampered_res["hash_mismatches"]


def test_tracker_generation_and_anomaly_logging(temp_artifact_dir: Path) -> None:
    """Verify logging of per-example generations and anomalies."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    meta = tracker.init_run(experiment_name="log_test", config={"test": True})

    # Log generation
    gen_file = tracker.log_generation(
        meta.run_id,
        {"example_id": "spider_01", "generated_sql": "SELECT 1;", "latency_ms": 12.5},
    )
    assert gen_file.is_file()
    assert "spider_01" in gen_file.read_text(encoding="utf-8")

    # Log anomaly
    anomaly_file = tracker.log_anomaly(
        meta.run_id,
        {"example_id": "spider_02", "error_code": "E09", "reason": "Query execution timed out"},
    )
    assert anomaly_file.is_file()
    assert "E09" in anomaly_file.read_text(encoding="utf-8")
