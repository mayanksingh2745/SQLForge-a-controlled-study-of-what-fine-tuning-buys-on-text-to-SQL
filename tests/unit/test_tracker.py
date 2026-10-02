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
