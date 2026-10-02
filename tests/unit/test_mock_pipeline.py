"""Unit tests for MockPipelineHarness execution, tracking, and verification."""

from pathlib import Path

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.pipeline.harness import MockPipelineHarness


def test_mock_pipeline_dry_run(temp_artifact_dir: Path) -> None:
    """Verify that dry-run mode validates configuration without creating artifacts."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    result = harness.run(dry_run=True)
    assert result.status == "dry_run"
    assert result.verified is True
    assert result.total_examples > 0

    # Ensure no directories were created in base_artifact_dir
    created_dirs = list(temp_artifact_dir.glob("*"))
    assert len(created_dirs) == 0


def test_mock_pipeline_end_to_end_success(temp_artifact_dir: Path) -> None:
    """Verify clean end-to-end execution, artifact logging, and cryptographic verification."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    result = harness.run()
    assert result.status == "completed"
    assert result.verified is True
    assert result.total_examples == 5
    assert result.anomalies_count == 2  # mock_spider_004 (syntax) & mock_spider_005 (timeout)

    run_dir = Path(result.artifact_dir)
    assert run_dir.is_dir()
    assert (run_dir / "run_metadata.json").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "generations.jsonl").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "eval_anomalies.json").is_file()
    assert (run_dir / "manifest.json").is_file()

    # Re-verify through tracker directly
    verification = tracker.verify_run(result.run_id, strict=True)
    assert verification["verified"] is True
    assert len(verification["missing_files"]) == 0
    assert len(verification["hash_mismatches"]) == 0
    assert len(verification["untracked_files"]) == 0


def test_mock_pipeline_collision_safety(temp_artifact_dir: Path) -> None:
    """Verify that attempting to rerun with an identical run_id fails without overwriting artifacts."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)
    fixed_run_id = "test_fixed_pipeline_run"

    res1 = harness.run(run_id=fixed_run_id)
    assert res1.status == "completed"

    with pytest.raises(FileExistsError, match="Collision detected"):
        harness.run(run_id=fixed_run_id)


def test_mock_pipeline_failure_mode_preserves_artifacts(temp_artifact_dir: Path) -> None:
    """Verify that unexpected component failure halts gracefully and preserves diagnostics."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    result = harness.run(fail_mode=True)
    assert result.status == "failed"

    run_dir = Path(result.artifact_dir)
    assert run_dir.is_dir()
    assert (run_dir / "run_metadata.json").is_file()
    assert (run_dir / "eval_anomalies.json").is_file()

    # Anomaly file should document the failure
    anomalies_text = (run_dir / "eval_anomalies.json").read_text(encoding="utf-8")
    assert "PIPELINE_ERROR" in anomalies_text


def test_mock_pipeline_invalid_config_fails_early(temp_artifact_dir: Path) -> None:
    """Verify that invalid configuration raises ValueError before any run directory is created."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    bad_config = {
        # Missing required experiment_id, description, model_id
        "seed": "not_an_int"
    }

    with pytest.raises(ValueError, match="Invalid experiment config"):
        harness.run(config=bad_config)

    # Ensure no run directory was created
    assert len(list(temp_artifact_dir.glob("*"))) == 0


def test_mock_pipeline_missing_config_file_raises(temp_artifact_dir: Path) -> None:
    """Verify that a missing config file raises FileNotFoundError."""
    tracker = ExperimentTracker(base_artifact_dir=temp_artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    with pytest.raises(FileNotFoundError):
        harness.run(config=temp_artifact_dir / "non_existent_config.yaml")
