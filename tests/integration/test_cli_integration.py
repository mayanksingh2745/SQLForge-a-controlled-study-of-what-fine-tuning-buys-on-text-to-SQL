"""Integration tests for the complete SQLForge CLI workflow."""

from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main
from sqlforge.experiments.tracker import ExperimentTracker


def test_cli_experiment_workflow_integration(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test full CLI experiment init integration with custom artifact dir."""
    artifacts = tmp_path / "artifacts" / "runs"
    result = cli_runner.invoke(
        main,
        [
            "experiment",
            "init",
            "--name",
            "integration_exp",
            "--seed",
            "777",
            "--artifact-dir",
            str(artifacts),
        ],
    )
    assert result.exit_code == 0
    assert "[OK] Initialized experiment run:" in result.output
    assert "integration_exp" in result.output

    # Check generated files on disk
    tracker = ExperimentTracker(base_artifact_dir=artifacts)
    run_dirs = list(artifacts.glob("integration_exp_*"))
    assert len(run_dirs) == 1
    run_id = run_dirs[0].name
    run_record = tracker.get_run(run_id)
    assert run_record is not None
    assert run_record.seed == 777
    assert run_record.status == "running"
