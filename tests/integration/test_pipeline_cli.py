"""Integration tests for CLI pipeline commands."""

from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main


def test_cli_pipeline_mock_dry_run() -> None:
    """Verify that CLI pipeline mock --dry-run succeeds with exit code 0."""
    runner = CliRunner()
    result = runner.invoke(main, ["pipeline", "mock", "--dry-run"])

    assert result.exit_code == 0
    assert "SQLForge Mock Pipeline Harness" in result.output
    assert "DRY_RUN" in result.output
    assert "DISCLAIMER: Offline synthetic mock pipeline demonstration only" in result.output


def test_cli_pipeline_mock_full_run(tmp_path: Path) -> None:
    """Verify that CLI pipeline mock executes end-to-end and outputs summary."""
    runner = CliRunner()
    artifact_dir = tmp_path / "runs"
    result = runner.invoke(
        main,
        ["pipeline", "mock", "--artifact-dir", str(artifact_dir)],
    )

    assert result.exit_code == 0
    assert "COMPLETED" in result.output
    assert "PASSED (Strict)" in result.output
    assert "Pipeline execution completed and verified successfully" in result.output


def test_cli_pipeline_mock_fail_mode_exits_nonzero(tmp_path: Path) -> None:
    """Verify that simulated component failure exits with non-zero exit code."""
    runner = CliRunner()
    artifact_dir = tmp_path / "runs"
    result = runner.invoke(
        main,
        ["pipeline", "mock", "--artifact-dir", str(artifact_dir), "--fail-mode"],
    )

    assert result.exit_code != 0
    assert "FAILED" in result.output
    assert "Pipeline completed with errors or failed verification" in result.output


def test_cli_pipeline_mock_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    """Verify that passing an invalid config file fails with non-zero exit code."""
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text("invalid: yaml: [", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["pipeline", "mock", "--config", str(bad_config)],
    )

    assert result.exit_code != 0
