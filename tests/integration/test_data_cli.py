"""Integration tests for SQLForge data CLI commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main


def test_cli_data_help() -> None:
    """Verify sqlforge data --help prints available subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["data", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output
    assert "audit" in result.output
    assert "manifest" in result.output


def test_cli_data_validate_default_fixtures() -> None:
    """Verify sqlforge data validate validates offline test fixtures cleanly."""
    runner = CliRunner()
    result = runner.invoke(main, ["data", "validate"])
    assert result.exit_code == 0
    assert "Dataset Validation Results" in result.output
    assert "Spider 1.0" in result.output
    assert "PASSED" in result.output


def test_cli_data_audit_clean_and_report_output() -> None:
    """Verify sqlforge data audit executes clean audit and writes report file."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = Path(tmpdir) / "test_audit_report.json"
        result = runner.invoke(main, ["data", "audit", "--output-report", str(report_file)])
        assert result.exit_code == 0
        assert "PASSED (Zero Leakage)" in result.output
        assert report_file.is_file()


def test_cli_data_audit_contaminated_failure() -> None:
    """Verify sqlforge data audit aborts with non-zero exit code when contamination is detected."""
    runner = CliRunner()
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures" / "dataset"
    bad_train = fixtures_dir / "contaminated" / "train_contaminated.json"

    result = runner.invoke(main, ["data", "audit", "--train", str(bad_train)])
    assert result.exit_code != 0
    assert "FAILED (Leakage Detected)" in result.output
