"""Integration tests for sqlforge scaling CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main


def test_cli_scaling_plan_exp05(tmp_path: Path) -> None:
    """Verify sqlforge scaling plan generates 15 planned conditions for EXP-05."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scaling",
            "plan",
            "--experiment",
            "exp05",
            "--output-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code == 0
    assert "Planned 15 run conditions" in result.output
    assert "exp05_n500_s42" in result.output
    assert "exp05_n7000_s44" in result.output

    plan_file = tmp_path / "scaling" / "exp05_plan.json"
    assert plan_file.exists()
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    assert len(data) == 15


def test_cli_scaling_plan_exp06(tmp_path: Path) -> None:
    """Verify sqlforge scaling plan generates 9 planned conditions for EXP-06."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scaling",
            "plan",
            "--experiment",
            "exp06",
            "--output-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code == 0
    assert "Planned 9 run conditions" in result.output
    plan_file = tmp_path / "scaling" / "exp06_plan.json"
    assert plan_file.exists()
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    assert len(data) == 9
    run_ids = {c["run_id"] for c in data}
    assert "exp06_human_only_s42" in run_ids
    assert "exp06_human_synthetic_50_50_s44" in run_ids


def test_cli_scaling_validate(tmp_path: Path) -> None:
    """Verify sqlforge scaling validate checks research contracts."""
    runner = CliRunner()
    result = runner.invoke(main, ["scaling", "validate", "--experiment", "exp05"])
    assert result.exit_code == 0
    assert "EXP-05-DATA-SCALING contracts validated successfully" in result.output


def test_cli_scaling_run_dry_run(tmp_path: Path) -> None:
    """Verify sqlforge scaling run --dry-run completes cleanly with 0 compute."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scaling",
            "run",
            "--experiment",
            "exp05",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--scaling-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code == 0
    assert "Completed dry-run simulation for 15 runs" in result.output
    summary_file = tmp_path / "scaling" / "exp_05_data_scaling" / "scaling_summary.json"
    assert summary_file.exists()


def test_cli_scaling_run_without_execute_flag(tmp_path: Path) -> None:
    """Verify sqlforge scaling run fails safely when neither --dry-run nor --execute is provided."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scaling",
            "run",
            "--experiment",
            "exp05",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--scaling-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code != 0
    assert (
        "requires --execute" in result.output or "requires explicit confirmation" in result.output
    )


def test_cli_scaling_status(tmp_path: Path) -> None:
    """Verify sqlforge scaling status displays condition states."""
    runner = CliRunner()
    # First plan so status has something to show
    runner.invoke(
        main,
        [
            "scaling",
            "plan",
            "--experiment",
            "exp05",
            "--output-dir",
            str(tmp_path / "scaling"),
        ],
    )
    result = runner.invoke(
        main,
        [
            "scaling",
            "status",
            "--experiment",
            "exp05",
            "--output-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code == 0
    assert "Status: planned" in result.output


def test_cli_scaling_analyze_rejects_missing_data(tmp_path: Path) -> None:
    """Verify sqlforge scaling analyze fails with exit code 2 when empirical data is unavailable."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scaling",
            "analyze",
            "--experiment",
            "exp05",
            "--scaling-dir",
            str(tmp_path / "scaling"),
        ],
    )
    assert result.exit_code == 2
    assert "No verified empirical results are available" in result.output
