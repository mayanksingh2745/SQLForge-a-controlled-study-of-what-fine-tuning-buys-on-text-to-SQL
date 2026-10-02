"""Integration tests for baseline CLI commands (run, verify, stats)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main
from sqlforge.experiments.tracker import ExperimentTracker


def test_baseline_run_dry_run_cli(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        main,
        ["baseline", "run", "--model", "qwen25_coder_1_5b", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Baseline Execution Summary" in result.output
    assert "DRY_RUN" in result.output
    assert "PASSED" in result.output


def test_baseline_run_frontier_api_ref_dry_run(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(
        main,
        ["baseline", "run", "--model", "frontier_api_ref", "--k-shots", "3", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "gpt-4o-mini-2024-07-18" in result.output
    assert "DRY_RUN" in result.output


def test_baseline_run_opt_in_protection_fails_cleanly(cli_runner: CliRunner) -> None:
    # Running frontier API without --live-api and without --dry-run must fail
    result = cli_runner.invoke(
        main,
        ["baseline", "run", "--model", "frontier_api_ref"],
    )
    assert result.exit_code != 0
    assert (
        "Live commercial API execution is disabled by default" in result.output
        or "FAILED" in result.output
    )


def test_baseline_verify_cli(cli_runner: CliRunner, tmp_path: Path) -> None:
    # Initialize and complete a small synthetic run using the tracker directly
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
    _ = tracker.init_run("verify_test", {"model": "test"}, run_id="run_verify_001")
    tracker.log_metrics(
        "run_verify_001",
        {
            "total_examples": 1,
            "execution_accuracy": 1.0,
            "exact_match_accuracy": 1.0,
        },
    )
    tracker.finish_run("run_verify_001", status="completed")

    result = cli_runner.invoke(
        main,
        ["baseline", "verify", "run_verify_001", "--artifact-dir", str(tmp_path / "runs")],
    )
    assert result.exit_code == 0
    assert "Verifying Baseline Run: run_verify_001" in result.output
    assert "PASSED" in result.output


def test_baseline_stats_cli_paired_comparison(cli_runner: CliRunner, tmp_path: Path) -> None:
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")

    # Create run A
    tracker.init_run("exp_a", {"model": "model_a"}, run_id="run_a")
    tracker.log_generation(
        "run_a",
        {
            "example_id": "ex1",
            "generated_sql": "SELECT 1;",
            "gold_sql": "SELECT 1;",
            "latency_ms": 10.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )
    tracker.log_generation(
        "run_a",
        {
            "example_id": "ex2",
            "generated_sql": "SELECT 2;",
            "gold_sql": "SELECT 2;",
            "latency_ms": 10.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )
    tracker.finish_run("run_a", status="completed")

    # Create run B
    tracker.init_run("exp_b", {"model": "model_b"}, run_id="run_b")
    tracker.log_generation(
        "run_b",
        {
            "example_id": "ex1",
            "generated_sql": "SELECT 1;",
            "gold_sql": "SELECT 1;",
            "latency_ms": 10.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )
    tracker.log_generation(
        "run_b",
        {
            "example_id": "ex2",
            "generated_sql": "SELECT 999;",  # wrong
            "gold_sql": "SELECT 2;",
            "latency_ms": 10.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )
    tracker.finish_run("run_b", status="completed")

    result = cli_runner.invoke(
        main,
        [
            "baseline",
            "stats",
            "run_a",
            "--compare-to",
            "run_b",
            "--artifact-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0
    assert "Paired Comparison: run_a vs run_b" in result.output
    assert "Common Evaluated Examples (N)" in result.output
    assert "McNemar" in result.output
