"""Integration tests for CLI sweep command group (plan, validate, run, status, plot)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main


class TestSweepCli:
    """Test suite for 'sqlforge sweep' CLI command group."""

    def test_sweep_help(self) -> None:
        """Verify sweep --help lists all subcommands."""
        runner = CliRunner()
        result = runner.invoke(main, ["sweep", "--help"])
        assert result.exit_code == 0
        assert "plan" in result.output
        assert "validate" in result.output
        assert "run" in result.output
        assert "status" in result.output
        assert "plot" in result.output

    def test_sweep_plan_cli(self) -> None:
        """Verify sweep plan displays planned runs and estimates."""
        runner = CliRunner()
        result = runner.invoke(main, ["sweep", "plan", "--ranks", "8,16", "--seeds", "42"])
        assert result.exit_code == 0
        assert "SQLForge LoRA Rank Sweep Plan" in result.output
        assert "exp04_r8_a16" in result.output
        assert "exp04_r16_a32" in result.output

    def test_sweep_validate_cli(self) -> None:
        """Verify sweep validate displays controlled invariant variables table."""
        runner = CliRunner()
        result = runner.invoke(main, ["sweep", "validate", "--ranks", "8,16,32,64"])
        assert result.exit_code == 0
        assert "Sweep Configuration Validated Successfully" in result.output
        assert "Controlled Invariant Variables" in result.output
        assert "r=8, r=16, r=32, r=64" in result.output

    def test_sweep_run_requires_confirmation(self) -> None:
        """Verify sweep run requires --execute or --dry-run flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["sweep", "run", "--ranks", "8,16"])
        assert result.exit_code != 0
        assert "Safety Guard:" in result.output
        assert "--execute" in result.output

    def test_sweep_run_dry_run_cli(self, tmp_path: Path) -> None:
        """Verify sweep run --dry-run completes cleanly."""
        runner = CliRunner()
        runs_dir = tmp_path / "runs"
        sweeps_dir = tmp_path / "sweeps"
        result = runner.invoke(
            main,
            [
                "sweep",
                "run",
                "--ranks",
                "8,16",
                "--dry-run",
                "--output-dir",
                str(runs_dir),
                "--sweep-dir",
                str(sweeps_dir),
            ],
        )
        assert result.exit_code == 0
        assert "SQLForge LoRA Rank Sweep Execution" in result.output
        assert "DRY RUN" in result.output
        assert "Sweep Completed" in result.output
        assert "Runs: 2 completed" in result.output
        assert "exp04_r8_a16" in result.output
        assert "exp04_r16_a32" in result.output

    def test_sweep_status_and_plot_no_empirical_data(self, tmp_path: Path) -> None:
        """Verify sweep status displays run table and plot warns about missing empirical data."""
        # Create a mock dry-run summary file
        summary_path = tmp_path / "mock_summary.json"
        summary_data = {
            "sweep_id": "sweep_test_123",
            "experiment_id": "EXP-04-RANK-SWEEP",
            "base_model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "method": "qlora",
            "total_planned": 2,
            "completed": 2,
            "cached": 0,
            "failed": 0,
            "skipped": 0,
            "is_dry_run": True,
            "created_at": "2026-10-04T00:00:00Z",
            "results": [
                {
                    "run_id": "exp04_r8_a16_all_linear_s42",
                    "rank": 8,
                    "alpha": 16,
                    "target_modules_tag": "all-linear",
                    "target_modules": ["q_proj", "v_proj"],
                    "seed": 42,
                    "status": "dry_run",
                    "trainable_parameters": 4_000_000,
                    "adapter_size_mb": 8.0,
                    "duration_seconds": 1.2,
                    "metrics": {},
                },
                {
                    "run_id": "exp04_r16_a32_all_linear_s42",
                    "rank": 16,
                    "alpha": 32,
                    "target_modules_tag": "all-linear",
                    "target_modules": ["q_proj", "v_proj"],
                    "seed": 42,
                    "status": "dry_run",
                    "trainable_parameters": 8_000_000,
                    "adapter_size_mb": 16.0,
                    "duration_seconds": 1.4,
                    "metrics": {},
                },
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f)

        runner = CliRunner()
        # Test status command
        status_res = runner.invoke(main, ["sweep", "status", str(summary_path)])
        assert status_res.exit_code == 0
        assert "SQLForge Sweep Status" in status_res.output
        assert "exp04_r8_a16_all_linear_s42" in status_res.output

        # Test plot command with no empirical data
        plot_res = runner.invoke(main, ["sweep", "plot", "--summary-file", str(summary_path)])
        assert plot_res.exit_code == 0
        assert "No Empirical Evaluation Data" in plot_res.output
        assert "No illustrative or fabricated curves were generated" in plot_res.output

    def test_sweep_plot_with_empirical_data(self, tmp_path: Path) -> None:
        """Verify sweep plot generates SVG figure when empirical data is present."""
        summary_path = tmp_path / "empirical_summary.json"
        out_svg = tmp_path / "pareto_out.svg"
        out_csv = tmp_path / "table_out.csv"

        summary_data = {
            "sweep_id": "sweep_emp_456",
            "experiment_id": "EXP-04-RANK-SWEEP",
            "base_model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "method": "qlora",
            "total_planned": 2,
            "completed": 2,
            "cached": 0,
            "failed": 0,
            "skipped": 0,
            "is_dry_run": False,
            "created_at": "2026-10-04T00:00:00Z",
            "results": [
                {
                    "run_id": "exp04_r8_a16_s42",
                    "rank": 8,
                    "alpha": 16,
                    "target_modules_tag": "all-linear",
                    "target_modules": ["q_proj"],
                    "seed": 42,
                    "status": "completed",
                    "trainable_parameters": 4_000_000,
                    "adapter_size_mb": 8.0,
                    "duration_seconds": 120.0,
                    "metrics": {"execution_accuracy": 0.69},
                },
                {
                    "run_id": "exp04_r16_a32_s42",
                    "rank": 16,
                    "alpha": 32,
                    "target_modules_tag": "all-linear",
                    "target_modules": ["q_proj"],
                    "seed": 42,
                    "status": "completed",
                    "trainable_parameters": 8_000_000,
                    "adapter_size_mb": 16.0,
                    "duration_seconds": 140.0,
                    "metrics": {"execution_accuracy": 0.74},
                },
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f)

        runner = CliRunner()
        plot_res = runner.invoke(
            main,
            [
                "sweep",
                "plot",
                "--summary-file",
                str(summary_path),
                "--output",
                str(out_svg),
                "--export-csv",
                str(out_csv),
            ],
        )
        assert plot_res.exit_code == 0
        assert "Pareto Plot Generated Successfully" in plot_res.output
        assert out_svg.exists()
        assert out_csv.exists()
