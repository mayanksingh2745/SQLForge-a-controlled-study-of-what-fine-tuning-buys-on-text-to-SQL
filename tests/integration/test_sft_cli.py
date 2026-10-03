"""Integration tests for CLI train command group (preflight, validate, run, inspect)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sqlforge.cli import main


class TestSFTCli:
    """Test suite for 'sqlforge train' CLI command group."""

    def test_train_help(self) -> None:
        """Verify train --help shows available subcommands."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "preflight" in result.output
        assert "validate" in result.output
        assert "run" in result.output
        assert "inspect" in result.output

    def test_train_preflight_cli(self) -> None:
        """Verify train preflight runs and outputs diagnostic table."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "preflight", "--method", "lora"])
        assert result.exit_code == 0
        assert "SQLForge SFT Pre-Flight Feasibility Check" in result.output
        assert "Overall Status:" in result.output
        assert "CUDA Available:" in result.output

    def test_train_validate_cli_defaults(self) -> None:
        """Verify train validate parses and displays default training specification."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "validate", "--method", "lora", "--rank", "16"])
        assert result.exit_code == 0
        assert "Configuration Validated Successfully" in result.output
        assert "LoRA Rank (r)" in result.output
        assert "16" in result.output

    def test_train_validate_cli_qlora(self) -> None:
        """Verify train validate displays QLoRA parameters."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "validate", "--method", "qlora"])
        assert result.exit_code == 0
        assert "QLoRA Quant Bits" in result.output
        assert "4-bit (nf4)" in result.output

    def test_train_run_requires_confirmation(self) -> None:
        """Verify that running without --execute or --dry-run aborts with safety guard message."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "run"])
        assert result.exit_code != 0
        assert "Safety Guard:" in result.output
        assert "--execute" in result.output

    def test_train_run_dry_run_cli(self) -> None:
        """Verify train run --dry-run completes successfully without downloading weights."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "run", "--dry-run", "--method", "lora"])
        assert result.exit_code == 0
        assert "Run Completed (DRY_RUN)" in result.output
        assert "Execution Summary" in result.output

    def test_train_inspect_cli(self, tmp_path: Path) -> None:
        """Verify train inspect reads checkpoint_metadata.json and renders table."""
        ckpt_dir = tmp_path / "mock_adapter"
        ckpt_dir.mkdir()

        meta = {
            "run_id": "test_inspect_run",
            "step": 250,
            "epoch": 2.5,
            "base_model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "base_model_revision": "main",
            "adapter_summary": {
                "method": "lora",
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "training_metrics": {"loss": 0.42, "lr": 1e-4},
            "dataset_info": {},
            "seed": 42,
            "git_commit": "abcdef123456",
            "git_dirty": False,
            "python_version": "3.12.0",
            "platform_details": "Linux",
            "checkpoint_type": "adapter_only",
            "adapter_files": ["adapter_config.json"],
        }
        with open(ckpt_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        runner = CliRunner()
        result = runner.invoke(main, ["train", "inspect", str(ckpt_dir)])
        assert result.exit_code == 0
        assert "Checkpoint Metadata Inspection" in result.output
        assert "test_inspect_run" in result.output
        assert "250" in result.output
        assert "0.42" in result.output
