"""Integration tests for SQLForge prompt CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from sqlforge.cli import main


def test_cli_prompt_help() -> None:
    """Verify sqlforge prompt --help prints available subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "serialize" in result.output
    assert "assemble" in result.output


def test_cli_prompt_serialize_formats() -> None:
    """Verify sqlforge prompt serialize executes for DDL, compact, and JSON formats."""
    runner = CliRunner()

    # DDL format
    res_ddl = runner.invoke(main, ["prompt", "serialize", "--format", "ddl"])
    assert res_ddl.exit_code == 0
    assert "CREATE TABLE" in res_ddl.output

    # Compact format
    res_compact = runner.invoke(main, ["prompt", "serialize", "--format", "compact"])
    assert res_compact.exit_code == 0
    assert "Serialized Schema" in res_compact.output

    # JSON format
    res_json = runner.invoke(main, ["prompt", "serialize", "--format", "json"])
    assert res_json.exit_code == 0
    assert '"tables"' in res_json.output


def test_cli_prompt_assemble_zero_and_few_shot() -> None:
    """Verify sqlforge prompt assemble works with zero-shot and few-shot flags."""
    runner = CliRunner()

    # Zero-shot
    res_zero = runner.invoke(
        main,
        ["prompt", "assemble", "--k-shots", "0", "--format", "ddl"],
    )
    assert res_zero.exit_code == 0
    assert "Prompt Assembly Summary" in res_zero.output
    assert "k-shots Attached" in res_zero.output
    assert "CREATE TABLE" in res_zero.output

    # Few-shot k=3
    res_few = runner.invoke(
        main,
        ["prompt", "assemble", "--k-shots", "3", "--format", "compact"],
    )
    assert res_few.exit_code == 0
    assert "Retrieved Demos" in res_few.output
    assert "Example 1:" in res_few.output
