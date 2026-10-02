"""Unit tests for the SQLForge CLI."""

from click.testing import CliRunner

from sqlforge import __version__
from sqlforge.cli import main


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test 'sqlforge --version' displays correct package version."""
    result = cli_runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_help(cli_runner: CliRunner) -> None:
    """Test 'sqlforge --help' renders usage and commands."""
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "SQLForge" in result.output
    assert "config" in result.output
    assert "env" in result.output
    assert "experiment" in result.output


def test_cli_env(cli_runner: CliRunner) -> None:
    """Test 'sqlforge env' outputs system diagnostic tables."""
    result = cli_runner.invoke(main, ["env"])
    assert result.exit_code == 0
    assert "Host Environment" in result.output
    assert "Operating System" in result.output
    assert "Python Version" in result.output
    assert "Git Version Control Status" in result.output


def test_cli_config_validate(cli_runner: CliRunner) -> None:
    """Test 'sqlforge config validate' validates default configs."""
    result = cli_runner.invoke(main, ["config", "validate"])
    assert result.exit_code == 0
    assert "defaults.yaml" in result.output
    assert "PASSED" in result.output
