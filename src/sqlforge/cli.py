"""Command Line Interface for SQLForge."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sqlforge import __version__
from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.logging_config import configure_logging
from sqlforge.settings import (
    get_default_config_dir,
    load_system_defaults,
    load_yaml_config,
    validate_all_configs,
)
from sqlforge.utils.env_info import collect_system_diagnostics

console = Console()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-v", "--version", message="SQLForge %(version)s")
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    """SQLForge: Controlled study of what fine-tuning buys on text-to-SQL systems."""
    configure_logging(level=log_level)


@main.command("env")
def env_cmd() -> None:
    """Inspect environment, hardware, Python, GPU, and Git diagnostics."""
    diagnostics = collect_system_diagnostics()

    console.print("\n[bold cyan]SQLForge System & Hardware Diagnostics[/bold cyan]")
    console.print("=" * 60)

    # System table
    sys_table = Table(title="Host Environment", show_header=True, header_style="bold magenta")
    sys_table.add_column("Property", style="dim", width=24)
    sys_table.add_column("Value")

    sys_table.add_row("Operating System", str(diagnostics.get("platform")))
    sys_table.add_row("System Architecture", str(diagnostics.get("architecture")))
    sys_table.add_row("Python Version", str(diagnostics.get("python_version")))
    sys_table.add_row("Logical CPU Cores", str(diagnostics.get("cpu_count")))
    sys_table.add_row("Physical Memory (RAM)", f"{diagnostics.get('total_ram_gb', 0.0)} GB")

    gpu_info = diagnostics.get("gpu", {})
    cuda_status = (
        "[green]Available[/green]"
        if gpu_info.get("cuda_available")
        else "[yellow]Not Available (CPU mode)[/yellow]"
    )
    sys_table.add_row("CUDA Acceleration", cuda_status)
    sys_table.add_row("GPU Devices Detected", str(gpu_info.get("device_count", 0)))
    if gpu_info.get("device_names"):
        sys_table.add_row("GPU Models", ", ".join(gpu_info.get("device_names", [])))

    console.print(sys_table)

    # Git table
    git_info = diagnostics.get("git", {})
    git_table = Table(
        title="Git Version Control Status", show_header=True, header_style="bold magenta"
    )
    git_table.add_column("Property", style="dim", width=24)
    git_table.add_column("Value")

    git_table.add_row("HEAD Commit", str(git_info.get("commit") or "N/A"))
    git_table.add_row("Current Branch", str(git_info.get("branch") or "N/A"))
    dirty_val = (
        "[red]Dirty (uncommitted changes)[/red]"
        if git_info.get("is_dirty")
        else "[green]Clean[/green]"
    )
    git_table.add_row("Working Tree Status", dirty_val)

    console.print(git_table)

    # Packages table
    pkg_table = Table(
        title="Installed Toolchain & Key Packages", show_header=True, header_style="bold magenta"
    )
    pkg_table.add_column("Package", style="dim", width=24)
    pkg_table.add_column("Status / Version")

    for pkg, status in diagnostics.get("packages", {}).items():
        style = "green" if status != "not installed" else "dim red"
        pkg_table.add_row(pkg, f"[{style}]{status}[/{style}]")

    console.print(pkg_table)


@main.group("config")
def config_group() -> None:
    """Manage and validate YAML configuration files."""
    pass


@config_group.command("validate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a specific YAML config file to validate.",
)
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing YAML configuration files.",
)
def config_validate_cmd(config_path: Path | None, config_dir: Path | None) -> None:
    """Validate YAML configuration files against typed contracts."""
    if config_path:
        console.print(f"Validating configuration file: [cyan]{config_path}[/cyan]")
        try:
            data = load_yaml_config(config_path)
            console.print(
                f"[green][OK] Successfully parsed and validated {config_path.name}[/green]"
            )
            console.print(f"Top-level keys: {list(data.keys())}")
        except Exception as exc:
            console.print(f"[red][FAILED] Validation failed for {config_path.name}: {exc}[/red]")
            raise click.Abort() from exc
        return

    target_dir = config_dir or get_default_config_dir()
    console.print(f"Validating standard configurations in: [cyan]{target_dir}[/cyan]")

    results = validate_all_configs(target_dir)
    table = Table(
        title="Configuration Validation Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Configuration File", width=25)
    table.add_column("Status", width=12)
    table.add_column("Details")

    all_valid = True
    for filename, res in results.items():
        if res.get("valid"):
            table.add_row(filename, "[green]PASSED[/green]", "Schema matches expected structure")
        else:
            all_valid = False
            table.add_row(filename, "[red]FAILED[/red]", str(res.get("error")))

    console.print(table)
    if not all_valid:
        raise click.Abort()


@main.group("experiment")
def experiment_group() -> None:
    """Initialize and inspect experiment runs."""
    pass


@experiment_group.command("init")
@click.option("--name", default="pilot_run", help="Experiment name or prefix")
@click.option("--seed", default=42, type=int, help="Random seed for the run")
@click.option(
    "--artifact-dir",
    default="artifacts/runs",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where experiment run artifacts are stored",
)
def experiment_init_cmd(name: str, seed: int, artifact_dir: Path) -> None:
    """Initialize a reproducible experiment run record and config snapshot."""
    tracker = ExperimentTracker(base_artifact_dir=artifact_dir)

    try:
        defaults = load_system_defaults()
        config_snapshot = defaults.model_dump()
    except Exception:
        config_snapshot = {"experiment_name": name, "seed": seed}

    metadata = tracker.init_run(
        experiment_name=name,
        config=config_snapshot,
        seed=seed,
    )

    console.print(f"[green][OK] Initialized experiment run:[/green] [bold]{metadata.run_id}[/bold]")
    console.print(f"  Directory: {tracker.base_dir / metadata.run_id}")
    console.print(f"  Timestamp: {metadata.timestamp_utc}")
    console.print(f"  Config Hash: {metadata.config_hash}")


@main.group("data")
def data_group() -> None:
    """Ingest, validate, audit, and manifest text-to-SQL datasets."""
    pass


@data_group.command("validate")
@click.option(
    "--spider-tables",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to Spider tables.json file.",
)
@click.option(
    "--spider-train",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to Spider train JSON file.",
)
@click.option(
    "--spider-dev",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to Spider dev JSON file.",
)
@click.option(
    "--bird-tables",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to BIRD dev_tables.json file.",
)
@click.option(
    "--bird-dev",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to BIRD dev JSON file.",
)
@click.option(
    "--bird-variant",
    type=click.Choice(["mini_dev_500", "mini_dev_780"]),
    default="mini_dev_500",
    help="BIRD Mini-Dev subset variant.",
)
@click.option(
    "--held-out-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to custom held-out JSON file.",
)
def data_validate_cmd(
    spider_tables: Path | None,
    spider_train: Path | None,
    spider_dev: Path | None,
    bird_tables: Path | None,
    bird_dev: Path | None,
    bird_variant: str,
    held_out_file: Path | None,
) -> None:
    """Validate dataset files, schema definitions, and record contracts."""
    from sqlforge.data.bird import BirdSubsetVariant, load_bird_mini_dev, parse_bird_tables
    from sqlforge.data.held_out import get_subscription_analytics_schema, load_held_out_dataset
    from sqlforge.data.spider import load_spider_split, parse_spider_tables
    from sqlforge.schemas.examples import DatasetSplit

    console.print("\n[bold cyan]SQLForge Dataset Ingestion & Contract Validation[/bold cyan]")

    results_table = Table(
        title="Dataset Validation Results", show_header=True, header_style="bold magenta"
    )
    results_table.add_column("Dataset Component", style="dim", width=26)
    results_table.add_column("Status", width=12)
    results_table.add_column("Schemas / Examples", width=22)
    results_table.add_column("Details")

    # If no options provided, use offline test fixtures
    root = Path(__file__).resolve().parent.parent.parent
    fixtures_dir = root / "tests" / "fixtures" / "dataset"

    # Spider validation
    sp_tbl = spider_tables or (fixtures_dir / "spider" / "tables.json")
    sp_tr = spider_train or (fixtures_dir / "spider" / "train_spider.json")
    sp_dv = spider_dev or (fixtures_dir / "spider" / "dev.json")

    if sp_tbl.is_file():
        try:
            sp_schemas = parse_spider_tables(sp_tbl)
            tr_ex = (
                load_spider_split(sp_tr, sp_schemas, split=DatasetSplit.TRAIN)
                if sp_tr.is_file()
                else []
            )
            dv_ex = (
                load_spider_split(sp_dv, sp_schemas, split=DatasetSplit.DEV)
                if sp_dv.is_file()
                else []
            )
            results_table.add_row(
                "Spider 1.0",
                "[green]PASSED[/green]",
                f"{len(sp_schemas)} schemas, {len(tr_ex)} train, {len(dv_ex)} dev",
                f"Tables: {sp_tbl.name}, Train: {sp_tr.name if sp_tr.is_file() else 'N/A'}",
            )
        except Exception as exc:
            results_table.add_row("Spider 1.0", "[red]FAILED[/red]", "Error", str(exc))

    # BIRD validation
    b_tbl = bird_tables or (fixtures_dir / "bird" / "dev_tables.json")
    b_dv = bird_dev or (fixtures_dir / "bird" / "dev.json")

    if b_tbl.is_file() and b_dv.is_file():
        try:
            b_schemas = parse_bird_tables(b_tbl)
            variant_enum = BirdSubsetVariant(bird_variant)
            b_ex = load_bird_mini_dev(
                b_dv, b_schemas, variant=variant_enum, enforce_expected_count=False
            )
            results_table.add_row(
                f"BIRD ({bird_variant})",
                "[green]PASSED[/green]",
                f"{len(b_schemas)} schemas, {len(b_ex)} examples",
                f"Tables: {b_tbl.name}, Dev: {b_dv.name} (CC BY-NC-SA 4.0)",
            )
        except Exception as exc:
            results_table.add_row(f"BIRD ({bird_variant})", "[red]FAILED[/red]", "Error", str(exc))

    # Custom held-out validation
    h_file = held_out_file or (fixtures_dir / "custom_held_out" / "held_out_examples.json")
    if h_file.is_file():
        try:
            _ = get_subscription_analytics_schema()
            h_ex = load_held_out_dataset(h_file)
            results_table.add_row(
                "Custom Held-Out",
                "[green]PASSED[/green]",
                f"1 schema (6 tables), {len(h_ex)} examples",
                f"File: {h_file.name} (Apache-2.0, isolated)",
            )
        except Exception as exc:
            results_table.add_row("Custom Held-Out", "[red]FAILED[/red]", "Error", str(exc))

    console.print(results_table)


@data_group.command("audit")
@click.option(
    "--train",
    "train_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to training examples (JSON or JSONL).",
)
@click.option(
    "--dev",
    "dev_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to dev examples (JSON or JSONL).",
)
@click.option(
    "--held-out",
    "held_out_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to custom held-out examples (JSON or JSONL).",
)
@click.option(
    "--output-report",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to write machine-readable contamination report JSON.",
)
@click.option(
    "--n-gram-size",
    default=4,
    type=int,
    help="Word n-gram size for fuzzy overlap (default: 4).",
)
@click.option(
    "--threshold",
    default=0.85,
    type=float,
    help="Fuzzy overlap similarity threshold [0.0, 1.0] (default: 0.85).",
)
def data_audit_cmd(
    train_path: Path | None,
    dev_path: Path | None,
    held_out_path: Path | None,
    output_report: Path | None,
    n_gram_size: int,
    threshold: float,
) -> None:
    """Audit datasets for cross-partition duplicates, fuzzy overlap, and schema leakage."""
    from sqlforge.data.audit import ContaminationAuditor
    from sqlforge.data.held_out import load_held_out_dataset
    from sqlforge.data.manifest import load_examples_from_file
    from sqlforge.data.spider import load_spider_split, parse_spider_tables
    from sqlforge.schemas.examples import DatasetSplit

    console.print("\n[bold cyan]SQLForge Contamination & Partition Leakage Audit[/bold cyan]")

    root = Path(__file__).resolve().parent.parent.parent
    fixtures_dir = root / "tests" / "fixtures" / "dataset"

    # Resolve train examples
    if train_path:
        train_examples = load_examples_from_file(train_path)
    else:
        sp_tbl = fixtures_dir / "spider" / "tables.json"
        sp_tr = fixtures_dir / "spider" / "train_spider.json"
        schemas = parse_spider_tables(sp_tbl)
        train_examples = load_spider_split(sp_tr, schemas, split=DatasetSplit.TRAIN)

    # Resolve eval partitions
    eval_partitions = {}
    if dev_path:
        eval_partitions["dev"] = load_examples_from_file(dev_path)
    else:
        sp_tbl = fixtures_dir / "spider" / "tables.json"
        sp_dv = fixtures_dir / "spider" / "dev.json"
        schemas = parse_spider_tables(sp_tbl)
        eval_partitions["dev"] = load_spider_split(sp_dv, schemas, split=DatasetSplit.DEV)

    if held_out_path:
        eval_partitions["held_out"] = load_examples_from_file(held_out_path)
    else:
        h_file = fixtures_dir / "custom_held_out" / "held_out_examples.json"
        if h_file.is_file():
            eval_partitions["held_out"] = load_held_out_dataset(h_file)

    auditor = ContaminationAuditor(n_gram_size=n_gram_size, fuzzy_threshold=threshold)
    report = auditor.audit_partitions(train_examples, eval_partitions)

    audit_table = Table(
        title="Contamination Audit Summary", show_header=True, header_style="bold magenta"
    )
    audit_table.add_column("Audit Metric", style="dim", width=32)
    audit_table.add_column("Result")

    audit_table.add_row(
        "Audit Status",
        "[green]PASSED (Zero Leakage)[/green]"
        if report.passed
        else "[red]FAILED (Leakage Detected)[/red]",
    )
    audit_table.add_row("Total Cross-Pairs Evaluated", str(report.total_pairs_evaluated))
    audit_table.add_row("Exact Question Duplicates", str(len(report.exact_question_duplicates)))
    audit_table.add_row("Exact SQL Duplicates", str(len(report.exact_sql_duplicates)))
    audit_table.add_row(
        f"Fuzzy Overlaps (score >= {threshold})", str(len(report.fuzzy_question_overlaps))
    )
    audit_table.add_row(
        "Schema Disjointness Violations", str(len(report.schema_disjointness_violations))
    )
    audit_table.add_row("Split Leakage Violations", str(len(report.split_leakage_violations)))

    console.print(audit_table)
    console.print(f"Summary: [dim]{report.summary}[/dim]\n")

    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2) + "\n")
        console.print(f"[green]Saved machine-readable audit report to:[/green] {out_p}")

    if not report.passed:
        raise click.Abort()


@data_group.command("manifest")
@click.option(
    "--verify",
    "manifest_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to dataset_manifest.json to verify.",
)
@click.option(
    "--project-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root directory for resolving relative paths in manifest.",
)
def data_manifest_cmd(
    manifest_file: Path,
    project_root: Path | None,
) -> None:
    """Verify cryptographic integrity of dataset manifest files."""
    from sqlforge.data.manifest import verify_dataset_manifest

    console.print(f"Verifying dataset manifest: [cyan]{manifest_file}[/cyan]")
    res = verify_dataset_manifest(manifest_file, project_root=project_root)

    table = Table(
        title="Manifest Verification Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Status", "[green]VERIFIED[/green]" if res["verified"] else "[red]FAILED[/red]")
    table.add_row("Dataset ID", str(res.get("dataset_id", "N/A")))
    table.add_row("Total Examples", str(res.get("total_examples", 0)))
    table.add_row("Missing Files", str(len(res.get("missing_files", []))))
    table.add_row("Hash Mismatches", str(len(res.get("hash_mismatches", []))))

    console.print(table)

    if not res["verified"]:
        for err in res.get("errors", []):
            console.print(f"[red]Error:[/red] {err}")
        for mf in res.get("missing_files", []):
            console.print(f"[red]Missing file:[/red] {mf}")
        for hm in res.get("hash_mismatches", []):
            console.print(f"[red]Hash mismatch:[/red] {hm}")
        raise click.Abort()


@main.group("pipeline")
def pipeline_group() -> None:
    """Execute end-to-end experiment pipelines and test harnesses."""
    pass


@pipeline_group.command("mock")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML experiment config file (defaults to configs/mock_pipeline.yaml).",
)
@click.option(
    "--fixtures",
    "fixtures_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to fixture dataset JSON (defaults to tests/fixtures/dataset/mock_spider.json).",
)
@click.option(
    "--artifact-dir",
    default="artifacts/runs",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where experiment run artifacts are stored.",
)
@click.option(
    "--run-id",
    help="Optional explicit run ID.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration and fixtures without creating an experiment run.",
)
@click.option(
    "--fail-mode",
    is_flag=True,
    help="Simulate component failure to verify failure state preservation.",
)
def pipeline_mock_cmd(
    config_path: Path | None,
    fixtures_path: Path | None,
    artifact_dir: Path,
    run_id: str | None,
    dry_run: bool,
    fail_mode: bool,
) -> None:
    """Run the deterministic end-to-end mock pipeline vertical slice.

    This command exercises configuration validation, prompt assembly, mock inference,
    mock evaluation, artifact logging, and cryptographic manifest verification.
    Operates 100% offline without downloading models or accessing real benchmark datasets.
    """
    from sqlforge.pipeline.harness import MockPipelineHarness

    console.print("\n[bold cyan]SQLForge Mock Pipeline Harness[/bold cyan]")
    console.print(
        "[yellow]DISCLAIMER: Offline synthetic mock pipeline demonstration only.\n"
        "Outputs do NOT represent real model inferences or benchmark accuracy claims.[/yellow]\n"
    )

    tracker = ExperimentTracker(base_artifact_dir=artifact_dir)
    harness = MockPipelineHarness(tracker=tracker)

    try:
        result = harness.run(
            config=config_path,
            fixtures_path=fixtures_path,
            run_id=run_id,
            dry_run=dry_run,
            fail_mode=fail_mode,
        )
    except Exception as exc:
        console.print(f"[bold red][FAILED] Pipeline setup error: {exc}[/bold red]")
        raise click.Abort() from exc

    table = Table(title="Pipeline Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Run ID", result.run_id)
    table.add_row(
        "Status",
        f"[bold green]{result.status.upper()}[/bold green]"
        if result.status == "completed"
        else f"[bold yellow]{result.status.upper()}[/bold yellow]",
    )
    table.add_row("Total Examples", str(result.total_examples))
    table.add_row("Artifact Directory", result.artifact_dir or "(none - dry run)")

    verified_badge = "[green]PASSED (Strict)[/green]" if result.verified else "[red]FAILED[/red]"
    table.add_row("Manifest Verification", verified_badge)
    table.add_row("Anomalies Recorded", str(result.anomalies_count))

    if result.metrics:
        em_acc = result.metrics.get("exact_match_accuracy", "N/A")
        exec_acc = result.metrics.get("execution_accuracy", "N/A")
        valid_rate = result.metrics.get("valid_sql_rate", "N/A")
        table.add_row("Mock EM Accuracy", f"{em_acc} (synthetic)")
        table.add_row("Mock Exec Accuracy", f"{exec_acc} (synthetic)")
        table.add_row("Mock Valid SQL Rate", f"{valid_rate} (synthetic)")

    console.print(table)

    if result.status == "failed" or not result.verified:
        console.print("[bold red]Pipeline completed with errors or failed verification.[/bold red]")
        raise click.Abort()

    console.print(
        "\n[bold green]Pipeline execution completed and verified successfully.[/bold green]\n"
    )


if __name__ == "__main__":
    main()
