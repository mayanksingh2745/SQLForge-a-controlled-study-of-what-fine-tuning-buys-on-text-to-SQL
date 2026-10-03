"""Command Line Interface for SQLForge."""

from pathlib import Path
from typing import Any

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


@main.group("prompt")
def prompt_group() -> None:
    """Serialize database schemas and assemble Text-to-SQL prompts."""
    pass


@prompt_group.command("serialize")
@click.option(
    "--tables-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to tables.json file (defaults to Spider fixture).",
)
@click.option(
    "--db-id",
    help="Database ID to serialize (defaults to first available in tables file).",
)
@click.option(
    "--format",
    "schema_format",
    type=click.Choice(["ddl", "compact", "json"], case_sensitive=False),
    default="ddl",
    help="Serialization format: 'ddl', 'compact', or 'json'.",
)
def prompt_serialize_cmd(
    tables_file: Path | None,
    db_id: str | None,
    schema_format: str,
) -> None:
    """Serialize a database schema to DDL, compact pipe, or structured JSON."""
    from sqlforge.data.spider import parse_spider_tables
    from sqlforge.prompting.serializers import get_serializer

    root = Path(__file__).resolve().parent.parent.parent
    tbl_path = tables_file or (root / "tests" / "fixtures" / "dataset" / "spider" / "tables.json")

    schemas = parse_spider_tables(tbl_path)
    target_id = db_id or next(iter(schemas.keys()))

    if target_id not in schemas:
        console.print(f"[red]Error: Database '{target_id}' not found in '{tbl_path}'.[/red]")
        raise click.Abort()

    serializer = get_serializer(schema_format)
    output = serializer.serialize(schemas[target_id])
    console.print(
        f"\n[bold cyan]Serialized Schema ({target_id} - {schema_format.upper()}):[/bold cyan]\n"
    )
    console.print(output)


@prompt_group.command("assemble")
@click.option(
    "--question",
    default="What is the name and capacity of each stadium?",
    help="Natural language question.",
)
@click.option(
    "--db-id",
    default="stadium",
    help="Target database ID.",
)
@click.option(
    "--k-shots",
    type=click.Choice(["0", "1", "3", "5"]),
    default="0",
    help="Few-shot demonstration count (0, 1, 3, 5).",
)
@click.option(
    "--format",
    "schema_format",
    type=click.Choice(["ddl", "compact", "json"], case_sensitive=False),
    default="ddl",
    help="Schema format ('ddl', 'compact', 'json').",
)
@click.option(
    "--max-tokens",
    type=int,
    default=4096,
    help="Context window token budget.",
)
def prompt_assemble_cmd(
    question: str,
    db_id: str,
    k_shots: str,
    schema_format: str,
    max_tokens: int,
) -> None:
    """Assemble a zero-shot or few-shot Text-to-SQL prompt with schema and RAG demonstrations."""
    from sqlforge.data.spider import load_spider_split, parse_spider_tables
    from sqlforge.prompting.engine import PromptEngine
    from sqlforge.prompting.retriever import BM25Retriever
    from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample

    root = Path(__file__).resolve().parent.parent.parent
    fixtures_dir = root / "tests" / "fixtures" / "dataset"
    tbl_path = fixtures_dir / "spider" / "tables.json"
    schemas = parse_spider_tables(tbl_path)

    if db_id not in schemas:
        console.print(f"[red]Error: Database '{db_id}' not found in fixtures.[/red]")
        raise click.Abort()

    k = int(k_shots)
    retriever = None
    if k > 0:
        tr_path = fixtures_dir / "spider" / "train_spider.json"
        train_examples = load_spider_split(tr_path, schemas, split=DatasetSplit.TRAIN)
        retriever = BM25Retriever(train_examples)

    engine = PromptEngine(
        schema_format=schema_format,
        retriever=retriever,
        default_max_tokens=max_tokens,
    )

    query_example = TextToSQLExample(
        id="query_cli_01",
        question=question,
        db_id=db_id,
        gold_sql="SELECT Name, Capacity FROM stadium;",
        dataset_name="spider",
        split=DatasetSplit.DEV,
    )

    assembled = engine.assemble(
        example=query_example,
        schema=schemas[db_id],
        k_shots=k,
        max_tokens=max_tokens,
    )

    table = Table(title="Prompt Assembly Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Database ID", assembled.db_id)
    table.add_row("Schema Format", assembled.schema_format.upper())
    table.add_row("k-shots Attached", str(assembled.k_shots))
    table.add_row("Retrieved Demos", ", ".join(assembled.retrieved_demonstration_ids) or "(none)")
    table.add_row("Estimated Tokens", f"{assembled.token_count_estimate} / {max_tokens}")
    table.add_row(
        "Truncation Applied",
        "[yellow]Yes[/yellow]" if assembled.truncation_applied else "[green]No[/green]",
    )

    console.print(table)
    console.print("\n[bold cyan]Assembled Prompt Text:[/bold cyan]\n")
    console.print(assembled.prompt_text)


@main.group("baseline")
def baseline_group() -> None:
    """Run and verify baseline evaluations across open-weight and frontier models (EXP-01)."""
    pass


@baseline_group.command("run")
@click.option(
    "--model",
    "model_key",
    type=click.Choice(["qwen25_coder_1_5b", "qwen25_coder_7b", "frontier_api_ref"]),
    default="qwen25_coder_1_5b",
    help="Model identifier to evaluate.",
)
@click.option(
    "--k-shots",
    default=0,
    type=int,
    help="Number of few-shot demonstrations (0 for zero-shot, 3 for 3-shot BM25).",
)
@click.option(
    "--dataset",
    "dataset_spec",
    default="spider:dev",
    help="Dataset split specification (e.g. 'spider:dev', 'bird_mini:dev', 'held_out_custom:test').",
)
@click.option(
    "--schema-format",
    default="ddl",
    type=click.Choice(["ddl", "compact", "json"]),
    help="Schema representation format.",
)
@click.option(
    "--max-examples",
    type=int,
    default=None,
    help="Maximum number of evaluation examples to process.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration, prompts, and retrieval isolation without executing inference.",
)
@click.option(
    "--live-api",
    is_flag=True,
    help="Explicit opt-in required to invoke paid frontier API endpoints.",
)
@click.option(
    "--spending-limit",
    default=50.0,
    type=float,
    help="Hard dollar budget limit for API calls (maximum $50.00 USD).",
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
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility.",
)
def baseline_run_cmd(
    model_key: str,
    k_shots: int,
    dataset_spec: str,
    schema_format: str,
    max_examples: int | None,
    dry_run: bool,
    live_api: bool,
    spending_limit: float,
    artifact_dir: Path,
    run_id: str | None,
    seed: int,
) -> None:
    """Execute or dry-run a baseline evaluation following the EXP-01 research protocol."""
    from sqlforge.data.held_out import get_subscription_analytics_schema, load_held_out_dataset
    from sqlforge.data.manifest import load_examples_from_file
    from sqlforge.data.spider import load_spider_split, parse_spider_tables
    from sqlforge.pipeline.baseline import BaselinePipelineHarness
    from sqlforge.schemas.evaluation import EvaluationMetrics
    from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
    from sqlforge.schemas.metadata import SchemaMetadata

    console.print("\n[bold cyan]SQLForge Baseline Evaluation (EXP-01)[/bold cyan]")

    root = Path(__file__).resolve().parent.parent.parent
    fixtures_dir = root / "tests" / "fixtures" / "dataset"
    processed_dir = root / "data" / "processed"

    eval_examples: list[TextToSQLExample] = []
    train_examples: list[TextToSQLExample] = []
    schemas: dict[str, SchemaMetadata] = {}
    is_synthetic = True

    # Resolve dataset and schemas
    if dataset_spec == "spider:dev":
        real_dev = processed_dir / "spider" / "dev.jsonl"
        real_train = processed_dir / "spider" / "train.jsonl"
        real_tables = processed_dir / "spider" / "tables.json"

        if real_dev.is_file() and real_tables.is_file():
            is_synthetic = False
            schemas = parse_spider_tables(real_tables)
            eval_examples = load_examples_from_file(real_dev)
            if real_train.is_file():
                train_examples = load_examples_from_file(real_train)
        else:
            fix_tables = fixtures_dir / "spider" / "tables.json"
            fix_dev = fixtures_dir / "spider" / "dev.json"
            fix_train = fixtures_dir / "spider" / "train_spider.json"
            schemas = parse_spider_tables(fix_tables)
            eval_examples = load_spider_split(fix_dev, schemas, split=DatasetSplit.DEV)
            train_examples = load_spider_split(fix_train, schemas, split=DatasetSplit.TRAIN)
    elif dataset_spec == "held_out_custom:test":
        h_schema = get_subscription_analytics_schema()
        schemas = {h_schema.db_id: h_schema}
        h_file = fixtures_dir / "custom_held_out" / "held_out_examples.json"
        eval_examples = load_held_out_dataset(h_file)
    else:
        console.print(f"[red]Error: Unsupported or uningested dataset '{dataset_spec}'.[/red]")
        raise click.Abort()

    if is_synthetic:
        console.print(
            "[yellow]NOTE: Real benchmark archive not detected in data/processed/. "
            "Using synthetic test fixtures from tests/fixtures/dataset/.\n"
            "Results reflect software verification only and must NOT be cited as empirical benchmark claims.[/yellow]\n"
        )

    tracker = ExperimentTracker(base_artifact_dir=artifact_dir)
    harness = BaselinePipelineHarness(tracker=tracker)

    try:
        result = harness.run(
            model_key=model_key,
            eval_examples=eval_examples,
            schemas=schemas,
            train_examples=train_examples if k_shots > 0 else None,
            k_shots=k_shots,
            schema_format=schema_format,
            max_examples=max_examples,
            run_id=run_id,
            seed=seed,
            dry_run=dry_run,
            allow_live_api=live_api,
            api_spending_limit=spending_limit,
        )
    except Exception as exc:
        console.print(f"[bold red][FAILED] Baseline execution error: {exc}[/bold red]")
        raise click.Abort() from exc

    table = Table(title="Baseline Execution Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Run ID", result.run_id)
    table.add_row("Model", result.model_id)
    table.add_row("k-shots", str(result.k_shots))
    table.add_row("Total Examples (N)", str(result.total_examples))
    table.add_row(
        "Status",
        f"[bold green]{result.status.upper()}[/bold green]"
        if result.status in ("completed", "dry_run")
        else f"[bold red]{result.status.upper()}[/bold red]",
    )
    table.add_row(
        "Manifest Verified",
        "[bold green]PASSED[/bold green]" if result.verified else "[bold red]FAILED[/bold red]",
    )

    if result.status == "completed" and isinstance(result.metrics, EvaluationMetrics):
        metrics = result.metrics
        acc_pct = metrics.execution_accuracy * 100.0
        ci = metrics.confidence_interval
        ci_str = f"[{ci.lower * 100.0:.1f}%, {ci.upper * 100.0:.1f}%]" if ci else "N/A"
        syntax_rate = (metrics.syntax_valid_rate or 0.0) * 100.0
        exec_rate = (metrics.execution_success_rate or 0.0) * 100.0
        table.add_row("Execution Accuracy", f"{acc_pct:.2f}% (95% CI: {ci_str})")
        table.add_row("Syntax Valid Rate", f"{syntax_rate:.2f}%")
        table.add_row("Execution Success Rate", f"{exec_rate:.2f}%")
        table.add_row("Exact Match Rate", f"{metrics.exact_match_accuracy * 100.0:.2f}%")
        table.add_row("Total Tokens", str(result.total_tokens))
        table.add_row("Total Cost (USD)", f"${result.total_cost_usd:.4f}")

    console.print(table)


@baseline_group.command("verify")
@click.argument("run_id")
@click.option(
    "--artifact-dir",
    default="artifacts/runs",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where experiment run artifacts are stored.",
)
def baseline_verify_cmd(run_id: str, artifact_dir: Path) -> None:
    """Verify cryptographic manifest integrity of a completed baseline run."""
    tracker = ExperimentTracker(base_artifact_dir=artifact_dir)

    console.print(f"\n[bold cyan]Verifying Baseline Run: {run_id}[/bold cyan]")
    try:
        report = tracker.verify_run(run_id, strict=True)
    except Exception as exc:
        console.print(f"[bold red]Verification failed with exception: {exc}[/bold red]")
        raise click.Abort() from exc

    table = Table(
        title="Manifest Verification Diagnostics", show_header=True, header_style="bold magenta"
    )
    table.add_column("Property", style="dim", width=24)
    table.add_column("Value")

    table.add_row("Run ID", run_id)
    table.add_row(
        "Verification Status",
        "[bold green]PASSED[/bold green]" if report["verified"] else "[bold red]FAILED[/bold red]",
    )
    table.add_row("Total Files Verified", str(len(report.get("file_details", {}))))

    if report.get("errors"):
        for err in report["errors"]:
            table.add_row("[red]Error[/red]", str(err))

    console.print(table)


@baseline_group.command("stats")
@click.argument("run_id_a")
@click.option(
    "--compare-to",
    "run_id_b",
    default=None,
    help="Second run ID for paired difference bootstrap CI and McNemar test.",
)
@click.option(
    "--artifact-dir",
    default="artifacts/runs",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where experiment run artifacts are stored.",
)
def baseline_stats_cmd(run_id_a: str, run_id_b: str | None, artifact_dir: Path) -> None:
    """Inspect statistical confidence intervals or compare two runs with paired tests."""
    import json

    from sqlforge.evaluation.statistics import compute_paired_difference_ci, mcnemar_test

    tracker = ExperimentTracker(base_artifact_dir=artifact_dir)
    run_dir_a = tracker.base_dir / run_id_a
    if not run_dir_a.is_dir():
        console.print(f"[red]Error: Run directory not found: {run_dir_a}[/red]")
        raise click.Abort()

    gen_file_a = run_dir_a / "generations.jsonl"
    metrics_file_a = run_dir_a / "metrics.json"

    console.print(f"\n[bold cyan]Statistical Analysis for Run: {run_id_a}[/bold cyan]")

    if metrics_file_a.is_file():
        with open(metrics_file_a, encoding="utf-8") as f:
            m_a = json.load(f)
        ci_a = m_a.get("confidence_interval")
        acc_a = m_a.get("execution_accuracy", 0.0) * 100.0
        ci_str = f"[{ci_a['lower'] * 100.0:.2f}%, {ci_a['upper'] * 100.0:.2f}%]" if ci_a else "N/A"
        console.print(
            f"Execution Accuracy: [bold green]{acc_a:.2f}%[/bold green] (95% Bootstrap CI: {ci_str})"
        )

    if run_id_b:
        run_dir_b = tracker.base_dir / run_id_b
        if not run_dir_b.is_dir():
            console.print(f"[red]Error: Run directory not found: {run_dir_b}[/red]")
            raise click.Abort()

        gen_file_b = run_dir_b / "generations.jsonl"
        if not gen_file_a.is_file() or not gen_file_b.is_file():
            console.print(
                "[red]Error: Both runs must contain generations.jsonl for paired statistical comparison.[/red]"
            )
            raise click.Abort()

        # Load generations and align by example_id
        def _load_gens(p: Path) -> dict[str, Any]:
            records = {}
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        records[obj["example_id"]] = obj
            return records

        gens_a = _load_gens(gen_file_a)
        gens_b = _load_gens(gen_file_b)

        common_ids = sorted(set(gens_a.keys()) & set(gens_b.keys()))
        if not common_ids:
            console.print(
                "[red]Error: Runs share 0 common example_ids. Cannot perform paired comparison.[/red]"
            )
            raise click.Abort()

        # For paired difference: score is 1.0 if generated query matches gold, 0.0 otherwise
        scores_a = [
            1.0
            if gens_a[eid].get("generated_sql", "").strip()
            == gens_a[eid].get("gold_sql", "").strip()
            else 0.0
            for eid in common_ids
        ]
        scores_b = [
            1.0
            if gens_b[eid].get("generated_sql", "").strip()
            == gens_b[eid].get("gold_sql", "").strip()
            else 0.0
            for eid in common_ids
        ]

        paired_ci = compute_paired_difference_ci(scores_a, scores_b)
        mcnemar = mcnemar_test(scores_a, scores_b)

        table = Table(
            title=f"Paired Comparison: {run_id_a} vs {run_id_b}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Property", style="dim", width=36)
        table.add_column("Value")

        table.add_row("Common Evaluated Examples (N)", str(len(common_ids)))
        table.add_row("Mean Score A", f"{sum(scores_a) / len(scores_a) * 100.0:.2f}%")
        table.add_row("Mean Score B", f"{sum(scores_b) / len(scores_b) * 100.0:.2f}%")
        table.add_row(
            "Difference (A - B)", f"{(sum(scores_a) - sum(scores_b)) / len(scores_a) * 100.0:+.2f}%"
        )
        table.add_row(
            "Paired 95% Bootstrap CI",
            f"[{paired_ci.lower * 100.0:+.2f}%, {paired_ci.upper * 100.0:+.2f}%]",
        )
        table.add_row("McNemar Chi-Square Statistic", f"{mcnemar['statistic']:.4f}")
        table.add_row("McNemar p-value", f"{mcnemar['p_value']:.4e}")
        table.add_row(
            "Statistically Significant (p < 0.05)",
            "[bold green]Yes[/bold green]" if mcnemar["significant"] else "[yellow]No[/yellow]",
        )

        console.print(table)


# =====================================================================
# SFT Training Workflow CLI Group (Step 6)
# =====================================================================


@main.group("train")
def train_group() -> None:
    """Supervised fine-tuning infrastructure for LoRA and QLoRA."""
    pass


@train_group.command("preflight")
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="lora",
    help="Target fine-tuning method",
)
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--check-dataset/--no-check-dataset",
    default=True,
    help="Verify training dataset availability",
)
def train_preflight_cmd(method: str, base_model: str, check_dataset: bool) -> None:
    """Check hardware feasibility, VRAM, RAM, disk, and dependencies."""
    from sqlforge.training import FineTuningMethod, PreflightChecker, SFTTrainingConfig

    ft_method = FineTuningMethod(method.lower())
    config = SFTTrainingConfig(method=ft_method, base_model_id=base_model)
    checker = PreflightChecker(config)
    report = checker.run_preflight(check_dataset=check_dataset)

    console.print(
        f"\n[bold cyan]SQLForge SFT Pre-Flight Feasibility Check ({method.upper()})[/bold cyan]"
    )
    console.print("=" * 65)

    status_color = (
        "green"
        if report.status.value == "ready"
        else ("yellow" if report.status.value == "warning" else "red")
    )
    console.print(
        f"Overall Status: [{status_color} bold]{report.status.value.upper()}[/{status_color} bold]"
    )
    console.print(f"Platform:       {report.platform} (Python {report.python_version})")
    console.print(
        f"CUDA Available: {'Yes (' + str(report.device_count) + ' device(s))' if report.cuda_available else 'No (CPU only)'}"
    )
    if report.device_names:
        console.print(f"GPU Devices:    {', '.join(report.device_names)}")
    console.print(
        f"Total VRAM:     {report.total_vram_gb:.1f} GB (Free: {report.free_vram_gb:.1f} GB)"
    )
    console.print(
        f"Host RAM:       {report.free_ram_gb:.1f} GB free / {report.total_ram_gb:.1f} GB total"
    )
    console.print(f"Target Disk:    {report.free_disk_gb:.1f} GB free")

    table = Table(title="Diagnostic Checks", show_header=True, header_style="bold magenta")
    table.add_column("Category", width=14)
    table.add_column("Check Name", width=24)
    table.add_column("Status", width=10)
    table.add_column("Details")

    for c in report.checks:
        status_str = (
            "[green]PASS[/green]"
            if c.passed
            else f"[{'red' if c.level == 'error' else 'yellow'}]{c.level.upper()}[/]"
        )
        table.add_row(c.category.capitalize(), c.name, status_str, c.message)

    console.print("\n", table)

    if report.blockers:
        console.print("\n[bold red]Blocking Issues:[/bold red]")
        for b in report.blockers:
            console.print(f"  [red]x[/red] {b}")

    if report.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in report.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    if report.recommendations:
        console.print("\n[bold cyan]Actionable Recommendations:[/bold cyan]")
        for r in report.recommendations:
            console.print(f"  -> {r}")

    if not report.is_runnable:
        console.print(
            "\n[dim yellow]Note: Hardware is insufficient for live GPU training on this host. Use --dry-run or remote GPU.[/dim yellow]"
        )


@train_group.command("validate")
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="lora",
    help="Target fine-tuning method",
)
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--rank",
    type=int,
    default=16,
    help="LoRA rank dimension r",
)
@click.option(
    "--alpha",
    type=int,
    default=32,
    help="LoRA alpha scaling factor",
)
@click.option(
    "--epochs",
    type=int,
    default=3,
    help="Training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Per-device batch size",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="Peak learning rate",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML training configuration file",
)
def train_validate_cmd(
    method: str,
    base_model: str,
    rank: int,
    alpha: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    config_file: Path | None,
) -> None:
    """Validate SFT training configuration parameters and settings."""
    import yaml

    from sqlforge.schemas.models import LoRAHyperparameters
    from sqlforge.training import FineTuningMethod, PEFTConfigFactory, SFTTrainingConfig

    try:
        if config_file is not None:
            with open(config_file, encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
            config = SFTTrainingConfig.model_validate(raw_cfg)
        else:
            ft_method = FineTuningMethod(method.lower())
            lora_params = LoRAHyperparameters(rank=rank, alpha=alpha)
            config = SFTTrainingConfig(
                method=ft_method,
                base_model_id=base_model,
                lora=lora_params,
                epochs=epochs,
                per_device_batch_size=batch_size,
                learning_rate=learning_rate,
            )

        summary = PEFTConfigFactory.summarize_adapter_config(config)

        console.print("\n[bold green]Configuration Validated Successfully[/bold green]")
        console.print("=" * 60)

        table = Table(
            title="Resolved Training Specification", show_header=True, header_style="bold magenta"
        )
        table.add_column("Parameter", style="dim", width=28)
        table.add_column("Value")

        table.add_row("Experiment ID", config.experiment_id)
        table.add_row("Base Model", config.base_model_id)
        table.add_row("Fine-Tuning Method", config.method.value.upper())
        table.add_row("LoRA Rank (r)", str(config.lora.rank))
        table.add_row("LoRA Alpha", str(config.lora.alpha))
        table.add_row("Target Modules", ", ".join(summary.target_modules))
        table.add_row("Epochs", str(config.epochs))
        table.add_row("Per-Device Batch Size", str(config.per_device_batch_size))
        table.add_row("Gradient Accumulation", str(config.gradient_accumulation_steps))
        table.add_row("Effective Batch Size", str(config.effective_batch_size))
        table.add_row("Learning Rate", str(config.learning_rate))
        table.add_row("LR Scheduler", config.lr_scheduler)
        table.add_row("Mixed Precision", config.mixed_precision)
        table.add_row("Max Sequence Length", str(config.max_seq_length))
        table.add_row("Mask Prompt Loss", str(config.mask_prompt_loss))
        table.add_row("Train Dataset", config.train_dataset)
        table.add_row("Eval Dataset", config.eval_dataset or "None")

        if config.method == FineTuningMethod.QLORA:
            table.add_row(
                "QLoRA Quant Bits", f"{config.qlora.bits}-bit ({config.qlora.quant_type})"
            )
            table.add_row("Double Quantization", str(config.qlora.use_double_quant))
            table.add_row("Compute Dtype", config.qlora.compute_dtype)

        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Configuration Validation Error:[/bold red] {exc}")
        raise click.Abort() from None


@train_group.command("run")
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="lora",
    help="Fine-tuning method ('lora' or 'qlora')",
)
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Per-device batch size",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="Peak learning rate",
)
@click.option(
    "--schema-format",
    default="ddl",
    type=click.Choice(["ddl", "compact_pipe", "json"], case_sensitive=False),
    help="Schema representation format",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum token sequence length",
)
@click.option(
    "--max-examples",
    type=int,
    default=None,
    help="Cap training examples for quick smoke tests",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Execute pre-flight, data preparation, and tokenization without training weights",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="Explicit confirmation required to launch live model weight training",
)
def train_run_cmd(
    method: str,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    schema_format: str,
    max_seq_length: int,
    max_examples: int | None,
    dry_run: bool,
    execute: bool,
) -> None:
    """Execute supervised fine-tuning pipeline for LoRA or QLoRA."""
    from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
    from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata
    from sqlforge.training import (
        FineTuningMethod,
        PreflightSafetyError,
        SFTFineTuningPipeline,
        SFTTrainingConfig,
    )

    if not dry_run and not execute:
        console.print(
            "[bold red]Safety Guard:[/bold red] Live training requires explicit confirmation via [bold cyan]--execute[/bold cyan] "
            "or use [bold cyan]--dry-run[/bold cyan] for zero-compute validation."
        )
        raise click.Abort()

    ft_method = FineTuningMethod(method.lower())
    config = SFTTrainingConfig(
        method=ft_method,
        base_model_id=base_model,
        epochs=epochs,
        per_device_batch_size=batch_size,
        learning_rate=learning_rate,
        schema_format=schema_format,
        max_seq_length=max_seq_length,
        max_train_examples=max_examples,
    )

    console.print(
        f"\n[bold cyan]SQLForge SFT Execution: {method.upper()} on {base_model}[/bold cyan]"
    )
    console.print("=" * 65)

    # Prepare minimal synthetic fixture data for dry run or fallback verification
    toy_schema = SchemaMetadata(
        db_id="academic",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="author",
                columns=[
                    ColumnMetadata(name="author_id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="name", data_type="TEXT"),
                ],
            )
        ],
    )
    schemas = {"academic": toy_schema}
    examples = [
        TextToSQLExample(
            id=f"spider_train_{i:04d}",
            question=f"List all author names in the database? (example {i})",
            db_id="academic",
            gold_sql="SELECT name FROM author;",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
        for i in range(1, 11)
    ]

    pipeline = SFTFineTuningPipeline(config=config)

    try:
        result = pipeline.run(
            examples=examples,
            schemas=schemas,
            dry_run=dry_run,
        )

        console.print(f"\n[bold green]Run Completed ({result.status.upper()})[/bold green]")
        console.print("=" * 60)
        table = Table(title="Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="dim", width=28)
        table.add_column("Value")

        table.add_row("Run ID", result.run_id)
        table.add_row("Status", result.status)
        table.add_row("Method", result.method.upper())
        table.add_row("Final Checkpoint Dir", result.final_checkpoint_dir)
        table.add_row("Valid Examples Formatted", str(result.data_summary.valid_examples_count))
        table.add_row("Duration", f"{result.duration_seconds}s")
        table.add_row("Total Steps", str(result.total_steps))
        table.add_row("Manifest Verified", str(result.manifest_verified))

        console.print(table)
    except PreflightSafetyError as err:
        console.print(f"\n[bold red]Pre-Flight Safety Halt:[/bold red]\n{err}")
        raise click.Abort() from None


@train_group.command("inspect")
@click.argument("checkpoint_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def train_inspect_cmd(checkpoint_dir: Path) -> None:
    """Inspect saved checkpoint metadata and training metrics."""
    from sqlforge.training import CheckpointManager

    try:
        meta = CheckpointManager.load_metadata(checkpoint_dir)

        console.print(
            f"\n[bold cyan]Checkpoint Metadata Inspection: {checkpoint_dir.name}[/bold cyan]"
        )
        console.print("=" * 65)

        table = Table(title="Checkpoint Details", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="dim", width=28)
        table.add_column("Value")

        table.add_row("Run ID", meta.run_id)
        table.add_row("Step", str(meta.step))
        table.add_row("Epoch", f"{meta.epoch:.2f}")
        table.add_row("Base Model", meta.base_model_id)
        table.add_row("Adapter Method", meta.adapter_summary.method.upper())
        table.add_row("LoRA Rank (r)", str(meta.adapter_summary.rank))
        table.add_row("LoRA Alpha", str(meta.adapter_summary.alpha))
        table.add_row("Target Modules", ", ".join(meta.adapter_summary.target_modules))
        table.add_row("Git Commit", str(meta.git_commit or "unknown"))
        table.add_row("Python Version", meta.python_version)
        table.add_row("Checkpoint Type", meta.checkpoint_type)
        table.add_row(
            "Adapter Files", ", ".join(meta.adapter_files) if meta.adapter_files else "None"
        )

        for k, v in meta.training_metrics.items():
            table.add_row(f"Metric: {k}", str(v))

        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Failed to load checkpoint metadata:[/bold red] {exc}")
        raise click.Abort() from None


# =====================================================================
# LoRA Hyperparameter & Rank Scaling Sweep CLI Group (Step 7)
# =====================================================================


@main.group("sweep")
def sweep_group() -> None:
    """Systematic LoRA hyperparameter and rank scaling sweeps (EXP-04)."""
    pass


@sweep_group.command("plan")
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="qlora",
    help="Fine-tuning method ('lora' or 'qlora')",
)
@click.option(
    "--ranks",
    default="8,16,32,64",
    help="Comma-separated LoRA rank dimensions",
)
@click.option(
    "--seeds",
    default="42",
    help="Comma-separated random seeds",
)
@click.option(
    "--target-modules",
    default="all-linear",
    help="Target modules configuration ('all-linear', 'attention-only')",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional YAML configuration file",
)
def sweep_plan_cmd(
    base_model: str,
    method: str,
    ranks: str,
    seeds: str,
    target_modules: str,
    config_file: Path | None,
) -> None:
    """Plan systematic LoRA rank scaling sweep and inspect cached runs."""
    import yaml

    from sqlforge.training import FineTuningMethod, LoRARankSweepOrchestrator, RankSweepConfig

    try:
        if config_file is not None:
            with open(config_file, encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
            exp04_data = raw_data.get("experiments", {}).get("exp04_lora_rank_sweep", raw_data)
            cfg = RankSweepConfig(
                experiment_id=exp04_data.get("experiment_id", "EXP-04-RANK-SWEEP"),
                base_model_id=exp04_data.get("base_model", base_model),
                ranks=exp04_data.get("ranks", [8, 16, 32, 64]),
                seeds=exp04_data.get("seeds", [42]),
                target_modules_configs=[exp04_data.get("target_modules", target_modules)],
            )
        else:
            parsed_ranks = [int(r.strip()) for r in ranks.split(",") if r.strip()]
            parsed_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]
            cfg = RankSweepConfig(
                base_model_id=base_model,
                method=FineTuningMethod(method.lower()),
                ranks=parsed_ranks,
                seeds=parsed_seeds,
                target_modules_configs=[target_modules],
            )

        orchestrator = LoRARankSweepOrchestrator(cfg)
        plans = orchestrator.plan_sweep()

        console.print(
            f"\n[bold cyan]SQLForge LoRA Rank Sweep Plan: {cfg.experiment_id}[/bold cyan]"
        )
        console.print("=" * 70)
        console.print(f"Base Model:     {cfg.base_model_id}")
        console.print(f"Method:         {cfg.method.value.upper()}")
        console.print(f"Total Runs:     {len(plans)}")

        table = Table(title="Planned Sweep Runs", show_header=True, header_style="bold magenta")
        table.add_column("Run ID", width=34)
        table.add_column("Rank (r)", justify="right", width=9)
        table.add_column("Alpha", justify="right", width=8)
        table.add_column("Seed", justify="right", width=6)
        table.add_column("Modules", width=14)
        table.add_column("Est. Params", justify="right", width=12)
        table.add_column("Est. Size", justify="right", width=10)
        table.add_column("Status", width=10)

        for p in plans:
            status_str = "[green]CACHED[/green]" if p.is_cached else "[cyan]PLANNED[/cyan]"
            table.add_row(
                p.run_id,
                str(p.rank),
                str(p.alpha),
                str(p.seed),
                p.target_modules_tag,
                f"{p.estimated_params:,}",
                f"{p.estimated_size_mb:.1f} MB",
                status_str,
            )

        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Failed to plan sweep:[/bold red] {exc}")
        raise click.Abort() from None


@sweep_group.command("validate")
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="qlora",
    help="Fine-tuning method ('lora' or 'qlora')",
)
@click.option(
    "--ranks",
    default="8,16,32,64",
    help="Comma-separated LoRA rank dimensions",
)
@click.option(
    "--seeds",
    default="42",
    help="Comma-separated random seeds",
)
@click.option(
    "--target-modules",
    default="all-linear",
    help="Target modules configuration",
)
def sweep_validate_cmd(
    base_model: str,
    method: str,
    ranks: str,
    seeds: str,
    target_modules: str,
) -> None:
    """Validate LoRA rank sweep configuration consistency and controls."""
    from sqlforge.training import FineTuningMethod, RankSweepConfig

    try:
        parsed_ranks = [int(r.strip()) for r in ranks.split(",") if r.strip()]
        parsed_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        cfg = RankSweepConfig(
            base_model_id=base_model,
            method=FineTuningMethod(method.lower()),
            ranks=parsed_ranks,
            seeds=parsed_seeds,
            target_modules_configs=[target_modules],
        )

        configs = cfg.generate_run_configs()

        console.print("\n[bold green]Sweep Configuration Validated Successfully[/bold green]")
        console.print("=" * 65)

        table = Table(
            title="Controlled Invariant Variables", show_header=True, header_style="bold magenta"
        )
        table.add_column("Controlled Parameter", style="dim", width=28)
        table.add_column("Fixed Value")

        table.add_row("Base Model", cfg.base_model_id)
        table.add_row("Method", cfg.method.value.upper())
        table.add_row("Epochs", str(cfg.epochs))
        table.add_row("Per-Device Batch Size", str(cfg.per_device_batch_size))
        table.add_row("Gradient Accumulation", str(cfg.gradient_accumulation_steps))
        table.add_row("Learning Rate", str(cfg.learning_rate))
        table.add_row("LR Scheduler", cfg.lr_scheduler)
        table.add_row("Max Sequence Length", str(cfg.max_seq_length))
        table.add_row("Schema Format", cfg.schema_format)
        table.add_row("Prompt Loss Masking", str(cfg.mask_prompt_loss))
        table.add_row("Train Dataset", cfg.train_dataset)
        table.add_row("Eval Dataset", cfg.eval_dataset)
        table.add_row("Evaluated Ranks", ", ".join(f"r={r}" for r in cfg.ranks))
        table.add_row("Alpha Rule", f"alpha = round(rank * {cfg.alpha_multiplier})")
        table.add_row("Seeds", ", ".join(str(s) for s in cfg.seeds))
        table.add_row("Total Controlled Runs", str(len(configs)))

        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Sweep Configuration Invalid:[/bold red] {exc}")
        raise click.Abort() from None


@sweep_group.command("run")
@click.option(
    "--base-model",
    default="Qwen/Qwen2.5-Coder-7B-Instruct",
    help="Target base model identifier",
)
@click.option(
    "--method",
    type=click.Choice(["lora", "qlora"], case_sensitive=False),
    default="qlora",
    help="Fine-tuning method ('lora' or 'qlora')",
)
@click.option(
    "--ranks",
    default="8,16,32,64",
    help="Comma-separated LoRA rank dimensions",
)
@click.option(
    "--seeds",
    default="42",
    help="Comma-separated random seeds",
)
@click.option(
    "--target-modules",
    default="all-linear",
    help="Target modules configuration",
)
@click.option(
    "--max-examples",
    type=int,
    default=None,
    help="Cap training examples for quick smoke testing",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("artifacts/runs"),
    help="Directory for individual run artifacts",
)
@click.option(
    "--sweep-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("artifacts/sweeps"),
    help="Directory for sweep summary manifests",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Execute zero-compute dry run across all sweep conditions",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="Explicit confirmation required to launch live model weight training",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume interrupted sweep, skipping already completed and verified runs",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Halt sweep immediately upon first failure",
)
def sweep_run_cmd(
    base_model: str,
    method: str,
    ranks: str,
    seeds: str,
    target_modules: str,
    max_examples: int | None,
    output_dir: Path,
    sweep_dir: Path,
    dry_run: bool,
    execute: bool,
    resume: bool,
    fail_fast: bool,
) -> None:
    """Execute systematic LoRA rank scaling sweep."""
    from sqlforge.training import (
        FineTuningMethod,
        LoRARankSweepOrchestrator,
        PreflightSafetyError,
        RankSweepConfig,
    )

    if not dry_run and not execute:
        console.print(
            "[bold red]Safety Guard:[/bold red] Live hyperparameter sweeps require explicit confirmation "
            "via [bold cyan]--execute[/bold cyan] or use [bold cyan]--dry-run[/bold cyan] for zero-compute orchestration."
        )
        raise click.Abort()

    try:
        parsed_ranks = [int(r.strip()) for r in ranks.split(",") if r.strip()]
        parsed_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        cfg = RankSweepConfig(
            base_model_id=base_model,
            method=FineTuningMethod(method.lower()),
            ranks=parsed_ranks,
            seeds=parsed_seeds,
            target_modules_configs=[target_modules],
            max_train_examples=max_examples,
            output_dir=output_dir,
            sweep_dir=sweep_dir,
        )

        orchestrator = LoRARankSweepOrchestrator(cfg)

        console.print(
            f"\n[bold cyan]SQLForge LoRA Rank Sweep Execution: {cfg.experiment_id}[/bold cyan]"
        )
        console.print("=" * 70)
        console.print(f"Base Model:     {cfg.base_model_id}")
        console.print(f"Method:         {cfg.method.value.upper()}")
        console.print(f"Execution Mode: {'DRY RUN' if dry_run else 'LIVE TRAINING'}")
        console.print(f"Resume Enabled: {resume}")

        summary = orchestrator.run_sweep(
            dry_run=dry_run,
            execute=execute,
            resume=resume,
            fail_fast=fail_fast,
        )

        console.print(f"\n[bold green]Sweep Completed: {summary.sweep_id}[/bold green]")
        console.print("=" * 70)
        console.print(
            f"Runs: {summary.completed} completed, {summary.cached} cached, "
            f"{summary.failed} failed, {summary.skipped} skipped out of {summary.total_planned} planned."
        )

        table = Table(title="Sweep Results Summary", show_header=True, header_style="bold magenta")
        table.add_column("Run ID", width=34)
        table.add_column("Rank", justify="right", width=6)
        table.add_column("Seed", justify="right", width=6)
        table.add_column("Status", width=10)
        table.add_column("Trainable Params", justify="right", width=16)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Loss", justify="right", width=8)

        for r in summary.results:
            status_style = (
                "green"
                if r.status in ("completed", "cached")
                else ("cyan" if r.status == "dry_run" else "red")
            )
            table.add_row(
                r.run_id,
                str(r.rank),
                str(r.seed),
                f"[{status_style}]{r.status.upper()}[/{status_style}]",
                f"{r.trainable_parameters:,}",
                f"{r.duration_seconds:.1f}s",
                f"{r.final_loss:.4f}" if r.final_loss is not None else "N/A",
            )

        console.print(table)
        console.print(
            f"\nSummary artifact saved to: [dim]{cfg.sweep_dir / summary.sweep_id / 'sweep_summary.json'}[/dim]"
        )
    except PreflightSafetyError as err:
        console.print(f"\n[bold red]Pre-Flight Safety Halt:[/bold red]\n{err}")
        raise click.Abort() from None
    except Exception as exc:
        console.print(f"[bold red]Sweep execution error:[/bold red] {exc}")
        raise click.Abort() from None


@sweep_group.command("status")
@click.argument("summary_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def sweep_status_cmd(summary_file: Path) -> None:
    """Inspect the status and results of a completed or partial sweep."""
    from sqlforge.training import SweepSummary

    try:
        summary = SweepSummary.load_json(summary_file)

        console.print(f"\n[bold cyan]SQLForge Sweep Status: {summary.sweep_id}[/bold cyan]")
        console.print("=" * 70)
        console.print(f"Experiment ID: {summary.experiment_id}")
        console.print(f"Base Model:    {summary.base_model_id}")
        console.print(f"Method:        {summary.method.upper()}")
        console.print(f"Created At:    {summary.created_at}")
        console.print(f"Is Dry Run:    {summary.is_dry_run}")
        console.print(
            f"Progress:      {summary.completed + summary.cached} / {summary.total_planned} completed "
            f"({summary.failed} failed, {summary.skipped} skipped)"
        )

        table = Table(title="Condition Results", show_header=True, header_style="bold magenta")
        table.add_column("Run ID", width=34)
        table.add_column("Rank", justify="right", width=6)
        table.add_column("Alpha", justify="right", width=6)
        table.add_column("Seed", justify="right", width=6)
        table.add_column("Status", width=10)
        table.add_column("Params", justify="right", width=12)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Loss", justify="right", width=8)

        for r in summary.results:
            status_style = "green" if r.status in ("completed", "cached") else "yellow"
            table.add_row(
                r.run_id,
                str(r.rank),
                str(r.alpha),
                str(r.seed),
                f"[{status_style}]{r.status.upper()}[/{status_style}]",
                f"{r.trainable_parameters:,}",
                f"{r.duration_seconds:.1f}s",
                f"{r.final_loss:.4f}" if r.final_loss is not None else "N/A",
            )

        console.print(table)
    except Exception as exc:
        console.print(f"[bold red]Failed to load sweep summary:[/bold red] {exc}")
        raise click.Abort() from None


@sweep_group.command("plot")
@click.option(
    "--summary-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to sweep_summary.json artifact",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("reports/figures/exp04_rank_saturation_pareto.svg"),
    help="Destination file for Pareto SVG figure",
)
@click.option(
    "--export-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to export tabular metrics as CSV",
)
def sweep_plot_cmd(
    summary_file: Path,
    output: Path,
    export_csv: Path | None,
) -> None:
    """Generate rank-saturation Pareto curve strictly from empirical results."""
    from sqlforge.training import NoEmpiricalDataError, RankSaturationPlotter, SweepSummary

    try:
        summary = SweepSummary.load_json(summary_file)

        if export_csv is not None:
            csv_path = RankSaturationPlotter.export_summary_csv(summary, export_csv)
            console.print(f"[green]Sweep metrics exported to CSV:[/green] {csv_path}")

        try:
            plot_path = RankSaturationPlotter.plot_rank_saturation(summary, output_path=output)
            console.print(
                f"\n[bold green]Pareto Plot Generated Successfully:[/bold green] {plot_path}"
            )
        except NoEmpiricalDataError as data_err:
            console.print(
                f"\n[bold yellow]No Empirical Evaluation Data:[/bold yellow]\n{data_err}\n"
                "[dim]Rank saturation plots require genuine execution accuracy measurements on spider:dev. "
                "No illustrative or fabricated curves were generated.[/dim]"
            )
    except Exception as exc:
        console.print(f"[bold red]Plotting error:[/bold red] {exc}")
        raise click.Abort() from None


if __name__ == "__main__":
    main()
