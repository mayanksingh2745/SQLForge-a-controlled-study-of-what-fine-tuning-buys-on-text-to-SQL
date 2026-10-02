"""Automated validation checks for SQLForge research specifications and documentation."""

import re
from pathlib import Path

import yaml

from sqlforge.schemas.evaluation import ConfidenceInterval, EvaluationMetrics
from sqlforge.schemas.execution import ExecutionStatus


def test_research_documents_presence_and_substance() -> None:
    """Verify that all 9 required research specification documents exist and are substantive."""
    research_dir = Path("docs/research")
    assert research_dir.is_dir(), "docs/research directory must exist"

    required_docs = [
        "hypotheses.md",
        "baseline_protocol.md",
        "dataset_protocol.md",
        "experiment_matrix.md",
        "metrics_and_statistics.md",
        "error_taxonomy.md",
        "compute_plan.md",
        "experiment_record_spec.md",
        "step_dependencies.md",
    ]

    for doc_name in required_docs:
        doc_path = research_dir / doc_name
        assert doc_path.is_file(), f"Missing required specification document: {doc_name}"
        content = doc_path.read_text(encoding="utf-8")
        word_count = len(content.split())
        assert word_count > 150, (
            f"Document {doc_name} is too brief ({word_count} words); requires substantive detail"
        )


def test_experiment_matrix_specification_integrity() -> None:
    """Verify that all 10 staged experiments in experiment_matrix.md are uniquely defined."""
    matrix_file = Path("docs/research/experiment_matrix.md")
    content = matrix_file.read_text(encoding="utf-8")

    # Look for Experiment ID matches: EXP-01 through EXP-10
    exp_pattern = re.compile(r"\* \*\*Experiment ID:\*\* `(EXP-\d{2}-[A-Z0-9\-]+)`")
    matches = exp_pattern.findall(content)

    assert len(matches) == 10, (
        f"Expected exactly 10 experiment definitions in matrix, found {len(matches)}: {matches}"
    )
    assert len(set(matches)) == 10, "Experiment IDs must be unique"

    # Verify key sections exist for each experiment
    for exp_id in matches:
        assert f"`{exp_id}`" in content
        assert "* **Primary Metric:**" in content
        assert "* **Completion Criteria:**" in content


def test_error_taxonomy_coverage() -> None:
    """Verify that the error taxonomy defines all 10 canonical error categories."""
    taxonomy_file = Path("docs/research/error_taxonomy.md")
    content = taxonomy_file.read_text(encoding="utf-8")

    expected_codes = [f"E{i:02d}" for i in range(1, 11)] + ["E99"]
    for code in expected_codes:
        assert code in content and (f"{code}:" in content or f"{code} " in content), (
            f"Missing error code {code} in taxonomy"
        )


def test_schemas_support_research_matrix() -> None:
    """Verify that foundational Pydantic schemas support the attributes required by Step 1 research."""
    # Confidence Interval supports bounds and level
    ci = ConfidenceInterval(lower=0.65, upper=0.75, confidence_level=0.95)
    assert ci.lower < ci.upper

    # EvaluationMetrics supports difficulty breakdown and bootstrap CI
    metrics = EvaluationMetrics(
        total_examples=1000,
        valid_sql_rate=0.95,
        exact_match_accuracy=0.60,
        execution_accuracy=0.72,
        bootstrap_ci_execution_accuracy=ci,
        difficulty_breakdown={"easy": 0.85, "medium": 0.70, "hard": 0.55, "extra": 0.40},
    )
    assert metrics.execution_accuracy == 0.72

    # ExecutionStatus contains the required runtime statuses
    assert ExecutionStatus.SUCCESS == "success"
    assert ExecutionStatus.TIMEOUT == "timeout"
    assert ExecutionStatus.SYNTAX_ERROR == "syntax_error"


def test_config_experiments_yaml_validity() -> None:
    """Verify that configs/experiments.yaml parses and aligns with experiment concepts."""
    config_file = Path("configs/experiments.yaml")
    assert config_file.is_file()

    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "experiments" in data
    exps = data["experiments"]
    assert "exp01_zeroshot_baselines" in exps
    assert "exp03_lora_vs_qlora" in exps
    assert "exp04_lora_rank_sweep" in exps
    assert "exp05_data_scaling_law" in exps
