"""Unit tests for typed foundational data contracts."""

import pytest
from pydantic import ValidationError

from sqlforge.schemas.evaluation import ConfidenceInterval, EvaluationMetrics
from sqlforge.schemas.examples import (
    DatasetSplit,
    DifficultyLevel,
    SyntheticProvenance,
    TextToSQLExample,
)
from sqlforge.schemas.execution import ExecutionResult, ExecutionStatus
from sqlforge.schemas.experiments import (
    ExperimentConfig,
    PromptConfig,
)
from sqlforge.schemas.metadata import (
    ColumnMetadata,
    SchemaMetadata,
)
from sqlforge.schemas.models import (
    LoRAHyperparameters,
    ModelConfig,
    ModelType,
    QLoRAHyperparameters,
)


def test_schema_metadata_structure(sample_schema: SchemaMetadata) -> None:
    """Verify SchemaMetadata integrity and table lookup."""
    assert sample_schema.db_id == "ecommerce_test"
    assert len(sample_schema.tables) == 2
    users = sample_schema.get_table("users")
    assert users is not None
    assert len(users.columns) == 3
    assert users.primary_keys == ["user_id"]

    non_existent = sample_schema.get_table("non_existent")
    assert non_existent is None


def test_text_to_sql_example_contract(sample_example: TextToSQLExample) -> None:
    """Verify TextToSQLExample serialization and attributes."""
    assert sample_example.id == "test_001"
    assert sample_example.split == DatasetSplit.DEV
    assert sample_example.difficulty == DifficultyLevel.MEDIUM

    # Serialization check
    dump = sample_example.model_dump(mode="json")
    assert dump["split"] == "dev"
    assert dump["db_id"] == "ecommerce_test"


def test_synthetic_provenance_validation() -> None:
    """Verify SyntheticProvenance records audit details."""
    prov = SyntheticProvenance(
        generator_model="qwen25_coder_7b",
        prompt_template_id="v1_nl_to_sql",
        validation_status="execution_verified",
        execution_verified=True,
        pass_rate=0.85,
        date_generated="2026-10-02T12:00:00Z",
    )
    assert prov.execution_verified is True
    assert prov.pass_rate == 0.85


def test_model_config_lora_and_qlora() -> None:
    """Verify ModelConfig with LoRA and QLoRA hyperparameters."""
    lora = LoRAHyperparameters(rank=32, alpha=64, dropout=0.1, target_modules=["q_proj", "v_proj"])
    qlora = QLoRAHyperparameters(bits=4, quant_type="nf4", use_double_quant=True)
    config = ModelConfig(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_type=ModelType.QLORA,
        context_window=8192,
        lora=lora,
        qlora=qlora,
    )
    assert config.model_type == ModelType.QLORA
    assert config.lora is not None
    assert config.lora.rank == 32
    assert config.qlora is not None
    assert config.qlora.bits == 4


def test_execution_result_contract() -> None:
    """Verify ExecutionResult fields."""
    res = ExecutionResult(
        query="SELECT COUNT(*) FROM users;",
        status=ExecutionStatus.SUCCESS,
        execution_time_ms=12.4,
        row_count=1,
        columns=["COUNT(*)"],
        result_fingerprint="abc123hash",
    )
    assert res.status == ExecutionStatus.SUCCESS
    assert res.row_count == 1
    assert res.error_message is None


def test_evaluation_metrics_contract() -> None:
    """Verify EvaluationMetrics with ConfidenceInterval."""
    ci = ConfidenceInterval(lower=0.72, upper=0.78, confidence_level=0.95)
    metrics = EvaluationMetrics(
        total_examples=500,
        valid_sql_rate=0.92,
        exact_match_accuracy=0.61,
        execution_accuracy=0.75,
        syntax_error_rate=0.08,
        timeout_rate=0.01,
        latency_p50_ms=45.0,
        latency_p95_ms=120.0,
        bootstrap_ci_execution_accuracy=ci,
        difficulty_breakdown={"easy": 0.88, "medium": 0.74, "hard": 0.60},
    )
    assert metrics.execution_accuracy == 0.75
    assert metrics.bootstrap_ci_execution_accuracy is not None
    assert metrics.bootstrap_ci_execution_accuracy.lower == 0.72


def test_experiment_config_validation() -> None:
    """Verify ExperimentConfig contract."""
    exp = ExperimentConfig(
        experiment_id="exp01_zero_shot",
        description="Zero-shot baseline evaluation",
        seed=42,
        model_id="qwen25_coder_7b",
        paradigm="zero-shot",
        eval_datasets=["spider:dev"],
        prompt=PromptConfig(template_name="standard", schema_format="ddl"),
    )
    assert exp.experiment_id == "exp01_zero_shot"
    assert exp.prompt.schema_format == "ddl"


def test_schema_metadata_missing_field_raises() -> None:
    """Ensure missing required field raises ValidationError."""
    with pytest.raises(ValidationError):
        ColumnMetadata(name="col")  # missing data_type
