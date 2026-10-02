"""Unit tests for PromptEngine, token budgeting, and few-shot assembly."""

from __future__ import annotations

import pytest

from sqlforge.prompting.engine import (
    PromptBudgetExceededError,
    PromptEngine,
    estimate_token_count,
)
from sqlforge.prompting.retriever import BM25Retriever
from sqlforge.prompting.serializers import DDLSerializer
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata


@pytest.fixture
def sample_schema() -> SchemaMetadata:
    """Fixture schema for testing prompt assembly."""
    return SchemaMetadata(
        db_id="car_1",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="cars_data",
                columns=[
                    ColumnMetadata(name="id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="model", data_type="TEXT"),
                    ColumnMetadata(name="horsepower", data_type="INTEGER"),
                ],
                primary_keys=["id"],
            )
        ],
    )


@pytest.fixture
def sample_train_examples() -> list[TextToSQLExample]:
    """Fixture training examples for retriever."""
    return [
        TextToSQLExample(
            id=f"train_{i}",
            question=f"Find car model {i} with high horsepower",
            db_id="car_1",
            gold_sql=f"SELECT model FROM cars_data WHERE horsepower > {i * 10};",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
        for i in range(1, 10)
    ]


@pytest.fixture
def sample_query_example() -> TextToSQLExample:
    """Target inference query example."""
    return TextToSQLExample(
        id="dev_001",
        question="What car model has the highest horsepower?",
        db_id="car_1",
        gold_sql="SELECT model FROM cars_data ORDER BY horsepower DESC LIMIT 1;",
        dataset_name="spider",
        split=DatasetSplit.DEV,
    )


def test_estimate_token_count() -> None:
    """Test token estimation heuristic."""
    assert estimate_token_count("") == 0
    text = "SELECT model FROM cars_data WHERE horsepower > 100;"
    est = estimate_token_count(text)
    assert est > 0
    assert est >= len(text.split())


def test_zero_shot_prompt_assembly(
    sample_schema: SchemaMetadata,
    sample_query_example: TextToSQLExample,
) -> None:
    """Test assembling a zero-shot prompt with DDL schema."""
    engine = PromptEngine(schema_format="ddl")
    prompt = engine.assemble(sample_query_example, sample_schema, k_shots=0)

    assert prompt.k_shots == 0
    assert prompt.db_id == "car_1"
    assert "### Database Schema (car_1):" in prompt.prompt_text
    assert "CREATE TABLE cars_data" in prompt.prompt_text
    assert "### Question:" in prompt.prompt_text
    assert "What car model has the highest horsepower?" in prompt.prompt_text
    assert prompt.prompt_text.endswith("SELECT ")
    assert len(prompt.retrieved_demonstration_ids) == 0
    assert prompt.evidence_included is False


def test_few_shot_prompt_assembly(
    sample_schema: SchemaMetadata,
    sample_train_examples: list[TextToSQLExample],
    sample_query_example: TextToSQLExample,
) -> None:
    """Test assembling few-shot prompts with k in {1, 3, 5}."""
    retriever = BM25Retriever(sample_train_examples)
    engine = PromptEngine(schema_format="compact", retriever=retriever)

    for k in (1, 3, 5):
        prompt = engine.assemble(sample_query_example, sample_schema, k_shots=k)
        assert prompt.k_shots == k
        assert len(prompt.retrieved_demonstration_ids) == k
        assert "### Examples:" in prompt.prompt_text
        assert f"Example {k}:" in prompt.prompt_text


def test_bird_domain_evidence_integration(sample_schema: SchemaMetadata) -> None:
    """Test that BIRD external evidence is attached to the prompt when present."""
    bird_example = TextToSQLExample(
        id="bird_dev_01",
        question="Which car has the highest acceleration?",
        db_id="car_1",
        gold_sql="SELECT model FROM cars_data ORDER BY acceleration DESC LIMIT 1;",
        dataset_name="bird_mini",
        split=DatasetSplit.DEV,
        evidence="Acceleration is measured in seconds from 0 to 60 mph.",
    )
    engine = PromptEngine(schema_format="ddl")
    prompt = engine.assemble(bird_example, sample_schema, k_shots=0)

    assert prompt.evidence_included is True
    assert "### Domain Evidence:" in prompt.prompt_text
    assert "Acceleration is measured in seconds from 0 to 60 mph." in prompt.prompt_text


def test_budget_pruning_demonstrations(
    sample_schema: SchemaMetadata,
    sample_train_examples: list[TextToSQLExample],
    sample_query_example: TextToSQLExample,
) -> None:
    """Test that demonstrations are pruned when exceeding context token budget."""
    retriever = BM25Retriever(sample_train_examples)
    engine = PromptEngine(schema_format="ddl", retriever=retriever)

    # First, measure unconstrained token count with k=5
    unconstrained = engine.assemble(sample_query_example, sample_schema, k_shots=5, max_tokens=None)
    full_tokens = unconstrained.token_count_estimate

    # Now enforce a budget that is smaller than full_tokens, but sufficient for k=2
    budget = full_tokens - 80
    pruned = engine.assemble(sample_query_example, sample_schema, k_shots=5, max_tokens=budget)

    assert pruned.truncation_applied is True
    assert pruned.k_shots < 5
    assert len(pruned.truncation_log) > 0
    assert pruned.token_count_estimate <= budget


def test_budget_pruning_ddl_comments_and_samples(
    sample_train_examples: list[TextToSQLExample],
    sample_query_example: TextToSQLExample,
) -> None:
    """Test that DDL serializer comments and sample rows are pruned if demonstrations are gone."""
    schema = SchemaMetadata(
        db_id="car_1",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="cars_data",
                description="Comprehensive vehicle telemetry dataset",
                columns=[
                    ColumnMetadata(
                        name="id",
                        data_type="INTEGER",
                        is_primary_key=True,
                        description="Unique vehicle identification record",
                        sample_values=[101, 102, 103],
                    ),
                    ColumnMetadata(
                        name="model",
                        data_type="TEXT",
                        description="Vehicle manufacturing model name",
                        sample_values=["Model S", "Mustang", "Corvette"],
                    ),
                ],
            )
        ],
    )
    serializer = DDLSerializer(include_comments=True, include_sample_rows=3)
    engine = PromptEngine(serializer=serializer, default_max_tokens=None)

    unconstrained = engine.assemble(sample_query_example, schema, k_shots=0)
    assert "-- Table: cars_data" in unconstrained.prompt_text

    # Force a tight budget that requires dropping comments
    tight_budget = unconstrained.token_count_estimate - 25
    pruned = engine.assemble(sample_query_example, schema, k_shots=0, max_tokens=tight_budget)

    assert pruned.truncation_applied is True
    assert any("Pruned DDL" in log for log in pruned.truncation_log)
    assert "-- Table: cars_data" not in pruned.prompt_text


def test_budget_hard_exceeded_raises_error(
    sample_schema: SchemaMetadata,
    sample_query_example: TextToSQLExample,
) -> None:
    """Test that PromptBudgetExceededError is raised when even k=0 exceeds budget."""
    engine = PromptEngine(schema_format="ddl")
    # Impossibly small token budget (e.g. 5 tokens)
    with pytest.raises(PromptBudgetExceededError, match="exceeding budget limit"):
        engine.assemble(sample_query_example, sample_schema, k_shots=0, max_tokens=5)


def test_missing_retriever_for_few_shot_raises(
    sample_schema: SchemaMetadata,
    sample_query_example: TextToSQLExample,
) -> None:
    """Test that requesting k_shots > 0 without a retriever raises ValueError."""
    engine = PromptEngine(retriever=None)
    with pytest.raises(ValueError, match="Retriever is required"):
        engine.assemble(sample_query_example, sample_schema, k_shots=3)


def test_all_serialization_formats_supported(
    sample_schema: SchemaMetadata,
    sample_query_example: TextToSQLExample,
) -> None:
    """Test engine supports DDL, Compact Pipe, and JSON formats."""
    for fmt in ("ddl", "compact", "json"):
        engine = PromptEngine(schema_format=fmt)
        prompt = engine.assemble(sample_query_example, sample_schema, k_shots=0)
        assert prompt.schema_format == fmt
        assert len(prompt.prompt_text) > 0
