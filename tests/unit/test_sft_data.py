"""Unit tests for training data formatting, split isolation, and schema serialization."""

from __future__ import annotations

import pytest

from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata
from sqlforge.training.data import (
    DisallowedSplitError,
    MalformedExampleError,
    MissingSchemaError,
    SFTDatasetFormatter,
)


@pytest.fixture
def sample_schema() -> SchemaMetadata:
    """Fixture providing a test database schema."""
    return SchemaMetadata(
        db_id="university",
        dialect="sqlite",
        tables=[
            TableMetadata(
                table_name="student",
                columns=[
                    ColumnMetadata(name="student_id", data_type="INTEGER", is_primary_key=True),
                    ColumnMetadata(name="name", data_type="TEXT"),
                    ColumnMetadata(name="gpa", data_type="REAL"),
                ],
            )
        ],
    )


@pytest.fixture
def sample_train_example() -> TextToSQLExample:
    """Fixture providing a valid training example."""
    return TextToSQLExample(
        id="spider_train_0001",
        question="What is the name of the student with the highest GPA?",
        db_id="university",
        gold_sql="SELECT name FROM student ORDER BY gpa DESC LIMIT 1;",
        dataset_name="spider",
        split=DatasetSplit.TRAIN,
    )


class TestSFTDatasetFormatter:
    """Test suite for SFTDatasetFormatter and split enforcement."""

    def test_format_valid_single_example(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Verify successful formatting of an eligible training instance."""
        formatter = SFTDatasetFormatter(schema_format="ddl")
        sft_ex = formatter.format_single(sample_train_example, sample_schema)

        assert sft_ex.example_id == "spider_train_0001"
        assert sft_ex.db_id == "university"
        assert sft_ex.split == DatasetSplit.TRAIN
        assert "CREATE TABLE student" in sft_ex.prompt_text
        assert "### Question:" in sft_ex.prompt_text
        assert sample_train_example.question in sft_ex.prompt_text
        assert sft_ex.target_text.endswith(";")
        assert sft_ex.target_text == "SELECT name FROM student ORDER BY gpa DESC LIMIT 1;"

    def test_quarantine_rejects_dev_split(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Enforce that DEV split examples raise DisallowedSplitError."""
        formatter = SFTDatasetFormatter()
        dev_example = sample_train_example.model_copy(update={"split": DatasetSplit.DEV})

        with pytest.raises(DisallowedSplitError, match="strictly rejects DEV, TEST, and HELD_OUT"):
            formatter.format_single(dev_example, sample_schema)

    def test_quarantine_rejects_test_and_held_out_splits(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Enforce that TEST and HELD_OUT split examples raise DisallowedSplitError."""
        formatter = SFTDatasetFormatter()
        test_example = sample_train_example.model_copy(update={"split": DatasetSplit.TEST})
        held_out_example = sample_train_example.model_copy(update={"split": DatasetSplit.HELD_OUT})

        with pytest.raises(DisallowedSplitError):
            formatter.format_single(test_example, sample_schema)

        with pytest.raises(DisallowedSplitError):
            formatter.format_single(held_out_example, sample_schema)

    def test_reject_missing_gold_sql(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Verify malformed examples without gold SQL raise MalformedExampleError."""
        formatter = SFTDatasetFormatter()
        bad_example = sample_train_example.model_copy(update={"gold_sql": ""})

        with pytest.raises(MalformedExampleError, match="missing or empty gold_sql"):
            formatter.format_single(bad_example, sample_schema)

    def test_reject_missing_question(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Verify malformed examples without question raise MalformedExampleError."""
        formatter = SFTDatasetFormatter()
        bad_example = sample_train_example.model_copy(update={"question": "  "})

        with pytest.raises(MalformedExampleError, match="missing or empty question"):
            formatter.format_single(bad_example, sample_schema)

    def test_reject_mismatched_schema(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Verify that passing mismatched schema db_id raises MissingSchemaError."""
        formatter = SFTDatasetFormatter()
        bad_schema = sample_schema.model_copy(update={"db_id": "other_db"})

        with pytest.raises(MissingSchemaError, match="does not match example db_id"):
            formatter.format_single(sample_train_example, bad_schema)

    def test_format_dataset_batch(
        self, sample_train_example: TextToSQLExample, sample_schema: SchemaMetadata
    ) -> None:
        """Verify batch dataset formatting and summary statistics."""
        formatter = SFTDatasetFormatter()
        examples = [sample_train_example.model_copy(update={"id": f"ex_{i}"}) for i in range(5)]
        schemas = {"university": sample_schema}

        sft_records, summary = formatter.format_dataset(examples, schemas)

        assert len(sft_records) == 5
        assert summary.valid_examples_count == 5
        assert summary.rejected_count == 0
        assert summary.unique_databases == 1
        assert summary.avg_prompt_chars > 0
        assert summary.avg_target_chars > 0
