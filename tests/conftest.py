"""Pytest fixtures for SQLForge test suite."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from sqlforge.schemas.examples import DatasetSplit, DifficultyLevel, TextToSQLExample
from sqlforge.schemas.metadata import (
    ColumnMetadata,
    ForeignKeyMetadata,
    SchemaMetadata,
    TableMetadata,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click test CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_schema() -> SchemaMetadata:
    """Provide a representative test SchemaMetadata instance."""
    users_table = TableMetadata(
        table_name="users",
        columns=[
            ColumnMetadata(
                name="user_id",
                data_type="INTEGER",
                is_primary_key=True,
                sample_values=[1, 2, 3],
            ),
            ColumnMetadata(
                name="username",
                data_type="VARCHAR(50)",
                is_primary_key=False,
                sample_values=["alice", "bob"],
            ),
            ColumnMetadata(
                name="created_at",
                data_type="DATETIME",
                is_primary_key=False,
            ),
        ],
        primary_keys=["user_id"],
        row_count=100,
        description="Application registered users",
    )

    orders_table = TableMetadata(
        table_name="orders",
        columns=[
            ColumnMetadata(
                name="order_id",
                data_type="INTEGER",
                is_primary_key=True,
                sample_values=[101, 102],
            ),
            ColumnMetadata(
                name="user_id",
                data_type="INTEGER",
                is_foreign_key=True,
                foreign_table="users",
                foreign_column="user_id",
                sample_values=[1, 2],
            ),
            ColumnMetadata(
                name="total_amount",
                data_type="DECIMAL(10,2)",
                sample_values=[49.99, 120.00],
            ),
        ],
        primary_keys=["order_id"],
        foreign_keys=[
            ForeignKeyMetadata(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="user_id",
            )
        ],
        row_count=500,
        description="Customer placed orders",
    )

    return SchemaMetadata(
        db_id="ecommerce_test",
        dialect="sqlite",
        tables=[users_table, orders_table],
        foreign_keys=[
            ForeignKeyMetadata(
                source_table="orders",
                source_column="user_id",
                target_table="users",
                target_column="user_id",
            )
        ],
        description="Test e-commerce database",
    )


@pytest.fixture
def sample_example() -> TextToSQLExample:
    """Provide a representative test TextToSQLExample instance."""
    return TextToSQLExample(
        id="test_001",
        question="How many users registered in 2026?",
        db_id="ecommerce_test",
        gold_sql="SELECT COUNT(*) FROM users WHERE strftime('%Y', created_at) = '2026';",
        dataset_name="spider",
        split=DatasetSplit.DEV,
        difficulty=DifficultyLevel.MEDIUM,
        evidence="Extract the year component using strftime",
    )


@pytest.fixture
def temp_artifact_dir(tmp_path: Path) -> Path:
    """Provide a temporary artifact directory for isolated testing."""
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True, exist_ok=True)
    return artifact_path
