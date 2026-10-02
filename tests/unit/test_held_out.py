"""Unit tests for Custom Held-Out Benchmark foundation and schema contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sqlforge.data.held_out import (
    HELD_OUT_DB_ID,
    HeldOutAdapterError,
    create_held_out_provenance,
    get_subscription_analytics_schema,
    load_held_out_dataset,
)
from sqlforge.schemas.examples import DatasetSplit


@pytest.fixture
def held_out_fixtures_dir() -> Path:
    """Return path to offline Custom Held-Out test fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "dataset" / "custom_held_out"


def test_subscription_analytics_schema() -> None:
    """Verify subscription_analytics_db contract has all 6 required tables and foreign keys."""
    schema = get_subscription_analytics_schema()

    assert schema.db_id == HELD_OUT_DB_ID
    table_names = {t.table_name for t in schema.tables}
    expected_tables = {
        "customers",
        "plans",
        "subscriptions",
        "invoices",
        "transactions",
        "support_tickets",
    }
    assert table_names == expected_tables

    # Check foreign keys exist and reference expected tables
    assert len(schema.foreign_keys) == 5
    fk_targets = {fk.target_table for fk in schema.foreign_keys}
    assert "customers" in fk_targets
    assert "plans" in fk_targets
    assert "subscriptions" in fk_targets
    assert "invoices" in fk_targets


def test_load_held_out_dataset(held_out_fixtures_dir: Path) -> None:
    """Test loading custom held-out examples and ensuring strict isolation partition."""
    examples_file = held_out_fixtures_dir / "held_out_examples.json"
    examples = load_held_out_dataset(examples_file)

    assert len(examples) == 2
    for ex in examples:
        assert ex.dataset_name == "custom_held_out"
        assert ex.split == DatasetSplit.HELD_OUT
        assert ex.db_id == HELD_OUT_DB_ID
        assert ex.gold_sql is not None


def test_held_out_db_id_mismatch() -> None:
    """Test rejection when an example references an unexpected database ID."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(
            [{"id": "bad_01", "db_id": "spider_db", "question": "q?", "gold_sql": "SELECT 1;"}], f
        )
        bad_file = Path(f.name)

    try:
        with pytest.raises(HeldOutAdapterError, match="unexpected db_id 'spider_db'"):
            load_held_out_dataset(bad_file)
    finally:
        bad_file.unlink()


def test_held_out_provenance() -> None:
    """Test Custom Held-Out provenance generation under Apache-2.0."""
    prov = create_held_out_provenance()
    assert prov.dataset_id == "custom_held_out"
    assert "Apache-2.0" in prov.license_id
