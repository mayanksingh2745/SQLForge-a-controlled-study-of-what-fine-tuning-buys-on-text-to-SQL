"""Unit tests for deterministic fixture dataset loading and validation."""

import json
from pathlib import Path

import pytest

from sqlforge.data.fixtures import FixtureExample, load_fixture_dataset


def test_load_default_fixtures() -> None:
    """Verify that default fixtures load cleanly, deterministically, and validate schema."""
    examples = load_fixture_dataset()
    assert len(examples) >= 3

    # Check deterministic ordering by example_id
    ids = [ex.example_id for ex in examples]
    assert ids == sorted(ids)

    for ex in examples:
        assert isinstance(ex, FixtureExample)
        assert len(ex.example_id) > 0
        assert len(ex.question) > 0
        assert len(ex.db_id) > 0
        assert len(ex.schema_context) > 0
        assert len(ex.gold_sql) > 0


def test_fixtures_duplicate_id_raises(tmp_path: Path) -> None:
    """Verify that duplicate example IDs are rejected with a clear error message."""
    bad_data = [
        {
            "example_id": "dup_01",
            "question": "Q1?",
            "db_id": "db1",
            "schema_context": "CREATE TABLE t (a int);",
            "gold_sql": "SELECT 1;",
        },
        {
            "example_id": "dup_01",
            "question": "Q2?",
            "db_id": "db1",
            "schema_context": "CREATE TABLE t (a int);",
            "gold_sql": "SELECT 2;",
        },
    ]
    fixture_file = tmp_path / "dup_fixtures.json"
    fixture_file.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate example_id 'dup_01' detected"):
        load_fixture_dataset(fixture_file)


def test_fixtures_empty_dataset_raises(tmp_path: Path) -> None:
    """Verify that an empty fixture dataset is rejected."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        load_fixture_dataset(empty_file)


def test_fixtures_malformed_json_raises(tmp_path: Path) -> None:
    """Verify that malformed JSON raises ValueError."""
    corrupted_file = tmp_path / "corrupted.json"
    corrupted_file.write_text("[{ malformed ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON in fixture dataset"):
        load_fixture_dataset(corrupted_file)


def test_fixtures_missing_required_field_raises(tmp_path: Path) -> None:
    """Verify that missing required fields trigger validation failure."""
    bad_item = [
        {
            "example_id": "incomplete_01",
            # missing question
            "db_id": "db1",
            "schema_context": "CREATE TABLE t (a int);",
            "gold_sql": "SELECT 1;",
        }
    ]
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad_item), encoding="utf-8")

    with pytest.raises(ValueError, match="Validation failed for fixture example"):
        load_fixture_dataset(bad_file)


def test_fixtures_missing_file_raises() -> None:
    """Verify that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_fixture_dataset("non_existent_fixture_path_xyz.json")
