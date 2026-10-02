"""Unit tests for Spider 1.0 dataset ingestion adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sqlforge.data.spider import (
    SpiderAdapterError,
    create_spider_provenance,
    load_spider_split,
    parse_spider_tables,
)
from sqlforge.schemas.examples import DatasetSplit


@pytest.fixture
def spider_fixtures_dir() -> Path:
    """Return path to offline Spider test fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "dataset" / "spider"


def test_spider_tables_parsing(spider_fixtures_dir: Path) -> None:
    """Test parsing Spider tables.json into valid SchemaMetadata objects."""
    tables_file = spider_fixtures_dir / "tables.json"
    schemas = parse_spider_tables(tables_file)

    assert "stadium" in schemas
    assert "concert_singer" in schemas
    assert "car_1" in schemas

    # Check stadium table structure
    stadium_schema = schemas["stadium"]
    assert stadium_schema.db_id == "stadium"
    assert len(stadium_schema.tables) == 1
    assert stadium_schema.tables[0].table_name == "stadium"
    assert "Stadium_ID" in stadium_schema.tables[0].primary_keys

    # Check foreign keys in concert_singer
    concert_schema = schemas["concert_singer"]
    assert len(concert_schema.foreign_keys) >= 1
    fk = concert_schema.foreign_keys[0]
    assert fk.target_table == "stadium"


def test_spider_split_loading(spider_fixtures_dir: Path) -> None:
    """Test loading Spider train and dev splits."""
    tables_file = spider_fixtures_dir / "tables.json"
    schemas = parse_spider_tables(tables_file)

    train_file = spider_fixtures_dir / "train_spider.json"
    train_examples = load_spider_split(train_file, schemas, split=DatasetSplit.TRAIN)

    assert len(train_examples) == 3
    for ex in train_examples:
        assert ex.dataset_name == "spider"
        assert ex.split == DatasetSplit.TRAIN
        assert ex.db_id in schemas
        assert ex.gold_sql is not None

    dev_file = spider_fixtures_dir / "dev.json"
    dev_examples = load_spider_split(dev_file, schemas, split=DatasetSplit.DEV)
    assert len(dev_examples) == 2
    for ex in dev_examples:
        assert ex.split == DatasetSplit.DEV
        assert ex.db_id == "car_1"


def test_spider_tables_file_not_found() -> None:
    """Test error handling when tables.json is missing."""
    with pytest.raises(FileNotFoundError):
        parse_spider_tables("non_existent_tables.json")


def test_spider_missing_db_reference(spider_fixtures_dir: Path) -> None:
    """Test rejection when an example references an unknown db_id."""
    tables_file = spider_fixtures_dir / "tables.json"
    schemas = parse_spider_tables(tables_file)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"db_id": "ghost_db", "question": "test?", "query": "SELECT 1;"}], f)
        bad_file = Path(f.name)

    try:
        with pytest.raises(SpiderAdapterError, match="Unknown db_id 'ghost_db'"):
            load_spider_split(bad_file, schemas, split=DatasetSplit.TRAIN)
    finally:
        bad_file.unlink()


def test_spider_malformed_example(spider_fixtures_dir: Path) -> None:
    """Test rejection when required fields are missing."""
    tables_file = spider_fixtures_dir / "tables.json"
    schemas = parse_spider_tables(tables_file)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"db_id": "stadium", "question": "", "query": "SELECT 1;"}], f)
        bad_file = Path(f.name)

    try:
        with pytest.raises(SpiderAdapterError, match="Missing or empty 'question'"):
            load_spider_split(bad_file, schemas, split=DatasetSplit.TRAIN)
    finally:
        bad_file.unlink()


def test_spider_provenance_creation(spider_fixtures_dir: Path) -> None:
    """Test generating a Spider provenance record with raw file checksums."""
    tables_file = spider_fixtures_dir / "tables.json"
    train_file = spider_fixtures_dir / "train_spider.json"
    dev_file = spider_fixtures_dir / "dev.json"

    prov = create_spider_provenance(tables_file, train_file, dev_file)
    assert prov.dataset_id == "spider"
    assert "tables.json" in prov.raw_files_checksums
    assert "train_spider.json" in prov.raw_files_checksums
    assert len(prov.raw_files_checksums["tables.json"]) == 64
