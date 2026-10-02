"""Unit tests for BIRD Mini-Dev ingestion adapter and subset ambiguity resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlforge.data.bird import (
    BIRDAdapterError,
    BirdSubsetVariant,
    create_bird_provenance,
    load_bird_mini_dev,
    parse_bird_tables,
)
from sqlforge.schemas.examples import DatasetSplit, DifficultyLevel


@pytest.fixture
def bird_fixtures_dir() -> Path:
    """Return path to offline BIRD test fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "dataset" / "bird"


def test_bird_mini_dev_loading(bird_fixtures_dir: Path) -> None:
    """Test loading BIRD Mini-Dev examples and preserving external evidence."""
    tables_file = bird_fixtures_dir / "dev_tables.json"
    schemas = parse_bird_tables(tables_file)

    assert "financial" in schemas

    dev_file = bird_fixtures_dir / "dev.json"
    examples = load_bird_mini_dev(dev_file, schemas, variant=BirdSubsetVariant.MINI_DEV_500)

    assert len(examples) == 2
    for ex in examples:
        assert ex.dataset_name == "bird_mini"
        assert ex.split == DatasetSplit.DEV
        assert ex.db_id == "financial"
        assert ex.evidence is not None  # Evidence must be preserved
        assert ex.gold_sql is not None

    # Check difficulty mapping ("simple" -> EASY, "moderate" -> MEDIUM)
    assert examples[0].difficulty == DifficultyLevel.EASY
    assert examples[1].difficulty == DifficultyLevel.MEDIUM


def test_bird_subset_variants() -> None:
    """Test enum representations and metadata for 500 and 780 subset variants."""
    assert BirdSubsetVariant.MINI_DEV_500.value == "mini_dev_500"
    assert BirdSubsetVariant.MINI_DEV_780.value == "mini_dev_780"


def test_bird_count_enforcement(bird_fixtures_dir: Path) -> None:
    """Test that enforce_expected_count raises when fixture has fewer than 500 instances."""
    tables_file = bird_fixtures_dir / "dev_tables.json"
    schemas = parse_bird_tables(tables_file)
    dev_file = bird_fixtures_dir / "dev.json"

    with pytest.raises(BIRDAdapterError, match="expected 500 examples, found 2"):
        load_bird_mini_dev(
            dev_file,
            schemas,
            variant=BirdSubsetVariant.MINI_DEV_500,
            enforce_expected_count=True,
        )


def test_bird_provenance_and_license(bird_fixtures_dir: Path) -> None:
    """Test BIRD provenance captures CC BY-NC-SA 4.0 license notes."""
    tables_file = bird_fixtures_dir / "dev_tables.json"
    dev_file = bird_fixtures_dir / "dev.json"

    prov = create_bird_provenance(tables_file, dev_file, variant=BirdSubsetVariant.MINI_DEV_500)
    assert prov.dataset_id == "bird_mini"
    assert "CC-BY-NC-SA-4.0" in prov.license_id
    assert "Non-Commercial" in str(prov.license_notes)
    assert f"{BirdSubsetVariant.MINI_DEV_500.value}.json" in prov.raw_files_checksums
