"""Unit tests for dataset contracts, provenance models, normalization, and hashing."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sqlforge.data.manifest import (
    build_dataset_manifest,
    load_examples_from_file,
    save_manifest,
    verify_dataset_manifest,
    write_processed_dataset,
)
from sqlforge.data.normalization import (
    compute_ngram_jaccard,
    normalize_question,
    normalize_sql,
)
from sqlforge.schemas.dataset import (
    DatasetProvenance,
    LicenseIdentifier,
)
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata


def test_normalization_question() -> None:
    """Test question normalization strips punctuation, whitespace, and lowercases."""
    raw = "  What is the Total Capacity of Stadiums in France???  \n"
    norm = normalize_question(raw)
    assert norm == "what is the total capacity of stadiums in france"


def test_normalization_sql() -> None:
    """Test SQL normalization strips line and block comments, trailing semicolons, and collapses spaces."""
    raw_sql = """
        -- Select highest capacity
        SELECT Name, Capacity /* inline comment */
        FROM stadium
        WHERE Capacity > 50000;   ;
    """
    norm = normalize_sql(raw_sql)
    assert norm == "select name, capacity from stadium where capacity > 50000"


def test_ngram_jaccard_similarity() -> None:
    """Test n-gram Jaccard similarity edge cases and scoring."""
    # Identical
    assert (
        compute_ngram_jaccard("Find all singers from France", "find all singers from france", n=3)
        == 1.0
    )

    # Completely different
    assert (
        compute_ngram_jaccard(
            "Find all singers from France", "List aircraft types by manufacturer", n=3
        )
        == 0.0
    )

    # Empty inputs
    assert compute_ngram_jaccard("", "", n=3) == 0.0
    assert compute_ngram_jaccard("Some text", "", n=3) == 0.0

    # Partial overlap
    q1 = "What is the average capacity of stadiums in France?"
    q2 = "What is the maximum capacity of stadiums in France?"
    sim = compute_ngram_jaccard(q1, q2, n=3)
    assert 0.4 <= sim <= 0.8


def test_dataset_provenance_contract() -> None:
    """Test validation of DatasetProvenance model."""
    prov = DatasetProvenance(
        dataset_id="spider",
        source_name="Spider 1.0",
        release_version="1.0",
        license_id=LicenseIdentifier.CC_BY_SA_4_0,
        dialect="sqlite",
        created_at=datetime.now(UTC).isoformat(),
    )
    assert prov.dataset_id == "spider"
    assert prov.license_id == LicenseIdentifier.CC_BY_SA_4_0

    # Empty dataset_id must fail validation
    with pytest.raises(ValidationError):
        DatasetProvenance(
            dataset_id="",
            source_name="Spider",
            release_version="1.0",
            license_id="CC-BY-SA-4.0",
            created_at=datetime.now(UTC).isoformat(),
        )


def test_write_and_verify_dataset_manifest() -> None:
    """Test deterministic JSONL serialization and manifest verification."""
    examples = [
        TextToSQLExample(
            id=f"ex_{i:02d}",
            question=f"Question {i}",
            db_id="test_db",
            gold_sql=f"SELECT {i};",
            dataset_name="test_data",
            split=DatasetSplit.TRAIN,
        )
        for i in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        jsonl_path = tmp_path / "data" / "processed" / "train.jsonl"
        count, file_hash = write_processed_dataset(examples, jsonl_path)

        assert count == 5
        assert len(file_hash) == 64

        # Read back examples
        loaded = load_examples_from_file(jsonl_path)
        assert len(loaded) == 5
        assert loaded[0].id == "ex_00"

        prov = DatasetProvenance(
            dataset_id="test_data",
            source_name="Test Source",
            release_version="1.0",
            license_id=LicenseIdentifier.APACHE_2_0,
            created_at=datetime.now(UTC).isoformat(),
        )
        dummy_schemas = {"test_db": SchemaMetadata(db_id="test_db")}

        manifest = build_dataset_manifest(
            dataset_id="test_data",
            version="1.0",
            splits_data={DatasetSplit.TRAIN: (examples, jsonl_path)},
            schemas=dummy_schemas,
            provenance=prov,
        )

        manifest_file = tmp_path / "dataset_manifest.json"
        save_manifest(manifest, manifest_file)

        # Verify manifest
        res = verify_dataset_manifest(manifest_file, project_root=tmp_path)
        assert res["verified"] is True
        assert res["total_examples"] == 5
        assert len(res["missing_files"]) == 0
        assert len(res["hash_mismatches"]) == 0

        # Tamper with file to verify hash mismatch detection
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write("\n")
        tampered_res = verify_dataset_manifest(manifest_file, project_root=tmp_path)
        assert tampered_res["verified"] is False
        assert len(tampered_res["hash_mismatches"]) == 1
