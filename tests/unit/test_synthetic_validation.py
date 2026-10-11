"""Unit tests for synthetic dataset validation, provenance verification, and contamination auditing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.training.synthetic_validation import (
    SyntheticDataValidator,
    SyntheticRecord,
)


def _create_mock_valid_synthetic_record(record_id: str = "syn_001") -> SyntheticRecord:
    """Construct a fully valid synthetic record matching the Pydantic schema."""
    return SyntheticRecord(
        synthetic_id=record_id,
        question="How many singers are from France?",
        sql="SELECT count(*) FROM singer WHERE country = 'France';",
        database_id="concert_singer",
        schema_reference="schemas/concert_singer.sql",
        source_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        source_model_revision="main_rev_abc123",
        generation_timestamp="2026-10-10T12:00:00Z",
        generation_config_hash="c" * 64,
        source_dataset_reference="spider:train",
        source_record_ids=["spider_train_00010"],
        verification_status="verified",
        verification_method="semantic_execution_and_reference_match",
        verifier_version="1.0.0",
        verification_artifact_hash="d" * 64,
        license_or_usage_rights="Apache-2.0 research use",
    )


def test_synthetic_record_valid() -> None:
    """Verify valid synthetic record parses cleanly."""
    rec = _create_mock_valid_synthetic_record()
    assert rec.synthetic_id == "syn_001"
    assert rec.verification_status == "verified"


def test_synthetic_record_missing_provenance_rejected() -> None:
    """Verify missing required provenance fields fails validation."""
    with pytest.raises(ValidationError):
        SyntheticRecord(
            synthetic_id="syn_bad",
            question="What is the total?",
            sql="SELECT 1;",
            database_id="db_01",
            schema_reference="ref.sql",
            source_model_id="",  # Empty source model rejected
            source_model_revision="rev1",
            generation_timestamp="2026-10-10T12:00:00Z",
            generation_config_hash="c" * 64,
            source_dataset_reference="spider:train",
            source_record_ids=["id1"],
            verification_status="verified",
            verification_method="rule",
            verifier_version="1.0.0",
            verification_artifact_hash="d" * 64,
            license_or_usage_rights="CC-BY-4.0",
        )


def test_synthetic_record_forbidden_eval_source_rejected() -> None:
    """Verify referencing spider:dev or spider:test in source_dataset_reference is rejected."""
    with pytest.raises(ValidationError, match="source_dataset_reference"):
        SyntheticRecord(
            synthetic_id="syn_bad",
            question="What is the total?",
            sql="SELECT 1;",
            database_id="db_01",
            schema_reference="ref.sql",
            source_model_id="model",
            source_model_revision="rev1",
            generation_timestamp="2026-10-10T12:00:00Z",
            generation_config_hash="c" * 64,
            source_dataset_reference="spider:dev",  # Forbidden eval split!
            source_record_ids=["id1"],
            verification_status="verified",
            verification_method="rule",
            verifier_version="1.0.0",
            verification_artifact_hash="d" * 64,
            license_or_usage_rights="CC-BY-4.0",
        )


def test_synthetic_record_unverified_status() -> None:
    """Verify unverified status has is_verified = False."""
    rec = _create_mock_valid_synthetic_record().model_copy(
        update={"verification_status": "unverified"}
    )
    assert rec.is_verified is False


def test_validator_clean_batch_accepted(tmp_path: Path) -> None:
    """Verify clean, valid records are accepted."""
    # Setup mock sqlite DB for execution test
    db_path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE singer (id INT, country TEXT);")
    conn.execute("INSERT INTO singer VALUES (1, 'France');")
    conn.commit()
    conn.close()

    validator = SyntheticDataValidator(db_paths={"concert_singer": db_path})
    records = [_create_mock_valid_synthetic_record("syn_001")]
    report = validator.validate_dataset(records)

    assert report.total_records == 1
    assert report.accepted_records == 1
    assert report.rejected_records == 0
    assert report.review_required_records == 0
    assert report.is_training_eligible is True


def test_validator_rejects_duplicate_ids() -> None:
    """Verify duplicate synthetic IDs are detected and rejected."""
    validator = SyntheticDataValidator()
    rec1 = _create_mock_valid_synthetic_record("syn_dup")
    rec2 = _create_mock_valid_synthetic_record("syn_dup")

    report = validator.validate_dataset([rec1, rec2])
    assert report.accepted_records == 1
    assert report.rejected_records == 1
    detail = report.record_details[1]
    assert detail.status == "rejected"
    assert "Duplicate synthetic_id" in detail.reasons[0]


def test_validator_rejects_duplicate_sql_question_pairs() -> None:
    """Verify exact duplicate normalized question/SQL is detected and rejected."""
    validator = SyntheticDataValidator()
    rec1 = _create_mock_valid_synthetic_record("syn_001")
    rec2 = SyntheticRecord(
        synthetic_id="syn_002",
        question="  how many singers are from france?  ",  # Normalizes to rec1
        sql="select count(*) from singer where country = 'France';  ",
        database_id="concert_singer",
        schema_reference="schemas/concert_singer.sql",
        source_model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        source_model_revision="main_rev_abc123",
        generation_timestamp="2026-10-10T12:00:00Z",
        generation_config_hash="c" * 64,
        source_dataset_reference="spider:train",
        source_record_ids=["spider_train_00010"],
        verification_status="verified",
        verification_method="semantic_execution_and_reference_match",
        verifier_version="1.0.0",
        verification_artifact_hash="d" * 64,
        license_or_usage_rights="Apache-2.0",
    )

    report = validator.validate_dataset([rec1, rec2])
    assert report.accepted_records == 1
    assert report.rejected_records == 1
    assert "Duplicate normalized question/SQL" in report.record_details[1].reasons[0]


def test_validator_rejects_human_data_overlap() -> None:
    """Verify overlap with condition's human data is detected and rejected."""
    human_ex = TextToSQLExample(
        id="human_001",
        question="How many singers are from France?",
        db_id="concert_singer",
        gold_sql="SELECT count(*) FROM singer WHERE country = 'France';",
        dataset_name="spider",
        split=DatasetSplit.TRAIN,
    )
    validator = SyntheticDataValidator(human_examples=[human_ex])
    rec = _create_mock_valid_synthetic_record("syn_overlap")

    report = validator.validate_dataset([rec])
    assert report.rejected_records == 1
    assert "Exact duplicate of human training example" in report.record_details[0].reasons[0]


def test_validator_rejects_eval_split_contamination() -> None:
    """Verify exact overlap with dev/test benchmark split is rejected as contamination."""
    eval_ex = TextToSQLExample(
        id="eval_dev_001",
        question="How many singers are from France?",
        db_id="concert_singer",
        gold_sql="SELECT count(*) FROM singer WHERE country = 'France';",
        dataset_name="spider",
        split=DatasetSplit.DEV,
    )
    validator = SyntheticDataValidator(eval_examples=[eval_ex])
    rec = _create_mock_valid_synthetic_record("syn_leak")

    report = validator.validate_dataset([rec])
    assert report.rejected_records == 1
    assert report.record_details[0].contamination_cleared is False
    assert any("Contamination with evaluation split" in r for r in report.record_details[0].reasons)


def test_validator_fuzzy_overlap_flags_review_required() -> None:
    """Verify fuzzy n-gram overlap flags record as review-required rather than silent pass."""
    # Similar but not identical question
    eval_ex = TextToSQLExample(
        id="eval_dev_002",
        question="How many singers are from France country?",
        db_id="concert_singer",
        gold_sql="SELECT count(singer_id) FROM singer WHERE country = 'France';",
        dataset_name="spider",
        split=DatasetSplit.DEV,
    )
    validator = SyntheticDataValidator(eval_examples=[eval_ex], fuzzy_threshold=0.70)
    rec = _create_mock_valid_synthetic_record("syn_fuzzy")

    report = validator.validate_dataset([rec])
    # Should flag review_required
    assert report.review_required_records == 1
    assert (
        report.is_training_eligible is False
    )  # Cannot train with unresolved review-required records!


def test_validator_execution_failure_rejected(tmp_path: Path) -> None:
    """Verify SQL execution syntax/table error in sandbox database causes rejection."""
    # Database without 'singer' table
    db_path = tmp_path / "empty_db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE different_table (id INT);")
    conn.commit()
    conn.close()

    validator = SyntheticDataValidator(db_paths={"concert_singer": db_path})
    rec = _create_mock_valid_synthetic_record("syn_exec_fail")

    report = validator.validate_dataset([rec])
    assert report.rejected_records == 1
    assert report.record_details[0].sql_execution_valid is False
    assert any("execution failed" in r for r in report.record_details[0].reasons)


def test_validator_unsafe_sql_rejected() -> None:
    """Verify destructive SQL statements (DROP, DELETE, UPDATE) are rejected."""
    validator = SyntheticDataValidator()
    rec = _create_mock_valid_synthetic_record("syn_unsafe").model_copy(
        update={"sql": "DROP TABLE singer;"}
    )
    report = validator.validate_dataset([rec])
    assert report.rejected_records == 1
    assert report.record_details[0].schema_valid is False
    assert any("Unsafe SQL statement" in r for r in report.record_details[0].reasons)
