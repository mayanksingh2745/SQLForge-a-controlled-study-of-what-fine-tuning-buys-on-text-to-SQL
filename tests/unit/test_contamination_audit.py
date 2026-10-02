"""Unit tests for contamination audit engine, duplicate detection, and isolation guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlforge.data.audit import (
    ContaminationAuditor,
    IsolationGuard,
    LeakageContaminationError,
)
from sqlforge.data.manifest import load_examples_from_file
from sqlforge.data.spider import load_spider_split, parse_spider_tables
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample


@pytest.fixture
def fixtures_root() -> Path:
    """Return path to offline dataset fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "dataset"


def test_clean_partitions_audit(fixtures_root: Path) -> None:
    """Test that disjoint Spider train, Spider dev, and custom held-out pass contamination audit."""
    tables_file = fixtures_root / "spider" / "tables.json"
    schemas = parse_spider_tables(tables_file)

    train_ex = load_spider_split(
        fixtures_root / "spider" / "train_spider.json", schemas, split=DatasetSplit.TRAIN
    )
    dev_ex = load_spider_split(
        fixtures_root / "spider" / "dev.json", schemas, split=DatasetSplit.DEV
    )

    auditor = ContaminationAuditor(n_gram_size=4, fuzzy_threshold=0.85)
    report = auditor.audit_partitions(train_ex, {"dev": dev_ex})

    assert report.passed is True
    assert len(report.exact_question_duplicates) == 0
    assert len(report.exact_sql_duplicates) == 0
    assert len(report.fuzzy_question_overlaps) == 0
    assert len(report.schema_disjointness_violations) == 0
    assert len(report.split_leakage_violations) == 0


def test_contaminated_partitions_audit(fixtures_root: Path) -> None:
    """Test that deliberate contamination triggers all violation detectors."""
    tables_file = fixtures_root / "spider" / "tables.json"
    schemas = parse_spider_tables(tables_file)

    bad_train_ex = load_examples_from_file(
        fixtures_root / "contaminated" / "train_contaminated.json"
    )
    dev_ex = load_spider_split(
        fixtures_root / "spider" / "dev.json", schemas, split=DatasetSplit.DEV
    )

    auditor = ContaminationAuditor(n_gram_size=4, fuzzy_threshold=0.85)
    report = auditor.audit_partitions(bad_train_ex, {"dev": dev_ex})

    assert report.passed is False
    # Detected exact duplicate question
    assert len(report.exact_question_duplicates) >= 1
    # Detected schema collision (car_1 appears in train and dev)
    assert len(report.schema_disjointness_violations) >= 1
    # Detected split leakage (split='dev' in train file)
    assert len(report.split_leakage_violations) >= 1
    # Detected fuzzy overlap
    assert len(report.fuzzy_question_overlaps) >= 1


def test_isolation_guard_training_and_retrieval() -> None:
    """Test runtime isolation assertions raise LeakageContaminationError on leakage."""
    valid_train = [
        TextToSQLExample(
            id="tr_01",
            question="Find users",
            db_id="train_db",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
    ]
    # Clean train should pass without exception
    IsolationGuard.assert_training_isolation(valid_train)
    IsolationGuard.assert_retrieval_isolation(valid_train)

    # Dev example mistakenly in training must raise error
    leaked_dev = [
        TextToSQLExample(
            id="dev_01",
            question="Find users",
            db_id="eval_db",
            dataset_name="spider",
            split=DatasetSplit.DEV,
        )
    ]
    with pytest.raises(LeakageContaminationError, match="CRITICAL DATA LEAKAGE"):
        IsolationGuard.assert_training_isolation(leaked_dev)

    with pytest.raises(LeakageContaminationError, match="RETRIEVAL INDEX CONTAMINATION"):
        IsolationGuard.assert_retrieval_isolation(leaked_dev)

    # Quarantined database reference in training must raise error
    with pytest.raises(LeakageContaminationError, match="CRITICAL SCHEMA LEAKAGE"):
        IsolationGuard.assert_training_isolation(valid_train, forbidden_db_ids={"train_db"})
