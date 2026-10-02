"""Unit tests for training-only BM25 demonstration retriever and quarantine isolation."""

from __future__ import annotations

import pytest

from sqlforge.data.audit import LeakageContaminationError
from sqlforge.prompting.retriever import BM25Retriever, tokenize_text
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample


@pytest.fixture
def clean_train_examples() -> list[TextToSQLExample]:
    """Sample valid training examples."""
    return [
        TextToSQLExample(
            id="train_001",
            question="What are the names of all singers in France?",
            db_id="concert_singer",
            gold_sql="SELECT Name FROM singer WHERE Country = 'France';",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        ),
        TextToSQLExample(
            id="train_002",
            question="Find the maximum capacity of all stadiums in Germany.",
            db_id="stadium",
            gold_sql="SELECT max(Capacity) FROM stadium WHERE Country = 'Germany';",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        ),
        TextToSQLExample(
            id="train_003",
            question="List songs sung by french singers.",
            db_id="concert_singer",
            gold_sql="SELECT Song_Name FROM singer JOIN song ON singer.Id = song.Singer_Id;",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        ),
        TextToSQLExample(
            id="train_004",
            question="How many concerts were held in the year 2020?",
            db_id="concert_singer",
            gold_sql="SELECT count(*) FROM concert WHERE Year = 2020;",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        ),
    ]


def test_tokenize_text() -> None:
    """Test word tokenization strips punctuation and lowercases."""
    tokens = tokenize_text("Find ALL singers from France (and Paris)!")
    assert tokens == ["find", "all", "singers", "from", "france", "and", "paris"]


def test_bm25_retrieval_relevance(clean_train_examples: list[TextToSQLExample]) -> None:
    """Test BM25 retrieves semantically relevant training examples by query."""
    retriever = BM25Retriever(clean_train_examples)
    assert retriever.corpus_size == 4

    # Query for singers
    results = retriever.retrieve("Singers from France", k=2)
    assert len(results) == 2
    assert results[0].id == "train_001"
    assert "singers" in results[0].question.lower()

    # Query for stadiums
    results_stadium = retriever.retrieve("stadium capacity", k=1)
    assert len(results_stadium) == 1
    assert results_stadium[0].id == "train_002"


def test_bm25_retrieval_records_and_scores(clean_train_examples: list[TextToSQLExample]) -> None:
    """Test DemonstrationRecord outputs contain traceable IDs and descending scores."""
    retriever = BM25Retriever(clean_train_examples)
    records = retriever.retrieve_demonstrations("concerts held in year", k=3)

    assert len(records) == 3
    assert records[0].example_id == "train_004"
    assert records[0].score > records[1].score
    # Verify scores are positive
    assert records[0].score > 0.0


def test_bm25_exclude_example_ids(clean_train_examples: list[TextToSQLExample]) -> None:
    """Test that exclude_example_ids prevents self-retrieval."""
    retriever = BM25Retriever(clean_train_examples)

    # Without exclusion, train_001 is top match
    top = retriever.retrieve("names of all singers in France", k=1)
    assert top[0].id == "train_001"

    # With exclusion, train_001 is omitted
    top_excluded = retriever.retrieve(
        "names of all singers in France", k=1, exclude_example_ids={"train_001"}
    )
    assert top_excluded[0].id != "train_001"


def test_bm25_edge_cases(clean_train_examples: list[TextToSQLExample]) -> None:
    """Test edge cases: k=0, empty corpus, empty query."""
    # k <= 0
    retriever = BM25Retriever(clean_train_examples)
    assert retriever.retrieve("singers", k=0) == []

    # Empty corpus
    empty_retriever = BM25Retriever([])
    assert empty_retriever.corpus_size == 0
    assert empty_retriever.retrieve("singers", k=3) == []

    # Empty query fallback
    fallback = retriever.retrieve("", k=2)
    assert len(fallback) == 2


def test_bm25_isolation_guard_rejects_eval_splits(
    clean_train_examples: list[TextToSQLExample],
) -> None:
    """Test that indexing DEV or TEST examples triggers LeakageContaminationError."""
    contaminated_pool = list(clean_train_examples)
    contaminated_pool.append(
        TextToSQLExample(
            id="dev_leak_001",
            question="What is the average capacity?",
            db_id="stadium",
            gold_sql="SELECT avg(Capacity) FROM stadium;",
            dataset_name="spider",
            split=DatasetSplit.DEV,  # Forbidden split
        )
    )

    with pytest.raises(LeakageContaminationError, match="CRITICAL DATA LEAKAGE"):
        BM25Retriever(contaminated_pool)


def test_bm25_isolation_guard_rejects_held_out(
    clean_train_examples: list[TextToSQLExample],
) -> None:
    """Test that indexing HELD_OUT instances triggers LeakageContaminationError."""
    contaminated_pool = list(clean_train_examples)
    contaminated_pool.append(
        TextToSQLExample(
            id="held_out_leak",
            question="Total revenue this month",
            db_id="subscription_analytics_db",
            gold_sql="SELECT sum(amount_paid_usd) FROM transactions;",
            dataset_name="custom_held_out",
            split=DatasetSplit.HELD_OUT,
        )
    )

    with pytest.raises(LeakageContaminationError, match="CRITICAL DATA LEAKAGE"):
        BM25Retriever(contaminated_pool)


def test_bm25_isolation_guard_rejects_quarantined_dbs(
    clean_train_examples: list[TextToSQLExample],
) -> None:
    """Test that candidate training instances referencing quarantined evaluation DBs are rejected."""
    with pytest.raises(LeakageContaminationError, match="CRITICAL SCHEMA LEAKAGE"):
        BM25Retriever(
            clean_train_examples,
            forbidden_db_ids=["concert_singer"],
        )
