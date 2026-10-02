"""Unit tests for SQL execution sandboxing and result set comparator."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sqlforge.evaluation.comparator import (
    ExecutionComparator,
    compare_result_sets,
    execute_sqlite_query,
)
from sqlforge.schemas.evaluation import GenerationResult
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample


@pytest.fixture
def temp_sqlite_db(tmp_path: Path) -> Path:
    """Create a temporary test SQLite database."""
    db_file = tmp_path / "test_store.sqlite"
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
        cursor.execute("INSERT INTO products VALUES (1, 'Widget', 19.99);")
        cursor.execute("INSERT INTO products VALUES (2, 'Gadget', 29.50002);")
        cursor.execute("INSERT INTO products VALUES (3, 'Doohickey', NULL);")
        conn.commit()
    return db_file


class TestExecuteSQLiteQuery:
    """Tests for safe read-only execution."""

    def test_read_only_select_success(self, temp_sqlite_db: Path) -> None:
        status, rows, err = execute_sqlite_query(
            temp_sqlite_db, "SELECT name, price FROM products ORDER BY id ASC;"
        )
        assert status == "SUCCESS"
        assert rows is not None
        assert len(rows) == 3
        assert rows[0] == ("Widget", 19.99)
        assert err is None

    def test_write_attempt_blocked_by_read_only_mode(self, temp_sqlite_db: Path) -> None:
        # SQLite opened with mode=ro rejects any INSERT / UPDATE / DELETE statement
        status, rows, err = execute_sqlite_query(
            temp_sqlite_db, "INSERT INTO products VALUES (4, 'Forbidden', 9.99);"
        )
        assert status in ("EXECUTION_ERROR", "SYNTAX_ERROR")
        assert err is not None
        assert "attempt to write a readonly database" in err.lower() or "readonly" in err.lower()

    def test_empty_sql_returns_syntax_error(self, temp_sqlite_db: Path) -> None:
        status, rows, err = execute_sqlite_query(temp_sqlite_db, "   ")
        assert status == "SYNTAX_ERROR"
        assert rows is None

    def test_nonexistent_db_returns_execution_error(self, tmp_path: Path) -> None:
        status, rows, err = execute_sqlite_query(tmp_path / "missing.db", "SELECT 1;")
        assert status == "EXECUTION_ERROR"
        assert rows is None


class TestCompareResultSets:
    """Tests for formal equivalence comparison rules."""

    def test_identical_single_row(self) -> None:
        gold = [("Alice", 100)]
        pred = [("Alice", 100)]
        match, err = compare_result_sets(pred, gold, gold_sql="SELECT name, score FROM students;")
        assert match is True
        assert err is None

    def test_multiset_unordered_matching_when_no_order_by(self) -> None:
        gold = [("Alice", 100), ("Bob", 90)]
        pred = [("Bob", 90), ("Alice", 100)]
        match, err = compare_result_sets(
            pred, gold, gold_sql="SELECT name, score FROM students WHERE score > 80;"
        )
        assert match is True
        assert err is None

    def test_strict_order_by_sequence_sensitivity(self) -> None:
        gold = [("Alice", 100), ("Bob", 90)]
        pred = [("Bob", 90), ("Alice", 100)]
        # When Gold SQL specifies ORDER BY, row order matters!
        match, err = compare_result_sets(
            pred, gold, gold_sql="SELECT name, score FROM students ORDER BY score DESC;"
        )
        assert match is False
        assert err == "ORDER_BY_MISMATCH"

    def test_floating_point_epsilon_tolerance(self) -> None:
        gold = [(29.50000,)]
        # Within 1e-4 tolerance
        pred = [(29.50002,)]
        match, _ = compare_result_sets(pred, gold, gold_sql="SELECT price FROM items;")
        assert match is True

        # Outside 1e-4 tolerance
        pred_bad = [(29.502,)]
        match_bad, _ = compare_result_sets(pred_bad, gold, gold_sql="SELECT price FROM items;")
        assert match_bad is False

    def test_null_value_strict_handling(self) -> None:
        gold = [(None,)]
        pred_matching = [(None,)]
        pred_zero = [(0,)]
        pred_empty_str = [("",)]

        assert compare_result_sets(pred_matching, gold, gold_sql="SELECT x;")[0] is True
        assert compare_result_sets(pred_zero, gold, gold_sql="SELECT x;")[0] is False
        assert compare_result_sets(pred_empty_str, gold, gold_sql="SELECT x;")[0] is False

    def test_ambiguous_empty_result_sets(self) -> None:
        # Both queries return 0 rows
        match, err = compare_result_sets([], [], gold_sql="SELECT name FROM users WHERE id = -1;")
        assert match is True
        assert err == "AMBIGUOUS_EMPTY_SET"


class TestExecutionComparatorDenominatorRules:
    """Tests verifying denominator N accounting and scoring."""

    def test_error_breakdown_and_confidence_interval(self, temp_sqlite_db: Path) -> None:
        comparator = ExecutionComparator(db_root=temp_sqlite_db.parent)

        gold_examples = [
            TextToSQLExample(
                id="q1",
                question="q1",
                db_id="test_store",
                gold_sql="SELECT name FROM products WHERE id = 1;",
                dataset_name="spider",
                split=DatasetSplit.DEV,
            ),
            TextToSQLExample(
                id="q2",
                question="q2",
                db_id="test_store",
                gold_sql="SELECT name FROM products WHERE id = 2;",
                dataset_name="spider",
                split=DatasetSplit.DEV,
            ),
            TextToSQLExample(
                id="q3",
                question="q3",
                db_id="test_store",
                gold_sql="SELECT name FROM products WHERE id = 3;",
                dataset_name="spider",
                split=DatasetSplit.DEV,
            ),
        ]

        generations = [
            # q1: correct
            GenerationResult(
                example_id="q1",
                generated_sql="SELECT name FROM products WHERE id = 1;",
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=15.0,
            ),
            # q2: wrong result set
            GenerationResult(
                example_id="q2",
                generated_sql="SELECT name FROM products WHERE id = 1;",
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=15.0,
            ),
            # q3: syntax error
            GenerationResult(
                example_id="q3",
                generated_sql="SELEC FORM products WHERE;",
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=15.0,
            ),
        ]

        metrics = comparator.evaluate(generations, gold_examples)
        assert metrics.total_examples == 3
        # 1 correct out of 3 = 0.3333...
        assert pytest.approx(metrics.execution_accuracy, 0.01) == 0.3333
        assert pytest.approx(metrics.syntax_valid_rate, 0.01) == 0.6667
        assert pytest.approx(metrics.execution_success_rate, 0.01) == 0.6667
        assert metrics.confidence_interval is not None
        assert 0.0 <= metrics.confidence_interval.lower <= metrics.confidence_interval.upper <= 1.0
        assert metrics.error_breakdown["SYNTAX_ERROR"] == 1
        assert metrics.error_breakdown["EXECUTION_MISMATCH"] == 1
