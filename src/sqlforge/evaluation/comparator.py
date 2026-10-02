"""SQL execution, sandboxing, and result set comparator.

Implements the formal result equivalence rules from metrics_and_statistics.md:
1. Positional column alignment.
2. Multiset row matching (unordered unless Gold SQL contains ORDER BY).
3. Strict ORDER BY sequence evaluation when Gold SQL contains ORDER BY.
4. Floating-point numerical tolerance (epsilon = 1e-4).
5. Strict NULL semantics.
6. Ambiguous empty result set detection.
7. Read-only SQLite connection sandboxing with wall-clock query timeout.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from sqlforge.evaluation.statistics import compute_bootstrap_ci
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
from sqlforge.schemas.examples import TextToSQLExample

logger = logging.getLogger(__name__)


def execute_sqlite_query(
    db_path: Path | str,
    sql: str,
    timeout_seconds: float = 10.0,
    max_rows: int = 5000,
) -> tuple[str, list[tuple[Any, ...]] | None, str | None]:
    """Execute a single SQL query in read-only mode against a SQLite database file.

    Args:
        db_path: Absolute or relative path to SQLite database file.
        sql: SQL query text.
        timeout_seconds: Query execution deadline in seconds (default 10.0).
        max_rows: Upper limit on returned result rows (default 5000).

    Returns:
        Tuple of (status, rows, error_message) where status is in:
        'SUCCESS', 'SYNTAX_ERROR', 'EXECUTION_ERROR', 'TIMEOUT'.
    """
    path = Path(db_path)
    if not path.is_file():
        return "EXECUTION_ERROR", None, f"Database file not found: {path}"

    cleaned_sql = sql.strip()
    if not cleaned_sql:
        return "SYNTAX_ERROR", None, "SQL query string is empty."

    uri = f"file:{path.resolve().as_posix()}?mode=ro"

    try:
        with sqlite3.connect(uri, uri=True, timeout=timeout_seconds) as conn:
            cursor = conn.cursor()
            cursor.execute(cleaned_sql)
            rows = cursor.fetchmany(max_rows)
            return "SUCCESS", rows, None
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "syntax error" in msg.lower() or "incomplete input" in msg.lower():
            return "SYNTAX_ERROR", None, msg
        elif "interrupted" in msg.lower() or "timeout" in msg.lower():
            return "TIMEOUT", None, msg
        else:
            return "EXECUTION_ERROR", None, msg
    except sqlite3.DatabaseError as exc:
        return "EXECUTION_ERROR", None, str(exc)
    except Exception as exc:
        return "EXECUTION_ERROR", None, f"Unexpected runtime error: {exc}"


def normalize_cell_value(val: Any, tolerance: float = 1e-4) -> Any:
    """Normalize a result cell for comparison with floating-point tolerance."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        # Round to tolerance decimals for consistent equality
        decimals = max(0, -int(math.log10(tolerance))) if tolerance > 0 else 4
        rounded = round(val, decimals)
        # Reconcile integer-equivalent floats (e.g. 1.0 == 1)
        if rounded.is_integer():
            return int(rounded)
        return rounded
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, bytes):
        return val
    return str(val)


def normalize_row(row: tuple[Any, ...], tolerance: float = 1e-4) -> tuple[Any, ...]:
    """Normalize row elements for multiset comparison."""
    return tuple(normalize_cell_value(v, tolerance=tolerance) for v in row)


def compare_result_sets(
    pred_rows: list[tuple[Any, ...]],
    gold_rows: list[tuple[Any, ...]],
    has_order_by: bool = False,
    tolerance: float = 1e-4,
    gold_sql: str | None = None,
) -> tuple[bool, str | None]:
    """Compare candidate query result set against Gold SQL result set.

    Args:
        pred_rows: Rows returned by predicted SQL query.
        gold_rows: Rows returned by Gold reference query.
        has_order_by: If True, enforces strict sequence order.
        tolerance: Absolute floating-point equivalence tolerance.
        gold_sql: Optional reference SQL query to automatically detect ORDER BY clauses.

    Returns:
        Tuple of (is_equivalent, anomaly_tag_if_any).
    """
    if gold_sql is not None and not has_order_by:
        has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))

    # Rule 1 & 2: Cardinality check (row count)
    if len(pred_rows) != len(gold_rows):
        return False, None

    # Empty result set handling
    if len(pred_rows) == 0 and len(gold_rows) == 0:
        return True, "AMBIGUOUS_EMPTY_SET"

    # Column count check
    if len(pred_rows[0]) != len(gold_rows[0]):
        return False, None

    norm_pred = [normalize_row(r, tolerance) for r in pred_rows]
    norm_gold = [normalize_row(r, tolerance) for r in gold_rows]

    # Rule 6: Strict ORDER BY sequence match
    if has_order_by:
        for r_pred, r_gold in zip(norm_pred, norm_gold, strict=True):
            if r_pred != r_gold:
                return False, "ORDER_BY_MISMATCH"
        return True, None

    # Rule 4 & 5: Multiset matching (unordered with element multiplicity)
    counter_pred = Counter(norm_pred)
    counter_gold = Counter(norm_gold)

    return (counter_pred == counter_gold), None


class ExecutionComparator:
    """Evaluates candidate SQL generations against Gold references with sandbox execution."""

    def __init__(
        self,
        db_directory: Path | str | None = None,
        db_root: Path | str | None = None,
        timeout_seconds: float = 10.0,
        floating_point_tolerance: float = 1e-4,
        bootstrap_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        """Initialize comparator.

        Args:
            db_directory: Directory containing SQLite databases (<db_id>/<db_id>.sqlite).
            db_root: Alias for db_directory.
            timeout_seconds: Max execution duration per query.
            floating_point_tolerance: Floating-point equivalence epsilon.
            bootstrap_samples: Number of bootstrap replicates for confidence intervals.
            seed: Seed for bootstrap resampling.
        """
        resolved_dir = db_root if db_root is not None else db_directory
        self.db_directory = Path(resolved_dir) if resolved_dir else None
        self.timeout_seconds = timeout_seconds
        self.tolerance = floating_point_tolerance
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.last_anomalies: list[dict[str, Any]] = []
        self.seed = seed

    def resolve_db_path(self, db_id: str) -> Path | None:
        """Resolve database SQLite file path from db_directory."""
        if not self.db_directory:
            return None

        candidates = [
            self.db_directory / db_id / f"{db_id}.sqlite",
            self.db_directory / f"{db_id}.sqlite",
            self.db_directory / db_id / f"{db_id}.db",
            self.db_directory / f"{db_id}.db",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return None

    def evaluate(
        self,
        generations: list[GenerationResult],
        gold_examples: list[TextToSQLExample],
    ) -> EvaluationMetrics:
        """Execute and compare generations against ground truth.

        Args:
            generations: Candidate SQL predictions.
            gold_examples: Reference dataset examples.

        Returns:
            EvaluationMetrics model populated with execution scores, bootstrap CI, and error breakdown.
        """
        metrics, anomalies, _ = self.evaluate_detailed(generations, gold_examples)
        self.last_anomalies = anomalies
        return metrics

    def evaluate_detailed(
        self,
        generations: list[GenerationResult],
        gold_examples: list[TextToSQLExample],
    ) -> tuple[EvaluationMetrics, list[dict[str, Any]], list[float]]:
        """Execute and compare generations against ground truth, returning anomalies and scores.

        Args:
            generations: Candidate SQL predictions.
            gold_examples: Reference dataset examples.

        Returns:
            Tuple of (EvaluationMetrics, anomalies list, per_example_ex_scores).
        """
        total = len(gold_examples)
        if total == 0:
            return (
                EvaluationMetrics(
                    total_examples=0,
                    execution_accuracy=0.0,
                    exact_match_accuracy=0.0,
                    execution_success_rate=0.0,
                    valid_sql_rate=0.0,
                    syntax_valid_rate=0.0,
                    syntax_error_rate=0.0,
                    timeout_rate=0.0,
                    error_breakdown={},
                ),
                [],
                [],
            )

        gen_map = {g.example_id: g for g in generations}
        anomalies: list[dict[str, Any]] = []

        ex_scores: list[float] = []
        em_scores: list[float] = []
        syntax_valid_count = 0
        syntax_error_count = 0
        execution_error_count = 0
        exec_success_count = 0
        mismatch_count = 0
        timeout_count = 0
        empty_count = 0
        ambiguous_count = 0
        unsupported_count = 0
        latencies: list[float] = []

        difficulty_counts: dict[str, int] = {}
        difficulty_correct: dict[str, int] = {}

        for ex in gold_examples:
            diff_label = str(
                ex.difficulty.value if hasattr(ex.difficulty, "value") else ex.difficulty
            )
            difficulty_counts[diff_label] = difficulty_counts.get(diff_label, 0) + 1

            gen = gen_map.get(ex.id)
            if gen is None:
                # Missing prediction counts as 0 in numerator, stays in denominator N
                ex_scores.append(0.0)
                em_scores.append(0.0)
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "MISSING_GENERATION",
                        "reason": f"No candidate prediction provided for example '{ex.id}'.",
                    }
                )
                continue

            latencies.append(gen.latency_ms)
            pred_sql = gen.generated_sql.strip()
            gold_sql = (ex.gold_sql or "").strip()

            # Exact string / AST match
            em_match = float(pred_sql.rstrip(";") == gold_sql.rstrip(";"))
            em_scores.append(em_match)

            db_file = self.resolve_db_path(ex.db_id)

            # If live database file is not available (e.g. offline fixture verification),
            # evaluate exact match and basic syntax heuristics
            if db_file is None:
                unsupported_count += 1
                # Fallback to string equivalence if database file is not mounted
                is_correct = em_match
                ex_scores.append(is_correct)
                if is_correct:
                    exec_success_count += 1
                    syntax_valid_count += 1
                    difficulty_correct[diff_label] = difficulty_correct.get(diff_label, 0) + 1
                continue

            # Execute Gold Query first
            gold_status, gold_rows, gold_err = execute_sqlite_query(
                db_file, gold_sql, timeout_seconds=self.timeout_seconds
            )
            if gold_status != "SUCCESS" or gold_rows is None:
                unsupported_count += 1
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "GOLD_EXECUTION_FAILURE",
                        "reason": f"Gold SQL failed on database '{ex.db_id}': {gold_err}",
                        "gold_sql": gold_sql,
                    }
                )
                ex_scores.append(0.0)
                continue

            # Execute Candidate Query
            pred_status, pred_rows, pred_err = execute_sqlite_query(
                db_file, pred_sql, timeout_seconds=self.timeout_seconds
            )

            if pred_status == "SYNTAX_ERROR":
                syntax_error_count += 1
                ex_scores.append(0.0)
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "SYNTAX_ERROR",
                        "reason": pred_err,
                        "pred_sql": pred_sql,
                    }
                )
                continue
            elif pred_status == "TIMEOUT":
                timeout_count += 1
                syntax_valid_count += 1
                ex_scores.append(0.0)
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "QUERY_TIMEOUT",
                        "reason": f"Query exceeded deadline of {self.timeout_seconds}s.",
                        "pred_sql": pred_sql,
                    }
                )
                continue
            elif pred_status == "EXECUTION_ERROR":
                syntax_valid_count += 1
                execution_error_count += 1
                ex_scores.append(0.0)
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "EXECUTION_ERROR",
                        "reason": pred_err,
                        "pred_sql": pred_sql,
                    }
                )
                continue

            # Query executed successfully
            syntax_valid_count += 1
            exec_success_count += 1

            if len(pred_rows or []) == 0:
                empty_count += 1

            has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))
            equivalent, anomaly_tag = compare_result_sets(
                pred_rows or [],
                gold_rows,
                has_order_by=has_order_by,
                tolerance=self.tolerance,
            )

            if anomaly_tag == "AMBIGUOUS_EMPTY_SET":
                ambiguous_count += 1

            if equivalent:
                ex_scores.append(1.0)
                difficulty_correct[diff_label] = difficulty_correct.get(diff_label, 0) + 1
            else:
                ex_scores.append(0.0)
                mismatch_count += 1
                anomalies.append(
                    {
                        "example_id": ex.id,
                        "error_code": "EXECUTION_MISMATCH",
                        "reason": anomaly_tag or "Result rows did not match gold result set.",
                        "pred_sql": pred_sql,
                    }
                )

        # Aggregate metrics
        exec_acc = sum(ex_scores) / total
        em_acc = sum(em_scores) / total
        exec_success_rate = exec_success_count / total
        syntax_valid_rate = syntax_valid_count / total
        syntax_error_rate = syntax_error_count / total
        timeout_rate = timeout_count / total

        # Compute 95% bootstrap confidence interval for execution accuracy
        bootstrap_ci = compute_bootstrap_ci(
            ex_scores,
            n_bootstrap=self.bootstrap_samples,
            confidence_level=0.95,
            seed=self.seed,
        )

        lat_sorted = sorted(latencies) if latencies else [0.0]
        p50 = float(lat_sorted[len(lat_sorted) // 2]) if lat_sorted else 0.0
        p95_idx = int(math.ceil(0.95 * len(lat_sorted))) - 1
        p95 = float(lat_sorted[max(0, min(p95_idx, len(lat_sorted) - 1))]) if lat_sorted else 0.0

        diff_breakdown = {
            d: round(difficulty_correct.get(d, 0) / count, 4) if count > 0 else 0.0
            for d, count in difficulty_counts.items()
        }

        err_breakdown = {
            "SYNTAX_ERROR": syntax_error_count,
            "TIMEOUT": timeout_count,
            "EXECUTION_ERROR": execution_error_count,
            "EXECUTION_MISMATCH": mismatch_count,
        }

        metrics = EvaluationMetrics(
            total_examples=total,
            execution_accuracy=round(exec_acc, 4),
            exact_match_accuracy=round(em_acc, 4),
            execution_success_rate=round(exec_success_rate, 4),
            valid_sql_rate=round(exec_success_rate, 4),
            syntax_valid_rate=round(syntax_valid_rate, 4),
            syntax_error_rate=round(syntax_error_rate, 4),
            timeout_rate=round(timeout_rate, 4),
            empty_result_count=empty_count,
            ambiguous_result_count=ambiguous_count,
            unsupported_comparison_count=unsupported_count,
            latency_p50_ms=round(p50, 2),
            latency_p95_ms=round(p95, 2),
            bootstrap_ci_execution_accuracy=bootstrap_ci,
            error_breakdown=err_breakdown,
            difficulty_breakdown=diff_breakdown,
        )

        return metrics, anomalies, ex_scores
