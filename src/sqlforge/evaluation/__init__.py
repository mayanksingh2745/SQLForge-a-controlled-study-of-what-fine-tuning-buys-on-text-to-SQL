"""Evaluation layer: Execution accuracy, exact match, statistical confidence intervals.

Provides:
- Result set set-equivalence comparator (`ExecutionComparator`, `compare_result_sets`).
- SQLite read-only connection query executor (`execute_sqlite_query`).
- Non-parametric bootstrap resampling for 95% confidence intervals (`compute_bootstrap_ci`).
- Paired difference bootstrap CI (`compute_paired_difference_ci`).
- McNemar's test with continuity correction (`mcnemar_test`).
"""

from typing import Protocol

from sqlforge.evaluation.comparator import (
    ExecutionComparator,
    compare_result_sets,
    execute_sqlite_query,
)
from sqlforge.evaluation.statistics import (
    compute_bootstrap_ci,
    compute_paired_difference_ci,
    mcnemar_test,
)
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
from sqlforge.schemas.examples import TextToSQLExample


class Evaluator(Protocol):
    """Protocol for scoring model generations against gold standards."""

    def evaluate(
        self,
        generations: list[GenerationResult],
        gold_examples: list[TextToSQLExample],
    ) -> EvaluationMetrics:
        """Calculate execution accuracy, exact match, valid-SQL rate, and error breakdown."""
        ...


__all__ = [
    "Evaluator",
    "ExecutionComparator",
    "compare_result_sets",
    "execute_sqlite_query",
    "compute_bootstrap_ci",
    "compute_paired_difference_ci",
    "mcnemar_test",
]
