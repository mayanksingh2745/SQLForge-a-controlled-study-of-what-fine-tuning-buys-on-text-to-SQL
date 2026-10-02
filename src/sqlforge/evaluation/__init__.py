"""Evaluation layer: Execution accuracy, exact match, statistical confidence intervals.

Planned for future steps:
- Result set set-equivalence comparator.
- Non-parametric bootstrap resampling for 95% confidence intervals.
- Error categorization taxonomy engine.
"""

from typing import Protocol

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
