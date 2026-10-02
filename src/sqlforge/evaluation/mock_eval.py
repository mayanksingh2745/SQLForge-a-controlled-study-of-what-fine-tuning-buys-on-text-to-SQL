"""Offline mock evaluator for pipeline testing and contract validation.

IMPORTANT DISCLAIMER:
This mock evaluator is strictly for software pipeline integration testing.
It DOES NOT execute queries against a live SQLite sandbox, does not establish
SQL correctness, execution accuracy, SQL safety, or benchmark compatibility,
and cannot produce empirical research findings.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlforge.data.fixtures import FixtureExample
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult

logger = logging.getLogger(__name__)


class MockEvaluator:
    """Mock evaluation module for testing pipeline serialization and metrics contracts."""

    def evaluate(
        self,
        generations: list[GenerationResult],
        gold_examples: list[FixtureExample],
    ) -> tuple[EvaluationMetrics, list[dict[str, Any]]]:
        """Evaluate synthetic generations against fixture gold references.

        Args:
            generations: Model inference predictions.
            gold_examples: Reference fixture examples with ground truth.

        Returns:
            Tuple of (EvaluationMetrics, list of anomaly dicts).
        """
        logger.info(
            "Running MOCK evaluation (test-only synthetic evaluator; does NOT establish empirical correctness)."
        )

        gold_map = {ex.example_id: ex for ex in gold_examples}
        total = len(generations)
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
                ),
                [],
            )

        exact_matches = 0
        execution_successes = 0
        syntax_valid = 0
        syntax_errors = 0
        timeouts = 0
        anomalies: list[dict[str, Any]] = []
        difficulty_counts: dict[str, int] = {}
        difficulty_correct: dict[str, int] = {}

        for gen in generations:
            gold = gold_map.get(gen.example_id)
            diff = gold.difficulty if gold else "unclassified"
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            sql = gen.generated_sql

            # Check for simulated syntax error
            if "MOCK_SYNTHETIC_MALFORMED" in sql:
                syntax_errors += 1
                anomalies.append(
                    {
                        "example_id": gen.example_id,
                        "error_code": "MOCK_SYNTAX_ERROR",
                        "reason": "Synthetic malformed SQL syntax",
                        "generated_sql": sql,
                    }
                )
                continue

            # Check for simulated timeout
            if "MOCK_SYNTHETIC_TIMEOUT" in sql or gen.finish_reason == "timeout":
                timeouts += 1
                anomalies.append(
                    {
                        "example_id": gen.example_id,
                        "error_code": "MOCK_TIMEOUT",
                        "reason": "Synthetic execution timeout",
                        "generated_sql": sql,
                    }
                )
                continue

            # Normal synthetic execution path
            syntax_valid += 1
            execution_successes += 1

            # Clean mock prefix to compare with gold query
            clean_sql = sql.replace("/* MOCK_SYNTHETIC */", "").strip()
            if gold and clean_sql.rstrip(";") == gold.gold_sql.strip().rstrip(";"):
                exact_matches += 1
                difficulty_correct[diff] = difficulty_correct.get(diff, 0) + 1

        exec_acc = exact_matches / total
        em_acc = exact_matches / total
        exec_success_rate = execution_successes / total
        syntax_valid_rate = syntax_valid / total
        syntax_error_rate = syntax_errors / total
        timeout_rate = timeouts / total

        breakdown = {
            d: (difficulty_correct.get(d, 0) / count if count > 0 else 0.0)
            for d, count in difficulty_counts.items()
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
            difficulty_breakdown=breakdown,
        )

        return metrics, anomalies
