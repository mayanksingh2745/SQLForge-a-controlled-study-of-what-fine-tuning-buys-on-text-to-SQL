"""Deterministic mock model runner for pipeline integration and verification.

All outputs are visibly synthetic and cannot be mistaken for real research results.
Operates completely offline with zero weight downloads and zero network requests.
"""

from __future__ import annotations

from typing import Any

from sqlforge.schemas.evaluation import GenerationResult


class MockModelError(RuntimeError):
    """Raised when the mock model runner encounters a simulated or operational failure."""


class MockModelRunner:
    """Offline deterministic mock inference engine for integration harnesses."""

    def __init__(
        self,
        model_id: str = "mock-pipeline-model-v0",
        simulate_failure_on_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.simulate_failure_on_id = simulate_failure_on_id

    def generate(
        self,
        prompt: str,
        example_id: str = "mock_example",
        gold_sql: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a deterministic synthetic SQL prediction.

        Args:
            prompt: Text prompt string.
            example_id: Identifier of the current fixture example.
            gold_sql: Optional reference SQL query to derive synthetic success prediction.
            **kwargs: Extra generation parameters (ignored by mock).

        Returns:
            GenerationResult labeled with explicit synthetic markers.

        Raises:
            MockModelError: If simulate_failure_on_id matches example_id.
        """
        if self.simulate_failure_on_id and example_id == self.simulate_failure_on_id:
            raise MockModelError(f"Simulated mock inference failure for example '{example_id}'.")

        # Token count estimates for testing
        prompt_tokens = len(prompt.split())

        # Path 1: Synthetic malformed output trigger
        if "malformed" in example_id.lower() or "France" in prompt:
            generated_sql = "/* MOCK_SYNTHETIC_MALFORMED */ SELECT FROM WHERE (syntax_error);"
            finish_reason = "stop"
        # Path 2: Synthetic timeout output trigger
        elif "timeout" in example_id.lower() or "complex subqueries" in prompt:
            generated_sql = (
                "/* MOCK_SYNTHETIC_TIMEOUT */ SELECT * FROM heavy_table WHERE sleep(100);"
            )
            finish_reason = "timeout"
        # Path 3: Standard synthetic success output
        else:
            if gold_sql:
                # Use gold SQL prefixed with clear mock synthetic disclaimer
                generated_sql = f"/* MOCK_SYNTHETIC */ {gold_sql.strip()}"
            else:
                generated_sql = "/* MOCK_SYNTHETIC */ SELECT 1;"
            finish_reason = "stop"

        completion_tokens = len(generated_sql.split())

        return GenerationResult(
            example_id=example_id,
            generated_sql=generated_sql,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=10.0,
            finish_reason=finish_reason,
        )
