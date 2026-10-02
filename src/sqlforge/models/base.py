"""Base protocols and common contracts for model inference engines."""

from __future__ import annotations

from typing import Any, Protocol

from sqlforge.schemas.evaluation import GenerationResult


class ModelRunner(Protocol):
    """Unified interface for model inference runners (local, API, and mock)."""

    @property
    def model_id(self) -> str:
        """Model identifier or repository slug."""
        ...

    def generate(
        self,
        prompt: str,
        example_id: str = "example",
        **kwargs: Any,
    ) -> GenerationResult:
        """Execute deterministic generation for a structured text prompt.

        Args:
            prompt: Formatted prompt text.
            example_id: Associated evaluation example identifier.
            **kwargs: Extra parameters (e.g. stop_tokens, max_new_tokens).

        Returns:
            GenerationResult containing generated SQL, token accounting, and latency.
        """
        ...
