"""Prompt construction module for text-to-SQL inference.

Provides deterministic prompt assembly independent from model inference.
"""

from __future__ import annotations

from sqlforge.data.fixtures import FixtureExample


class PromptBuilder:
    """Constructs deterministic structured prompts from schema context and natural language questions."""

    def __init__(self, template_version: str = "v1") -> None:
        self.template_version = template_version

    def build_prompt(self, example: FixtureExample) -> str:
        """Assemble a prompt from a fixture example.

        Args:
            example: Validated FixtureExample instance.

        Returns:
            Deterministic multi-line prompt string.

        Raises:
            ValueError: If required fields in example are blank.
        """
        question = example.question.strip()
        schema_context = example.schema_context.strip()
        db_id = example.db_id.strip()

        if not question:
            raise ValueError(f"Cannot build prompt for '{example.example_id}': question is empty.")
        if not schema_context:
            raise ValueError(
                f"Cannot build prompt for '{example.example_id}': schema_context is empty."
            )
        if not db_id:
            raise ValueError(f"Cannot build prompt for '{example.example_id}': db_id is empty.")

        lines = [
            "### Instructions:",
            "Given the SQLite database schema below, write a syntactically valid SQL query that accurately answers the question.",
            "",
            f"### Database Schema ({db_id}):",
            schema_context,
            "",
            "### Natural Language Question:",
            question,
            "",
            "### SQL Query:",
            "SELECT ",
        ]
        return "\n".join(lines)
