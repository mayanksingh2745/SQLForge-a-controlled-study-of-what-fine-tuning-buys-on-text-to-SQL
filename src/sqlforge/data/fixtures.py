"""Deterministic test fixtures for pipeline harness and offline testing.

Provides typed fixture contracts, strict validation, and deterministic ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class FixtureExample(BaseModel):
    """Minimal typed contract for deterministic test fixture examples."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., min_length=1, description="Unique, non-empty example identifier")
    question: str = Field(..., min_length=1, description="Natural language user question")
    db_id: str = Field(..., min_length=1, description="Database schema identifier")
    schema_context: str = Field(
        ..., min_length=1, description="Schema definition (DDL or column list)"
    )
    gold_sql: str = Field(..., min_length=1, description="Reference ground truth SQL query")
    difficulty: str = Field(default="easy", description="Query difficulty category")


def get_default_fixture_path() -> Path:
    """Return the absolute path to the default version-controlled test fixture dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    return project_root / "tests" / "fixtures" / "dataset" / "mock_spider.json"


def load_fixture_dataset(path: Path | str | None = None) -> list[FixtureExample]:
    """Load and strictly validate a deterministic fixture dataset from JSON.

    Args:
        path: Path to fixture JSON file. If None, uses default mock_spider.json fixture.

    Returns:
        List of validated FixtureExample instances, deterministically sorted by example_id.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
        ValueError: If JSON is malformed, dataset is empty, schema is violated, or duplicate IDs exist.
    """
    target_path = Path(path) if path is not None else get_default_fixture_path()

    if not target_path.exists():
        raise FileNotFoundError(f"Fixture dataset file not found: {target_path}")

    try:
        with open(target_path, encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in fixture dataset '{target_path}': {exc}") from exc

    if not isinstance(raw_data, list):
        raise ValueError(
            f"Invalid fixture dataset format in '{target_path}': root structure must be a list."
        )

    if not raw_data:
        raise ValueError(
            f"Fixture dataset '{target_path}' is empty. At least one example is required."
        )

    examples: list[FixtureExample] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Fixture example at index {idx} must be a dictionary, got {type(item).__name__}."
            )

        try:
            example = FixtureExample(**item)
        except ValidationError as exc:
            raise ValueError(
                f"Validation failed for fixture example at index {idx}: {exc}"
            ) from exc

        if example.example_id in seen_ids:
            raise ValueError(
                f"Duplicate example_id '{example.example_id}' detected at index {idx} in '{target_path}'."
            )
        seen_ids.add(example.example_id)
        examples.append(example)

    # Enforce deterministic ordering by example_id
    examples.sort(key=lambda ex: ex.example_id)
    return examples
