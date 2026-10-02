"""Unit tests for deterministic prompt builder."""

import pytest

from sqlforge.data.fixtures import FixtureExample
from sqlforge.prompting.builder import PromptBuilder


def test_prompt_builder_deterministic_output() -> None:
    """Verify prompt builder output contains all required sections and is deterministic."""
    example = FixtureExample(
        example_id="test_01",
        question="Find all singer names.",
        db_id="concert_singer",
        schema_context="CREATE TABLE singer (Singer_ID int, Name text);",
        gold_sql="SELECT Name FROM singer;",
        difficulty="easy",
    )
    builder = PromptBuilder()
    prompt1 = builder.build_prompt(example)
    prompt2 = builder.build_prompt(example)

    assert prompt1 == prompt2
    assert "concert_singer" in prompt1
    assert "CREATE TABLE singer" in prompt1
    assert "Find all singer names." in prompt1
    assert prompt1.endswith("SELECT ")


def test_prompt_builder_empty_fields_raise() -> None:
    """Verify prompt builder rejects empty inputs."""
    builder = PromptBuilder()

    # Empty question
    with pytest.raises(ValueError, match="question is empty"):
        builder.build_prompt(
            FixtureExample(
                example_id="bad_q",
                question="   ",
                db_id="db1",
                schema_context="CREATE TABLE t (a int);",
                gold_sql="SELECT 1;",
            )
        )

    # Empty schema
    with pytest.raises(ValueError, match="schema_context is empty"):
        builder.build_prompt(
            FixtureExample(
                example_id="bad_s",
                question="Valid question?",
                db_id="db1",
                schema_context="   ",
                gold_sql="SELECT 1;",
            )
        )
