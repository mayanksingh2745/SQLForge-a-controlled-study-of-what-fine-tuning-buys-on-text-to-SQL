"""Prompting layer: Schema serialization, prompt templates, and few-shot retrieval.

Planned for future steps:
- Serializers for DDL, compact schema strings, and markdown table notations.
- Dynamic few-shot demonstration selectors (BM25, embedding semantic search).
- Chat template formatters for instruction-tuned LLMs.
"""

from typing import Protocol

from sqlforge.schemas.metadata import SchemaMetadata


class SchemaSerializer(Protocol):
    """Protocol for converting SchemaMetadata into LLM prompt text."""

    def serialize(self, schema: SchemaMetadata) -> str:
        """Format schema metadata into serialized prompt representation."""
        ...
