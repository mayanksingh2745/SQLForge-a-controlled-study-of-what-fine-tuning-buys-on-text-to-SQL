"""Data layer: Ingestion, schema normalization, and split management.

Planned for future steps:
- Dataset loaders for Spider, BIRD, and custom held-out datasets.
- Schema extraction and canonical metadata graph construction.
- Contamination and n-gram overlap leakage checkers.
"""

from typing import Protocol

from sqlforge.schemas.examples import TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata


class DatasetLoader(Protocol):
    """Protocol for dataset ingest adapters."""

    def load_examples(self, split: str) -> list[TextToSQLExample]:
        """Load normalized examples for a split."""
        ...

    def get_schema(self, db_id: str) -> SchemaMetadata:
        """Retrieve schema metadata for a database ID."""
        ...
