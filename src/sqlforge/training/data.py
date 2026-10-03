"""Supervised training data preparation, formatting, and split validation.

Converts TextToSQLExample records and SchemaMetadata into structured (prompt, target)
pairs for causal LM supervised fine-tuning with strict train/dev/test split isolation.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.prompting.serializers import SchemaSerializer, get_serializer
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata

logger = logging.getLogger(__name__)


class DisallowedSplitError(ValueError):
    """Raised when an example from a non-train split is passed to training data preparation."""


class MalformedExampleError(ValueError):
    """Raised when a training example lacks required fields (question, gold_sql, db_id)."""


class MissingSchemaError(KeyError):
    """Raised when schema metadata for a required db_id is not provided."""


class SFTExample(BaseModel):
    """Single prepared supervised fine-tuning training pair."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique example identifier")
    db_id: str = Field(..., description="Target database identifier")
    dataset_name: str = Field(..., description="Origin dataset name")
    split: DatasetSplit = Field(..., description="Partition split (must be train)")
    prompt_text: str = Field(
        ..., description="Formatted input prompt containing schema and question"
    )
    target_text: str = Field(..., description="Target completion SQL query")
    gold_sql: str = Field(..., description="Ground truth reference SQL query")
    difficulty: str = Field(default="unclassified", description="Query complexity tier")
    is_synthetic: bool = Field(default=False, description="True if synthetically generated")
    schema_version: str | None = Field(default=None, description="Schema version hash")


class TrainingDataSummary(BaseModel):
    """Audit summary of prepared training dataset."""

    model_config = ConfigDict(frozen=True)

    total_input_examples: int
    valid_examples_count: int
    rejected_count: int
    unique_databases: int
    synthetic_count: int
    avg_prompt_chars: float
    avg_target_chars: float
    schema_format: str


class SFTDatasetFormatter:
    """Formats TextToSQLExample instances into supervised prompt/completion pairs."""

    def __init__(
        self,
        schema_format: str = "ddl",
        serializer: SchemaSerializer | None = None,
        system_instruction: str | None = None,
        append_semicolon: bool = True,
        **serializer_kwargs: Any,
    ) -> None:
        """Initialize SFTDatasetFormatter.

        Args:
            schema_format: Schema serialization format ('ddl', 'compact_pipe', 'json').
            serializer: Optional pre-configured SchemaSerializer.
            system_instruction: Custom instruction text for prompt header.
            append_semicolon: Ensure target SQL ends with a semicolon.
            **serializer_kwargs: Arguments passed to get_serializer if serializer is None.
        """
        self.schema_format = schema_format.lower().strip()
        self.serializer = serializer or get_serializer(self.schema_format, **serializer_kwargs)
        self.system_instruction = (
            system_instruction
            or "Given the database schema below, write a syntactically valid SQLite query that accurately answers the question."
        )
        self.append_semicolon = append_semicolon

    def format_single(
        self,
        example: TextToSQLExample,
        schema: SchemaMetadata,
    ) -> SFTExample:
        """Format an individual TextToSQLExample into an SFTExample.

        Args:
            example: Valid TextToSQLExample instance from train partition.
            schema: SchemaMetadata corresponding to example.db_id.

        Returns:
            SFTExample ready for tokenization.

        Raises:
            DisallowedSplitError: If example split is not TRAIN.
            MalformedExampleError: If question, db_id, or gold_sql is missing/empty.
            MissingSchemaError: If schema db_id does not match example db_id.
        """
        # Strict training-split quarantine enforcement
        if example.split != DatasetSplit.TRAIN:
            raise DisallowedSplitError(
                f"Example '{example.id}' belongs to split '{example.split.value}'. "
                "Training pipeline strictly rejects DEV, TEST, and HELD_OUT splits to prevent data leakage."
            )

        # Field validation
        if not example.question or not example.question.strip():
            raise MalformedExampleError(f"Example '{example.id}' has missing or empty question.")
        if not example.db_id or not example.db_id.strip():
            raise MalformedExampleError(f"Example '{example.id}' has missing or empty db_id.")
        if not example.gold_sql or not example.gold_sql.strip():
            raise MalformedExampleError(
                f"Example '{example.id}' has missing or empty gold_sql required for supervised training."
            )
        if schema.db_id != example.db_id:
            raise MissingSchemaError(
                f"Schema db_id '{schema.db_id}' does not match example db_id '{example.db_id}'."
            )

        # Build prompt
        serialized_schema = self.serializer.serialize(schema)
        prompt_text = self._build_prompt_text(
            db_id=schema.db_id,
            serialized_schema=serialized_schema,
            question=example.question,
            evidence=example.evidence,
        )

        # Build target SQL
        target_sql = example.gold_sql.strip()
        if self.append_semicolon and not target_sql.endswith(";"):
            target_sql = f"{target_sql};"

        is_synthetic = bool(example.synthetic_metadata is not None)

        return SFTExample(
            example_id=example.id,
            db_id=example.db_id,
            dataset_name=example.dataset_name,
            split=example.split,
            prompt_text=prompt_text,
            target_text=target_sql,
            gold_sql=example.gold_sql,
            difficulty=str(example.difficulty),
            is_synthetic=is_synthetic,
            schema_version=example.schema_version,
        )

    def format_dataset(
        self,
        examples: list[TextToSQLExample],
        schemas: dict[str, SchemaMetadata],
        skip_malformed: bool = False,
        max_examples: int | None = None,
    ) -> tuple[list[SFTExample], TrainingDataSummary]:
        """Format a list of examples with schema lookups.

        Args:
            examples: List of TextToSQLExample instances.
            schemas: Mapping from db_id to SchemaMetadata.
            skip_malformed: If True, log and skip malformed examples instead of failing.
            max_examples: Optional cap on total formatted examples.

        Returns:
            Tuple of (formatted SFTExamples, TrainingDataSummary).
        """
        formatted: list[SFTExample] = []
        rejected_count = 0
        seen_dbs: set[str] = set()
        synthetic_count = 0
        total_prompt_chars = 0
        total_target_chars = 0

        target_slice = examples if max_examples is None else examples[:max_examples]

        for ex in target_slice:
            if ex.db_id not in schemas:
                if skip_malformed:
                    logger.warning(
                        f"Skipping example '{ex.id}': schema for db_id '{ex.db_id}' not found."
                    )
                    rejected_count += 1
                    continue
                raise MissingSchemaError(
                    f"Schema metadata for db_id '{ex.db_id}' (example '{ex.id}') not provided in schemas."
                )

            schema = schemas[ex.db_id]

            try:
                sft_ex = self.format_single(ex, schema)
                formatted.append(sft_ex)
                seen_dbs.add(sft_ex.db_id)
                if sft_ex.is_synthetic:
                    synthetic_count += 1
                total_prompt_chars += len(sft_ex.prompt_text)
                total_target_chars += len(sft_ex.target_text)
            except (DisallowedSplitError, MalformedExampleError) as err:
                if skip_malformed:
                    logger.warning(f"Skipping example '{ex.id}': {err}")
                    rejected_count += 1
                else:
                    raise

        valid_count = len(formatted)
        avg_prompt = (total_prompt_chars / valid_count) if valid_count > 0 else 0.0
        avg_target = (total_target_chars / valid_count) if valid_count > 0 else 0.0

        summary = TrainingDataSummary(
            total_input_examples=len(target_slice),
            valid_examples_count=valid_count,
            rejected_count=rejected_count,
            unique_databases=len(seen_dbs),
            synthetic_count=synthetic_count,
            avg_prompt_chars=round(avg_prompt, 1),
            avg_target_chars=round(avg_target, 1),
            schema_format=self.schema_format,
        )

        return formatted, summary

    def _build_prompt_text(
        self,
        db_id: str,
        serialized_schema: str,
        question: str,
        evidence: str | None,
    ) -> str:
        """Compose structured instruction prompt for training."""
        sections = [
            "### Instructions:",
            self.system_instruction.strip(),
            "",
            f"### Database Schema ({db_id}):",
            serialized_schema.strip(),
            "",
        ]

        if evidence and evidence.strip():
            sections.extend(
                [
                    "### Domain Evidence:",
                    evidence.strip(),
                    "",
                ]
            )

        sections.extend(
            [
                "### Question:",
                question.strip(),
                "",
                "### SQL Query:",
                "",
            ]
        )

        return "\n".join(sections)
