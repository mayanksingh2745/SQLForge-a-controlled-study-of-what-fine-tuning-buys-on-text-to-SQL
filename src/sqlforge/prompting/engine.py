"""Prompt assembly engine for zero-shot and few-shot Text-to-SQL tasks.

Integrates:
1. Schema serialization (DDL, compact pipe, JSON).
2. Demonstration retrieval (k in {0, 1, 3, 5}).
3. Domain evidence grounding (BIRD external knowledge).
4. Graceful context budget enforcement with auditable truncation logging.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.prompting.retriever import BM25Retriever, DemonstrationRecord
from sqlforge.prompting.serializers import DDLSerializer, SchemaSerializer, get_serializer
from sqlforge.schemas.examples import TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata

SUPPORTED_FEW_SHOT_K: frozenset[int] = frozenset({0, 1, 3, 5})


class PromptBudgetExceededError(ValueError):
    """Raised when minimum required prompt components exceed hard context budget."""


class AssembledPrompt(BaseModel):
    """Complete, auditable snapshot of an assembled Text-to-SQL inference prompt."""

    model_config = ConfigDict(frozen=True)

    prompt_text: str = Field(..., description="Deterministic complete prompt string")
    example_id: str = Field(..., description="ID of the evaluation target example")
    db_id: str = Field(..., description="Database ID referenced by the prompt")
    k_shots: int = Field(..., description="Number of few-shot demonstrations attached")
    schema_format: str = Field(
        ..., description="Schema serialization format ('ddl', 'compact', 'json')"
    )
    retrieved_demonstration_ids: list[str] = Field(
        default_factory=list, description="IDs of demonstrations retrieved by BM25"
    )
    evidence_included: bool = Field(
        default=False, description="Whether BIRD external evidence was attached"
    )
    token_count_estimate: int = Field(
        ..., description="Deterministic heuristic token count estimate"
    )
    max_tokens_budget: int | None = Field(default=None, description="Configured token budget limit")
    budget_exceeded: bool = Field(
        default=False, description="True if prompt exceeded target token budget"
    )
    truncation_applied: bool = Field(
        default=False, description="True if any truncation degradation was performed"
    )
    truncation_log: list[str] = Field(
        default_factory=list, description="Auditable record of truncation actions"
    )


def estimate_token_count(text: str) -> int:
    """Deterministic token estimation heuristic.

    Uses character length divided by 3.8 rounded up, bounded by word count.
    Provides stable, non-stochastic approximations across environments.
    """
    if not text:
        return 0
    char_estimate = math.ceil(len(text) / 3.8)
    # Ensure token estimate is at least word count
    word_count = len(text.split())
    return max(char_estimate, word_count)


class PromptEngine:
    """Configurable prompt constructor integrating schema serialization and RAG demonstrations."""

    def __init__(
        self,
        schema_format: str = "ddl",
        serializer: SchemaSerializer | None = None,
        retriever: BM25Retriever | None = None,
        default_max_tokens: int | None = 4096,
        system_instruction: str | None = None,
        **serializer_kwargs: Any,
    ) -> None:
        """Initialize PromptEngine.

        Args:
            schema_format: Format name ('ddl', 'compact', or 'json').
            serializer: Explicit serializer instance, or instantiated via get_serializer.
            retriever: BM25Retriever instance for few-shot demonstration retrieval.
            default_max_tokens: Context window budget limit.
            system_instruction: Custom instruction override.
            **serializer_kwargs: Options passed to serializer constructor if serializer is None.
        """
        self.schema_format = schema_format.lower().strip()
        self.serializer = serializer or get_serializer(self.schema_format, **serializer_kwargs)
        self.retriever = retriever
        self.default_max_tokens = default_max_tokens
        self.system_instruction = (
            system_instruction
            or "Given the database schema below, write a syntactically valid SQLite query that accurately answers the question."
        )

    def assemble(
        self,
        example: TextToSQLExample,
        schema: SchemaMetadata,
        k_shots: int = 0,
        max_tokens: int | None = None,
        same_db_only: bool = False,
    ) -> AssembledPrompt:
        """Assemble a prompt for a TextToSQLExample and SchemaMetadata.

        Args:
            example: Evaluation or inference query example.
            schema: Schema metadata corresponding to example.db_id.
            k_shots: Number of demonstrations (0, 1, 3, or 5).
            max_tokens: Optional token budget overriding default_max_tokens.
            same_db_only: If True, restricts demonstration retrieval to the same db_id.

        Returns:
            AssembledPrompt model containing prompt text and audit metadata.

        Raises:
            ValueError: If k_shots is negative, or if k_shots > 0 but no retriever is configured.
            PromptBudgetExceededError: If core schema + question exceeds budget even with k=0.
        """
        if k_shots < 0:
            raise ValueError(f"k_shots must be >= 0, got {k_shots}")

        budget = max_tokens if max_tokens is not None else self.default_max_tokens
        truncation_log: list[str] = []
        truncation_applied = False

        # Retrieve demonstrations if requested
        demonstrations: list[DemonstrationRecord] = []
        if k_shots > 0:
            if self.retriever is None:
                raise ValueError(
                    f"Retriever is required when k_shots > 0 (requested k_shots={k_shots})."
                )
            demonstrations = self.retriever.retrieve_demonstrations(
                query=example.question,
                k=k_shots,
                exclude_example_ids={example.id},
                same_db_only=same_db_only,
                target_db_id=schema.db_id,
            )

        # Build initial prompt candidate
        serialized_schema = self.serializer.serialize(schema)

        prompt_text = self._format_prompt(
            instruction=self.system_instruction,
            db_id=schema.db_id,
            serialized_schema=serialized_schema,
            evidence=example.evidence,
            demonstrations=demonstrations,
            question=example.question,
        )

        tokens = estimate_token_count(prompt_text)

        # Graceful budget enforcement if budget is specified and exceeded
        if budget is not None and tokens > budget:
            truncation_applied = True

            # Degradation Phase 1: Prune demonstrations from lowest score to highest
            while demonstrations and tokens > budget:
                dropped = demonstrations.pop()
                truncation_log.append(
                    f"Dropped demonstration '{dropped.example_id}' (score={dropped.score}) to satisfy token budget."
                )
                prompt_text = self._format_prompt(
                    instruction=self.system_instruction,
                    db_id=schema.db_id,
                    serialized_schema=serialized_schema,
                    evidence=example.evidence,
                    demonstrations=demonstrations,
                    question=example.question,
                )
                tokens = estimate_token_count(prompt_text)

            # Degradation Phase 2: If still exceeding budget, try removing sample rows and comments from schema
            if (
                tokens > budget
                and isinstance(self.serializer, DDLSerializer)
                and (self.serializer.include_sample_rows > 0 or self.serializer.include_comments)
            ):
                truncation_log.append(
                    "Pruned DDL sample rows and comments to reduce schema footprint."
                )
                compact_ddl_serializer = DDLSerializer(
                    include_comments=False,
                    include_sample_rows=0,
                    sort_tables=self.serializer.sort_tables,
                )
                serialized_schema = compact_ddl_serializer.serialize(schema)
                prompt_text = self._format_prompt(
                    instruction=self.system_instruction,
                    db_id=schema.db_id,
                    serialized_schema=serialized_schema,
                    evidence=example.evidence,
                    demonstrations=demonstrations,
                    question=example.question,
                )
                tokens = estimate_token_count(prompt_text)

            # If core schema + question still exceeds hard budget, raise explicit error
            if tokens > budget:
                truncation_log.append(
                    f"CRITICAL: Core schema and question ({tokens} tokens) exceeds context budget ({budget} tokens)."
                )
                raise PromptBudgetExceededError(
                    f"Cannot assemble prompt for '{example.id}': core schema ({schema.db_id}) and question "
                    f"require {tokens} tokens, exceeding budget limit of {budget}."
                )

        retrieved_ids = [d.example_id for d in demonstrations]

        return AssembledPrompt(
            prompt_text=prompt_text,
            example_id=example.id,
            db_id=schema.db_id,
            k_shots=len(demonstrations),
            schema_format=self.schema_format,
            retrieved_demonstration_ids=retrieved_ids,
            evidence_included=bool(example.evidence and example.evidence.strip()),
            token_count_estimate=tokens,
            max_tokens_budget=budget,
            budget_exceeded=bool(budget is not None and tokens > budget),
            truncation_applied=truncation_applied,
            truncation_log=truncation_log,
        )

    @staticmethod
    def _format_prompt(
        instruction: str,
        db_id: str,
        serialized_schema: str,
        evidence: str | None,
        demonstrations: list[DemonstrationRecord],
        question: str,
    ) -> str:
        """Compose structured prompt sections deterministically."""
        sections: list[str] = [
            "### Instructions:",
            instruction.strip(),
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

        if demonstrations:
            demo_lines: list[str] = ["### Examples:"]
            for idx, demo in enumerate(demonstrations, 1):
                demo_lines.append(f"Example {idx}:")
                demo_lines.append(f"Question: {demo.question.strip()}")
                if demo.evidence and demo.evidence.strip():
                    demo_lines.append(f"Evidence: {demo.evidence.strip()}")
                demo_lines.append(f"SQL: {demo.gold_sql.strip()}\n")
            sections.extend(["\n".join(demo_lines).strip(), ""])

        sections.extend(
            [
                "### Question:",
                question.strip(),
                "",
                "### SQL Query:",
                "SELECT ",
            ]
        )

        return "\n".join(sections)
