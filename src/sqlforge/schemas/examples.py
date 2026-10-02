"""Data contracts for Text-to-SQL examples, splits, and provenance."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DatasetSplit(StrEnum):
    """Canonical dataset partition splits."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    HELD_OUT = "held_out"


class DifficultyLevel(StrEnum):
    """Difficulty classification tiers (aligning with Spider/BIRD hierarchies)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTRA = "extra"
    UNCLASSIFIED = "unclassified"


class SyntheticProvenance(BaseModel):
    """Audit metadata for synthetically generated training examples."""

    model_config = ConfigDict(frozen=True)

    generator_model: str = Field(..., description="Model ID used to produce the synthetic pair")
    prompt_template_id: str = Field(
        ..., description="Template version identifier used during generation"
    )
    validation_status: str = Field(
        default="pending",
        description="Validation stage ('unverified', 'syntax_checked', 'execution_verified')",
    )
    execution_verified: bool = Field(
        default=False,
        description="Whether query was executed against the database and produced non-empty valid result",
    )
    pass_rate: float | None = Field(
        default=None,
        description="Self-consistency agreement rate across multiple generation samplings",
    )
    date_generated: str = Field(
        ..., description="ISO 8601 timestamp string when example was synthesized"
    )


class TextToSQLExample(BaseModel):
    """Canonical data model for a single text-to-SQL instance."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        ..., description="Stable, unique example identifier (e.g. 'spider_train_00042')"
    )
    question: str = Field(..., description="Natural language question or request")
    db_id: str = Field(..., description="Database/schema identifier to query against")
    gold_sql: str | None = Field(default=None, description="Ground truth reference SQL query")
    dataset_name: str = Field(
        ..., description="Origin dataset name ('spider', 'bird', 'custom_held_out')"
    )
    split: DatasetSplit = Field(..., description="Partition split of the example")
    difficulty: DifficultyLevel | str = Field(
        default=DifficultyLevel.UNCLASSIFIED,
        description="Complexity rating of the query",
    )
    evidence: str | None = Field(
        default=None,
        description="External domain hints, column clarifications, or formulas provided with question",
    )
    synthetic_metadata: SyntheticProvenance | None = Field(
        default=None,
        description="Audit metadata if this example was synthetically produced",
    )
    schema_version: str | None = Field(
        default=None,
        description="Version or checksum of the target schema used during annotation",
    )
