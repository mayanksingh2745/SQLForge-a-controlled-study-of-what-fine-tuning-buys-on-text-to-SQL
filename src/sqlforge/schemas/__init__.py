"""Typed foundational data contracts for SQLForge."""

from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
from sqlforge.schemas.examples import (
    DatasetSplit,
    DifficultyLevel,
    SyntheticProvenance,
    TextToSQLExample,
)
from sqlforge.schemas.execution import ExecutionResult, ExecutionStatus
from sqlforge.schemas.experiments import ExperimentConfig, RunMetadata, TrainingParams
from sqlforge.schemas.metadata import (
    ColumnMetadata,
    ForeignKeyMetadata,
    SchemaMetadata,
    TableMetadata,
)
from sqlforge.schemas.models import GenerationDefaults, ModelConfig, ModelType

__all__ = [
    "ColumnMetadata",
    "DatasetSplit",
    "DifficultyLevel",
    "EvaluationMetrics",
    "ExecutionResult",
    "ExecutionStatus",
    "ExperimentConfig",
    "ForeignKeyMetadata",
    "GenerationDefaults",
    "GenerationResult",
    "ModelConfig",
    "ModelType",
    "RunMetadata",
    "SchemaMetadata",
    "SyntheticProvenance",
    "TableMetadata",
    "TextToSQLExample",
    "TrainingParams",
]
