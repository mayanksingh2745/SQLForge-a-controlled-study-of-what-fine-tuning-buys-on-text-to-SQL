"""Typed foundational data contracts for SQLForge."""

from sqlforge.schemas.dataset import (
    ContaminationReport,
    DatasetManifest,
    DatasetProvenance,
    DatasetSplitManifest,
    DuplicateDetail,
    FuzzyOverlapDetail,
    LicenseIdentifier,
    SchemaDisjointnessViolation,
    SplitLeakageViolation,
)
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
    "ContaminationReport",
    "DatasetManifest",
    "DatasetProvenance",
    "DatasetSplit",
    "DatasetSplitManifest",
    "DifficultyLevel",
    "DuplicateDetail",
    "EvaluationMetrics",
    "ExecutionResult",
    "ExecutionStatus",
    "ExperimentConfig",
    "ForeignKeyMetadata",
    "FuzzyOverlapDetail",
    "GenerationDefaults",
    "GenerationResult",
    "LicenseIdentifier",
    "ModelConfig",
    "ModelType",
    "RunMetadata",
    "SchemaDisjointnessViolation",
    "SchemaMetadata",
    "SplitLeakageViolation",
    "SyntheticProvenance",
    "TableMetadata",
    "TextToSQLExample",
    "TrainingParams",
]
