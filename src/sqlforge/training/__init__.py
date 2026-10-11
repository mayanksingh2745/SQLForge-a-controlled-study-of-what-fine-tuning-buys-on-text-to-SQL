"""Training layer: SFT trainers, LoRA/QLoRA configuration, and checkpointing.

Provides supervised fine-tuning infrastructure for comparing LoRA and QLoRA
with pre-flight feasibility gates, loss masking, and safe checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sqlforge.schemas.experiments import ExperimentConfig
from sqlforge.training.adapters import (
    AdapterConfigSummary,
    PEFTConfigFactory,
    UnsupportedQuantizationPlatformError,
)
from sqlforge.training.checkpoints import CheckpointManager, CheckpointMetadata
from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig
from sqlforge.training.data import (
    DisallowedSplitError,
    MalformedExampleError,
    MissingSchemaError,
    SFTDatasetFormatter,
    SFTExample,
    TrainingDataSummary,
)
from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP05_SEEDS,
    EXP05_TRAINING_SIZES,
    EXP06_EXPERIMENT_ID,
    EXP06_GROUPS,
    EXP06_SEEDS,
    DataScalingSamplingError,
    NoEmpiricalScalingDataError,
    RankSelectionReference,
    ScalingRunCondition,
    SyntheticDataValidationError,
    UnresolvedRankSelectionError,
    generate_exp05_run_conditions,
    generate_exp06_run_conditions,
)
from sqlforge.training.evaluation_handoff import (
    create_evaluation_config,
    prepare_evaluation_runner,
    validate_adapter_checkpoint,
)
from sqlforge.training.plotting import (
    NoEmpiricalDataError,
    ParetoPoint,
    RankAggregateMetric,
    RankSaturationPlotter,
)
from sqlforge.training.preflight import (
    PreflightChecker,
    PreflightCheckItem,
    PreflightReport,
    PreflightStatus,
)
from sqlforge.training.scaling_analysis import (
    DataScalingAnalyzer,
    EXP05AnalysisResult,
    EXP06AnalysisResult,
    GroupComparisonMetric,
    ScalingFitResult,
    SizeScalingMetric,
)
from sqlforge.training.scaling_orchestrator import (
    DataScalingExperimentOrchestrator,
    ScalingExperimentSummary,
    ScalingRunPlan,
    ScalingRunResult,
)
from sqlforge.training.scaling_sampler import (
    NestedStratifiedSpiderSampler,
    SamplingManifest,
)
from sqlforge.training.sweeps import (
    LoRARankSweepOrchestrator,
    RankSweepConfig,
    SweepRunPlan,
    SweepRunResult,
    SweepSummary,
)
from sqlforge.training.synthetic_validation import (
    RecordValidationDetail,
    SyntheticDataValidator,
    SyntheticRecord,
    SyntheticValidationReport,
)
from sqlforge.training.tokenization import (
    CompletionLossMasker,
    TargetTruncationError,
    TokenizedSFTExample,
    ToyDeterministicTokenizer,
)
from sqlforge.training.trainer import (
    PreflightSafetyError,
    SFTFineTuningPipeline,
    SFTTrainResult,
)


class FineTuningEngine(Protocol):
    """Protocol for executing model training runs."""

    def train(self, config: ExperimentConfig) -> Path:
        """Execute fine-tuning and return checkpoint directory path."""
        ...


__all__ = [
    "AdapterConfigSummary",
    "CheckpointManager",
    "CheckpointMetadata",
    "CompletionLossMasker",
    "DataScalingAnalyzer",
    "DataScalingExperimentOrchestrator",
    "DataScalingSamplingError",
    "DisallowedSplitError",
    "EXP05AnalysisResult",
    "EXP05_EXPERIMENT_ID",
    "EXP05_SEEDS",
    "EXP05_TRAINING_SIZES",
    "EXP06AnalysisResult",
    "EXP06_EXPERIMENT_ID",
    "EXP06_GROUPS",
    "EXP06_SEEDS",
    "FineTuningEngine",
    "FineTuningMethod",
    "GroupComparisonMetric",
    "LoRARankSweepOrchestrator",
    "MalformedExampleError",
    "MissingSchemaError",
    "NestedStratifiedSpiderSampler",
    "NoEmpiricalDataError",
    "NoEmpiricalScalingDataError",
    "PEFTConfigFactory",
    "ParetoPoint",
    "PreflightCheckItem",
    "PreflightChecker",
    "PreflightReport",
    "PreflightSafetyError",
    "PreflightStatus",
    "RankAggregateMetric",
    "RankSaturationPlotter",
    "RankSelectionReference",
    "RankSweepConfig",
    "RecordValidationDetail",
    "SFTDatasetFormatter",
    "SFTExample",
    "SFTFineTuningPipeline",
    "SFTTrainResult",
    "SFTTrainingConfig",
    "SamplingManifest",
    "ScalingExperimentSummary",
    "ScalingFitResult",
    "ScalingRunCondition",
    "ScalingRunPlan",
    "ScalingRunResult",
    "SizeScalingMetric",
    "SweepRunPlan",
    "SweepRunResult",
    "SweepSummary",
    "SyntheticDataValidationError",
    "SyntheticDataValidator",
    "SyntheticRecord",
    "SyntheticValidationReport",
    "TargetTruncationError",
    "TokenizedSFTExample",
    "ToyDeterministicTokenizer",
    "TrainingDataSummary",
    "UnresolvedRankSelectionError",
    "UnsupportedQuantizationPlatformError",
    "create_evaluation_config",
    "generate_exp05_run_conditions",
    "generate_exp06_run_conditions",
    "prepare_evaluation_runner",
    "validate_adapter_checkpoint",
]
