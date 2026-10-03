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
from sqlforge.training.evaluation_handoff import (
    create_evaluation_config,
    prepare_evaluation_runner,
    validate_adapter_checkpoint,
)
from sqlforge.training.preflight import (
    PreflightChecker,
    PreflightCheckItem,
    PreflightReport,
    PreflightStatus,
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
    "DisallowedSplitError",
    "FineTuningEngine",
    "FineTuningMethod",
    "MalformedExampleError",
    "MissingSchemaError",
    "PEFTConfigFactory",
    "PreflightCheckItem",
    "PreflightChecker",
    "PreflightReport",
    "PreflightSafetyError",
    "PreflightStatus",
    "SFTDatasetFormatter",
    "SFTExample",
    "SFTFineTuningPipeline",
    "SFTTrainResult",
    "SFTTrainingConfig",
    "TargetTruncationError",
    "TokenizedSFTExample",
    "ToyDeterministicTokenizer",
    "TrainingDataSummary",
    "UnsupportedQuantizationPlatformError",
    "create_evaluation_config",
    "prepare_evaluation_runner",
    "validate_adapter_checkpoint",
]
