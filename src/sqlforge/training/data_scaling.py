"""Research contracts, configuration schemas, and validation for EXP-05 and EXP-06.

Defines typed specifications for:
- EXP-05: Training data scaling (500, 1000, 2000, 5000, 7000 examples across seeds 42, 43, 44).
- EXP-06: Synthetic vs. human data composition (human_only, synthetic_only, human_synthetic_50_50).
- Rank-selection dependency linking Step 8 back to verified Step 7 (EXP-04) artifacts.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sqlforge.schemas.models import LoRAHyperparameters
from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig

# ==============================================================================
# Canonical Research Constants
# ==============================================================================

EXP05_EXPERIMENT_ID = "EXP-05-DATA-SCALING"
EXP06_EXPERIMENT_ID = "EXP-06-SYNTHETIC-DATA"

EXP05_TRAINING_SIZES: tuple[int, ...] = (500, 1000, 2000, 5000, 7000)
EXP05_SEEDS: tuple[int, ...] = (42, 43, 44)

EXP06_GROUPS: tuple[str, ...] = ("human_only", "synthetic_only", "human_synthetic_50_50")
EXP06_SEEDS: tuple[int, ...] = (42, 43, 44)
EXP06_TOTAL_SIZE = 3000
EXP06_GROUP_COMPOSITION: dict[str, tuple[int, int]] = {
    # (human_count, synthetic_count)
    "human_only": (3000, 0),
    "synthetic_only": (0, 3000),
    "human_synthetic_50_50": (1500, 1500),
}

CANONICAL_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
CANONICAL_MODEL_FAMILY = "qwen"
CANONICAL_MODEL_REVISION = "main"

CANONICAL_OPTIMIZER = "paged_adamw_32bit"
CANONICAL_LR = 0.0002
CANONICAL_LR_SCHEDULER = "cosine"
CANONICAL_WARMUP_RATIO = 0.03
CANONICAL_EPOCHS = 3
CANONICAL_PER_DEVICE_BATCH_SIZE = 8
CANONICAL_GRADIENT_ACCUMULATION = 2  # Effective batch size = 16
CANONICAL_MAX_SEQ_LENGTH = 2048
CANONICAL_SCHEMA_FORMAT = "ddl"
CANONICAL_MASK_PROMPT_LOSS = True

TRAIN_SPLIT_SPIDER = "spider:train"
EVAL_SPLIT_SPIDER = "spider:dev"

ALLOWED_LORA_RANKS: tuple[int, ...] = (8, 16, 32, 64)


# ==============================================================================
# Dedicated Research Exceptions
# ==============================================================================


class UnresolvedRankSelectionError(RuntimeError):
    """Raised when live training is attempted without a verified EXP-04 rank selection artifact."""


class NoEmpiricalScalingDataError(RuntimeError):
    """Raised when scaling law or comparative analysis is attempted without genuine empirical data."""


class DataScalingSamplingError(ValueError):
    """Raised when stratified sampling violates dataset constraints, nesting, or eligibility."""


class SyntheticDataValidationError(ValueError):
    """Raised when synthetic dataset records fail schema, execution, provenance, or isolation audits."""


# ==============================================================================
# Rank Selection Dependency Contract
# ==============================================================================


class RankSelectionReference(BaseModel):
    """Cryptographic and experimental provenance linking data scaling to an EXP-04 rank choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_experiment_id: str = Field(
        default="EXP-04-RANK-SWEEP",
        description="Source experiment ID (must strictly be 'EXP-04-RANK-SWEEP')",
    )
    source_run_ids: list[str] = Field(
        description="List of EXP-04 run IDs evaluated to produce the selection",
    )
    source_manifest_paths: list[str] = Field(
        description="Paths to verified manifest.json files of source runs",
    )
    source_manifest_hashes: list[str] = Field(
        description="Cryptographic SHA-256 hashes of the source run manifests",
    )
    selected_rank: int = Field(
        description="Selected LoRA rank r in {8, 16, 32, 64}",
    )
    selected_alpha: int = Field(
        description="Selected LoRA alpha (must equal 2 * selected_rank)",
    )
    selected_target_modules: list[str] = Field(
        description="Target module projection names (e.g. all-linear)",
    )
    selection_metric: str = Field(
        default="execution_accuracy",
        description="Evaluation metric used for ranking (must be 'execution_accuracy')",
    )
    selection_rule: str = Field(
        default="highest seed-42 spider:dev execution accuracy, lower parameter count breaking ties",
        description="Explicit decision rule used to select optimal rank",
    )
    selection_timestamp: str = Field(
        description="ISO 8601 UTC timestamp of rank selection resolution",
    )
    verification_status: str = Field(
        description="Verification state: 'verified' or 'provisional_non_empirical'",
    )

    @field_validator("source_experiment_id")
    @classmethod
    def validate_source_experiment(cls, v: str) -> str:
        if v != "EXP-04-RANK-SWEEP":
            raise ValueError(
                f"source_experiment_id must strictly be 'EXP-04-RANK-SWEEP', got {v!r}"
            )
        return v

    @field_validator("selected_rank")
    @classmethod
    def validate_selected_rank(cls, v: int) -> int:
        if v not in ALLOWED_LORA_RANKS:
            raise ValueError(f"selected_rank must be one of {ALLOWED_LORA_RANKS}, got {v}")
        return v

    @field_validator("selected_alpha")
    @classmethod
    def validate_selected_alpha(cls, v: int, info: Any) -> int:
        # Cross-checked in model_validator for selected_rank
        return v

    @model_validator(mode="after")
    def validate_alpha_and_status(self) -> RankSelectionReference:
        if self.selected_alpha != self.selected_rank * 2:
            raise ValueError(
                f"selected_alpha must be strictly 2 * selected_rank (expected {self.selected_rank * 2}, got {self.selected_alpha})"
            )
        if self.verification_status not in ("verified", "provisional_non_empirical"):
            raise ValueError(
                f"verification_status must be 'verified' or 'provisional_non_empirical', got {self.verification_status!r}"
            )
        return self

    @property
    def is_empirical_eligible(self) -> bool:
        """Whether this selection reference authorizes genuine live GPU execution."""
        return self.verification_status == "verified"

    @classmethod
    def create_provisional(cls, rank: int = 16) -> RankSelectionReference:
        """Create an explicitly provisional, non-empirical reference for CPU dry runs and planning only."""
        if rank not in ALLOWED_LORA_RANKS:
            raise ValueError(f"Rank must be one of {ALLOWED_LORA_RANKS}, got {rank}")
        return cls(
            source_experiment_id="EXP-04-RANK-SWEEP",
            source_run_ids=[f"exp04_r{rank}_a{rank * 2}_all_linear_s42_provisional"],
            source_manifest_paths=["provisional_non_empirical_placeholder"],
            source_manifest_hashes=["0" * 64],
            selected_rank=rank,
            selected_alpha=rank * 2,
            selected_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            selection_metric="execution_accuracy",
            selection_rule="provisional default (rank 16) solely for zero-compute planning and dry-run validation",
            selection_timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            verification_status="provisional_non_empirical",
        )


# ==============================================================================
# EXP-05 & EXP-06 Condition Specification
# ==============================================================================


class ScalingRunCondition(BaseModel):
    """Specification of an individual run condition in EXP-05 or EXP-06."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    experiment_id: str
    seed: int
    total_examples: int
    human_examples_count: int
    synthetic_examples_count: int
    group_name: str | None = None  # None for EXP-05, group name for EXP-06
    rank_selection: RankSelectionReference = Field(
        default_factory=RankSelectionReference.create_provisional
    )
    sampling_manifest_hash: str | None = None
    source_dataset_manifest_hash: str | None = None
    status: str = "planned"  # "planned", "dry_run", "cached", "completed", "failed"
    is_provisional: bool = False

    def to_sft_training_config(
        self,
        output_dir: Path | str = "artifacts/runs",
    ) -> SFTTrainingConfig:
        """Generate the typed SFTTrainingConfig representing this exact condition."""
        return SFTTrainingConfig(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            description=(
                f"{self.experiment_id} run: total_n={self.total_examples}, "
                f"human_n={self.human_examples_count}, synth_n={self.synthetic_examples_count}, "
                f"seed={self.seed}, rank={self.rank_selection.selected_rank}"
            ),
            base_model_id=CANONICAL_BASE_MODEL_ID,
            model_family=CANONICAL_MODEL_FAMILY,
            model_revision=CANONICAL_MODEL_REVISION,
            method=FineTuningMethod.QLORA,
            lora=LoRAHyperparameters(
                rank=self.rank_selection.selected_rank,
                alpha=self.rank_selection.selected_alpha,
                dropout=0.05,
                target_modules=self.rank_selection.selected_target_modules,
            ),
            epochs=CANONICAL_EPOCHS,
            per_device_batch_size=CANONICAL_PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=CANONICAL_GRADIENT_ACCUMULATION,
            learning_rate=CANONICAL_LR,
            lr_scheduler=CANONICAL_LR_SCHEDULER,
            schema_format=CANONICAL_SCHEMA_FORMAT,
            max_seq_length=CANONICAL_MAX_SEQ_LENGTH,
            mask_prompt_loss=CANONICAL_MASK_PROMPT_LOSS,
            train_dataset=TRAIN_SPLIT_SPIDER,
            eval_dataset=EVAL_SPLIT_SPIDER,
            seed=self.seed,
            output_dir=str(output_dir),
            max_train_examples=self.total_examples,
        )


# ==============================================================================
# EXP-05 & EXP-06 Experiment Matrix Generators
# ==============================================================================


def generate_exp05_run_conditions(
    rank_selection: RankSelectionReference | None = None,
    sizes: tuple[int, ...] = EXP05_TRAINING_SIZES,
    seeds: tuple[int, ...] = EXP05_SEEDS,
) -> list[ScalingRunCondition]:
    """Generate the exact 15 planned EXP-05 data-scaling run conditions (5 sizes x 3 seeds)."""
    resolved_rank = (
        rank_selection
        if rank_selection is not None
        else RankSelectionReference.create_provisional()
    )
    conditions: list[ScalingRunCondition] = []
    is_provisional = not resolved_rank.is_empirical_eligible

    for size in sizes:
        for seed in seeds:
            run_id = f"exp05_n{size}_s{seed}"
            cond = ScalingRunCondition(
                run_id=run_id,
                experiment_id=EXP05_EXPERIMENT_ID,
                seed=seed,
                total_examples=size,
                human_examples_count=size,
                synthetic_examples_count=0,
                group_name=f"n_{size}",
                rank_selection=resolved_rank,
                status="planned",
                is_provisional=is_provisional,
            )
            conditions.append(cond)
    return conditions


def generate_exp06_run_conditions(
    rank_selection: RankSelectionReference | None = None,
    groups: tuple[str, ...] = EXP06_GROUPS,
    seeds: tuple[int, ...] = EXP06_SEEDS,
) -> list[ScalingRunCondition]:
    """Generate the exact 9 planned EXP-06 synthetic vs. human run conditions (3 groups x 3 seeds)."""
    resolved_rank = (
        rank_selection
        if rank_selection is not None
        else RankSelectionReference.create_provisional()
    )
    conditions: list[ScalingRunCondition] = []
    is_provisional = not resolved_rank.is_empirical_eligible

    for group in groups:
        if group not in EXP06_GROUP_COMPOSITION:
            raise ValueError(
                f"Unknown EXP-06 group {group!r}. Allowed: {list(EXP06_GROUP_COMPOSITION.keys())}"
            )
        human_count, synth_count = EXP06_GROUP_COMPOSITION[group]
        for seed in seeds:
            run_id = f"exp06_{group}_s{seed}"
            cond = ScalingRunCondition(
                run_id=run_id,
                experiment_id=EXP06_EXPERIMENT_ID,
                seed=seed,
                total_examples=EXP06_TOTAL_SIZE,
                human_examples_count=human_count,
                synthetic_examples_count=synth_count,
                group_name=group,
                rank_selection=resolved_rank,
                status="planned",
                is_provisional=is_provisional,
            )
            conditions.append(cond)
    return conditions
