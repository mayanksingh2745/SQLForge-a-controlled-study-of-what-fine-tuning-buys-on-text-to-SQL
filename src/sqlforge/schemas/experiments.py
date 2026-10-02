"""Data contracts for experiment specifications, runs, and metadata."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingParams(BaseModel):
    """Hyperparameters for supervised fine-tuning runs."""

    model_config = ConfigDict(frozen=True)

    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=16, ge=1)
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    lr_scheduler: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)


class PromptConfig(BaseModel):
    """Prompt engineering and schema representation settings."""

    model_config = ConfigDict(frozen=True)

    template_name: str = Field(default="default_text2sql")
    schema_format: str = Field(default="ddl", description="'ddl', 'compact', or 'json'")
    k_shots: int = Field(default=0, ge=0, description="Number of in-context demonstration examples")
    retriever_type: str | None = Field(
        default=None, description="'bm25', 'dense_embedding', or None"
    )


class ExecutionSandboxConfig(BaseModel):
    """Runtime limits and isolation rules for database query execution."""

    model_config = ConfigDict(frozen=True)

    timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_memory_mb: int = Field(default=1024, ge=128)
    single_statement_only: bool = Field(default=True)
    enforce_read_only: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Complete specification of an individual experimental run."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(
        ..., description="Unique slug for the experiment (e.g. 'exp03_lora_r16')"
    )
    description: str = Field(..., description="Detailed description of hypothesis being tested")
    seed: int = Field(default=42, description="Random seed for data shuffling and initialization")
    model_id: str = Field(..., description="Model key or identifier")
    paradigm: str = Field(
        default="zero-shot", description="'zero-shot', 'few-shot', or 'fine-tuning'"
    )
    train_dataset: str | None = Field(
        default=None, description="Dataset key for training (e.g. 'spider:train')"
    )
    eval_datasets: list[str] = Field(
        default_factory=lambda: ["spider:dev"], description="Evaluation datasets"
    )
    training: TrainingParams | None = Field(
        default=None, description="Training params if paradigm is fine-tuning"
    )
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    execution: ExecutionSandboxConfig = Field(default_factory=ExecutionSandboxConfig)
    output_dir: str = Field(default="artifacts/runs")


class HardwareMetadata(BaseModel):
    """Snapshot of execution hardware resources."""

    model_config = ConfigDict(frozen=True)

    cpu_count: int = Field(default=1)
    total_ram_gb: float = Field(default=0.0)
    gpu_count: int = Field(default=0)
    gpu_names: list[str] = Field(default_factory=list)
    cuda_available: bool = Field(default=False)


class RunMetadata(BaseModel):
    """Immutable audit record generated at the start and completion of an experiment run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Globally unique run identifier")
    experiment_name: str = Field(
        ..., description="Name or identifier of the experiment configuration"
    )
    timestamp_utc: str = Field(..., description="ISO 8601 UTC timestamp of execution start")
    git_commit: str | None = Field(default=None, description="HEAD commit hash")
    git_branch: str | None = Field(default=None, description="Git branch name")
    git_dirty: bool = Field(
        default=False, description="True if working tree had uncommitted modifications"
    )
    python_version: str = Field(..., description="Python runtime version")
    platform: str = Field(..., description="Operating system platform details")
    hardware: HardwareMetadata = Field(..., description="Hardware resource snapshot")
    seed: int = Field(default=42, description="Primary random seed")
    status: str = Field(
        default="initialized", description="'initialized', 'running', 'completed', 'failed'"
    )
    config_hash: str | None = Field(
        default=None, description="SHA256 checksum of resolved configuration"
    )
    extra: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extension metadata")
