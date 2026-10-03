"""Configuration contracts and validation for Supervised Fine-Tuning (SFT).

Provides typed specifications for LoRA and QLoRA fine-tuning, training hyperparameters,
hardware constraints, tokenization parameters, and checkpoint management.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sqlforge.schemas.models import LoRAHyperparameters, QLoRAHyperparameters


class FineTuningMethod(StrEnum):
    """Supervised fine-tuning adaptation method."""

    LORA = "lora"
    QLORA = "qlora"


# Architecture-aware default target linear projection modules
DEFAULT_TARGET_MODULES_MAP: dict[str, list[str]] = {
    "qwen": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "default": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
}


class SFTTrainingConfig(BaseModel):
    """Complete specification for a supervised fine-tuning run (LoRA or QLoRA)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Experiment identity
    run_id: str | None = Field(
        default=None, description="Unique run identifier (generated if None)"
    )
    experiment_id: str = Field(
        default="EXP-03-LORA-VS-QLORA",
        description="Canonical experiment identifier from experiment matrix",
    )
    description: str = Field(
        default="Supervised fine-tuning on text-to-SQL",
        description="Detailed research description",
    )

    # Base model
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="Hugging Face repository ID or local base model path",
    )
    model_family: str = Field(
        default="qwen", description="Model family ('qwen', 'llama', 'mistral')"
    )
    model_revision: str = Field(
        default="main", description="Git commit hash or branch tag for base weights"
    )

    # Method & LoRA/QLoRA
    method: FineTuningMethod = Field(
        default=FineTuningMethod.LORA,
        description="Fine-tuning adaptation paradigm ('lora' or 'qlora')",
    )
    lora: LoRAHyperparameters = Field(
        default_factory=lambda: LoRAHyperparameters(
            rank=16,
            alpha=32,
            dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        ),
        description="LoRA adapter hyperparameters",
    )
    qlora: QLoRAHyperparameters = Field(
        default_factory=lambda: QLoRAHyperparameters(
            bits=4,
            quant_type="nf4",
            use_double_quant=True,
            compute_dtype="bfloat16",
        ),
        description="QLoRA 4-bit quantization configuration",
    )

    # Training hyperparameters
    epochs: int = Field(default=3, ge=1, le=50, description="Number of complete training epochs")
    per_device_batch_size: int = Field(
        default=8, ge=1, le=128, description="Micro-batch size per device"
    )
    gradient_accumulation_steps: int = Field(
        default=2, ge=1, le=64, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(
        default=2e-4, gt=0.0, le=1e-2, description="Peak learning rate for AdamW"
    )
    lr_scheduler: str = Field(
        default="cosine", description="Learning rate scheduler ('cosine', 'linear', 'constant')"
    )
    warmup_ratio: float = Field(
        default=0.03, ge=0.0, le=0.5, description="Fraction of steps for linear warmup"
    )
    weight_decay: float = Field(
        default=0.01, ge=0.0, le=1.0, description="AdamW weight decay coefficient"
    )
    max_grad_norm: float = Field(
        default=1.0, gt=0.0, description="Maximum gradient norm for clipping"
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing to save VRAM"
    )
    mixed_precision: str = Field(
        default="bf16", description="Mixed precision mode ('bf16', 'fp16', 'no')"
    )
    optimizer: str = Field(
        default="paged_adamw_32bit",
        description="Optimizer name (e.g. 'adamw_torch', 'paged_adamw_32bit', 'paged_adamw_8bit')",
    )

    # Data & Prompting
    train_dataset: str = Field(
        default="spider:train", description="Dataset key for training (must be train split)"
    )
    eval_dataset: str | None = Field(
        default="spider:dev", description="Dataset key for dev evaluation (must be dev split)"
    )
    schema_format: str = Field(
        default="ddl", description="Schema serialization format ('ddl', 'compact_pipe', 'json')"
    )
    max_seq_length: int = Field(
        default=2048, ge=256, le=8192, description="Maximum token sequence length (prompt + SQL)"
    )
    mask_prompt_loss: bool = Field(
        default=True,
        description="Whether to mask prompt/schema tokens with -100 so loss is on SQL tokens only",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    max_train_examples: int | None = Field(
        default=None, ge=1, description="Optional cap on training examples for quick smoke tests"
    )
    max_eval_examples: int | None = Field(
        default=None, ge=1, description="Optional cap on validation examples"
    )

    # Logging & Checkpoints
    logging_steps: int = Field(
        default=10, ge=1, description="Frequency of metric logging in training steps"
    )
    save_steps: int = Field(
        default=100, ge=1, description="Frequency of checkpoint saves in training steps"
    )
    save_total_limit: int = Field(
        default=3, ge=1, description="Maximum number of intermediate checkpoints retained"
    )
    output_dir: str = Field(
        default="artifacts/checkpoints",
        description="Base directory for model checkpoints and run artifacts",
    )
    resume_from_checkpoint: str | None = Field(
        default=None, description="Path to checkpoint directory to resume training from"
    )
    use_wandb: bool = Field(
        default=False, description="Enable Weights & Biases logging (strictly optional)"
    )

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size across gradient accumulation steps."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    @field_validator("train_dataset")
    @classmethod
    def validate_train_split(cls, v: str) -> str:
        """Enforce that training dataset references the training partition."""
        parts = v.split(":")
        if len(parts) == 2 and parts[1].lower() != "train":
            raise ValueError(
                f"Training dataset must use 'train' split, got '{parts[1]}' in '{v}'. "
                "Dev, test, and held-out partitions are strictly forbidden for training."
            )
        return v

    @field_validator("eval_dataset")
    @classmethod
    def validate_eval_split(cls, v: str | None) -> str | None:
        """Enforce that eval dataset does not use training or test partitions."""
        if v is None:
            return None
        parts = v.split(":")
        if len(parts) == 2 and parts[1].lower() not in {"dev", "val", "validation"}:
            raise ValueError(
                f"Evaluation dataset must use 'dev' split, got '{parts[1]}' in '{v}'. "
                "Test and held-out evaluation sets must remain isolated from intermediate training validation."
            )
        return v

    @field_validator("mixed_precision")
    @classmethod
    def validate_mixed_precision(cls, v: str) -> str:
        """Validate mixed precision specification."""
        allowed = {"bf16", "fp16", "no", "float32"}
        if v.lower() not in allowed:
            raise ValueError(f"mixed_precision must be one of {allowed}, got '{v}'")
        return v.lower()

    @model_validator(mode="after")
    def validate_method_settings(self) -> SFTTrainingConfig:
        """Cross-validate LoRA and QLoRA configuration coherence."""
        if self.method == FineTuningMethod.QLORA:
            if self.qlora.bits not in {4, 8}:
                raise ValueError(f"QLoRA bits must be 4 or 8, got {self.qlora.bits}")
            if self.qlora.quant_type not in {"nf4", "fp4"}:
                raise ValueError(
                    f"QLoRA quant_type must be 'nf4' or 'fp4', got '{self.qlora.quant_type}'"
                )

        # Validate LoRA alpha / rank
        if self.lora.rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {self.lora.rank}")
        if self.lora.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.lora.alpha}")

        return self

    def resolve_target_modules(self) -> list[str]:
        """Resolve target module list, falling back to architecture-aware defaults."""
        if self.lora.target_modules:
            return list(self.lora.target_modules)
        family = self.model_family.lower()
        return list(DEFAULT_TARGET_MODULES_MAP.get(family, DEFAULT_TARGET_MODULES_MAP["default"]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a clean dictionary."""
        data = self.model_dump(mode="json")
        data["effective_batch_size"] = self.effective_batch_size
        return data


def resolve_target_modules(
    base_model_id: str,
    target_modules_tag: str = "all-linear",
) -> list[str]:
    """Resolve target module list given base model and preset tag ('all-linear' or 'attention-only')."""
    if target_modules_tag == "attention-only":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    lower = base_model_id.lower()
    family = "qwen" if "qwen" in lower else ("llama" if "llama" in lower else "default")
    return list(DEFAULT_TARGET_MODULES_MAP.get(family, DEFAULT_TARGET_MODULES_MAP["default"]))
