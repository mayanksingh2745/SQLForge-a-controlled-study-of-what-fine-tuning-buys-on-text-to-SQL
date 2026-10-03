"""Unit tests for SFT training configuration, LoRA/QLoRA schemas, and parameter validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sqlforge.schemas.models import LoRAHyperparameters, QLoRAHyperparameters
from sqlforge.training.config import (
    DEFAULT_TARGET_MODULES_MAP,
    FineTuningMethod,
    SFTTrainingConfig,
)


class TestSFTTrainingConfig:
    """Test suite for SFTTrainingConfig schema and validation rules."""

    def test_default_config_initialization(self) -> None:
        """Verify default configuration values."""
        config = SFTTrainingConfig()
        assert config.method == FineTuningMethod.LORA
        assert config.base_model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.epochs == 3
        assert config.per_device_batch_size == 8
        assert config.gradient_accumulation_steps == 2
        assert config.effective_batch_size == 16
        assert config.learning_rate == 2e-4
        assert config.mixed_precision == "bf16"
        assert config.max_seq_length == 2048
        assert config.mask_prompt_loss is True
        assert config.train_dataset == "spider:train"
        assert config.eval_dataset == "spider:dev"

    def test_target_modules_resolution(self) -> None:
        """Verify architecture-aware target module defaults."""
        qwen_cfg = SFTTrainingConfig(
            model_family="qwen",
            lora=LoRAHyperparameters(target_modules=[]),
        )
        assert qwen_cfg.resolve_target_modules() == DEFAULT_TARGET_MODULES_MAP["qwen"]

        llama_cfg = SFTTrainingConfig(
            model_family="llama",
            lora=LoRAHyperparameters(target_modules=[]),
        )
        assert llama_cfg.resolve_target_modules() == DEFAULT_TARGET_MODULES_MAP["llama"]

        custom_cfg = SFTTrainingConfig(
            lora=LoRAHyperparameters(target_modules=["c_attn", "c_proj"]),
        )
        assert custom_cfg.resolve_target_modules() == ["c_attn", "c_proj"]

    def test_reject_disallowed_train_splits(self) -> None:
        """Enforce that train_dataset must reference 'train' partition."""
        with pytest.raises(ValidationError, match="Training dataset must use 'train' split"):
            SFTTrainingConfig(train_dataset="spider:dev")

        with pytest.raises(ValidationError, match="Training dataset must use 'train' split"):
            SFTTrainingConfig(train_dataset="spider:test")

        with pytest.raises(ValidationError, match="Training dataset must use 'train' split"):
            SFTTrainingConfig(train_dataset="custom:held_out")

    def test_reject_disallowed_eval_splits(self) -> None:
        """Enforce that eval_dataset must reference 'dev' partition."""
        with pytest.raises(ValidationError, match="Evaluation dataset must use 'dev' split"):
            SFTTrainingConfig(eval_dataset="spider:train")

        with pytest.raises(ValidationError, match="Evaluation dataset must use 'dev' split"):
            SFTTrainingConfig(eval_dataset="spider:test")

        # None is allowed (no evaluation during training)
        cfg = SFTTrainingConfig(eval_dataset=None)
        assert cfg.eval_dataset is None

    def test_qlora_configuration_validation(self) -> None:
        """Verify QLoRA configuration constraints."""
        valid_qlora = SFTTrainingConfig(
            method=FineTuningMethod.QLORA,
            qlora=QLoRAHyperparameters(bits=4, quant_type="nf4", compute_dtype="bfloat16"),
        )
        assert valid_qlora.method == FineTuningMethod.QLORA
        assert valid_qlora.qlora.bits == 4

        with pytest.raises(ValidationError, match="QLoRA quant_type must be 'nf4' or 'fp4'"):
            SFTTrainingConfig(
                method=FineTuningMethod.QLORA,
                qlora=QLoRAHyperparameters(quant_type="int4"),
            )

    def test_invalid_hyperparameters(self) -> None:
        """Ensure out-of-bound numerical hyperparameters raise validation errors."""
        with pytest.raises(ValidationError):
            SFTTrainingConfig(epochs=0)

        with pytest.raises(ValidationError):
            SFTTrainingConfig(per_device_batch_size=-1)

        with pytest.raises(ValidationError):
            SFTTrainingConfig(learning_rate=-0.001)

        with pytest.raises(ValidationError):
            SFTTrainingConfig(mixed_precision="invalid_precision")

    def test_serialization_to_dict(self) -> None:
        """Verify serialization to dictionary."""
        config = SFTTrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["method"] == "lora"
        assert d["effective_batch_size"] == 16
