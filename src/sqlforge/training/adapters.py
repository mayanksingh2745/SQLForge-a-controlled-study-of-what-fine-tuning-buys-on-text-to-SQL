"""PEFT LoRA and QLoRA configuration factories and model preparation.

Builds typed Hugging Face PEFT LoraConfig and BitsAndBytesConfig objects with
architecture-aware target module detection, validation, and zero silent fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig

logger = logging.getLogger(__name__)


class UnsupportedQuantizationPlatformError(RuntimeError):
    """Raised when QLoRA is requested on an incompatible platform (e.g. CPU-only or Windows without bitsandbytes)."""


class AdapterConfigSummary(BaseModel):
    """Auditable summary of resolved LoRA/QLoRA adapter configuration."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(..., description="'lora' or 'qlora'")
    rank: int = Field(..., description="LoRA rank r")
    alpha: int = Field(..., description="LoRA alpha scaling factor")
    dropout: float = Field(..., description="LoRA dropout rate")
    target_modules: list[str] = Field(..., description="Target linear projection modules")
    bias: str = Field(..., description="Bias training strategy")
    task_type: str = Field(default="CAUSAL_LM", description="PEFT task type")
    quantization_bits: int | None = Field(
        default=None, description="Quantization bit-width (4 for QLoRA)"
    )
    quant_type: str | None = Field(default=None, description="Quantization algorithm ('nf4')")
    use_double_quant: bool | None = Field(
        default=None, description="Whether nested double quantization is active"
    )
    compute_dtype: str | None = Field(
        default=None, description="Quantization dequantize compute dtype"
    )


class PEFTConfigFactory:
    """Constructs PEFT and BitsAndBytes configurations with architecture-aware defaults."""

    @staticmethod
    def build_lora_config(config: SFTTrainingConfig) -> Any:
        """Build Hugging Face PEFT LoraConfig from SFTTrainingConfig.

        Args:
            config: SFTTrainingConfig specifying LoRA hyperparameters.

        Returns:
            peft.LoraConfig instance (if peft is installed) or AdapterConfigSummary dictionary.

        Raises:
            ImportError: If peft is not installed and real config object is needed.
        """
        targets = config.resolve_target_modules()

        try:
            from peft import LoraConfig, TaskType

            return LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=targets,
                bias=config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
        except ImportError:
            logger.warning(
                "peft package not installed; returning dictionary representation of LoraConfig."
            )
            return {
                "r": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "lora_dropout": config.lora.dropout,
                "target_modules": targets,
                "bias": config.lora.bias,
                "task_type": "CAUSAL_LM",
            }

    @staticmethod
    def build_bitsandbytes_config(config: SFTTrainingConfig) -> Any:
        """Build Hugging Face BitsAndBytesConfig for 4-bit NF4 QLoRA.

        Args:
            config: SFTTrainingConfig with QLoRA configuration.

        Returns:
            transformers.BitsAndBytesConfig instance or configuration dictionary.

        Raises:
            UnsupportedQuantizationPlatformError: If QLoRA requested without CUDA or bitsandbytes.
        """
        if config.method != FineTuningMethod.QLORA:
            return None

        # Verify bitsandbytes and torch
        try:
            import torch
        except ImportError as exc:
            raise UnsupportedQuantizationPlatformError(
                "QLoRA requires 'torch' with CUDA support, but torch is not installed."
            ) from exc

        if not torch.cuda.is_available():
            raise UnsupportedQuantizationPlatformError(
                "QLoRA 4-bit NF4 quantization requires a CUDA-capable GPU. "
                "No CUDA device is available in this environment. "
                "Silent fallback to LoRA is strictly disabled to preserve research reproducibility."
            )

        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError as exc:
            raise UnsupportedQuantizationPlatformError(
                "QLoRA requires 'bitsandbytes' for 4-bit NF4 weight quantization. "
                "Install via: pip install bitsandbytes."
            ) from exc

        # Resolve compute dtype
        dtype_map = {
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(config.qlora.compute_dtype, torch.bfloat16)

        try:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=(config.qlora.bits == 4),
                load_in_8bit=(config.qlora.bits == 8),
                bnb_4bit_quant_type=config.qlora.quant_type,
                bnb_4bit_use_double_quant=config.qlora.use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except ImportError as exc:
            raise UnsupportedQuantizationPlatformError(
                "transformers package is required to build BitsAndBytesConfig."
            ) from exc

    @staticmethod
    def summarize_adapter_config(config: SFTTrainingConfig) -> AdapterConfigSummary:
        """Produce an auditable summary of adapter parameters."""
        targets = config.resolve_target_modules()
        is_qlora = config.method == FineTuningMethod.QLORA

        return AdapterConfigSummary(
            method=config.method.value,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            target_modules=targets,
            bias=config.lora.bias,
            task_type="CAUSAL_LM",
            quantization_bits=config.qlora.bits if is_qlora else None,
            quant_type=config.qlora.quant_type if is_qlora else None,
            use_double_quant=config.qlora.use_double_quant if is_qlora else None,
            compute_dtype=config.qlora.compute_dtype if is_qlora else None,
        )

    @staticmethod
    def prepare_model_for_training(model: Any, is_qlora: bool = False) -> Any:
        """Prepare causal LM for PEFT adapter training.

        Enables gradient checkpointing and input-gradient hooks as required by LoRA/QLoRA.

        Args:
            model: PyTorch PreTrainedModel instance.
            is_qlora: Whether the model was loaded with 4-bit quantization.

        Returns:
            Prepared model ready for PEFT wrapping.
        """
        if is_qlora:
            try:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                return model
            except ImportError:
                logger.warning("peft.prepare_model_for_kbit_training not available.")

        # Standard LoRA gradient checkpointing
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        return model
