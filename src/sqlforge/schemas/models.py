"""Data contracts for model architectures, fine-tuning specs, and generation."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelType(StrEnum):
    """Categorization of model access and adaptation method."""

    BASE = "base"
    LORA = "lora"
    QLORA = "qlora"
    API = "api"


class LoRAHyperparameters(BaseModel):
    """Low-Rank Adaptation hyperparameters."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(default=16, ge=1, le=256, description="Intrinsic low-rank dimension r")
    alpha: int = Field(default=32, ge=1, description="Scaling factor alpha")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout probability")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Linear attention projection modules to adapt",
    )
    bias: str = Field(default="none", description="Bias parameter adaptation strategy")


class QLoRAHyperparameters(BaseModel):
    """Quantized LoRA settings for memory-efficient fine-tuning."""

    model_config = ConfigDict(frozen=True)

    bits: int = Field(default=4, description="Quantization bit-width (4 or 8)")
    quant_type: str = Field(
        default="nf4", description="Quantization representation ('nf4' or 'fp4')"
    )
    use_double_quant: bool = Field(
        default=True, description="Nested quantization for additional memory reduction"
    )
    compute_dtype: str = Field(
        default="bfloat16", description="Computation precision ('bfloat16' or 'float16')"
    )


class GenerationDefaults(BaseModel):
    """Inference decoding parameters."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    stop_tokens: list[str] = Field(default_factory=lambda: [";", "```", "<|im_end|>"])


class ModelConfig(BaseModel):
    """Configuration contract for model instantiation and adaptation."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(..., description="Hugging Face repo ID or API model identifier")
    model_type: ModelType = Field(
        default=ModelType.BASE, description="Architecture/adaptation paradigm"
    )
    revision: str = Field(
        default="main", description="Git commit hash or branch tag for model weights"
    )
    context_window: int = Field(
        default=8192, ge=512, description="Maximum context length in tokens"
    )
    torch_dtype: str = Field(default="bfloat16", description="Base model weight precision")
    lora: LoRAHyperparameters | None = Field(
        default=None, description="LoRA config if adapting with LoRA/QLoRA"
    )
    qlora: QLoRAHyperparameters | None = Field(
        default=None, description="QLoRA quantization configuration"
    )
    is_api: bool = Field(default=False, description="True if accessed via external HTTP API")
    provider: str | None = Field(
        default=None, description="Provider name if is_api is True (e.g. 'openai')"
    )
