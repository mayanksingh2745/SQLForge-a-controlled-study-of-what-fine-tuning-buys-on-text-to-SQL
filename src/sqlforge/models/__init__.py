"""Model layer: Unified interfaces for local models and frontier API baselines.

Planned for future steps:
- Hugging Face causal LM wrappers (Qwen2.5-Coder, Llama-3.1, DeepSeek-Coder).
- Frontier API wrappers (OpenAI, Anthropic) with retry and rate-limiting.
- LoRA/QLoRA adapter loader and weight merger.
"""

from typing import Protocol

from sqlforge.schemas.evaluation import GenerationResult


class ModelRunner(Protocol):
    """Protocol for model inference engines."""

    def generate(self, prompt: str, **kwargs: object) -> GenerationResult:
        """Generate SQL completion for a prompt."""
        ...
