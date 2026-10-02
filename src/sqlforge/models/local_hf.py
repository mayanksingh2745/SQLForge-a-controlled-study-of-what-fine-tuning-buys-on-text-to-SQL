"""Local Hugging Face model inference runner with lazy-loaded dependencies.

Supports Qwen2.5-Coder (1.5B, 7B) and Llama-3.1 with greedy decoding,
device detection, and per-example token accounting.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from sqlforge.schemas.evaluation import GenerationResult

logger = logging.getLogger(__name__)


class LocalModelDependencyError(RuntimeError):
    """Raised when local inference is invoked but torch or transformers is uninstalled."""


class LocalHFModelRunner:
    """Inference runner for local Hugging Face causal language models."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        inference_fn: Callable[[str, int], tuple[str, int, int]] | None = None,
    ) -> None:
        """Initialize LocalHFModelRunner.

        Args:
            model_id: Hugging Face repository slug or local checkpoint path.
            device: Target compute device ('cuda', 'cpu', 'auto').
            torch_dtype: Model tensor precision ('bfloat16', 'float16', 'float32').
            temperature: Sampling temperature (0.0 for deterministic greedy decoding).
            max_new_tokens: Maximum completion token budget (default 512).
            inference_fn: Optional mock inference callable for offline verification.
        """
        self._model_id = model_id
        self.device = device or "auto"
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._inference_fn = inference_fn

        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model_id(self) -> str:
        """Hugging Face model repository ID."""
        return self._model_id

    def _ensure_loaded(self) -> None:
        """Lazily load torch, transformers, tokenizer, and model weights."""
        if self._inference_fn is not None:
            return

        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalModelDependencyError(
                "Local Hugging Face model inference requires 'torch' and 'transformers'. "
                'Install via: pip install -e ".[train]".'
            ) from exc

        logger.info(f"Loading local tokenizer and model for '{self._model_id}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)

        dtype_map = {
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(self.torch_dtype, torch.float32)

        device_map = (
            self.device
            if self.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        example_id: str = "example",
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        stop_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Execute deterministic greedy generation for a prompt.

        Args:
            prompt: Text prompt string.
            example_id: Evaluation example identifier.
            temperature: Sampling temperature (must be 0.0 for deterministic greedy decoding).
            max_new_tokens: Optional completion token limit.
            stop_tokens: Stop token strings.
            **kwargs: Extra arguments.

        Returns:
            GenerationResult with predicted SQL, token counts, and latency.
        """
        effective_max = max_new_tokens or self.max_new_tokens
        start_time = time.monotonic()

        # Handle test injection / mock callable
        if self._inference_fn is not None:
            raw_sql, p_tok, c_tok = self._inference_fn(prompt, effective_max)
            latency_ms = (time.monotonic() - start_time) * 1000.0
            clean_sql = self._clean_sql(raw_sql)
            return GenerationResult(
                example_id=example_id,
                generated_sql=clean_sql,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=round(latency_ms, 2),
                finish_reason="stop",
                model_id=self._model_id,
            )

        self._ensure_loaded()
        import torch

        # Tokenize prompt
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=effective_max,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Slice new generated tokens
        new_tokens = outputs[0][prompt_tokens:]
        completion_tokens = len(new_tokens)
        decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency_ms = (time.monotonic() - start_time) * 1000.0

        clean_sql = self._clean_sql(decoded)

        return GenerationResult(
            example_id=example_id,
            generated_sql=clean_sql,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason="stop",
            model_id=self._model_id,
        )

    @staticmethod
    def _clean_sql(text: str) -> str:
        """Strip markdown fences and whitespace from model output."""
        cleaned = text.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for p in parts:
                p_strip = p.strip()
                if p_strip.startswith("sql"):
                    cleaned = p_strip[3:].strip()
                    break
                elif p_strip.upper().startswith("SELECT") or p_strip.upper().startswith("WITH"):
                    cleaned = p_strip
                    break

        if not cleaned.endswith(";"):
            cleaned = f"{cleaned};"
        return cleaned
