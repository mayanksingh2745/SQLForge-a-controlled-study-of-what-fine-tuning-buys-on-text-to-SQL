"""Tokenization, sequence padding, and completion-only loss masking for SFT.

Ensures the cross-entropy training loss is computed exclusively on the target SQL completion
tokens by masking prompt/schema tokens with -100 (IGNORE_INDEX), while handling EOS termination,
attention masks, and truncation detection.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.training.data import SFTExample

logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100


class TargetTruncationError(ValueError):
    """Raised when an example's target SQL completion is truncated to an unusable length."""


class TokenizedSFTExample(BaseModel):
    """Tokenized supervised training instance with loss mask and attention mask."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., description="Unique example identifier")
    input_ids: list[int] = Field(..., description="Token ID sequence (prompt + target)")
    labels: list[int] = Field(
        ..., description="Training labels with prompt masked to IGNORE_INDEX (-100)"
    )
    attention_mask: list[int] = Field(
        ..., description="Binary attention mask (1 for real, 0 for pad)"
    )
    prompt_len: int = Field(..., description="Number of tokens in the prompt portion")
    target_len: int = Field(
        ..., description="Number of tokens in the target portion (including EOS)"
    )
    is_truncated: bool = Field(
        default=False, description="True if sequence exceeded max_seq_length"
    )
    target_truncated: bool = Field(
        default=False, description="True if target SQL was truncated severely"
    )


class ToyDeterministicTokenizer:
    """Deterministic token-level mock tokenizer for unit tests without external models.

    Maps words and special symbols to stable integer token IDs with deterministic encoding/decoding.
    """

    def __init__(
        self,
        pad_token: str = "<|pad|>",
        eos_token: str = "<|im_end|>",
        bos_token: str | None = None,
    ) -> None:
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.vocab: dict[str, int] = {
            self.pad_token: 0,
            self.eos_token: 1,
        }
        if bos_token:
            self.vocab[bos_token] = 2
        self.reverse_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._next_id = len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text by splitting on whitespace and punctuation tokens."""
        tokens = text.replace("\n", " \n ").split()
        ids: list[int] = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = self._next_id
                self.reverse_vocab[self._next_id] = t
                self._next_id += 1
            ids.append(self.vocab[t])
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        parts: list[str] = []
        special_ids = {self.pad_token_id, self.eos_token_id}
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            parts.append(self.reverse_vocab.get(tid, f"<unk_{tid}>"))
        return " ".join(parts).replace(" \n ", "\n")

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        """HF-compatible call interface."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class CompletionLossMasker:
    """Constructs prompt-target sequences and applies completion-only loss masking."""

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int = 2048,
        mask_prompt_loss: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        min_target_tokens: int = 3,
    ) -> None:
        """Initialize CompletionLossMasker.

        Args:
            tokenizer: HF tokenizer or compatible tokenizer instance.
            max_seq_length: Maximum allowed token sequence length.
            mask_prompt_loss: If True, set prompt token labels to IGNORE_INDEX (-100).
            pad_token_id: Token ID for padding. Defaults to tokenizer.pad_token_id or eos_token_id.
            eos_token_id: Token ID for EOS. Defaults to tokenizer.eos_token_id.
            min_target_tokens: Minimum required surviving target tokens before raising truncation error.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt_loss = mask_prompt_loss
        self.min_target_tokens = min_target_tokens

        # Resolve pad and eos token IDs
        self.eos_token_id: int = (
            int(eos_token_id)
            if eos_token_id is not None
            else int(getattr(tokenizer, "eos_token_id", 1))
        )
        resolved_pad = (
            pad_token_id if pad_token_id is not None else getattr(tokenizer, "pad_token_id", None)
        )
        self.pad_token_id: int = (
            int(resolved_pad) if resolved_pad is not None else self.eos_token_id
        )

    def process_example(
        self,
        example: SFTExample,
        raise_on_target_truncation: bool = False,
    ) -> TokenizedSFTExample:
        """Tokenize an SFTExample, apply loss mask, and enforce length boundaries.

        Args:
            example: SFTExample instance containing prompt_text and target_text.
            raise_on_target_truncation: If True, raise TargetTruncationError if target is cut off.

        Returns:
            TokenizedSFTExample with input_ids, labels, and attention_mask.
        """
        # 1. Tokenize prompt
        if hasattr(self.tokenizer, "encode"):
            prompt_ids = self.tokenizer.encode(example.prompt_text, add_special_tokens=False)
        else:
            prompt_ids = self.tokenizer(example.prompt_text, add_special_tokens=False)["input_ids"]

        # 2. Tokenize target SQL and append EOS
        if hasattr(self.tokenizer, "encode"):
            target_ids = self.tokenizer.encode(example.target_text, add_special_tokens=False)
        else:
            target_ids = self.tokenizer(example.target_text, add_special_tokens=False)["input_ids"]

        # Append EOS token to target if not already present
        if not target_ids or target_ids[-1] != self.eos_token_id:
            target_ids.append(self.eos_token_id)

        prompt_len = len(prompt_ids)
        target_len = len(target_ids)
        total_len = prompt_len + target_len

        # 3. Construct input_ids and labels
        input_ids = prompt_ids + target_ids

        if self.mask_prompt_loss:
            labels = [IGNORE_INDEX] * prompt_len + list(target_ids)
        else:
            labels = list(input_ids)

        is_truncated = False
        target_truncated = False

        # 4. Handle truncation against max_seq_length
        if total_len > self.max_seq_length:
            is_truncated = True

            # If prompt consumes the entire budget, target is completely truncated
            if prompt_len >= self.max_seq_length:
                target_truncated = True
                surviving_target = 0
            else:
                surviving_target = self.max_seq_length - prompt_len
                if surviving_target < self.min_target_tokens:
                    target_truncated = True

            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

            if target_truncated and raise_on_target_truncation:
                raise TargetTruncationError(
                    f"Example '{example.example_id}' exceeds max sequence length ({total_len} > {self.max_seq_length}). "
                    f"Target SQL truncated to {surviving_target} tokens (minimum required: {self.min_target_tokens})."
                )

        attention_mask = [1] * len(input_ids)

        return TokenizedSFTExample(
            example_id=example.example_id,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            target_len=target_len,
            is_truncated=is_truncated,
            target_truncated=target_truncated,
        )

    def pad_batch(
        self,
        batch: list[TokenizedSFTExample],
        padding_side: str = "right",
    ) -> dict[str, Any]:
        """Pad a batch of tokenized examples to the longest sequence in the batch.

        Args:
            batch: List of TokenizedSFTExample instances.
            padding_side: 'right' (standard for causal LM training) or 'left'.

        Returns:
            Dictionary with padded 'input_ids', 'labels', and 'attention_mask' as nested lists.
        """
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": []}

        max_len = max(len(ex.input_ids) for ex in batch)
        padded_inputs: list[list[int]] = []
        padded_labels: list[list[int]] = []
        padded_masks: list[list[int]] = []

        for ex in batch:
            pad_len = max_len - len(ex.input_ids)
            if padding_side == "right":
                p_in = ex.input_ids + [self.pad_token_id] * pad_len
                p_lab = ex.labels + [IGNORE_INDEX] * pad_len
                p_mask = ex.attention_mask + [0] * pad_len
            else:
                p_in = [self.pad_token_id] * pad_len + ex.input_ids
                p_lab = [IGNORE_INDEX] * pad_len + ex.labels
                p_mask = [0] * pad_len + ex.attention_mask

            padded_inputs.append(p_in)
            padded_labels.append(p_lab)
            padded_masks.append(p_mask)

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": padded_masks,
        }
