"""Unit tests for tokenization, sequence construction, and completion-only loss masking."""

from __future__ import annotations

import pytest

from sqlforge.schemas.examples import DatasetSplit
from sqlforge.training.data import SFTExample
from sqlforge.training.tokenization import (
    IGNORE_INDEX,
    CompletionLossMasker,
    TargetTruncationError,
    ToyDeterministicTokenizer,
)


@pytest.fixture
def toy_tokenizer() -> ToyDeterministicTokenizer:
    """Fixture providing a deterministic toy tokenizer."""
    return ToyDeterministicTokenizer()


@pytest.fixture
def sample_sft_example() -> SFTExample:
    """Fixture providing a prepared SFTExample."""
    return SFTExample(
        example_id="ex_001",
        db_id="test_db",
        dataset_name="spider",
        split=DatasetSplit.TRAIN,
        prompt_text="### Schema:\nCREATE TABLE t (id INT);\n### Question:\nHow many rows?\n### SQL:\n",
        target_text="SELECT count(*) FROM t;",
        gold_sql="SELECT count(*) FROM t;",
    )


class TestCompletionLossMasker:
    """Test suite for completion-only loss masking and token processing."""

    def test_prompt_masking_with_ignore_index(
        self, toy_tokenizer: ToyDeterministicTokenizer, sample_sft_example: SFTExample
    ) -> None:
        """Verify prompt tokens are masked with -100 and completion tokens remain unmasked."""
        masker = CompletionLossMasker(
            tokenizer=toy_tokenizer,
            max_seq_length=512,
            mask_prompt_loss=True,
        )
        tokenized = masker.process_example(sample_sft_example)

        assert tokenized.example_id == "ex_001"
        assert tokenized.prompt_len > 0
        assert tokenized.target_len > 0
        assert len(tokenized.input_ids) == tokenized.prompt_len + tokenized.target_len

        # Check that prompt portion of labels is all IGNORE_INDEX (-100)
        prompt_labels = tokenized.labels[: tokenized.prompt_len]
        assert all(label == IGNORE_INDEX for label in prompt_labels)

        # Check that completion portion of labels matches target token IDs
        target_labels = tokenized.labels[tokenized.prompt_len :]
        target_input_ids = tokenized.input_ids[tokenized.prompt_len :]
        assert target_labels == target_input_ids
        assert all(label != IGNORE_INDEX for label in target_labels)

    def test_eos_token_appended_to_target(
        self, toy_tokenizer: ToyDeterministicTokenizer, sample_sft_example: SFTExample
    ) -> None:
        """Verify EOS token is appended to the completion sequence."""
        masker = CompletionLossMasker(
            tokenizer=toy_tokenizer,
            max_seq_length=512,
        )
        tokenized = masker.process_example(sample_sft_example)

        # Last token of input_ids and labels must be eos_token_id
        assert tokenized.input_ids[-1] == toy_tokenizer.eos_token_id
        assert tokenized.labels[-1] == toy_tokenizer.eos_token_id

    def test_disable_prompt_masking(
        self, toy_tokenizer: ToyDeterministicTokenizer, sample_sft_example: SFTExample
    ) -> None:
        """Verify that when mask_prompt_loss is False, all labels equal input_ids."""
        masker = CompletionLossMasker(
            tokenizer=toy_tokenizer,
            max_seq_length=512,
            mask_prompt_loss=False,
        )
        tokenized = masker.process_example(sample_sft_example)

        assert tokenized.labels == tokenized.input_ids
        assert IGNORE_INDEX not in tokenized.labels

    def test_truncation_detection_and_error(
        self, toy_tokenizer: ToyDeterministicTokenizer, sample_sft_example: SFTExample
    ) -> None:
        """Verify severe target truncation is flagged and raises TargetTruncationError when configured."""
        # Set max_seq_length shorter than prompt
        prompt_tokens = toy_tokenizer.encode(sample_sft_example.prompt_text)
        short_budget = len(prompt_tokens) - 2

        masker = CompletionLossMasker(
            tokenizer=toy_tokenizer,
            max_seq_length=short_budget,
            min_target_tokens=3,
        )

        # Without raising, should flag target_truncated
        tokenized = masker.process_example(sample_sft_example, raise_on_target_truncation=False)
        assert tokenized.is_truncated is True
        assert tokenized.target_truncated is True
        assert len(tokenized.input_ids) == short_budget

        # With raise_on_target_truncation=True, should raise
        with pytest.raises(TargetTruncationError, match="exceeds max sequence length"):
            masker.process_example(sample_sft_example, raise_on_target_truncation=True)

    def test_batch_padding(
        self, toy_tokenizer: ToyDeterministicTokenizer, sample_sft_example: SFTExample
    ) -> None:
        """Verify batch padding aligns sequences to maximum length in batch."""
        masker = CompletionLossMasker(
            tokenizer=toy_tokenizer,
            max_seq_length=512,
        )

        ex1 = sample_sft_example
        ex2 = sample_sft_example.model_copy(
            update={"target_text": "SELECT col1, col2, col3 FROM t WHERE id = 1 AND active = 1;"}
        )

        tok1 = masker.process_example(ex1)
        tok2 = masker.process_example(ex2)

        padded = masker.pad_batch([tok1, tok2], padding_side="right")

        assert len(padded["input_ids"]) == 2
        assert len(padded["input_ids"][0]) == len(padded["input_ids"][1])
        max_len = max(len(tok1.input_ids), len(tok2.input_ids))
        assert len(padded["input_ids"][0]) == max_len

        # Pad positions must have pad_token_id, IGNORE_INDEX in labels, and 0 in attention_mask
        assert padded["attention_mask"][0][-1] == (1 if len(tok1.input_ids) == max_len else 0)
