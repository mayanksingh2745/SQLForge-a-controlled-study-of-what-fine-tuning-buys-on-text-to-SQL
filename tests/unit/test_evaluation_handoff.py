"""Unit tests for evaluation handoff bridge and PEFT adapter integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlforge.models.local_hf import LocalHFModelRunner
from sqlforge.training.config import SFTTrainingConfig
from sqlforge.training.evaluation_handoff import (
    create_evaluation_config,
    prepare_evaluation_runner,
    validate_adapter_checkpoint,
)


class TestEvaluationHandoff:
    """Test suite for post-training evaluation handoff contracts."""

    def test_validate_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        """Verify error is raised if checkpoint directory is missing."""
        with pytest.raises(FileNotFoundError, match="Checkpoint directory does not exist"):
            validate_adapter_checkpoint(tmp_path / "nonexistent")

    def test_prepare_evaluation_runner(self, tmp_path: Path) -> None:
        """Verify LocalHFModelRunner is correctly configured with adapter_path."""
        ckpt_dir = tmp_path / "adapter_ckpt"
        ckpt_dir.mkdir()

        def mock_infer(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            return "SELECT 1;", 10, 5

        runner = prepare_evaluation_runner(
            base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_checkpoint_dir=ckpt_dir,
            inference_fn=mock_infer,
        )

        assert isinstance(runner, LocalHFModelRunner)
        assert runner.adapter_path == ckpt_dir
        assert "adapter_ckpt" in runner.model_id

        # Verify generation with mock callable works
        res = runner.generate(prompt="SELECT ")
        assert res.generated_sql == "SELECT 1;"

    def test_create_evaluation_config_enforces_dev_split(self, tmp_path: Path) -> None:
        """Verify evaluation configuration allows dev split and rejects test/held-out."""
        train_cfg = SFTTrainingConfig(run_id="run_exp03")
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()

        # Valid dev evaluation config
        eval_cfg = create_evaluation_config(
            train_config=train_cfg,
            adapter_checkpoint_dir=ckpt_dir,
            eval_dataset="spider:dev",
        )
        assert eval_cfg.eval_datasets == ["spider:dev"]
        assert eval_cfg.train_dataset is None
        assert "checkpoint-100" in eval_cfg.model_id
        assert "artifacts/runs/run_exp03_checkpoint-100_eval" in eval_cfg.output_dir

        # Disallowed test split raises ValueError
        with pytest.raises(ValueError, match="strictly prohibits evaluating on 'test' split"):
            create_evaluation_config(
                train_config=train_cfg,
                adapter_checkpoint_dir=ckpt_dir,
                eval_dataset="spider:test",
            )

        with pytest.raises(ValueError, match="strictly prohibits evaluating on 'held_out' split"):
            create_evaluation_config(
                train_config=train_cfg,
                adapter_checkpoint_dir=ckpt_dir,
                eval_dataset="spider:held_out",
            )
