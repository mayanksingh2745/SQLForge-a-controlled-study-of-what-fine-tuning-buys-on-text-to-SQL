"""Unit tests for PEFT adapter builders, quantization safety, and checkpoint layout."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sqlforge.schemas.models import LoRAHyperparameters, QLoRAHyperparameters
from sqlforge.training.adapters import (
    PEFTConfigFactory,
    UnsupportedQuantizationPlatformError,
)
from sqlforge.training.checkpoints import CheckpointManager
from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig


class TestPEFTConfigFactory:
    """Test suite for PEFT and BitsAndBytes configuration factories."""

    def test_build_lora_config_dict_fallback(self) -> None:
        """Verify fallback to dict when peft is not installed."""
        config = SFTTrainingConfig(
            lora=LoRAHyperparameters(rank=32, alpha=64, dropout=0.1),
        )
        lora_cfg = PEFTConfigFactory.build_lora_config(config)

        # Either LoraConfig object or dict
        if isinstance(lora_cfg, dict):
            assert lora_cfg["r"] == 32
            assert lora_cfg["lora_alpha"] == 64
            assert lora_cfg["lora_dropout"] == 0.1
        else:
            assert lora_cfg.r == 32
            assert lora_cfg.lora_alpha == 64

    def test_qlora_no_silent_fallback_on_cpu(self) -> None:
        """Ensure QLoRA raises UnsupportedQuantizationPlatformError when CUDA is absent."""
        config = SFTTrainingConfig(
            method=FineTuningMethod.QLORA,
            qlora=QLoRAHyperparameters(bits=4, quant_type="nf4"),
        )

        # Simulate torch without CUDA
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with (
            patch.dict("sys.modules", {"torch": mock_torch}),
            pytest.raises(
                UnsupportedQuantizationPlatformError,
                match="requires a CUDA-capable GPU",
            ),
        ):
            PEFTConfigFactory.build_bitsandbytes_config(config)

    def test_adapter_summary_generation(self) -> None:
        """Verify AdapterConfigSummary reflects configuration accurately."""
        config = SFTTrainingConfig(
            method=FineTuningMethod.QLORA,
            lora=LoRAHyperparameters(rank=16, alpha=32),
            qlora=QLoRAHyperparameters(bits=4, quant_type="nf4", use_double_quant=True),
        )
        summary = PEFTConfigFactory.summarize_adapter_config(config)

        assert summary.method == "qlora"
        assert summary.rank == 16
        assert summary.alpha == 32
        assert summary.quantization_bits == 4
        assert summary.quant_type == "nf4"
        assert summary.use_double_quant is True


class TestCheckpointManager:
    """Test suite for CheckpointManager and CheckpointMetadata."""

    def test_checkpoint_paths_and_metadata(self, tmp_path: Path) -> None:
        """Verify deterministic checkpoint path resolution and metadata serialization."""
        mgr = CheckpointManager(base_dir=tmp_path, run_id="test_run_123")

        step_dir = mgr.get_step_dir(100)
        final_dir = mgr.get_final_dir()

        assert step_dir.name == "checkpoint-100"
        assert final_dir.name == "final_adapter"

        config = SFTTrainingConfig()
        adapter_summary = PEFTConfigFactory.summarize_adapter_config(config)

        meta_file = mgr.save_checkpoint_metadata(
            target_dir=step_dir,
            step=100,
            epoch=1.0,
            config=config,
            adapter_summary=adapter_summary,
            training_metrics={"loss": 0.52},
            dataset_info={"count": 500},
        )

        assert meta_file.exists()
        loaded = mgr.load_metadata(step_dir)
        assert loaded.run_id == "test_run_123"
        assert loaded.step == 100
        assert loaded.epoch == 1.0
        assert loaded.training_metrics["loss"] == 0.52
        assert loaded.base_model_id == config.base_model_id

    def test_list_and_find_latest_checkpoint(self, tmp_path: Path) -> None:
        """Verify sorting and discovery of checkpoints."""
        mgr = CheckpointManager(base_dir=tmp_path, run_id="multi_ckpt_run")
        config = SFTTrainingConfig()
        summary = PEFTConfigFactory.summarize_adapter_config(config)

        for step in [50, 200, 100]:
            sdir = mgr.get_step_dir(step)
            mgr.save_checkpoint_metadata(
                target_dir=sdir,
                step=step,
                epoch=float(step) / 100.0,
                config=config,
                adapter_summary=summary,
                training_metrics={"loss": 1.0 / step},
            )

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 3
        assert [c[0] for c in ckpts] == [50, 100, 200]

        latest = mgr.find_latest_checkpoint()
        assert latest is not None
        assert latest.name == "checkpoint-200"
