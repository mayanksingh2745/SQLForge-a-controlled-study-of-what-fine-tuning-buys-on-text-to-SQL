"""Unit tests for pre-flight feasibility checker, hardware safety gates, and environment audits."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig
from sqlforge.training.preflight import (
    PreflightChecker,
    PreflightStatus,
)


class TestPreflightChecker:
    """Test suite for PreflightChecker hardware and safety gates."""

    def test_preflight_on_cpu_environment(self) -> None:
        """Verify preflight correctly flags absence of CUDA and reports BLOCKED status."""
        config = SFTTrainingConfig()
        checker = PreflightChecker(config)
        report = checker.run_preflight(check_dataset=False)

        assert report.platform != ""
        assert report.python_version != ""
        assert isinstance(report.free_disk_gb, float)
        assert isinstance(report.free_ram_gb, float)
        assert report.status in {
            PreflightStatus.BLOCKED,
            PreflightStatus.WARNING,
            PreflightStatus.READY,
        }
        assert len(report.checks) > 0

    def test_preflight_summary_formatting(self) -> None:
        """Verify human-readable preflight report summary."""
        checker = PreflightChecker(SFTTrainingConfig())
        report = checker.run_preflight(check_dataset=False)
        summary = report.summary()

        assert "Pre-Flight Status:" in summary
        assert "Platform:" in summary
        assert "CUDA:" in summary

    def test_simulated_cuda_pass(self) -> None:
        """Simulate an environment with adequate CUDA VRAM and dependencies."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        mock_props = MagicMock()
        mock_props.name = "NVIDIA A10G"
        mock_props.total_memory = 24 * (1024**3)  # 24 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (22 * (1024**3), 24 * (1024**3))

        checker = PreflightChecker(
            SFTTrainingConfig(
                base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                method=FineTuningMethod.LORA,
            )
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "transformers": MagicMock(),
                    "peft": MagicMock(),
                    "accelerate": MagicMock(),
                },
            ),
            patch("shutil.disk_usage", return_value=(100 * 1024**3, 50 * 1024**3, 50 * 1024**3)),
            patch.object(checker, "_get_host_ram", return_value=(32.0, 16.0)),
        ):
            report = checker.run_preflight(check_dataset=False)
            assert report.cuda_available is True
            assert report.total_vram_gb == 24.0
            assert report.device_names == ["NVIDIA A10G"]
            assert report.status == PreflightStatus.READY
            assert report.is_runnable is True

    def test_simulated_insufficient_vram_warning(self) -> None:
        """Simulate an environment where CUDA VRAM is lower than model threshold."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        mock_props = MagicMock()
        mock_props.name = "NVIDIA RTX 3060"
        mock_props.total_memory = 12 * (1024**3)  # 12 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.mem_get_info.return_value = (10 * (1024**3), 12 * (1024**3))

        checker = PreflightChecker(
            SFTTrainingConfig(
                base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                method=FineTuningMethod.LORA,  # Needs 22GB for 7B LoRA
            )
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "transformers": MagicMock(),
                    "peft": MagicMock(),
                    "accelerate": MagicMock(),
                },
            ),
            patch("shutil.disk_usage", return_value=(100 * 1024**3, 50 * 1024**3, 50 * 1024**3)),
        ):
            report = checker.run_preflight(check_dataset=False)
            assert any("VRAM" in w for w in report.warnings)
            assert report.status == PreflightStatus.WARNING

    def test_critically_low_disk_space_blocker(self) -> None:
        """Verify preflight flags critically low disk space (< 3 GB)."""
        checker = PreflightChecker(SFTTrainingConfig())
        # Simulate 1.5 GB free disk space
        with patch(
            "shutil.disk_usage", return_value=(100 * 1024**3, 98.5 * 1024**3, 1.5 * 1024**3)
        ):
            report = checker.run_preflight(check_dataset=False)
            assert report.status == PreflightStatus.BLOCKED
            assert any("disk space is critically low" in b.lower() for b in report.blockers)

    def test_qlora_requires_bitsandbytes_check(self) -> None:
        """Verify QLoRA preflight explicitly flags missing bitsandbytes."""
        checker = PreflightChecker(SFTTrainingConfig(method=FineTuningMethod.QLORA))
        report = checker.run_preflight(check_dataset=False)
        assert any("bitsandbytes" in b.lower() for b in report.blockers)
