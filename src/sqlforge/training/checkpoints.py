"""Safe checkpoint management, resumability, and audit metadata.

Handles deterministic checkpoint directory layouts, distinguishes lightweight adapter weights
from full optimizer states, records reproducibility metadata, and guards against accidental overwrites.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.reproducibility import get_git_metadata
from sqlforge.training.adapters import AdapterConfigSummary
from sqlforge.training.config import SFTTrainingConfig

logger = logging.getLogger(__name__)


class CheckpointType(BaseModel):
    """Categorization of checkpoint content."""

    ADAPTER_ONLY: str = "adapter_only"
    FULL_TRAINING_STATE: str = "full_training_state"
    MERGED_MODEL: str = "merged_model"


class CheckpointMetadata(BaseModel):
    """Immutable audit record stored with each saved checkpoint."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier")
    step: int = Field(..., ge=0, description="Global training step number")
    epoch: float = Field(..., ge=0.0, description="Training epoch reached")
    base_model_id: str = Field(..., description="Base pretrained model repository ID")
    base_model_revision: str = Field(..., description="Base model revision/hash")
    adapter_summary: AdapterConfigSummary = Field(
        ..., description="LoRA/QLoRA configuration summary"
    )
    training_metrics: dict[str, float] = Field(
        default_factory=dict, description="Loss, learning rate, and throughput metrics"
    )
    dataset_info: dict[str, Any] = Field(
        default_factory=dict, description="Dataset names, hashes, and split details"
    )
    seed: int = Field(..., description="Random seed")
    git_commit: str | None = Field(default=None, description="Repository Git commit hash")
    git_dirty: bool = Field(
        default=False, description="True if working tree had uncommitted modifications"
    )
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform_details: str = Field(
        default_factory=lambda: f"{platform.system()} {platform.release()} ({platform.machine()})"
    )
    checkpoint_type: str = Field(
        default="adapter_only", description="'adapter_only' or 'full_training_state'"
    )
    adapter_files: list[str] = Field(default_factory=list, description="Saved adapter file names")


class CheckpointManager:
    """Manages creation, layout, verification, and discovery of SFT model checkpoints."""

    def __init__(
        self,
        base_dir: str | Path = "artifacts/checkpoints",
        run_id: str = "sft_run",
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            base_dir: Root directory for checkpoints.
            run_id: Experiment run identifier to partition checkpoints.
        """
        self.base_dir = Path(base_dir)
        self.run_id = run_id
        self.run_dir = self.base_dir / self.run_id

    def get_step_dir(self, step: int) -> Path:
        """Get deterministic checkpoint directory for a specific training step."""
        return self.run_dir / f"checkpoint-{step}"

    def get_final_dir(self) -> Path:
        """Get deterministic checkpoint directory for the final completed adapter."""
        return self.run_dir / "final_adapter"

    def save_checkpoint_metadata(
        self,
        target_dir: Path,
        step: int,
        epoch: float,
        config: SFTTrainingConfig,
        adapter_summary: AdapterConfigSummary,
        training_metrics: dict[str, float],
        dataset_info: dict[str, Any] | None = None,
        checkpoint_type: str = "adapter_only",
        adapter_files: list[str] | None = None,
    ) -> Path:
        """Save structured CheckpointMetadata to target checkpoint directory.

        Args:
            target_dir: Path to directory containing checkpoint files.
            step: Global training step.
            epoch: Epoch number.
            config: SFTTrainingConfig instance.
            adapter_summary: AdapterConfigSummary instance.
            training_metrics: Training losses and learning rates.
            dataset_info: Dataset metadata and splits.
            checkpoint_type: 'adapter_only' or 'full_training_state'.
            adapter_files: List of file names in the checkpoint.

        Returns:
            Path to the saved checkpoint_metadata.json file.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_file = target_dir / "checkpoint_metadata.json"

        # Capture git status
        try:
            git_info = get_git_metadata()
            commit = git_info.get("commit")
            dirty = bool(git_info.get("is_dirty", False))
        except Exception:
            commit = None
            dirty = False

        metadata = CheckpointMetadata(
            run_id=self.run_id,
            step=step,
            epoch=epoch,
            base_model_id=config.base_model_id,
            base_model_revision=config.model_revision,
            adapter_summary=adapter_summary,
            training_metrics=training_metrics,
            dataset_info=dataset_info or {},
            seed=config.seed,
            git_commit=commit,
            git_dirty=dirty,
            checkpoint_type=checkpoint_type,
            adapter_files=adapter_files or [],
        )

        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        logger.info(f"Saved checkpoint metadata to {meta_file}")
        return meta_file

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """List all valid step checkpoints sorted by step number ascending."""
        if not self.run_dir.exists():
            return []

        checkpoints: list[tuple[int, Path]] = []
        for p in self.run_dir.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                try:
                    step_num = int(p.name.split("-")[1])
                    checkpoints.append((step_num, p))
                except (IndexError, ValueError):
                    continue

        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def find_latest_checkpoint(self) -> Path | None:
        """Find path to the most recent step checkpoint for resumption."""
        ckpts = self.list_checkpoints()
        if ckpts:
            return ckpts[-1][1]
        final = self.get_final_dir()
        if final.exists() and (final / "checkpoint_metadata.json").exists():
            return final
        return None

    @staticmethod
    def load_metadata(checkpoint_dir: str | Path) -> CheckpointMetadata:
        """Load CheckpointMetadata from a checkpoint directory.

        Args:
            checkpoint_dir: Directory containing checkpoint_metadata.json.

        Returns:
            Validated CheckpointMetadata instance.

        Raises:
            FileNotFoundError: If checkpoint_metadata.json is missing.
        """
        meta_path = Path(checkpoint_dir) / "checkpoint_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found at {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointMetadata.model_validate(data)
