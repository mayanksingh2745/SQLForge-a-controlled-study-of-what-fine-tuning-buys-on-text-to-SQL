"""Unit tests for DataScalingExperimentOrchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP06_EXPERIMENT_ID,
    RankSelectionReference,
    UnresolvedRankSelectionError,
)
from sqlforge.training.scaling_orchestrator import (
    DataScalingExperimentOrchestrator,
)
from sqlforge.training.trainer import PreflightSafetyError


def test_orchestrator_exp05_plan_count(tmp_path: Path) -> None:
    """Verify orchestrator plans exactly 15 conditions for EXP-05."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference.create_provisional(rank=16)
    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP05_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
        output_dir=tmp_path / "checkpoints",
        scaling_dir=tmp_path / "scaling",
    )

    plans = orchestrator.plan_experiment()
    assert len(plans) == 15
    assert all(p.experiment_id == EXP05_EXPERIMENT_ID for p in plans)
    assert all(p.rank == 16 and p.alpha == 32 for p in plans)
    assert all(p.status == "planned" for p in plans)


def test_orchestrator_exp06_plan_count(tmp_path: Path) -> None:
    """Verify orchestrator plans exactly 9 conditions for EXP-06."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference.create_provisional(rank=16)
    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP06_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
        output_dir=tmp_path / "checkpoints",
        scaling_dir=tmp_path / "scaling",
    )

    plans = orchestrator.plan_experiment()
    assert len(plans) == 9
    assert all(p.experiment_id == EXP06_EXPERIMENT_ID for p in plans)
    assert all(p.total_examples == 3000 for p in plans)


def test_orchestrator_live_execution_guard_requires_execute_flag(tmp_path: Path) -> None:
    """Verify live training is rejected if execute=False and dry_run=False."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference(
        source_experiment_id="EXP-04-RANK-SWEEP",
        source_run_ids=["r1"],
        source_manifest_paths=["m1.json"],
        source_manifest_hashes=["a" * 64],
        selected_rank=16,
        selected_alpha=32,
        selected_target_modules=["q_proj", "v_proj"],
        selection_metric="acc",
        selection_rule="rule",
        selection_timestamp="2026-10-10T00:00:00Z",
        verification_status="verified",
    )
    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP05_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
    )
    with pytest.raises(
        PreflightSafetyError, match="requires explicit confirmation via execute=True"
    ):
        orchestrator.run_experiment(dry_run=False, execute=False)


def test_orchestrator_live_execution_rejects_provisional_rank(tmp_path: Path) -> None:
    """Verify live training rejects provisional rank selection."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference.create_provisional(rank=16)
    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP05_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
    )
    with pytest.raises(UnresolvedRankSelectionError, match="requires a verified EXP-04 rank"):
        orchestrator.run_experiment(dry_run=False, execute=True)


def test_orchestrator_dry_run_execution(tmp_path: Path) -> None:
    """Verify zero-compute dry-run executes cleanly and outputs summary."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference.create_provisional(rank=16)
    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP05_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
        output_dir=tmp_path / "checkpoints",
        scaling_dir=tmp_path / "scaling",
    )

    summary = orchestrator.run_experiment(dry_run=True)
    assert summary.is_dry_run is True
    assert summary.total_planned == 15
    assert summary.dry_run_count == 15
    assert summary.completed == 0
    assert summary.failed == 0
    assert summary.empirical_execution_allowed is False
    assert len(summary.results) == 15
    assert all(r.status == "dry_run" for r in summary.results)
    assert all(r.final_loss is None for r in summary.results)
    assert summary.output_path is not None
    assert Path(summary.output_path).exists()


def test_orchestrator_mock_training_execution(tmp_path: Path) -> None:
    """Verify live execution works with mock training callable and empirical rank."""
    tracker = ExperimentTracker(base_artifact_dir=tmp_path / "experiments")
    ref = RankSelectionReference(
        source_experiment_id="EXP-04-RANK-SWEEP",
        source_run_ids=["exp04_r16_s42"],
        source_manifest_paths=["manifest.json"],
        source_manifest_hashes=["a" * 64],
        selected_rank=16,
        selected_alpha=32,
        selected_target_modules=["q_proj", "v_proj"],
        selection_metric="acc",
        selection_rule="rule",
        selection_timestamp="2026-10-10T00:00:00Z",
        verification_status="verified",
    )

    def mock_train(cfg: Any, sft_examples: Any, final_dir: Path) -> dict[str, Any]:
        return {"total_steps": 50, "final_loss": 0.35, "metrics": {"loss": 0.35}}

    def mock_eval(ckpt: str, dataset: str) -> dict[str, Any]:
        return {"execution_accuracy": 0.72, "valid_sql_rate": 0.98, "exact_match": 0.65}

    orchestrator = DataScalingExperimentOrchestrator(
        experiment_id=EXP05_EXPERIMENT_ID,
        rank_selection=ref,
        tracker=tracker,
        output_dir=tmp_path / "checkpoints",
        scaling_dir=tmp_path / "scaling",
        train_fn=mock_train,
        eval_fn=mock_eval,
    )

    summary = orchestrator.run_experiment(dry_run=False, execute=True)
    assert summary.is_dry_run is False
    assert summary.completed == 15
    assert summary.failed == 0
    assert len(summary.results) == 15
    assert summary.results[0].status == "completed"
    assert summary.results[0].final_loss == 0.35
    assert summary.results[0].total_steps == 50
    assert summary.results[0].metrics["execution_accuracy"] == 0.72
