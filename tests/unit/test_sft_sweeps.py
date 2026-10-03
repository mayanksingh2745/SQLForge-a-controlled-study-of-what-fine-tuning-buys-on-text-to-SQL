"""Unit tests for LoRA hyperparameter and rank scaling sweeps (EXP-04)."""

from pathlib import Path
from typing import Any

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata
from sqlforge.training.config import SFTTrainingConfig
from sqlforge.training.sweeps import (
    LoRARankSweepOrchestrator,
    PreflightSafetyError,
    RankSweepConfig,
    SweepSummary,
)


@pytest.fixture
def sample_schema() -> dict[str, SchemaMetadata]:
    return {
        "academic": SchemaMetadata(
            db_id="academic",
            dialect="sqlite",
            tables=[
                TableMetadata(
                    table_name="author",
                    columns=[
                        ColumnMetadata(name="author_id", data_type="INTEGER", is_primary_key=True),
                        ColumnMetadata(name="name", data_type="TEXT"),
                    ],
                )
            ],
        )
    }


@pytest.fixture
def sample_examples() -> list[TextToSQLExample]:
    return [
        TextToSQLExample(
            id=f"spider_train_{i:04d}",
            question=f"What are the names of authors? (example {i})",
            db_id="academic",
            gold_sql="SELECT name FROM author;",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
        for i in range(1, 6)
    ]


class TestRankSweepConfig:
    """Test suite for RankSweepConfig validation and planning."""

    def test_default_config_initialization(self) -> None:
        cfg = RankSweepConfig()
        assert cfg.experiment_id == "EXP-04-RANK-SWEEP"
        assert cfg.ranks == [8, 16, 32, 64]
        assert cfg.seeds == [42]
        assert cfg.alpha_multiplier == 2.0
        assert cfg.get_alpha(8) == 16
        assert cfg.get_alpha(16) == 32
        assert cfg.get_alpha(32) == 64
        assert cfg.get_alpha(64) == 128

    def test_custom_alphas_override(self) -> None:
        cfg = RankSweepConfig(custom_alphas={8: 8, 16: 16, 32: 32, 64: 64})
        assert cfg.get_alpha(8) == 8
        assert cfg.get_alpha(16) == 16
        assert cfg.get_alpha(32) == 32
        assert cfg.get_alpha(64) == 64

    def test_invalid_ranks_and_seeds(self) -> None:
        with pytest.raises(ValueError, match="ranks list must contain at least one rank"):
            RankSweepConfig(ranks=[])

        with pytest.raises(ValueError, match="Rank dimension must be positive"):
            RankSweepConfig(ranks=[-4])

        with pytest.raises(ValueError, match="Seed must be non-negative"):
            RankSweepConfig(seeds=[-1])

    def test_parameter_estimation(self) -> None:
        cfg = RankSweepConfig(base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct")
        params_r8 = cfg.estimate_trainable_parameters(8, "all-linear")
        params_r16 = cfg.estimate_trainable_parameters(16, "all-linear")
        params_r32 = cfg.estimate_trainable_parameters(32, "all-linear")
        params_r64 = cfg.estimate_trainable_parameters(64, "all-linear")

        assert params_r8 > 0
        assert params_r16 == params_r8 * 2
        assert params_r32 == params_r16 * 2
        assert params_r64 == params_r32 * 2

        # Attention-only should have strictly fewer parameters than all-linear
        attn_r16 = cfg.estimate_trainable_parameters(16, "attention-only")
        assert attn_r16 < params_r16

        # Adapter size calculation in MB
        size_r16 = cfg.estimate_adapter_size_mb(16, "all-linear")
        assert size_r16 > 0.0

    def test_generate_run_configs_invariance(self) -> None:
        cfg = RankSweepConfig(
            ranks=[8, 16, 32],
            seeds=[42, 43],
            learning_rate=2e-4,
            epochs=3,
            per_device_batch_size=8,
            schema_format="ddl",
        )
        run_configs = cfg.generate_run_configs()
        assert len(run_configs) == 3 * 2  # 3 ranks * 2 seeds = 6 runs

        # Verify deterministic run_id naming and strict parameter invariance
        for run_cfg in run_configs:
            assert run_cfg.learning_rate == 2e-4
            assert run_cfg.epochs == 3
            assert run_cfg.per_device_batch_size == 8
            assert run_cfg.schema_format == "ddl"
            assert run_cfg.run_id is not None
            assert run_cfg.run_id.startswith("exp04_r")


class TestLoRARankSweepOrchestrator:
    """Test suite for LoRARankSweepOrchestrator execution and recovery."""

    def test_plan_sweep(self, tmp_path: Path) -> None:
        cfg = RankSweepConfig(
            ranks=[8, 16],
            seeds=[42],
            output_dir=tmp_path / "runs",
            sweep_dir=tmp_path / "sweeps",
        )
        orchestrator = LoRARankSweepOrchestrator(cfg)
        plans = orchestrator.plan_sweep()

        assert len(plans) == 2
        assert plans[0].rank == 8
        assert plans[0].alpha == 16
        assert plans[0].is_cached is False
        assert plans[1].rank == 16

    def test_dry_run_sweep_execution(
        self,
        tmp_path: Path,
        sample_schema: dict[str, SchemaMetadata],
        sample_examples: list[TextToSQLExample],
    ) -> None:
        cfg = RankSweepConfig(
            ranks=[8, 16],
            seeds=[42],
            output_dir=tmp_path / "runs",
            sweep_dir=tmp_path / "sweeps",
        )
        orchestrator = LoRARankSweepOrchestrator(cfg)
        summary = orchestrator.run_sweep(
            examples=sample_examples,
            schemas=sample_schema,
            dry_run=True,
            execute=False,
            resume=False,
        )

        assert summary.total_planned == 2
        assert summary.completed == 2
        assert summary.is_dry_run is True
        assert len(summary.results) == 2
        assert summary.results[0].status == "dry_run"
        assert summary.results[1].status == "dry_run"

        # Verify summary JSON file exists
        summary_file = cfg.sweep_dir / summary.sweep_id / "sweep_summary.json"
        assert summary_file.exists()

        # Load back
        loaded = SweepSummary.load_json(summary_file)
        assert loaded.sweep_id == summary.sweep_id
        assert len(loaded.results) == 2

    def test_safety_guard_prevents_unprompted_live_training(self) -> None:
        cfg = RankSweepConfig(ranks=[8, 16])
        orchestrator = LoRARankSweepOrchestrator(cfg)

        with pytest.raises(PreflightSafetyError, match="requires explicit confirmation"):
            orchestrator.run_sweep(dry_run=False, execute=False)

    def test_resume_skips_verified_cached_runs(
        self,
        tmp_path: Path,
        sample_schema: dict[str, SchemaMetadata],
        sample_examples: list[TextToSQLExample],
    ) -> None:
        cfg = RankSweepConfig(
            ranks=[8, 16],
            seeds=[42],
            output_dir=tmp_path / "runs",
            sweep_dir=tmp_path / "sweeps",
        )
        tracker = ExperimentTracker(base_artifact_dir=str(cfg.output_dir))
        orchestrator = LoRARankSweepOrchestrator(cfg, tracker=tracker)

        # 1. Run first time
        summary1 = orchestrator.run_sweep(
            examples=sample_examples,
            schemas=sample_schema,
            dry_run=True,
            execute=False,
            resume=True,
        )
        assert summary1.completed == 2
        assert summary1.cached == 0

        # 2. Run second time with resume=True -> should identify verified runs as cached
        summary2 = orchestrator.run_sweep(
            examples=sample_examples,
            schemas=sample_schema,
            dry_run=True,
            execute=False,
            resume=True,
        )
        assert summary2.cached == 2
        assert summary2.completed == 0
        assert summary2.results[0].status == "cached"

    def test_failure_handling_and_fail_fast(
        self,
        tmp_path: Path,
        sample_schema: dict[str, SchemaMetadata],
        sample_examples: list[TextToSQLExample],
    ) -> None:
        # Mock training callable that fails when rank == 16
        def failing_train_fn(
            c: SFTTrainingConfig,
            exs: list[TextToSQLExample],
            ckpt: Path,
        ) -> dict[str, Any]:
            if c.lora.rank == 16:
                raise RuntimeError("Simulated OOM on rank=16")
            return {"training_loss": 0.35, "total_steps": 10}

        cfg = RankSweepConfig(
            ranks=[8, 16, 32],
            seeds=[42],
            output_dir=tmp_path / "runs",
            sweep_dir=tmp_path / "sweeps",
        )
        tracker = ExperimentTracker(base_artifact_dir=str(cfg.output_dir))

        # Test fail_fast=False: should record failed run and continue
        orch = LoRARankSweepOrchestrator(cfg, tracker=tracker, train_fn=failing_train_fn)
        summary = orch.run_sweep(
            examples=sample_examples,
            schemas=sample_schema,
            dry_run=False,
            execute=True,
            resume=False,
            fail_fast=False,
        )

        assert summary.total_planned == 3
        assert summary.completed == 2
        assert summary.failed == 1
        failed_r16 = next(r for r in summary.results if r.rank == 16)
        assert failed_r16.status == "failed"
        assert "Simulated OOM on rank=16" in str(failed_r16.error_message)
