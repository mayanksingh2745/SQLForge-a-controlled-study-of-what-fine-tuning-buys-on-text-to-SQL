"""Integration tests for SFTFineTuningPipeline, dry-run, mock execution, and artifact verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata
from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig
from sqlforge.training.data import SFTExample
from sqlforge.training.trainer import PreflightSafetyError, SFTFineTuningPipeline


@pytest.fixture
def integration_schema() -> dict[str, SchemaMetadata]:
    """Test schema mapping."""
    return {
        "company": SchemaMetadata(
            db_id="company",
            dialect="sqlite",
            tables=[
                TableMetadata(
                    table_name="employee",
                    columns=[
                        ColumnMetadata(name="emp_id", data_type="INTEGER", is_primary_key=True),
                        ColumnMetadata(name="name", data_type="TEXT"),
                        ColumnMetadata(name="salary", data_type="INTEGER"),
                    ],
                )
            ],
        )
    }


@pytest.fixture
def integration_train_examples() -> list[TextToSQLExample]:
    """Test training examples."""
    return [
        TextToSQLExample(
            id=f"spider_train_{i:03d}",
            question=f"List all employee names with salary > {1000 * i}?",
            db_id="company",
            gold_sql=f"SELECT name FROM employee WHERE salary > {1000 * i};",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
        for i in range(1, 6)
    ]


class TestSFTFineTuningPipeline:
    """Integration test suite for SFTFineTuningPipeline."""

    def test_dry_run_pipeline_execution(
        self,
        tmp_path: Path,
        integration_schema: dict[str, SchemaMetadata],
        integration_train_examples: list[TextToSQLExample],
    ) -> None:
        """Verify dry-run completes, formats data, creates dry-run checkpoint, and verifies manifest."""
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        config = SFTTrainingConfig(
            run_id="test_dry_run_01",
            method=FineTuningMethod.LORA,
            output_dir=str(tmp_path / "checkpoints"),
        )
        pipeline = SFTFineTuningPipeline(config=config, tracker=tracker)

        result = pipeline.run(
            examples=integration_train_examples,
            schemas=integration_schema,
            dry_run=True,
        )

        assert result.status == "dry_run"
        assert result.total_steps == 0
        assert result.data_summary.valid_examples_count == 5
        assert result.manifest_verified is True
        assert Path(result.final_checkpoint_dir).exists()
        assert (Path(result.final_checkpoint_dir) / "checkpoint_metadata.json").exists()
        assert (Path(result.final_checkpoint_dir) / "adapter_config.json").exists()

    def test_mock_training_execution(
        self,
        tmp_path: Path,
        integration_schema: dict[str, SchemaMetadata],
        integration_train_examples: list[TextToSQLExample],
    ) -> None:
        """Verify simulated training callable saves checkpoints, logs metrics, and verifies manifest."""
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        config = SFTTrainingConfig(
            run_id="test_mock_train_01",
            method=FineTuningMethod.LORA,
            epochs=2,
            output_dir=str(tmp_path / "checkpoints"),
        )

        def mock_train(
            cfg: SFTTrainingConfig, sft_exs: list[SFTExample], out_dir: Path
        ) -> dict[str, Any]:
            # Simulate saving adapter weights
            (out_dir / "adapter_model.safetensors").write_bytes(b"WEIGHTS")
            return {
                "total_steps": 50,
                "final_loss": 0.38,
                "metrics": {"loss": 0.38, "lr": 1e-4},
            }

        pipeline = SFTFineTuningPipeline(
            config=config,
            tracker=tracker,
            train_fn=mock_train,
        )

        result = pipeline.run(
            examples=integration_train_examples,
            schemas=integration_schema,
            dry_run=False,
        )

        assert result.status == "completed"
        assert result.total_steps == 50
        assert result.final_loss == 0.38
        assert result.manifest_verified is True
        assert (Path(result.final_checkpoint_dir) / "adapter_model.safetensors").exists()
        assert (Path(result.final_checkpoint_dir) / "checkpoint_metadata.json").exists()

    def test_preflight_safety_halt_on_cpu(
        self,
        tmp_path: Path,
        integration_schema: dict[str, SchemaMetadata],
        integration_train_examples: list[TextToSQLExample],
    ) -> None:
        """Verify that live execution on CPU-only environment halts immediately via PreflightSafetyError."""
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        config = SFTTrainingConfig(
            run_id="test_safety_halt_01",
            method=FineTuningMethod.LORA,
            output_dir=str(tmp_path / "checkpoints"),
        )
        # Without mock_train_fn, in an environment without CUDA, should raise PreflightSafetyError
        pipeline = SFTFineTuningPipeline(config=config, tracker=tracker)

        with pytest.raises(PreflightSafetyError, match="pre-flight check failed"):
            pipeline.run(
                examples=integration_train_examples,
                schemas=integration_schema,
                dry_run=False,
            )
