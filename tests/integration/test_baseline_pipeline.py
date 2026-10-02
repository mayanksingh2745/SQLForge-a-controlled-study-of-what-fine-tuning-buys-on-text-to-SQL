"""Integration tests for baseline evaluation pipeline harness (EXP-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlforge.data.audit import LeakageContaminationError
from sqlforge.data.spider import load_spider_split, parse_spider_tables
from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.models.local_hf import LocalHFModelRunner
from sqlforge.models.mock import MockModelRunner
from sqlforge.pipeline.baseline import BaselinePipelineHarness
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata


@pytest.fixture
def spider_fixtures_dir() -> Path:
    """Path to version-controlled Spider fixtures."""
    root = Path(__file__).resolve().parent.parent.parent
    return root / "tests" / "fixtures" / "dataset" / "spider"


@pytest.fixture
def loaded_spider_fixtures(
    spider_fixtures_dir: Path,
) -> tuple[dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]]:
    """Load Spider tables, dev examples, and train examples."""
    tables_file = spider_fixtures_dir / "tables.json"
    dev_file = spider_fixtures_dir / "dev.json"
    train_file = spider_fixtures_dir / "train_spider.json"

    schemas = parse_spider_tables(tables_file)
    dev_examples = load_spider_split(dev_file, schemas, split=DatasetSplit.DEV)
    train_examples = load_spider_split(train_file, schemas, split=DatasetSplit.TRAIN)
    return schemas, dev_examples, train_examples


class TestBaselinePipelineHarness:
    """Integration test suite for BaselinePipelineHarness."""

    def test_dry_run_does_not_create_artifacts(
        self,
        tmp_path: Path,
        loaded_spider_fixtures: tuple[
            dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]
        ],
    ) -> None:
        schemas, dev_examples, train_examples = loaded_spider_fixtures
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        harness = BaselinePipelineHarness(tracker=tracker)

        result = harness.run(
            model_key="qwen25_coder_1_5b",
            eval_examples=dev_examples,
            schemas=schemas,
            train_examples=train_examples,
            k_shots=3,
            dry_run=True,
        )

        assert result.status == "dry_run"
        assert result.verified is True
        assert result.total_examples == len(dev_examples)
        assert result.k_shots == 3
        # No runs created under artifact directory
        assert not (tmp_path / "runs").exists() or list((tmp_path / "runs").iterdir()) == []

    def test_zero_shot_baseline_with_mock_runner(
        self,
        tmp_path: Path,
        loaded_spider_fixtures: tuple[
            dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]
        ],
    ) -> None:
        schemas, dev_examples, _ = loaded_spider_fixtures
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        mock_runner = MockModelRunner()
        harness = BaselinePipelineHarness(tracker=tracker, model_runner=mock_runner)

        result = harness.run(
            model_key="qwen25_coder_1_5b",
            eval_examples=dev_examples,
            schemas=schemas,
            k_shots=0,
            run_id="test_exp01_zeroshot_01",
        )

        assert result.status == "completed"
        assert result.run_id == "test_exp01_zeroshot_01"
        assert result.verified is True
        assert result.total_examples == len(dev_examples)

        # Inspect generated files in run directory
        run_dir = tmp_path / "runs" / "test_exp01_zeroshot_01"
        assert (run_dir / "generations.jsonl").is_file()
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "run_metadata.json").is_file()
        assert (run_dir / "manifest.json").is_file()

        # Strict manifest verification check
        verify_report = tracker.verify_run("test_exp01_zeroshot_01", strict=True)
        assert verify_report["verified"] is True

    def test_three_shot_bm25_retrieval_baseline(
        self,
        tmp_path: Path,
        loaded_spider_fixtures: tuple[
            dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]
        ],
    ) -> None:
        schemas, dev_examples, train_examples = loaded_spider_fixtures
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")

        # Injected runner that returns gold SQL to simulate accurate generation
        def smart_infer(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            assert "Question:" in prompt
            return "SELECT Name, Capacity FROM stadium ORDER BY Capacity DESC;", 80, 15

        runner = LocalHFModelRunner(
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            inference_fn=smart_infer,
        )
        harness = BaselinePipelineHarness(tracker=tracker, model_runner=runner)

        result = harness.run(
            model_key="qwen25_coder_7b",
            eval_examples=dev_examples,
            schemas=schemas,
            train_examples=train_examples,
            k_shots=3,
            run_id="test_exp01_threeshot_01",
        )

        assert result.status == "completed"
        assert result.k_shots == 3
        assert result.verified is True
        assert result.total_examples == len(dev_examples)

    def test_retrieval_isolation_leakage_rejection(
        self,
        tmp_path: Path,
        loaded_spider_fixtures: tuple[
            dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]
        ],
    ) -> None:
        schemas, dev_examples, _ = loaded_spider_fixtures
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        harness = BaselinePipelineHarness(tracker=tracker, model_runner=MockModelRunner())

        # Intentionally pass dev_examples (DEV split) as train_examples
        with pytest.raises(LeakageContaminationError, match="RETRIEVAL INDEX CONTAMINATION"):
            harness.run(
                model_key="qwen25_coder_7b",
                eval_examples=dev_examples,
                schemas=schemas,
                train_examples=dev_examples,  # Contaminated!
                k_shots=3,
            )

    def test_mid_pipeline_failure_preservation(
        self,
        tmp_path: Path,
        loaded_spider_fixtures: tuple[
            dict[str, SchemaMetadata], list[TextToSQLExample], list[TextToSQLExample]
        ],
    ) -> None:
        schemas, dev_examples, _ = loaded_spider_fixtures
        tracker = ExperimentTracker(base_artifact_dir=tmp_path / "runs")
        harness = BaselinePipelineHarness(tracker=tracker, model_runner=MockModelRunner())

        with pytest.raises(RuntimeError, match="Simulated mid-pipeline component failure"):
            harness.run(
                model_key="qwen25_coder_1_5b",
                eval_examples=dev_examples,
                schemas=schemas,
                k_shots=0,
                run_id="test_failure_run",
                fail_mode=True,
            )

        # Run directory and anomaly log must be preserved
        run_dir = tmp_path / "runs" / "test_failure_run"
        assert run_dir.is_dir()
        assert (run_dir / "eval_anomalies.json").is_file()
        with open(run_dir / "run_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["status"] == "failed"
