"""Mock pipeline harness orchestrator.

Coordinates configuration validation, deterministic fixtures, prompt construction,
mock inference, mock evaluation, and local experiment tracking.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sqlforge.data.fixtures import FixtureExample, load_fixture_dataset
from sqlforge.evaluation.mock_eval import MockEvaluator
from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.models.mock import MockModelRunner
from sqlforge.prompting.builder import PromptBuilder
from sqlforge.schemas.evaluation import GenerationResult
from sqlforge.schemas.experiments import ExperimentConfig
from sqlforge.settings import load_yaml_config

logger = logging.getLogger(__name__)


class MockPipelineResult(BaseModel):
    """Execution summary and verification report for a mock pipeline run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Run identifier")
    status: str = Field(..., description="Run outcome ('completed', 'failed', 'dry_run')")
    total_examples: int = Field(..., ge=0, description="Number of examples processed")
    artifact_dir: str = Field(..., description="Filesystem directory where artifacts are stored")
    verified: bool = Field(
        ..., description="Whether the run passed cryptographic manifest verification"
    )
    verification_details: dict[str, Any] = Field(
        default_factory=dict, description="Detailed manifest verification diagnostics"
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="Logged evaluation metrics")
    anomalies_count: int = Field(default=0, ge=0, description="Count of detected anomalies")


class MockPipelineHarness:
    """Orchestrator for deterministic mock pipeline runs."""

    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        model: MockModelRunner | None = None,
        evaluator: MockEvaluator | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.tracker = tracker or ExperimentTracker()
        self.model = model or MockModelRunner()
        self.evaluator = evaluator or MockEvaluator()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def run(
        self,
        config: Path | str | dict[str, Any] | ExperimentConfig | None = None,
        fixtures_path: Path | str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        fail_mode: bool = False,
    ) -> MockPipelineResult:
        """Execute the end-to-end mock pipeline.

        Args:
            config: Path to YAML config file, dict, or ExperimentConfig. If None, uses default mock config.
            fixtures_path: Path to fixture dataset JSON. If None, uses default mock fixtures.
            run_id: Explicit run ID. If None, auto-generated.
            dry_run: If True, validates config and fixtures without creating a run directory.
            fail_mode: If True, simulates a component failure to verify failure handling and artifact preservation.

        Returns:
            MockPipelineResult summarizing execution and verification.

        Raises:
            ValueError: If configuration or fixtures are invalid or missing.
            FileNotFoundError: If configuration file does not exist.
        """
        # Stage 1: Load and validate configuration
        validated_config = self._load_and_validate_config(config)

        # Stage 2: Load and validate fixture dataset
        examples = self._load_and_validate_fixtures(fixtures_path)

        # Stage 3: Dry-run check
        if dry_run:
            logger.info(
                "Dry-run mode: Configuration and fixtures validated successfully. No artifacts created."
            )
            return MockPipelineResult(
                run_id="dry_run",
                status="dry_run",
                total_examples=len(examples),
                artifact_dir="",
                verified=True,
                verification_details={"dry_run": True, "message": "Validation passed"},
                metrics={},
                anomalies_count=0,
            )

        # Stage 4: Initialize experiment run in tracker
        run_meta = self.tracker.init_run(
            experiment_name=validated_config.experiment_id,
            config=validated_config,
            seed=validated_config.seed,
            run_id=run_id,
        )
        active_run_id = run_meta.run_id
        run_dir = str(self.tracker.base_dir / active_run_id)

        try:
            # Stage 5: Inference generation
            generations: list[GenerationResult] = []
            for example in examples:
                if fail_mode and example.example_id.endswith("_003"):
                    raise RuntimeError("Simulated mid-pipeline component failure for verification.")

                prompt = self.prompt_builder.build_prompt(example)
                gen_result = self.model.generate(
                    prompt=prompt,
                    example_id=example.example_id,
                    gold_sql=example.gold_sql,
                )
                generations.append(gen_result)
                self.tracker.log_generation(active_run_id, gen_result)

            # Stage 6: Mock Evaluation
            metrics, anomalies = self.evaluator.evaluate(generations, examples)

            # Stage 7: Persist metrics & anomalies
            self.tracker.log_metrics(active_run_id, metrics)
            for anomaly in anomalies:
                self.tracker.log_anomaly(active_run_id, anomaly)

            # Stage 8: Finish run cleanly and compute manifest
            self.tracker.finish_run(active_run_id, status="completed")

            # Stage 9: Strict verification
            verification = self.tracker.verify_run(active_run_id, strict=True)

            return MockPipelineResult(
                run_id=active_run_id,
                status="completed",
                total_examples=len(examples),
                artifact_dir=run_dir,
                verified=bool(verification.get("verified", False)),
                verification_details=verification,
                metrics=metrics.model_dump(mode="json"),
                anomalies_count=len(anomalies),
            )

        except Exception as exc:
            logger.error("Pipeline failure in run '%s': %s", active_run_id, exc)
            # Preserve failure state and diagnostics
            with contextlib.suppress(Exception):
                self.tracker.log_anomaly(
                    active_run_id,
                    {
                        "error_code": "PIPELINE_ERROR",
                        "exception_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

            self.tracker.finish_run(active_run_id, status="failed")
            verification = self.tracker.verify_run(active_run_id, strict=False)

            return MockPipelineResult(
                run_id=active_run_id,
                status="failed",
                total_examples=len(examples),
                artifact_dir=run_dir,
                verified=bool(verification.get("verified", False)),
                verification_details=verification,
                metrics={},
                anomalies_count=1,
            )

    def _load_and_validate_config(
        self, config: Path | str | dict[str, Any] | ExperimentConfig | None
    ) -> ExperimentConfig:
        """Load and validate experiment configuration against ExperimentConfig."""
        if config is None:
            # Default mock pipeline config
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            default_config_path = project_root / "configs" / "mock_pipeline.yaml"
            if not default_config_path.exists():
                raise FileNotFoundError(f"Default mock config not found at: {default_config_path}")
            raw = load_yaml_config(default_config_path)
            try:
                return ExperimentConfig(**raw)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid default mock config at '{default_config_path}': {exc}"
                ) from exc

        if isinstance(config, ExperimentConfig):
            return config

        if isinstance(config, (str, Path)):
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"Experiment config file not found: {path}")
            raw = load_yaml_config(path)
            try:
                return ExperimentConfig(**raw)
            except ValidationError as exc:
                raise ValueError(f"Invalid experiment config in '{path}': {exc}") from exc

        if isinstance(config, dict):
            try:
                return ExperimentConfig(**config)
            except ValidationError as exc:
                raise ValueError(f"Invalid experiment config dict: {exc}") from exc

        raise ValueError(f"Unsupported config type: {type(config).__name__}")

    def _load_and_validate_fixtures(self, fixtures_path: Path | str | None) -> list[FixtureExample]:
        """Load and validate fixture dataset."""
        return load_fixture_dataset(fixtures_path)
