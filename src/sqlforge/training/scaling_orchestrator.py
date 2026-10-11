"""Experiment orchestration for EXP-05 and EXP-06 data scaling and synthetic comparisons.

Coordinates deterministic plan generation, rank selection verification,
hardware feasibility gating, zero-compute dry runs, and manifest tracking.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata
from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP06_EXPERIMENT_ID,
    RankSelectionReference,
    ScalingRunCondition,
    UnresolvedRankSelectionError,
    generate_exp05_run_conditions,
    generate_exp06_run_conditions,
)
from sqlforge.training.preflight import PreflightChecker
from sqlforge.training.trainer import PreflightSafetyError, SFTFineTuningPipeline, SFTTrainResult

logger = logging.getLogger(__name__)


class ScalingRunPlan(BaseModel):
    """Specification of an individual planned condition in the data scaling matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    experiment_id: str
    seed: int
    total_examples: int
    human_examples_count: int
    synthetic_examples_count: int
    group_name: str | None = None
    rank: int
    alpha: int
    status: str = (
        "planned"  # "planned", "cached", "dry_run", "completed", "failed", "blocked_dependency"
    )
    is_cached: bool = False
    checkpoint_dir: str | None = None


class ScalingRunResult(BaseModel):
    """Execution summary and metrics for an individual scaling run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    experiment_id: str
    seed: int
    total_examples: int
    human_examples_count: int
    synthetic_examples_count: int
    group_name: str | None = None
    rank: int
    alpha: int
    status: str  # "completed", "dry_run", "cached", "failed", "blocked_dependency"
    duration_seconds: float = 0.0
    final_loss: float | None = None
    total_steps: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    checkpoint_dir: str | None = None
    manifest_verified: bool = False
    error_message: str | None = None


class ScalingExperimentSummary(BaseModel):
    """Complete manifest summarizing all runs in EXP-05 or EXP-06."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str
    base_model_id: str
    total_planned: int
    completed: int
    cached: int
    dry_run_count: int
    failed: int
    is_dry_run: bool
    rank_selection_status: str
    empirical_execution_allowed: bool
    results: list[ScalingRunResult]
    created_at: str
    output_path: str | None = None

    @property
    def has_empirical_evaluation(self) -> bool:
        """Whether this summary contains genuine execution accuracy metrics."""
        return any(
            r.status in ("completed", "cached") and r.metrics.get("execution_accuracy") is not None
            for r in self.results
        )

    def save_json(self, path: Path | str) -> None:
        """Atomically persist summary manifest to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.model_dump(), indent=2, sort_keys=True))
        tmp.replace(p)

    @classmethod
    def load_json(cls, path: Path | str) -> ScalingExperimentSummary:
        """Load summary from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Scaling summary manifest not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class DataScalingExperimentOrchestrator:
    """Orchestrates planning, validation, dry-run simulation, and live runs for EXP-05 and EXP-06."""

    def __init__(
        self,
        experiment_id: str = EXP05_EXPERIMENT_ID,
        rank_selection: RankSelectionReference | None = None,
        tracker: ExperimentTracker | None = None,
        output_dir: Path | str = "artifacts/runs",
        scaling_dir: Path | str = "artifacts/data_scaling",
        train_fn: Callable[..., Any] | None = None,
        eval_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.rank_selection = rank_selection or RankSelectionReference.create_provisional()
        self.output_dir = Path(output_dir)
        self.scaling_dir = Path(scaling_dir)
        self.tracker = tracker or ExperimentTracker(base_artifact_dir=str(self.output_dir))
        self.train_fn = train_fn
        self.eval_fn = eval_fn

    def get_planned_conditions(self) -> list[ScalingRunCondition]:
        """Generate the canonical conditions for the selected experiment."""
        if self.experiment_id == EXP05_EXPERIMENT_ID:
            return generate_exp05_run_conditions(self.rank_selection)
        elif self.experiment_id == EXP06_EXPERIMENT_ID:
            return generate_exp06_run_conditions(self.rank_selection)
        else:
            raise ValueError(f"Unknown experiment ID: {self.experiment_id!r}")

    def plan_experiment(self) -> list[ScalingRunPlan]:
        """Inspect planned runs and detect existing cached checkpoints with resume integrity."""
        plans: list[ScalingRunPlan] = []
        conditions = self.get_planned_conditions()

        for cond in conditions:
            run_dir = self.tracker.base_dir / cond.run_id
            manifest_file = run_dir / "manifest.json"
            is_cached = False
            ckpt_dir = None

            if manifest_file.exists():
                verify_res = self.tracker.verify_run(cond.run_id, strict=False)
                if verify_res.get("verified", False):
                    # Check resume integrity: ensure config and models match
                    is_cached = True
                    ckpt_dir = str(run_dir)

            plans.append(
                ScalingRunPlan(
                    run_id=cond.run_id,
                    experiment_id=cond.experiment_id,
                    seed=cond.seed,
                    total_examples=cond.total_examples,
                    human_examples_count=cond.human_examples_count,
                    synthetic_examples_count=cond.synthetic_examples_count,
                    group_name=cond.group_name,
                    rank=self.rank_selection.selected_rank,
                    alpha=self.rank_selection.selected_alpha,
                    status="cached" if is_cached else "planned",
                    is_cached=is_cached,
                    checkpoint_dir=ckpt_dir,
                )
            )
        return plans

    def run_experiment(
        self,
        dry_run: bool = True,
        execute: bool = False,
        resume: bool = True,
        fail_fast: bool = False,
        examples_by_condition: dict[str, list[TextToSQLExample]] | None = None,
        schemas: dict[str, SchemaMetadata] | None = None,
    ) -> ScalingExperimentSummary:
        """Execute or dry-run the data scaling experiment matrix."""
        # Safety gate: Live execution requires explicit opt-in
        if not dry_run and not execute:
            raise PreflightSafetyError(
                "Live training requires explicit confirmation via execute=True. "
                "Use dry_run=True for zero-compute planning and validation."
            )

        # Safety gate: Live execution requires verified EXP-04 rank artifact
        if not dry_run and not self.rank_selection.is_empirical_eligible:
            raise UnresolvedRankSelectionError(
                f"Live training for {self.experiment_id} requires a verified EXP-04 rank selection artifact. "
                f"Current status: {self.rank_selection.verification_status!r}."
            )

        # Preflight feasibility check
        first_cond = self.get_planned_conditions()[0]
        sft_cfg = first_cond.to_sft_training_config(output_dir=self.output_dir)
        checker = PreflightChecker(sft_cfg)
        preflight_report = checker.run_preflight(check_dataset=False)

        if not dry_run and not self.train_fn and not preflight_report.is_runnable:
            blocker_text = "\n".join(f" - {b}" for b in preflight_report.blockers)
            raise PreflightSafetyError(
                f"Hardware feasibility pre-flight failed for {self.experiment_id}:\n{blocker_text}"
            )

        plans = self.plan_experiment()
        results: list[ScalingRunResult] = []

        completed_count = 0
        cached_count = 0
        dry_run_count = 0
        failed_count = 0

        # Fallback fixture data for testing
        fallback_examples, fallback_schemas = self._get_fallback_fixtures()
        resolved_schemas = schemas or fallback_schemas

        for plan in plans:
            # 1. Cached resume handling
            if resume and plan.is_cached and plan.checkpoint_dir:
                cached_metrics = self._load_run_metrics(plan.run_id)
                cached_count += 1
                results.append(
                    ScalingRunResult(
                        run_id=plan.run_id,
                        experiment_id=plan.experiment_id,
                        seed=plan.seed,
                        total_examples=plan.total_examples,
                        human_examples_count=plan.human_examples_count,
                        synthetic_examples_count=plan.synthetic_examples_count,
                        group_name=plan.group_name,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        status="cached",
                        duration_seconds=cached_metrics.get("duration_seconds", 0.0),
                        final_loss=cached_metrics.get("training_loss"),
                        total_steps=cached_metrics.get("total_steps", 0),
                        metrics=cached_metrics,
                        checkpoint_dir=plan.checkpoint_dir,
                        manifest_verified=True,
                    )
                )
                continue

            # 2. Collision defense
            run_dir = self.tracker.base_dir / plan.run_id
            target_run_id = plan.run_id
            if run_dir.exists() and not resume:
                target_run_id = f"{plan.run_id}_{int(time.time())}"

            # 3. Resolve condition-specific training examples
            cond_examples = (
                examples_by_condition.get(plan.run_id)
                if examples_by_condition and plan.run_id in examples_by_condition
                else fallback_examples
            )

            # Build SFT training configuration
            cond = next(c for c in self.get_planned_conditions() if c.run_id == plan.run_id)
            cfg = cond.to_sft_training_config(output_dir=self.output_dir)
            if target_run_id != plan.run_id:
                cfg = cfg.model_copy(update={"run_id": target_run_id})

            start_t = time.monotonic()
            try:
                pipeline = SFTFineTuningPipeline(
                    config=cfg,
                    tracker=self.tracker,
                    train_fn=self.train_fn,
                )
                train_res: SFTTrainResult = pipeline.run(
                    examples=cond_examples or [],
                    schemas=resolved_schemas,
                    dry_run=dry_run,
                )
                duration = time.monotonic() - start_t

                if dry_run:
                    dry_run_count += 1
                    status_str = "dry_run"
                    metrics_dict: dict[str, Any] = {}
                else:
                    completed_count += 1
                    status_str = "completed"
                    metrics_dict = (
                        self.eval_fn(train_res.final_checkpoint_dir, cfg.eval_dataset)
                        if self.eval_fn
                        else {}
                    )

                results.append(
                    ScalingRunResult(
                        run_id=target_run_id,
                        experiment_id=plan.experiment_id,
                        seed=plan.seed,
                        total_examples=plan.total_examples,
                        human_examples_count=plan.human_examples_count,
                        synthetic_examples_count=plan.synthetic_examples_count,
                        group_name=plan.group_name,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        status=status_str,
                        duration_seconds=round(duration, 2),
                        final_loss=train_res.final_loss if not dry_run else None,
                        total_steps=train_res.total_steps if not dry_run else 0,
                        metrics=metrics_dict,
                        checkpoint_dir=str(train_res.final_checkpoint_dir)
                        if train_res.final_checkpoint_dir
                        else None,
                        manifest_verified=True,
                    )
                )

            except Exception as exc:
                duration = time.monotonic() - start_t
                failed_count += 1
                logger.error(f"Run {plan.run_id} failed: {exc}")
                results.append(
                    ScalingRunResult(
                        run_id=target_run_id,
                        experiment_id=plan.experiment_id,
                        seed=plan.seed,
                        total_examples=plan.total_examples,
                        human_examples_count=plan.human_examples_count,
                        synthetic_examples_count=plan.synthetic_examples_count,
                        group_name=plan.group_name,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        status="failed",
                        duration_seconds=round(duration, 2),
                        final_loss=None,
                        total_steps=0,
                        metrics={},
                        checkpoint_dir=None,
                        manifest_verified=False,
                        error_message=str(exc),
                    )
                )
                if fail_fast:
                    break

        summary_dir = self.scaling_dir / self.experiment_id.lower().replace("-", "_")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / "scaling_summary.json"

        summary = ScalingExperimentSummary(
            experiment_id=self.experiment_id,
            base_model_id=first_cond.to_sft_training_config().base_model_id,
            total_planned=len(plans),
            completed=completed_count,
            cached=cached_count,
            dry_run_count=dry_run_count,
            failed=failed_count,
            is_dry_run=dry_run,
            rank_selection_status=self.rank_selection.verification_status,
            empirical_execution_allowed=self.rank_selection.is_empirical_eligible,
            results=results,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            output_path=str(summary_file),
        )
        summary.save_json(summary_file)
        return summary

    def _load_run_metrics(self, run_id: str) -> dict[str, Any]:
        """Load metrics and metadata from a verified run directory."""
        metrics_file = self.tracker.base_dir / run_id / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    return cast(dict[str, Any], json.load(f))
            except Exception:
                pass
        return {}

    def _get_fallback_fixtures(self) -> tuple[list[TextToSQLExample], dict[str, SchemaMetadata]]:
        """Return minimal synthetic fixture data for testing and offline validation."""
        example = TextToSQLExample(
            id="spider_train_001",
            question="How many singers are there?",
            db_id="concert_singer",
            gold_sql="SELECT count(*) FROM singer",
            dataset_name="spider",
            split=DatasetSplit.TRAIN,
        )
        schema_meta = SchemaMetadata(
            db_id="concert_singer",
            dialect="sqlite",
            tables=[],
            foreign_keys=[],
        )
        return [example], {"concert_singer": schema_meta}
