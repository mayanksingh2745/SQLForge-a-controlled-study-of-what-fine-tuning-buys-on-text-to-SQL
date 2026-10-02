"""Frontier and open-weight model baseline evaluation pipeline harness (EXP-01).

Implements the standard experimental baseline protocol:
1. Qwen2.5-Coder-1.5B-Instruct — zero-shot.
2. Qwen2.5-Coder-7B-Instruct — zero-shot.
3. Qwen2.5-Coder-7B-Instruct — three-shot BM25 retrieval.
4. GPT-4o mini (`gpt-4o-mini-2024-07-18`) — zero-shot and three-shot BM25 retrieval.

Safeguards:
- Strict training-only BM25 retrieval isolation with `IsolationGuard`.
- Hard cumulative API spending limit capped at <= $50.00 USD.
- Explicit opt-in requirement (`allow_live_api=True`) for live API calls.
- Deterministic greedy decoding (temperature=0.0, max_tokens=512).
- Denominator consistency: every eligible example is accounted for in N.
- 95% bootstrap confidence intervals for primary execution accuracy.
- Cryptographic manifest verification via `ExperimentTracker`.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.evaluation.comparator import ExecutionComparator
from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.models.base import ModelRunner
from sqlforge.models.local_hf import LocalHFModelRunner
from sqlforge.models.openai_runner import OpenAIRunner
from sqlforge.prompting.engine import PromptEngine
from sqlforge.prompting.retriever import BM25Retriever
from sqlforge.prompting.serializers import get_serializer
from sqlforge.schemas.evaluation import EvaluationMetrics, GenerationResult
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.experiments import ExperimentConfig, PromptConfig
from sqlforge.schemas.metadata import SchemaMetadata

logger = logging.getLogger(__name__)

# Canonical model aliases mapped to resolved model IDs and families
MODEL_CONFIG_MAP: dict[str, dict[str, Any]] = {
    "qwen25_coder_1_5b": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "family": "qwen",
        "is_api": False,
    },
    "qwen25_coder_7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "family": "qwen",
        "is_api": False,
    },
    "frontier_api_ref": {
        "model_id": "gpt-4o-mini-2024-07-18",
        "family": "openai",
        "is_api": True,
    },
}


class BaselineRunResult(BaseModel):
    """Complete summary and audit report for a baseline evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique experiment run ID")
    experiment_id: str = Field(..., description="Experiment identifier (e.g. 'EXP-01-ZEROSHOT')")
    model_id: str = Field(..., description="Model identifier used for inference")
    k_shots: int = Field(..., ge=0, description="Few-shot demonstration count")
    status: str = Field(..., description="Run status ('completed', 'failed', 'dry_run')")
    total_examples: int = Field(..., ge=0, description="Total number of evaluated examples (N)")
    artifact_dir: str = Field(
        ..., description="Filesystem directory where run artifacts are stored"
    )
    verified: bool = Field(..., description="Whether cryptographic manifest verification passed")
    verification_details: dict[str, Any] = Field(
        default_factory=dict, description="Diagnostics from manifest verification"
    )
    metrics: EvaluationMetrics | dict[str, Any] = Field(
        default_factory=dict, description="Primary and secondary evaluation metrics"
    )
    anomalies_count: int = Field(
        default=0, ge=0, description="Total number of recorded execution anomalies"
    )
    total_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Cumulative dollar expenditure incurred by API runner"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total prompt and completion tokens processed"
    )


class BaselinePipelineHarness:
    """Orchestrator for baseline model evaluations across open-weight and frontier models."""

    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        model_runner: ModelRunner | None = None,
        comparator: ExecutionComparator | None = None,
    ) -> None:
        """Initialize the baseline harness.

        Args:
            tracker: ExperimentTracker instance. Defaults to default tracker.
            model_runner: Injected ModelRunner instance (e.g. for offline testing or mock runners).
            comparator: ExecutionComparator for running and scoring SQL queries.
        """
        self.tracker = tracker or ExperimentTracker()
        self.injected_runner = model_runner
        self.comparator = comparator or ExecutionComparator()

    def run(
        self,
        model_key: str,
        eval_examples: list[TextToSQLExample],
        schemas: dict[str, SchemaMetadata],
        train_examples: list[TextToSQLExample] | None = None,
        k_shots: int = 0,
        schema_format: str = "ddl",
        db_root: Path | str | None = None,
        max_examples: int | None = None,
        experiment_id: str = "EXP-01-ZEROSHOT",
        run_id: str | None = None,
        seed: int = 42,
        dry_run: bool = False,
        allow_live_api: bool = False,
        api_spending_limit: float = 50.0,
        openai_api_key: str | None = None,
        fail_mode: bool = False,
    ) -> BaselineRunResult:
        """Execute a baseline evaluation run.

        Args:
            model_key: Key identifying model ('qwen25_coder_1_5b', 'qwen25_coder_7b', 'frontier_api_ref').
            eval_examples: List of TextToSQLExample instances to evaluate (must be DEV or TEST).
            schemas: Mapping of db_id -> SchemaMetadata.
            train_examples: Optional training partition examples for BM25 retrieval (must be TRAIN only).
            k_shots: Number of few-shot demonstrations (0 for zero-shot, 3 for 3-shot BM25).
            schema_format: Schema representation format ('ddl', 'compact', 'json').
            db_root: Optional path to directory containing SQLite database files.
            max_examples: Optional limit on number of examples to evaluate.
            experiment_id: Identifier for the experiment matrix entry.
            run_id: Optional explicit run ID.
            seed: Random seed for reproducibility.
            dry_run: If True, validates configuration, schemas, and isolation without generating or writing run files.
            allow_live_api: Explicit opt-in flag required for live OpenAI API inference.
            api_spending_limit: Hard cumulative dollar ceiling for OpenAI API (<= $50.00).
            openai_api_key: Optional explicit API key override (never logged).
            fail_mode: Simulate a failure mid-run for testing failure recovery.

        Returns:
            BaselineRunResult summarizing run execution, metrics, and verification.

        Raises:
            ValueError: If configuration, parameters, or retrieval isolation rules are violated.
        """
        # Step 1: Validate inputs and model configuration
        if not eval_examples:
            raise ValueError("eval_examples list cannot be empty.")
        if not schemas:
            raise ValueError("schemas mapping cannot be empty.")
        if k_shots < 0:
            raise ValueError(f"k_shots must be non-negative, got {k_shots}")

        # Limit examples if requested
        active_examples = (
            eval_examples[:max_examples] if max_examples is not None else eval_examples
        )

        # Step 2: Initialize PromptEngine and BM25Retriever (with isolation enforcement)
        retriever: BM25Retriever | None = None
        if k_shots > 0:
            if not train_examples:
                raise ValueError(
                    f"train_examples must be provided when k_shots > 0 (requested k_shots={k_shots})."
                )
            # BM25Retriever constructor enforces IsolationGuard.assert_retrieval_isolation
            retriever = BM25Retriever(
                examples=train_examples,
                forbidden_splits={DatasetSplit.DEV, DatasetSplit.TEST, DatasetSplit.HELD_OUT},
            )

        prompt_engine = PromptEngine(
            schema_format=schema_format,
            serializer=get_serializer(schema_format),
            retriever=retriever,
            default_max_tokens=4096,
        )

        # Step 3: Resolve Model Runner
        resolved_model_id: str
        runner: ModelRunner

        if self.injected_runner is not None:
            runner = self.injected_runner
            resolved_model_id = getattr(runner, "model_id", model_key)
        else:
            model_info = MODEL_CONFIG_MAP.get(model_key)
            if model_info is None:
                raise ValueError(
                    f"Unknown model_key '{model_key}'. Supported keys: {list(MODEL_CONFIG_MAP.keys())}"
                )
            resolved_model_id = model_info["model_id"]

            if model_info["is_api"]:
                runner = OpenAIRunner(
                    model_id=resolved_model_id,
                    api_key=openai_api_key,
                    allow_live_api=allow_live_api,
                    max_cumulative_budget_usd=api_spending_limit,
                    temperature=0.0,
                    max_new_tokens=512,
                )
            else:
                runner = LocalHFModelRunner(
                    model_id=resolved_model_id,
                    temperature=0.0,
                    max_new_tokens=512,
                )

        # Step 4: Handle Dry Run
        if dry_run:
            logger.info("Executing dry-run validation for model '%s' (k=%d).", model_key, k_shots)
            # Verify prompt assembly on the first example
            first_ex = active_examples[0]
            if first_ex.db_id not in schemas:
                raise ValueError(f"Schema not found for db_id '{first_ex.db_id}'")
            sample_prompt = prompt_engine.assemble(
                example=first_ex,
                schema=schemas[first_ex.db_id],
                k_shots=k_shots,
            )
            return BaselineRunResult(
                run_id="dry_run",
                experiment_id=experiment_id,
                model_id=resolved_model_id,
                k_shots=k_shots,
                status="dry_run",
                total_examples=len(active_examples),
                artifact_dir="",
                verified=True,
                verification_details={
                    "dry_run": True,
                    "prompt_sample_len": len(sample_prompt.prompt_text),
                    "prompt_token_estimate": sample_prompt.token_count_estimate,
                },
                metrics={},
                anomalies_count=0,
                total_cost_usd=0.0,
                total_tokens=0,
            )

        # Step 5: Initialize Experiment Run
        exp_config = ExperimentConfig(
            experiment_id=experiment_id,
            description=f"Baseline evaluation for {resolved_model_id} (k_shots={k_shots})",
            seed=seed,
            model_id=resolved_model_id,
            paradigm="zero-shot" if k_shots == 0 else "few-shot",
            prompt=PromptConfig(
                schema_format=schema_format,
                k_shots=k_shots,
                retriever_type="bm25" if k_shots > 0 else None,
            ),
        )

        run_meta = self.tracker.init_run(
            experiment_name=experiment_id,
            config=exp_config,
            seed=seed,
            run_id=run_id,
        )
        active_run_id = run_meta.run_id
        run_dir = str(self.tracker.base_dir / active_run_id)

        # Step 6: Execute Model Inference and Stream Generations
        generations: list[GenerationResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0

        try:
            for idx, example in enumerate(active_examples):
                if fail_mode and idx >= 1:
                    raise RuntimeError("Simulated mid-pipeline component failure for verification.")

                schema = schemas.get(example.db_id)
                if schema is None:
                    raise ValueError(
                        f"Missing SchemaMetadata for db_id '{example.db_id}' at example '{example.id}'"
                    )

                assembled = prompt_engine.assemble(
                    example=example,
                    schema=schema,
                    k_shots=k_shots,
                )

                gen_result = runner.generate(
                    prompt=assembled.prompt_text,
                    example_id=example.id,
                    gold_sql=example.gold_sql,
                )
                generations.append(gen_result)
                self.tracker.log_generation(active_run_id, gen_result)

                total_prompt_tokens += gen_result.prompt_tokens
                total_completion_tokens += gen_result.completion_tokens
                if gen_result.cost_usd is not None:
                    total_cost_usd += gen_result.cost_usd

            # Step 7: Evaluate Generations with ExecutionComparator
            comparator = ExecutionComparator(
                db_root=db_root,
                timeout_seconds=10.0,
            )
            metrics = comparator.evaluate(
                generations=generations,
                gold_examples=active_examples,
            )

            # Update metrics with token and cost totals
            metrics_dict = metrics.model_dump()
            metrics_dict["total_prompt_tokens"] = total_prompt_tokens
            metrics_dict["total_completion_tokens"] = total_completion_tokens
            metrics_dict["total_cost_usd"] = round(total_cost_usd, 6)

            # Step 8: Persist Metrics and Anomalies
            self.tracker.log_metrics(active_run_id, metrics)
            anomalies_count = 0
            if metrics.error_breakdown:
                for error_type, count in metrics.error_breakdown.items():
                    if count > 0:
                        anomalies_count += count
                        self.tracker.log_anomaly(
                            active_run_id,
                            {
                                "error_type": error_type,
                                "count": count,
                                "description": f"Encountered {count} instances of {error_type}",
                            },
                        )

            # Step 9: Finalize Run and Manifest
            self.tracker.finish_run(active_run_id, status="completed")
            verification = self.tracker.verify_run(active_run_id, strict=True)

            return BaselineRunResult(
                run_id=active_run_id,
                experiment_id=experiment_id,
                model_id=resolved_model_id,
                k_shots=k_shots,
                status="completed",
                total_examples=len(active_examples),
                artifact_dir=run_dir,
                verified=verification["verified"],
                verification_details=verification,
                metrics=metrics,
                anomalies_count=anomalies_count,
                total_cost_usd=round(total_cost_usd, 6),
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

        except Exception as exc:
            logger.error("Run %s failed: %s", active_run_id, exc)
            with contextlib.suppress(Exception):
                self.tracker.log_anomaly(
                    active_run_id,
                    {
                        "error_type": "PIPELINE_FAILURE",
                        "exception": str(exc),
                        "completed_generations": len(generations),
                    },
                )
                self.tracker.finish_run(active_run_id, status="failed")
                self.tracker.verify_run(active_run_id, strict=False)
            raise
