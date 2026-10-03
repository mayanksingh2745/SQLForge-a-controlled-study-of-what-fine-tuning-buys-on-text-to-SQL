"""Systematic LoRA hyperparameter and rank scaling sweeps (EXP-04).

Provides typed configuration, deterministic run planning, hardware safety gating,
interrupted-run recovery, execution orchestration, and artifact summarization
for isolating LoRA intrinsic rank scaling (r in {8, 16, 32, 64}) and target modules.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.examples import DatasetSplit, TextToSQLExample
from sqlforge.schemas.metadata import ColumnMetadata, SchemaMetadata, TableMetadata
from sqlforge.schemas.models import LoRAHyperparameters
from sqlforge.training.config import (
    DEFAULT_TARGET_MODULES_MAP,
    FineTuningMethod,
    SFTTrainingConfig,
    resolve_target_modules,
)
from sqlforge.training.preflight import PreflightChecker
from sqlforge.training.trainer import PreflightSafetyError, SFTFineTuningPipeline, SFTTrainResult
from sqlforge.utils.env_info import collect_system_diagnostics

logger = logging.getLogger(__name__)


# Standard model dimensions for parameter estimation
MODEL_ARCH_DIMENSIONS: dict[str, dict[str, int]] = {
    "qwen25_coder_7b": {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_layers": 28,
        "kv_channels": 512,  # 4 heads * 128 dim
    },
    "qwen25_coder_1_5b": {
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_layers": 28,
        "kv_channels": 256,  # 2 heads * 128 dim
    },
    "default": {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_layers": 32,
        "kv_channels": 1024,
    },
}


class RankSweepConfig(BaseModel):
    """Specification for systematic LoRA rank and hyperparameter sweeps."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(
        default="EXP-04-RANK-SWEEP",
        description="Canonical experiment identifier",
    )
    description: str = Field(
        default="LoRA intrinsic rank sweep (r in {8, 16, 32, 64}) and target modules",
        description="Detailed description of the sweep",
    )
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="Target base model identifier",
    )
    model_family: str = Field(
        default="qwen",
        description="Model family ('qwen', 'llama', etc.)",
    )
    method: FineTuningMethod = Field(
        default=FineTuningMethod.QLORA,
        description="Fine-tuning method ('lora' or 'qlora')",
    )
    ranks: list[int] = Field(
        default_factory=lambda: [8, 16, 32, 64],
        description="LoRA rank dimensions to evaluate",
    )
    alpha_multiplier: float = Field(
        default=2.0,
        description="Scaling factor for LoRA alpha: alpha = round(rank * alpha_multiplier)",
    )
    custom_alphas: dict[int, int] | None = Field(
        default=None,
        description="Optional explicit mapping from rank to alpha",
    )
    target_modules_configs: list[str] = Field(
        default_factory=lambda: ["all-linear"],
        description="List of target module configurations ('all-linear', 'attention-only')",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="Random seeds for replication across rank conditions",
    )
    epochs: int = Field(
        default=3,
        ge=1,
        description="Number of fine-tuning epochs per run",
    )
    per_device_batch_size: int = Field(
        default=8,
        ge=1,
        description="Per-device batch size",
    )
    gradient_accumulation_steps: int = Field(
        default=2,
        ge=1,
        description="Gradient accumulation steps",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        description="Learning rate held strictly constant across runs",
    )
    lr_scheduler: str = Field(
        default="cosine",
        description="Learning rate scheduler",
    )
    schema_format: str = Field(
        default="ddl",
        description="Schema serialization format ('ddl', 'compact_pipe', 'json')",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=64,
        description="Maximum sequence length",
    )
    mask_prompt_loss: bool = Field(
        default=True,
        description="Mask prompt tokens with -100 for completion-only loss",
    )
    train_dataset: str = Field(
        default="spider:train",
        description="Authorized training dataset split identifier",
    )
    eval_dataset: str = Field(
        default="spider:dev",
        description="Target evaluation dataset split identifier",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("artifacts/runs"),
        description="Directory for individual run artifacts",
    )
    sweep_dir: Path = Field(
        default_factory=lambda: Path("artifacts/sweeps"),
        description="Directory for sweep summary manifests",
    )
    max_train_examples: int | None = Field(
        default=None,
        description="Optional limit on training examples for dry-run/smoke testing",
    )

    @field_validator("ranks")
    @classmethod
    def validate_ranks(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("ranks list must contain at least one rank dimension.")
        for r in v:
            if r <= 0:
                raise ValueError(f"Rank dimension must be positive, got {r}.")
        return sorted(list(set(v)))

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seeds list must contain at least one seed.")
        for s in v:
            if s < 0:
                raise ValueError(f"Seed must be non-negative, got {s}.")
        return sorted(list(set(v)))

    @field_validator("target_modules_configs")
    @classmethod
    def validate_target_modules_configs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_modules_configs must contain at least one configuration.")
        valid_presets = {"all-linear", "attention-only"}
        for cfg in v:
            if cfg not in valid_presets and not isinstance(cfg, str):
                raise ValueError(f"Invalid target module configuration '{cfg}'.")
        return v

    def get_alpha(self, rank: int) -> int:
        """Get the LoRA alpha scaling factor for a given rank."""
        if self.custom_alphas and rank in self.custom_alphas:
            return self.custom_alphas[rank]
        return int(round(rank * self.alpha_multiplier))

    def estimate_trainable_parameters(
        self, rank: int, target_modules_tag: str = "all-linear"
    ) -> int:
        """Estimate trainable parameter count for a given rank and module config."""
        model_key = "default"
        lower_id = self.base_model_id.lower()
        if "7b" in lower_id:
            model_key = "qwen25_coder_7b"
        elif "1.5b" in lower_id or "1_5b" in lower_id:
            model_key = "qwen25_coder_1_5b"

        dims = MODEL_ARCH_DIMENSIONS.get(model_key, MODEL_ARCH_DIMENSIONS["default"])
        h = dims["hidden_size"]
        inter = dims["intermediate_size"]
        layers = dims["num_layers"]
        kv = dims["kv_channels"]

        # Attention projections: q (h->h), k (h->kv), v (h->kv), o (h->h)
        attn_params_per_layer = (h + h) * rank + (h + kv) * rank + (h + kv) * rank + (h + h) * rank

        if target_modules_tag == "attention-only":
            return attn_params_per_layer * layers

        # MLP projections: gate (h->inter), up (h->inter), down (inter->h)
        mlp_params_per_layer = (h + inter) * rank + (h + inter) * rank + (inter + h) * rank
        return (attn_params_per_layer + mlp_params_per_layer) * layers

    def estimate_adapter_size_mb(self, rank: int, target_modules_tag: str = "all-linear") -> float:
        """Estimate adapter checkpoint size in MB (assuming 16-bit weights)."""
        params = self.estimate_trainable_parameters(rank, target_modules_tag)
        return round((params * 2) / (1024 * 1024), 2)

    def generate_run_configs(self) -> list[SFTTrainingConfig]:
        """Generate controlled, deterministic SFTTrainingConfig instances for each condition."""
        configs: list[SFTTrainingConfig] = []
        for r in self.ranks:
            alpha = self.get_alpha(r)
            for mod_tag in self.target_modules_configs:
                target_modules = resolve_target_modules(self.base_model_id, mod_tag)
                for seed in self.seeds:
                    run_id = f"exp04_r{r}_a{alpha}_{mod_tag.replace('-', '_')}_s{seed}"
                    cfg = SFTTrainingConfig(
                        run_id=run_id,
                        experiment_id=self.experiment_id,
                        description=f"EXP-04 sweep: rank={r}, alpha={alpha}, modules={mod_tag}, seed={seed}",
                        base_model_id=self.base_model_id,
                        model_family=self.model_family,
                        method=self.method,
                        lora=LoRAHyperparameters(
                            rank=r,
                            alpha=alpha,
                            dropout=0.05,
                            target_modules=target_modules,
                        ),
                        epochs=self.epochs,
                        per_device_batch_size=self.per_device_batch_size,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        learning_rate=self.learning_rate,
                        lr_scheduler=self.lr_scheduler,
                        schema_format=self.schema_format,
                        max_seq_length=self.max_seq_length,
                        mask_prompt_loss=self.mask_prompt_loss,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.eval_dataset,
                        seed=seed,
                        output_dir=str(self.output_dir),
                        max_train_examples=self.max_train_examples,
                    )
                    configs.append(cfg)
        return configs


class SweepRunPlan(BaseModel):
    """Specification of an individual planned run in the sweep."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    rank: int
    alpha: int
    target_modules_tag: str
    target_modules: list[str]
    seed: int
    estimated_params: int
    estimated_size_mb: float
    is_cached: bool = False
    checkpoint_dir: str | None = None


class SweepRunResult(BaseModel):
    """Execution summary and metrics for an individual sweep run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    rank: int
    alpha: int
    target_modules_tag: str
    target_modules: list[str]
    seed: int
    status: str  # "completed", "dry_run", "cached", "failed", "skipped"
    trainable_parameters: int
    adapter_size_mb: float
    duration_seconds: float = 0.0
    final_loss: float | None = None
    total_steps: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    checkpoint_dir: str | None = None
    manifest_verified: bool = False
    error_message: str | None = None


class SweepSummary(BaseModel):
    """Complete manifest summarizing all runs in a hyperparameter/rank sweep."""

    model_config = ConfigDict(frozen=True)

    sweep_id: str
    experiment_id: str
    base_model_id: str
    method: str
    total_planned: int
    completed: int
    cached: int
    failed: int
    skipped: int
    is_dry_run: bool
    results: list[SweepRunResult]
    created_at: str
    hardware_metadata: dict[str, Any] = Field(default_factory=dict)
    output_path: str | None = None

    @property
    def all_completed(self) -> bool:
        """Whether all planned runs finished successfully or were cached."""
        return (self.completed + self.cached) == self.total_planned

    def get_completed_results(self) -> list[SweepRunResult]:
        """Return only successfully completed or cached results."""
        return [r for r in self.results if r.status in ("completed", "cached", "dry_run")]

    def has_empirical_evaluation(self) -> bool:
        """Check if any run in the sweep has recorded execution accuracy metrics."""
        return any("execution_accuracy" in r.metrics for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Convert sweep summary to a dictionary."""
        return self.model_dump()

    def save_json(self, path: Path) -> Path:
        """Save sweep summary to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load_json(cls, path: Path) -> SweepSummary:
        """Load sweep summary from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


class LoRARankSweepOrchestrator:
    """Orchestrator for systematic LoRA rank scaling sweeps."""

    def __init__(
        self,
        config: RankSweepConfig,
        tracker: ExperimentTracker | None = None,
        train_fn: Callable[[SFTTrainingConfig, list[TextToSQLExample], Path], dict[str, Any]]
        | None = None,
        eval_fn: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize LoRARankSweepOrchestrator.

        Args:
            config: RankSweepConfig defining the sweep parameters.
            tracker: Optional ExperimentTracker for run management.
            train_fn: Optional mock training callable for offline verification.
            eval_fn: Optional evaluation callable (adapter_dir, eval_split) -> metrics dict.
        """
        self.config = config
        self.tracker = tracker or ExperimentTracker(base_artifact_dir=str(config.output_dir))
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.sweep_id = f"sweep_{config.experiment_id.lower()}_{int(time.time())}"

    def plan_sweep(self) -> list[SweepRunPlan]:
        """Inspect planned runs and detect existing cached checkpoints."""
        plans: list[SweepRunPlan] = []
        run_configs = self.config.generate_run_configs()

        for cfg in run_configs:
            assert cfg.run_id is not None
            r = cfg.lora.rank
            alpha = cfg.lora.alpha
            mod_tag = (
                "all-linear"
                if len(cfg.lora.target_modules)
                == len(DEFAULT_TARGET_MODULES_MAP.get(self.config.model_family, []))
                else "attention-only"
            )
            est_params = self.config.estimate_trainable_parameters(r, mod_tag)
            est_size = self.config.estimate_adapter_size_mb(r, mod_tag)

            # Check if run exists and is verified
            run_dir = self.tracker.base_dir / cfg.run_id
            manifest_file = run_dir / "manifest.json"
            is_cached = False
            ckpt_dir = None

            if manifest_file.exists():
                verify_res = self.tracker.verify_run(cfg.run_id, strict=False)
                if verify_res.get("verified", False):
                    is_cached = True
                    ckpt_dir = str(run_dir)

            plans.append(
                SweepRunPlan(
                    run_id=cfg.run_id,
                    rank=r,
                    alpha=alpha,
                    target_modules_tag=mod_tag,
                    target_modules=cfg.lora.target_modules,
                    seed=cfg.seed,
                    estimated_params=est_params,
                    estimated_size_mb=est_size,
                    is_cached=is_cached,
                    checkpoint_dir=ckpt_dir,
                )
            )
        return plans

    def run_sweep(
        self,
        examples: list[TextToSQLExample] | None = None,
        schemas: dict[str, SchemaMetadata] | None = None,
        dry_run: bool = True,
        execute: bool = False,
        resume: bool = True,
        fail_fast: bool = False,
    ) -> SweepSummary:
        """Execute the hyperparameter/rank scaling sweep.

        Args:
            examples: Training examples (if None, synthetic fixtures used for testing).
            schemas: Schema metadata mapping (if None, synthetic schema used for testing).
            dry_run: If True, execute without live training weights.
            execute: Must be True for live compute spend when dry_run is False.
            resume: If True, skip already verified completed runs.
            fail_fast: If True, halt immediately upon any run failure.

        Returns:
            SweepSummary with complete execution results.

        Raises:
            PreflightSafetyError: If live training requested without explicit execution flag.
        """
        if not dry_run and not execute:
            raise PreflightSafetyError(
                "Live hyperparameter sweep requires explicit confirmation via execute=True "
                "or dry_run=True for zero-compute orchestration."
            )

        logger.info(
            f"Starting LoRA Rank Sweep '{self.sweep_id}' ({len(self.config.ranks)} ranks, "
            f"{len(self.config.seeds)} seeds, dry_run={dry_run})..."
        )

        # Preflight verification on base configuration
        first_cfg = self.config.generate_run_configs()[0]
        checker = PreflightChecker(first_cfg)
        preflight_report = checker.run_preflight(check_dataset=False)

        if not dry_run and not self.train_fn and not preflight_report.is_runnable:
            blocker_text = "\n".join(f" - {b}" for b in preflight_report.blockers)
            raise PreflightSafetyError(
                f"Hardware feasibility pre-flight failed for sweep:\n{blocker_text}"
            )

        # Setup fallback fixtures if not provided
        resolved_schemas = schemas or self._get_fallback_schemas()
        resolved_examples = examples or self._get_fallback_examples()

        plans = self.plan_sweep()
        run_configs = {cfg.run_id: cfg for cfg in self.config.generate_run_configs()}
        results: list[SweepRunResult] = []

        completed_count = 0
        cached_count = 0
        failed_count = 0
        skipped_count = 0

        for plan in plans:
            cfg = run_configs[plan.run_id]

            # 1. Check for cached/resumed run
            if resume and plan.is_cached and plan.checkpoint_dir:
                logger.info(
                    f"Run '{plan.run_id}' already completed and verified. Resuming/skipping."
                )
                cached_count += 1
                # Try loading existing metrics
                cached_metrics = self._load_run_metrics(plan.run_id)
                results.append(
                    SweepRunResult(
                        run_id=plan.run_id,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        target_modules_tag=plan.target_modules_tag,
                        target_modules=plan.target_modules,
                        seed=plan.seed,
                        status="cached",
                        trainable_parameters=plan.estimated_params,
                        adapter_size_mb=plan.estimated_size_mb,
                        duration_seconds=cached_metrics.get("duration_seconds", 0.0),
                        final_loss=cached_metrics.get("training_loss"),
                        total_steps=cached_metrics.get("total_steps", 0),
                        metrics=cached_metrics,
                        checkpoint_dir=plan.checkpoint_dir,
                        manifest_verified=True,
                    )
                )
                continue

            # 2. Execute Run
            assert cfg.run_id is not None
            run_dir = self.tracker.base_dir / cfg.run_id
            if run_dir.exists():
                new_run_id = f"{cfg.run_id}_{int(time.time())}"
                cfg = cfg.model_copy(update={"run_id": new_run_id})

            logger.info(
                f"Executing sweep condition: rank={plan.rank}, alpha={plan.alpha}, seed={plan.seed}"
            )
            start_t = time.monotonic()
            try:
                pipeline = SFTFineTuningPipeline(
                    config=cfg,
                    tracker=self.tracker,
                    train_fn=self._wrap_train_fn(self.train_fn, cfg) if self.train_fn else None,
                )
                run_res: SFTTrainResult = pipeline.run(
                    examples=resolved_examples,
                    schemas=resolved_schemas,
                    dry_run=dry_run,
                )
                duration = time.monotonic() - start_t

                # Run post-training evaluation if eval_fn is provided
                eval_metrics: dict[str, Any] = {}
                if self.eval_fn is not None and not dry_run:
                    try:
                        eval_metrics = self.eval_fn(
                            run_res.final_checkpoint_dir, cfg.eval_dataset or "spider:dev"
                        )
                    except Exception as eval_err:
                        logger.warning(f"Evaluation failed for {plan.run_id}: {eval_err}")

                completed_count += 1
                results.append(
                    SweepRunResult(
                        run_id=plan.run_id,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        target_modules_tag=plan.target_modules_tag,
                        target_modules=plan.target_modules,
                        seed=plan.seed,
                        status="dry_run" if dry_run else "completed",
                        trainable_parameters=plan.estimated_params,
                        adapter_size_mb=plan.estimated_size_mb,
                        duration_seconds=round(duration, 2),
                        final_loss=run_res.final_loss,
                        total_steps=run_res.total_steps,
                        metrics=eval_metrics,
                        checkpoint_dir=run_res.final_checkpoint_dir,
                        manifest_verified=run_res.manifest_verified,
                    )
                )
            except Exception as exc:
                duration = time.monotonic() - start_t
                failed_count += 1
                logger.error(f"Sweep run '{plan.run_id}' failed: {exc}", exc_info=True)
                results.append(
                    SweepRunResult(
                        run_id=plan.run_id,
                        rank=plan.rank,
                        alpha=plan.alpha,
                        target_modules_tag=plan.target_modules_tag,
                        target_modules=plan.target_modules,
                        seed=plan.seed,
                        status="failed",
                        trainable_parameters=plan.estimated_params,
                        adapter_size_mb=plan.estimated_size_mb,
                        duration_seconds=round(duration, 2),
                        error_message=str(exc),
                    )
                )
                if fail_fast:
                    logger.warning("Failing fast on error during hyperparameter sweep.")
                    break

        summary = SweepSummary(
            sweep_id=self.sweep_id,
            experiment_id=self.config.experiment_id,
            base_model_id=self.config.base_model_id,
            method=self.config.method.value,
            total_planned=len(plans),
            completed=completed_count,
            cached=cached_count,
            failed=failed_count,
            skipped=skipped_count,
            is_dry_run=dry_run,
            results=results,
            created_at=datetime.now(UTC).isoformat(),
            hardware_metadata=collect_system_diagnostics(),
        )

        # Save summary artifact
        summary_path = self.config.sweep_dir / self.sweep_id / "sweep_summary.json"
        summary.save_json(summary_path)
        logger.info(f"Sweep summary saved to: {summary_path}")

        return summary

    def _load_run_metrics(self, run_id: str) -> dict[str, Any]:
        """Load recorded metrics for a completed run."""
        metrics_file = self.tracker.base_dir / run_id / "training_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        return loaded
            except Exception:
                pass
        return {}

    def _wrap_train_fn(
        self,
        fn: Callable[[SFTTrainingConfig, list[TextToSQLExample], Path], dict[str, Any]],
        cfg: SFTTrainingConfig,
    ) -> Callable[[SFTTrainingConfig, list[Any], Path], dict[str, Any]]:
        """Wrap train_fn to match expected signature."""

        def wrapped(_cfg: SFTTrainingConfig, ex_list: list[Any], ckpt: Path) -> dict[str, Any]:
            return fn(cfg, ex_list, ckpt)

        return wrapped

    def _get_fallback_schemas(self) -> dict[str, SchemaMetadata]:
        return {
            "academic": SchemaMetadata(
                db_id="academic",
                dialect="sqlite",
                tables=[
                    TableMetadata(
                        table_name="author",
                        columns=[
                            ColumnMetadata(
                                name="author_id", data_type="INTEGER", is_primary_key=True
                            ),
                            ColumnMetadata(name="name", data_type="TEXT"),
                        ],
                    )
                ],
            )
        }

    def _get_fallback_examples(self) -> list[TextToSQLExample]:
        return [
            TextToSQLExample(
                id=f"spider_train_{i:04d}",
                question=f"List all author names in the database? (example {i})",
                db_id="academic",
                gold_sql="SELECT name FROM author;",
                dataset_name="spider",
                split=DatasetSplit.TRAIN,
            )
            for i in range(1, 11)
        ]
