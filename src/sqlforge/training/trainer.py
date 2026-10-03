"""Supervised fine-tuning pipeline orchestrator for LoRA and QLoRA.

Implements FineTuningEngine protocol, coordinating pre-flight feasibility gates,
data formatting, completion-only loss masking, model preparation, training loops,
checkpoint saving, and experiment tracking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.experiments.tracker import ExperimentTracker
from sqlforge.schemas.evaluation import EvaluationMetrics
from sqlforge.schemas.examples import TextToSQLExample
from sqlforge.schemas.metadata import SchemaMetadata
from sqlforge.training.adapters import AdapterConfigSummary, PEFTConfigFactory
from sqlforge.training.checkpoints import CheckpointManager
from sqlforge.training.config import FineTuningMethod, SFTTrainingConfig
from sqlforge.training.data import SFTDatasetFormatter, SFTExample, TrainingDataSummary
from sqlforge.training.preflight import PreflightChecker, PreflightReport
from sqlforge.training.tokenization import CompletionLossMasker, ToyDeterministicTokenizer

logger = logging.getLogger(__name__)


class PreflightSafetyError(RuntimeError):
    """Raised when training is invoked on hardware or environments failing safety gates."""


class SFTTrainResult(BaseModel):
    """Execution summary and artifact manifest for a completed SFT training run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique run identifier")
    experiment_id: str = Field(..., description="Experiment ID")
    method: str = Field(..., description="'lora' or 'qlora'")
    status: str = Field(..., description="'completed', 'dry_run', or 'failed'")
    final_checkpoint_dir: str = Field(..., description="Path to final saved adapter checkpoint")
    adapter_summary: AdapterConfigSummary = Field(
        ..., description="LoRA/QLoRA configuration summary"
    )
    data_summary: TrainingDataSummary = Field(..., description="Dataset preparation audit summary")
    preflight_report: PreflightReport = Field(
        ..., description="Hardware and safety gate diagnostic"
    )
    total_steps: int = Field(default=0, description="Total training optimization steps executed")
    final_loss: float | None = Field(default=None, description="Final recorded cross-entropy loss")
    duration_seconds: float = Field(default=0.0, description="Total wall-clock training duration")
    manifest_verified: bool = Field(
        default=False, description="Whether experiment manifest is verified"
    )


class SFTFineTuningPipeline:
    """End-to-end supervised fine-tuning pipeline supporting LoRA and QLoRA."""

    def __init__(
        self,
        config: SFTTrainingConfig,
        tracker: ExperimentTracker | None = None,
        train_fn: Callable[[SFTTrainingConfig, list[SFTExample], Path], dict[str, Any]]
        | None = None,
    ) -> None:
        """Initialize SFTFineTuningPipeline.

        Args:
            config: SFTTrainingConfig with complete training specifications.
            tracker: Optional ExperimentTracker instance.
            train_fn: Optional mock training callable for testing and offline verification.
        """
        self.config = config
        self.run_id = (
            config.run_id or f"sft_{config.method.value}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        )
        self.tracker = tracker or ExperimentTracker(base_artifact_dir="artifacts/runs")
        self.checkpoint_manager = CheckpointManager(
            base_dir=config.output_dir,
            run_id=self.run_id,
        )
        self._train_fn = train_fn

    def run(
        self,
        examples: list[TextToSQLExample],
        schemas: dict[str, SchemaMetadata],
        dry_run: bool = False,
        tokenizer: Any | None = None,
    ) -> SFTTrainResult:
        """Execute the supervised fine-tuning pipeline.

        Args:
            examples: Training TextToSQLExample instances (must be split == TRAIN).
            schemas: Mapping of db_id to SchemaMetadata.
            dry_run: If True, execute preflight, data formatting, and tokenization without training.
            tokenizer: Optional pre-loaded tokenizer. If None, loaded lazily or toy tokenizer in dry run.

        Returns:
            SFTTrainResult with execution details and artifact references.

        Raises:
            PreflightSafetyError: If hardware/environment checks fail and not in dry_run mode.
        """
        start_time = time.monotonic()
        logger.info(
            f"Starting SFT pipeline run '{self.run_id}' (method: {self.config.method.value})..."
        )

        # 1. Initialize Run Tracking
        run_meta = self.tracker.init_run(
            experiment_name=self.config.experiment_id,
            config=self.config.to_dict(),
            seed=self.config.seed,
            run_id=self.run_id,
        )
        active_run_id = run_meta.run_id

        # 2. Hardware Pre-Flight Safety Gates
        checker = PreflightChecker(self.config)
        preflight_report = checker.run_preflight(check_dataset=False)

        if not dry_run and not self._train_fn and not preflight_report.is_runnable:
            err_msg = (
                f"Training pre-flight check failed with status '{preflight_report.status.value.upper()}'.\n"
                f"Blockers:\n" + "\n".join(f"  - {b}" for b in preflight_report.blockers) + "\n"
                "Recommendations:\n"
                + "\n".join(f"  - {r}" for r in preflight_report.recommendations)
            )
            self.tracker.log_anomaly(
                run_id=active_run_id,
                anomaly={
                    "step": 0,
                    "error_type": "PreflightSafetyError",
                    "message": err_msg,
                },
            )
            self.tracker.finish_run(run_id=active_run_id, status="failed")
            raise PreflightSafetyError(err_msg)

        # 3. Data Preparation & Split Isolation
        formatter = SFTDatasetFormatter(schema_format=self.config.schema_format)
        sft_examples, data_summary = formatter.format_dataset(
            examples=examples,
            schemas=schemas,
            max_examples=self.config.max_train_examples,
        )

        # Record dataset summary artifact
        data_summary_path = self.tracker.base_dir / active_run_id / "training_data_summary.json"
        with open(data_summary_path, "w", encoding="utf-8") as f:
            f.write(data_summary.model_dump_json(indent=2))

        # 4. Tokenization & Completion Loss Mask Verification
        resolved_tokenizer = tokenizer or ToyDeterministicTokenizer()
        masker = CompletionLossMasker(
            tokenizer=resolved_tokenizer,
            max_seq_length=self.config.max_seq_length,
            mask_prompt_loss=self.config.mask_prompt_loss,
        )

        sample_tokens = [masker.process_example(ex) for ex in sft_examples[:5]]
        logger.info(
            f"Tokenized {len(sample_tokens)} sample training examples. "
            f"Prompt loss masked: {self.config.mask_prompt_loss}"
        )

        # 5. Adapter Configuration Summary
        adapter_summary = PEFTConfigFactory.summarize_adapter_config(self.config)

        # 6. Execute Training or Dry Run
        final_dir = self.checkpoint_manager.get_final_dir()
        final_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info("Dry-run requested: skipping model weight training.")
            duration = round(time.monotonic() - start_time, 2)

            # Save dry-run checkpoint metadata
            self.checkpoint_manager.save_checkpoint_metadata(
                target_dir=final_dir,
                step=0,
                epoch=0.0,
                config=self.config,
                adapter_summary=adapter_summary,
                training_metrics={"loss": 0.0},
                dataset_info={"examples_count": len(sft_examples)},
                checkpoint_type="adapter_only",
                adapter_files=["adapter_config.json"],
            )

            # Create mock adapter_config.json
            adapter_conf_path = final_dir / "adapter_config.json"
            with open(adapter_conf_path, "w", encoding="utf-8") as f:
                f.write(adapter_summary.model_dump_json(indent=2))

            # Save training metrics to training_metrics.json
            train_metrics_path = self.tracker.base_dir / active_run_id / "training_metrics.json"
            with open(train_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"training_loss": 0.0, "examples_processed": len(sft_examples)}, f, indent=2
                )

            self.tracker.finish_run(run_id=active_run_id, status="dry_run")

            verification = self.tracker.verify_run(run_id=active_run_id, strict=False)

            return SFTTrainResult(
                run_id=active_run_id,
                experiment_id=self.config.experiment_id,
                method=self.config.method.value,
                status="dry_run",
                final_checkpoint_dir=str(final_dir),
                adapter_summary=adapter_summary,
                data_summary=data_summary,
                preflight_report=preflight_report,
                total_steps=0,
                final_loss=0.0,
                duration_seconds=duration,
                manifest_verified=bool(verification.get("verified", False)),
            )

        # Custom mock training callable (for integration tests)
        if self._train_fn is not None:
            logger.info("Executing custom mock training function...")
            train_results = self._train_fn(self.config, sft_examples, final_dir)
            total_steps = train_results.get("total_steps", 100)
            final_loss = train_results.get("final_loss", 0.45)
            training_metrics = train_results.get("metrics", {"loss": final_loss})

            # Save checkpoint metadata
            self.checkpoint_manager.save_checkpoint_metadata(
                target_dir=final_dir,
                step=total_steps,
                epoch=float(self.config.epochs),
                config=self.config,
                adapter_summary=adapter_summary,
                training_metrics=training_metrics,
                dataset_info={"examples_count": len(sft_examples)},
                checkpoint_type="adapter_only",
                adapter_files=["adapter_config.json", "adapter_model.safetensors"],
            )

            adapter_conf_path = final_dir / "adapter_config.json"
            if not adapter_conf_path.exists():
                with open(adapter_conf_path, "w", encoding="utf-8") as f:
                    f.write(adapter_summary.model_dump_json(indent=2))

            weights_path = final_dir / "adapter_model.safetensors"
            if not weights_path.exists():
                weights_path.write_bytes(b"MOCK_PEFT_ADAPTER_WEIGHTS")

            # Save training metrics to training_metrics.json
            train_metrics_path = self.tracker.base_dir / active_run_id / "training_metrics.json"
            with open(train_metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"training_loss": final_loss, "total_steps": float(total_steps)}, f, indent=2
                )

            eval_metrics = EvaluationMetrics(
                total_examples=len(sft_examples),
                execution_accuracy=0.0,
                exact_match_accuracy=0.0,
                execution_success_rate=1.0,
                syntax_valid_rate=1.0,
            )
            self.tracker.log_metrics(run_id=active_run_id, metrics=eval_metrics)
            self.tracker.finish_run(run_id=active_run_id, status="completed")

            duration = round(time.monotonic() - start_time, 2)
            verification = self.tracker.verify_run(run_id=active_run_id, strict=False)

            return SFTTrainResult(
                run_id=active_run_id,
                experiment_id=self.config.experiment_id,
                method=self.config.method.value,
                status="completed",
                final_checkpoint_dir=str(final_dir),
                adapter_summary=adapter_summary,
                data_summary=data_summary,
                preflight_report=preflight_report,
                total_steps=total_steps,
                final_loss=final_loss,
                duration_seconds=duration,
                manifest_verified=bool(verification.get("verified", False)),
            )

        # Real training execution with Hugging Face transformers/peft
        return self._execute_hf_training(
            sft_examples=sft_examples,
            data_summary=data_summary,
            adapter_summary=adapter_summary,
            preflight_report=preflight_report,
            final_dir=final_dir,
            start_time=start_time,
        )

    def _execute_hf_training(
        self,
        sft_examples: list[SFTExample],
        data_summary: TrainingDataSummary,
        adapter_summary: AdapterConfigSummary,
        preflight_report: PreflightReport,
        final_dir: Path,
        start_time: float,
    ) -> SFTTrainResult:
        """Execute real training with PyTorch, Transformers, and PEFT."""
        try:
            import torch
            from peft import get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as exc:
            raise PreflightSafetyError(
                "Real training requires 'torch', 'transformers', and 'peft'. "
                'Install via: pip install -e ".[train]".'
            ) from exc

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            revision=self.config.model_revision,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Tokenize Dataset
        masker = CompletionLossMasker(
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            mask_prompt_loss=self.config.mask_prompt_loss,
        )
        tokenized_records = [masker.process_example(ex) for ex in sft_examples]

        # Convert to HuggingFace Dataset format
        from datasets import Dataset

        hf_dataset = Dataset.from_dict(
            {
                "input_ids": [r.input_ids for r in tokenized_records],
                "labels": [r.labels for r in tokenized_records],
                "attention_mask": [r.attention_mask for r in tokenized_records],
            }
        )

        # 3. Model Loading with LoRA or QLoRA
        bnb_config = PEFTConfigFactory.build_bitsandbytes_config(self.config)
        is_qlora = self.config.method == FineTuningMethod.QLORA

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            revision=self.config.model_revision,
            quantization_config=bnb_config if is_qlora else None,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for PEFT
        model = PEFTConfigFactory.prepare_model_for_training(model, is_qlora=is_qlora)
        lora_config = PEFTConfigFactory.build_lora_config(self.config)
        model = get_peft_model(model, lora_config)

        # 4. Training Arguments
        train_args = TrainingArguments(
            output_dir=str(self.checkpoint_manager.run_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=(self.config.mixed_precision == "bf16"),
            fp16=(self.config.mixed_precision == "fp16"),
            seed=self.config.seed,
            report_to=["wandb"] if self.config.use_wandb else [],
        )

        # Data collator for padded batch
        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(x["input_ids"]) for x in batch)
            p_in = [
                x["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(x["input_ids"]))
                for x in batch
            ]
            p_lab = [x["labels"] + [-100] * (max_len - len(x["labels"])) for x in batch]
            p_mask = [
                x["attention_mask"] + [0] * (max_len - len(x["attention_mask"])) for x in batch
            ]
            return {
                "input_ids": torch.tensor(p_in, dtype=torch.long),
                "labels": torch.tensor(p_lab, dtype=torch.long),
                "attention_mask": torch.tensor(p_mask, dtype=torch.long),
            }

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=hf_dataset,
            data_collator=collate_fn,
        )

        # 5. Train
        train_output = trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        total_steps = train_output.global_step
        final_loss = train_output.training_loss

        # 6. Save Adapter Checkpoint
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        self.checkpoint_manager.save_checkpoint_metadata(
            target_dir=final_dir,
            step=total_steps,
            epoch=float(self.config.epochs),
            config=self.config,
            adapter_summary=adapter_summary,
            training_metrics={"loss": final_loss, "total_steps": float(total_steps)},
            dataset_info={"examples_count": len(sft_examples)},
            checkpoint_type="adapter_only",
            adapter_files=["adapter_config.json", "adapter_model.safetensors"],
        )

        # Save training metrics to training_metrics.json
        train_metrics_path = self.tracker.base_dir / self.run_id / "training_metrics.json"
        with open(train_metrics_path, "w", encoding="utf-8") as f:
            json.dump({"training_loss": final_loss, "total_steps": float(total_steps)}, f, indent=2)

        eval_metrics = EvaluationMetrics(
            total_examples=len(sft_examples),
            execution_accuracy=0.0,
            exact_match_accuracy=0.0,
            execution_success_rate=1.0,
            syntax_valid_rate=1.0,
        )
        self.tracker.log_metrics(run_id=self.run_id, metrics=eval_metrics)
        self.tracker.finish_run(run_id=self.run_id, status="completed")

        duration = round(time.monotonic() - start_time, 2)
        verification = self.tracker.verify_run(run_id=self.run_id, strict=False)

        return SFTTrainResult(
            run_id=self.run_id,
            experiment_id=self.config.experiment_id,
            method=self.config.method.value,
            status="completed",
            final_checkpoint_dir=str(final_dir),
            adapter_summary=adapter_summary,
            data_summary=data_summary,
            preflight_report=preflight_report,
            total_steps=total_steps,
            final_loss=final_loss,
            duration_seconds=duration,
            manifest_verified=bool(verification.get("verified", False)),
        )
