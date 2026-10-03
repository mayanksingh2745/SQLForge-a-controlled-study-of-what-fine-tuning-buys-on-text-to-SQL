"""Evaluation handoff bridge connecting trained PEFT checkpoints to the baseline evaluation pipeline.

Configures LocalHFModelRunner with adapter weights and sets up ExperimentConfig
strictly targeting the evaluation development split (dev) while isolating held-out partitions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlforge.models.local_hf import LocalHFModelRunner
from sqlforge.schemas.experiments import ExperimentConfig, PromptConfig
from sqlforge.training.checkpoints import CheckpointManager, CheckpointMetadata
from sqlforge.training.config import SFTTrainingConfig

logger = logging.getLogger(__name__)


def validate_adapter_checkpoint(checkpoint_dir: str | Path) -> CheckpointMetadata | None:
    """Validate that a checkpoint directory exists and inspect its metadata if available.

    Args:
        checkpoint_dir: Path to directory containing saved adapter files.

    Returns:
        CheckpointMetadata instance if metadata file exists, else None.

    Raises:
        FileNotFoundError: If the checkpoint directory itself does not exist.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")

    meta_file = ckpt_path / "checkpoint_metadata.json"
    if meta_file.exists():
        return CheckpointManager.load_metadata(ckpt_path)
    return None


def prepare_evaluation_runner(
    base_model_id: str,
    adapter_checkpoint_dir: str | Path,
    device: str | None = None,
    torch_dtype: str = "bfloat16",
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    inference_fn: Callable[[str, int], tuple[str, int, int]] | None = None,
) -> LocalHFModelRunner:
    """Instantiate a LocalHFModelRunner configured with the trained PEFT adapter.

    Args:
        base_model_id: Hugging Face repo ID or path of the base model.
        adapter_checkpoint_dir: Path to the trained PEFT adapter directory.
        device: Compute device ('cuda', 'cpu', 'auto').
        torch_dtype: Weight precision.
        temperature: Sampling temperature (must be 0.0 for deterministic greedy decoding).
        max_new_tokens: Maximum completion token budget.
        inference_fn: Optional mock callable for testing.

    Returns:
        Configured LocalHFModelRunner.
    """
    validate_adapter_checkpoint(adapter_checkpoint_dir)
    logger.info(
        f"Preparing evaluation model runner: base='{base_model_id}', adapter='{adapter_checkpoint_dir}'"
    )

    return LocalHFModelRunner(
        model_id=base_model_id,
        adapter_path=adapter_checkpoint_dir,
        device=device,
        torch_dtype=torch_dtype,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        inference_fn=inference_fn,
    )


def create_evaluation_config(
    train_config: SFTTrainingConfig,
    adapter_checkpoint_dir: str | Path,
    eval_dataset: str = "spider:dev",
) -> ExperimentConfig:
    """Create an ExperimentConfig for evaluating a fine-tuned adapter on dev split.

    Args:
        train_config: SFTTrainingConfig used during fine-tuning.
        adapter_checkpoint_dir: Directory where fine-tuned adapter is stored.
        eval_dataset: Dataset key for evaluation (strictly dev partition).

    Returns:
        ExperimentConfig conforming to the evaluation pipeline contracts.

    Raises:
        ValueError: If eval_dataset references a test or held-out partition.
    """
    parts = eval_dataset.split(":")
    if len(parts) == 2 and parts[1].lower() in {"test", "held_out"}:
        raise ValueError(
            f"Evaluation handoff strictly prohibits evaluating on '{parts[1]}' split. "
            "Use 'dev' partition for post-training validation."
        )

    ckpt_name = Path(adapter_checkpoint_dir).name
    run_id_eval = f"{train_config.run_id or 'sft'}_{ckpt_name}_eval"

    return ExperimentConfig(
        experiment_id=f"{train_config.experiment_id}-EVAL",
        description=f"Evaluation of fine-tuned adapter {ckpt_name} on {eval_dataset}",
        seed=train_config.seed,
        model_id=f"{train_config.base_model_id}[adapter={ckpt_name}]",
        paradigm="fine-tuning-eval",
        train_dataset=None,
        eval_datasets=[eval_dataset],
        prompt=PromptConfig(
            template_name="default_text2sql",
            schema_format=train_config.schema_format,
            k_shots=0,
            retriever_type=None,
        ),
        output_dir=f"artifacts/runs/{run_id_eval}",
    )
