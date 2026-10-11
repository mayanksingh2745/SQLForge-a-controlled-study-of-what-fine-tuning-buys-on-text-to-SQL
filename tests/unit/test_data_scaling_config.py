"""Unit tests for EXP-05 and EXP-06 data scaling configuration contracts and rank selection."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP05_SEEDS,
    EXP05_TRAINING_SIZES,
    EXP06_EXPERIMENT_ID,
    EXP06_GROUPS,
    EXP06_SEEDS,
    RankSelectionReference,
    ScalingRunCondition,
    generate_exp05_run_conditions,
    generate_exp06_run_conditions,
)


def test_exp05_constants() -> None:
    """Verify EXP-05 research constants strictly match the specification."""
    assert EXP05_EXPERIMENT_ID == "EXP-05-DATA-SCALING"
    assert EXP05_TRAINING_SIZES == (500, 1000, 2000, 5000, 7000)
    assert EXP05_SEEDS == (42, 43, 44)


def test_exp06_constants() -> None:
    """Verify EXP-06 research constants strictly match the specification."""
    assert EXP06_EXPERIMENT_ID == "EXP-06-SYNTHETIC-DATA"
    assert EXP06_GROUPS == ("human_only", "synthetic_only", "human_synthetic_50_50")
    assert EXP06_SEEDS == (42, 43, 44)


def test_rank_selection_reference_valid() -> None:
    """Verify valid empirical rank selection reference passes validation."""
    ref = RankSelectionReference(
        source_experiment_id="EXP-04-RANK-SWEEP",
        source_run_ids=["exp04_r16_s42", "exp04_r16_s43", "exp04_r16_s44"],
        source_manifest_paths=["manifests/exp04_r16_s42.json"],
        source_manifest_hashes=["a" * 64],
        selected_rank=16,
        selected_alpha=32,
        selected_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        selection_metric="spider_dev_execution_accuracy",
        selection_rule="highest_execution_accuracy_tiebreak_min_params",
        selection_timestamp="2026-10-10T00:00:00Z",
        verification_status="verified",
    )
    assert ref.selected_rank == 16
    assert ref.selected_alpha == 32
    assert ref.is_empirical_eligible is True


def test_rank_selection_reference_invalid_rank() -> None:
    """Verify selected rank outside {8, 16, 32, 64} is rejected."""
    with pytest.raises(ValidationError, match="selected_rank"):
        RankSelectionReference(
            source_experiment_id="EXP-04-RANK-SWEEP",
            source_run_ids=["exp04_r4_s42"],
            source_manifest_paths=["manifests/exp04_r4_s42.json"],
            source_manifest_hashes=["a" * 64],
            selected_rank=4,  # Invalid
            selected_alpha=8,
            selected_target_modules=["q_proj", "v_proj"],
            selection_metric="spider_dev_execution_accuracy",
            selection_rule="highest_execution_accuracy_tiebreak_min_params",
            selection_timestamp="2026-10-10T00:00:00Z",
            verification_status="verified",
        )


def test_rank_selection_reference_alpha_mismatch() -> None:
    """Verify selected alpha must equal 2 * selected rank."""
    with pytest.raises(ValidationError, match="selected_alpha"):
        RankSelectionReference(
            source_experiment_id="EXP-04-RANK-SWEEP",
            source_run_ids=["exp04_r16_s42"],
            source_manifest_paths=["manifests/exp04_r16_s42.json"],
            source_manifest_hashes=["a" * 64],
            selected_rank=16,
            selected_alpha=16,  # Should be 32
            selected_target_modules=["q_proj", "v_proj"],
            selection_metric="spider_dev_execution_accuracy",
            selection_rule="highest_execution_accuracy_tiebreak_min_params",
            selection_timestamp="2026-10-10T00:00:00Z",
            verification_status="verified",
        )


def test_rank_selection_reference_invalid_source_exp() -> None:
    """Verify source_experiment_id must be EXP-04-RANK-SWEEP."""
    with pytest.raises(ValidationError, match="source_experiment_id"):
        RankSelectionReference(
            source_experiment_id="EXP-01-BASELINE",
            source_run_ids=["run1"],
            source_manifest_paths=["manifest1.json"],
            source_manifest_hashes=["a" * 64],
            selected_rank=16,
            selected_alpha=32,
            selected_target_modules=["q_proj", "v_proj"],
            selection_metric="spider_dev_execution_accuracy",
            selection_rule="highest_execution_accuracy_tiebreak_min_params",
            selection_timestamp="2026-10-10T00:00:00Z",
            verification_status="verified",
        )


def test_rank_selection_provisional_for_dry_run() -> None:
    """Verify provisional rank reference is created with empirical_execution_allowed=False."""
    prov = RankSelectionReference.create_provisional(rank=16)
    assert prov.selected_rank == 16
    assert prov.selected_alpha == 32
    assert prov.verification_status == "provisional_non_empirical"
    assert prov.is_empirical_eligible is False


def test_rank_selection_extra_fields_forbidden() -> None:
    """Verify extra fields are strictly forbidden."""
    with pytest.raises(ValidationError):
        RankSelectionReference(
            source_experiment_id="EXP-04-RANK-SWEEP",
            source_run_ids=["exp04_r16_s42"],
            source_manifest_paths=["manifests/exp04_r16_s42.json"],
            source_manifest_hashes=["a" * 64],
            selected_rank=16,
            selected_alpha=32,
            selected_target_modules=["q_proj", "v_proj"],
            selection_metric="spider_dev_execution_accuracy",
            selection_rule="highest_execution_accuracy_tiebreak_min_params",
            selection_timestamp="2026-10-10T00:00:00Z",
            verification_status="verified",
            arbitrary_field="not_allowed",  # type: ignore[call-arg]
        )


def test_generate_exp05_conditions_count() -> None:
    """Verify EXP-05 generates exactly 15 run conditions (5 sizes x 3 seeds)."""
    conditions = generate_exp05_run_conditions()
    assert len(conditions) == 15

    expected_ids = [
        f"exp05_n{size}_s{seed}" for size in EXP05_TRAINING_SIZES for seed in EXP05_SEEDS
    ]
    actual_ids = [c.run_id for c in conditions]
    assert actual_ids == expected_ids

    for c in conditions:
        assert c.experiment_id == EXP05_EXPERIMENT_ID
        assert c.group_name == f"n_{c.total_examples}"
        assert c.human_examples_count == c.total_examples
        assert c.synthetic_examples_count == 0


def test_generate_exp06_conditions_count() -> None:
    """Verify EXP-06 generates exactly 9 run conditions (3 groups x 3 seeds)."""
    conditions = generate_exp06_run_conditions()
    assert len(conditions) == 9

    expected_ids = [f"exp06_{grp}_s{seed}" for grp in EXP06_GROUPS for seed in EXP06_SEEDS]
    actual_ids = [c.run_id for c in conditions]
    assert actual_ids == expected_ids

    for c in conditions:
        assert c.experiment_id == EXP06_EXPERIMENT_ID
        assert c.total_examples == 3000
        if c.group_name == "human_only":
            assert c.human_examples_count == 3000
            assert c.synthetic_examples_count == 0
        elif c.group_name == "synthetic_only":
            assert c.human_examples_count == 0
            assert c.synthetic_examples_count == 3000
        elif c.group_name == "human_synthetic_50_50":
            assert c.human_examples_count == 1500
            assert c.synthetic_examples_count == 1500


def test_scaling_run_condition_to_sft_config() -> None:
    """Verify ScalingRunCondition accurately constructs SFTTrainingConfig."""
    cond = ScalingRunCondition(
        run_id="exp05_n1000_s42",
        experiment_id=EXP05_EXPERIMENT_ID,
        seed=42,
        total_examples=1000,
        human_examples_count=1000,
        synthetic_examples_count=0,
        group_name="n_1000",
    )
    sft_cfg = cond.to_sft_training_config()
    assert sft_cfg.run_id == "exp05_n1000_s42"
    assert sft_cfg.seed == 42
    assert sft_cfg.base_model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert sft_cfg.train_dataset == "spider:train"
    assert sft_cfg.eval_dataset == "spider:dev"
    assert sft_cfg.max_train_examples == 1000
    assert sft_cfg.epochs == 3
    assert sft_cfg.learning_rate == 0.0002
    assert sft_cfg.lr_scheduler == "cosine"
    assert sft_cfg.warmup_ratio == 0.03
    assert sft_cfg.per_device_batch_size == 8
    assert sft_cfg.gradient_accumulation_steps == 2
    assert sft_cfg.max_seq_length == 2048
    assert sft_cfg.schema_format == "ddl"
    assert sft_cfg.mask_prompt_loss is True
    assert sft_cfg.method.value == "qlora"
    assert sft_cfg.lora.rank == 16
    assert sft_cfg.lora.alpha == 32
    assert sft_cfg.qlora.quant_type == "nf4"
    assert sft_cfg.qlora.use_double_quant is True
