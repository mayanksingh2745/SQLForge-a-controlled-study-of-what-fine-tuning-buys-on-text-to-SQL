"""Unit tests for DataScalingAnalyzer and empirical scaling statistical fits."""

from __future__ import annotations

import pytest

from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP06_EXPERIMENT_ID,
    NoEmpiricalScalingDataError,
)
from sqlforge.training.scaling_analysis import (
    DataScalingAnalyzer,
    EXP05AnalysisResult,
    EXP06AnalysisResult,
)


def _generate_synthetic_test_metrics_exp05() -> list[dict[str, object]]:
    """Generate explicitly labeled test fixture records for mathematical verification.

    Notice: Test fixture only. Never used as empirical research results.
    """
    records: list[dict[str, object]] = []
    # Sizes: 500, 1000, 2000, 5000, 7000
    base_accs = {500: 0.50, 1000: 0.58, 2000: 0.65, 5000: 0.72, 7000: 0.75}
    for sz, acc in base_accs.items():
        for s_idx, s in enumerate([42, 43, 44]):
            # slight jitter per seed
            seed_acc = acc + (s_idx - 1) * 0.01
            records.append(
                {
                    "run_id": f"exp05_n{sz}_s{s}",
                    "experiment_id": EXP05_EXPERIMENT_ID,
                    "seed": s,
                    "status": "completed",
                    "sample_size": sz,
                    "total_examples": sz,
                    "metrics": {
                        "execution_accuracy": round(seed_acc, 4),
                        "valid_sql_rate": 0.98,
                        "exact_match": round(seed_acc - 0.08, 4),
                    },
                    "checkpoint_dir": f"/mock/ckpt/n{sz}_s{s}",
                }
            )
    return records


def test_analyzer_rejects_empty_runs() -> None:
    """Verify analyzer raises NoEmpiricalScalingDataError when run list is empty."""
    analyzer = DataScalingAnalyzer()
    with pytest.raises(NoEmpiricalScalingDataError, match="No empirical evaluation runs provided"):
        analyzer.analyze_exp05([])


def test_analyzer_rejects_dry_runs() -> None:
    """Verify analyzer strictly rejects dry-run records and raises NoEmpiricalScalingDataError."""
    dry_runs = [
        {
            "run_id": "exp05_n500_s42",
            "experiment_id": EXP05_EXPERIMENT_ID,
            "seed": 42,
            "status": "dry_run",
            "sample_size": 500,
            "metrics": {},
        }
    ]
    analyzer = DataScalingAnalyzer()
    with pytest.raises(NoEmpiricalScalingDataError, match="No verified empirical runs"):
        analyzer.analyze_exp05(dry_runs)


def test_analyzer_rejects_missing_execution_accuracy() -> None:
    """Verify runs missing execution_accuracy are rejected."""
    bad_runs = [
        {
            "run_id": "exp05_n500_s42",
            "experiment_id": EXP05_EXPERIMENT_ID,
            "seed": 42,
            "status": "completed",
            "sample_size": 500,
            "metrics": {"valid_sql_rate": 0.98},  # Missing execution_accuracy!
        }
    ]
    analyzer = DataScalingAnalyzer()
    with pytest.raises(NoEmpiricalScalingDataError):
        analyzer.analyze_exp05(bad_runs)


def test_analyzer_exp05_computes_statistics() -> None:
    """Verify analyzer computes per-size means, standard deviations, and diffs correctly."""
    runs = _generate_synthetic_test_metrics_exp05()
    analyzer = DataScalingAnalyzer()
    res = analyzer.analyze_exp05(runs)

    assert isinstance(res, EXP05AnalysisResult)
    assert res.experiment_id == EXP05_EXPERIMENT_ID
    assert len(res.size_metrics) == 5

    # Check 7000 condition
    m7000 = next(m for m in res.size_metrics if m.sample_size == 7000)
    assert m7000.seeds == [42, 43, 44]
    assert pytest.approx(m7000.mean_execution_accuracy, abs=1e-3) == 0.75
    assert m7000.accuracy_diff_relative_to_max == 0.0

    # Check 500 condition has negative diff relative to 7000
    m500 = next(m for m in res.size_metrics if m.sample_size == 500)
    assert pytest.approx(m500.mean_execution_accuracy, abs=1e-3) == 0.50
    assert pytest.approx(m500.accuracy_diff_relative_to_max, abs=1e-3) == -0.25


def test_analyzer_exp05_log_linear_fit() -> None:
    """Verify OLS log-linear fitting produces valid coefficients and high R2 on monotonic data."""
    runs = _generate_synthetic_test_metrics_exp05()
    analyzer = DataScalingAnalyzer()
    res = analyzer.analyze_exp05(runs)

    assert res.log_linear_fit.fit_status == "converged"
    assert res.log_linear_fit.r_squared is not None
    assert res.log_linear_fit.r_squared > 0.90
    assert "a" in res.log_linear_fit.coefficients
    assert "b" in res.log_linear_fit.coefficients
    assert res.log_linear_fit.coefficients["b"] > 0.0  # Positive scaling slope


def test_analyzer_exp05_power_law_fit() -> None:
    """Verify power-law EX(N) = a - b * N^(-c) fit produces bounded parameters."""
    runs = _generate_synthetic_test_metrics_exp05()
    analyzer = DataScalingAnalyzer()
    res = analyzer.analyze_exp05(runs)

    if res.power_law_fit.fit_status == "converged":
        coeffs = res.power_law_fit.coefficients
        assert coeffs["b"] >= 0.0
        assert coeffs["c"] > 0.0
        assert 0.0 <= coeffs["a"] <= 1.0


def test_analyzer_exp06_group_comparisons() -> None:
    """Verify EXP-06 analyzes human vs synthetic groups with seed-matched paired differences."""
    runs: list[dict[str, object]] = []
    # Seed 42, 43, 44 for each of 3 groups
    groups = {
        "human_only": 0.70,
        "synthetic_only": 0.60,
        "human_synthetic_50_50": 0.68,
    }
    for grp, base_acc in groups.items():
        for s in [42, 43, 44]:
            runs.append(
                {
                    "run_id": f"exp06_{grp}_s{s}",
                    "experiment_id": EXP06_EXPERIMENT_ID,
                    "group_name": grp,
                    "seed": s,
                    "status": "completed",
                    "total_examples": 3000,
                    "metrics": {
                        "execution_accuracy": base_acc + (s - 43) * 0.01,
                        "valid_sql_rate": 0.97,
                        "exact_match": base_acc - 0.05,
                    },
                }
            )

    analyzer = DataScalingAnalyzer()
    res = analyzer.analyze_exp06(runs)

    assert isinstance(res, EXP06AnalysisResult)
    assert res.experiment_id == EXP06_EXPERIMENT_ID
    assert len(res.group_metrics) == 3

    human_metric = next(m for m in res.group_metrics if m.group_name == "human_only")
    synth_metric = next(m for m in res.group_metrics if m.group_name == "synthetic_only")

    assert pytest.approx(human_metric.mean_execution_accuracy, abs=1e-3) == 0.70
    assert pytest.approx(synth_metric.mean_execution_accuracy, abs=1e-3) == 0.60

    # Paired differences: human_vs_synthetic should be ~+0.10
    paired = res.paired_comparisons["human_vs_synthetic"]
    assert pytest.approx(paired["mean_difference"], abs=1e-3) == 0.10
