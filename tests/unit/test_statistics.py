"""Unit tests for statistical evaluation, bootstrap confidence intervals, and hypothesis testing."""

from __future__ import annotations

import pytest

from sqlforge.evaluation.statistics import (
    compute_bootstrap_ci,
    compute_paired_difference_ci,
    mcnemar_test,
)


class TestBootstrapCI:
    """Tests for non-parametric bootstrap percentile confidence intervals."""

    def test_empty_scores(self) -> None:
        ci = compute_bootstrap_ci([])
        assert ci.lower == 0.0
        assert ci.upper == 0.0
        assert ci.confidence_level == 0.95

    def test_single_score(self) -> None:
        ci = compute_bootstrap_ci([1.0])
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_identical_all_zeros(self) -> None:
        ci = compute_bootstrap_ci([0.0] * 50)
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_identical_all_ones(self) -> None:
        ci = compute_bootstrap_ci([1.0] * 50)
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_known_binary_distribution_coverage(self) -> None:
        # 80 ones, 20 zeros -> mean = 0.80
        scores = [1.0] * 80 + [0.0] * 20
        ci = compute_bootstrap_ci(scores, n_bootstrap=1000, seed=42)
        assert 0.70 <= ci.lower <= 0.78
        assert 0.82 <= ci.upper <= 0.89
        assert ci.lower <= 0.80 <= ci.upper

    def test_seed_reproducibility(self) -> None:
        scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        ci_1 = compute_bootstrap_ci(scores, n_bootstrap=500, seed=123)
        ci_2 = compute_bootstrap_ci(scores, n_bootstrap=500, seed=123)
        assert ci_1.lower == ci_2.lower
        assert ci_1.upper == ci_2.upper

    def test_different_confidence_levels(self) -> None:
        scores = [1.0] * 60 + [0.0] * 40
        ci_90 = compute_bootstrap_ci(scores, confidence_level=0.90, seed=42)
        ci_99 = compute_bootstrap_ci(scores, confidence_level=0.99, seed=42)
        # Higher confidence level produces wider interval
        assert ci_99.lower <= ci_90.lower
        assert ci_99.upper >= ci_90.upper

    def test_boolean_input_list(self) -> None:
        scores = [True, False, True, True]
        ci = compute_bootstrap_ci(scores, seed=42)
        assert 0.0 <= ci.lower <= 1.0
        assert 0.0 <= ci.upper <= 1.0
        assert ci.lower <= ci.upper


class TestPairedDifferenceCI:
    """Tests for paired bootstrap confidence intervals on difference metric (d_i = A_i - B_i)."""

    def test_mismatched_lengths_raise_error(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_paired_difference_ci([1.0, 0.0], [1.0])

    def test_empty_lists(self) -> None:
        ci = compute_paired_difference_ci([], [])
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_identical_performance_produces_zero_interval(self) -> None:
        scores_a = [1.0, 0.0, 1.0, 1.0, 0.0]
        scores_b = [1.0, 0.0, 1.0, 1.0, 0.0]
        ci = compute_paired_difference_ci(scores_a, scores_b)
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_strictly_superior_model(self) -> None:
        # Model A solves all 50 examples, Model B solves 25
        scores_a = [1.0] * 50
        scores_b = [1.0] * 25 + [0.0] * 25
        ci = compute_paired_difference_ci(scores_a, scores_b, seed=42)
        assert ci.lower > 0.35
        assert ci.upper <= 0.65
        assert ci.lower <= 0.50 <= ci.upper


class TestMcNemarTest:
    """Tests for McNemar's test with continuity correction."""

    def test_mismatched_lengths_raise_error(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mcnemar_test([1.0], [1.0, 0.0])

    def test_empty_inputs_raise_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            mcnemar_test([], [])

    def test_identical_predictions_zero_discordant_pairs(self) -> None:
        scores_a = [1.0, 0.0, 1.0]
        scores_b = [1.0, 0.0, 1.0]
        result = mcnemar_test(scores_a, scores_b)
        assert result["b_discordant"] == 0
        assert result["c_discordant"] == 0
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_symmetric_discordant_pairs(self) -> None:
        # A right & B wrong: 10 times; B right & A wrong: 10 times
        scores_a = [1.0] * 10 + [0.0] * 10
        scores_b = [0.0] * 10 + [1.0] * 10
        result = mcnemar_test(scores_a, scores_b)
        assert result["b_discordant"] == 10
        assert result["c_discordant"] == 10
        # Continuity correction gives (|10 - 10| - 1)^2 / 20 = 1 / 20 = 0.05
        assert pytest.approx(result["statistic"], 0.001) == 0.05
        assert result["p_value"] > 0.05
        assert not result["significant"]

    def test_statistically_significant_difference(self) -> None:
        # A right & B wrong: 35 times; B right & A wrong: 5 times
        scores_a = [1.0] * 35 + [0.0] * 5 + [1.0] * 50
        scores_b = [0.0] * 35 + [1.0] * 5 + [1.0] * 50
        result = mcnemar_test(scores_a, scores_b)
        assert result["b_discordant"] == 35
        assert result["c_discordant"] == 5
        # (|35 - 5| - 1)^2 / 40 = 29^2 / 40 = 841 / 40 = 21.025
        assert pytest.approx(result["statistic"], 0.01) == 21.025
        assert result["p_value"] < 0.001
        assert result["significant"]
