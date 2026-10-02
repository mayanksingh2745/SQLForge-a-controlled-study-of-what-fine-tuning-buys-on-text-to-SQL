"""Statistical analysis module for Text-to-SQL evaluation.

Implements:
1. Non-parametric bootstrap confidence intervals (B=1000) for execution accuracy and metrics.
2. Paired difference confidence intervals for rigorous model comparisons (d_i = score_a,i - score_b,i).
3. McNemar's test with continuity correction for binary paired outcomes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sqlforge.schemas.evaluation import ConfidenceInterval


def compute_bootstrap_ci(
    scores: list[float] | list[int] | list[bool],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute non-parametric bootstrap percentile confidence interval.

    Args:
        scores: List of example-level scores (e.g. 1.0 for correct, 0.0 for incorrect).
        n_bootstrap: Number of bootstrap resamples (default 1000).
        confidence_level: Desired confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducible bootstrap sampling.

    Returns:
        ConfidenceInterval model with empirical lower and upper bounds.
    """
    if not scores:
        return ConfidenceInterval(lower=0.0, upper=0.0, confidence_level=confidence_level)

    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)

    # Edge case: single score or identical scores
    if n == 1 or np.all(arr == arr[0]):
        val = float(arr[0])
        val = max(0.0, min(1.0, val))
        return ConfidenceInterval(
            lower=round(val, 4), upper=round(val, 4), confidence_level=confidence_level
        )

    rng = np.random.default_rng(seed)
    # Draw (n_bootstrap, n) indices with replacement
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower = float(np.percentile(bootstrap_means, lower_pct))
    upper = float(np.percentile(bootstrap_means, upper_pct))

    # Clamp bounds to [0.0, 1.0] and ensure lower <= upper
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))
    if lower > upper:
        lower = upper

    return ConfidenceInterval(
        lower=round(lower, 4),
        upper=round(upper, 4),
        confidence_level=confidence_level,
    )


class PairedDifferenceResult(BaseModel):
    """Result of paired difference bootstrap confidence interval."""

    model_config = ConfigDict(frozen=True)

    mean_difference: float = Field(..., description="Sample mean difference (A - B)")
    lower: float = Field(..., description="Lower bootstrap confidence bound")
    upper: float = Field(..., description="Upper bootstrap confidence bound")
    significant: bool = Field(..., description="True if 0 is excluded from confidence interval")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g. 0.95)")

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def compute_paired_difference_ci(
    scores_a: list[float] | list[int] | list[bool],
    scores_b: list[float] | list[int] | list[bool],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedDifferenceResult:
    """Compute bootstrap confidence interval for paired model performance difference.

    d_i = score_a,i - score_b,i

    Args:
        scores_a: Per-example scores for Model A.
        scores_b: Per-example scores for Model B (evaluated on identical examples).
        n_bootstrap: Number of bootstrap iterations (default 1000).
        confidence_level: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        PairedDifferenceResult with mean_difference, lower, upper, significant.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Paired comparison requires same length: got {len(scores_a)} vs {len(scores_b)}"
        )

    if not scores_a:
        return PairedDifferenceResult(
            mean_difference=0.0,
            lower=0.0,
            upper=0.0,
            significant=False,
            confidence_level=confidence_level,
        )

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    diffs = arr_a - arr_b
    n = len(diffs)
    mean_diff = float(np.mean(diffs))

    if n == 1 or np.all(diffs == diffs[0]):
        return PairedDifferenceResult(
            mean_difference=round(mean_diff, 4),
            lower=round(mean_diff, 4),
            upper=round(mean_diff, 4),
            significant=abs(mean_diff) > 1e-5,
            confidence_level=confidence_level,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, (alpha / 2.0) * 100.0))
    upper = float(np.percentile(bootstrap_means, (1.0 - alpha / 2.0) * 100.0))

    # Significant if 0 is strictly outside [lower, upper]
    significant = bool((lower > 0.0) or (upper < 0.0))

    return PairedDifferenceResult(
        mean_difference=round(mean_diff, 4),
        lower=round(lower, 4),
        upper=round(upper, 4),
        significant=significant,
        confidence_level=confidence_level,
    )


def mcnemar_test(
    scores_a: list[bool | int | float],
    scores_b: list[bool | int | float],
) -> dict[str, Any]:
    """Perform McNemar's test with continuity correction on binary paired outcomes.

    Args:
        scores_a: Binary outcomes for Model A (1/True = correct, 0/False = incorrect).
        scores_b: Binary outcomes for Model B on the identical examples.

    Returns:
        Dict with statistic (chi-square), b, c, discordant_count, p_value, significant.
    """
    if not scores_a or not scores_b:
        raise ValueError("Inputs cannot be empty.")

    if len(scores_a) != len(scores_b):
        raise ValueError("McNemar's test requires same length observation lists.")

    b = 0  # A correct, B incorrect
    c = 0  # A incorrect, B correct

    for sa, sb in zip(scores_a, scores_b, strict=True):
        a_pass = bool(sa)
        b_pass = bool(sb)
        if a_pass and not b_pass:
            b += 1
        elif not a_pass and b_pass:
            c += 1

    discordant = b + c
    if discordant == 0:
        return {
            "statistic": 0.0,
            "b": b,
            "c": c,
            "b_discordant": b,
            "c_discordant": c,
            "discordant": 0,
            "p_value": 1.0,
            "p_value_approx": 1.0,
            "significant": False,
        }

    # Continuity-corrected McNemar statistic: (|b - c| - 1)^2 / (b + c)
    stat = ((abs(b - c) - 1.0) ** 2) / discordant
    stat = max(0.0, stat)

    # Standard normal approximation p-value for 1 degree of freedom: 2 * (1 - Phi(sqrt(stat)))
    z = math.sqrt(stat)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    p_val = max(0.0, min(1.0, p_val))

    return {
        "statistic": round(stat, 4),
        "b": b,
        "c": c,
        "b_discordant": b,
        "c_discordant": c,
        "discordant": discordant,
        "p_value": round(p_val, 6),
        "p_value_approx": round(p_val, 6),
        "significant": bool(p_val < 0.05),
    }
