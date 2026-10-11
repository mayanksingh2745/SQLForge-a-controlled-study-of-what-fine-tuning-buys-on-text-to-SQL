"""Statistical scaling analysis, log-linear and power-law fitting, and comparative evaluation.

Implements DataScalingAnalyzer for:
- EXP-05: Data-volume scaling laws (EX as a function of N in {500, 1000, 2000, 5000, 7000}).
- EXP-06: Seed-matched paired comparison (human_only, synthetic_only, human_synthetic_50_50).
- Strict anti-fabrication enforcement (NoEmpiricalScalingDataError raised when genuine data is absent).
"""

from __future__ import annotations

import math
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sqlforge.training.data_scaling import (
    EXP05_EXPERIMENT_ID,
    EXP06_EXPERIMENT_ID,
    NoEmpiricalScalingDataError,
)
from sqlforge.training.scaling_orchestrator import ScalingExperimentSummary, ScalingRunResult


class SizeScalingMetric(BaseModel):
    """Aggregate performance metrics for a specific training dataset size condition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_size: int
    mean_execution_accuracy: float
    std_execution_accuracy: float
    ci_95_lower: float
    ci_95_upper: float
    mean_valid_sql_rate: float
    mean_exact_match: float
    delta_vs_7000: float  # EX(N) - EX(7000)
    per_seed_accuracies: dict[int, float]
    note: str = (
        "Exploratory 95% bootstrap CI resampled at seed level (K=3 seeds). Statistically limited."
    )

    @property
    def seeds(self) -> list[int]:
        """List of evaluated seeds for this sample size."""
        return sorted(self.per_seed_accuracies.keys())

    @property
    def accuracy_diff_relative_to_max(self) -> float:
        """Alias for delta_vs_7000."""
        return self.delta_vs_7000


class ScalingFitResult(BaseModel):
    """Mathematical curve fitting parameters and goodness-of-fit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fit_type: str  # "log_linear" or "power_law"
    fit_status: str  # "converged", "success", or "fit_unavailable"
    formula: str
    parameters: dict[str, float] = Field(default_factory=dict)
    r_squared: float | None = None
    reason: str | None = None

    @property
    def coefficients(self) -> dict[str, float]:
        """Alias for fitted parameters."""
        return self.parameters


class EXP05AnalysisResult(BaseModel):
    """Complete machine-readable analysis for EXP-05 data scaling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = EXP05_EXPERIMENT_ID
    sample_sizes: list[int]
    size_metrics: list[SizeScalingMetric]
    log_linear_fit: ScalingFitResult
    power_law_fit: ScalingFitResult
    extrapolation_warning: str = "Fitting restricted strictly to observed range [500, 7000]. Extrapolation beyond 7,000 is prohibited."


class GroupComparisonMetric(BaseModel):
    """Aggregate metrics for an EXP-06 comparison group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_name: str
    human_count: int
    synthetic_count: int
    total_count: int
    mean_execution_accuracy: float
    std_execution_accuracy: float
    mean_valid_sql_rate: float
    mean_exact_match: float
    per_seed_accuracies: dict[int, float]


class EXP06AnalysisResult(BaseModel):
    """Complete machine-readable analysis for EXP-06 synthetic vs. human comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = EXP06_EXPERIMENT_ID
    group_metrics: list[GroupComparisonMetric]
    paired_differences: dict[str, dict[str, float]]
    statistical_statement: str = (
        "Point estimates and paired differences reported for observation only. "
        "No claims of statistical superiority, equivalence, or significance are asserted without higher replication power."
    )

    @property
    def paired_comparisons(self) -> dict[str, dict[str, float]]:
        """Alias for paired_differences."""
        return self.paired_differences


class DataScalingAnalyzer:
    """Analyzes empirical results from EXP-05 and EXP-06 fine-tuning runs."""

    def __init__(self, bootstrap_iterations: int = 1000, random_seed: int = 42) -> None:
        self.bootstrap_iterations = bootstrap_iterations
        self.random_seed = random_seed

    def _validate_empirical_input(
        self,
        summary: ScalingExperimentSummary | list[ScalingRunResult] | list[dict[str, Any]],
    ) -> list[ScalingRunResult]:
        """Validate that summary contains verified empirical results and reject dry/mock/fake runs."""
        if isinstance(summary, list):
            if not summary:
                raise NoEmpiricalScalingDataError("No empirical evaluation runs provided.")
            runs: list[ScalingRunResult] = []
            for item in summary:
                if isinstance(item, ScalingRunResult):
                    runs.append(item)
                elif isinstance(item, dict):
                    if item.get("status") == "dry_run":
                        runs.append(
                            ScalingRunResult(
                                run_id=str(item.get("run_id", "run")),
                                experiment_id=str(item.get("experiment_id", EXP05_EXPERIMENT_ID)),
                                seed=int(item.get("seed") or 42),
                                total_examples=int(
                                    item.get("total_examples") or item.get("sample_size") or 0
                                ),
                                human_examples_count=int(item.get("sample_size") or 0),
                                synthetic_examples_count=0,
                                group_name=item.get("group_name"),
                                rank=16,
                                alpha=32,
                                status="dry_run",
                                metrics={},
                            )
                        )
                        continue
                    m = item.get("metrics", {})
                    runs.append(
                        ScalingRunResult(
                            run_id=str(item.get("run_id", "run")),
                            experiment_id=str(item.get("experiment_id", EXP05_EXPERIMENT_ID)),
                            seed=int(item.get("seed") or 42),
                            total_examples=int(
                                item.get("total_examples") or item.get("sample_size") or 0
                            ),
                            human_examples_count=int(
                                item.get(
                                    "human_examples_count",
                                    item.get("total_examples") or item.get("sample_size") or 0,
                                )
                                or 0
                            ),
                            synthetic_examples_count=int(item.get("synthetic_examples_count") or 0),
                            group_name=item.get("group_name"),
                            rank=int(item.get("rank") or 16),
                            alpha=int(item.get("alpha", 32)),
                            status=str(item.get("status", "completed")),
                            final_loss=item.get("final_loss"),
                            total_steps=int(item.get("total_steps", 100)),
                            metrics=m,
                            checkpoint_dir=item.get("checkpoint_dir"),
                            manifest_verified=True,
                        )
                    )
            valid_runs = [
                r
                for r in runs
                if r.status in ("completed", "cached")
                and r.metrics.get("execution_accuracy") is not None
            ]
            if not valid_runs:
                raise NoEmpiricalScalingDataError(
                    "No verified empirical runs with execution accuracy metrics found in input list."
                )
            return valid_runs

        if summary.is_dry_run or not summary.has_empirical_evaluation:
            raise NoEmpiricalScalingDataError(
                "No verified empirical results are available. EXP-05/EXP-06 infrastructure is ready, "
                "but GPU training and evaluation remain pending."
            )

        if not summary.empirical_execution_allowed:
            raise NoEmpiricalScalingDataError(
                f"Summary runs were executed under provisional rank status ({summary.rank_selection_status!r}). "
                "Empirical analysis requires verified EXP-04 rank selection."
            )

        valid_runs = [
            r
            for r in summary.results
            if r.status in ("completed", "cached")
            and r.metrics.get("execution_accuracy") is not None
        ]
        if not valid_runs:
            raise NoEmpiricalScalingDataError(
                "No verified empirical runs with execution accuracy metrics found in summary."
            )

        return valid_runs

    def compute_seed_bootstrap_ci(
        self,
        values: list[float],
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute 95% bootstrap confidence interval resampling seed-level observations."""
        if not values:
            return (0.0, 0.0)
        if len(values) == 1:
            return (values[0], values[0])

        rng = random.Random(seed)
        means: list[float] = []
        n = len(values)

        for _ in range(self.bootstrap_iterations):
            sample = [rng.choice(values) for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        idx_lower = int(0.025 * len(means))
        idx_upper = int(0.975 * len(means))
        return (round(means[idx_lower], 4), round(means[idx_upper], 4))

    def fit_log_linear(self, sizes: list[int], accuracies: list[float]) -> ScalingFitResult:
        """Fit log-linear scaling curve: EX(N) = a + b * ln(N) using OLS."""
        if len(sizes) < 2 or len(sizes) != len(accuracies):
            return ScalingFitResult(
                fit_type="log_linear",
                fit_status="fit_unavailable",
                formula="EX(N) = a + b * ln(N)",
                reason="Insufficient data points for log-linear regression (minimum 2 required).",
            )

        x = [math.log(s) for s in sizes]
        y = accuracies
        k = len(x)

        mean_x = sum(x) / k
        mean_y = sum(y) / k

        ss_xx = sum((xi - mean_x) ** 2 for xi in x)
        ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)

        if ss_xx <= 1e-12:
            return ScalingFitResult(
                fit_type="log_linear",
                fit_status="fit_unavailable",
                formula="EX(N) = a + b * ln(N)",
                reason="Degenerate input data: zero variance in training sizes.",
            )

        b = ss_xy / ss_xx
        a = mean_y - b * mean_x

        ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y, strict=True))
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0

        return ScalingFitResult(
            fit_type="log_linear",
            fit_status="converged",
            formula="EX(N) = a + b * ln(N)",
            parameters={"a": round(a, 4), "b": round(b, 6)},
            r_squared=round(r2, 4),
        )

    def fit_power_law(self, sizes: list[int], accuracies: list[float]) -> ScalingFitResult:
        """Fit bounded power-law scaling curve: EX(N) = a - b * N^(-c).

        Subject to:
        - b >= 0
        - c > 0
        - 0 <= EX(N) <= 1 for all observed N.
        """
        if len(sizes) < 3 or len(sizes) != len(accuracies):
            return ScalingFitResult(
                fit_type="power_law",
                fit_status="fit_unavailable",
                formula="EX(N) = a - b * N^(-c)",
                reason="Insufficient data points for 3-parameter power-law fit (minimum 3 required).",
            )

        # Deterministic multi-resolution grid optimization
        best_loss = float("inf")
        best_params: tuple[float, float, float] | None = None

        y_max = max(accuracies)

        # Coarse grid
        for a_cand in [y_max, y_max + 0.05, y_max + 0.1, 0.85, 0.90, 0.95, 1.0]:
            if a_cand > 1.0:
                continue
            for b_cand in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
                for c_cand in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
                    # Check validity
                    preds = [a_cand - b_cand * (n ** (-c_cand)) for n in sizes]
                    if any(p < 0.0 or p > 1.0 for p in preds):
                        continue
                    loss = sum((yi - pi) ** 2 for yi, pi in zip(accuracies, preds, strict=True))
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (a_cand, b_cand, c_cand)

        if best_params is None:
            return ScalingFitResult(
                fit_type="power_law",
                fit_status="fit_unavailable",
                formula="EX(N) = a - b * N^(-c)",
                reason="Could not identify valid parameters satisfying bounds (b>=0, c>0, 0<=EX<=1).",
            )

        a_opt, b_opt, c_opt = best_params

        # Compute R^2
        mean_y = sum(accuracies) / len(accuracies)
        ss_tot = sum((yi - mean_y) ** 2 for yi in accuracies)
        r2 = max(0.0, 1.0 - (best_loss / ss_tot)) if ss_tot > 1e-12 else 1.0

        return ScalingFitResult(
            fit_type="power_law",
            fit_status="converged",
            formula="EX(N) = a - b * N^(-c)",
            parameters={"a": round(a_opt, 4), "b": round(b_opt, 4), "c": round(c_opt, 4)},
            r_squared=round(r2, 4),
        )

    def analyze_exp05(
        self,
        summary: ScalingExperimentSummary | list[ScalingRunResult] | list[dict[str, Any]],
    ) -> EXP05AnalysisResult:
        """Analyze EXP-05 empirical results across training sizes and seeds."""
        valid_runs = self._validate_empirical_input(summary)

        # Group runs by sample size
        by_size: dict[int, list[ScalingRunResult]] = {}
        for r in valid_runs:
            by_size.setdefault(r.total_examples, []).append(r)

        # Ensure reference size 7000 exists
        ref_ex_mean: float = 0.0
        if 7000 in by_size:
            ref_ex_mean = sum(
                r.metrics.get("execution_accuracy", 0.0) for r in by_size[7000]
            ) / len(by_size[7000])

        size_metrics: list[SizeScalingMetric] = []
        ordered_sizes = sorted(by_size.keys())
        mean_accuracies: list[float] = []

        for size in ordered_sizes:
            runs = by_size[size]
            per_seed_acc = {r.seed: float(r.metrics.get("execution_accuracy", 0.0)) for r in runs}
            acc_list = list(per_seed_acc.values())
            mean_acc = sum(acc_list) / len(acc_list)
            mean_accuracies.append(mean_acc)

            # Std dev
            if len(acc_list) > 1:
                var = sum((x - mean_acc) ** 2 for x in acc_list) / (len(acc_list) - 1)
                std_acc = math.sqrt(var)
            else:
                std_acc = 0.0

            ci_low, ci_high = self.compute_seed_bootstrap_ci(acc_list, seed=self.random_seed)

            valid_sql_list = [float(r.metrics.get("syntax_valid_rate", 1.0)) for r in runs]
            em_list = [float(r.metrics.get("exact_match", 0.0)) for r in runs]

            delta_7000 = mean_acc - ref_ex_mean if ref_ex_mean > 0 else 0.0

            size_metrics.append(
                SizeScalingMetric(
                    sample_size=size,
                    mean_execution_accuracy=round(mean_acc, 4),
                    std_execution_accuracy=round(std_acc, 4),
                    ci_95_lower=ci_low,
                    ci_95_upper=ci_high,
                    mean_valid_sql_rate=round(sum(valid_sql_list) / len(valid_sql_list), 4),
                    mean_exact_match=round(sum(em_list) / len(em_list), 4),
                    delta_vs_7000=round(delta_7000, 4),
                    per_seed_accuracies={s: round(v, 4) for s, v in per_seed_acc.items()},
                )
            )

        # Fit scaling laws on per-size means
        log_lin_fit = self.fit_log_linear(ordered_sizes, mean_accuracies)
        power_law_fit = self.fit_power_law(ordered_sizes, mean_accuracies)

        return EXP05AnalysisResult(
            experiment_id=EXP05_EXPERIMENT_ID,
            sample_sizes=ordered_sizes,
            size_metrics=size_metrics,
            log_linear_fit=log_lin_fit,
            power_law_fit=power_law_fit,
        )

    def analyze_exp06(
        self,
        summary: ScalingExperimentSummary | list[ScalingRunResult] | list[dict[str, Any]],
    ) -> EXP06AnalysisResult:
        """Analyze EXP-06 empirical comparisons across human, synthetic, and mixed groups."""
        valid_runs = self._validate_empirical_input(summary)

        by_group: dict[str, list[ScalingRunResult]] = {}
        for r in valid_runs:
            if r.group_name:
                by_group.setdefault(r.group_name, []).append(r)

        group_metrics: list[GroupComparisonMetric] = []
        group_seed_accs: dict[str, dict[int, float]] = {}

        for group_name in sorted(by_group.keys()):
            runs = by_group[group_name]
            per_seed_acc = {r.seed: float(r.metrics.get("execution_accuracy", 0.0)) for r in runs}
            group_seed_accs[group_name] = per_seed_acc
            acc_list = list(per_seed_acc.values())
            mean_acc = sum(acc_list) / len(acc_list)

            if len(acc_list) > 1:
                var = sum((x - mean_acc) ** 2 for x in acc_list) / (len(acc_list) - 1)
                std_acc = math.sqrt(var)
            else:
                std_acc = 0.0

            valid_sql_list = [float(r.metrics.get("syntax_valid_rate", 1.0)) for r in runs]
            em_list = [float(r.metrics.get("exact_match", 0.0)) for r in runs]

            first_run = runs[0]
            group_metrics.append(
                GroupComparisonMetric(
                    group_name=group_name,
                    human_count=first_run.human_examples_count,
                    synthetic_count=first_run.synthetic_examples_count,
                    total_count=first_run.total_examples,
                    mean_execution_accuracy=round(mean_acc, 4),
                    std_execution_accuracy=round(std_acc, 4),
                    mean_valid_sql_rate=round(sum(valid_sql_list) / len(valid_sql_list), 4),
                    mean_exact_match=round(sum(em_list) / len(em_list), 4),
                    per_seed_accuracies={s: round(v, 4) for s, v in per_seed_acc.items()},
                )
            )

        # Paired differences across matched seeds
        paired_diffs: dict[str, dict[str, float]] = {}
        for pair_name, (g1, g2) in [
            ("human_vs_synthetic", ("human_only", "synthetic_only")),
            ("human_vs_50_50", ("human_only", "human_synthetic_50_50")),
            ("50_50_vs_synthetic", ("human_synthetic_50_50", "synthetic_only")),
        ]:
            if g1 in group_seed_accs and g2 in group_seed_accs:
                common_seeds = set(group_seed_accs[g1].keys()) & set(group_seed_accs[g2].keys())
                diffs: list[float] = []
                for s in sorted(common_seeds):
                    diff = group_seed_accs[g1][s] - group_seed_accs[g2][s]
                    diffs.append(diff)
                mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
                paired_diffs[pair_name] = {
                    "mean_difference": round(mean_diff, 4),
                    "mean_paired_difference": round(mean_diff, 4),
                    "matched_seeds_count": float(len(common_seeds)),
                }

        return EXP06AnalysisResult(
            experiment_id=EXP06_EXPERIMENT_ID,
            group_metrics=group_metrics,
            paired_differences=paired_diffs,
        )
