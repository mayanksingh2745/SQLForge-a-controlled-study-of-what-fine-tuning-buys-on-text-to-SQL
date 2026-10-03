"""Unit tests for Pareto plotting and rank saturation analysis (EXP-04)."""

from pathlib import Path

import pytest

from sqlforge.training.plotting import (
    NoEmpiricalDataError,
    RankSaturationPlotter,
)
from sqlforge.training.sweeps import (
    SweepRunResult,
    SweepSummary,
)


@pytest.fixture
def dry_run_summary() -> SweepSummary:
    """Sweep summary containing only dry-run runs without empirical evaluation accuracy."""
    results = [
        SweepRunResult(
            run_id=f"exp04_r{r}_a{r * 2}_all_linear_s42",
            rank=r,
            alpha=r * 2,
            target_modules_tag="all-linear",
            target_modules=["q_proj", "v_proj"],
            seed=42,
            status="dry_run",
            trainable_parameters=r * 1_000_000,
            adapter_size_mb=round(r * 2.0, 1),
            duration_seconds=1.5,
            metrics={},  # No execution accuracy!
        )
        for r in [8, 16, 32, 64]
    ]
    return SweepSummary(
        sweep_id="sweep_dry_run_test",
        experiment_id="EXP-04-RANK-SWEEP",
        base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        method="qlora",
        total_planned=4,
        completed=4,
        cached=0,
        failed=0,
        skipped=0,
        is_dry_run=True,
        results=results,
        created_at="2026-10-04T00:00:00Z",
    )


@pytest.fixture
def empirical_eval_summary() -> SweepSummary:
    """Sweep summary containing genuine/fixture empirical evaluation results."""
    # Synthetic fixture data simulating realistic accuracy saturation
    # r=8: ~0.68, r=16: ~0.73, r=32: ~0.75, r=64: ~0.748 (saturates)
    acc_map = {
        8: [0.675, 0.682, 0.678],
        16: [0.728, 0.735, 0.731],
        32: [0.748, 0.755, 0.752],
        64: [0.746, 0.750, 0.749],
    }

    results: list[SweepRunResult] = []
    for r, accs in acc_map.items():
        for s_idx, acc in enumerate(accs):
            seed = 42 + s_idx
            results.append(
                SweepRunResult(
                    run_id=f"exp04_r{r}_a{r * 2}_all_linear_s{seed}",
                    rank=r,
                    alpha=r * 2,
                    target_modules_tag="all-linear",
                    target_modules=["q_proj", "v_proj"],
                    seed=seed,
                    status="completed",
                    trainable_parameters=r * 500_000,
                    adapter_size_mb=round(r * 1.0, 1),
                    duration_seconds=300.0,
                    final_loss=0.35,
                    total_steps=500,
                    metrics={
                        "execution_accuracy": acc,
                        "syntax_valid_rate": 0.98,
                        "exact_match": acc - 0.12,
                    },
                    manifest_verified=True,
                )
            )

    return SweepSummary(
        sweep_id="sweep_empirical_test",
        experiment_id="EXP-04-RANK-SWEEP",
        base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        method="qlora",
        total_planned=len(results),
        completed=len(results),
        cached=0,
        failed=0,
        skipped=0,
        is_dry_run=False,
        results=results,
        created_at="2026-10-04T00:00:00Z",
    )


class TestRankSaturationPlotter:
    """Test suite for Pareto analysis and visualization safety."""

    def test_missing_empirical_data_raises_error(self, dry_run_summary: SweepSummary) -> None:
        """Verify that dry-run/unevaluated runs raise NoEmpiricalDataError without fabricating plots."""
        with pytest.raises(NoEmpiricalDataError, match="No empirical evaluation accuracy metrics"):
            RankSaturationPlotter.extract_pareto_data(dry_run_summary)

        with pytest.raises(NoEmpiricalDataError):
            RankSaturationPlotter.plot_rank_saturation(dry_run_summary)

    def test_extract_pareto_data_with_empirical_results(
        self, empirical_eval_summary: SweepSummary
    ) -> None:
        """Verify extraction of Pareto points, mean accuracy, and confidence intervals."""
        points, aggs = RankSaturationPlotter.extract_pareto_data(empirical_eval_summary)

        assert len(points) == 12  # 4 ranks * 3 seeds
        assert len(aggs) == 4  # 4 ranks

        # Verify ranks are ordered
        ranks = [a.rank for a in aggs]
        assert ranks == [8, 16, 32, 64]

        # Verify mean accuracy increases up to r=32
        agg_r8 = next(a for a in aggs if a.rank == 8)
        agg_r16 = next(a for a in aggs if a.rank == 16)
        agg_r32 = next(a for a in aggs if a.rank == 32)
        agg_r64 = next(a for a in aggs if a.rank == 64)

        assert agg_r8.mean_accuracy < agg_r16.mean_accuracy < agg_r32.mean_accuracy
        # r=64 saturates or slightly regresses
        assert agg_r64.mean_accuracy <= agg_r32.mean_accuracy + 0.01

        # Check confidence intervals
        for a in aggs:
            assert a.ci_95_lower <= a.mean_accuracy <= a.ci_95_upper
            assert a.num_seeds == 3
            assert a.std_err > 0.0

        # Verify Pareto optimal points: r=8, r=16, r=32 should be pareto-optimal
        pareto_points = [p for p in points if p.is_pareto_optimal]
        pareto_ranks = {p.rank for p in pareto_points}
        assert 8 in pareto_ranks
        assert 16 in pareto_ranks
        assert 32 in pareto_ranks

    def test_plot_rank_saturation_svg_generation(
        self, tmp_path: Path, empirical_eval_summary: SweepSummary
    ) -> None:
        """Verify publication-grade SVG chart is generated with expected elements."""
        out_svg = tmp_path / "exp04_pareto.svg"
        generated_path = RankSaturationPlotter.plot_rank_saturation(
            empirical_eval_summary,
            output_path=out_svg,
            title="EXP-04 Pareto Frontier Test",
        )

        assert generated_path.exists()
        assert generated_path == out_svg

        # Verify SVG content
        svg_text = out_svg.read_text(encoding="utf-8")
        assert "<svg" in svg_text
        assert "</svg>" in svg_text
        assert "EXP-04 Pareto Frontier Test" in svg_text
        assert "LoRA Rank (r)" in svg_text
        assert "Trainable Parameters (Millions)" in svg_text
        assert "r=8" in svg_text
        assert "r=16" in svg_text
        assert "r=32" in svg_text
        assert "r=64" in svg_text

    def test_export_summary_csv(self, tmp_path: Path, empirical_eval_summary: SweepSummary) -> None:
        """Verify CSV export contains all required columns."""
        out_csv = tmp_path / "sweep_results.csv"
        csv_path = RankSaturationPlotter.export_summary_csv(empirical_eval_summary, out_csv)

        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split(",")
        assert "run_id" in header
        assert "rank" in header
        assert "trainable_parameters" in header
        assert "execution_accuracy" in header
        assert len(lines) == 13  # Header + 12 rows
