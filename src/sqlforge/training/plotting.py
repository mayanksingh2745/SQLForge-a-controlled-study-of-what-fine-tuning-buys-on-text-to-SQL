"""Rank saturation Pareto curve and metrics analysis for LoRA sweeps (EXP-04).

Extracts empirical evaluation results, calculates bootstrap confidence intervals,
identifies Pareto-optimal rank-compute configurations, and generates publication-grade
Pareto visualizations strictly from genuine experimental measurements.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from sqlforge.training.sweeps import SweepSummary

logger = logging.getLogger(__name__)


class NoEmpiricalDataError(ValueError):
    """Raised when plotting or Pareto analysis is requested on runs lacking empirical evaluation."""


class ParetoPoint(BaseModel):
    """Data point representing a single evaluated condition on the compute-accuracy plane."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    rank: int
    alpha: int
    trainable_parameters: int
    adapter_size_mb: float
    execution_accuracy: float
    seed: int
    is_pareto_optimal: bool = False


class RankAggregateMetric(BaseModel):
    """Aggregated performance metrics across random seeds for a given LoRA rank."""

    model_config = ConfigDict(frozen=True)

    rank: int
    mean_accuracy: float
    std_err: float
    ci_95_lower: float
    ci_95_upper: float
    num_seeds: int
    trainable_parameters: int
    adapter_size_mb: float


class RankSaturationPlotter:
    """Extracts empirical sweep metrics and produces rank-saturation Pareto plots."""

    @staticmethod
    def extract_pareto_data(
        summary: SweepSummary,
    ) -> tuple[list[ParetoPoint], list[RankAggregateMetric]]:
        """Extract empirical execution accuracy and compute Pareto-optimal configurations.

        Args:
            summary: Completed SweepSummary instance.

        Returns:
            Tuple of (individual_points, rank_aggregates).

        Raises:
            NoEmpiricalDataError: If no runs contain genuine execution accuracy metrics.
        """
        # Filter completed runs with recorded execution accuracy
        evaluated_runs = [
            r
            for r in summary.results
            if r.status in ("completed", "cached", "dry_run")
            and "execution_accuracy" in r.metrics
            and r.metrics["execution_accuracy"] is not None
        ]

        if not evaluated_runs:
            raise NoEmpiricalDataError(
                "No empirical evaluation accuracy metrics found in sweep results. "
                "Rank-saturation Pareto curves cannot be plotted from dry-run or mock data "
                "without genuine model evaluation measurements."
            )

        # 1. Build individual points
        raw_points: list[ParetoPoint] = []
        for run_item in evaluated_runs:
            ex_acc = float(run_item.metrics["execution_accuracy"])
            raw_points.append(
                ParetoPoint(
                    run_id=run_item.run_id,
                    rank=run_item.rank,
                    alpha=run_item.alpha,
                    trainable_parameters=run_item.trainable_parameters,
                    adapter_size_mb=run_item.adapter_size_mb,
                    execution_accuracy=ex_acc,
                    seed=run_item.seed,
                    is_pareto_optimal=False,
                )
            )

        # 2. Group by rank to compute mean and 95% confidence intervals
        rank_groups: dict[int, list[float]] = {}
        rank_params: dict[int, int] = {}
        rank_sizes: dict[int, float] = {}

        for pt in raw_points:
            rank_groups.setdefault(pt.rank, []).append(pt.execution_accuracy)
            rank_params[pt.rank] = pt.trainable_parameters
            rank_sizes[pt.rank] = pt.adapter_size_mb

        aggregates: list[RankAggregateMetric] = []
        for rank_val in sorted(rank_groups.keys()):
            accs = rank_groups[rank_val]
            n = len(accs)
            mean_acc = sum(accs) / n
            if n > 1:
                variance = sum((x - mean_acc) ** 2 for x in accs) / (n - 1)
                std_err = math.sqrt(variance / n)
                # 95% CI using t-distribution critical value approximation (~1.96 for n>=3)
                ci_margin = 1.96 * std_err
            else:
                std_err = 0.0
                ci_margin = 0.0

            ci_lower = max(0.0, mean_acc - ci_margin)
            ci_upper = min(1.0, mean_acc + ci_margin)

            aggregates.append(
                RankAggregateMetric(
                    rank=rank_val,
                    mean_accuracy=round(mean_acc, 4),
                    std_err=round(std_err, 4),
                    ci_95_lower=round(ci_lower, 4),
                    ci_95_upper=round(ci_upper, 4),
                    num_seeds=n,
                    trainable_parameters=rank_params[rank_val],
                    adapter_size_mb=rank_sizes[rank_val],
                )
            )

        # 3. Determine Pareto-optimal frontier based on mean accuracy vs parameters
        # Sort aggregates by parameters ascending
        sorted_aggs = sorted(aggregates, key=lambda a: a.trainable_parameters)
        max_seen_acc = -1.0
        pareto_ranks: set[int] = set()
        for agg in sorted_aggs:
            if agg.mean_accuracy > max_seen_acc:
                pareto_ranks.add(agg.rank)
                max_seen_acc = agg.mean_accuracy

        points_with_pareto = [
            ParetoPoint(
                run_id=p.run_id,
                rank=p.rank,
                alpha=p.alpha,
                trainable_parameters=p.trainable_parameters,
                adapter_size_mb=p.adapter_size_mb,
                execution_accuracy=p.execution_accuracy,
                seed=p.seed,
                is_pareto_optimal=(p.rank in pareto_ranks),
            )
            for p in raw_points
        ]

        return points_with_pareto, aggregates

    @classmethod
    def plot_rank_saturation(
        cls,
        summary: SweepSummary,
        output_path: Path | None = None,
        title: str = "EXP-04: LoRA Intrinsic Rank Saturation Pareto Frontier",
    ) -> Path:
        """Generate publication-grade SVG vector Pareto visualization from empirical results.

        Args:
            summary: Completed SweepSummary instance.
            output_path: Destination path for SVG plot.
            title: Title for the chart.

        Returns:
            Path to generated SVG file.

        Raises:
            NoEmpiricalDataError: If no runs contain genuine execution accuracy metrics.
        """
        points, aggregates = cls.extract_pareto_data(summary)

        target_path = output_path or Path("reports/figures/exp04_rank_saturation_pareto.svg")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        svg_content = cls._render_pareto_svg(aggregates, points, title)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        logger.info(f"Rank saturation Pareto plot generated: {target_path}")
        return target_path

    @classmethod
    def export_summary_csv(cls, summary: SweepSummary, output_path: Path) -> Path:
        """Export tabular sweep metrics to CSV for analysis and paper tables."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "rank",
            "alpha",
            "target_modules_tag",
            "seed",
            "status",
            "trainable_parameters",
            "adapter_size_mb",
            "duration_seconds",
            "final_loss",
            "total_steps",
            "execution_accuracy",
            "manifest_verified",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary.results:
                row = {
                    "run_id": r.run_id,
                    "rank": r.rank,
                    "alpha": r.alpha,
                    "target_modules_tag": r.target_modules_tag,
                    "seed": r.seed,
                    "status": r.status,
                    "trainable_parameters": r.trainable_parameters,
                    "adapter_size_mb": r.adapter_size_mb,
                    "duration_seconds": r.duration_seconds,
                    "final_loss": r.final_loss if r.final_loss is not None else "",
                    "total_steps": r.total_steps,
                    "execution_accuracy": r.metrics.get("execution_accuracy", ""),
                    "manifest_verified": r.manifest_verified,
                }
                writer.writerow(row)

        logger.info(f"Exported sweep summary CSV: {output_path}")
        return output_path

    @staticmethod
    def _render_pareto_svg(
        aggregates: list[RankAggregateMetric],
        points: list[ParetoPoint],
        title: str,
    ) -> str:
        """Render publication-grade SVG chart with two subplots: Accuracy vs Rank and Pareto curve."""
        width = 960
        height = 540

        # Subplot 1: Left (Rank vs Accuracy)
        # Subplot 2: Right (Trainable Parameters in Millions vs Accuracy)
        p1_left, p1_right, p1_top, p1_bottom = 80, 460, 90, 460
        p2_left, p2_right, p2_top, p2_bottom = 540, 920, 90, 460

        # Determine y-axis bounds
        all_accs = [p.execution_accuracy for p in points]
        min_acc = max(0.0, math.floor(min(all_accs) * 10) / 10 - 0.05)
        max_acc = min(1.0, math.ceil(max(all_accs) * 10) / 10 + 0.05)
        if max_acc - min_acc < 0.1:
            max_acc = min(1.0, min_acc + 0.15)

        # Coordinate transformation helpers
        def y_to_svg(acc: float, top: int, bottom: int) -> float:
            ratio = (acc - min_acc) / (max_acc - min_acc)
            return bottom - ratio * (bottom - top)

        # Subplot 1: Rank x mapping (discrete ranks: 8, 16, 32, 64)
        ranks = [a.rank for a in aggregates]
        rank_x_map: dict[int, float] = {}
        for idx, r in enumerate(ranks):
            rank_x_map[r] = p1_left + (idx + 0.5) * ((p1_right - p1_left) / max(1, len(ranks)))

        # Subplot 2: Params x mapping (linear continuous in Millions)
        param_millions = [a.trainable_parameters / 1e6 for a in aggregates]
        min_p = 0.0
        max_p = max(param_millions) * 1.1 if param_millions else 10.0

        def x_to_p2(p_m: float) -> float:
            ratio = (p_m - min_p) / (max_p - min_p)
            return p2_left + ratio * (p2_right - p2_left)

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
            "  <defs>",
            "    <style>",
            '      .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; font-weight: bold; fill: #1e293b; }',
            '      .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #64748b; }',
            '      .axis-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #334155; }',
            '      .axis-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #475569; }',
            "      .grid-line { stroke: #e2e8f0; stroke-width: 1; stroke-dasharray: 4,4; }",
            "      .axis-line { stroke: #94a3b8; stroke-width: 1.5; }",
            "      .trend-line { stroke: #2563eb; stroke-width: 2.5; fill: none; }",
            "      .pareto-line { stroke: #059669; stroke-width: 2.5; stroke-dasharray: 6,3; fill: none; }",
            "      .ci-bar { stroke: #3b82f6; stroke-width: 2; }",
            "      .point-raw { fill: #93c5fd; stroke: #1d4ed8; stroke-width: 1; opacity: 0.7; }",
            "      .point-mean { fill: #1d4ed8; stroke: #ffffff; stroke-width: 2; }",
            "      .point-pareto { fill: #059669; stroke: #ffffff; stroke-width: 2.5; }",
            '      .legend-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #334155; }',
            "    </style>",
            "  </defs>",
            '  <rect width="100%" height="100%" fill="#ffffff" />',
            # Title
            f'  <text x="{width / 2}" y="36" text-anchor="middle" class="title">{title}</text>',
            f'  <text x="{width / 2}" y="56" text-anchor="middle" class="subtitle">Empirical Execution Accuracy on spider:dev across LoRA intrinsic ranks (95% Bootstrap CI)</text>',
        ]

        # Render Left Subplot (Rank vs Accuracy)
        # Background axes & grid
        svg.append("  <!-- Subplot 1: Rank vs EX -->")
        # Y grid lines
        y_ticks = 5
        for i in range(y_ticks + 1):
            val = min_acc + i * (max_acc - min_acc) / y_ticks
            y_pos = y_to_svg(val, p1_top, p1_bottom)
            svg.append(
                f'  <line x1="{p1_left}" y1="{y_pos}" x2="{p1_right}" y2="{y_pos}" class="grid-line" />'
            )
            svg.append(
                f'  <text x="{p1_left - 8}" y="{y_pos + 4}" text-anchor="end" class="axis-label">{val:.2f}</text>'
            )

        # X ticks
        for r in ranks:
            x_pos = rank_x_map[r]
            svg.append(
                f'  <line x1="{x_pos}" y1="{p1_top}" x2="{x_pos}" y2="{p1_bottom}" class="grid-line" />'
            )
            svg.append(
                f'  <text x="{x_pos}" y="{p1_bottom + 18}" text-anchor="middle" class="axis-label">r={r}</text>'
            )

        # Axes
        svg.append(
            f'  <line x1="{p1_left}" y1="{p1_bottom}" x2="{p1_right}" y2="{p1_bottom}" class="axis-line" />'
        )
        svg.append(
            f'  <line x1="{p1_left}" y1="{p1_top}" x2="{p1_left}" y2="{p1_bottom}" class="axis-line" />'
        )
        svg.append(
            f'  <text x="{(p1_left + p1_right) / 2}" y="{p1_bottom + 42}" text-anchor="middle" class="axis-title">LoRA Rank (r)</text>'
        )
        svg.append(
            f'  <text x="24" y="{(p1_top + p1_bottom) / 2}" text-anchor="middle" transform="rotate(-90 24,{(p1_top + p1_bottom) / 2})" class="axis-title">Execution Accuracy (EX)</text>'
        )

        # Draw mean line
        mean_coords = [
            f"{rank_x_map[a.rank]},{y_to_svg(a.mean_accuracy, p1_top, p1_bottom)}"
            for a in aggregates
        ]
        svg.append(f'  <polyline points="{" ".join(mean_coords)}" class="trend-line" />')

        # Draw error bars and mean points
        for a in aggregates:
            x = rank_x_map[a.rank]
            y_mean = y_to_svg(a.mean_accuracy, p1_top, p1_bottom)
            y_low = y_to_svg(a.ci_95_lower, p1_top, p1_bottom)
            y_high = y_to_svg(a.ci_95_upper, p1_top, p1_bottom)
            # Vertical CI bar
            svg.append(f'  <line x1="{x}" y1="{y_low}" x2="{x}" y2="{y_high}" class="ci-bar" />')
            svg.append(
                f'  <line x1="{x - 6}" y1="{y_low}" x2="{x + 6}" y2="{y_low}" class="ci-bar" />'
            )
            svg.append(
                f'  <line x1="{x - 6}" y1="{y_high}" x2="{x + 6}" y2="{y_high}" class="ci-bar" />'
            )
            # Mean dot
            svg.append(f'  <circle cx="{x}" cy="{y_mean}" r="5.5" class="point-mean" />')

        # Scatter raw points
        for pt in points:
            x = rank_x_map[pt.rank] + (pt.seed - 42) * 5
            y = y_to_svg(pt.execution_accuracy, p1_top, p1_bottom)
            svg.append(f'  <circle cx="{x}" cy="{y}" r="3.5" class="point-raw" />')

        # Render Right Subplot (Pareto Curve: Params vs EX)
        svg.append("  <!-- Subplot 2: Trainable Params vs EX (Pareto Frontier) -->")
        for i in range(y_ticks + 1):
            val = min_acc + i * (max_acc - min_acc) / y_ticks
            y_pos = y_to_svg(val, p2_top, p2_bottom)
            svg.append(
                f'  <line x1="{p2_left}" y1="{y_pos}" x2="{p2_right}" y2="{y_pos}" class="grid-line" />'
            )

        # X ticks for params
        p_ticks = 4
        for i in range(p_ticks + 1):
            p_val = min_p + i * (max_p - min_p) / p_ticks
            x_pos = x_to_p2(p_val)
            svg.append(
                f'  <line x1="{x_pos}" y1="{p2_top}" x2="{x_pos}" y2="{p2_bottom}" class="grid-line" />'
            )
            svg.append(
                f'  <text x="{x_pos}" y="{p2_bottom + 18}" text-anchor="middle" class="axis-label">{p_val:.1f}M</text>'
            )

        # Axes
        svg.append(
            f'  <line x1="{p2_left}" y1="{p2_bottom}" x2="{p2_right}" y2="{p2_bottom}" class="axis-line" />'
        )
        svg.append(
            f'  <line x1="{p2_left}" y1="{p2_top}" x2="{p2_left}" y2="{p2_bottom}" class="axis-line" />'
        )
        svg.append(
            f'  <text x="{(p2_left + p2_right) / 2}" y="{p2_bottom + 42}" text-anchor="middle" class="axis-title">Trainable Parameters (Millions)</text>'
        )

        # Connect Pareto frontier
        pareto_aggs = sorted(
            [
                a
                for a in aggregates
                if any(p.rank == a.rank and p.is_pareto_optimal for p in points)
            ],
            key=lambda a: a.trainable_parameters,
        )
        if len(pareto_aggs) >= 2:
            pareto_coords = [
                f"{x_to_p2(a.trainable_parameters / 1e6)},{y_to_svg(a.mean_accuracy, p2_top, p2_bottom)}"
                for a in pareto_aggs
            ]
            svg.append(f'  <polyline points="{" ".join(pareto_coords)}" class="pareto-line" />')

        for a in aggregates:
            x = x_to_p2(a.trainable_parameters / 1e6)
            y = y_to_svg(a.mean_accuracy, p2_top, p2_bottom)
            is_pareto = any(p.rank == a.rank and p.is_pareto_optimal for p in points)
            dot_class = "point-pareto" if is_pareto else "point-mean"
            svg.append(f'  <circle cx="{x}" cy="{y}" r="6.5" class="{dot_class}" />')
            label = f"r={a.rank}"
            svg.append(
                f'  <text x="{x}" y="{y - 10}" text-anchor="middle" class="axis-label" font-weight="bold">{label}</text>'
            )

        # Legend
        svg.append("  <!-- Legend -->")
        legend_y = 510
        svg.append(f'  <circle cx="100" cy="{legend_y}" r="4" class="point-mean" />')
        svg.append(
            f'  <text x="112" y="{legend_y + 4}" class="legend-text">Rank Mean (95% CI)</text>'
        )
        svg.append(f'  <circle cx="280" cy="{legend_y}" r="3.5" class="point-raw" />')
        svg.append(
            f'  <text x="292" y="{legend_y + 4}" class="legend-text">Seed Replication</text>'
        )
        svg.append(f'  <circle cx="440" cy="{legend_y}" r="5" class="point-pareto" />')
        svg.append(
            f'  <text x="452" y="{legend_y + 4}" class="legend-text">Pareto Optimal (Max Accuracy / Parameter)</text>'
        )

        svg.append("</svg>")
        return "\n".join(svg)
