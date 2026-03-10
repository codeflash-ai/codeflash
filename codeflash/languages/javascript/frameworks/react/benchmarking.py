"""React render benchmarking and comparison.

Compares original vs optimized render profiles from React Profiler
instrumentation to quantify re-render reduction and render time improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.languages.javascript.parse import DomMutationProfile, RenderProfile


@dataclass(frozen=True)
class RenderBenchmark:
    """Comparison of original vs optimized render metrics."""

    component_name: str
    original_render_count: int
    optimized_render_count: int
    original_avg_duration_ms: float
    optimized_avg_duration_ms: float
    original_dom_mutations: int = 0
    optimized_dom_mutations: int = 0

    @property
    def render_count_reduction_pct(self) -> float:
        """Percentage reduction in render count (0-100)."""
        if self.original_render_count == 0:
            return 0.0
        return (self.original_render_count - self.optimized_render_count) / self.original_render_count * 100

    @property
    def duration_reduction_pct(self) -> float:
        """Percentage reduction in render duration (0-100)."""
        if self.original_avg_duration_ms == 0:
            return 0.0
        return (self.original_avg_duration_ms - self.optimized_avg_duration_ms) / self.original_avg_duration_ms * 100

    @property
    def render_speedup_x(self) -> float:
        """Render time speedup factor (e.g., 2.5x means 2.5 times faster)."""
        if self.optimized_avg_duration_ms == 0:
            return 0.0
        return self.original_avg_duration_ms / self.optimized_avg_duration_ms

    @property
    def dom_mutation_reduction_pct(self) -> float:
        """Percentage reduction in DOM mutations (0-100)."""
        if self.original_dom_mutations == 0:
            return 0.0
        return (self.original_dom_mutations - self.optimized_dom_mutations) / self.original_dom_mutations * 100


def compare_render_benchmarks(
    original_profiles: list[RenderProfile],
    optimized_profiles: list[RenderProfile],
    original_dom_mutations: list[DomMutationProfile] | None = None,
    optimized_dom_mutations: list[DomMutationProfile] | None = None,
) -> RenderBenchmark | None:
    """Compare original and optimized render profiles.

    Aggregates render counts and durations across all render events
    for the same component, then computes the benchmark comparison.
    """
    if not original_profiles or not optimized_profiles:
        return None

    # Use the first profile's component name
    component_name = original_profiles[0].component_name

    # Aggregate original metrics
    orig_count = max((p.render_count for p in original_profiles), default=0)
    orig_durations = [p.actual_duration_ms for p in original_profiles]
    orig_avg_duration = sum(orig_durations) / len(orig_durations) if orig_durations else 0.0

    # Aggregate optimized metrics
    opt_count = max((p.render_count for p in optimized_profiles), default=0)
    opt_durations = [p.actual_duration_ms for p in optimized_profiles]
    opt_avg_duration = sum(opt_durations) / len(opt_durations) if opt_durations else 0.0

    # Aggregate DOM mutation counts
    orig_dom = sum(p.mutation_count for p in original_dom_mutations) if original_dom_mutations else 0
    opt_dom = sum(p.mutation_count for p in optimized_dom_mutations) if optimized_dom_mutations else 0

    return RenderBenchmark(
        component_name=component_name,
        original_render_count=orig_count,
        optimized_render_count=opt_count,
        original_avg_duration_ms=orig_avg_duration,
        optimized_avg_duration_ms=opt_avg_duration,
        original_dom_mutations=orig_dom,
        optimized_dom_mutations=opt_dom,
    )


def format_render_benchmark_for_pr(benchmark: RenderBenchmark) -> str:
    """Format render benchmark data for PR comment body."""
    lines = [
        "### React Render Performance",
        "",
        "| Metric | Before | After | Improvement |",
        "|--------|--------|-------|-------------|",
        f"| Renders | {benchmark.original_render_count} | {benchmark.optimized_render_count} "
        f"| {benchmark.render_count_reduction_pct:.1f}% fewer |",
        f"| Avg render time | {benchmark.original_avg_duration_ms:.2f}ms "
        f"| {benchmark.optimized_avg_duration_ms:.2f}ms "
        f"| {benchmark.duration_reduction_pct:.1f}% faster |",
    ]

    if benchmark.original_dom_mutations > 0 or benchmark.optimized_dom_mutations > 0:
        lines.append(
            f"| DOM mutations | {benchmark.original_dom_mutations} | {benchmark.optimized_dom_mutations} "
            f"| {benchmark.dom_mutation_reduction_pct:.1f}% fewer |"
        )

    if benchmark.render_speedup_x > 1:
        lines.append(f"\nRender time improved **{benchmark.render_speedup_x:.1f}x**.")

    return "\n".join(lines)
