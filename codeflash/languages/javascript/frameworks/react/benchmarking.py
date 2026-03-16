"""React render benchmarking and comparison.

Compares original vs optimized render profiles from React Profiler
instrumentation to quantify re-render reduction and render time improvement.

Phase-aware: separates mount-phase renders (expected for both original and
optimized) from update-phase renders (the primary signal for optimization
effectiveness — fewer updates means better memoization, debounce, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.languages.javascript.parse import DomMutationProfile, InteractionDurationProfile, RenderProfile

logger = logging.getLogger(__name__)


def _group_by_component(profiles: list[RenderProfile]) -> dict[str, list[RenderProfile]]:
    """Group render profiles by component name."""
    grouped: dict[str, list[RenderProfile]] = {}
    for p in profiles:
        grouped.setdefault(p.component_name, []).append(p)
    return grouped


def _split_by_phase(profiles: list[RenderProfile]) -> tuple[list[RenderProfile], list[RenderProfile]]:
    """Split render profiles into mount-phase and update-phase lists."""
    mount: list[RenderProfile] = []
    update: list[RenderProfile] = []
    for p in profiles:
        if p.phase == "mount":
            mount.append(p)
        else:
            update.append(p)
    return mount, update


def _aggregate_render_count(profiles: list[RenderProfile]) -> int:
    """Get the max render count from a list of profiles (cumulative counter)."""
    return max((p.render_count for p in profiles), default=0)


def _aggregate_avg_duration(profiles: list[RenderProfile]) -> float:
    """Get the average actual_duration_ms from a list of profiles."""
    if not profiles:
        return 0.0
    return sum(p.actual_duration_ms for p in profiles) / len(profiles)


@dataclass(frozen=True)
class RenderBenchmark:
    """Comparison of original vs optimized render metrics.

    Provides both total and phase-separated (mount vs update) metrics.
    Update-phase render count is the primary signal for optimization value.
    """

    component_name: str
    # Total (all phases combined)
    original_render_count: int
    optimized_render_count: int
    original_avg_duration_ms: float
    optimized_avg_duration_ms: float
    original_dom_mutations: int = 0
    optimized_dom_mutations: int = 0
    # Update-phase only (primary signal)
    original_update_render_count: int = 0
    optimized_update_render_count: int = 0
    original_update_avg_duration_ms: float = 0.0
    optimized_update_avg_duration_ms: float = 0.0
    # Mount-phase only (informational)
    original_mount_render_count: int = 0
    optimized_mount_render_count: int = 0
    # Child component render reduction (sum of reductions across all children)
    child_render_reduction: int = 0
    # Interaction duration metrics (from MutationObserver timestamps)
    original_interaction_duration_ms: float = 0.0
    optimized_interaction_duration_ms: float = 0.0
    original_burst_count: int = 0
    optimized_burst_count: int = 0

    @property
    def render_count_reduction_pct(self) -> float:
        """Percentage reduction in total render count (0-100)."""
        if self.original_render_count == 0:
            return 0.0
        return (self.original_render_count - self.optimized_render_count) / self.original_render_count * 100

    @property
    def update_render_count_reduction_pct(self) -> float:
        """Percentage reduction in update-phase render count (0-100)."""
        if self.original_update_render_count == 0:
            return 0.0
        return (
            (self.original_update_render_count - self.optimized_update_render_count)
            / self.original_update_render_count
            * 100
        )

    @property
    def duration_reduction_pct(self) -> float:
        """Percentage reduction in total render duration (0-100)."""
        if self.original_avg_duration_ms == 0:
            return 0.0
        return (self.original_avg_duration_ms - self.optimized_avg_duration_ms) / self.original_avg_duration_ms * 100

    @property
    def update_duration_reduction_pct(self) -> float:
        """Percentage reduction in update-phase render duration (0-100)."""
        if self.original_update_avg_duration_ms == 0:
            return 0.0
        return (
            (self.original_update_avg_duration_ms - self.optimized_update_avg_duration_ms)
            / self.original_update_avg_duration_ms
            * 100
        )

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

    @property
    def has_update_phase_data(self) -> bool:
        """Whether update-phase data is available (tests triggered re-renders)."""
        return self.original_update_render_count > 0 or self.optimized_update_render_count > 0

    @property
    def interaction_duration_reduction_pct(self) -> float:
        """Percentage reduction in interaction-to-settle duration (0-100)."""
        if self.original_interaction_duration_ms == 0:
            return 0.0
        return (
            (self.original_interaction_duration_ms - self.optimized_interaction_duration_ms)
            / self.original_interaction_duration_ms
            * 100
        )

    @property
    def has_child_render_data(self) -> bool:
        """Whether child component render reduction data is available."""
        return self.child_render_reduction > 0

    @property
    def has_interaction_duration_data(self) -> bool:
        """Whether interaction duration data is available."""
        return self.original_interaction_duration_ms > 0 or self.optimized_interaction_duration_ms > 0


def compare_render_benchmarks(
    original_profiles: list[RenderProfile],
    optimized_profiles: list[RenderProfile],
    original_dom_mutations: list[DomMutationProfile] | None = None,
    optimized_dom_mutations: list[DomMutationProfile] | None = None,
    target_component_name: str | None = None,
    original_interaction_durations: list[InteractionDurationProfile] | None = None,
    optimized_interaction_durations: list[InteractionDurationProfile] | None = None,
) -> RenderBenchmark | None:
    """Compare original and optimized render profiles with phase awareness.

    When target_component_name is provided, uses only the target component's
    profiles for the primary comparison and computes child render reductions
    from all other components. Falls back to using all profiles when no target
    is specified.

    Separates mount-phase and update-phase render profiles. Update-phase
    render count is the primary signal for optimization value.
    """
    if not original_profiles or not optimized_profiles:
        return None

    # Group by component for multi-component analysis
    orig_by_comp = _group_by_component(original_profiles)
    opt_by_comp = _group_by_component(optimized_profiles)

    # Select target component profiles
    if target_component_name and target_component_name in orig_by_comp:
        component_name = target_component_name
        target_orig = orig_by_comp[target_component_name]
        target_opt = opt_by_comp.get(target_component_name, [])
    else:
        component_name = original_profiles[0].component_name
        target_orig = original_profiles
        target_opt = optimized_profiles

    # Split by phase
    orig_mount, orig_update = _split_by_phase(target_orig)
    opt_mount, opt_update = _split_by_phase(target_opt)

    if not orig_update and not opt_update:
        logger.debug("No update-phase render markers found — tests may lack interactions")

    # Aggregate total metrics (all phases)
    orig_count = _aggregate_render_count(target_orig)
    opt_count = _aggregate_render_count(target_opt)
    orig_avg_duration = _aggregate_avg_duration(target_orig)
    opt_avg_duration = _aggregate_avg_duration(target_opt)

    # Aggregate update-phase metrics (primary signal)
    orig_update_count = _aggregate_render_count(orig_update)
    opt_update_count = _aggregate_render_count(opt_update)
    orig_update_avg_duration = _aggregate_avg_duration(orig_update)
    opt_update_avg_duration = _aggregate_avg_duration(opt_update)

    # Aggregate mount-phase metrics (informational)
    orig_mount_count = _aggregate_render_count(orig_mount)
    opt_mount_count = _aggregate_render_count(opt_mount)

    # Aggregate DOM mutation counts
    orig_dom = sum(p.mutation_count for p in original_dom_mutations) if original_dom_mutations else 0
    opt_dom = sum(p.mutation_count for p in optimized_dom_mutations) if optimized_dom_mutations else 0

    # Compute child render reduction (sum of reductions across all non-target children)
    child_render_reduction = 0
    if target_component_name:
        for comp_name, orig_comp_profiles in orig_by_comp.items():
            if comp_name == target_component_name:
                continue
            opt_comp_profiles = opt_by_comp.get(comp_name, [])
            orig_child_count = _aggregate_render_count(orig_comp_profiles)
            opt_child_count = _aggregate_render_count(opt_comp_profiles)
            if orig_child_count > opt_child_count:
                child_render_reduction += orig_child_count - opt_child_count

    # Aggregate interaction duration metrics
    orig_interaction_ms = 0.0
    opt_interaction_ms = 0.0
    orig_bursts = 0
    opt_bursts = 0
    if original_interaction_durations:
        orig_interaction_ms = sum(d.duration_ms for d in original_interaction_durations) / len(
            original_interaction_durations
        )
        orig_bursts = max((d.burst_count for d in original_interaction_durations), default=0)
    if optimized_interaction_durations:
        opt_interaction_ms = sum(d.duration_ms for d in optimized_interaction_durations) / len(
            optimized_interaction_durations
        )
        opt_bursts = max((d.burst_count for d in optimized_interaction_durations), default=0)

    return RenderBenchmark(
        component_name=component_name,
        original_render_count=orig_count,
        optimized_render_count=opt_count,
        original_avg_duration_ms=orig_avg_duration,
        optimized_avg_duration_ms=opt_avg_duration,
        original_dom_mutations=orig_dom,
        optimized_dom_mutations=opt_dom,
        original_update_render_count=orig_update_count,
        optimized_update_render_count=opt_update_count,
        original_update_avg_duration_ms=orig_update_avg_duration,
        optimized_update_avg_duration_ms=opt_update_avg_duration,
        original_mount_render_count=orig_mount_count,
        optimized_mount_render_count=opt_mount_count,
        child_render_reduction=child_render_reduction,
        original_interaction_duration_ms=orig_interaction_ms,
        optimized_interaction_duration_ms=opt_interaction_ms,
        original_burst_count=orig_bursts,
        optimized_burst_count=opt_bursts,
    )


def format_render_benchmark_for_pr(benchmark: RenderBenchmark) -> str:
    """Format render benchmark data for PR comment body with mount/update breakdown."""
    lines = [
        "### React Render Performance",
        "",
        "| Metric | Before | After | Improvement |",
        "|--------|--------|-------|-------------|",
    ]

    if benchmark.has_update_phase_data:
        lines.append(
            f"| Re-renders (update) | {benchmark.original_update_render_count} "
            f"| {benchmark.optimized_update_render_count} "
            f"| {benchmark.update_render_count_reduction_pct:.1f}% fewer |"
        )
        if benchmark.original_update_avg_duration_ms > 0:
            lines.append(
                f"| Avg update render time | {benchmark.original_update_avg_duration_ms:.2f}ms "
                f"| {benchmark.optimized_update_avg_duration_ms:.2f}ms "
                f"| {benchmark.update_duration_reduction_pct:.1f}% faster |"
            )

    lines.append(
        f"| Total renders | {benchmark.original_render_count} | {benchmark.optimized_render_count} "
        f"| {benchmark.render_count_reduction_pct:.1f}% fewer |"
    )
    lines.append(
        f"| Avg render time | {benchmark.original_avg_duration_ms:.2f}ms "
        f"| {benchmark.optimized_avg_duration_ms:.2f}ms "
        f"| {benchmark.duration_reduction_pct:.1f}% faster |"
    )

    if benchmark.original_dom_mutations > 0 or benchmark.optimized_dom_mutations > 0:
        lines.append(
            f"| DOM mutations | {benchmark.original_dom_mutations} | {benchmark.optimized_dom_mutations} "
            f"| {benchmark.dom_mutation_reduction_pct:.1f}% fewer |"
        )

    if benchmark.has_child_render_data:
        lines.append(f"| Child re-renders saved | — | — | {benchmark.child_render_reduction} fewer |")

    if benchmark.has_interaction_duration_data:
        lines.append(
            f"| Interaction duration | {benchmark.original_interaction_duration_ms:.2f}ms "
            f"| {benchmark.optimized_interaction_duration_ms:.2f}ms "
            f"| {benchmark.interaction_duration_reduction_pct:.1f}% faster |"
        )

    if benchmark.render_speedup_x > 1:
        lines.append(f"\nRender time improved **{benchmark.render_speedup_x:.1f}x**.")

    return "\n".join(lines)
