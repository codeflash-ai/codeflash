"""Tests for React render benchmarking and comparison."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.benchmarking import (
    RenderBenchmark,
    compare_render_benchmarks,
    format_render_benchmark_for_pr,
)
from codeflash.languages.javascript.frameworks.react.testgen import has_high_density_interactions, has_react_test_interactions
from codeflash.languages.javascript.parse import InteractionDurationProfile, RenderProfile
from codeflash.result.critic import render_efficiency_critic


class TestRenderBenchmark:
    def test_render_count_reduction_pct(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=50,
            optimized_render_count=5,
            original_avg_duration_ms=10.0,
            optimized_avg_duration_ms=2.0,
        )
        assert b.render_count_reduction_pct == 90.0

    def test_duration_reduction_pct(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=10,
            optimized_render_count=5,
            original_avg_duration_ms=20.0,
            optimized_avg_duration_ms=5.0,
        )
        assert b.duration_reduction_pct == 75.0

    def test_render_speedup(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=10,
            optimized_render_count=2,
            original_avg_duration_ms=100.0,
            optimized_avg_duration_ms=10.0,
        )
        assert b.render_speedup_x == 10.0

    def test_zero_original_render_count(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=0,
            optimized_render_count=0,
            original_avg_duration_ms=0.0,
            optimized_avg_duration_ms=0.0,
        )
        assert b.render_count_reduction_pct == 0.0
        assert b.duration_reduction_pct == 0.0

    def test_zero_optimized_duration(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=10,
            optimized_render_count=1,
            original_avg_duration_ms=10.0,
            optimized_avg_duration_ms=0.0,
        )
        assert b.render_speedup_x == 0.0


class TestCompareRenderBenchmarks:
    def test_basic_comparison(self):
        original = [
            RenderProfile("Counter", "mount", 12.0, 15.0, 1),
            RenderProfile("Counter", "update", 3.0, 15.0, 5),
            RenderProfile("Counter", "update", 2.5, 15.0, 10),
        ]
        optimized = [
            RenderProfile("Counter", "mount", 11.0, 14.0, 1),
            RenderProfile("Counter", "update", 1.5, 14.0, 2),
        ]
        benchmark = compare_render_benchmarks(original, optimized)
        assert benchmark is not None
        assert benchmark.component_name == "Counter"
        assert benchmark.original_render_count == 10  # max of [1, 5, 10]
        assert benchmark.optimized_render_count == 2  # max of [1, 2]
        # Phase-aware fields
        assert benchmark.original_update_render_count == 10  # max of update profiles [5, 10]
        assert benchmark.optimized_update_render_count == 2  # max of update profiles [2]
        assert benchmark.original_mount_render_count == 1
        assert benchmark.optimized_mount_render_count == 1
        assert benchmark.has_update_phase_data is True

    def test_mount_only_comparison(self):
        original = [RenderProfile("Widget", "mount", 5.0, 8.0, 1)]
        optimized = [RenderProfile("Widget", "mount", 4.0, 7.0, 1)]
        benchmark = compare_render_benchmarks(original, optimized)
        assert benchmark is not None
        assert benchmark.original_update_render_count == 0
        assert benchmark.optimized_update_render_count == 0
        assert benchmark.has_update_phase_data is False

    def test_empty_profiles(self):
        assert compare_render_benchmarks([], []) is None
        assert compare_render_benchmarks([RenderProfile("A", "mount", 1.0, 1.0, 1)], []) is None
        assert compare_render_benchmarks([], [RenderProfile("A", "mount", 1.0, 1.0, 1)]) is None

    def test_avg_duration(self):
        original = [
            RenderProfile("Counter", "mount", 10.0, 15.0, 1),
            RenderProfile("Counter", "update", 6.0, 15.0, 2),
        ]
        optimized = [
            RenderProfile("Counter", "mount", 8.0, 14.0, 1),
            RenderProfile("Counter", "update", 2.0, 14.0, 2),
        ]
        benchmark = compare_render_benchmarks(original, optimized)
        assert benchmark is not None
        assert benchmark.original_avg_duration_ms == 8.0  # (10+6)/2
        assert benchmark.optimized_avg_duration_ms == 5.0  # (8+2)/2


class TestFormatRenderBenchmarkForPr:
    def test_markdown_table(self):
        b = RenderBenchmark(
            component_name="TaskList",
            original_render_count=47,
            optimized_render_count=3,
            original_avg_duration_ms=340.0,
            optimized_avg_duration_ms=12.0,
        )
        output = format_render_benchmark_for_pr(b)
        assert "React Render Performance" in output
        assert "Total renders" in output
        assert "47" in output
        assert "3" in output
        assert "93.6%" in output
        assert "28.3x" in output

    def test_no_speedup_line_when_not_faster(self):
        b = RenderBenchmark(
            component_name="Simple",
            original_render_count=2,
            optimized_render_count=2,
            original_avg_duration_ms=1.0,
            optimized_avg_duration_ms=1.5,
        )
        output = format_render_benchmark_for_pr(b)
        assert "improved" not in output


class TestRenderEfficiencyCritic:
    def test_accepts_significant_render_reduction(self):
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=100.0,
            original_update_render_count=48,
            optimized_update_render_count=8,
        ) is True

    def test_rejects_insignificant_reduction(self):
        assert render_efficiency_critic(
            original_render_count=10,
            optimized_render_count=9,
            original_render_duration=100.0,
            optimized_render_duration=99.0,
        ) is False

    def test_accepts_significant_duration_improvement(self):
        assert render_efficiency_critic(
            original_render_count=10,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=10.0,
            original_update_render_count=8,
            optimized_update_render_count=8,
        ) is True

    def test_rejects_worse_than_best(self):
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=50.0,
            best_render_count_until_now=5,
        ) is False

    def test_accepts_better_than_best(self):
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=3,
            original_render_duration=100.0,
            optimized_render_duration=10.0,
            best_render_count_until_now=5,
            original_update_render_count=48,
            optimized_update_render_count=1,
        ) is True

    def test_uses_update_phase_counts_when_available(self):
        # Total counts look similar, but update-phase shows big reduction
        assert render_efficiency_critic(
            original_render_count=10,
            optimized_render_count=9,
            original_render_duration=100.0,
            optimized_render_duration=95.0,
            original_update_render_count=8,
            optimized_update_render_count=2,
            original_update_duration=80.0,
            optimized_update_duration=10.0,
        ) is True

    def test_rejects_mount_only_when_no_secondary_signals(self):
        # No update-phase data AND no DOM/child/interaction signals → rejected
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=100.0,
            original_update_render_count=0,
            optimized_update_render_count=0,
        ) is False

    def test_falls_back_to_total_with_dom_signal(self):
        # No update-phase data but DOM mutations present → uses total counts
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=100.0,
            original_update_render_count=0,
            optimized_update_render_count=0,
            original_dom_mutations=100,
            optimized_dom_mutations=20,
        ) is True


class TestFormatWithPhaseData:
    def test_shows_update_phase_row(self):
        b = RenderBenchmark(
            component_name="Counter",
            original_render_count=12,
            optimized_render_count=4,
            original_avg_duration_ms=10.0,
            optimized_avg_duration_ms=3.0,
            original_update_render_count=10,
            optimized_update_render_count=2,
            original_update_avg_duration_ms=8.0,
            optimized_update_avg_duration_ms=1.5,
            original_mount_render_count=2,
            optimized_mount_render_count=2,
        )
        output = format_render_benchmark_for_pr(b)
        assert "Re-renders (update)" in output
        assert "Total renders" in output


class TestPerTestRenderCounting:
    def test_per_test_reset_with_max_aggregation(self):
        """With per-test counter reset, each test's markers start from 0.

        Test A: 3 renders → markers with counts 1,2,3 (max=3)
        Test B: 7 renders → markers with counts 1,2,...,7 (max=7)
        max() across all markers = 7 (worst single test), not 10 (cumulative).
        """
        # Simulate markers from two tests with per-test reset
        test_a_profiles = [
            RenderProfile("Counter", "update", 2.0, 5.0, 1),
            RenderProfile("Counter", "update", 2.0, 5.0, 2),
            RenderProfile("Counter", "update", 2.0, 5.0, 3),
        ]
        test_b_profiles = [
            RenderProfile("Counter", "update", 1.5, 5.0, 1),
            RenderProfile("Counter", "update", 1.5, 5.0, 2),
            RenderProfile("Counter", "update", 1.5, 5.0, 3),
            RenderProfile("Counter", "update", 1.5, 5.0, 4),
            RenderProfile("Counter", "update", 1.5, 5.0, 5),
            RenderProfile("Counter", "update", 1.5, 5.0, 6),
            RenderProfile("Counter", "update", 1.5, 5.0, 7),
        ]
        all_profiles = test_a_profiles + test_b_profiles
        # max(render_count) = 7, which is worst-case single test
        from codeflash.languages.javascript.frameworks.react.benchmarking import _aggregate_render_count

        assert _aggregate_render_count(all_profiles) == 7

    def test_profiler_code_includes_before_each_reset(self):
        from codeflash.languages.javascript.frameworks.react.profiler import generate_render_counter_code

        code = generate_render_counter_code("MyComponent")
        assert "beforeEach" in code
        assert "_codeflash_render_count_MyComponent = 0;" in code


class TestMultiComponentBenchmarking:
    def test_group_by_component(self):
        from codeflash.languages.javascript.frameworks.react.benchmarking import _group_by_component

        profiles = [
            RenderProfile("Parent", "mount", 5.0, 10.0, 1),
            RenderProfile("Parent", "update", 3.0, 10.0, 2),
            RenderProfile("Child", "mount", 2.0, 5.0, 1),
            RenderProfile("Child", "update", 1.0, 5.0, 5),
        ]
        grouped = _group_by_component(profiles)
        assert len(grouped) == 2
        assert len(grouped["Parent"]) == 2
        assert len(grouped["Child"]) == 2

    def test_compare_with_target_component(self):
        original = [
            RenderProfile("Parent", "mount", 5.0, 10.0, 1),
            RenderProfile("Parent", "update", 3.0, 10.0, 5),
            RenderProfile("Child", "mount", 2.0, 5.0, 1),
            RenderProfile("Child", "update", 1.0, 5.0, 10),
        ]
        optimized = [
            RenderProfile("Parent", "mount", 5.0, 10.0, 1),
            RenderProfile("Parent", "update", 3.0, 10.0, 5),
            RenderProfile("Child", "mount", 2.0, 5.0, 1),
            RenderProfile("Child", "update", 0.5, 5.0, 2),
        ]
        benchmark = compare_render_benchmarks(
            original, optimized, target_component_name="Parent"
        )
        assert benchmark is not None
        # Parent renders unchanged
        assert benchmark.original_render_count == 5
        assert benchmark.optimized_render_count == 5
        # Child renders reduced (10 → 2)
        assert benchmark.child_render_reduction == 8

    def test_compare_without_target_uses_first_component(self):
        """Backward compat: no target_component_name uses all profiles."""
        original = [
            RenderProfile("Counter", "mount", 5.0, 10.0, 1),
            RenderProfile("Counter", "update", 3.0, 10.0, 5),
        ]
        optimized = [
            RenderProfile("Counter", "mount", 5.0, 10.0, 1),
            RenderProfile("Counter", "update", 1.0, 10.0, 2),
        ]
        benchmark = compare_render_benchmarks(original, optimized)
        assert benchmark is not None
        assert benchmark.component_name == "Counter"
        assert benchmark.child_render_reduction == 0


class TestInteractionDurationParsing:
    def test_parse_interaction_duration_markers(self):
        from codeflash.languages.javascript.parse import parse_interaction_duration_markers

        stdout = (
            "!######REACT_INTERACTION_DURATION:Counter:15.50:3######!\n"
            "!######REACT_INTERACTION_DURATION:Timer:8.25:1######!\n"
        )
        profiles = parse_interaction_duration_markers(stdout)
        assert len(profiles) == 2
        assert profiles[0].component_name == "Counter"
        assert profiles[0].duration_ms == 15.50
        assert profiles[0].burst_count == 3
        assert profiles[1].component_name == "Timer"
        assert profiles[1].duration_ms == 8.25
        assert profiles[1].burst_count == 1

    def test_parse_empty_stdout(self):
        from codeflash.languages.javascript.parse import parse_interaction_duration_markers

        assert parse_interaction_duration_markers("") == []

    def test_compare_with_interaction_durations(self):
        original = [RenderProfile("Counter", "update", 3.0, 10.0, 5)]
        optimized = [RenderProfile("Counter", "update", 3.0, 10.0, 5)]
        orig_durations = [InteractionDurationProfile("Counter", 50.0, 5)]
        opt_durations = [InteractionDurationProfile("Counter", 20.0, 2)]
        benchmark = compare_render_benchmarks(
            original, optimized,
            original_interaction_durations=orig_durations,
            optimized_interaction_durations=opt_durations,
        )
        assert benchmark is not None
        assert benchmark.original_interaction_duration_ms == 50.0
        assert benchmark.optimized_interaction_duration_ms == 20.0
        assert benchmark.interaction_duration_reduction_pct == 60.0
        assert benchmark.original_burst_count == 5
        assert benchmark.optimized_burst_count == 2


class TestRenderEfficiencyCriticNewSignals:
    def test_accepts_child_render_reduction(self):
        """Accept when target renders are unchanged but children render less."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            child_render_reduction=8,
        ) is True

    def test_rejects_small_child_render_reduction(self):
        """Reject when child reduction is below threshold."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            child_render_reduction=1,
        ) is False

    def test_accepts_interaction_duration_reduction(self):
        """Accept when interaction duration decreases significantly (debounce/throttle)."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            original_interaction_duration_ms=100.0,
            optimized_interaction_duration_ms=30.0,
        ) is True

    def test_rejects_small_interaction_duration_reduction(self):
        """Reject when interaction duration reduction is too small."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            original_interaction_duration_ms=100.0,
            optimized_interaction_duration_ms=90.0,
        ) is False


class TestRenderEfficiencyCriticTrustDuration:
    def test_duration_ignored_when_trust_duration_false(self):
        """Duration-only improvement should be rejected when trust_duration=False."""
        assert render_efficiency_critic(
            original_render_count=10,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=10.0,
            trust_duration=False,
        ) is False

    def test_duration_accepted_when_trust_duration_true(self):
        """Duration-only improvement should be accepted when trust_duration=True (default)."""
        assert render_efficiency_critic(
            original_render_count=10,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=10.0,
            original_update_render_count=8,
            optimized_update_render_count=8,
            trust_duration=True,
        ) is True

    def test_render_count_still_works_with_trust_duration_false(self):
        """Render count reduction should still be accepted when trust_duration=False."""
        assert render_efficiency_critic(
            original_render_count=50,
            optimized_render_count=10,
            original_render_duration=100.0,
            optimized_render_duration=100.0,
            original_update_render_count=48,
            optimized_update_render_count=8,
            trust_duration=False,
        ) is True

    def test_dom_mutations_still_work_with_trust_duration_false(self):
        """DOM mutation reduction should still be accepted when trust_duration=False."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            original_dom_mutations=100,
            optimized_dom_mutations=50,
            trust_duration=False,
        ) is True

    def test_child_renders_still_work_with_trust_duration_false(self):
        """Child render reduction should still be accepted when trust_duration=False."""
        assert render_efficiency_critic(
            original_render_count=5,
            optimized_render_count=5,
            original_render_duration=10.0,
            optimized_render_duration=10.0,
            child_render_reduction=5,
            trust_duration=False,
        ) is True


class TestHasReactTestInteractions:
    def test_detects_fire_event(self):
        assert has_react_test_interactions("fireEvent.click(button);") is True

    def test_detects_user_event(self):
        assert has_react_test_interactions("await userEvent.type(input, 'hello');") is True

    def test_detects_rerender(self):
        assert has_react_test_interactions("rerender(<Counter count={2} />);") is True

    def test_detects_act(self):
        assert has_react_test_interactions("act(() => { setState(5); });") is True

    def test_rejects_render_only(self):
        assert has_react_test_interactions("const { container } = render(<Counter />);") is False

    def test_rejects_empty(self):
        assert has_react_test_interactions("") is False


class TestHasHighDensityInteractions:
    def test_detects_loop_with_fireEvent(self):
        source = """
        for (let i = 0; i < 10; i++) {
            fireEvent.click(button);
        }
        """
        assert has_high_density_interactions(source) is True

    def test_detects_loop_with_rerender(self):
        source = """
        for (let i = 0; i < 10; i++) {
            rerender(<Component {...props} />);
        }
        """
        assert has_high_density_interactions(source) is True

    def test_detects_many_sequential_interactions(self):
        source = """
        fireEvent.click(button);
        fireEvent.click(button);
        fireEvent.click(button);
        """
        assert has_high_density_interactions(source) is True

    def test_rejects_single_interaction(self):
        source = "fireEvent.click(button);"
        assert has_high_density_interactions(source) is False

    def test_rejects_two_interactions(self):
        source = """
        fireEvent.click(button);
        fireEvent.click(button);
        """
        assert has_high_density_interactions(source) is False

    def test_rejects_no_interactions(self):
        assert has_high_density_interactions("render(<Component />);") is False

    def test_detects_loop_with_userEvent(self):
        source = """
        for (const phrase of phrases) {
            userEvent.type(input, phrase);
        }
        """
        assert has_high_density_interactions(source) is True
