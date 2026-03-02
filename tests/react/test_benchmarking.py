"""Tests for React render benchmarking and comparison."""

from __future__ import annotations

from codeflash.languages.javascript.frameworks.react.benchmarking import (
    RenderBenchmark,
    compare_render_benchmarks,
    format_render_benchmark_for_pr,
)
from codeflash.languages.javascript.parse import RenderProfile
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
        ) is True
