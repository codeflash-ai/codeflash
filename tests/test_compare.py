from __future__ import annotations

from codeflash.benchmarking.compare import CompareResult, has_meaningful_memory_change, render_comparison
from codeflash.benchmarking.plugin.plugin import BenchmarkStats, MemoryStats
from codeflash.models.models import BenchmarkKey


def _make_stats(median_ns: float = 1000.0, rounds: int = 10) -> BenchmarkStats:
    return BenchmarkStats(
        min_ns=median_ns * 0.9,
        max_ns=median_ns * 1.1,
        mean_ns=median_ns,
        median_ns=median_ns,
        stddev_ns=median_ns * 0.05,
        iqr_ns=median_ns * 0.1,
        rounds=rounds,
        iterations=100,
        outliers="0;0",
    )


def _make_memory(peak: int = 4_194_304, allocs: int = 1000) -> MemoryStats:
    return MemoryStats(peak_memory_bytes=peak, total_allocations=allocs)


BM_KEY = BenchmarkKey(module_path="tests.benchmarks.test_example", function_name="test_func")


class TestFormatMarkdownMemoryOnly:
    def test_memory_only_no_timing_table(self) -> None:
        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_memory={BM_KEY: _make_memory(peak=10_000_000, allocs=500)},
            head_memory={BM_KEY: _make_memory(peak=7_000_000, allocs=400)},
        )
        md = result.format_markdown()

        # Should have memory data
        assert "Peak Memory" in md
        assert "Allocations" in md
        # Should NOT have timing table headers
        assert "Min | Median | Mean | OPS" not in md
        assert "Per-Function" not in md

    def test_memory_only_returns_empty_when_no_data(self) -> None:
        result = CompareResult(base_ref="abc123", head_ref="def456")
        md = result.format_markdown()
        assert md == "_No benchmark results to compare._"

    def test_mixed_timing_and_memory(self) -> None:
        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_stats={BM_KEY: _make_stats()},
            head_stats={BM_KEY: _make_stats(median_ns=500.0)},
            base_memory={BM_KEY: _make_memory(peak=10_000_000)},
            head_memory={BM_KEY: _make_memory(peak=5_000_000)},
        )
        md = result.format_markdown()

        # Should have both timing and memory
        assert "Min | Median | Mean | OPS" in md
        assert "Peak Memory" in md

    def test_memory_only_always_shows_memory(self) -> None:
        """Memory-only keys always render the memory table, even if delta is <1%."""
        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_memory={BM_KEY: _make_memory(peak=10_000_000, allocs=1000)},
            head_memory={BM_KEY: _make_memory(peak=10_000_000, allocs=1000)},
        )
        md = result.format_markdown()
        # Even with identical memory, memory-only keys always show the table
        assert "Peak Memory" in md

    def test_timing_with_negligible_memory_suppressed(self) -> None:
        """When timing data exists, negligible memory changes are suppressed."""
        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_stats={BM_KEY: _make_stats()},
            head_stats={BM_KEY: _make_stats()},
            base_memory={BM_KEY: _make_memory(peak=10_000_000, allocs=1000)},
            head_memory={BM_KEY: _make_memory(peak=10_000_000, allocs=1000)},
        )
        md = result.format_markdown()
        # Timing table should be there
        assert "Min | Median | Mean | OPS" in md
        # Memory table should be suppressed (delta <1% and timing exists)
        assert "Peak Memory" not in md

    def test_memory_only_key_mixed_with_timing_key(self) -> None:
        """Some keys have timing, others are memory-only."""
        timing_key = BenchmarkKey(module_path="tests.bench", function_name="test_timing")
        memory_key = BenchmarkKey(module_path="tests.bench", function_name="test_memory")

        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_stats={timing_key: _make_stats()},
            head_stats={timing_key: _make_stats(median_ns=500.0)},
            base_memory={
                timing_key: _make_memory(peak=10_000_000),
                memory_key: _make_memory(peak=8_000_000),
            },
            head_memory={
                timing_key: _make_memory(peak=5_000_000),
                memory_key: _make_memory(peak=6_000_000),
            },
        )
        md = result.format_markdown()

        # Both benchmark keys should appear
        assert "test_timing" in md
        assert "test_memory" in md
        # Timing table for timing_key
        assert "Min | Median | Mean | OPS" in md


class TestRenderComparisonMemoryOnly:
    def test_memory_only_no_crash(self, capsys: object) -> None:
        """render_comparison should not crash or warn with memory-only data."""
        result = CompareResult(
            base_ref="abc123",
            head_ref="def456",
            base_memory={BM_KEY: _make_memory(peak=10_000_000)},
            head_memory={BM_KEY: _make_memory(peak=7_000_000)},
        )
        # Should not raise
        render_comparison(result)

    def test_empty_result_warns(self) -> None:
        result = CompareResult(base_ref="abc123", head_ref="def456")
        # Should return without error (just logs a warning)
        render_comparison(result)


class TestHasMeaningfulMemoryChange:
    def test_both_none(self) -> None:
        assert not has_meaningful_memory_change(None, None)

    def test_one_none(self) -> None:
        assert has_meaningful_memory_change(_make_memory(), None)
        assert has_meaningful_memory_change(None, _make_memory())

    def test_both_zero(self) -> None:
        assert not has_meaningful_memory_change(_make_memory(0, 0), _make_memory(0, 0))

    def test_no_change(self) -> None:
        mem = _make_memory(peak=1000, allocs=100)
        assert not has_meaningful_memory_change(mem, mem)

    def test_significant_peak_change(self) -> None:
        base = _make_memory(peak=10_000_000, allocs=1000)
        head = _make_memory(peak=8_000_000, allocs=1000)
        assert has_meaningful_memory_change(base, head)

    def test_significant_alloc_change(self) -> None:
        base = _make_memory(peak=10_000_000, allocs=1000)
        head = _make_memory(peak=10_000_000, allocs=800)
        assert has_meaningful_memory_change(base, head)
