from __future__ import annotations

import gc
import importlib.util
import os
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.code_utils.code_utils import module_name_from_file_path

if TYPE_CHECKING:
    from codeflash.models.models import BenchmarkKey

PYTEST_BENCHMARK_INSTALLED = importlib.util.find_spec("pytest_benchmark") is not None

# Calibration defaults (matching pytest-benchmark)
MIN_TIME = 0.000005  # 5µs — minimum time per round during calibration
MAX_TIME = 1.0  # 1s — maximum wall-clock time per test
MIN_ROUNDS = 5
CALIBRATION_PRECISION = 10


@dataclass
class BenchmarkStats:
    min_ns: float
    max_ns: float
    mean_ns: float
    median_ns: float
    stddev_ns: float
    iqr_ns: float
    rounds: int
    iterations: int
    outliers: str

    @staticmethod
    def from_per_iteration_times(times_ns: list[float], iterations: int) -> BenchmarkStats:
        n = len(times_ns)
        sorted_times = sorted(times_ns)
        q1 = sorted_times[n // 4] if n >= 4 else sorted_times[0]
        q3 = sorted_times[3 * n // 4] if n >= 4 else sorted_times[-1]
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        mild_outliers = sum(1 for t in times_ns if t < low_fence or t > high_fence)
        severe_fence_low = q1 - 3.0 * iqr
        severe_fence_high = q3 + 3.0 * iqr
        severe_outliers = sum(1 for t in times_ns if t < severe_fence_low or t > severe_fence_high)

        return BenchmarkStats(
            min_ns=min(times_ns),
            max_ns=max(times_ns),
            mean_ns=statistics.mean(times_ns),
            median_ns=statistics.median(times_ns),
            stddev_ns=statistics.stdev(times_ns) if n > 1 else 0.0,
            iqr_ns=iqr,
            rounds=n,
            iterations=iterations,
            outliers=f"{severe_outliers};{mild_outliers}",
        )


@dataclass
class MemoryStats:
    peak_memory_bytes: int
    total_allocations: int

    @staticmethod
    def parse_memray_results(bin_dir: Path, bin_prefix: str) -> dict:
        from codeflash.models.models import BenchmarkKey

        try:
            from memray import FileReader
        except ImportError as e:
            msg = "memray is required for --memory profiling. Install with: uv add memray pytest-memray"
            raise ImportError(msg) from e

        results: dict[BenchmarkKey, MemoryStats] = {}
        for bin_file in sorted(bin_dir.glob(f"{bin_prefix}-*.bin")):
            stem = bin_file.stem
            # pytest-memray names: {prefix}-{nodeid with :: and os.sep replaced by -}.bin
            nodeid_part = stem[len(bin_prefix) + 1 :]  # strip "{prefix}-"
            # Extract the test function name (last segment after the final -)
            # Node IDs look like: tests-benchmarks-test_file.py-test_func_name
            # We need the module_path and function_name for BenchmarkKey
            # Split on ".py-" to separate module path from function name
            parts = nodeid_part.split(".py-", 1)
            if len(parts) == 2:
                module_part = parts[0].replace("-", ".")
                function_name = parts[1]
            else:
                module_part = nodeid_part.rsplit("-", 1)[0].replace("-", ".")
                function_name = nodeid_part.rsplit("-", 1)[-1] if "-" in nodeid_part else nodeid_part

            try:
                reader = FileReader(str(bin_file))
                meta = reader.metadata
                bm_key = BenchmarkKey(module_path=module_part, function_name=function_name)
                results[bm_key] = MemoryStats(
                    peak_memory_bytes=meta.peak_memory, total_allocations=meta.total_allocations
                )
                reader.close()
            except OSError:
                continue
        return results


class CodeFlashBenchmarkPlugin:
    def __init__(self) -> None:
        self._trace_path = None
        self._connection = None
        self.project_root = None
        self.benchmark_timings = []

    def setup(self, trace_path: str, project_root: str) -> None:
        try:
            self.project_root = project_root
            self._trace_path = trace_path
            self._connection = sqlite3.connect(self._trace_path)
            cur = self._connection.cursor()
            cur.execute("PRAGMA synchronous = OFF")
            cur.execute("PRAGMA journal_mode = MEMORY")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_timings("
                "benchmark_module_path TEXT, benchmark_function_name TEXT, benchmark_line_number INTEGER,"
                "round_index INTEGER, iterations INTEGER, round_time_ns INTEGER)"
            )
            self._connection.commit()
            self.close()
        except Exception as e:
            print(f"Database setup error: {e}")
            if self._connection:
                self._connection.close()
                self._connection = None
            raise

    def write_benchmark_timings(self) -> None:
        if not self.benchmark_timings:
            return

        if self._connection is None:
            self._connection = sqlite3.connect(self._trace_path)

        try:
            cur = self._connection.cursor()
            cur.executemany(
                "INSERT INTO benchmark_timings "
                "(benchmark_module_path, benchmark_function_name, benchmark_line_number, "
                "round_index, iterations, round_time_ns) VALUES (?, ?, ?, ?, ?, ?)",
                self.benchmark_timings,
            )
            self._connection.commit()
            self.benchmark_timings = []
        except Exception as e:
            print(f"Error writing to benchmark timings database: {e}")
            self._connection.rollback()
            raise

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    @staticmethod
    def get_function_benchmark_timings(trace_path: Path) -> dict[str, dict[BenchmarkKey, float]]:
        from codeflash.models.models import BenchmarkKey

        result: dict[str, dict[BenchmarkKey, float]] = {}
        connection = sqlite3.connect(trace_path)
        cursor = connection.cursor()

        try:
            # Get total iterations per benchmark to normalize
            cursor.execute(
                "SELECT benchmark_module_path, benchmark_function_name, "
                "SUM(iterations) FROM benchmark_timings "
                "GROUP BY benchmark_module_path, benchmark_function_name"
            )
            total_iterations: dict[BenchmarkKey, int] = {}
            for row in cursor.fetchall():
                bm_file, bm_func, total_iters = row
                key = BenchmarkKey(module_path=bm_file, function_name=bm_func)
                total_iterations[key] = total_iters

            cursor.execute(
                "SELECT module_name, class_name, function_name, "
                "benchmark_module_path, benchmark_function_name, benchmark_line_number, function_time_ns "
                "FROM benchmark_function_timings"
            )

            # Accumulate total function time
            raw_totals: dict[str, dict[BenchmarkKey, int]] = {}
            for row in cursor.fetchall():
                module_name, class_name, function_name, benchmark_file, benchmark_func, _benchmark_line, time_ns = row
                if class_name:
                    qualified_name = f"{module_name}.{class_name}.{function_name}"
                else:
                    qualified_name = f"{module_name}.{function_name}"
                benchmark_key = BenchmarkKey(module_path=benchmark_file, function_name=benchmark_func)
                if qualified_name not in raw_totals:
                    raw_totals[qualified_name] = {}
                raw_totals[qualified_name][benchmark_key] = raw_totals[qualified_name].get(benchmark_key, 0) + time_ns

            # Normalize to per-iteration average
            for qualified_name, bm_dict in raw_totals.items():
                result[qualified_name] = {}
                for bm_key, total_ns in bm_dict.items():
                    iters = total_iterations.get(bm_key, 1)
                    result[qualified_name][bm_key] = total_ns / iters

        finally:
            connection.close()

        return result

    @staticmethod
    def get_benchmark_timings(trace_path: Path) -> dict[BenchmarkKey, BenchmarkStats]:
        from codeflash.models.models import BenchmarkKey

        connection = sqlite3.connect(trace_path)
        cursor = connection.cursor()

        try:
            # Get overhead per benchmark to subtract
            cursor.execute(
                "SELECT benchmark_module_path, benchmark_function_name, benchmark_line_number, SUM(overhead_time_ns) "
                "FROM benchmark_function_timings "
                "GROUP BY benchmark_module_path, benchmark_function_name, benchmark_line_number"
            )
            overhead_by_benchmark: dict[BenchmarkKey, int] = {}
            for row in cursor.fetchall():
                bm_file, bm_func, _bm_line, total_overhead_ns = row
                key = BenchmarkKey(module_path=bm_file, function_name=bm_func)
                overhead_by_benchmark[key] = total_overhead_ns or 0

            # Get per-round data
            cursor.execute(
                "SELECT benchmark_module_path, benchmark_function_name, benchmark_line_number, "
                "round_index, iterations, round_time_ns "
                "FROM benchmark_timings ORDER BY round_index"
            )

            rounds_data: dict[BenchmarkKey, list[tuple[int, int]]] = {}
            for row in cursor.fetchall():
                bm_file, bm_func, _bm_line, _round_idx, iterations, round_time_ns = row
                key = BenchmarkKey(module_path=bm_file, function_name=bm_func)
                if key not in rounds_data:
                    rounds_data[key] = []
                rounds_data[key].append((iterations, round_time_ns))

            result: dict[BenchmarkKey, BenchmarkStats] = {}
            for bm_key, rounds in rounds_data.items():
                total_overhead = overhead_by_benchmark.get(bm_key, 0)
                total_rounds = len(rounds)
                overhead_per_round = total_overhead / total_rounds if total_rounds > 0 else 0
                iterations = rounds[0][0]  # All rounds have same iteration count

                per_iteration_times = []
                for iters, round_time_ns in rounds:
                    adjusted = max(0, round_time_ns - overhead_per_round)
                    per_iteration_times.append(adjusted / iters)

                result[bm_key] = BenchmarkStats.from_per_iteration_times(per_iteration_times, iterations)

        finally:
            connection.close()

        return result

    # Pytest hooks
    @pytest.hookimpl
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        codeflash_trace.close()
        if self.benchmark_timings:
            self.write_benchmark_timings()
        self.close()

    @staticmethod
    def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
        if not config.getoption("--codeflash-trace"):
            return

        skip_no_benchmark = pytest.mark.skip(reason="Test requires benchmark fixture")
        for item in items:
            has_fixture = hasattr(item, "fixturenames") and "benchmark" in item.fixturenames  # ty:ignore[unsupported-operator]
            has_marker = False
            if hasattr(item, "get_closest_marker"):
                marker = item.get_closest_marker("benchmark")
                if marker is not None:
                    has_marker = True
            if not (has_fixture or has_marker):
                item.add_marker(skip_no_benchmark)

    class Benchmark:  # noqa: D106
        def __init__(self, request: pytest.FixtureRequest) -> None:
            self.request = request

        def __call__(self, func, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
            if args or kwargs:
                return self.run_benchmark(func, *args, **kwargs)

            def wrapped_func(*args, **kwargs):  # noqa: ANN002, ANN003
                return func(*args, **kwargs)

            self.run_benchmark(func)
            return wrapped_func

        def run_benchmark(self, func, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            node_path = getattr(self.request.node, "path", None) or getattr(self.request.node, "fspath", None)
            if node_path is None:
                raise RuntimeError("Unable to determine test file path from pytest node")

            benchmark_module_path = module_name_from_file_path(
                Path(str(node_path)), Path(codeflash_benchmark_plugin.project_root), traverse_up=True
            )
            benchmark_function_name = self.request.node.name
            line_number = int(str(sys._getframe(2).f_lineno))  # noqa: SLF001

            os.environ["CODEFLASH_BENCHMARK_FUNCTION_NAME"] = benchmark_function_name
            os.environ["CODEFLASH_BENCHMARK_MODULE_PATH"] = benchmark_module_path
            os.environ["CODEFLASH_BENCHMARK_LINE_NUMBER"] = str(line_number)

            # Phase 1: Calibrate (tracing disabled to avoid overhead)
            os.environ["CODEFLASH_BENCHMARKING"] = "False"
            iterations, calibrated_duration = calibrate(func, args, kwargs)

            # Phase 2: Multi-round benchmark (tracing enabled)
            os.environ["CODEFLASH_BENCHMARKING"] = "True"
            rounds = max(MIN_ROUNDS, ceil(MAX_TIME / calibrated_duration)) if calibrated_duration > 0 else MIN_ROUNDS

            result = None
            for round_idx in range(rounds):
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    start = time.perf_counter_ns()
                    for _ in range(iterations):
                        result = func(*args, **kwargs)
                    end = time.perf_counter_ns()
                finally:
                    if gc_was_enabled:
                        gc.enable()

                round_time = end - start
                codeflash_benchmark_plugin.benchmark_timings.append(
                    (benchmark_module_path, benchmark_function_name, line_number, round_idx, iterations, round_time)
                )

                # Flush function timings per round
                codeflash_trace.write_function_timings()
                codeflash_trace.function_call_count = 0

            os.environ["CODEFLASH_BENCHMARKING"] = "False"
            return result


def compute_timer_precision() -> float:
    minimum = float("inf")
    for _ in range(20):
        t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns()
        dt = t2 - t1
        if dt > 0:
            minimum = min(minimum, dt)
    return minimum / 1e9  # Convert to seconds


def calibrate(func, args, kwargs) -> tuple[int, float]:
    timer_precision = compute_timer_precision()
    min_time = max(MIN_TIME, timer_precision * CALIBRATION_PRECISION)
    min_time_estimate = min_time * 5 / CALIBRATION_PRECISION

    iterations = 1
    while True:
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            for _ in range(iterations):
                func(*args, **kwargs)
            end = time.perf_counter_ns()
        finally:
            if gc_was_enabled:
                gc.enable()

        duration = (end - start) / 1e9  # Convert to seconds

        if duration >= min_time:
            break

        if duration >= min_time_estimate:
            iterations = ceil(min_time * iterations / duration)
        else:
            iterations *= 10

    return iterations, duration


codeflash_benchmark_plugin = CodeFlashBenchmarkPlugin()
